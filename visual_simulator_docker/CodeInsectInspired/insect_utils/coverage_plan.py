"""Deterministic coverage expansion and frozen provenance, without torch or Isaac."""

import math
from pathlib import Path

import numpy as np

from insect_utils.experiment_config import PROJECT, canonical_hash, digest
from insect_utils.experiment_state import atomic_json, read_json


PROTOCOL = "config/experiments/stage11_coverage_protocol.json"
PARTITIONS = ("train", "validation", "legacy_test", "new_test")


def radial_bin(x, y, edges):
    radius = math.hypot(x, y)
    if not math.isfinite(radius) or radius > edges[-1] + 1e-8:
        raise ValueError("Position outside the declared learning radius")
    return min(int(np.searchsorted(edges, radius, side="right")) - 1, len(edges) - 2)


def square_distances(points, references):
    if not len(references):
        return np.full(len(points), np.inf)
    minimum = np.full(len(points), np.inf)
    for reference in references:
        minimum = np.minimum(minimum, ((points - reference) ** 2).sum(axis=1))
    return minimum


def quadrant(x, y):
    return min(int((math.atan2(y, x) % (2 * math.pi)) / (math.pi / 2)), 3)


def build_plan(protocol, old_rows, old_split):
    """Use positions and split metadata only; no predictions or error-based sampling."""
    edges = protocol["radial_edges_m"]
    grouped = {}
    for row in old_rows:
        grouped.setdefault(row["position_id"], []).append(row)
    positions = []
    for split, old_name in (("validation", "validation"), ("legacy_test", "test")):
        for name in sorted(old_split["groups"][old_name]):
            rows = sorted(grouped[name], key=lambda row: row["heading_index"])
            positions.append(
                dict(
                    position_id=name,
                    split=split,
                    xyz_m=[rows[0][k] for k in ("x", "y", "z")],
                    yaws_deg=[r["yaw"] for r in rows],
                    legacy=True,
                )
            )
    held = [p["xyz_m"][:2] for p in positions]
    excluded = []
    for name in sorted(old_split["groups"]["train"]):
        rows = sorted(grouped[name], key=lambda row: row["heading_index"])
        xy = [rows[0]["x"], rows[0]["y"]]
        if min(math.dist(xy, p) for p in held) < protocol["train_holdout_buffer_m"]:
            excluded.append(name)
            continue
        positions.append(
            dict(
                position_id=name,
                split="train",
                xyz_m=xy + [protocol["height_agl_m"]],
                yaws_deg=[r["yaw"] for r in rows],
                legacy=True,
            )
        )
    old_xy = [[rows[0]["x"], rows[0]["y"]] for rows in grouped.values()]
    rng = np.random.RandomState(protocol["seed"])
    yaw_rng = np.random.RandomState(protocol["seed"])
    for split in ("validation", "new_test", "train"):
        for bin_index, quota in enumerate(protocol["position_quotas"][split]):
            count = sum(
                p["split"] == split and radial_bin(*p["xyz_m"][:2], edges) == bin_index
                for p in positions
            )
            if count > quota:
                raise ValueError("Existing positions exceed a bin quota")
            radii = np.sqrt(
                rng.uniform(
                    edges[bin_index] ** 2,
                    edges[bin_index + 1] ** 2,
                    protocol["candidate_pool_per_bin"],
                )
            )
            angles = rng.uniform(0, 2 * np.pi, len(radii))
            candidates = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            if split != "train":
                references = old_xy + [p["xyz_m"][:2] for p in positions if not p["legacy"]]
                distance = square_distances(candidates, references)
                threshold = protocol["fresh_holdout_distance_from_all_legacy_m"]
                selection_quadrants = []
                for sector, sector_quota in enumerate(
                    protocol["holdout_quadrant_quotas_per_radial_bin"][split]
                ):
                    present = sum(
                        p["split"] == split
                        and radial_bin(*p["xyz_m"][:2], edges) == bin_index
                        and quadrant(*p["xyz_m"][:2]) == sector
                        for p in positions
                    )
                    if present > sector_quota:
                        raise ValueError("Legacy holdout exceeds a quadrant quota")
                    selection_quadrants.extend([sector] * (sector_quota - present))
                candidate_quadrants = (angles / (np.pi / 2)).astype(int)
            else:
                held = [p["xyz_m"][:2] for p in positions if p["split"] != "train"]
                safe = square_distances(candidates, held) >= protocol["train_holdout_buffer_m"] ** 2
                candidates = candidates[safe]
                distance = square_distances(candidates, [p["xyz_m"][:2] for p in positions])
                threshold = protocol["fresh_train_spacing_m"]
                selection_quadrants = [None] * (quota - count)
            if len(selection_quadrants) != quota - count:
                raise ValueError("Angular quotas do not sum to the radial quota")
            for index, sector in enumerate(selection_quadrants, start=count):
                ranked = (
                    distance
                    if sector is None
                    else np.where(candidate_quadrants == sector, distance, -1)
                )
                chosen = int(ranked.argmax())
                if distance[chosen] < threshold ** 2:
                    raise ValueError("Candidate pool cannot satisfy spatial spacing")
                xy = candidates[chosen].tolist()
                yaw = float(yaw_rng.random_sample() * 720)
                positions.append(
                    dict(
                        position_id="fresh_{}_b{}_{:03d}".format(split, bin_index, index),
                        split=split,
                        xyz_m=xy + [protocol["height_agl_m"]],
                        yaws_deg=[(yaw + i * 45) % 360 for i in range(8)],
                        legacy=False,
                    )
                )
                distance = np.minimum(
                    distance, ((candidates - candidates[chosen]) ** 2).sum(axis=1)
                )
    positions.sort(key=lambda p: (PARTITIONS.index(p["split"]), p["position_id"]))
    plan = {
        "schema_version": 1,
        "protocol_sha256": canonical_hash(protocol),
        "positions": positions,
        "excluded_legacy_train_positions": excluded,
    }
    plan["audit"] = audit_plan(plan, protocol, old_split)
    return plan


def audit_plan(plan, protocol, old_split):
    positions = plan["positions"]
    if len({p["position_id"] for p in positions}) != len(positions):
        raise ValueError("Duplicate position ID")
    counts = {}
    for split in PARTITIONS:
        selected = [p for p in positions if p["split"] == split]
        bins = [0] * 4
        angular_counts = [[0] * 4 for _ in range(4)]
        for p in selected:
            bin_index = radial_bin(*p["xyz_m"][:2], protocol["radial_edges_m"])
            bins[bin_index] += 1
            angular_counts[bin_index][quadrant(*p["xyz_m"][:2])] += 1
            if len(p["yaws_deg"]) != 8 or not math.isclose(
                p["xyz_m"][2], protocol["height_agl_m"], rel_tol=0, abs_tol=1e-5
            ):
                raise ValueError("Incorrect height or heading count")
        if split != "legacy_test" and bins != protocol["position_quotas"][split]:
            raise ValueError("Partition radial quota mismatch")
        if split in ("validation", "new_test"):
            expected = protocol["holdout_quadrant_quotas_per_radial_bin"][split]
            if any(count != expected for count in angular_counts):
                raise ValueError("Holdout angular coverage quota mismatch")
        counts[split] = {
            "positions": len(selected),
            "images": len(selected) * 8,
            "bins": bins,
            "quadrants_per_radial_bin": angular_counts,
        }
    actual_old_test = {p["position_id"] for p in positions if p["split"] == "legacy_test"}
    if actual_old_test != set(old_split["groups"]["test"]):
        raise ValueError("Legacy test partition changed")
    train = [p["xyz_m"][:2] for p in positions if p["split"] == "train"]
    held = [p["xyz_m"][:2] for p in positions if p["split"] != "train"]
    minimum = float(np.sqrt(square_distances(np.array(train), held).min()))
    if minimum + 1e-9 < protocol["train_holdout_buffer_m"]:
        raise ValueError("Training center violates holdout buffer")
    return {
        "positions": len(positions),
        "images": len(positions) * 8,
        "counts": counts,
        "reused_positions": sum(p["legacy"] for p in positions),
        "fresh_images": sum(not p["legacy"] for p in positions) * 8,
        "minimum_train_holdout_center_distance_m": minimum,
        "footprint_overlap_possible": True,
    }


def source_paths():
    return [
        p
        for p in sorted(PROJECT.rglob("*.py"))
        if not {"runs", ".git", "__pycache__"}.intersection(p.relative_to(PROJECT).parts)
    ]


def prepare_plan(folder):
    folder = Path(folder)
    protocol = read_json(PROJECT / PROTOCOL)
    prepared = PROJECT / protocol["legacy_prepared"]
    rows = read_json(prepared / "records.json")
    split = read_json(prepared / "splits.json")
    plan = build_plan(protocol, rows, split)
    old_config = read_json(PROJECT / protocol["legacy_capture"] / "resolved_config.json")
    paths = source_paths() + [
        PROJECT / PROTOCOL,
        prepared / "records.json",
        prepared / "splits.json",
        PROJECT / protocol["legacy_weights"],
        PROJECT / protocol["scene"],
    ]
    bundle = old_config["provenance"]["scene_bundle"]
    scene_dir = (PROJECT / protocol["scene"]).parent
    for relative, expected in bundle["asset_sha256"].items():
        path = scene_dir / relative
        if digest(path) != expected:
            raise ValueError("Legacy scene dependency changed: " + relative)
        paths.append(path)
    if digest(PROJECT / protocol["scene"]) != old_config["provenance"]["scene_sha256"]:
        raise ValueError("Legacy scene changed")
    ledger = {str(p.relative_to(PROJECT)): digest(p) for p in paths}
    freeze = {"protocol": protocol, "plan": plan, "files": ledger}
    freeze["fingerprint"] = canonical_hash(freeze)
    if (folder / "frozen.json").exists():
        if read_json(folder / "frozen.json") != freeze:
            raise ValueError("Frozen experiment changed; use a new run directory")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder / "frozen.json", freeze)
        atomic_json(folder / "plan.json", plan)
        atomic_json(folder / "protocol.json", protocol)
    return freeze


def verify_freeze(folder):
    freeze = read_json(Path(folder) / "frozen.json")
    if (
        canonical_hash({k: v for k, v in freeze.items() if k != "fingerprint"})
        != freeze["fingerprint"]
    ):
        raise ValueError("Frozen experiment fingerprint mismatch")
    for name, expected in freeze["files"].items():
        if digest(PROJECT / name) != expected:
            raise ValueError("Frozen experiment source/asset changed: " + name)
    return freeze
