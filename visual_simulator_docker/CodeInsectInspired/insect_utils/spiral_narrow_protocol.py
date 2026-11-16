"""Freeze spiral-only acquisition, fresh training and requested10/20m evaluation."""

import math
from pathlib import Path

import numpy as np

from insect_utils.coverage_capture import check_observation, committed_records, planned_samples
from insect_utils.coverage_plan import verify_freeze
from insect_utils.data_preparation import grouped_split, validate_records
from insect_utils.experiment_config import PROJECT, canonical_hash, digest
from insect_utils.experiment_state import atomic_json, read_json
from insect_utils.flight_path_functions import generate_bee_path
from insect_utils.navigation_core import make_episodes, safe_segment
from insect_utils.scene_geometry import check_positions_and_segments, require_assets, spatial_budget


PROTOCOL = "config/experiments/stage22_spiral_h40_fov30_protocol.json"


def build_plan(protocol, permutation=None):
    learning = protocol["learning"]
    positions = generate_bee_path(*learning["shape_params"], [0, 0, protocol["height_agl_m"]])
    if len(positions) != learning["position_count"] or len(positions) != 147:
        raise ValueError("Expected exactly the original147 spiral positions")
    if permutation is None:
        import torch

        permutation = torch.randperm(
            len(positions), generator=torch.Generator().manual_seed(protocol["seed"])
        ).tolist()
    groups = [{"position_id": "position_{:06d}".format(i)} for i in range(len(positions))]
    split = grouped_split(groups, permutation, learning["split_ratio"])
    membership = {p: name for name, ids in split["groups"].items() for p in ids}
    yaws = np.random.RandomState(protocol["seed"]).random_sample(len(positions)) * 720
    rows = []
    for index, (position, yaw) in enumerate(zip(positions, yaws)):
        identifier = groups[index]["position_id"]
        rows.append(
            dict(
                position_id=identifier,
                split=membership[identifier],
                xyz_m=list(position),
                yaws_deg=[float((yaw + i * 45) % 360) for i in range(8)],
                legacy=False,
                trajectory_index=index,
            )
        )
    counts = {name: len(ids) for name, ids in split["groups"].items()}
    if counts != dict(train=133, validation=6, test=8):
        raise ValueError("Original position-group split proportions changed")
    return dict(
        positions=rows,
        group_split=split["groups"],
        group_permutation=split["group_permutation"],
        audit=dict(
            positions=len(rows),
            images=len(rows) * 8,
            groups=counts,
            images_by_split={k: v * 8 for k, v in counts.items()},
            off_trajectory_positions=0,
            reused_images=0,
            maximum_radius_m=max(math.hypot(*p[:2]) for p in positions),
            minimum_radius_m=min(math.hypot(*p[:2]) for p in positions),
        ),
    )


def scene_budget(protocol):
    nav = protocol["navigation"]
    spec = dict(
        camera=protocol["camera"],
        scene=dict(ground_z_m=nav["ground_z_m"]),
        flight=dict(camera_height_agl_m=protocol["height_agl_m"]),
        home=dict(ground_xy_m=nav["home_camera_xyz_m"][:2]),
        evaluation=dict(
            radius_m=max(nav["formal"]["radii_m"]),
            radius_scales=[1],
            max_steps=nav["controller"]["max_steps"],
            max_step_m=nav["controller"]["max_step_m"],
            pitch_roll_deg=0,
        ),
    )
    manifest = require_assets(PROJECT / protocol["scene"])
    budget = spatial_budget(spec, manifest, nav["runtime"]["safety_margin_m"])
    budget["boundary_guard_connected_to_controller"] = True
    return budget


def prepare(folder):
    folder = Path(folder).resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    if (folder / "frozen.json").exists():
        return verify_freeze(folder)
    protocol = read_json(PROJECT / PROTOCOL)
    if protocol["camera"]["hfov_deg"] != 30 or protocol["height_agl_m"] != 40:
        raise ValueError("Expected the user-requested40m/30deg camera")
    if protocol["navigation"]["controller"]["success_radius_m"] != 2:
        raise ValueError("Expected2m navigation success radius")
    plan = build_plan(protocol)
    budget = scene_budget(protocol)
    check_positions_and_segments([p["xyz_m"] for p in plan["positions"]], budget)
    paths = list(PROJECT.glob("*.py")) + list((PROJECT / "insect_utils").glob("*.py"))
    paths += list((PROJECT / protocol["scene"]).parent.rglob("*"))
    paths += [PROJECT / PROTOCOL, PROJECT / "tests/test_stage22_spiral_narrow.py"]
    value = dict(
        protocol=protocol,
        plan=plan,
        budget=budget,
        files={str(p.relative_to(PROJECT)): digest(p) for p in sorted(paths) if p.is_file()},
    )
    value["fingerprint"] = canonical_hash(value)
    atomic_json(folder / "frozen.json", value)
    atomic_json(folder / "protocol.json", protocol)
    atomic_json(folder / "plan.json", plan)
    return value


def audit_rows(records, frozen):
    samples = planned_samples(frozen["plan"])
    expected = {s["sample_id"]: s for s in samples}
    if len(records) != len(expected) or {r["sample_id"] for r in records} != set(expected):
        raise ValueError("Dataset must contain every spiral image exactly once")
    validate_records(records, [0, 0], headings=8)
    for row in records:
        sample = expected[row["sample_id"]]
        if row["position_id"] != sample["position_id"] or row["split"] != sample["split"]:
            raise ValueError("Position/split differs from the predeclared spiral plan")
        if any(
            abs(row[k] - v) > 1e-5 for k, v in zip(("x", "y", "z"), sample["planned_position_m"])
        ):
            raise ValueError("Off-trajectory observation in spiral-only dataset")
        if row["heading_index"] != sample["heading_index"]:
            raise ValueError("Heading differs from frozen spiral sample")
        if abs((row["yaw"] - sample["planned_body_yaw_deg"] + 180) % 360 - 180) > 1e-5:
            raise ValueError("Yaw differs from frozen spiral sample")
    return frozen["plan"]["audit"]


def prepare_data(folder):
    folder = Path(folder)
    frozen = verify_freeze(folder)
    captured = committed_records(folder)
    records = []
    for sample in planned_samples(frozen["plan"]):
        if sample["sample_id"] not in captured:
            raise ValueError("Spiral capture is incomplete")
        row = dict(captured[sample["sample_id"]], split=sample["split"])
        check_observation(sample, row, frozen["protocol"])
        records.append(row)
    if len(captured) != len(records):
        raise ValueError("Unexpected extra capture outside the spiral plan")
    audit_rows(records, frozen)
    payload = dict(
        fingerprint=frozen["fingerprint"],
        records=records,
        records_sha256=canonical_hash(records),
        audit=frozen["plan"]["audit"],
        spiral_only=True,
        old_images_used=False,
        test_images_used_for_training_or_selection=False,
    )
    path = folder / "dataset.json"
    if path.exists() and read_json(path) != payload:
        raise ValueError("Previously prepared spiral dataset changed")
    atomic_json(path, payload)
    return payload


def load_data(folder):
    frozen = verify_freeze(folder)
    payload = read_json(Path(folder) / "dataset.json")
    if payload["fingerprint"] != frozen["fingerprint"]:
        raise ValueError("Dataset belongs to another experiment")
    if canonical_hash(payload["records"]) != payload["records_sha256"]:
        raise ValueError("Spiral dataset records changed")
    audit_rows(payload["records"], frozen)
    return payload["records"]


def prepare_navigation(folder):
    folder = Path(folder)
    frozen = verify_freeze(folder)
    completed = read_json(folder / "training/completion.json")
    if completed["status"] != "completed" or completed["epochs"] != 600:
        raise ValueError("Complete600 fresh epochs before navigation")
    if completed["experiment_fingerprint"] != frozen["fingerprint"]:
        raise ValueError("Training belongs to another frozen experiment")
    files = dict(frozen["files"])
    for name, expected in completed["files"].items():
        path = folder / "training" / name
        if digest(path) != expected:
            raise ValueError("Completed training artifact changed")
        files[str(path.relative_to(PROJECT))] = expected
    for name in ("dataset.json", "training/completion.json"):
        path = folder / name
        files[str(path.relative_to(PROJECT))] = digest(path)
    protocol = dict(frozen["protocol"]["navigation"], camera=frozen["protocol"]["camera"])
    episodes = {
        phase: make_episodes(protocol[phase], protocol["home_camera_xyz_m"], phase)
        for phase in ("rendered_oracle", "formal")
    }
    if len(episodes["formal"]) != 128 or set(protocol["formal"]["radii_m"]) != {10, 20}:
        raise ValueError("Only64 neural trials each at10m and20m are authorized")
    for group in episodes.values():
        for episode in group:
            p = episode["initial_position_m"]
            safe_segment(p, p, frozen["budget"])
    model_dir = folder / "training"
    value = dict(
        protocol=protocol,
        episodes=episodes,
        budget=frozen["budget"],
        files=files,
        model=dict(
            arm="fresh_spiral30_mask_rotation",
            epoch=completed["best_validation"]["epoch"],
            weights=str((model_dir / "best_validation.pth").relative_to(PROJECT)),
            torchscript=str(
                (model_dir / "deployment/home_vector_m.torchscript.pt").relative_to(PROJECT)
            ),
            scale_m=44.0,
            output_units="body-frame home vector in meters",
        ),
        selection_frozen_at=completed["updated"],
    )
    value["fingerprint"] = canonical_hash(value)
    destination = folder / "navigation/frozen.json"
    if destination.exists() and read_json(destination) != value:
        raise ValueError("Navigation selection already frozen; use another run")
    atomic_json(destination, value)
    atomic_json(destination.with_name("episode_plan.json"), episodes)
    return value
