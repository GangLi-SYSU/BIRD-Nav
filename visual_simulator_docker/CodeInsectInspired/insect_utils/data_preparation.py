"""Frozen capture lineage and position-group splits without simulation imports."""

import json
import math
from pathlib import Path

from insect_utils.camera_geometry import body_to_world
from insect_utils.experiment_config import PROJECT, canonical_hash, digest, relative_path
from insect_utils.experiment_state import read_json, verify_completed


DATA_PROTOCOL = "config/experiments/stage06_data_protocol.json"
SPLITS = ("train", "validation", "test")
CORE_SOURCES = (
    "insect_utils/rgb_preprocessing.py",
    "insect_utils/data_preparation.py",
    "insect_utils/downward_data.py",
    "simple_network.py",
)
RECORD_FIELDS = (
    "sample_id",
    "pose_id",
    "frame_id",
    "image",
    "position_id",
    "heading_index",
    "x",
    "y",
    "z",
    "yaw",
    "target_x",
    "target_y",
    "camera_id",
    "split_group",
    "ground_z_m",
    "agl_m",
    "role",
    "sha256",
)


def project_path(value, project=PROJECT):
    path = Path(value)
    path = path if path.is_absolute() else project / relative_path(str(value))
    path.resolve().relative_to(project.resolve())
    return path


def verify_capture_source(run, project=PROJECT, full=True):
    """Verify historical capture artifacts against their own frozen fingerprint."""
    run = project_path(run, project)
    stored = read_json(run / "resolved_config.json")
    payload = {key: stored[key] for key in ("spec", "derived", "provenance")}
    if canonical_hash(payload) != stored["fingerprint"]:
        raise ValueError("Captured configuration was modified")
    if read_json(run / "run_manifest.json")["fingerprint"] != stored["fingerprint"]:
        raise ValueError("Capture run manifest fingerprint mismatch")
    state = read_json(run / "state/capture.json")
    manifest = read_json(run / "capture/dataset_manifest.json")
    if (
        state["status"] != "completed"
        or state["fingerprint"] != stored["fingerprint"]
        or manifest["fingerprint"] != stored["fingerprint"]
    ):
        raise ValueError("Source capture is not completed under its frozen fingerprint")
    if manifest["camera_mode"] != "downward_pinhole" or manifest["producer"] != "isaac":
        raise ValueError("Stage F requires a real downward capture")
    if manifest["purpose"] != "research" or manifest["role"] != "learning_only":
        raise ValueError("Engineering fixtures/evaluation data are not a learning dataset")
    if full:
        verify_completed(run, "capture", stored["fingerprint"])
    reference = {
        "run": str(run.resolve().relative_to(project.resolve())),
        "fingerprint": stored["fingerprint"],
        "resolved_config_sha256": digest(run / "resolved_config.json"),
        "run_manifest_sha256": digest(run / "run_manifest.json"),
        "capture_state_sha256": digest(run / "state/capture.json"),
        "capture_manifest_sha256": digest(run / "capture/dataset_manifest.json"),
        "records_sha256": digest(run / "capture/samples.jsonl"),
    }
    return reference, stored, manifest


def validate_records(records, home_xy, headings=8):
    if not records:
        raise ValueError("Prepared dataset is empty")
    names, images, groups = set(), set(), {}
    for row in records:
        relative_path(row["image"])
        if row["sample_id"] in names or row["image"] in images:
            raise ValueError("Duplicate sample/image in prepared data")
        names.add(row["sample_id"])
        images.add(row["image"])
        if row["role"] != "learning" or row["split_group"] != row["position_id"]:
            raise ValueError("Invalid learning role or position group")
        values = [
            row[k] for k in ["x", "y", "z", "yaw", "target_x", "target_y", "ground_z_m", "agl_m"]
        ]
        if not all(
            isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)
            for v in values
        ):
            raise ValueError("Non-finite position or label")
        world = body_to_world([row["target_x"], row["target_y"]], row["yaw"])
        if any(abs(row[k] + world[i] - home_xy[i]) > 1e-6 for i, k in enumerate(["x", "y"])):
            raise ValueError("Prepared label does not reconstruct home")
        if not math.isclose(row["z"] - row["ground_z_m"], row["agl_m"], abs_tol=1e-6, rel_tol=0):
            raise ValueError("Prepared AGL is inconsistent")
        groups.setdefault(row["position_id"], []).append(row)
    for rows in groups.values():
        if len(rows) != headings or {r["heading_index"] for r in rows} != set(range(headings)):
            raise ValueError("Position has incomplete/duplicate headings")
        first = rows[0]
        if any(abs(r[k] - first[k]) > 1e-6 for r in rows for k in ["x", "y", "z"]):
            raise ValueError("Position group contains different camera locations")
        ordered = sorted(rows, key=lambda r: r["heading_index"])
        for a, b in zip(ordered, ordered[1:] + ordered[:1]):
            if abs((b["yaw"] - a["yaw"]) % 360 - (360 / headings) % 360) > 1e-6:
                raise ValueError("Position heading spacing changed")
    return sorted(groups)


def grouped_split(records, permutation, ratio=0.95):
    groups = sorted({r["position_id"] for r in records})
    if sorted(permutation) != list(range(len(groups))):
        raise ValueError("Group permutation must cover each position exactly once")
    if isinstance(ratio, bool) or not isinstance(ratio, (float, int)) or not 0 < ratio < 1:
        raise ValueError("Invalid split ratio")
    ordered = [groups[i] for i in permutation]
    pool = int(ratio * len(groups))
    validation = int((1 - ratio) * pool)
    if min(validation, pool - validation, len(groups) - pool) < 1:
        raise ValueError("Too few positions for the declared three-way split")
    selected = {
        "train": ordered[validation:pool],
        "validation": ordered[:validation],
        "test": ordered[pool:],
    }
    result = {
        "schema_version": 1,
        "seed": 10,
        "ratio": ratio,
        "group_permutation": ordered,
        "groups": selected,
        "indices": {
            name: [i for i, r in enumerate(records) if r["position_id"] in set(selected[name])]
            for name in SPLITS
        },
    }
    verify_split(records, result)
    return result


def verify_split(records, split):
    groups = set()
    indices = set()
    expected = {r["position_id"] for r in records}
    for name in SPLITS:
        chosen = split["indices"][name]
        positions = split["groups"][name]
        if (
            not chosen
            or not positions
            or len(chosen) != len(set(chosen))
            or len(positions) != len(set(positions))
        ):
            raise ValueError("Empty or duplicated split members")
        if any(
            isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(records)
            for i in chosen
        ):
            raise ValueError("Invalid sample index")
        if groups.intersection(positions) or indices.intersection(chosen):
            raise ValueError("A sample/position leaks across splits")
        if {records[i]["position_id"] for i in chosen} != set(positions):
            raise ValueError("Split groups and sample indices disagree")
        if set(chosen) != {i for i, r in enumerate(records) if r["position_id"] in positions}:
            raise ValueError("Split lost one or more headings at a position")
        groups.update(positions)
        indices.update(chosen)
    if groups != expected or indices != set(range(len(records))):
        raise ValueError("Splits do not cover the complete learning dataset")


def load_prepared(folder, project=PROJECT):
    folder = project_path(folder, project)
    manifest = read_json(folder / "preparation_manifest.json")
    payload = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if canonical_hash(payload) != manifest["fingerprint"] or manifest["status"] != "prepared":
        raise ValueError("Prepared manifest changed or is incomplete")
    for name, expected in manifest["files_sha256"].items():
        if digest(folder / relative_path(name)) != expected:
            raise ValueError("Prepared artifact hash changed: " + name)
    if digest(project / DATA_PROTOCOL) != manifest["data_protocol_sha256"]:
        raise ValueError("Data protocol changed; prepare a new stage F output")
    for name, expected in manifest["core_source_sha256"].items():
        if digest(project / name) != expected:
            raise ValueError("Data/network source changed; revalidate stage F: " + name)
    reference = read_json(folder / "source_reference.json")
    actual, stored, capture = verify_capture_source(reference["run"], project, full=False)
    if actual != reference:
        raise ValueError("Source capture lineage changed")
    records = read_json(folder / "records.json")
    original = [
        json.loads(line)
        for line in (project_path(reference["run"], project) / "capture/samples.jsonl")
        .read_text()
        .splitlines()
    ]
    if records != [{name: r[name] for name in RECORD_FIELDS} for r in original]:
        raise ValueError("Prepared metadata differs from the captured observations")
    validate_records(
        records, stored["spec"]["home"]["ground_xy_m"], capture["headings_per_position"]
    )
    split = read_json(folder / "splits.json")
    verify_split(records, split)
    if len(records) != capture["sample_count"]:
        raise ValueError("Prepared record count differs from capture")
    if manifest["group_counts"] != {name: len(ids) for name, ids in split["groups"].items()}:
        raise ValueError("Prepared group counts disagree with the split")
    if manifest["image_counts"] != {name: len(ids) for name, ids in split["indices"].items()}:
        raise ValueError("Prepared image counts disagree with the split")
    if any(
        abs(r["agl_m"] - stored["spec"]["flight"]["camera_height_agl_m"]) > 1e-6 for r in records
    ):
        raise ValueError("Prepared altitude differs from the capture protocol")
    return {
        "folder": folder,
        "manifest": manifest,
        "records": records,
        "split": split,
        "source": reference,
        "capture_config": stored,
        "capture_manifest": capture,
        "capture_folder": project_path(reference["run"], project) / "capture",
    }
