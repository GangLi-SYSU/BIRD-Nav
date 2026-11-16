"""Composite captured-data audit and explicit, position-disjoint training partitions."""

import math
from pathlib import Path

from insect_utils.camera_geometry import home_vector_body
from insect_utils.coverage_capture import check_observation, committed_records, planned_samples
from insect_utils.coverage_plan import PARTITIONS, verify_freeze
from insect_utils.experiment_config import PROJECT, canonical_hash, digest
from insect_utils.experiment_state import atomic_json, read_json


def prepare_data(folder):
    from PIL import Image
    from insect_utils.data_preparation import verify_capture_source

    folder = Path(folder)
    freeze = verify_freeze(folder)
    protocol = freeze["protocol"]
    verify_capture_source(protocol["legacy_capture"], full=True)
    old_folder = PROJECT / protocol["legacy_capture"] / "capture"
    old_rows = read_json(PROJECT / protocol["legacy_prepared"] / "records.json")
    old = {(r["position_id"], r["heading_index"]): r for r in old_rows}
    captured = committed_records(folder)
    samples = planned_samples(freeze["plan"])
    if set(captured) != {s["sample_id"] for s in samples}:
        raise ValueError("Fresh capture is incomplete or has extra samples")
    for sample in samples:
        check_observation(sample, captured[sample["sample_id"]], protocol)
    fresh = {(r["position_id"], r["heading_index"]): r for r in captured.values()}
    records = []
    for p in freeze["plan"]["positions"]:
        for heading in range(8):
            key = (p["position_id"], heading)
            row = dict(old[key] if p["legacy"] else fresh[key])
            if p["legacy"]:
                row["image"] = str((old_folder / row["image"]).relative_to(PROJECT))
                row["sample_id"] = "legacy_" + row["sample_id"]
            row.update(split=p["split"], legacy=p["legacy"])
            wanted = home_vector_body([row[k] for k in ("x", "y", "z")], [0, 0, 40], row["yaw"])
            if any(abs(row[k] - v) > 1e-8 for k, v in zip(("target_x", "target_y"), wanted)):
                raise ValueError("Physical target does not reconstruct the home location")
            if any(abs(row[k] - v) > 1e-5 for k, v in zip(("x", "y", "z"), p["xyz_m"])):
                raise ValueError("Captured location differs from frozen plan")
            if abs((row["yaw"] - p["yaws_deg"][heading] + 180) % 360 - 180) > 1e-5:
                raise ValueError("Captured heading differs from frozen plan")
            image = PROJECT / row["image"]
            if digest(image) != row["sha256"]:
                raise ValueError("Captured image hash mismatch")
            with Image.open(image) as opened:
                if opened.mode != "RGB" or list(opened.size) != [1024, 1024]:
                    raise ValueError("Expected complete1024-square RGB images")
                opened.verify()
            records.append(row)
    audit_records(records)
    payload = {
        "fingerprint": freeze["fingerprint"],
        "records": records,
        "records_sha256": canonical_hash(records),
        "plan_audit": freeze["plan"]["audit"],
        "legacy_test_bytes_unchanged": True,
        "raw_resolution_wh": [1024, 1024],
        "target_units": "body-frame home vector in meters",
        "augmentation": False,
    }
    destination = folder / "dataset.json"
    if destination.exists() and read_json(destination) != payload:
        raise ValueError("Prepared dataset differs from its frozen committed observations")
    atomic_json(destination, payload)
    atomic_json(folder / "data_audit.json", {k: v for k, v in payload.items() if k != "records"})
    return payload


def audit_records(records):
    for field in ("sample_id", "image"):
        if len({r[field] for r in records}) != len(records):
            raise ValueError("Duplicate sample or image")
    groups = {}
    for r in records:
        groups.setdefault(r["position_id"], []).append(r)
        if r["split"] not in PARTITIONS or not all(
            math.isfinite(r[k]) for k in ("target_x", "target_y")
        ):
            raise ValueError("Invalid partition or non-finite target")
    for rows in groups.values():
        if len(rows) != 8 or {r["heading_index"] for r in rows} != set(range(8)):
            raise ValueError("Missing or duplicated physical headings")
        if len({r["split"] for r in rows}) != 1:
            raise ValueError("Position leakage between partitions")


def load_data(folder):
    value = read_json(Path(folder) / "dataset.json")
    freeze = verify_freeze(folder)
    if value["fingerprint"] != freeze["fingerprint"]:
        raise ValueError("Prepared dataset belongs to a different experiment")
    if canonical_hash(value["records"]) != value["records_sha256"]:
        raise ValueError("Prepared record bytes changed")
    audit_records(value["records"])
    return value["records"]


def make_loader(records, split, scale=1.0, workers=4, shuffle=False, allow_test=False):
    """Only explicit final-evaluation callers may open a test partition."""
    import torch
    from torch.utils.data import DataLoader
    from insect_utils.coverage_model import CoverageDataset
    from insect_utils.downward_data import seed_worker

    dataset = CoverageDataset(records, split, scale, allow_test=allow_test)
    return DataLoader(
        dataset,
        batch_size=8,
        shuffle=shuffle,
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(10),
    )
