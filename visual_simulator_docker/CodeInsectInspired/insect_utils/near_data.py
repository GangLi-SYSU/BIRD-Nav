"""Geometry-only antipodal local sampling and explicit import of new captured data."""

import math
from pathlib import Path

import numpy as np

from insect_utils.capture_geometry import make_samples
from insect_utils.coverage_data import audit_records
from insect_utils.experiment_config import PROJECT, canonical_hash, digest
from insect_utils.experiment_state import atomic_json, read_json
from insect_utils.local_probe_runtime import verify_capture
from insect_utils.navigation_core import safe_segment
from insect_utils.navigation_protocol import verify_navigation


PROTOCOL = "config/experiments/stage19_near_data_protocol.json"


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def original_centers(records, split):
    return list(
        {r["position_id"]: (r["x"], r["y"]) for r in records if r["split"] in split}.values()
    )


def make_positions(spec, records):
    rng = np.random.RandomState(spec["seed"])
    original_train = original_centers(records, {"train"})
    original_hold = original_centers(records, {"validation", "legacy_test", "new_test"})
    positions = []
    holds = []
    trains = []
    edges = spec["band_edges_m"]

    def add(split, band, xy):
        index = sum(p["split"] == split for p in positions)
        item = dict(
            position_id="near_{}_{:04d}".format(split, index),
            split=split,
            band=band,
            xyz_m=[float(xy[0]), float(xy[1]), 40.0],
        )
        positions.append(item)
        (trains if split == "train" else holds).append(item["xyz_m"][:2])

    def valid(xy, split):
        if split == "train":
            tests = [
                (original_hold, spec["old_holdout_train_separation_m"]),
                (holds, spec["new_holdout_separation_m"]),
                (original_train + trains, spec["new_train_min_spacing_m"]),
            ]
        else:
            tests = [(original_train + original_hold + holds, spec["new_holdout_separation_m"])]
        return all(distance(xy, p) >= minimum - 1e-10 for points, minimum in tests for p in points)

    def pairs(split, band, counts):
        low, high = edges[band], edges[band + 1]
        if split != "train" and band == 0:
            low = spec["inner_holdout_radius_min_m"]
        # QI/QIII and QII/QIV pairs keep each new partition symmetric about home.
        for quadrant, count in enumerate(counts):
            found = 0
            for _ in range(200000):
                if found == count:
                    break
                radius = math.sqrt(rng.uniform(low * low, high * high))
                angle = rng.uniform(quadrant * math.pi / 2, (quadrant + 1) * math.pi / 2)
                xy = [radius * math.cos(angle), radius * math.sin(angle)]
                opposite = [-xy[0], -xy[1]]
                if valid(xy, split) and valid(opposite, split) and distance(xy, opposite) >= 0.15:
                    add(split, band, xy)
                    add(split, band, opposite)
                    found += 1
            if found != count:
                raise ValueError("Cannot satisfy local spatial separation for " + split)

    for split, counts in (
        ("validation", spec["validation_counts_by_band"]),
        ("reserved_test", spec["reserved_counts_by_band"]),
    ):
        for band, count in enumerate(counts):
            if count % 2:
                raise ValueError("Holdout counts must support antipodal pairs")
            half = count // 2
            pairs(split, band, [(half + 1) // 2, half // 2])
    if not valid([0, 0], "train"):
        raise ValueError("Home anchor violates the declared separation")
    add("train", 0, [0, 0])
    for band, count in enumerate(spec["train_counts_by_band"]):
        remaining = count - (1 if band == 0 else 0)
        if remaining % 4:
            raise ValueError("Training rings must balance all4 quadrants")
        pairs("train", band, [remaining // 4, remaining // 4])
    return positions


def build_plan():
    spec = read_json(PROJECT / PROTOCOL)
    source_folder = PROJECT / spec["source_probe_run"]
    source = verify_navigation(source_folder)
    completed = read_json(source_folder / "completion.json")
    if (
        source["fingerprint"] != spec["source_probe_fingerprint"]
        or completed["status"] != "completed"
        or completed["images"] != 104
        or source["model"]["epoch"] != 589
    ):
        raise ValueError("Completed104-image diagnostic source changed")
    data = PROJECT / spec["source_dataset_run"] / "dataset.json"
    if digest(data) != spec["source_dataset_sha256"]:
        raise ValueError("Original dataset changed")
    original = read_json(data)["records"]
    positions = make_positions(spec, original)
    selected = [p for p in positions if p["split"] != "reserved_test"]
    samples = make_samples(
        [p["xyz_m"] for p in selected], [spec["headings_deg"] for _ in selected], [0, 0, 40], 0
    )
    for sample in samples:
        point = selected[int(sample["position_id"].split("_")[-1])]
        sample.update(
            position_id=point["position_id"],
            split_group=point["position_id"],
            split=point["split"],
            band=point["band"],
            role="explicit_near_data_capture",
        )
        safe_segment(sample["planned_position_m"], sample["planned_position_m"], source["budget"])
    files = dict(source["files"])
    extras = [
        PROJECT / PROTOCOL,
        source_folder / "frozen.json",
        source_folder / "completion.json",
        data,
        PROJECT / "run_near_data.py",
        PROJECT / "insect_utils/near_data.py",
        PROJECT / "insect_utils/near_data_report.py",
        PROJECT / "tests/test_stage19_near_data.py",
    ]
    files.update({str(p.relative_to(PROJECT)): digest(p) for p in extras})
    value = dict(
        spec=spec,
        protocol=source["protocol"],
        model=source["model"],
        budget=source["budget"],
        positions=positions,
        samples=samples,
        files=files,
    )
    value["fingerprint"] = canonical_hash(value)
    return value


def prepare(folder):
    folder = Path(folder).resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    value = build_plan()
    path = folder / "frozen.json"
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Near-data plan is already frozen")
    else:
        atomic_json(path, value)
        atomic_json(folder / "position_plan.json", value["positions"])
        atomic_json(
            folder / "reserved_positions.json",
            [p for p in value["positions"] if p["split"] == "reserved_test"],
        )
    return value


def prepare_dataset(folder):
    folder = Path(folder)
    frozen = verify_navigation(folder)
    capture = verify_capture(folder)
    old = read_json(PROJECT / frozen["spec"]["source_dataset_run"] / "dataset.json")["records"]
    records = [dict(r, origin="original") for r in old if r["split"] in ("train", "validation")]
    samples = {s["sample_id"]: s for s in frozen["samples"]}
    for row in capture["rows"]:
        sample = samples[row["sample_id"]]
        records.append(
            dict(
                sample,
                sample_id="near_" + row["sample_id"],
                x=row["position_m"][0],
                y=row["position_m"][1],
                z=row["position_m"][2],
                yaw=row["body_yaw_deg"],
                target_x=row["target_body_m"][0],
                target_y=row["target_body_m"][1],
                image=row["image"],
                sha256=row["image_sha256"],
                legacy=False,
                origin="near_supplement",
            )
        )
    audit_records(records)
    if (
        sum(r["split"] == "train" for r in records) != 5640
        or sum(r["split"] == "validation" for r in records) != 448
    ):
        raise ValueError("Unexpected combined training/validation count")
    payload = dict(
        fingerprint=frozen["fingerprint"],
        records=records,
        records_sha256=canonical_hash(records),
        reserved_positions_captured=False,
        source_original_sha256=frozen["spec"]["source_dataset_sha256"],
        training_near_5m_fraction=sum(
            r["split"] == "train" and math.hypot(r["x"], r["y"]) <= 5 for r in records
        )
        / 5640,
        limits=frozen["spec"]["limits"],
    )
    file = folder / "dataset.json"
    if file.exists() and read_json(file) != payload:
        raise ValueError("Prepared near dataset changed")
    atomic_json(file, payload)
    return payload
