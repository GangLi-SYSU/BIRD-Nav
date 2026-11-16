"""Fixed-position probe plans and physical vector diagnostics without Isaac imports."""

import math
from pathlib import Path

from insect_utils.camera_geometry import body_to_world, home_vector_body
from insect_utils.capture_geometry import make_samples
from insect_utils.experiment_config import PROJECT, canonical_hash, digest
from insect_utils.experiment_state import atomic_json, read_json
from insect_utils.navigation_core import safe_segment, shortest_angle_error
from insect_utils.navigation_protocol import verify_navigation
from insect_utils.training_state import distribution


PROTOCOL = "config/experiments/stage18_local_probe_protocol.json"


def probe_samples(design, home):
    if design == "cross13":
        offsets = [(0.0, 0.0)] + [
            (r * x, r * y) for r in (0.5, 1.0, 2.0) for x, y in ((1, 0), (-1, 0), (0, 1), (0, -1))
        ]
    elif design == "grid81":
        offsets = sorted(
            [(x / 2, y / 2) for x in range(-4, 5) for y in range(-4, 5)],
            key=lambda p: (p[0] ** 2 + p[1] ** 2, p[0], p[1]),
        )
    else:
        raise ValueError("Choose the explicitly confirmed cross13 or grid81 probe design")
    positions = [[home[0] + x, home[1] + y, home[2]] for x, y in offsets]
    samples = make_samples(positions, [list(range(0, 360, 45)) for _ in positions], home, 0)
    for sample in samples:
        sample["role"] = "diagnostic_only"
        sample["image"] = "observations/{}/image.png".format(sample["sample_id"])
    return samples


def build_plan(design):
    spec = read_json(PROJECT / PROTOCOL)
    if design not in spec["allowed_designs"]:
        raise ValueError("A user-confirmed diagnostic design is required")
    if (
        spec["headings_deg"] != list(range(0, 360, 45))
        or spec["cross_radii_m"] != [0.5, 1, 2]
        or spec["grid_half_width_m"] != 2
        or spec["grid_spacing_m"] != 0.5
    ):
        raise ValueError("Probe geometry changed; confirm a new protocol before capture")
    source_folder = PROJECT / spec["source_navigation_run"]
    source = verify_navigation(source_folder)
    completion = read_json(source_folder / "completion.json")
    if (
        source["fingerprint"] != spec["source_fingerprint"]
        or completion["fingerprint"] != source["fingerprint"]
        or completion["status"] != "completed"
        or source["model"]["epoch"] != spec["selected_epoch"]
    ):
        raise ValueError("The completed589-epoch navigation source changed")
    samples = probe_samples(design, source["protocol"]["home_camera_xyz_m"])
    for sample in samples:
        safe_segment(sample["planned_position_m"], sample["planned_position_m"], source["budget"])
    replay = read_json(PROJECT / spec["historical_replay"])
    historical = [r for r in replay["samples"] if r["role"].startswith("historical")]
    script_sha = source["files"][source["model"]["torchscript"]]
    if replay["model_sha256"] != script_sha or len(historical) != 2:
        raise ValueError("Historical home observations refer to another model")
    extras = [
        PROJECT / PROTOCOL,
        PROJECT / spec["historical_replay"],
        source_folder / "frozen.json",
        source_folder / "completion.json",
        PROJECT / "run_local_probe.py",
        PROJECT / "tests/test_stage18_local_probe.py",
    ]
    extras += [
        PROJECT / "insect_utils" / name
        for name in ("local_probe.py", "local_probe_runtime.py", "local_probe_report.py")
    ]
    for record in historical:
        image = PROJECT / record["image"]
        if digest(image) != record["image_sha256"]:
            raise ValueError("Historical home image changed")
        extras.extend([image, image.with_suffix(".receipt.json")])
    files = dict(source["files"])
    files.update({str(p.relative_to(PROJECT)): digest(p) for p in extras})
    protocol = {
        k: v for k, v in source["protocol"].items() if k not in ("formal", "rendered_oracle")
    }
    value = dict(
        spec=spec,
        design=design,
        protocol=protocol,
        samples=samples,
        model=source["model"],
        budget=source["budget"],
        historical_home=historical,
        files=files,
    )
    value["fingerprint"] = canonical_hash(value)
    return value


def prepare(folder, design):
    folder = Path(folder).resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    value = build_plan(design)
    path = folder / "frozen.json"
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Diagnostic plan is already frozen; use a new run directory")
    else:
        atomic_json(path, value)
        atomic_json(folder / "sample_plan.json", value["samples"])
    return value


def vector_diagnostic(position, yaw, prediction, home):
    if len(prediction) != 2 or not all(math.isfinite(float(x)) for x in prediction):
        raise ValueError("Expected a finite two-component physical prediction")
    truth = home_vector_body(position, home, yaw)
    world = body_to_world(prediction, yaw)
    estimate = [position[i] + world[i] for i in (0, 1)]
    return dict(
        position_m=list(position),
        body_yaw_deg=yaw,
        target_body_m=truth,
        prediction_body_m=list(prediction),
        prediction_world_m=world,
        predicted_home_world_xy_m=estimate,
        true_distance_m=math.hypot(position[0] - home[0], position[1] - home[1]),
        predicted_distance_m=math.hypot(*prediction),
        vector_error_m=math.hypot(estimate[0] - home[0], estimate[1] - home[1]),
        angle_error_deg=shortest_angle_error(prediction, truth),
    )


def summarize_rows(rows, home):
    if not rows:
        raise ValueError("Cannot summarize an empty diagnostic capture")
    groups = {}
    for row in rows:
        groups.setdefault(row["position_id"], []).append(row)
    positions = []
    for name, group in groups.items():
        estimate = [
            sum(r["predicted_home_world_xy_m"][i] for r in group) / len(group) for i in (0, 1)
        ]
        spread = math.sqrt(
            sum(
                sum((r["predicted_home_world_xy_m"][i] - estimate[i]) ** 2 for i in (0, 1))
                for r in group
            )
            / len(group)
        )
        angles = [r["angle_error_deg"] for r in group if r["angle_error_deg"] is not None]
        positions.append(
            dict(
                position_id=name,
                position_m=group[0]["position_m"],
                samples=len(group),
                true_distance_m=group[0]["true_distance_m"],
                predicted_home_mean_xy_m=estimate,
                heading_dispersion_rms_m=spread,
                vector_error_m=distribution([r["vector_error_m"] for r in group]),
                backwards_fraction=sum(a >= 90 for a in angles) / len(angles) if angles else None,
            )
        )
    home_rows = [r for r in rows if r["true_distance_m"] <= 1e-8]
    return dict(
        images=len(rows),
        positions=len(groups),
        home_output_norm_m=dict(
            distribution([r["predicted_distance_m"] for r in home_rows]),
            maximum=max((r["predicted_distance_m"] for r in home_rows), default=None),
        ),
        per_position=positions,
        vector_error_m=distribution([r["vector_error_m"] for r in rows]),
        notes=(
            "Fixed-position diagnostic observations; headings share a location and scene. "
            "Not navigation success rates."
        ),
    )


def saved_rows(folder, frozen):
    folder = Path(folder)
    expected = {s["sample_id"]: s for s in frozen["samples"]}
    values = {}
    for path in sorted((folder / "observations").glob("*/record.json")):
        record = read_json(path)
        if canonical_hash({k: v for k, v in record.items() if k != "record_sha256"}) != record.get(
            "record_sha256"
        ):
            raise ValueError("Committed diagnostic record digest changed")
        name = record["sample_id"]
        if (
            record["fingerprint"] != frozen["fingerprint"]
            or name not in expected
            or name in values
            or path.parent.name != name
        ):
            raise ValueError("Diagnostic record identity changed")
        sample = expected[name]
        if (
            record["position_id"] != sample["position_id"]
            or abs((record["body_yaw_deg"] - sample["planned_body_yaw_deg"] + 180) % 360 - 180)
            > 1e-5
            or any(
                abs(a - b) > 1e-4
                for a, b in zip(record["position_m"], sample["planned_position_m"])
            )
        ):
            raise ValueError("Diagnostic sample pose differs from its frozen plan")
        for name, sha in record["files"].items():
            file = folder / name
            file.resolve().relative_to(folder.resolve())
            if digest(file) != sha:
                raise ValueError("Captured diagnostic artifact changed")
        values[record["sample_id"]] = record
    return values
