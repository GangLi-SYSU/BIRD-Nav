"""Plan, acquire and verify explicitly configured downward learning observations."""

import csv
import json
import math
import os
from pathlib import Path

from insect_utils.camera_geometry import optics
from insect_utils.capture_geometry import contains, exact_batches, make_samples
from insect_utils.capture_geometry import measured_record, overlap_report
from insect_utils.experiment_config import canonical_hash, digest
from insect_utils.experiment_state import atomic_json, collect_artifacts
from insect_utils.scene_geometry import check_positions_and_segments, require_assets, spatial_budget


ROW_FIELDS = [
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
]


def capture_render_state(settings):
    return {
        "mode": settings.get("/rtx/rendermode"),
        "antialiasing": settings.get("/rtx/post/aa/op"),
        "aa_limited_ops": settings.get("/rtx-transient/post/aa/limitedOps"),
        "auto_exposure": settings.get("/rtx/post/histogram/enabled"),
    }


def configure_capture_rendering(settings, replicator_settings):
    """Apply the declared rendering contract after stage/camera initialization.

    Kit's AA limited-ops mode otherwise silently restores DLSS, even when the
    global AA option was set to zero before opening a stage.
    """
    replicator_settings.set_render_rtx_realtime(antialiasing="Off")
    settings.set("/rtx-transient/post/aa/limitedOps", False)
    settings.set("/rtx/post/aa/op", 0)
    settings.set("/rtx/post/histogram/enabled", False)


def require_capture_rendering(settings):
    actual = capture_render_state(settings)
    if (
        actual["antialiasing"] != 0
        or actual["aa_limited_ops"] is not False
        or actual["auto_exposure"] is not False
    ):
        raise ValueError("Capture render settings were overwritten: " + str(actual))
    return actual


def build_plan(config):
    """Use the original bee generator and an isolated original-style yaw stream."""
    import numpy as np
    from insect_utils.flight_path_functions import generate_bee_path

    count = config.get("headings_per_position")
    if isinstance(count, bool) or not isinstance(count, int) or count not in (1, 4, 8):
        raise ValueError("Explicitly select headings_per_position=1, 4 or 8 before capture")
    if config["shape_flight"] != "bee" or config["sample_count_policy"] != "exact":
        raise ValueError("Downward capture requires an exact-count bee learning path")
    if config.get("use_noisy_spiral") or config.get("only_point_north") or config.get("crop"):
        raise ValueError("This downward protocol has no noise, north lock or panorama crop")
    scene = Path(config["map"])
    manifest = require_assets(scene)
    spec = config["experiment_spec"]
    budget = spatial_budget(spec, manifest)
    positions = generate_bee_path(*config["shape_params"], config["home_position"])
    check_positions_and_segments(positions, budget)
    # Preserve the original rand()*720 orientation distribution and first draw,
    # while making it independent of camera allocation and renderer initialization.
    base_yaws = np.random.RandomState(config["seed"]).random_sample(len(positions)) * 720
    headings = [[float(yaw + i * 360 / count) for i in range(count)] for yaw in base_yaws]
    samples = make_samples(positions, headings, config["home_position"], config["ground_z_m"])
    polygons, adjacent = overlap_report(samples, config["ground_z_m"], optics(config["camera"]))
    maximum = max(math.dist(p[:2], config["home_position"][:2]) for p in positions)
    return {
        "schema_version": 1,
        "role": "learning",
        "positions": positions,
        "position_count": len(positions),
        "headings_per_position": count,
        "sample_count": len(samples),
        "camera_count": config["number_cameras"],
        "batch_sizes": [len(batch) for batch in exact_batches(samples, config["number_cameras"])],
        "pose_signature": canonical_hash(
            [[s["planned_position_m"], s["planned_body_yaw_deg"]] for s in samples]
        ),
        "samples": samples,
        "footprints_xy_m": polygons,
        "adjacent_steps": adjacent,
        "path_length_m": sum(p["distance_m"] for p in adjacent),
        "max_radius_m": maximum,
        "spatial_budget": budget,
        "yaw_policy": (
            "RandomState(seed).rand(position_count)*720 modulo360; extra headings offset evenly"
        ),
        "height_adjustments": [],
        "height_policy": (
            "flat ground referenced, unchanged XY; all landmark tops below flight plane"
        ),
        "coverage_note": (
            "Unoccluded reference-ground projections, not visible area or navigation success"
        ),
    }


def coverage_diagnostics(plan, spec, cell_m=0.25):
    """Estimate footprint union and disk holes on a documented cell-center grid."""
    import numpy as np

    polygons = plan["footprints_xy_m"]
    low = [
        math.floor(min(p[i] for polygon in polygons for p in polygon) / cell_m) * cell_m
        for i in range(2)
    ]
    high = [
        math.ceil(max(p[i] for polygon in polygons for p in polygon) / cell_m) * cell_m
        for i in range(2)
    ]
    xs = np.arange(low[0] + cell_m / 2, high[0], cell_m)
    ys = np.arange(low[1] + cell_m / 2, high[1], cell_m)
    xx, yy = np.meshgrid(xs, ys)
    covered = np.zeros(xx.shape, dtype=bool)
    for polygon in polygons:
        inside = np.ones(xx.shape, dtype=bool)
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            inside &= ((b[0] - a[0]) * (yy - a[1]) - (b[1] - a[1]) * (xx - a[0])) >= -1e-9
        covered |= inside
    hx, hy = spec["home"]["ground_xy_m"]
    disk = (xx - hx) ** 2 + (yy - hy) ** 2 <= plan["max_radius_m"] ** 2
    starts = []
    for scale in spec["evaluation"]["radius_scales"]:
        for index in range(spec["evaluation"]["points_per_radius"]):
            angle = index * 2 * math.pi / spec["evaluation"]["points_per_radius"]
            radius = scale * spec["evaluation"]["radius_m"]
            point = [hx + radius * math.cos(angle), hy + radius * math.sin(angle)]
            starts.append(
                {
                    "xy_m": point,
                    "radius_m": radius,
                    "inside_learning_footprint": any(contains(p, point) for p in polygons),
                }
            )
    report = {
        "method": "cell-center grid, reference ground plane",
        "cell_m": cell_m,
        "bounds_xy_m": [low, high],
        "projected_union_area_estimate_m2": float(covered.sum() * cell_m ** 2),
        "learning_disk_covered_fraction_estimate": float((covered & disk).sum() / disk.sum()),
        "learning_disk_uncovered_fraction_estimate": float((~covered & disk).sum() / disk.sum()),
        "evaluation_starts_analysis_only": starts,
        "evaluation_samples_added_to_learning": 0,
        "occlusion_accounted_for": False,
    }
    return report, (xx, yy, covered, disk)


def save_plan_plots(plan, spec, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    report, (xx, yy, covered, disk) = coverage_diagnostics(plan, spec)
    atomic_json(output / "coverage_report.json", report)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(
        xx[disk & ~covered],
        yy[disk & ~covered],
        s=1,
        c="#edb4ae",
        label="Uncovered disk cells (estimate)",
    )
    for polygon in plan["footprints_xy_m"]:
        ax.add_patch(Polygon(polygon, facecolor="#348abd", edgecolor="#348abd", alpha=0.07))
    positions = plan["positions"]
    ax.plot(
        [p[0] for p in positions],
        [p[1] for p in positions],
        "k.-",
        ms=2,
        lw=0.7,
        label="Original bee learning path",
    )
    for contained, color in [(True, "#2a9d45"), (False, "#d62728")]:
        selected = [
            s["xy_m"]
            for s in report["evaluation_starts_analysis_only"]
            if s["inside_learning_footprint"] == contained
        ]
        if selected:
            ax.scatter(
                [p[0] for p in selected],
                [p[1] for p in selected],
                s=18,
                marker="x",
                c=color,
                label="Evaluation starts " + ("inside" if contained else "outside") + " footprints",
            )
    ax.scatter(
        *spec["home"]["ground_xy_m"], marker="*", s=120, c="#f1b32b", edgecolors="k", label="Home"
    )
    ax.set(
        xlabel="World X (m)",
        ylabel="World Y (m)",
        title="Learning footprints and evaluation starts (reference ground plane)",
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "learning_coverage.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    steps = plan["adjacent_steps"]
    axes[0].plot([s["distance_m"] for s in steps])
    axes[0].set_ylabel("Step distance (m)")
    axes[1].plot([s["footprint_area_overlap_fraction"] for s in steps])
    axes[1].set(
        ylabel="Footprint area overlap", xlabel="Adjacent learning position pair", ylim=(0, 1)
    )
    fig.tight_layout()
    fig.savefig(output / "adjacent_overlap.png", dpi=160)
    plt.close(fig)
    return report


def write_tables(output, records):
    with (output / "samples.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    with (output / "dataset_navigation.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Filename", "Position", "Rotation", "Target"])
        for row in records:
            writer.writerow(
                [
                    Path(row["image"]).name,
                    repr((row["x"], row["y"], row["z"])),
                    repr((0.0, 0.0, row["yaw"])),
                    "[{:.17g} {:.17g}]".format(row["target_x"], row["target_y"]),
                ]
            )


def capture_dataset(config, world=None):
    import carb
    import omni.usd
    import omni.replicator.core as rep
    from omni.isaac.core import World
    from PIL import Image
    from render_dataset import create_downward_capture_camera
    from insect_utils.downward_camera import capture_batch

    plan = build_plan(config)
    output = Path(config["output_folder"])
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "Replicator/rgb"
    image_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "capture_plan.json", plan)
    save_plan_plots(plan, config["experiment_spec"], output)
    carb.settings.get_settings().set("/rtx/post/aa/op", 0)
    carb.settings.get_settings().set("/rtx/post/histogram/enabled", False)
    if world is None:
        omni.usd.get_context().open_stage(config["map"])
        world = World(stage_units_in_meters=1.0)
        world.reset()
    cameras = []
    records = []
    try:
        for index in range(min(config["number_cameras"], len(plan["samples"]))):
            cameras.append(
                create_downward_capture_camera(config, world, "LearningCamera_{:02d}".format(index))
            )
        settings = carb.settings.get_settings()
        configure_capture_rendering(settings, rep.settings)
        with (output / "samples.jsonl").open("x") as log:
            for batch in exact_batches(plan["samples"], len(cameras)):
                requests = [
                    (
                        cameras[index],
                        sample["planned_position_m"],
                        sample["planned_body_yaw_deg"],
                        sample["pose_id"],
                    )
                    for index, sample in batch
                ]
                observations = capture_batch(requests)
                for (index, sample), (rgb, receipt) in zip(batch, observations):
                    receipt["observed_render_settings"] = require_capture_rendering(settings)
                    record = measured_record(
                        sample,
                        receipt,
                        config["home_position"],
                        config["ground_z_m"],
                        config["camera_height_agl_m"],
                    )
                    Image.fromarray(rgb).save(output / sample["image"])
                    record.update(
                        camera_id="camera_{:02d}".format(index),
                        sha256=digest(output / sample["image"]),
                        receipt=receipt,
                    )
                    atomic_json(output / (sample["image"] + ".json"), record)
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                    log.flush()
                    os.fsync(log.fileno())
                    records.append(record)
                    print(
                        "DOWNWARD_SAMPLE {}/{} {} frame={}".format(
                            len(records),
                            len(plan["samples"]),
                            sample["sample_id"],
                            receipt["rendering_frame"],
                        ),
                        flush=True,
                    )
        write_tables(output, records)
        return world
    finally:
        for camera in reversed(cameras):
            camera.close()


def seal_dataset(resolved, run_dir, config, publish=True):
    """Rebuild the plan, verify every observation and both CSV representations."""
    from PIL import Image
    from insect_utils.augment_and_save import load_dataset_from_csv

    output = run_dir / "capture"
    if publish and (output / "dataset_manifest.json").exists():
        raise FileExistsError("Capture manifest already exists; verify instead of overwriting")
    plan = build_plan(config)
    saved = json.loads((output / "capture_plan.json").read_text())
    # JSON serializes path tuples as lists: compare canonical representations.
    if canonical_hash(saved) != canonical_hash(plan):
        raise ValueError("Capture plan changed")
    records = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
    if (
        len(records) != len(plan["samples"])
        or len(records) != resolved["derived"]["expected_samples"]
    ):
        raise ValueError("Missing or extra capture sample")
    actual_files = {str(p.relative_to(output)) for p in (output / "Replicator/rgb").glob("*.png")}
    if actual_files != {s["image"] for s in plan["samples"]}:
        raise ValueError("Missing or extra image file")
    last_frames = {}
    sample_ids = set()
    for index, (sample, record) in enumerate(zip(plan["samples"], records)):
        checked = measured_record(
            sample,
            record["receipt"],
            config["home_position"],
            config["ground_z_m"],
            config["camera_height_agl_m"],
        )
        if any(record.get(k) != value for k, value in checked.items()):
            raise ValueError("Image/pose/label association changed")
        camera = "camera_{:02d}".format(index % config["number_cameras"])
        expected_prim = "/World/LearningCamera_{:02d}".format(index % config["number_cameras"])
        if (
            record["receipt"]["backend"] != "replicator"
            or record["receipt"]["camera_prim"] != expected_prim
        ):
            raise ValueError("Image came from a different capture camera")
        if record["camera_id"] != camera or record["sample_id"] in sample_ids:
            raise ValueError("Duplicate sample or wrong camera assignment")
        sample_ids.add(record["sample_id"])
        if record["frame_id"] <= last_frames.get(camera, -1):
            raise ValueError("Stale camera frame")
        last_frames[camera] = record["frame_id"]
        if digest(output / record["image"]) != record["sha256"]:
            raise ValueError("Changed capture image")
        if json.loads((output / (record["image"] + ".json")).read_text()) != record:
            raise ValueError("Changed image sidecar")
        with Image.open(output / record["image"]) as image:
            if (
                image.mode != "RGB"
                or list(image.size) != resolved["derived"]["processed_resolution_wh"]
            ):
                raise ValueError("Wrong image mode or dimensions")
            image.verify()
    with (output / "samples.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ROW_FIELDS:
            raise ValueError("Invalid typed sample CSV schema")
        rows = list(reader)
    if rows != [{k: str(r[k]) for k in ROW_FIELDS} for r in records]:
        raise ValueError("Typed CSV differs from observations")
    with (output / "dataset_navigation.csv").open(newline="") as stream:
        legacy_rows = list(csv.reader(stream))
    if (
        not legacy_rows
        or legacy_rows[0] != ["Filename", "Position", "Rotation", "Target"]
        or len(legacy_rows) != len(records) + 1
        or any(len(row) != 4 for row in legacy_rows)
    ):
        raise ValueError("Invalid legacy CSV schema or row count")
    filenames, tx, ty, pm, rm, tm, *_ = load_dataset_from_csv(
        str(output / "dataset_navigation.csv")
    )
    if filenames != [Path(r["image"]).name for r in records]:
        raise ValueError("Legacy CSV image mapping changed")
    for row, x, y in zip(records, tx, ty):
        name = Path(row["image"]).name
        actual = list(pm[name]) + list(rm[name]) + [x, y]
        wanted = [row["x"], row["y"], row["z"], 0, 0, row["yaw"], row["target_x"], row["target_y"]]
        if any(not math.isfinite(a) or abs(a - b) > 1e-8 for a, b in zip(actual, wanted)):
            raise ValueError("Legacy CSV label parse mismatch")
    manifest = {
        "fingerprint": resolved["fingerprint"],
        "purpose": resolved["spec"]["purpose"],
        "producer": "isaac",
        "sample_count": len(records),
        "planned_samples": len(plan["samples"]),
        "position_count": plan["position_count"],
        "headings_per_position": plan["headings_per_position"],
        "camera_count": plan["camera_count"],
        "pose_signature": plan["pose_signature"],
        "camera_mode": "downward_pinhole",
        "count_policy": "exact",
        "camera_optics": optics(config["camera"]),
        "camera_contract": config["camera_contract"],
        "acquisition_policy": config["camera_acquisition_policy"],
        "scene_fingerprint": resolved["provenance"]["scene_bundle"],
        "role": "learning_only",
        "split_assignment": "pending_stage_F; group by split_group",
        "training_steps": 0,
        "files": {
            path: record
            for path, record in collect_artifacts(output, run_dir).items()
            if path != "capture/dataset_manifest.json"
        },
    }
    if publish:
        atomic_json(output / "dataset_manifest.json", manifest)
    elif json.loads((output / "dataset_manifest.json").read_text()) != manifest:
        raise ValueError("Capture manifest differs from verified observations")
    return manifest
