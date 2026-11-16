"""Resumable four-camera acquisition for the frozen coverage plan."""

import json
import math
import os
from pathlib import Path
import signal
import time
import uuid

from insect_utils.capture_geometry import make_samples, measured_record
from insect_utils.camera_geometry import optics
from insect_utils.coverage_plan import verify_freeze
from insect_utils.experiment_config import PROJECT, digest
from insect_utils.experiment_state import atomic_json, now, read_json


def planned_samples(plan):
    result = []
    for position in plan["positions"]:
        if position["legacy"]:
            continue
        samples = make_samples([position["xyz_m"]], [position["yaws_deg"]], [0, 0, 40], 0)
        for sample in samples:
            identifier = "{}_h{}".format(position["position_id"], sample["heading_index"])
            sample.update(
                sample_id=identifier,
                pose_id=identifier,
                position_id=position["position_id"],
                split_group=position["position_id"],
                split=position["split"],
            )
            sample.pop("image")
            result.append(sample)
    return result


def committed_records(folder, verify_images=True):
    """Only atomic batch receipts commit images; interrupted orphan PNGs are ignored."""
    records = {}
    last_frames = {}
    for file in sorted((Path(folder) / "capture/sessions").glob("*/batch_*.json")):
        for row in read_json(file):
            if row["sample_id"] in records:
                raise ValueError("Duplicate committed camera sample")
            if verify_images and digest(PROJECT / row["image"]) != row["sha256"]:
                raise ValueError("Committed camera image changed")
            if row["session_id"] != file.parent.name:
                raise ValueError("Camera observation belongs to a different capture session")
            camera = (row["session_id"], row["camera_id"])
            if row["frame_id"] <= last_frames.get(camera, -1):
                raise ValueError("Stale committed frame within a camera session")
            last_frames[camera] = row["frame_id"]
            records[row["sample_id"]] = row
    return records


def check_observation(sample, row, protocol):
    checked = measured_record(sample, row["receipt"], [0, 0, 40], 0, protocol["height_agl_m"])
    if any(row.get(k) != value for k, value in checked.items()):
        raise ValueError("Camera observation no longer matches frozen plan")
    rendering = row["receipt"]["observed_render_settings"]
    if (
        rendering["antialiasing"] != 0
        or rendering["aa_limited_ops"] is not False
        or rendering["auto_exposure"] is not False
    ):
        raise ValueError("Camera rendering contract mismatch")
    if row["receipt"]["backend"] != "replicator":
        raise ValueError("Expected actual Replicator camera capture")
    receipt = row["receipt"]
    expected = optics(protocol["camera"])
    actual = receipt["observed_usd_optics"]
    if receipt["resolution_wh"] != expected["resolution_wh"]:
        raise ValueError("Captured camera resolution changed")
    for key, value in (
        ("focalLength", expected["usd_focal_length"]),
        ("horizontalAperture", expected["usd_aperture_wh"][0]),
        ("verticalAperture", expected["usd_aperture_wh"][1]),
        ("fStop", 0),
    ):
        if not math.isclose(actual[key], value, rel_tol=0, abs_tol=1e-5):
            raise ValueError("Captured camera optical parameters changed")
    slot = int(row["camera_id"].rsplit("_", 1)[1])
    if (
        slot not in range(4)
        or receipt["camera_prim"] != "/World/CoverageCamera_{:02d}".format(slot)
        or receipt["reference_time_numerator"] != row["frame_id"]
    ):
        raise ValueError("Camera slot or fresh-frame association changed")


def capture(folder):
    """Resume only verified committed observations; never overwrite a previous image."""
    folder = Path(folder)
    freeze = verify_freeze(folder)
    protocol = freeze["protocol"]
    planned = planned_samples(freeze["plan"])
    completed = committed_records(folder)
    mapping = {s["sample_id"]: s for s in planned}
    if not completed.keys() <= mapping.keys():
        raise ValueError("Capture has observations outside the frozen plan")
    for identifier, row in completed.items():
        check_observation(mapping[identifier], row, protocol)
    pending = [s for s in planned if s["sample_id"] not in completed]
    if not pending:
        return
    from omni.isaac.kit import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "anti_aliasing": 0})
    cameras = []
    stop = []

    def request_stop(signum, _):
        stop.append(signum)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    session_id = uuid.uuid4().hex
    session = folder / "capture/sessions" / session_id
    session.mkdir(parents=True, exist_ok=False)
    atomic_json(
        session / "session.json",
        {"id": session_id, "started": now(), "fingerprint": freeze["fingerprint"]},
    )
    started = time.monotonic()
    initial = len(completed)
    try:
        import carb
        import omni.usd
        import omni.replicator.core as rep
        from PIL import Image
        from omni.isaac.core import World
        from insect_utils.downward_camera import DownwardCamera, capture_batch
        from insect_utils.downward_capture import (
            configure_capture_rendering,
            require_capture_rendering,
        )

        omni.usd.get_context().open_stage(str(PROJECT / protocol["scene"]))
        world = World(stage_units_in_meters=1.0)
        world.reset()
        for index in range(4):
            cameras.append(
                DownwardCamera(
                    protocol["camera"],
                    world,
                    "replicator",
                    name="CoverageCamera_{:02d}".format(index),
                )
            )
        settings = carb.settings.get_settings()
        configure_capture_rendering(settings, rep.settings)
        for batch_index, start in enumerate(range(0, len(pending), 4)):
            if stop:
                raise InterruptedError("Capture stopped at a committed batch boundary")
            batch = pending[start : start + 4]
            observations = capture_batch(
                [
                    (cameras[i], s["planned_position_m"], s["planned_body_yaw_deg"], s["pose_id"])
                    for i, s in enumerate(batch)
                ]
            )
            rows = []
            for index, (sample, (rgb, receipt)) in enumerate(zip(batch, observations)):
                receipt["observed_render_settings"] = require_capture_rendering(settings)
                row = measured_record(sample, receipt, [0, 0, 40], 0, protocol["height_agl_m"])
                image = session / (sample["sample_id"] + ".png")
                with image.open("xb") as stream:
                    Image.fromarray(rgb).save(stream, format="PNG")
                    stream.flush()
                    os.fsync(stream.fileno())
                row.update(
                    image=str(image.relative_to(PROJECT)),
                    sha256=digest(image),
                    receipt=receipt,
                    session_id=session_id,
                    camera_id="camera_{}".format(index),
                )
                check_observation(sample, row, protocol)
                rows.append(row)
            atomic_json(session / "batch_{:05d}.json".format(batch_index), rows)
            completed.update({r["sample_id"]: r for r in rows})
            elapsed = time.monotonic() - started
            progress = {
                "phase": "capture",
                "status": "running",
                "updated": now(),
                "new_images_completed": len(completed),
                "new_images_total": len(planned),
                "reused_images": freeze["plan"]["audit"]["images"] - len(planned),
                "session_images_per_minute": (len(completed) - initial) * 60 / elapsed,
                "session_elapsed_s": elapsed,
            }
            atomic_json(folder / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        atomic_json(
            folder / "capture/completion.json",
            {
                "status": "completed",
                "updated": now(),
                "new_images": len(completed),
                "fingerprint": freeze["fingerprint"],
            },
        )
    finally:
        for camera in reversed(cameras):
            camera.close()
        app.close()
