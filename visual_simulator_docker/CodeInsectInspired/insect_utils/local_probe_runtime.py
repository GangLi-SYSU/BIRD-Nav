"""Capture stationary local probes with the unchanged settled-frame camera pipeline."""

from pathlib import Path
import signal
import uuid

from insect_utils.experiment_config import PROJECT, canonical_hash, digest
from insect_utils.experiment_state import atomic_json, now, read_json
from insect_utils.local_probe import saved_rows, summarize_rows, vector_diagnostic
from insect_utils.navigation_protocol import verify_navigation
from insect_utils.navigation_runtime import (
    load_predictor,
    observation_receipt,
    predict_rgb,
    save_observation,
)
from insect_utils.rotation_runtime import reference_predictor


def verify_capture(folder):
    folder = Path(folder)
    frozen = verify_navigation(folder)
    completed = read_json(folder / "capture/completion.json")
    path = folder / "capture/results.json"
    if (
        completed["status"] != "completed"
        or completed["fingerprint"] != frozen["fingerprint"]
        or digest(path) != completed["results_sha256"]
    ):
        raise ValueError("Local capture completion changed")
    rows = saved_rows(folder, frozen)
    expected = {r["sample_id"] for r in frozen["samples"]}
    if set(rows) != expected or completed["images"] != len(expected):
        raise ValueError("Local probe is missing planned observations")
    result = read_json(path)
    if canonical_hash(list(rows.values())) != canonical_hash(result["rows"]):
        raise ValueError("Committed local capture records changed")
    return result


def capture_probe(folder):
    folder = Path(folder)
    frozen = verify_navigation(folder)
    if (folder / "capture/completion.json").exists():
        return verify_capture(folder)
    from omni.isaac.kit import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "anti_aliasing": 0})
    cameras, stop = [], []
    signal.signal(signal.SIGTERM, lambda sig, _: stop.append(sig))
    signal.signal(signal.SIGINT, lambda sig, _: stop.append(sig))
    try:
        import torch
        import carb
        import omni.usd
        import omni.replicator.core as rep
        from omni.isaac.core import World
        from insect_utils.downward_camera import DownwardCamera, capture_batch
        from insect_utils.downward_capture import configure_capture_rendering
        from insect_utils.rgb_preprocessing import rgb_tensor, load_rgb_tensor

        omni.usd.get_context().open_stage(str(PROJECT / frozen["protocol"]["scene"]))
        world = World(stage_units_in_meters=1.0)
        world.reset()
        for index in range(frozen["protocol"]["runtime"]["camera_count"]):
            cameras.append(
                DownwardCamera(
                    frozen["protocol"]["camera"],
                    world,
                    "replicator",
                    name="NavigationCamera_{:02d}".format(index),
                )
            )
        settings = carb.settings.get_settings()
        configure_capture_rendering(settings, rep.settings)
        model, reference = load_predictor(frozen), reference_predictor(frozen)
        completed = saved_rows(folder, frozen)
        pending = [s for s in frozen["samples"] if s["sample_id"] not in completed]
        for start in range(0, len(pending), len(cameras)):
            if stop:
                raise InterruptedError("Probe stopped; committed samples are preserved")
            batch = pending[start : start + len(cameras)]
            requests = [
                (cameras[i], s["planned_position_m"], s["planned_body_yaw_deg"], s["pose_id"])
                for i, s in enumerate(batch)
            ]
            observations = capture_batch(requests)
            predictions, _ = predict_rgb(model, observations)
            raw = torch.stack([rgb_tensor(rgb, [1024, 1024]) for rgb, _ in observations])
            with torch.inference_mode():
                expected = reference(raw)
                actual = torch.tensor(predictions, dtype=expected.dtype)
                singles = torch.cat([model(t[None]) for t in raw])
            if (
                not torch.isfinite(actual).all()
                or not torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
                or not torch.allclose(actual, singles, atol=1e-5, rtol=1e-4)
            ):
                raise ValueError("Live selected model/export/mask/scale or batching changed")
            for index, (sample, (rgb, receipt), prediction) in enumerate(
                zip(batch, observations, predictions)
            ):
                measured = observation_receipt(
                    cameras[index],
                    sample["planned_position_m"],
                    sample["planned_body_yaw_deg"],
                    sample["pose_id"],
                    receipt,
                    frozen["protocol"],
                    settings,
                )
                attempt = (
                    folder / "observations" / sample["sample_id"] / "attempts" / uuid.uuid4().hex
                )
                attempt.mkdir(parents=True, exist_ok=False)
                image = attempt / "image.png"
                sha = save_observation(image, rgb, receipt)
                if not torch.equal(raw[index], load_rgb_tensor(image, [1024, 1024], sha)):
                    raise ValueError("Live and saved probe input differ")
                position = [measured[k] for k in ("x", "y", "z")]
                result = vector_diagnostic(
                    position, measured["yaw"], prediction, frozen["protocol"]["home_camera_xyz_m"]
                )
                result.update(
                    sample_id=sample["sample_id"],
                    position_id=sample["position_id"],
                    planned_body_yaw_deg=sample["planned_body_yaw_deg"],
                    heading_index=sample["heading_index"],
                    fingerprint=frozen["fingerprint"],
                    role="diagnostic_only",
                    image=str(image.relative_to(PROJECT)),
                    image_sha256=sha,
                    script_reference_max_abs_m=float((actual[index] - expected[index]).abs().max()),
                    batch_single_max_abs_m=float((actual[index] - singles[index]).abs().max()),
                    files={
                        str(p.relative_to(folder)): digest(p)
                        for p in (image, image.with_suffix(".receipt.json"))
                    },
                )
                result["record_sha256"] = canonical_hash(result)
                atomic_json(folder / "observations" / sample["sample_id"] / "record.json", result)
                completed[sample["sample_id"]] = result
            atomic_json(
                folder / "progress.json",
                dict(
                    status="running",
                    phase="capture",
                    updated=now(),
                    fingerprint=frozen["fingerprint"],
                    images_completed=len(completed),
                    images_total=len(frozen["samples"]),
                    last_sample_id=batch[-1]["sample_id"],
                ),
            )
            print("LOCAL_PROBE {}/{}".format(len(completed), len(frozen["samples"])), flush=True)
        rows = list(saved_rows(folder, frozen).values())
        if len(rows) != len(frozen["samples"]):
            raise ValueError("Incomplete diagnostic image count")
        result = dict(
            fingerprint=frozen["fingerprint"],
            rows=rows,
            summary=summarize_rows(rows, frozen["protocol"]["home_camera_xyz_m"]),
        )
        atomic_json(folder / "capture/results.json", result)
        atomic_json(
            folder / "capture/completion.json",
            dict(
                status="completed",
                updated=now(),
                fingerprint=frozen["fingerprint"],
                images=len(rows),
                results_sha256=digest(folder / "capture/results.json"),
            ),
        )
        verify_capture(folder)
    except BaseException as error:
        atomic_json(
            folder / ("worker_error_" + uuid.uuid4().hex[:10] + ".json"),
            dict(error=str(error), updated=now()),
        )
        raise
    finally:
        for camera in reversed(cameras):
            camera.close()
        app.close()
