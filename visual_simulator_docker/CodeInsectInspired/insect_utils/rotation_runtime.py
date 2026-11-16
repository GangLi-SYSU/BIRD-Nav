"""Reuse the established navigation loop with circular models and20m-only IO checks."""

import math
from pathlib import Path
import signal
import uuid

from insect_utils.experiment_config import PROJECT, digest
from insect_utils.experiment_state import atomic_json, now, read_json
from insect_utils.navigation_protocol import verify_navigation
from insect_utils.navigation_runtime import (
    load_predictor,
    observation_receipt,
    run_episodes,
    save_observation,
)
from run_navigation_experiment import verify_phase


def reference_predictor(frozen):
    import torch
    from insect_utils.circular_augmentation import CircularAttention
    from insect_utils.coverage_model import PhysicalHomeVector

    reference = CircularAttention({"network_variant": "nadir384x384"}).cpu().eval()
    reference.load_state_dict(
        torch.load(PROJECT / frozen["model"]["weights"], map_location="cpu", weights_only=True)
    )
    return PhysicalHomeVector(reference, frozen["model"]["scale_m"]).eval()


def model_check(folder, frozen, cameras, settings):
    import torch
    from insect_utils.downward_camera import capture_batch
    from insect_utils.rgb_preprocessing import load_rgb_tensor, rgb_tensor

    scripted, reference = load_predictor(frozen), reference_predictor(frozen)
    home = frozen["protocol"]["home_camera_xyz_m"]
    poses = [
        (
            [
                home[0] + 20 * math.cos(math.radians(b)),
                home[1] + 20 * math.sin(math.radians(b)),
                home[2],
            ],
            yaw,
        )
        for b in (0, 90, 180, 270)
        for yaw in (0, 90)
    ]
    destination = Path(folder) / "model_check/sessions" / uuid.uuid4().hex
    destination.mkdir(parents=True)
    checks = []
    for start in range(0, len(poses), len(cameras)):
        batch = poses[start : start + len(cameras)]
        requests = [
            (cameras[i], position, yaw, "modelcheck_{:02d}".format(start + i))
            for i, (position, yaw) in enumerate(batch)
        ]
        observations = capture_batch(requests)
        tensors = []
        for i, ((rgb, receipt), request) in enumerate(zip(observations, requests)):
            observation_receipt(
                cameras[i],
                request[1],
                request[2],
                request[3],
                receipt,
                frozen["protocol"],
                settings,
            )
            image = destination / (request[3] + ".png")
            sha = save_observation(image, rgb, receipt)
            tensor = rgb_tensor(rgb, [1024, 1024])
            if not torch.equal(tensor, load_rgb_tensor(image, [1024, 1024], sha)):
                raise ValueError("Live and saved RGB inputs differ")
            tensors.append(tensor)
        stacked = torch.stack(tensors)
        with torch.inference_mode():
            actual, expected = scripted(stacked), reference(stacked)
            singles = torch.cat([scripted(t[None]) for t in tensors])
        if not torch.isfinite(actual).all() or not torch.allclose(
            actual, expected, atol=1e-5, rtol=1e-4
        ):
            raise ValueError("Live circular export differs from selected weights/mask/scale")
        if not torch.allclose(actual, singles, atol=1e-5, rtol=1e-4):
            raise ValueError("Navigation predictions depend on episode batch assignment")
        checks.append(
            dict(
                samples=len(batch),
                script_reference_max_abs_difference_m=float((actual - expected).abs().max()),
                batch_single_max_abs_difference_m=float((actual - singles).abs().max()),
            )
        )
    artifacts = {
        str(p.relative_to(Path(folder) / "model_check")): digest(p)
        for p in destination.iterdir()
        if p.is_file()
    }
    atomic_json(
        Path(folder) / "model_check/completion.json",
        dict(
            status="completed",
            fingerprint=frozen["fingerprint"],
            images=len(poses),
            online_disk_equal=True,
            checks=checks,
            output_unit="meters",
            extra_scale_applied=False,
            fixed_circle_inside_export=True,
            random_inference_rotation=False,
            artifacts=artifacts,
        ),
    )


def run_arm(folder):
    folder = Path(folder)
    frozen = verify_navigation(folder)
    from omni.isaac.kit import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "anti_aliasing": 0})
    cameras, stop = [], []
    signal.signal(signal.SIGTERM, lambda signum, _: stop.append(signum))
    signal.signal(signal.SIGINT, lambda signum, _: stop.append(signum))
    try:
        import carb
        import omni.usd
        import omni.replicator.core as rep
        from omni.isaac.core import World
        from insect_utils.downward_camera import DownwardCamera
        from insect_utils.downward_capture import configure_capture_rendering

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
        for phase in ("rendered_oracle", "model_check", "formal"):
            if (folder / phase / "completion.json").exists():
                verify_phase(folder, phase, frozen)
                continue
            if phase == "model_check":
                model_check(folder, frozen, cameras, settings)
            else:
                model = None if phase == "rendered_oracle" else load_predictor(frozen)
                run_episodes(folder, phase, frozen, cameras, model, settings, stop)
            completion = read_json(folder / phase / "completion.json")
            if (
                completion["fingerprint"] != frozen["fingerprint"]
                or completion["status"] != "completed"
            ):
                raise RuntimeError("Missing valid completion receipt for " + phase)
            print("COMPLETED " + phase, flush=True)
        verify_navigation(folder)
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
