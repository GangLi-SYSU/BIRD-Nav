"""Check the formal headless camera on real-scene poses; never collect training data."""

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    preview = json.loads((PROJECT / "config/experiments/stage09_preview.json").read_text())
    from omni.isaac.kit import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "anti_aliasing": 0})
    report = {
        "purpose": "camera_engineering_acceptance_only",
        "training_steps": 0,
        "scene": preview["scene"],
        "runs": {},
        "dataset_capture_accepted": False,
    }
    cameras = []
    try:
        import carb
        import omni.usd
        import omni.replicator.core as rep
        import numpy as np
        from PIL import Image
        from omni.isaac.core import World
        from insect_utils.downward_camera import DownwardCamera, capture_batch
        from insect_utils.frame_stability import STABILITY_POLICY

        settings = carb.settings.get_settings()
        settings.set("/rtx/post/aa/op", 0)
        settings.set("/rtx/post/histogram/enabled", False)
        omni.usd.get_context().open_stage(str(PROJECT / preview["scene"]))
        world = World(stage_units_in_meters=1.0)
        world.reset()
        report["settings_after_stage_reset_before_enforcement"] = {
            "aa_op": settings.get("/rtx/post/aa/op"),
            "histogram_enabled": settings.get("/rtx/post/histogram/enabled"),
        }
        rep.settings.set_render_rtx_realtime(antialiasing="Off")
        settings.set("/rtx-transient/post/aa/limitedOps", False)
        settings.set("/rtx/post/aa/op", 0)
        settings.set("/rtx/post/histogram/enabled", False)
        report["observed_render_settings"] = {
            "aa_op": settings.get("/rtx/post/aa/op"),
            "histogram_enabled": settings.get("/rtx/post/histogram/enabled"),
            "aa_limited_ops": settings.get("/rtx-transient/post/aa/limitedOps"),
        }
        report["stability_policy"] = dict(STABILITY_POLICY)
        poses = [
            (name, xy, yaw)
            for name, xy in preview["inspection_positions"].items()
            for yaw in (0, 45, 90, 135, 180, 225, 270, 315)
        ]
        for count in (1, 4):
            run_started = time.perf_counter()
            key = "{}_camera".format(count)
            records = []
            report["runs"][key] = {"status": "running", "observations": records}
            for index in range(count):
                cameras.append(
                    DownwardCamera(
                        preview["camera"],
                        world,
                        "replicator",
                        name="Acceptance{}_{}".format(count, index),
                    )
                )
            try:
                for start in range(0, len(poses), count):
                    batch = list(enumerate(poses[start : start + count]))
                    if args.batch:
                        requests = [
                            (cameras[index], xy + [40.0], yaw, "{}_{}_{}".format(key, name, yaw))
                            for index, (name, xy, yaw) in batch
                        ]
                        observations = capture_batch(requests)
                    else:
                        observations = [
                            cameras[index].capture(
                                xy + [40.0], yaw, "{}_{}_{}".format(key, name, yaw)
                            )
                            for index, (name, xy, yaw) in batch
                        ]
                    for (index, (name, xy, yaw)), (rgb, receipt) in zip(batch, observations):
                        pose_id = "{}_{}_{}".format(key, name, yaw)
                        if (
                            receipt["pose_id"] != pose_id
                            or receipt["camera_prim"] != cameras[index].path
                            or not np.allclose(
                                receipt["position_m"], xy + [40.0], atol=1e-6, rtol=0
                            )
                        ):
                            raise ValueError("Engineering image/pose association differs")
                        if rgb.shape != (1024, 1024, 3) or rgb.dtype != np.uint8:
                            raise ValueError("Unexpected actual RGB dimensions or dtype")
                        Image.fromarray(rgb).save(args.output / (pose_id + ".png"))
                        actual_aa = settings.get("/rtx/post/aa/op")
                        if actual_aa != 0:
                            raise ValueError(
                                "Renderer overwrote AA during capture: " + str(actual_aa)
                            )
                        records.append(
                            {"pose_id": pose_id, "receipt": receipt, "observed_aa": actual_aa}
                        )
                        print(
                            "CAMERA_ACCEPTED "
                            + pose_id
                            + " MAE="
                            + str(receipt["stability"]["last_rgb_mae"]),
                            flush=True,
                        )
                report["runs"][key]["status"] = "passed"
                report["runs"][key]["elapsed_wall_s"] = time.perf_counter() - run_started
                report["runs"][key]["shared_batch_render_updates"] = args.batch
            finally:
                for camera in cameras:
                    camera.close()
                cameras.clear()
            (args.output / "camera_report.json").write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "passed"
        print("STAGE10_CAMERA_ACCEPTANCE_PASS", flush=True)
    except BaseException as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        print("STAGE10_CAMERA_ACCEPTANCE_FAILED " + str(error), flush=True)
        raise
    finally:
        (args.output / "camera_report.json").write_text(json.dumps(report, indent=2) + "\n")
        for camera in cameras:
            camera.close()
        app.close()


if __name__ == "__main__":
    main()
