"""Calibrate both real camera endpoints; no training or closed-loop navigation."""

import argparse
import json
import math
from pathlib import Path
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.camera_geometry import optics, world_to_pixel
from insect_utils.experiment_config import digest, load_config
from insect_utils.experiment_state import atomic_json
from insect_utils.frame_stability import STABILITY_POLICY


def verify_settling(receipt):
    stability = receipt["stability"]
    if stability["policy"] != STABILITY_POLICY or stability["consecutive_changes"] < 2:
        raise ValueError("Image did not satisfy the approved settling policy")
    history = stability["history"]
    frames = [entry["frame"] for entry in history]
    if len(frames) < 3 or any(a >= b for a, b in zip(frames, frames[1:])):
        raise ValueError("Settling evidence must contain advancing frames")
    if any(entry["rgb_mae"] is None or entry["rgb_mae"] > 0.5 for entry in history[-2:]):
        raise ValueError("The last two RGB changes exceed the settling limit")
    if receipt["rendering_frame"] != frames[-1] or receipt["updates_waited"] > 90:
        raise ValueError("Settling receipt does not match the accepted frame/budget")


MARKERS = {
    "red": (2, 0, (1, 0, 0)),
    "green": (0, 2, (0, 1, 0)),
    "blue": (-2, 0, (0, 0, 1)),
    "yellow": (0, -2, (1, 1, 0)),
    "white": (0, 0, (1, 1, 1)),
}
BASE = (12.0, -8.0)
MARKER_Z = 0.004


def calibration_scene(destination):
    from pxr import Gf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateNew(str(destination))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())

    def quad(name, x, y, hx, hy, z, color):
        mesh = UsdGeom.Mesh.Define(stage, "/World/" + name)
        mesh.CreatePointsAttr(
            [(x - hx, y - hy, z), (x + hx, y - hy, z), (x + hx, y + hy, z), (x - hx, y + hy, z)]
        )
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateSubdivisionSchemeAttr("none")
        material = UsdShade.Material.Define(stage, "/World/Materials/" + name)
        shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        from pxr import Sdf

        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    quad("Ground", *BASE, 6, 6, 0, (0.06, 0.06, 0.06))
    for i in range(-10, 11):
        quad(
            "GridX_" + str(i + 10), BASE[0] + i * 0.5, BASE[1], 0.004, 5, 0.001, (0.10, 0.10, 0.10)
        )
        quad(
            "GridY_" + str(i + 10), BASE[0], BASE[1] + i * 0.5, 5, 0.004, 0.001, (0.10, 0.10, 0.10)
        )
    for name, (x, y, color) in MARKERS.items():
        quad(name, BASE[0] + x, BASE[1] + y, 0.08, 0.08, MARKER_Z, color)
    stage.GetRootLayer().Save()


def marker_centers(rgb):
    import numpy as np
    from scipy import ndimage

    r, g, b = [rgb[:, :, i].astype(float) for i in range(3)]
    masks = {
        "red": (r > 100) & (r > 1.8 * g) & (r > 1.8 * b),
        "green": (g > 100) & (g > 1.8 * r) & (g > 1.8 * b),
        "blue": (b > 100) & (b > 1.8 * r) & (b > 1.8 * g),
        "yellow": (r > 100) & (g > 100) & (b < 0.5 * np.minimum(r, g)),
        "white": (np.minimum(np.minimum(r, g), b) > 160)
        & (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b) < 35),
    }
    centers = {}
    for name, mask in masks.items():
        labels, _ = ndimage.label(mask)
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        if sizes.max() < 16:
            raise ValueError("Missing calibration marker: " + name)
        yy, xx = np.nonzero(labels == int(sizes.argmax()))
        centers[name] = [float(xx.mean()) + 0.5, float(yy.mean()) + 0.5]
    return centers


def check_calibration(rgb, position, yaw, settings):
    import numpy as np

    centers = marker_centers(rgb)
    errors = {}
    for name, (x, y, _) in MARKERS.items():
        predicted = world_to_pixel([BASE[0] + x, BASE[1] + y, MARKER_Z], position, yaw, settings)
        errors[name] = float(np.linalg.norm(np.asarray(centers[name]) - predicted))
    if max(errors.values()) > 1.5:
        raise ValueError("Image axes/pose projection error: " + str(errors))
    span = (
        np.linalg.norm(np.asarray(centers["red"]) - centers["blue"])
        + np.linalg.norm(np.asarray(centers["green"]) - centers["yellow"])
    ) / 2
    fx = float(span * (position[2] - MARKER_Z) / 4)
    measured_hfov = math.degrees(2 * math.atan(settings["resolution_wh"][0] / (2 * fx)))
    if abs(measured_hfov - settings["hfov_deg"]) / settings["hfov_deg"] > 0.01:
        raise ValueError("Measured HFOV differs by more than 1%")
    return {
        "marker_centers_uv": centers,
        "projection_errors_px": errors,
        "measured_hfov_deg": measured_hfov,
        "measured_ground_footprint_width_m": position[2] * settings["resolution_wh"][0] / fx,
    }


def run_check(args, resolved):
    import numpy as np
    from PIL import Image
    import omni.usd
    import carb
    from omni.isaac.core import World
    import render_dataset
    import virtual_homing_ANN

    settings = optics(resolved["spec"]["camera"])
    if args.kind == "calibration":
        scene = args.output / "calibration_only.usda"
        calibration_scene(scene)
        poses = [([*BASE, 10], yaw) for yaw in (0, 90, 180, 270)]
        poses += [([BASE[0] + 0.4, BASE[1] - 0.3, 10], 0), ([*BASE, 10], 0)]
    else:
        scene = PROJECT / resolved["spec"]["scene"]["path"]
        manifest = json.loads(scene.with_suffix(".manifest.json").read_text())
        poses = []
        for index, shape in enumerate(("Cube", "Cylinder", "Cone", "Sphere")):
            item = min(
                (p for p in manifest["landmarks"] if p["shape"] == shape),
                key=lambda p: sum(v * v for v in p["center_xyz_m"][:2]),
            )
            poses.append((item["center_xyz_m"][:2] + [10], index * 90))
    carb.settings.get_settings().set("/rtx/post/aa/op", 0)
    carb.settings.get_settings().set("/rtx/post/histogram/enabled", False)
    omni.usd.get_context().open_stage(str(scene))
    world = World(stage_units_in_meters=1.0)
    world.reset()
    cfg = {"camera": resolved["spec"]["camera"]}
    capture = render_dataset.create_downward_capture_camera(cfg, world)
    inference = virtual_homing_ANN.create_downward_inference_camera(cfg, world)
    cameras = {"replicator": capture, "sensor": inference}
    records = []
    pairs = []
    last = {"replicator": -1, "sensor": -1}
    try:
        for index, (position, yaw) in enumerate(poses):
            # Both paths start convergence from the same pose. Moving the second
            # camera only after sampling the first confounds the comparison.
            for camera in cameras.values():
                camera.set_body_pose(position, yaw)
            images = {}
            for backend, camera in cameras.items():
                if backend == "sensor":
                    half = math.radians(yaw) / 2
                    body_quaternion = [math.cos(half), 0, 0, math.sin(half)]
                    rgb, _ = virtual_homing_ANN.get_image(body_quaternion, position, camera, world)
                    receipt = dict(camera.last_receipt)
                else:
                    rgb, receipt = camera.capture(position, yaw, "pose_{:02d}".format(index))
                assert receipt["rendering_frame"] > last[backend]
                verify_settling(receipt)
                last[backend] = receipt["rendering_frame"]
                filename = "{}_{:02d}.png".format(backend, index)
                Image.fromarray(rgb).save(args.output / filename)
                entry = {
                    "image": filename,
                    "sha256": digest(args.output / filename),
                    "receipt": receipt,
                }
                atomic_json(args.output / (filename + ".json"), entry)
                if args.kind == "calibration":
                    entry["calibration"] = check_calibration(rgb, position, yaw, settings)
                tensor = virtual_homing_ANN.downward_rgb_tensor(rgb)
                if tuple(tensor.shape) != (1, 3, 1024, 1024):
                    raise ValueError("Panorama crop still active")
                expected = np.transpose(rgb.astype(np.float32) / 255, (2, 0, 1))[None]
                if not np.array_equal(tensor.numpy(), expected):
                    raise ValueError("RGB tensor preprocessing mismatch")
                images[backend] = rgb
                records.append(entry)
                print(
                    "STAGE04_FRAME " + filename + " frame=" + str(receipt["rendering_frame"]),
                    flush=True,
                )
            delta = np.abs(images["replicator"].astype(float) - images["sensor"].astype(float))
            mae = float(delta.mean())
            if mae > 2.0:
                raise ValueError("Capture/inference RGB mismatch, MAE=" + str(mae))
            pairs.append(
                {
                    "pose_index": index,
                    "mean_absolute_rgb_difference": mae,
                    "p95_rgb_difference": float(np.quantile(delta, 0.95)),
                }
            )
        import torch
        from simple_network import IncepAttentionCNN_rgb

        torch.manual_seed(10)
        torch.set_num_threads(2)
        model = IncepAttentionCNN_rgb().eval()
        with torch.no_grad():
            prediction = model(tensor)
        if (
            sum(p.numel() for p in model.parameters()) != 10820
            or not torch.isfinite(prediction).all()
        ):
            raise ValueError("Original network input contract failed")
        timeout_checks = {}
        for backend, camera in cameras.items():
            try:
                camera.capture(position, yaw, "forced_timeout", max_updates=1)
            except RuntimeError:
                if camera.last_receipt is not None or camera.last_failure["updates_waited"] != 1:
                    raise ValueError("Timeout reused an old image receipt")
                timeout_checks[backend] = True
            else:
                raise ValueError("Incomplete rendering unexpectedly passed the timeout guard")
        if load_config(args.config)["fingerprint"] != resolved["fingerprint"]:
            raise ValueError("Configuration or camera source changed during calibration")
        return {
            "status": "passed",
            "report_version": 2,
            "acquisition_policy": dict(STABILITY_POLICY),
            "both_cameras_prepositioned": True,
            "kind": args.kind,
            "confirmed_camera": resolved["spec"]["camera"],
            "optics": settings,
            "scene": str(scene),
            "frames": records,
            "pair_comparisons": pairs,
            "network_parameters": 10820,
            "network_weights": "fresh_untrained_seed10",
            "network_output_shape": list(prediction.shape),
            "training_steps": 0,
            "timeout_guard_verified": timeout_checks,
            "note": (
                "Camera calibration only. Dataset protocol and closed-loop controller remain gated."
            ),
        }
    finally:
        for camera in cameras.values():
            camera.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "config/experiments/downward_h10_hfov30_landmarks.json",
    )
    parser.add_argument("--kind", choices=("calibration", "landmarks"), default="calibration")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.verify_only:
        report = json.loads((args.output / "camera_report.json").read_text())
        if report["status"] != "passed":
            raise ValueError("Camera verification did not pass")
        if report.get("acquisition_policy") != STABILITY_POLICY:
            raise ValueError("Report predates the approved settling policy; render a new run")
        for frame in report["frames"]:
            verify_settling(frame["receipt"])
            if digest(args.output / frame["image"]) != frame["sha256"]:
                raise ValueError("Changed calibration image")
        print("STAGE04_VERIFIED " + str(args.output))
        return
    resolved = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "resolved_config.json", resolved)
    atomic_json(args.output / "camera_report.json", {"status": "running"})
    app = None
    try:
        from omni.isaac.kit import SimulationApp

        app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "anti_aliasing": 0})
        report = run_check(args, resolved)
        atomic_json(args.output / "camera_report.json", report)
        print("STAGE04_CAMERA_PASS " + str(args.output), flush=True)
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        atomic_json(args.output / "camera_report.json", {"status": "failed", "error": error})
        raise
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
