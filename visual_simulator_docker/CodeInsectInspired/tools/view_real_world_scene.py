"""Preview normal-scale houses, trees and pool; save inspection PNGs without training."""

import argparse
import json
import math
from pathlib import Path
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def capture_inspection(camera, position, yaw, pose_id, subframes, frame_budget):
    """Export fresh, pose-checked previews, explicitly retaining any settling failure."""
    import time
    import numpy as np
    import omni.replicator.core as rep
    from insect_utils.camera_geometry import usd_world_matrix
    from insect_utils.frame_stability import FrameStability

    camera.set_body_pose(position, yaw)
    expected = np.asarray(usd_world_matrix(position, yaw))
    previous_frame = max(camera._frame(), camera.last_frame)
    first_pose_frame = None
    previous_rgb = None
    settling = FrameStability()
    latest_valid = None
    started = time.perf_counter()
    for update in range(frame_budget):
        rep.orchestrator.step(rt_subframes=subframes, pause_timeline=False)
        frame = camera._frame()
        if frame <= previous_frame or frame <= settling.last_frame:
            continue
        reference = camera.reference.get_data()
        params = camera.params.get_data()
        if not reference or not params or int(reference["referenceTimeNumerator"]) != frame:
            settling.reset()
            first_pose_frame = None
            previous_rgb = None
            continue
        matrix = np.linalg.inv(np.asarray(params["cameraViewTransform"]).reshape(4, 4))
        if not np.allclose(matrix, expected, atol=1e-4, rtol=0):
            settling.reset()
            first_pose_frame = None
            previous_rgb = None
            continue
        if first_pose_frame is None:
            first_pose_frame = frame
            continue
        if frame <= first_pose_frame:
            continue
        raw = np.asarray(camera.rgb.get_data()).copy()
        if raw.ndim != 3 or raw.shape[:2] != (1024, 1024) or raw.shape[2] < 3:
            continue
        if camera._frame() != frame or raw.dtype != np.uint8:
            continue
        rgb = np.ascontiguousarray(raw[:, :, :3])
        if not np.any(rgb):
            continue
        actual_optics = {
            name: float(camera.prim.GetAttribute(name).Get())
            for name in ("focalLength", "horizontalAperture", "verticalAperture", "fStop")
        }
        expected_optics = dict(
            focalLength=camera.settings["usd_focal_length"],
            horizontalAperture=camera.settings["usd_aperture_wh"][0],
            verticalAperture=camera.settings["usd_aperture_wh"][1],
            fStop=0,
        )
        if any(abs(actual_optics[name] - value) > 1e-6 for name, value in expected_optics.items()):
            raise ValueError("Renderer changed inspection camera optics")
        mae = (
            float(np.abs(rgb.astype(np.float32) - previous_rgb).mean())
            if previous_rgb is not None
            else None
        )
        previous_rgb = rgb.astype(np.float32)
        settled = settling.observe(frame, mae)
        camera.last_frame = frame
        summary = settling.summary()
        summary["policy"]["render_subframes"] = subframes
        receipt = {
            "pose_id": pose_id,
            "position_m": position,
            "body_yaw_deg": yaw,
            "camera_world_matrix": matrix.tolist(),
            "camera_prim": camera.path,
            "rendering_frame": frame,
            "first_pose_matched_frame": first_pose_frame,
            "reference_time_numerator": int(reference["referenceTimeNumerator"]),
            "updates_waited": update + 1,
            "elapsed_s": time.perf_counter() - started,
            "stability": summary,
            "resolution_wh": [1024, 1024],
            "observed_usd_optics": actual_optics,
            "render_camera_projection": np.asarray(params["cameraProjection"])
            .reshape(4, 4)
            .tolist(),
            "purpose": "visual_preview_only",
            "settled": settled,
            "preview_frame_budget": frame_budget,
            "dataset_capture_accepted": False,
        }
        latest_valid = (rgb, receipt)
        if settled:
            return latest_valid
    if latest_valid is None:
        raise RuntimeError("No fresh pose-matched preview RGB within the render budget")
    print("PREVIEW_SETTLING_PENDING " + pose_id, flush=True)
    return latest_valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "config/experiments/stage09_preview.json"
    )
    parser.add_argument("--save-preview", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--exit-after-preview", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["purpose"] != "scene_preview_only" or config["training_steps"] != 0:
        raise ValueError("This viewer accepts only the preview protocol")
    if args.save_preview:
        args.save_preview.mkdir(parents=True, exist_ok=False)
    from omni.isaac.kit import SimulationApp

    app = SimulationApp(
        {
            "headless": args.headless,
            "width": 1280,
            "height": 900,
            "renderer": "RayTracedLighting",
            "anti_aliasing": config["preview_anti_aliasing"],
            "multi_gpu": False,
        }
    )
    capture = None
    try:
        import carb
        import numpy as np
        from PIL import Image
        from pxr import Gf, UsdGeom
        import omni.ui as ui
        import omni.usd
        import omni.replicator.core as rep
        from omni.kit.viewport.utility import get_active_viewport
        from insect_utils.camera_geometry import optics
        from insect_utils.downward_camera import DownwardCamera, configure_prim
        from insect_utils.scene_geometry import require_assets
        from omni.isaac.core import World

        settings = carb.settings.get_settings()
        settings.set("/app/runLoops/main/rateLimitEnabled", True)
        settings.set("/app/runLoops/main/rateLimitFrequency", 15)
        scene = PROJECT / config["scene"]
        manifest = require_assets(scene)
        omni.usd.get_context().open_stage(str(scene))
        for _ in range(5):
            app.update()
        world = World(stage_units_in_meters=1.0)
        world.reset()
        stage = omni.usd.get_context().get_stage()
        overview = UsdGeom.Camera.Define(stage, "/World/ResidentialOverview")
        overview.GetFocalLengthAttr().Set(24)
        overview.GetHorizontalApertureAttr().Set(36)
        overview.GetVerticalApertureAttr().Set(36)
        overview.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 1000))
        matrix = (
            Gf.Matrix4d()
            .SetLookAt(Gf.Vec3d(90, -115, 125), Gf.Vec3d(0, 0, 0), Gf.Vec3d(0, 0, 1))
            .GetInverse()
        )
        UsdGeom.Xformable(overview).AddTransformOp().Set(matrix)
        capture = DownwardCamera(config["camera"], world, "replicator", name="RealPreviewNadir")
        state = {"position": "junction", "fov": config["camera"]["hfov_deg"]}

        def set_view(path):
            if not args.headless:
                get_active_viewport().camera_path = path

        label = [None]

        def show_overview():
            set_view(str(overview.GetPath()))
            if label[0] is not None:
                label[0].text = "Oblique overview (inspection camera)"

        def set_nadir(position=None, fov=None, select_view=True):
            if position is not None:
                state["position"] = position
            if fov is not None:
                state["fov"] = fov
            capture.settings = optics(dict(config["camera"], hfov_deg=state["fov"]))
            configure_prim(capture.prim, capture.settings)
            xy = config["inspection_positions"][state["position"]]
            capture.set_body_pose(xy + [config["camera_height_agl_m"]], 0)
            if select_view:
                set_view(capture.path)
            if label[0] is not None:
                label[0].text = "{} | {} m AGL | HFOV {} deg".format(
                    state["position"], config["camera_height_agl_m"], state["fov"]
                )

        def inspect_rgb(rgb):
            if rgb.shape != (1024, 1024, 3) or rgb.dtype != np.uint8:
                raise ValueError("Preview image is not actual 1024-square uint8 RGB")
            std = float(rgb.astype(float).std(axis=(0, 1)).mean())
            black = float((rgb.max(axis=2) < 5).mean())
            if std < 3 or black > 0.02:
                raise ValueError("Preview is blank or missing terrain")
            return {"spatial_std_rgb_mean": std, "black_fraction": black}

        # Keep the interactive viewport off the capture camera: viewport auto
        # framing or user navigation can otherwise change its physical pose.
        set_view(str(overview.GetPath()))
        set_nadir(select_view=False)
        print("REAL_SCENE_LOADED; warming renderer and materials", flush=True)
        if args.save_preview:
            records = []
            try:
                # Keep the original rendering settings. An unsettled frame may
                # be viewed, with its failure recorded; it is never training data.
                cases = [(name, 70.0) for name in config["inspection_positions"]]
                cases.extend(
                    ("junction", fov) for fov in config["comparison_hfov_deg"] if fov != 70
                )
                for name, fov in cases:
                    set_nadir(name, fov, select_view=False)
                    xy = config["inspection_positions"][name]
                    rgb, receipt = capture_inspection(
                        capture,
                        xy + [config["camera_height_agl_m"]],
                        0,
                        name + str(fov),
                        config["preview_render_subframes"],
                        config["preview_frame_budget"],
                    )
                    projection = np.asarray(receipt["render_camera_projection"])
                    actual_hfov = math.degrees(2 * math.atan(1 / abs(projection[0, 0])))
                    actual_vfov = math.degrees(2 * math.atan(1 / abs(projection[1, 1])))
                    if not np.allclose([actual_hfov, actual_vfov], [fov, fov], atol=0.01, rtol=0):
                        raise ValueError("Rendered projection differs from requested square FOV")
                    filename = "{}_h40_fov{:g}.png".format(name, fov)
                    Image.fromarray(rgb).save(args.save_preview / filename)
                    records.append(
                        dict(
                            image=filename,
                            actual_hfov_deg=actual_hfov,
                            actual_vfov_deg=actual_vfov,
                            receipt=receipt,
                            **inspect_rgb(rgb),
                        )
                    )
                    print("REAL_PREVIEW_CAPTURED " + filename, flush=True)
                set_view(str(overview.GetPath()))
                product = rep.create.render_product(str(overview.GetPath()), (1024, 1024))
                rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                rgb_annotator.attach([product])
                try:
                    for _ in range(8):
                        rep.orchestrator.step(rt_subframes=2, pause_timeline=False)
                    rgb = np.asarray(rgb_annotator.get_data())[:, :, :3].copy()
                    stats = inspect_rgb(rgb)
                    Image.fromarray(rgb).save(args.save_preview / "overview.png")
                    records.append(
                        dict(image="overview.png", purpose="oblique_scene_inspection_only", **stats)
                    )
                finally:
                    rgb_annotator.detach([product.path])
                    product.destroy()
                report = {
                    "status": "preview_ready",
                    "all_downward_views_settled": all(
                        record["receipt"]["settled"] for record in records if "receipt" in record
                    ),
                    "dataset_capture_accepted": False,
                    "purpose": "scene_preview_only",
                    "training_steps": 0,
                    "views": records,
                    "camera_height_agl_m": config["camera_height_agl_m"],
                    "camera": config["camera"],
                    "scene": config["scene"],
                    "appearance": manifest["appearance"],
                    "preview_render_subframes": config["preview_render_subframes"],
                    "preview_anti_aliasing": config["preview_anti_aliasing"],
                }
                (args.save_preview / "preview_report.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                print("REAL_SCENE_PREVIEW_READY " + str(args.save_preview), flush=True)
            except Exception:
                error = traceback.format_exc()
                (args.save_preview / "preview_error.log").write_text(error)
                print(error, flush=True)
                if args.exit_after_preview:
                    raise
        # Destroy the extra RGB render product after saving; the GUI viewport
        # continues rendering its selected camera with less GPU work.
        capture.close()
        set_nadir("junction", 70)
        set_view(str(overview.GetPath()))
        if not args.headless:
            window = ui.Window("Real scene preview: 40 m AGL", width=360, height=400)
            with window.frame:
                with ui.VStack(spacing=5):
                    ui.Label("12 houses | 42 trees | 12 x 7 m pool")
                    ui.Label("Preview only - no dataset / training")
                    label[0] = ui.Label("Oblique overview (inspection camera)")
                    ui.Button("Scene overview", clicked_fn=show_overview)
                    for name in config["inspection_positions"]:
                        ui.Button("Downward: " + name, clicked_fn=lambda name=name: set_nadir(name))
                    for fov in config["comparison_hfov_deg"]:
                        ui.Button(
                            "Downward HFOV {:g} deg".format(fov),
                            clicked_fn=lambda fov=fov: set_nadir(fov=fov),
                        )
                    ui.Label("Square camera | 1024 x 1024 | body yaw 0")
            print("REAL_SCENE_GUI_READY", flush=True)
        if not args.exit_after_preview:
            while app.is_running():
                app.update()
    finally:
        if capture is not None:
            capture.close()
        app.close()


if __name__ == "__main__":
    main()
