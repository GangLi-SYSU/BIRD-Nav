"""Browse stage C terrain in Isaac; no capture, model, training or experiment run."""

import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def preview_layout(manifest):
    low, high = manifest["bounds_xy_m"]
    center = [(a + b) / 2 for a, b in zip(low, high)]
    side = max(b - a for a, b in zip(low, high))
    x, y = center
    ground = manifest["ground_z_m"]
    return {
        "side_m": side,
        "StageCOverview": ((x, y - 0.65 * side, ground + 1.4 * side), (x, y, ground)),
        "StageCLearningArea": ((x + 30, y - 40, ground + 55), (x, y, ground + 1)),
        "StageCAt10m": ((x, y, ground + 10), (x, y, ground)),
    }


def export_preview(output, manifest, set_view, set_inspection):
    import json
    import numpy as np
    from PIL import Image
    import omni.replicator.core as rep

    output.mkdir(parents=True, exist_ok=False)
    cases = [
        ("central_oblique", "StageCLearningArea", None),
        ("whole_field", "StageCOverview", None),
    ]
    for shape in ("Cube", "Cylinder", "Cone", "Sphere"):
        item = min(
            (p for p in manifest["landmarks"] if p["shape"] == shape),
            key=lambda p: sum(v ** 2 for v in p["center_xyz_m"][:2]),
        )
        cases.append((shape.lower() + "_at10m", "StageCAt10m", item))
    records = []
    for name, camera, landmark in cases:
        if landmark:
            set_inspection(landmark["center_xyz_m"][:2])
        set_view("/World/" + camera)
        product = rep.create.render_product("/World/" + camera, (768, 768))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([product])
        try:
            for _ in range(2):
                rep.orchestrator.step(rt_subframes=4, pause_timeline=False)
            raw = np.asarray(annotator.get_data()).copy()
            if raw.ndim != 3 or raw.shape[:2] != (768, 768) or raw.shape[2] < 3:
                raise ValueError("Preview renderer returned no RGB image")
            rgb = raw[:, :, :3]
            std = float(rgb.astype(float).std(axis=(0, 1)).mean())
            black_fraction = float((rgb.max(axis=2) < 5).mean())
            if landmark and (black_fraction > 0.005 or std < 3):
                raise ValueError("Landmark preview is blank, stale or missing terrain")
            Image.fromarray(rgb).save(output / (name + ".png"))
            records.append(
                {
                    "image": name + ".png",
                    "camera": camera,
                    "landmark_id": landmark["id"] if landmark else None,
                    "camera_xy_m": landmark["center_xyz_m"][:2] if landmark else None,
                    "spatial_std_rgb_mean": std,
                    "black_fraction": black_fraction,
                }
            )
            print("PREVIEW_CAPTURED " + name, flush=True)
        finally:
            annotator.detach([product.path])
            product.destroy()
    report = {
        "status": "passed",
        "views": records,
        "training_steps": 0,
        "purpose": "landmark_scene_visual_inspection_only",
        "experiment_camera_adapter_ready": False,
    }
    (output / "preview_report.json").write_text(json.dumps(report, indent=2) + "\n")
    set_inspection([0, 0])
    set_view("/World/StageCLearningArea")
    print("LANDMARK_PREVIEW_PASS " + str(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=PROJECT / "maps/downward_flat_v1")
    parser.add_argument(
        "--scene", type=Path, help="Browse one scene instead of the three flat profiles"
    )
    parser.add_argument(
        "--save-preview", type=Path, help="Export six landmark inspection views once"
    )
    args = parser.parse_args()
    from omni.isaac.kit import SimulationApp

    app = SimulationApp(
        {
            "headless": False,
            "width": 1280,
            "height": 900,
            "renderer": "RayTracedLighting",
            "anti_aliasing": 0,
        }
    )
    try:
        from pxr import Gf, UsdGeom
        import omni.ui as ui
        import omni.usd
        from omni.kit.viewport.utility import get_active_viewport
        from insect_utils.scene_geometry import require_assets
        import carb

        carb.settings.get_settings().set("/app/runLoops/main/rateLimitEnabled", True)
        carb.settings.get_settings().set("/app/runLoops/main/rateLimitFrequency", 30)
        pending = [None]
        loaded = {}
        options = (
            {args.scene.stem: args.scene.resolve()}
            if args.scene
            else {
                profile: args.bundle / (profile + ".usda")
                for profile in ("distinct", "repeated", "low_texture")
            }
        )

        def request_profile(profile):
            pending[0] = profile

        def set_view(path):
            get_active_viewport().camera_path = path

        def set_inspection(xy):
            ground = loaded["manifest"]["ground_z_m"]
            matrix = (
                Gf.Matrix4d()
                .SetLookAt(
                    Gf.Vec3d(xy[0], xy[1], ground + 10),
                    Gf.Vec3d(xy[0], xy[1], ground),
                    Gf.Vec3d(0, 1, 0),
                )
                .GetInverse()
            )
            loaded["cameras"]["StageCAt10m"].GetOrderedXformOps()[0].Set(matrix)

        def next_landmark():
            items = loaded["manifest"].get("landmarks", [])
            if not items:
                return
            nearest = sorted(items, key=lambda p: sum(v ** 2 for v in p["center_xyz_m"][:2]))
            index = loaded.get("landmark_index", 0) % len(nearest)
            set_inspection(nearest[index]["center_xyz_m"][:2])
            loaded["landmark_index"] = index + 1
            set_view("/World/StageCAt10m")

        def load(profile):
            scene = options[profile]
            manifest = require_assets(scene)
            omni.usd.get_context().open_stage(str(scene))
            for _ in range(5):
                app.update()
            stage = omni.usd.get_context().get_stage()
            loaded.update(manifest=manifest, cameras={}, landmark_index=0)
            layout = preview_layout(manifest)
            for name in ("StageCOverview", "StageCLearningArea", "StageCAt10m"):
                eye, target = layout[name]
                camera = UsdGeom.Camera.Define(stage, "/World/" + name)
                camera.GetFocalLengthAttr().Set(24)
                camera.GetHorizontalApertureAttr().Set(12.861561 if name.endswith("At10m") else 36)
                camera.GetVerticalApertureAttr().Set(12.861561 if name.endswith("At10m") else 24)
                camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, max(1000, layout["side_m"] * 5)))
                up = Gf.Vec3d(0, 0, 1) if eye[1] else Gf.Vec3d(0, 1, 0)
                matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), up).GetInverse()
                UsdGeom.Xformable(camera).AddTransformOp().Set(matrix)
                loaded["cameras"][name] = UsdGeom.Xformable(camera)
            set_view(
                "/World/StageCLearningArea"
                if manifest.get("landmarks")
                else "/World/StageCOverview"
            )
            print("SCENE_VIEW_READY " + profile, flush=True)

        load(next(iter(options)))
        if args.save_preview:
            if not loaded["manifest"].get("landmarks"):
                parser.error("--save-preview requires a landmark scene")
            try:
                export_preview(args.save_preview, loaded["manifest"], set_view, set_inspection)
            except Exception:
                import traceback

                error = traceback.format_exc()
                print(error, flush=True)
                args.save_preview.mkdir(parents=True, exist_ok=True)
                (args.save_preview / "preview_error.log").write_text(error)
                set_view("/World/StageCLearningArea")
        side = preview_layout(loaded["manifest"])["side_m"]
        window = ui.Window("Stage C: terrain preview", width=350, height=310)
        with window.frame:
            with ui.VStack(spacing=5):
                ui.Label("{:g} x {:g} m | meters".format(side, side))
                ui.Label("Scene preview | no training or homing")
                ui.Button(
                    "Whole field overview", clicked_fn=lambda: set_view("/World/StageCOverview")
                )
                if loaded["manifest"].get("landmarks"):
                    ui.Label(
                        "{} colored geometric landmarks".format(
                            len(loaded["manifest"]["landmarks"])
                        )
                    )
                    ui.Button(
                        "Central learning area",
                        clicked_fn=lambda: set_view("/World/StageCLearningArea"),
                    )
                    ui.Button("Next landmark at 10 m", clicked_fn=next_landmark)
                ui.Button(
                    "Downward inspection at 10 m", clicked_fn=lambda: set_view("/World/StageCAt10m")
                )
                if not args.scene:
                    ui.Button("Distinct ground", clicked_fn=lambda: request_profile("distinct"))
                    ui.Button(
                        "Repeated ground (diagnostic)",
                        clicked_fn=lambda: request_profile("repeated"),
                    )
                    ui.Button(
                        "Low texture (diagnostic)",
                        clicked_fn=lambda: request_profile("low_texture"),
                    )
        while app.is_running():
            if pending[0] is not None:
                profile, pending[0] = pending[0], None
                load(profile)
            app.update()
    finally:
        app.close()


if __name__ == "__main__":
    main()
