"""Render terrain inspection views only; this is not the experiment camera adapter."""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=PROJECT / "maps/downward_flat_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from omni.isaac.kit import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "width": 512,
            "height": 512,
            "renderer": "RayTracedLighting",
            "anti_aliasing": 0,
        }
    )
    try:
        import math
        import numpy as np
        from PIL import Image
        from pxr import Gf, UsdGeom
        import omni.usd
        import omni.replicator.core as rep

        records = []
        previous = None
        for profile in ("distinct", "repeated", "low_texture"):
            omni.usd.get_context().open_stage(str(args.bundle / (profile + ".usda")))
            for _ in range(10):
                app.update()
            stage = omni.usd.get_context().get_stage()
            camera = UsdGeom.Camera.Define(stage, "/World/StageCInspectionCamera")
            camera.GetFocalLengthAttr().Set(24.0)
            aperture = 2 * 24 * math.tan(math.radians(30) / 2)
            camera.GetHorizontalApertureAttr().Set(aperture)
            camera.GetVerticalApertureAttr().Set(aperture)
            camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 1000))
            translate = UsdGeom.Xformable(camera).AddTranslateOp()
            product = rep.create.render_product(str(camera.GetPath()), (512, 512))
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach([product])
            poses = [(0, 0, 10), (20, 0, 10), (40, 40, 10), (0, 0, 210)]
            if profile != "distinct":
                poses = poses[:1]
            for index, pose in enumerate(poses):
                translate.Set(Gf.Vec3d(*pose))
                for _ in range(2):
                    rep.orchestrator.step(rt_subframes=8, pause_timeline=False)
                raw = np.asarray(annotator.get_data()).copy()
                if raw.shape[:2] != (512, 512) or raw.ndim != 3 or raw.shape[2] < 3:
                    raise ValueError("No valid inspection image")
                rgb = raw[:, :, :3]
                black_fraction = float((rgb.max(axis=2) < 5).mean())
                if pose[2] == 10 and black_fraction > 0.005:
                    raise ValueError("Local inspection view has missing/black terrain")
                spatial_std = float(rgb.astype(float).std(axis=(0, 1)).mean())
                if profile == "distinct" and pose[2] == 10 and spatial_std < 3:
                    raise ValueError("Rendered distinct ground has no local detail")
                delta = None
                if profile == "distinct" and previous is not None:
                    delta = float(np.abs(rgb.astype(float) - previous).mean())
                    if delta < 1:
                        raise ValueError("Terrain inspection pose change produced a stale frame")
                if profile == "distinct":
                    previous = rgb.astype(float)
                filename = profile + "_{:02d}.png".format(index)
                Image.fromarray(rgb).save(args.output / filename)
                records.append(
                    {
                        "profile": profile,
                        "inspection_position_m": pose,
                        "image": filename,
                        "black_fraction": black_fraction,
                        "mean_rgb": rgb.mean(axis=(0, 1)).tolist(),
                        "spatial_std_rgb_mean": spatial_std,
                        "mean_abs_change_from_previous": delta,
                    }
                )
                print("Rendered terrain inspection: " + filename, flush=True)
            annotator.detach([product.path])
            product.destroy()
        report = {
            "status": "passed",
            "views": records,
            "training_steps": 0,
            "purpose": "stage_C_terrain_inspection_only",
            "experiment_camera_adapter_implemented": False,
            "note": (
                "Diagnostic USD -Z camera only. Body/yaw alignment, measured FOV "
                "and capture/inference agreement await stage D."
            ),
        }
        # Kit may terminate the process during close: publish the report first.
        (args.output / "render_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("STAGE03_RENDER_PASS " + str(args.output), flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
