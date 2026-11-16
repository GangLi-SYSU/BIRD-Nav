"""Verify real-scene physical units, complete dependencies and preview camera bounds."""

import argparse
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.camera_geometry import optics
from insect_utils.scene_geometry import require_assets
from generate_real_world_environment import mdl_texture_dependencies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "runs/stage09_real_preview/geometry_report.json"
    )
    args = parser.parse_args()
    from tools.check_stage03_scene import import_usd

    Usd, UsdGeom, UsdUtils = import_usd()
    config = json.loads((PROJECT / "config/experiments/stage09_preview.json").read_text())
    scene = PROJECT / config["scene"]
    manifest = require_assets(scene)
    stage = Usd.Stage.Open(str(scene))
    if UsdGeom.GetStageMetersPerUnit(stage) != 1 or UsdGeom.GetStageUpAxis(stage) != "Z":
        raise ValueError("Preview stage must use meters and Z-up")
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(scene))
    dependencies = {Path(layer.realPath).resolve() for layer in layers if layer.realPath}
    dependencies.update(Path(path).resolve() for path in assets)
    mdl_textures = mdl_texture_dependencies(dependencies)
    dependencies.update(mdl_textures)
    if unresolved or dependencies != {
        (scene.parent / p).resolve() for p in manifest["asset_sha256"]
    }:
        raise ValueError("Scene dependency closure differs from its hashed manifest")
    models = {model["id"]: model for model in manifest["models"]}
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    max_height = 0
    for item in manifest["instances"]:
        box = cache.ComputeWorldBound(
            stage.GetPrimAtPath("/World/" + item["id"])
        ).ComputeAlignedRange()
        low, high = box.GetMin(), box.GetMax()
        if box.IsEmpty() or low[2] < -0.002:
            raise ValueError("Asset is empty or lies below the ground")
        expected_height = models[item["asset_id"]]["size_xyz_m"][2]
        if not math.isclose(high[2] - low[2], expected_height, abs_tol=0.002):
            raise ValueError("Native asset height changed during placement")
        for actual, expected in zip(list(low) + list(high), sum(item["bbox_world_m"], [])):
            if not math.isclose(actual, expected, abs_tol=0.002):
                raise ValueError("Actual instance bounds differ from manifest")
        max_height = max(max_height, high[2])
    if len(manifest["instances"]) != manifest["house_instances"] + manifest["tree_instances"]:
        raise ValueError("Asset instance count differs")
    height = config["camera_height_agl_m"]
    if height - max_height < 10:
        raise ValueError("Preview has less than 10 m vertical clearance")
    cameras = []
    for fov in config["comparison_hfov_deg"]:
        spec = dict(config["camera"], hfov_deg=fov)
        settings = optics(spec)
        side = 2 * height * math.tan(math.radians(fov) / 2)
        radius = side / math.sqrt(2)
        for xy in config["inspection_positions"].values():
            if any(abs(v) + radius + 5 > 200 for v in xy):
                raise ValueError("Camera footprint crosses ground boundary")
        cameras.append(
            {
                "hfov_deg": fov,
                "vfov_deg": settings["vfov_deg"],
                "ground_footprint_wh_m": [side, side],
                "raw_ground_m_per_pixel": side / 1024,
                "ground_m_per_pixel_after_384_resize": side / 384,
            }
        )
    report = {
        "status": "passed",
        "purpose": "scene_preview_only",
        "training_steps": 0,
        "house_instances": manifest["house_instances"],
        "tree_instances": manifest["tree_instances"],
        "dependencies_checked": len(dependencies),
        "implicit_mdl_textures_checked": len(mdl_textures),
        "unresolved": [],
        "physical_scale_preserved": True,
        "max_surface_height_m": max_height,
        "camera_height_agl_m": height,
        "minimum_vertical_clearance_m": height - max_height,
        "comparison_cameras": cameras,
        "populated_region_xy_m": manifest["populated_region_xy_m"],
        "note": "Geometry/preview checks only; dataset coverage and navigation are not evaluated.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("REAL_SCENE_GEOMETRY_PASS " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
