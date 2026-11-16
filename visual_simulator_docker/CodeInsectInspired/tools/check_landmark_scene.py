"""Verify actual landmark USD geometry, dependencies and the current flight budget."""

import argparse
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.experiment_config import load_config
from insect_utils.scene_geometry import check_positions_and_segments, require_assets, spatial_budget
from tools.check_stage03_scene import import_usd, inspect_usd, planned_positions


def check_landmarks(scene, spec):
    manifest = require_assets(scene)
    geometry = inspect_usd(scene)
    if geometry["meters_per_unit"] != 1 or geometry["up_axis"] != "Z":
        raise ValueError("Landmark scene must use meters and Z-up")
    declared = {str((scene.parent / name).resolve()) for name in manifest["asset_sha256"]}
    if geometry["unresolved"] or set(geometry["dependencies"]) != declared:
        raise ValueError("Landmark scene dependency closure is incomplete")
    objects = {p["prim"]: p for p in geometry["objects"]}
    landmarks = manifest["landmarks"]
    if len(objects) != len(landmarks) + 1:
        raise ValueError("Unexpected geometry count in landmark scene")
    ground = objects.pop("/World/Ground")
    low, high = manifest["bounds_xy_m"]
    expected = low + [0] + high + [0]
    if any(
        not math.isclose(a, b, abs_tol=1e-4)
        for a, b in zip(ground["min_xyz_m"] + ground["max_xyz_m"], expected)
    ):
        raise ValueError("Ground bounds disagree with manifest")
    for item in landmarks:
        obj = objects["/World/Landmarks/" + item["id"]]
        if obj["type"] != item["shape"]:
            raise ValueError("Landmark shape disagrees with manifest")
        for axis in range(2):
            if obj["min_xyz_m"][axis] < low[axis] or obj["max_xyz_m"][axis] > high[axis]:
                raise ValueError("Landmark extends beyond ground")
            center = (obj["min_xyz_m"][axis] + obj["max_xyz_m"][axis]) / 2
            if not math.isclose(center, item["center_xyz_m"][axis], abs_tol=1e-4):
                raise ValueError("Landmark XY placement disagrees with manifest")
        if not math.isclose(obj["min_xyz_m"][2], 0, abs_tol=1e-4):
            raise ValueError("Landmark does not sit on ground")
        if not math.isclose(obj["max_xyz_m"][2], item["top_z_m"], abs_tol=1e-4):
            raise ValueError("Landmark height disagrees with manifest")
    actual_top = max(p["max_xyz_m"][2] for p in objects.values())
    if not math.isclose(actual_top, manifest["max_surface_height_agl_m"], abs_tol=1e-4):
        raise ValueError("Maximum surface height was understated")
    budget = spatial_budget(spec, manifest)
    learning, starts = planned_positions(spec)
    counts = {
        shape: sum(p["shape"] == shape for p in landmarks)
        for shape in ("Cube", "Cylinder", "Cone", "Sphere")
    }
    return {
        "status": "passed",
        "geometry": geometry,
        "shape_counts": counts,
        "budget": budget,
        "learning_positions": check_positions_and_segments(learning, budget),
        "evaluation_starts": check_positions_and_segments(starts, budget),
        "lowest_clearance_above_all_landmark_tops_m": budget["minimum_surface_clearance_m"],
        "note": (
            "Geometry preview only. Elevated surfaces change local AGL and visible footprints; "
            "this is not a trained navigation result."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "config/experiments/downward_h10_hfov30_landmarks.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import_usd()
    resolved = load_config(args.config)
    scene = PROJECT / resolved["spec"]["scene"]["path"]
    report = check_landmarks(scene, resolved["spec"])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "scene_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("LANDMARK_SCENE_PASS " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
