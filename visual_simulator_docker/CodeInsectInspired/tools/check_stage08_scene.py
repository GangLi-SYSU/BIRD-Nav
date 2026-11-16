"""Verify the enlarged landmark mesh scene and coverage before recapturing data."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.experiment_config import digest
from insect_utils.scene_geometry import require_assets
from tools.check_stage03_scene import import_usd, inspect_usd
from tools.check_stage07_overfit import candidates


def check(scene, output):
    Usd, UsdGeom, _ = import_usd()
    manifest = require_assets(scene)
    geometry = inspect_usd(scene)
    if geometry["unresolved"] or geometry["geometry_count"] != 1 + len(manifest["landmarks"]):
        raise ValueError("Scene geometry or dependency closure is incomplete")
    if geometry["meters_per_unit"] != 1 or geometry["up_axis"] != "Z":
        raise ValueError("Scene must be meter-scaled and Z-up")
    stage = Usd.Stage.Open(str(scene))
    checked = []
    for item in manifest["landmarks"]:
        prim = stage.GetPrimAtPath("/World/Landmarks/" + item["id"])
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            raise ValueError("Missing landmark mesh")
        points = mesh.GetPointsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        if sum(counts) != len(indices) or any(i < 0 or i >= len(points) for i in indices):
            raise ValueError("Invalid mesh topology")
        height = item["center_xyz_m"][2]
        low = min(float(p[2]) for p in points) + height
        high = max(float(p[2]) for p in points) + height
        if abs(low) > 1e-5 or abs(high - item["top_z_m"]) > 1e-5 or high > 3.00001:
            raise ValueError("Mesh floats, penetrates ground or exceeds the height limit")
        accent = prim.GetChild("Accent")
        marked = accent.GetAttribute("indices").Get()
        if not marked or any(i < 0 or i >= len(counts) for i in marked):
            raise ValueError("Landmark has no valid asymmetric color subset")
        checked.append(item["id"])
    if digest(scene.parent / "ground.png") != digest(
        PROJECT / "maps/downward_landmarks_v1/ground.png"
    ):
        raise ValueError("This comparison must preserve the original ground texture")
    prepared = PROJECT / "runs/stage06_data/position_groups_01"
    rows = json.loads((prepared / "records.json").read_text())
    split = json.loads((prepared / "splits.json").read_text())["indices"]
    coverage = {}
    for name, indices in split.items():
        numbers = [candidates(rows[i], manifest["landmarks"]) for i in indices]
        coverage[name] = {
            "images": len(numbers),
            "zero_candidates": numbers.count(0),
            "zero_candidate_fraction": numbers.count(0) / len(numbers),
            "histogram": dict(Counter(numbers)),
        }
    if coverage["train"]["zero_candidate_fraction"] > 0.05:
        raise ValueError("Training zero-candidate fraction exceeds the user-approved 5%")
    report = {
        "status": "passed",
        "geometry": geometry,
        "checked_meshes": len(checked),
        "coverage": coverage,
        "shape_counts": dict(Counter(p["shape"] for p in manifest["landmarks"])),
        "shape_color_pairs": len(
            {(p["shape"], p["material_index"]) for p in manifest["landmarks"]}
        ),
        "height_limit_m": 3,
        "ground_texture_unchanged": True,
        "coverage_scope": "Conservative geometric candidates, not semantic visibility",
        "closed_loop_run": False,
        "training_steps": 0,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "scene_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("STAGE08_SCENE_PASS", str(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", type=Path, default=PROJECT / "maps/downward_landmarks_v2/landmarks.usda"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.scene.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
