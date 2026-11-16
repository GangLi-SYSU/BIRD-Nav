"""Audit USD dependencies, geometry, texture diagnostics and flight-space budgets."""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.experiment_config import digest, load_config
from insect_utils.scene_geometry import check_positions_and_segments, require_assets, spatial_budget


def import_usd():
    """Use the installed Isaac 4.2 USD libraries without starting Kit."""
    try:
        return tuple(
            importlib.import_module("pxr." + name) for name in ("Usd", "UsdGeom", "UsdUtils")
        )
    except ModuleNotFoundError:
        kit = Path("/isaac-sim/kit")
        libraries = []
        for base in (kit / "exts", Path("/isaac-sim/extscache")):
            for pxr in base.glob("omni.usd.libs*/**/pxr"):
                sys.path.insert(0, str(pxr.parent))
                libraries.append(pxr.parent)
        if libraries and not os.environ.get("BEE_NAV_USD_BOOTSTRAPPED"):
            native = sorted({str(p.parent) for base in libraries for p in base.rglob("libtf.so")})
            plugins = sorted(
                {str(p.parent) for base in libraries for p in base.rglob("plugInfo.json")}
            )
            environment = dict(os.environ, BEE_NAV_USD_BOOTSTRAPPED="1")
            environment["LD_LIBRARY_PATH"] = ":".join(
                native + [os.environ.get("LD_LIBRARY_PATH", "")]
            )
            environment["PXR_PLUGINPATH_NAME"] = ":".join(
                plugins + [os.environ.get("PXR_PLUGINPATH_NAME", "")]
            )
            os.execve(sys.executable, [sys.executable] + sys.argv, environment)
        return tuple(
            importlib.import_module("pxr." + name) for name in ("Usd", "UsdGeom", "UsdUtils")
        )


def inspect_usd(path):
    Usd, UsdGeom, UsdUtils = import_usd()
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise ValueError("Cannot open USD: " + str(path))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    objects = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Boundable) or prim.IsA(UsdGeom.Camera):
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            continue
        low, high = list(box.GetMin()), list(box.GetMax())
        objects.append(
            {
                "prim": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "min_xyz_m": low,
                "max_xyz_m": high,
            }
        )
    dependencies = sorted(
        set(
            [str(Path(layer.realPath).resolve()) for layer in layers if layer.realPath]
            + [str(Path(a).resolve()) for a in assets]
        )
    )
    return {
        "entry": str(path),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "dependencies": dependencies,
        "unresolved": sorted(set(str(p) for p in unresolved)),
        "objects": objects,
        "geometry_count": len(objects),
    }


def planned_positions(spec):
    import numpy as np
    from insect_utils.flight_path_functions import generate_bee_path

    home = spec["home"]["ground_xy_m"] + [
        spec["scene"]["ground_z_m"] + spec["flight"]["camera_height_agl_m"]
    ]
    learning = generate_bee_path(*spec["capture"]["shape_params"], home)
    starts = []
    for scale in spec["evaluation"]["radius_scales"]:
        for theta in np.linspace(
            0, 2 * np.pi, spec["evaluation"]["points_per_radius"], endpoint=False
        ):
            radius = spec["evaluation"]["radius_m"] * scale
            starts.append(
                [
                    home[0] + radius * float(np.cos(theta)),
                    home[1] + radius * float(np.sin(theta)),
                    home[2],
                ]
            )
    return learning, starts


def texture_diagnostics(scene, manifest):
    import numpy as np
    from PIL import Image

    texture = next(name for name in manifest["asset_sha256"] if name.endswith(".png"))
    with Image.open(scene.parent / texture) as image:
        pixels = np.asarray(image.convert("RGB"))
    size = pixels.shape[0]
    side = manifest["texture"]["physical_extent_wh_m"][0]
    window = max(8, round(5.35898385 / side * size))
    patches = []
    # Every pose is interior; do not include corners or periodic interpolation seams.
    for fy, fx in ((0.3, 0.3), (0.3, 0.6), (0.6, 0.3), (0.6, 0.6)):
        y, x = round(fy * size), round(fx * size)
        patches.append(
            float(pixels[y : y + window, x : x + window].astype(float).std(axis=(0, 1)).mean())
        )
    shift = max(1, round(2.0 / side * size))
    a = pixels[size // 4 : size // 2, size // 4 : size // 2].astype(float)
    b = pixels[size // 4 : size // 2, size // 4 + shift : size // 2 + shift].astype(float)
    values = {
        "local_5_36m_spatial_std_rgb_mean": patches,
        "translation_2m_mean_absolute_rgb_difference": float(np.abs(a - b).mean()),
        "footprint_texture_pixels_at_10m_hfov30": 5.35898385 / side * size,
        "note": "Texture-space checks only; no navigation performance claim.",
    }
    profile = manifest["profile"]
    if profile == "distinct" and (
        min(patches) < 3 or values["translation_2m_mean_absolute_rgb_difference"] < 3
    ):
        raise ValueError("Distinct ground lacks local texture variation")
    if profile == "repeated" and values["translation_2m_mean_absolute_rgb_difference"] != 0:
        raise ValueError("Repeated diagnostic does not repeat exactly")
    if profile == "low_texture" and max(patches) != 0:
        raise ValueError("Low-texture diagnostic unexpectedly contains spatial detail")
    return values


def check_bundle(spec, bundle):
    import math

    learning, starts = planned_positions(spec)
    reports = {}
    for profile in ("distinct", "repeated", "low_texture"):
        scene = bundle / (profile + ".usda")
        manifest = require_assets(scene)
        geometry = inspect_usd(scene)
        if geometry["unresolved"] or geometry["geometry_count"] != 1:
            raise ValueError("New scene has missing dependencies or unexpected geometry")
        if geometry["meters_per_unit"] != 1 or geometry["up_axis"] != "Z":
            raise ValueError("New scene must use meters and Z-up")
        declared = {str((bundle / name).resolve()) for name in manifest["asset_sha256"]}
        if set(geometry["dependencies"]) != declared:
            raise ValueError("USD dependency closure differs from the declared bundle assets")
        obj = geometry["objects"][0]
        low, high = manifest["bounds_xy_m"]
        for actual, expected in zip(
            obj["min_xyz_m"] + obj["max_xyz_m"],
            low + [manifest["ground_z_m"]] + high + [manifest["ground_z_m"]],
        ):
            if not math.isclose(actual, expected, abs_tol=1e-5):
                raise ValueError("USD geometry differs from manifest bounds")
        budget = spatial_budget(spec, manifest)
        reports[profile] = {
            "geometry": geometry,
            "manifest_sha256": digest(scene.with_suffix(".manifest.json")),
            "budget": budget,
            "learning_path_check": check_positions_and_segments(learning, budget),
            "evaluation_start_check": check_positions_and_segments(starts, budget),
            "texture_diagnostics": texture_diagnostics(scene, manifest),
            "clearance_verification": (
                "Entire ground is one flat quad with no elevated surfaces; "
                "endpoint/segment checks are analytic."
            ),
            "status": "passed",
        }
    return reports


def audit_legacy():
    reports = {}
    for name in ("park", "urban", "courtyard", "warehouse", "greenhouse"):
        geometry = inspect_usd(PROJECT / ("maps/env_" + name + ".usd"))
        units = geometry["meters_per_unit"]
        # Legacy authored units may differ; explicitly convert world extents to meters.
        for obj in geometry["objects"]:
            for key in ("min_xyz_m", "max_xyz_m"):
                obj[key] = [v * units for v in obj[key]]
        nearby = [
            o
            for o in geometry["objects"]
            if all(o["min_xyz_m"][i] <= 30 and o["max_xyz_m"][i] >= -30 for i in range(2))
        ]
        intersections = [o["prim"] for o in nearby if o["min_xyz_m"][2] <= 10 <= o["max_xyz_m"][2]]
        high_surfaces = [o["prim"] for o in nearby if o["max_xyz_m"][2] > 1]
        geometry["migration"] = {
            "status": "blocked_missing_assets"
            if geometry["unresolved"]
            else "requires_surface_and_clearance_review",
            "aabb_intersections_at_z10_near_xy30m": intersections,
            "elevated_geometry_near_xy30m": high_surfaces,
            "automatic_migration_approved": False,
            "note": (
                "AABB screening is conservative, not triangle ray casting or proof of free space. "
                "Keep original USD unchanged."
            ),
        }
        reports[name] = geometry
        print("Audited legacy scene: " + name, flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "config/experiments/downward_h10_hfov30.json"
    )
    parser.add_argument("--bundle", type=Path, default=PROJECT / "maps/downward_flat_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    import_usd()
    args.output.mkdir(parents=True, exist_ok=False)
    resolved = load_config(args.config)
    report = {"new_scenes": check_bundle(resolved["spec"], args.bundle)}
    if args.legacy:
        report["legacy_scenes"] = audit_legacy()
    report.update(
        status="passed",
        camera_adapter_ready=False,
        note="Stage C scene geometry only; no model training or homing results.",
    )
    (args.output / "scene_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print("STAGE03_SCENE_PASS " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
