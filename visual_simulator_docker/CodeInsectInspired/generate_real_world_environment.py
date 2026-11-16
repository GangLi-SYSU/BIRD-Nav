"""Build an independent meter-scale residential/park preview from actual USD assets."""

import argparse
import itertools
import json
import math
from pathlib import Path
import random
import re
import shutil

from insect_utils.experiment_config import PROJECT, digest


HOUSES = (
    ("modern-house", -20, -18, 0),
    ("japanese-house", -43, -20, 12),
    ("suburban-family-house", -20, -44, 180),
    ("alpine-chalet", -44, -45, 90),
    ("japanese-house", -20, 24, 180),
    ("modern-house", -43, 24, 180),
    ("alpine-chalet", -20, 48, 0),
    ("suburban-family-house", -45, 49, 180),
    ("alpine-chalet", 21, 24, 180),
    ("suburban-family-house", 43, 24, 180),
    ("japanese-house", 43, -20, 0),
    ("modern-house", 43, -44, 90),
)
TREES = (
    "Japanese_Maple",
    "Common_Apple",
    "Dogwood",
    "Blue_Berry_Elder",
    "Japanese_Cherry",
    "Service_Berry",
    "Black_Oak",
)


def converted_bounds(low, high, units, up_axis):
    """Convert asset bounds to Z-up meters, retaining the model's physical size."""
    if not math.isfinite(units) or units <= 0 or up_axis not in ("Y", "Z"):
        raise ValueError("Unsupported asset units/up axis")
    if (
        len(low) != 3
        or len(high) != 3
        or any(not math.isfinite(a) or not math.isfinite(b) or b <= a for a, b in zip(low, high))
    ):
        raise ValueError("Asset has empty or non-finite native bounds")
    points = []
    for point in itertools.product(*zip(low, high)):
        x, y, z = [float(v) * units for v in point]
        points.append((x, -z, y) if up_axis == "Y" else (x, y, z))
    result = [
        [min(p[i] for p in points) for i in range(3)],
        [max(p[i] for p in points) for i in range(3)],
    ]
    if any(not math.isfinite(v) for end in result for v in end) or any(
        b <= a for a, b in zip(*result)
    ):
        raise ValueError("Asset has empty or non-finite bounds")
    return result


def mdl_texture_dependencies(paths):
    """USD dependency scans omit texture filenames embedded in MDL defaults."""
    result = set()
    for path in paths:
        path = Path(path)
        if path.suffix.lower() != ".mdl":
            continue
        for name in re.findall(r'texture_2d\(\s*"([^"]+)"', path.read_text()):
            texture = (path.parent / name).resolve()
            if not texture.is_file():
                raise FileNotFoundError("Missing MDL texture: " + str(texture))
            result.add(texture)
    return result


def tree_layout():
    """Landscape independently of flight paths, home positions and training labels."""
    points = [
        (-10, -18),
        (-31, -9),
        (-51, -10),
        (-10, -37),
        (-34, -36),
        (-51, -36),
        (-33, -55),
        (-8, -57),
        (-55, -53),
        (-10, 19),
        (-32, 14),
        (-54, 15),
        (-10, 39),
        (-34, 37),
        (-55, 39),
        (-33, 60),
        (-9, 59),
        (-55, 61),
        (12, 16),
        (32, 14),
        (54, 14),
        (54, 34),
        (33, 35),
        (11, 34),
        (12, -8),
        (31, -9),
        (55, -10),
        (54, -34),
        (33, -34),
        (54, -55),
        (9, -38),
        (22, -43),
        (12, -55),
        (35, -57),
        (12, 47),
        (22, 54),
        (31, 48),
        (9, 61),
        (40, 60),
        (53, 51),
    ]
    rng = random.Random(91)
    items = [(TREES[i % 6], x, y, rng.uniform(0, 360)) for i, (x, y) in enumerate(points)]
    # Large mature trees remain at their native size, away from houses and the pool.
    items.extend([("Black_Oak", 70, 48, 17), ("Black_Oak", -75, -25, 120)])
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "maps/downward_real_h40_v1")
    args = parser.parse_args()
    from tools.check_stage03_scene import import_usd

    Usd, UsdGeom, UsdUtils = import_usd()
    from pxr import Gf, Sdf, UsdLux, UsdShade
    import numpy as np
    from PIL import Image

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = output / "real_scene.usda"
    if scene.exists():
        raise FileExistsError("Use a new bundle directory; existing scene is preserved")
    native_assets = PROJECT.parent / "isaac-sim_assets"
    buildings = PROJECT.parent / "downloads/real_preview_v1"
    catalog = json.loads((buildings / "building_assets.json").read_text())
    metadata = []

    def copy_dependencies(source, root, folder):
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(source))
        if unresolved:
            raise ValueError("Missing source dependencies: " + str(unresolved))
        paths = {Path(layer.realPath).resolve() for layer in layers if layer.realPath}
        paths.update(Path(path).resolve() for path in assets)
        paths.add(source.resolve())
        paths.update(mdl_texture_dependencies(paths))
        for path in sorted(paths):
            relative = path.relative_to(root.resolve())
            destination = output / "assets" / folder / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(path, destination)
        return output / "assets" / folder / source.resolve().relative_to(root.resolve())

    def audit(source, identifier, license_record):
        asset_stage = Usd.Stage.Open(str(source))
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
        box = cache.ComputeWorldBound(asset_stage.GetPseudoRoot()).ComputeAlignedRange()
        if box.IsEmpty() or not asset_stage.GetDefaultPrim():
            raise ValueError("Asset lacks valid geometry/default prim: " + identifier)
        units = UsdGeom.GetStageMetersPerUnit(asset_stage)
        axis = str(UsdGeom.GetStageUpAxis(asset_stage))
        low, high = converted_bounds(box.GetMin(), box.GetMax(), units, axis)
        record = dict(
            id=identifier,
            path=str(source.relative_to(output)),
            source_units_m=units,
            source_up_axis=axis,
            bounds_z_up_m=[low, high],
            size_xyz_m=[b - a for a, b in zip(low, high)],
            source=license_record,
        )
        metadata.append(record)
        return record

    models = {}
    for entry in catalog:
        slug = entry["slug"]
        source = buildings / entry["usd_files"][0]
        root = buildings / slug / "unpacked"
        copied = copy_dependencies(source, root, "houses/" + slug)
        for name in ("LICENSE.txt", "manifest.json"):
            shutil.copyfile(buildings / slug / name, copied.parent / name)
        models[slug] = audit(copied, slug, entry)
    for name in TREES:
        copied = copy_dependencies(
            native_assets / "Trees" / (name + ".usd"), native_assets, "nvidia"
        )
        models[name] = audit(
            copied,
            name,
            {
                "origin": "Existing Isaac asset pack; retain original NVIDIA terms",
                "source_relative_path": "isaac-sim_assets/Trees/" + name + ".usd",
            },
        )

    stage = Usd.Stage.CreateNew(str(scene))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())

    def material(name, color, roughness=0.75, texture=None, metallic=0):
        result = UsdShade.Material.Define(stage, "/World/Looks/" + name)
        shader = UsdShade.Shader.Define(stage, str(result.GetPath()) + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        result.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        if texture:
            reader = UsdShade.Shader.Define(stage, str(result.GetPath()) + "/UV")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            tex = UsdShade.Shader.Define(stage, str(result.GetPath()) + "/Texture")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                reader.ConnectableAPI(), "result"
            )
            shader.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
        return result

    textures = output / "textures"
    textures.mkdir(exist_ok=True)
    shutil.copyfile(
        native_assets / "Grass_Countryside/Grass_Countryside_BaseColor.png", textures / "grass.png"
    )
    rng = np.random.default_rng(91)
    noise = rng.normal(0, 4, (512, 512, 1))
    Image.fromarray(np.uint8(np.clip(np.array([57, 60, 63]) + noise, 0, 255))).save(
        textures / "asphalt.png"
    )
    yy, xx = np.mgrid[:512, :512]
    ripples = (np.sin(xx / 11 + np.sin(yy / 29)) + np.sin(yy / 13 + xx / 37)) * 6
    Image.fromarray(np.uint8(np.clip(np.array([25, 133, 165]) + ripples[..., None], 0, 255))).save(
        textures / "water.png"
    )
    grass = material("Grass", (0.14, 0.25, 0.08), texture="textures/grass.png")
    road = material("Asphalt", (0.08, 0.09, 0.10), texture="textures/asphalt.png")
    paving = material("Paving", (0.55, 0.51, 0.43))
    curb = material("Curb", (0.64, 0.65, 0.63))
    stripe = material("RoadMarking", (0.83, 0.78, 0.53))
    pool_mat = material(
        "PoolWater", (0.02, 0.39, 0.53), roughness=0.16, texture="textures/water.png", metallic=0.08
    )
    tile = material("PoolTile", (0.70, 0.76, 0.73))

    def quad(name, x, y, width, depth, z, mat, repeat_m=4):
        mesh = UsdGeom.Mesh.Define(stage, "/World/" + name)
        mesh.CreatePointsAttr(
            [
                (x - width / 2, y - depth / 2, z),
                (x + width / 2, y - depth / 2, z),
                (x + width / 2, y + depth / 2, z),
                (x - width / 2, y + depth / 2, z),
            ]
        )
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        ).Set(
            [
                (0, 0),
                (width / repeat_m, 0),
                (width / repeat_m, depth / repeat_m),
                (0, depth / repeat_m),
            ]
        )
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        return mesh

    quad("Ground", 0, 0, 400, 400, 0, grass)
    quad("RoadNS", 0, 0, 7, 180, 0.015, road)
    quad("RoadEW", 0, 5, 190, 7, 0.020, road)
    for sign in (-1, 1):
        quad("SidewalkNS" + str(sign + 1), sign * 4.4, 0, 1.6, 180, 0.04, curb)
        quad("SidewalkEW" + str(sign + 1), 0, 5 + sign * 4.4, 190, 1.6, 0.05, curb)
    for i, coordinate in enumerate(range(-84, 85, 8)):
        if abs(coordinate - 5) > 7:
            quad("LineNS_%02d" % i, 0, coordinate, 0.12, 3, 0.03, stripe)
        if abs(coordinate) > 7:
            quad("LineEW_%02d" % i, coordinate, 5, 3, 0.12, 0.035, stripe)
    instances = []

    def place(name, identifier, x, y, yaw, base_z=0):
        model = models[identifier]
        low, high = model["bounds_z_up_m"]
        center = [(a + b) / 2 for a, b in zip(low, high)]
        parent = UsdGeom.Xform.Define(stage, "/World/" + name)
        parent.AddTranslateOp().Set(Gf.Vec3d(x, y, base_z))
        parent.AddRotateZOp().Set(yaw)
        align = UsdGeom.Xform.Define(stage, str(parent.GetPath()) + "/GroundAlign")
        align.AddTranslateOp().Set(Gf.Vec3d(-center[0], -center[1], -low[2]))
        orientation = UsdGeom.Xform.Define(stage, str(align.GetPath()) + "/UnitsAndAxis")
        if model["source_up_axis"] == "Y":
            orientation.AddRotateXOp().Set(90)
        scale = model["source_units_m"]
        orientation.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
        asset = stage.DefinePrim(str(orientation.GetPath()) + "/Asset")
        asset.GetReferences().AddReference(model["path"])
        asset.SetInstanceable(True)
        box = (
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
            .ComputeWorldBound(parent.GetPrim())
            .ComputeAlignedRange()
        )
        entry = dict(
            id=name,
            asset_id=identifier,
            xy_m=[x, y],
            yaw_deg=yaw,
            bbox_world_m=[list(box.GetMin()), list(box.GetMax())],
        )
        if box.IsEmpty() or abs(box.GetMin()[2] - base_z) > 0.002:
            raise ValueError("Asset did not land on its intended ground surface")
        instances.append(entry)

    for i, (slug, x, y, yaw) in enumerate(HOUSES):
        quad("HousePad_%02d" % i, x, y, 18, 16, 0.055, paving)
        place("House_%02d" % i, slug, x, y, yaw, base_z=0.055)
    for i, (slug, x, y, yaw) in enumerate(tree_layout()):
        place("Tree_%02d" % i, slug, x, y, yaw)
    # Normal-scale designed pool, explicitly separate from downloaded assets.
    quad("PoolDeck", 20, -20, 16, 11, 0.07, paving)
    quad("PoolCoping", 20, -20, 13, 8, 0.09, tile)
    quad("PoolWater", 20, -20, 12, 7, 0.095, pool_mat, repeat_m=3)
    for i in range(9):
        quad("PoolDeckJoint_%02d" % i, 12 + 2 * i, -20, 0.025, 11, 0.08, curb)
    quad("ParkPath", 20, 43, 25, 1.8, 0.04, paving)
    quad("ParkPathBranch", 29, 50, 1.5, 14, 0.045, paving)
    dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
    dome.CreateIntensityAttr(650)
    dome.CreateColorAttr(Gf.Vec3f(0.82, 0.88, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(2500)
    sun.CreateAngleAttr(0.8)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-28, -32, 18))
    stage.GetRootLayer().Save()
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(scene))
    dependencies = {Path(layer.realPath).resolve() for layer in layers if layer.realPath}
    dependencies.update(Path(path).resolve() for path in assets)
    dependencies.update(mdl_texture_dependencies(dependencies))
    if unresolved:
        raise ValueError("Generated scene has unresolved dependencies: " + str(unresolved))
    sha = {str(path.relative_to(output)): digest(path) for path in sorted(dependencies)}
    manifest = {
        "schema_version": 1,
        "entry": scene.name,
        "profile": "residential_park_preview",
        "purpose": "scene_preview_only",
        "meters_per_unit": 1,
        "up_axis": "Z",
        "bounds_xy_m": [[-200, -200], [200, 200]],
        "ground_z_m": 0,
        "max_surface_height_agl_m": max(item["bbox_world_m"][1][2] for item in instances),
        "asset_sha256": sha,
        "models": metadata,
        "instances": instances,
        "house_instances": len(HOUSES),
        "tree_instances": len(tree_layout()),
        "pool": {
            "water_size_wh_m": [12, 7],
            "center_xy_m": [20, -20],
            "origin": "procedural static surface, no water dynamics",
        },
        "populated_region_xy_m": [[-90, -70], [90, 75]],
        "layout_note": (
            "Fixed world coordinates; not derived from home, routes, labels or splits. "
            "Outer terrain is a visual buffer, not a populated navigation benchmark."
        ),
        "appearance": (
            "Authored stylized houses, textured existing trees and procedural road/pool; "
            "not photogrammetry."
        ),
    }
    scene.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        "REAL_SCENE_GENERATED "
        + json.dumps(
            {
                "scene": str(scene),
                "houses": len(HOUSES),
                "trees": len(tree_layout()),
                "dependencies": len(sha),
                "max_height_m": manifest["max_surface_height_agl_m"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
