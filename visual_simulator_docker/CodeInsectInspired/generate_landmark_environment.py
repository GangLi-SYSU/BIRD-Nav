"""Create a larger, independent terrain with static colored geometric landmarks."""

import argparse
import json
import math
from pathlib import Path
import random

from generate_downward_environment import make_texture, usd_source
from insect_utils.experiment_config import PROJECT, digest


PALETTE = [
    (0.85, 0.13, 0.12),
    (0.10, 0.42, 0.92),
    (0.96, 0.68, 0.06),
    (0.12, 0.75, 0.68),
    (0.66, 0.19, 0.83),
    (0.96, 0.36, 0.08),
    (0.74, 0.82, 0.91),
    (0.16, 0.28, 0.38),
    (0.84, 0.28, 0.53),
    (0.48, 0.72, 0.14),
    (0.29, 0.51, 0.66),
    (0.63, 0.40, 0.17),
]
SHAPES = ("Cube", "Cylinder", "Cone", "Sphere")


def landmark_layout(seed=10, side_m=400.0, central_count=96, outer_count=96):
    """Seeded random packing, no special home object or path-conditioned placement."""
    if side_m < 120 or not math.isfinite(side_m):
        raise ValueError("Landmark terrain requires a finite side >=120 m")
    if (
        any(
            isinstance(v, bool) or not isinstance(v, int) or v < 0
            for v in (central_count, outer_count)
        )
        or central_count + outer_count < 4
    ):
        raise ValueError("Use nonnegative integer counts with at least four landmarks")
    rng = random.Random(seed)
    layout = []
    for zone, count in (("central", central_count), ("outer", outer_count)):
        for _ in range(count):
            index = len(layout)
            shape = SHAPES[index % len(SHAPES)]
            radius = rng.uniform(0.55, 1.15)
            height = rng.uniform(0.9, 3.0)
            size = [rng.uniform(0.9, 2.4), rng.uniform(0.9, 2.4), height]
            if shape == "Sphere":
                size = [2 * radius] * 3
                height = size[2]
            elif shape != "Cube":
                size = [2 * radius, 2 * radius, height]
            reach = math.hypot(size[0], size[1]) / 2 if shape == "Cube" else radius
            bound = 36.0 if zone == "central" else side_m / 2 - 15
            for attempt in range(50000):
                x, y = rng.uniform(-bound, bound), rng.uniform(-bound, bound)
                if zone == "outer" and max(abs(x), abs(y)) < 45:
                    continue
                if all(
                    math.hypot(x - p["center_xyz_m"][0], y - p["center_xyz_m"][1])
                    >= reach + p["bounding_radius_xy_m"] + 1.0
                    for p in layout
                ):
                    break
            else:
                raise ValueError("Cannot pack requested landmark count without overlaps")
            layout.append(
                {
                    "id": "Landmark_{:03d}".format(index),
                    "shape": shape,
                    "zone": zone,
                    "center_xyz_m": [round(x, 6), round(y, 6), height / 2],
                    "size_xyz_m": size,
                    "radius_m": radius if shape != "Cube" else None,
                    "yaw_deg": rng.uniform(0, 360) if shape == "Cube" else 0.0,
                    "color_rgb": PALETTE[(index * 5) % len(PALETTE)],
                    "material_index": (index * 5) % len(PALETTE),
                    "bounding_radius_xy_m": reach,
                    "top_z_m": height,
                }
            )
    return layout


def landmark_usd(layout):
    parts = ['    def Scope "LandmarkMaterials" {']
    for index, color in enumerate(PALETTE):
        parts.append(
            """        def Material "Color_INDEX" {
            SURFACE_LINK
            def Shader "Surface" {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = COLOR
                float inputs:roughness = 0.65
                float inputs:metallic = 0
                token outputs:surface
            }
        }""".replace(
                "SURFACE_LINK",
                "token outputs:surface.connect = "
                "</World/LandmarkMaterials/Color_INDEX/Surface.outputs:surface>",
            )
            .replace("INDEX", str(index))
            .replace("COLOR", str(color))
        )
    parts.append('    }\n    def Scope "Landmarks" {')
    for item in layout:
        shape = item["shape"]
        size = item["size_xyz_m"]
        height = size[2]
        if shape == "Cube":
            properties = (
                "double size = 1\n            float3[] extent = [(-0.5,-0.5,-0.5),(0.5,0.5,0.5)]"
            )
            scale = size
        else:
            radius = item["radius_m"]
            scale = [1, 1, 1]
            half_z = radius if shape == "Sphere" else height / 2
            properties = "double radius = {}\n".format(radius)
            if shape != "Sphere":
                properties += "            double height = {}\n".format(height)
                properties += '            uniform token axis = "Z"\n'
            properties += "            float3[] extent = [{},{}]".format(
                (-radius, -radius, -half_z), (radius, radius, half_z)
            )
        parts.append(
            """        def SHAPE "NAME" (
            prepend apiSchemas = ["MaterialBindingAPI", "PhysicsCollisionAPI"]
        ) {
            PROPERTIES
            double3 xformOp:translate = POSITION
            float xformOp:rotateZ = YAW
            float3 xformOp:scale = SCALE
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ", "xformOp:scale"]
            bool physics:collisionEnabled = true
            rel material:binding = </World/LandmarkMaterials/Color_MATERIAL>
        }""".replace(
                "SHAPE", shape
            )
            .replace("NAME", item["id"])
            .replace("PROPERTIES", properties)
            .replace("POSITION", str(tuple(item["center_xyz_m"])))
            .replace("YAW", str(item["yaw_deg"]))
            .replace("SCALE", str(tuple(scale)))
            .replace("MATERIAL", str(item["material_index"]))
        )
    parts.append(
        """    }
    def DistantLight "Sun" {
        float inputs:intensity = 1200
        float inputs:angle = 1
        color3f inputs:color = (1, 0.96, 0.90)
        float3 xformOp:rotateXYZ = (-35, -20, -20)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }"""
    )
    return "\n".join(parts)


def generate_scene(
    output_dir, side_m=400.0, resolution=8192, seed=10, central_count=96, outer_count=96
):
    from PIL import Image

    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 128:
        raise ValueError("Resolution must be an integer >=128")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    layout = landmark_layout(seed, side_m, central_count, outer_count)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    texture = output_dir / "ground.png"
    image = make_texture("distinct", side_m, resolution, seed)
    image.save(texture)
    image.resize((1200, 1200), Image.Resampling.LANCZOS).save(output_dir / "texture_overview.png")
    scene = output_dir / "landmarks.usda"
    ground = usd_source(side_m, 0.0, texture.name)
    scene.write_text(ground.rsplit("\n}", 1)[0] + "\n" + landmark_usd(layout) + "\n}\n")
    manifest = {
        "schema_version": 1,
        "entry": scene.name,
        "profile": "landmarks",
        "seed": seed,
        "meters_per_unit": 1.0,
        "up_axis": "Z",
        "ground_z_m": 0.0,
        "bounds_xy_m": [[-side_m / 2] * 2, [side_m / 2] * 2],
        "max_surface_height_agl_m": max(p["top_z_m"] for p in layout),
        "geometry": "flat collision mesh and static colored geometric landmarks",
        "landmarks": layout,
        "layout": {
            "central_count": central_count,
            "central_xy_bounds_m": [-36, 36],
            "outer_count": outer_count,
            "minimum_edge_gap_m": 1.0,
            "home_specific_marker": False,
            "path_conditioned": False,
        },
        "texture": {
            "resolution_wh": [resolution] * 2,
            "physical_extent_wh_m": [side_m] * 2,
            "meters_per_texel": side_m / resolution,
            "color_space": "sRGB",
            "wrapping": "clamp",
            "uv_range": [[0, 0], [1, 1]],
            "provenance": "procedural terrain; not real-world imagery",
        },
        "asset_sha256": {scene.name: digest(scene), texture.name: digest(texture)},
        "generator_sha256": digest(Path(__file__)),
        "terrain_generator_sha256": digest(PROJECT / "generate_downward_environment.py"),
    }
    scene.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Created landmark scene: {} ({} objects)".format(scene, len(layout)), flush=True)
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "maps/downward_landmarks_v1")
    parser.add_argument("--side-m", type=float, default=400.0)
    parser.add_argument("--resolution", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--central-count", type=int, default=96)
    parser.add_argument("--outer-count", type=int, default=96)
    args = parser.parse_args()
    generate_scene(
        args.output_dir,
        args.side_m,
        args.resolution,
        args.seed,
        args.central_count,
        args.outer_count,
    )


if __name__ == "__main__":
    main()
