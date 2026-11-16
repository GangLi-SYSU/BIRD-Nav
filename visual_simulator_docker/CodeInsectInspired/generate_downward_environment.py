"""Generate an independent, meter-scale flat USD bundle; never overwrite a bundle."""

import argparse
import json
from pathlib import Path

from insect_utils.experiment_config import PROJECT, digest


PROFILES = ("distinct", "repeated", "low_texture")


def make_texture(profile, side_m, resolution, seed):
    import numpy as np
    from PIL import Image, ImageDraw

    rng = np.random.RandomState(seed)
    if profile == "low_texture":
        return Image.new("RGB", (resolution, resolution), (112, 120, 100))
    if profile == "repeated":
        # Exactly 2 m periodic in both axes; intentionally ambiguous diagnostic.
        period = max(2, round(2.0 * resolution / side_m))
        xy = np.indices((resolution, resolution))
        light = (xy[0] % period < period // 2) ^ (xy[1] % period < period // 2)
        colors = np.array([[84, 99, 74], [156, 144, 112]], dtype=np.uint8)
        return Image.fromarray(colors[light.astype(np.uint8)])
    if profile != "distinct":
        raise ValueError("Unknown texture profile")
    # Physical correlation lengths; one full-scene texture, no tiled home beacon.
    scales = [(16.0, 0.24), (4.0, 0.24), (1.0, 0.22), (0.25, 0.18), (0.0625, 0.12)]
    field = np.zeros((resolution, resolution), dtype=np.float32)
    for spacing, weight in scales:
        size = max(2, round(side_m / spacing) + 1)
        noise = Image.fromarray(rng.randint(0, 256, (size, size), dtype=np.uint8))
        smooth = noise.resize((resolution, resolution), Image.Resampling.BICUBIC)
        field += weight * (np.asarray(smooth, dtype=np.float32) / 255)
    soil = np.clip((field - 0.25) / 0.5, 0, 1)
    dark = np.array([42, 62, 36], dtype=np.float32)
    pale = np.array([194, 172, 130], dtype=np.float32)
    rgb = dark + soil[:, :, None] * (pale - dark)
    image = Image.fromarray(rgb.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    # Uniformly distributed fine cracks/stone strokes, independent of home/path.
    for _ in range(round(side_m ** 2 / 1.5)):
        x, y = rng.uniform(0, resolution, 2)
        length = rng.uniform(0.08, 0.7) * resolution / side_m
        angle = rng.uniform(0, 2 * np.pi)
        color = tuple(int(v) for v in rng.uniform([55, 60, 43], [155, 150, 120]))
        draw.line(
            (x, y, x + length * np.cos(angle), y + length * np.sin(angle)),
            fill=color,
            width=max(1, round(0.025 * resolution / side_m)),
        )
    return image


def usd_source(side_m, ground_z, texture_name):
    half = side_m / 2
    # Counterclockwise face viewed from +Z; V=0 corresponds to world -Y.
    points = [
        (-half, -half, ground_z),
        (half, -half, ground_z),
        (half, half, ground_z),
        (-half, half, ground_z),
    ]
    return """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World" {
    def Mesh "Ground" (prepend apiSchemas = ["MaterialBindingAPI", "PhysicsCollisionAPI"]) {
        point3f[] points = POINTS
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        normal3f[] normals = [(0, 0, 1)] (interpolation = "constant")
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
        uniform token subdivisionScheme = "none"
        rel material:binding = </World/GroundMaterial>
        bool physics:collisionEnabled = true
    }
    def Material "GroundMaterial" {
        token outputs:surface.connect = </World/GroundMaterial/Surface.outputs:surface>
        def Shader "Surface" {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor.connect = </World/GroundMaterial/Texture.outputs:rgb>
            float inputs:roughness = 0.85
            float inputs:metallic = 0
            token outputs:surface
        }
        def Shader "UV" {
            uniform token info:id = "UsdPrimvarReader_float2"
            token inputs:varname = "st"
            float2 outputs:result
        }
        def Shader "Texture" {
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @TEXTURE@
            token inputs:sourceColorSpace = "sRGB"
            token inputs:wrapS = "clamp"
            token inputs:wrapT = "clamp"
            float2 inputs:st.connect = </World/GroundMaterial/UV.outputs:result>
            float3 outputs:rgb
        }
    }
    def DomeLight "Ambient" {
        float inputs:intensity = 600
        color3f inputs:color = (1, 1, 1)
    }
}
""".replace(
        "POINTS", str(points)
    ).replace(
        "TEXTURE", texture_name
    )


def generate_bundle(output_dir, side_m=100.0, resolution=4096, seed=10, ground_z=0.0):
    import math
    from PIL import Image

    if not math.isfinite(side_m) or side_m < 20 or not math.isfinite(ground_z):
        raise ValueError("Use a finite side >=20 m and a finite ground height")
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 128:
        raise ValueError("Texture resolution must be an integer >=128")
    if not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    for profile in PROFILES:
        texture = output_dir / (profile + ".png")
        image = make_texture(profile, side_m, resolution, seed)
        image.save(texture)
        # A texture-space preview, explicitly not a camera render or calibration.
        image.resize((1000, 1000), Image.Resampling.LANCZOS).save(
            output_dir / (profile + "_texture_overview.png")
        )
        scene = output_dir / (profile + ".usda")
        scene.write_text(usd_source(side_m, ground_z, texture.name))
        manifest = {
            "schema_version": 1,
            "entry": scene.name,
            "profile": profile,
            "seed": seed,
            "meters_per_unit": 1.0,
            "up_axis": "Z",
            "bounds_xy_m": [[-side_m / 2] * 2, [side_m / 2] * 2],
            "ground_z_m": ground_z,
            "max_surface_height_agl_m": 0.0,
            "geometry": "one flat collision mesh; no elevated obstacles or roofs",
            "texture": {
                "resolution_wh": [resolution, resolution],
                "physical_extent_wh_m": [side_m, side_m],
                "meters_per_texel": side_m / resolution,
                "uv_range": [[0, 0], [1, 1]],
                "uv_world_axes": "U:+X,V:+Y; PNG top row is +Y",
                "wrapping": "clamp",
                "color_space": "sRGB",
                "correlation_lengths_m": [16, 4, 1, 0.25, 0.0625] if profile == "distinct" else [],
                "realized_period_m": round(2 * resolution / side_m) * side_m / resolution
                if profile == "repeated"
                else None,
                "provenance": "procedural engineering terrain; not measured real-world imagery",
                "home_specific_marker": False,
            },
            "asset_sha256": {scene.name: digest(scene), texture.name: digest(texture)},
            "generator_sha256": digest(Path(__file__)),
        }
        scene.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Created independent scene bundle: " + str(output_dir), flush=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "maps/downward_flat_v1")
    parser.add_argument("--side-m", type=float, default=100.0)
    parser.add_argument("--resolution", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--ground-z", type=float, default=0.0)
    args = parser.parse_args()
    generate_bundle(args.output_dir, args.side_m, args.resolution, args.seed, args.ground_z)


if __name__ == "__main__":
    main()
