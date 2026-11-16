"""Build a separate, denser landmark scene with large, asymmetric color patterns."""

import argparse
import colorsys
import json
import math
from pathlib import Path
import random
import shutil

from generate_downward_environment import usd_source
from insect_utils.experiment_config import PROJECT, digest


SHAPES = ("Cube", "Cylinder", "Cone", "Sphere", "Triangle", "Hexagon", "Trapezoid", "Arrow")
PALETTE = [colorsys.hsv_to_rgb(i / 24, 0.85, 0.92) for i in range(24)]
PALETTE += [(0.96, 0.96, 0.96), (0.035, 0.035, 0.035)]


def layout(seed=10):
    """Fixed world grid, independent of home, learning path, headings and labels."""
    rng = random.Random(seed)
    centers = []
    spacing = 5.2
    dy = spacing * math.sqrt(3) / 2
    for j in range(-8, 9):
        for i in range(-7, 8):
            x = i * spacing + (j % 2) * spacing / 2 + rng.uniform(-0.12, 0.12)
            y = j * dy + rng.uniform(-0.12, 0.12)
            if max(abs(x), abs(y)) <= 36:
                centers.append((x, y, "central"))
    # Keep peripheral landmarks, while leaving their placement independent of the path.
    for _ in range(96):
        for attempt in range(50000):
            x, y = rng.uniform(-185, 185), rng.uniform(-185, 185)
            if max(abs(x), abs(y)) < 45:
                continue
            if all(math.hypot(x - a, y - b) >= 5.2 for a, b, _ in centers):
                centers.append((x, y, "outer"))
                break
        else:
            raise ValueError("Cannot place outer landmarks")
    shapes = [SHAPES[i % len(SHAPES)] for i in range(len(centers))]
    rng.shuffle(shapes)
    result = []
    for index, ((x, y, zone), shape) in enumerate(zip(centers, shapes)):
        height = rng.uniform(1.6, 2.9)
        radius = rng.uniform(1.4, 1.75)
        if shape == "Sphere":
            radius = rng.uniform(1.3, 1.49)
            height = 2 * radius
        item = {
            "id": "Landmark_{:03d}".format(index),
            "shape": shape,
            "usd_type": "Mesh",
            "zone": zone,
            "center_xyz_m": [x, y, height / 2],
            "height_m": height,
            "radius_m": radius,
            "yaw_deg": rng.uniform(0, 360),
            "material_index": rng.randrange(24),
            "pattern_phase": rng.randrange(24),
            "pattern_width": rng.choice([1, 2, 3]),
            "top_z_m": height,
        }
        item["color_rgb"] = PALETTE[item["material_index"]]
        luminance = sum(a * b for a, b in zip(item["color_rgb"], [0.2126, 0.7152, 0.0722]))
        item["accent_material_index"] = 25 if luminance > 0.45 else 24
        vertices, faces, accents = mesh(item)
        item["bounding_radius_xy_m"] = max(math.hypot(p[0], p[1]) for p in vertices)
        item["size_xyz_m"] = [
            max(p[k] for p in vertices) - min(p[k] for p in vertices) for k in range(3)
        ]
        item["accent_face_count"] = len(accents)
        result.append(item)
    for i, first in enumerate(result):
        for second in result[i + 1 :]:
            distance = math.dist(first["center_xyz_m"][:2], second["center_xyz_m"][:2])
            if distance < first["bounding_radius_xy_m"] + second["bounding_radius_xy_m"] + 0.25:
                raise ValueError("Landmarks overlap or violate the 0.25 m edge gap")
    return result


def mesh(item):
    shape, radius, height = item["shape"], item["radius_m"], item["height_m"]
    phase, width = item["pattern_phase"], item["pattern_width"]
    if shape == "Sphere":
        points = [(0, 0, -radius), (0, 0, radius)]
        n, bands = 24, 12
        for j in range(1, bands):
            latitude = -math.pi / 2 + j * math.pi / bands
            points.extend(
                (
                    radius * math.cos(latitude) * math.cos(i * 2 * math.pi / n),
                    radius * math.cos(latitude) * math.sin(i * 2 * math.pi / n),
                    radius * math.sin(latitude),
                )
                for i in range(n)
            )
        faces, accents = [], []
        for i in range(n):
            nxt = (i + 1) % n
            faces.append([0, 2 + nxt, 2 + i])
            for j in range(bands - 2):
                a, b = 2 + j * n + i, 2 + j * n + nxt
                faces.extend([[a, b, b + n], [a, b + n, a + n]])
            faces.append([1, 2 + (bands - 2) * n + i, 2 + (bands - 2) * n + nxt])
            # A noncentral wedge breaks the solid sphere's rotational ambiguity.
            if (i - phase) % n < 3 + width:
                accents.extend(range(len(faces) - 2 * (bands - 2) - 2, len(faces)))
        return points, faces, accents
    if shape == "Cube":
        polygon = [(-1.45, -1.45), (1.45, -1.45), (1.45, 1.45), (-1.45, 1.45)]
    elif shape == "Trapezoid":
        polygon = [(-1.5, -1.35), (1.5, -1.35), (0.85, 1.35), (-1.0, 1.35)]
    elif shape == "Arrow":
        polygon = [(-1.5, -1.15), (0.45, -1.15), (1.75, 0), (0.45, 1.15), (-1.5, 1.15)]
    else:
        n = {"Triangle": 3, "Hexagon": 6}.get(shape, 24)
        polygon = [
            (radius * math.cos(i * 2 * math.pi / n), radius * math.sin(i * 2 * math.pi / n))
            for i in range(n)
        ]
    n = len(polygon)
    points = [(x, y, -height / 2) for x, y in polygon]
    faces, accents = [], []
    if shape == "Cone":
        points.extend([(0, 0, height / 2), (0, 0, -height / 2)])
        for i in range(n):
            faces.extend([[i, (i + 1) % n, n], [n + 1, (i + 1) % n, i]])
            if (i - phase) % n < 3 + width:
                accents.append(len(faces) - 2)
    else:
        points += [(x, y, height / 2) for x, y in polygon]
        points.extend([(0, 0, height / 2), (0, 0, -height / 2)])
        for i in range(n):
            nxt = (i + 1) % n
            faces.extend([[i, nxt, n + nxt, n + i], [2 * n, n + i, n + nxt], [2 * n + 1, nxt, i]])
            if (i - phase) % n < max(1, n // 4):
                accents.append(len(faces) - 2)
    return points, faces, accents


def usd_landmarks(items):
    lines = ['    def Scope "LandmarkMaterials" {']
    for i, color in enumerate(PALETTE):
        lines.append(
            """        def Material "Color_%d" {
            token outputs:surface.connect = SURFACE_PATH
            def Shader "Surface" {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = %s
                float inputs:roughness = 0.7
                token outputs:surface
            }
        }"""
            % (i, str(tuple(color)))
        )
        lines[-1] = lines[-1].replace(
            "SURFACE_PATH", "</World/LandmarkMaterials/Color_%d/Surface.outputs:surface>" % i
        )
    lines.append('    }\n    def Scope "Landmarks" {')
    for item in items:
        points, faces, accents = mesh(item)
        low = tuple(min(p[k] for p in points) for k in range(3))
        high = tuple(max(p[k] for p in points) for k in range(3))
        lines.append(
            """        def Mesh "%s" (
            prepend apiSchemas = ["MaterialBindingAPI", "PhysicsCollisionAPI"]
        ) {
            point3f[] points = %s
            int[] faceVertexCounts = %s
            int[] faceVertexIndices = %s
            float3[] extent = [%s, %s]
            uniform token subdivisionScheme = "none"
            bool doubleSided = true
            bool physics:collisionEnabled = true
            double3 xformOp:translate = %s
            float xformOp:rotateZ = %s
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ"]
            rel material:binding = </World/LandmarkMaterials/Color_%d>
            def GeomSubset "Accent" (prepend apiSchemas = ["MaterialBindingAPI"]) {
                uniform token elementType = "face"
                uniform token familyName = "materialBind"
                int[] indices = %s
                rel material:binding = </World/LandmarkMaterials/Color_%d>
            }
        }"""
            % (
                item["id"],
                str(points),
                str([len(f) for f in faces]),
                str([i for f in faces for i in f]),
                str(low),
                str(high),
                str(tuple(item["center_xyz_m"])),
                str(item["yaw_deg"]),
                item["material_index"],
                str(accents),
                item["accent_material_index"],
            )
        )
    lines.append(
        """    }
    def DistantLight "Sun" {
        float inputs:intensity = 1200
        float inputs:angle = 1
        color3f inputs:color = (1, 0.96, 0.90)
        float3 xformOp:rotateXYZ = (-35, -20, -20)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }"""
    )
    return "\n".join(lines)


def generate(output, seed=10):
    output = Path(output)
    items = layout(seed)
    output.mkdir(parents=True, exist_ok=False)
    old = PROJECT / "maps/downward_landmarks_v1"
    shutil.copyfile(old / "ground.png", output / "ground.png")
    scene = output / "landmarks.usda"
    ground = usd_source(400, 0, "ground.png")
    scene.write_text(ground.rsplit("\n}", 1)[0] + "\n" + usd_landmarks(items) + "\n}\n")
    original = json.loads((old / "landmarks.manifest.json").read_text())
    manifest = {
        "schema_version": 1,
        "entry": scene.name,
        "profile": "identifiable_landmarks",
        "seed": seed,
        "meters_per_unit": 1.0,
        "up_axis": "Z",
        "ground_z_m": 0.0,
        "bounds_xy_m": [[-200, -200], [200, 200]],
        "max_surface_height_agl_m": max(p["top_z_m"] for p in items),
        "landmarks": items,
        "texture": original["texture"],
        "layout": {
            "method": "jittered triangular world lattice plus random peripheral objects",
            "spacing_m": 5.2,
            "jitter_m": 0.12,
            "minimum_edge_gap_m": 0.25,
            "home_specific_marker": False,
            "path_conditioned": False,
        },
        "geometry": "large eight-shape meshes with independent colors and asymmetric accents",
        "asset_sha256": {scene.name: digest(scene), "ground.png": digest(output / "ground.png")},
        "generator_sha256": digest(Path(__file__)),
        "source_ground_sha256": digest(old / "ground.png"),
    }
    scene.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("IDENTIFIABLE_SCENE_GENERATED", str(scene), "landmarks", len(items), flush=True)
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "maps/downward_landmarks_v2")
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
