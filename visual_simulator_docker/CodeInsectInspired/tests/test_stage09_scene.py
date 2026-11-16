"""Regressions for metric asset conversion and the wider nadir preview camera."""

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from generate_real_world_environment import converted_bounds, mdl_texture_dependencies
from insect_utils.camera_geometry import optics, world_to_pixel


class RealSceneUnitsTests(unittest.TestCase):
    def test_y_up_house_retains_real_size_and_height(self):
        low, high = converted_bounds([-7.5, 0, -5], [7.5, 7.8, 5], 1, "Y")
        self.assertEqual(low, [-7.5, -5, 0])
        self.assertEqual(high, [7.5, 5, 7.8])

    def test_centimeter_tree_becomes_meter_tree(self):
        low, high = converted_bounds([-150, -200, -1], [150, 200, 900], 0.01, "Z")
        self.assertEqual(low, [-1.5, -2, -0.01])
        self.assertEqual(high, [1.5, 2, 9])

    def test_empty_asset_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            converted_bounds([1, 1, 1], [-1, -1, -1], 0.01, "Z")

    def test_mdl_texture_defaults_are_required_and_resolved_from_material_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            material = root / "leaf.mdl"
            material.write_text('diffuse: texture_2d("./leaf.png", ::tex::gamma_srgb)')
            with self.assertRaises(FileNotFoundError):
                mdl_texture_dependencies([material])
            texture = root / "leaf.png"
            texture.write_bytes(b"texture")
            self.assertEqual(mdl_texture_dependencies([material]), {texture.resolve()})

    def test_preview_footprint_and_image_axes(self):
        config = json.loads((PROJECT / "config/experiments/stage09_preview.json").read_text())
        settings = optics(config["camera"])
        half_side = config["camera_height_agl_m"] * math.tan(math.radians(35))
        self.assertAlmostEqual(half_side * 2, 56.0166030568)
        self.assertEqual(settings["resolution_wh"], [1024, 1024])
        self.assertEqual(settings["vfov_deg"], 70)
        # Body forward is image up, world -Y is image right at zero yaw.
        up = world_to_pixel([half_side, 0, 0], [0, 0, 40], 0, settings)
        right = world_to_pixel([0, -half_side, 0], [0, 0, 40], 0, settings)
        for actual, expected in zip(up + right, [512, 0, 1024, 512]):
            self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
