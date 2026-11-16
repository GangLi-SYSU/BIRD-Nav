"""Landmark packing, flight clearance and preview scale regressions without Kit."""

import json
import math
from pathlib import Path
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from generate_landmark_environment import landmark_layout
from insect_utils.scene_geometry import check_positions_and_segments, spatial_budget
from tools.view_stage03_scene import preview_layout


class LandmarkSceneTests(unittest.TestCase):
    def setUp(self):
        self.layout = landmark_layout()
        self.spec = json.loads(
            (PROJECT / "config/experiments/downward_h10_hfov30.json").read_text()
        )
        self.manifest = {
            "bounds_xy_m": [[-200, -200], [200, 200]],
            "ground_z_m": 0,
            "max_surface_height_agl_m": max(p["top_z_m"] for p in self.layout),
        }

    def test_all_four_shapes_and_both_placement_zones_are_present(self):
        for shape in ("Cube", "Cylinder", "Cone", "Sphere"):
            self.assertEqual(sum(p["shape"] == shape for p in self.layout), 48)
        for zone in ("central", "outer"):
            self.assertEqual(sum(p["zone"] == zone for p in self.layout), 96)

    def test_same_seed_reproduces_layout_and_other_seed_changes_it(self):
        self.assertEqual(self.layout, landmark_layout())
        self.assertNotEqual(self.layout, landmark_layout(seed=11))

    def test_landmarks_are_separated_even_for_rotated_cubes(self):
        for index, first in enumerate(self.layout):
            for second in self.layout[index + 1 :]:
                distance = math.hypot(
                    first["center_xyz_m"][0] - second["center_xyz_m"][0],
                    first["center_xyz_m"][1] - second["center_xyz_m"][1],
                )
                self.assertGreaterEqual(
                    distance + 2e-6,
                    first["bounding_radius_xy_m"] + second["bounding_radius_xy_m"] + 1,
                )

    def test_400m_ground_covers_full_current_action_budget(self):
        budget = spatial_budget(self.spec, self.manifest)
        self.assertFalse(budget["boundary_guard_required"])
        self.assertGreater(budget["allowed_camera_center_xy_m"][1][0], 191)
        self.assertLess(budget["required_side_m_without_boundary_guard"], 400)

    def test_every_object_is_below_3m_and_clears_10m_flight_plane(self):
        self.assertLessEqual(self.manifest["max_surface_height_agl_m"], 3)
        budget = spatial_budget(self.spec, self.manifest)
        self.assertGreaterEqual(budget["minimum_surface_clearance_m"], 7)
        self.assertTrue(check_positions_and_segments([[0, 0, 10], [150, 0, 10]], budget)["valid"])

    def test_overview_scales_with_scene_bounds_but_local_camera_stays_at_10m(self):
        layout = preview_layout(self.manifest)
        self.assertEqual(layout["side_m"], 400)
        self.assertEqual(layout["StageCOverview"][0], (0, -260, 560))
        self.assertEqual(layout["StageCAt10m"][0], (0, 0, 10))

    def test_invalid_counts_and_small_field_are_rejected(self):
        for kwargs in ({"central_count": -1}, {"outer_count": True}, {"side_m": 100}):
            with self.assertRaises(ValueError):
                landmark_layout(**kwargs)


if __name__ == "__main__":
    unittest.main()
