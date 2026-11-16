"""Regressions for spiral-only data and the independently defined2m arrival rule."""

import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insect_utils.camera_geometry import home_vector_body, optics, world_to_pixel
from insect_utils.coverage_capture import planned_samples
from insect_utils.experiment_config import PROJECT
from insect_utils.experiment_state import read_json
from insect_utils.navigation_core import make_episodes, simulate_oracle, termination
from insect_utils.spiral_narrow_protocol import PROTOCOL, audit_rows, build_plan


class SpiralNarrowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = read_json(PROJECT / PROTOCOL)
        cls.plan = build_plan(cls.protocol, permutation=list(reversed(range(147))))
        cls.frozen = dict(plan=cls.plan, protocol=cls.protocol)
        cls.rows = []
        for sample in planned_samples(cls.plan):
            x, y, z = sample["planned_position_m"]
            tx, ty = sample["planned_target_body_m"]
            cls.rows.append(
                dict(
                    sample,
                    x=x,
                    y=y,
                    z=z,
                    yaw=sample["planned_body_yaw_deg"],
                    target_x=tx,
                    target_y=ty,
                    ground_z_m=0.0,
                    agl_m=40.0,
                    image="fixtures/" + sample["sample_id"] + ".png",
                )
            )

    def test_original_spiral_has_no_added_near_home_or_coverage_positions(self):
        audit = audit_rows(self.rows, self.frozen)
        self.assertEqual(audit["positions"], 147)
        self.assertEqual(audit["images_by_split"], dict(train=1064, validation=48, test=64))
        self.assertAlmostEqual(audit["maximum_radius_m"], 43.982297150257104)
        self.assertEqual(audit["off_trajectory_positions"], 0)

    def test_reject_off_spiral_position_even_with_consistent_target(self):
        records = copy.deepcopy(self.rows)
        group = records[0]["position_id"]
        for row in records:
            if row["position_id"] == group:
                row["x"] += 0.25
                row["target_x"], row["target_y"] = home_vector_body(
                    [row["x"], row["y"], row["z"]], [0, 0, 40], row["yaw"]
                )
        with self.assertRaisesRegex(ValueError, "Off-trajectory"):
            audit_rows(records, self.frozen)

    def test_reject_split_reassignment(self):
        records = copy.deepcopy(self.rows)
        group = records[0]["position_id"]
        for row in records:
            if row["position_id"] == group:
                row["split"] = "validation" if row["split"] != "validation" else "train"
        with self.assertRaisesRegex(ValueError, "Position/split"):
            audit_rows(records, self.frozen)

    def test_missing_view_cannot_be_replaced_by_duplicate(self):
        records = self.rows[:-1] + [self.rows[0]]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            audit_rows(records, self.frozen)

    def test_success_boundary_is_two_meters_not_camera_visibility(self):
        controller = self.protocol["navigation"]["controller"]
        self.assertEqual(termination([2, 0, 40], [0, 0, 40], 1, controller), "success")
        self.assertIsNone(termination([2.001, 0, 40], [0, 0, 40], 1, controller))
        self.assertIsNone(termination([10, 0, 40], [0, 0, 40], 0, controller))
        pixel = world_to_pixel([0, 0, 0], [10, 0, 40], 0, optics(self.protocol["camera"]))
        self.assertTrue(all(0 <= value < 1024 for value in pixel))

    def test_requested_starting_poses_are_nontrivial_and_oracle_can_return(self):
        nav = self.protocol["navigation"]
        episodes = make_episodes(nav["formal"], nav["home_camera_xyz_m"], "formal")
        self.assertEqual(len(episodes), 128)
        self.assertEqual({e["radius_m"] for e in episodes}, {10, 20})
        for episode in episodes:
            result = simulate_oracle(episode, nav["home_camera_xyz_m"], nav["controller"])
            self.assertEqual(result["reason"], "success")
            self.assertGreater(result["steps"], 0)
            self.assertLessEqual(result["final_distance_m"], 2)

    def test_narrow_optics_and_training_budget(self):
        settings = optics(self.protocol["camera"])
        expected = 1024 / (2 * math.tan(math.radians(15)))
        self.assertAlmostEqual(settings["intrinsics"][0][0], expected)
        training = self.protocol["training"]
        self.assertEqual(training["epochs"], 600)
        self.assertEqual(training["epochs"] * math.ceil(1064 / training["batch_size"]), 79800)


if __name__ == "__main__":
    unittest.main()
