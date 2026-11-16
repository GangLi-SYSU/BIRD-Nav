"""Requested starting radii, immutable model/controller and oracle convergence."""

import copy
from pathlib import Path
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.experiment_state import read_json
from insect_utils.navigation_core import make_episodes, simulate_oracle
from insect_utils.radius_extension import build_plan, extended_protocol


class RadiusNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = read_json(PROJECT / "runs/stage16_rotation_600_01/navigation/frozen.json")

    def test_only_requested_radii_change_the_formal_protocol(self):
        source = copy.deepcopy(self.source)
        protocol = extended_protocol(source, [10, 5])
        expected = copy.deepcopy(source["protocol"])
        expected["formal"]["radii_m"] = [10, 5]
        expected["rendered_oracle"]["radii_m"] = [10, 5]
        self.assertEqual(protocol, expected)
        self.assertEqual(source, self.source)
        for radii in ([5], [10], [5, 10, 20], [20]):
            with self.assertRaises(ValueError):
                extended_protocol(source, radii)

    def test128_unique_initial_conditions_and8_engineering_returns(self):
        protocol = extended_protocol(self.source, [10, 5])
        episodes = make_episodes(protocol["formal"], protocol["home_camera_xyz_m"], "formal")
        self.assertEqual(len(episodes), 128)
        self.assertEqual(len({e["episode_id"] for e in episodes}), 128)
        for radius in (10, 5):
            selected = [e for e in episodes if e["radius_m"] == radius]
            self.assertEqual(len(selected), 64)
            self.assertEqual({e["initial_yaw_deg"] for e in selected}, {0, 90, 180, 270})
            self.assertEqual(len({e["bearing_deg"] for e in selected}), 16)
        engineering = make_episodes(
            protocol["rendered_oracle"], protocol["home_camera_xyz_m"], "oracle"
        )
        self.assertEqual(len(engineering), 8)
        self.assertEqual({e["radius_m"] for e in engineering}, {5, 10})

    def test_oracle_reaches_success_circle_at_both_radii_and_translated_home(self):
        protocol = extended_protocol(self.source, [10, 5])
        for home in ([0, 0, 40], [13, -8, 40]):
            for episode in make_episodes(protocol["formal"], home, "oracle"):
                result = simulate_oracle(
                    episode, home, protocol["controller"], self.source["budget"]
                )
                self.assertEqual(result["reason"], "success")
                self.assertLessEqual(result["final_distance_m"], 0.5)
                self.assertLessEqual(result["steps"], 50)

    def test_frozen_plan_reuses_exact_weights_controller_and_historical_reference(self):
        plan = build_plan()
        self.assertEqual(plan["model"], self.source["model"])
        self.assertEqual(plan["protocol"]["controller"], self.source["protocol"]["controller"])
        self.assertEqual(plan["budget"], self.source["budget"])
        self.assertEqual(len(plan["episodes"]["formal"]), 128)
        for name, sha in self.source["files"].items():
            self.assertEqual(plan["files"][name], sha)

    def test_another_selected_epoch_or_output_scale_is_rejected(self):
        for key, value in (("epoch", 600), ("scale_m", 1)):
            changed = copy.deepcopy(self.source)
            changed["model"][key] = value
            with self.assertRaises(ValueError):
                extended_protocol(changed, [10, 5])


if __name__ == "__main__":
    unittest.main()
