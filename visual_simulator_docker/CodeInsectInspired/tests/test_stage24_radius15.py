"""Check that the additional radius does not change the selected model or controller."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insect_utils.experiment_config import PROJECT
from insect_utils.experiment_state import read_json
from insect_utils.navigation_core import make_episodes, simulate_oracle
from insect_utils.radius15_extension import extend_protocol


class Radius15Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = read_json(
            PROJECT / "runs/stage23_compact_spiral_h40_fov30_01/navigation/frozen.json"
        )

    def test_only_starting_radius_changes(self):
        source = copy.deepcopy(self.source)
        before = copy.deepcopy(source["protocol"])
        protocol = extend_protocol(source, [15])
        self.assertEqual(source["protocol"], before)
        for phase in ("formal", "rendered_oracle"):
            self.assertEqual(protocol[phase]["radii_m"], [15])
            before[phase]["radii_m"] = [15]
        self.assertEqual(protocol, before)
        protocol["controller"]["max_steps"] = 1
        self.assertEqual(source["protocol"]["controller"]["max_steps"], 50)

    def test_all64_requested_starting_poses_are_nontrivial(self):
        protocol = extend_protocol(self.source, [15])
        episodes = make_episodes(protocol["formal"], protocol["home_camera_xyz_m"], "formal")
        self.assertEqual(len(episodes), 64)
        self.assertEqual(len({e["episode_id"] for e in episodes}), 64)
        self.assertEqual({e["radius_m"] for e in episodes}, {15})
        for episode in episodes:
            result = simulate_oracle(episode, protocol["home_camera_xyz_m"], protocol["controller"])
            self.assertEqual(result["reason"], "success")
            self.assertGreater(result["steps"], 0)
            self.assertLessEqual(result["final_distance_m"], 2)

    def test_reject_other_radii_or_wrong_checkpoint(self):
        with self.assertRaises(ValueError):
            extend_protocol(self.source, [10, 20])
        source = copy.deepcopy(self.source)
        source["model"]["epoch"] = 600
        with self.assertRaisesRegex(ValueError, "epoch484"):
            extend_protocol(source, [15])

    def test_success_radius_stays_at_two_meters(self):
        source = copy.deepcopy(self.source)
        source["protocol"]["controller"]["success_radius_m"] = 0.5
        with self.assertRaisesRegex(ValueError, "2m"):
            extend_protocol(source, [15])


if __name__ == "__main__":
    unittest.main()
