"""Regression checks: reaching home must not stop a50-action trajectory."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insect_utils.experiment_config import PROJECT
from insect_utils.experiment_state import read_json
from insect_utils.fixed50_metrics import (
    COMPLETED,
    complete_fixed_episode,
    endpoint_metrics,
    fixed_termination,
    summarize_fixed,
)
from insect_utils.fixed50_protocol import fixed_protocol
from run_fixed50_navigation import simulate_fixed_oracle


class Fixed50Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = read_json(PROJECT / "runs/stage24_radius15_01/frozen.json")
        cls.spec = read_json(PROJECT / "config/experiments/stage25_fixed50_protocol.json")
        cls.controller = cls.source["protocol"]["controller"]

    def test_arrival_even_at_exact_home_does_not_terminate_before50(self):
        for position in ([0, 0, 40], [0.1, 0, 40], [2, 0, 40], [100, 0, 40]):
            for step in range(50):
                self.assertIsNone(fixed_termination(position, [0, 0, 40], step, self.controller))
            self.assertEqual(
                fixed_termination(position, [0, 0, 40], 50, self.controller), COMPLETED
            )

    def test_pure_oracle_keeps_moving_after_first2m_entry(self):
        episode = self.source["episodes"]["formal"][0]
        points = simulate_fixed_oracle(episode, [0, 0, 40], self.controller, self.source["budget"])
        metrics = endpoint_metrics(points, [0, 0, 40])
        self.assertEqual(len(points), 51)
        self.assertEqual(metrics["actions"], 50)
        self.assertLess(metrics["first_entry_2m_step"], 50)
        for a, b in zip(points, points[1:]):
            self.assertNotEqual(a[:2], b[:2])

    def test_ever_entering_and_final_arrival_are_distinct(self):
        positions = [[15, 0, 40]] + [[1, 0, 40] for _ in range(49)] + [[3, 0, 40]]
        metrics = endpoint_metrics(positions, [0, 0, 40])
        self.assertEqual(metrics["first_entry_2m_step"], 1)
        self.assertTrue(metrics["ever_entered_2m"])
        self.assertFalse(metrics["final_within_2m"])
        self.assertTrue(metrics["entered_then_outside_2m_at_end"])
        self.assertEqual(metrics["final_at_step50_m"], 3)

    def test_incomplete_trial_cannot_be_committed_as_fixed50(self):
        with self.assertRaisesRegex(ValueError, "all50"):
            complete_fixed_episode(Path("/tmp"), "formal", dict(actions=[{}] * 49), COMPLETED, {})
        partial = endpoint_metrics([[15, 0, 40], [1, 0, 40]], [0, 0, 40])
        self.assertIsNone(partial["final_at_step50_m"])

    def test_only_stopping_rule_changes_not_motion_or_starting_poses(self):
        before = copy.deepcopy(self.source)
        protocol = fixed_protocol(self.source, self.spec)
        self.assertEqual(self.source, before)
        for key in ("controller", "formal", "camera", "runtime", "scene", "home_camera_xyz_m"):
            self.assertEqual(protocol[key], self.source["protocol"][key])
        self.assertFalse(protocol["termination_policy"]["distance_based"])

    def test_invalid_fixed_horizon_requests_are_rejected(self):
        spec = copy.deepcopy(self.spec)
        spec["distance_based_termination"] = True
        with self.assertRaises(ValueError):
            fixed_protocol(self.source, spec)
        with self.assertRaises(ValueError):
            fixed_termination([0, 0, 40], [0, 0, 40], 51, self.controller)

    def test_scoring_excludes_abort_and_uses_the50th_endpoint(self):
        metrics = endpoint_metrics([[15, 0, 40]] + [[1, 0, 40]] * 49 + [[3, 0, 40]], [0, 0, 40])
        complete = dict(
            reason=COMPLETED,
            steps=50,
            final_distance_m=3,
            minimum_endpoint_distance_m=1,
            final_position_m=[3, 0, 40],
            detail=dict(fixed_horizon=metrics),
        )
        aborted = dict(reason="invalid_prediction_zero", steps=17, final_distance_m=0.1)
        summary = summarize_fixed([complete, aborted], [0.5, 1, 2, 3, 5])
        self.assertEqual(summary["completed50"], 1)
        self.assertEqual(summary["aborted"], 1)
        self.assertEqual(summary["ever_entered_2m"], 1)
        self.assertEqual(summary["final_within_2m"], 0)
        self.assertEqual(summary["final_within_radius"]["3"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
