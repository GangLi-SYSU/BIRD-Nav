"""Settling behavior must reject spikes, stale frames and interrupted sequences."""

import json
from pathlib import Path
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.frame_stability import FrameStability, STABILITY_POLICY


class FrameStabilityTests(unittest.TestCase):
    def test_baseline_is_followed_by_two_small_changes(self):
        gate = FrameStability()
        self.assertFalse(gate.observe(10, None))
        self.assertFalse(gate.observe(11, 0.4))
        self.assertTrue(gate.observe(12, 0.3))
        self.assertEqual(gate.summary()["last_rgb_mae"], 0.3)

    def test_threshold_is_inclusive(self):
        gate = FrameStability()
        gate.observe(1, None)
        self.assertFalse(gate.observe(2, 0.5))
        self.assertTrue(gate.observe(3, 0.5))

    def test_spike_resets_the_consecutive_count(self):
        gate = FrameStability()
        for frame, value in ((1, None), (2, 0.2), (3, 2.1), (4, 0.1)):
            self.assertFalse(gate.observe(frame, value))
        self.assertTrue(gate.observe(5, 0.2))

    def test_same_frame_cannot_count_twice(self):
        gate = FrameStability()
        gate.observe(1, None)
        gate.observe(2, 0.1)
        for _ in range(4):
            self.assertFalse(gate.observe(2, 0))
        self.assertEqual(gate.consecutive, 1)
        self.assertEqual(len(gate.history), 2)
        self.assertTrue(gate.observe(3, 0.1))

    def test_older_frame_is_ignored(self):
        gate = FrameStability()
        gate.observe(10, None)
        gate.observe(12, 0.1)
        self.assertFalse(gate.observe(11, 0))
        self.assertEqual(gate.last_frame, 12)
        self.assertTrue(gate.observe(13, 0.1))

    def test_invalid_pose_or_image_reset_requires_a_new_baseline(self):
        gate = FrameStability()
        gate.observe(1, None)
        gate.observe(2, 0.1)
        gate.reset()
        self.assertFalse(gate.observe(3, 0.1))
        self.assertFalse(gate.observe(4, 0.1))
        self.assertTrue(gate.observe(5, 0.1))
        self.assertIsNone(gate.history[-3]["rgb_mae"])

    def test_nonfinite_or_negative_change_is_rejected(self):
        for value in (float("nan"), float("inf"), -0.1, True):
            gate = FrameStability()
            gate.observe(1, None)
            with self.assertRaises(ValueError):
                gate.observe(2, value)

    def test_invalid_frame_ids_are_rejected(self):
        for frame in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                FrameStability().observe(frame, None)

    def test_new_capture_does_not_reuse_previous_acceptance(self):
        old = FrameStability()
        old.observe(1, None)
        old.observe(2, 0.1)
        self.assertTrue(old.observe(3, 0.1))
        new = FrameStability()
        self.assertFalse(new.observe(4, 0.1))
        self.assertFalse(new.observe(5, 0.1))
        self.assertTrue(new.observe(6, 0.1))

    def test_applied_policy_matches_user_confirmed_protocol(self):
        protocol = json.loads(
            (PROJECT / "config/experiments/stage04_camera_protocol.json").read_text()
        )
        self.assertEqual(protocol["image_acquisition"], STABILITY_POLICY)


if __name__ == "__main__":
    unittest.main()
