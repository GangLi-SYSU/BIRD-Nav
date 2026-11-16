"""Verify20m compression preserves counts,headings,splits and all other settings."""

import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insect_utils.compact_spiral import PROTOCOL, REFERENCE_PROTOCOL, build_compact_plan
from insect_utils.experiment_config import PROJECT
from insect_utils.experiment_state import read_json
from insect_utils.spiral_narrow_protocol import build_plan


class CompactSpiralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = read_json(PROJECT / PROTOCOL)
        cls.reference = read_json(PROJECT / REFERENCE_PROTOCOL)
        cls.permutation = list(reversed(range(147)))
        cls.plan = build_compact_plan(cls.protocol, cls.permutation)
        cls.original = build_plan(cls.reference, cls.permutation)

    def test_same147_positions_and1176_images_with_outermost_position_at20m(self):
        self.assertEqual(self.plan["audit"]["positions"], 147)
        self.assertEqual(self.plan["audit"]["images"], 1176)
        self.assertAlmostEqual(self.plan["audit"]["maximum_radius_m"], 20)
        self.assertTrue(
            all(math.hypot(*p["xyz_m"][:2]) <= 20 + 1e-9 for p in self.plan["positions"])
        )
        self.assertEqual(self.plan["audit"]["groups"], dict(train=133, validation=6, test=8))

    def test_split_membership_and_eight_headings_are_preserved(self):
        self.assertEqual(self.plan["group_split"], self.original["group_split"])
        for new, old in zip(self.plan["positions"], self.original["positions"]):
            self.assertEqual(new["position_id"], old["position_id"])
            self.assertEqual(new["yaws_deg"], old["yaws_deg"])
            self.assertEqual(len(new["yaws_deg"]), 8)
            self.assertEqual(new["xyz_m"][2], 40)

    def test_more_positions_are_available_in_target10_to20m_annulus(self):
        def count(plan):
            return sum(10 < math.hypot(*p["xyz_m"][:2]) <= 20 + 1e-9 for p in plan["positions"])

        self.assertEqual(count(self.original), 34)
        self.assertEqual(count(self.plan), 75)
        self.assertAlmostEqual(self.plan["compression"]["nominal_radial_pitch_m"], 4)
        self.assertAlmostEqual(self.plan["compression"]["nominal_turns_between10_and20m"], 2.5)

    def test_unchanged_training_and_navigation_are_enforced(self):
        for section, key, value in (
            ("training", "learning_rate", 0.001),
            ("camera", "hfov_deg", 70),
        ):
            changed = copy.deepcopy(self.protocol)
            changed[section][key] = value
            with self.assertRaisesRegex(ValueError, "Only the sampling range"):
                build_compact_plan(changed, self.permutation)
        self.assertEqual(self.protocol["training"]["scale_m"], 44)
        self.assertEqual(self.protocol["navigation"]["controller"]["success_radius_m"], 2)

    def test_reject_spiral_scale_that_exceeds20m(self):
        changed = copy.deepcopy(self.protocol)
        changed["learning"]["shape_params"][2] = 0.7
        with self.assertRaisesRegex(ValueError, "requested20m"):
            build_compact_plan(changed, self.permutation)


if __name__ == "__main__":
    unittest.main()
