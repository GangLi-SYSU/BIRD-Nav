"""Symmetric geometry-only local samples and explicit position holdout distances."""

from collections import Counter
from pathlib import Path
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.experiment_state import read_json
from insect_utils.near_data import PROTOCOL, build_plan, distance, make_positions, original_centers


class NearDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = read_json(PROJECT / PROTOCOL)
        cls.records = read_json(PROJECT / "runs/stage11_coverage_02/dataset.json")["records"]
        cls.positions = make_positions(cls.spec, cls.records)

    def test_counts_origin_and_antipodal_symmetry_in_each_partition(self):
        self.assertEqual(
            Counter(p["split"] for p in self.positions),
            dict(train=193, validation=16, reserved_test=18),
        )
        self.assertEqual(len({p["position_id"] for p in self.positions}), 227)
        for split in ("train", "validation", "reserved_test"):
            pts = [p["xyz_m"][:2] for p in self.positions if p["split"] == split]
            for x, y in pts:
                self.assertTrue(any(distance([-x, -y], q) < 1e-10 for q in pts))
        homes = [p for p in self.positions if distance(p["xyz_m"], [0, 0]) < 1e-10]
        self.assertEqual(len(homes), 1)
        self.assertEqual(homes[0]["split"], "train")

    def test_local_and_original_holdout_distances_are_satisfied(self):
        trains = [p["xyz_m"] for p in self.positions if p["split"] == "train"]
        held = [p["xyz_m"] for p in self.positions if p["split"] != "train"]
        oldtrain = original_centers(self.records, {"train"})
        oldhold = original_centers(self.records, {"validation", "legacy_test", "new_test"})
        self.assertGreaterEqual(min(distance(a, b) for a in trains for b in oldhold), 1 - 1e-10)
        self.assertGreaterEqual(
            min(distance(a, b) for a in held for b in trains + oldtrain), 0.15 - 1e-10
        )
        self.assertGreaterEqual(
            min(distance(a, b) for i, a in enumerate(held) for b in held[i + 1 :]), 0.15 - 1e-10
        )

    def test_seeded_geometry_is_reproducible_and_does_not_use_predictions(self):
        self.assertEqual(make_positions(self.spec, self.records), self.positions)
        self.assertTrue(
            all(0 <= p["band"] <= 2 and distance(p["xyz_m"], [0, 0]) <= 5 for p in self.positions)
        )
        self.assertEqual(
            Counter(p["band"] for p in self.positions if p["split"] == "train"),
            {0: 33, 1: 80, 2: 80},
        )

    def test_only_train_and_validation_positions_are_captured(self):
        plan = build_plan()
        self.assertEqual(len(plan["samples"]), 1672)
        self.assertEqual(
            Counter(s["split"] for s in plan["samples"]), dict(train=1544, validation=128)
        )
        self.assertEqual(len({s["sample_id"] for s in plan["samples"]}), 1672)
        for position in {s["position_id"] for s in plan["samples"]}:
            group = [s for s in plan["samples"] if s["position_id"] == position]
            self.assertEqual(len(group), 8)
            self.assertEqual(len({s["split"] for s in group}), 1)
            self.assertEqual({s["planned_body_yaw_deg"] for s in group}, set(range(0, 360, 45)))


if __name__ == "__main__":
    unittest.main()
