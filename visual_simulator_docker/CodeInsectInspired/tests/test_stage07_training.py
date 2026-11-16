"""Regression metrics, position subsets and checkpoint byte guards without torch."""

from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.experiment_config import digest
from insect_utils.experiment_state import atomic_json
from insect_utils.training_state import (
    checkpoint_pointer,
    position_subset,
    quantile,
    vector_metrics,
)


class RegressionMetricTests(unittest.TestCase):
    def test_component_mse_and_vector_mse_have_explicit_distinct_normalization(self):
        result = vector_metrics([[0, 0]], [[3, 4]])
        self.assertEqual(result["component_mse_m2"], 12.5)
        self.assertEqual(result["mean_squared_vector_error_m2"], 25)
        self.assertEqual(result["vector_error_m"]["mean"], 5)

    def test_shortest_angle_wraps_at_180_degrees(self):
        import math

        a = math.radians(179)
        b = math.radians(-179)
        result = vector_metrics([[math.cos(a), math.sin(a)]], [[math.cos(b), math.sin(b)]])
        self.assertAlmostEqual(result["absolute_shortest_angle_error_deg"]["mean"], 2)
        result = vector_metrics([[-1, 0]], [[1, 0]])
        self.assertEqual(result["absolute_shortest_angle_error_deg"]["mean"], 180)

    def test_zero_vectors_exclude_undefined_angle_instead_of_reporting_zero_error(self):
        result = vector_metrics([[0, 0], [1, 0], [0, 1]], [[1, 0], [0, 0], [1, 0]])
        self.assertEqual(result["valid_angle_samples"], 1)
        self.assertEqual(result["excluded_angle_samples"], 2)
        self.assertEqual(result["zero_target_samples"], 1)
        self.assertEqual(result["zero_prediction_samples"], 1)
        self.assertEqual(result["absolute_shortest_angle_error_deg"]["mean"], 90)

    def test_nonfinite_empty_or_mismatched_outputs_are_rejected(self):
        for prediction, target in [([], []), ([[1, 0]], []), ([[float("nan"), 0]], [[1, 0]])]:
            with self.assertRaises(ValueError):
                vector_metrics(prediction, target)

    def test_quantiles_interpolate_and_handle_empty_angles(self):
        self.assertEqual(quantile([0, 10], 0.5), 5)
        self.assertIsNone(quantile([], 0.95))
        with self.assertRaises(ValueError):
            quantile([1, 2], 1.1)


class CheckpointAndSubsetTests(unittest.TestCase):
    def test_position_subset_keeps_every_heading_and_spreads_positions(self):
        records = [{"position_id": "p{:03d}".format(i)} for i in range(10) for _ in range(8)]
        indices, groups = position_subset(records, 4)
        self.assertEqual(groups, ["p000", "p003", "p006", "p009"])
        self.assertEqual(len(indices), 32)
        self.assertEqual({records[i]["position_id"] for i in indices}, set(groups))

    def test_invalid_subset_size_is_rejected(self):
        for value in [0, True, 3, 1.5]:
            with self.assertRaises(ValueError):
                position_subset([{"position_id": "p0"}], value)

    def test_checkpoint_pointer_rejects_changed_config_and_changed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkpoint = folder / "epoch.pt"
            checkpoint.write_bytes(b"committed checkpoint")
            atomic_json(
                folder / "checkpoint_latest.json",
                {
                    "file": "epoch.pt",
                    "sha256": digest(checkpoint),
                    "fingerprint": "configuration_a",
                },
            )
            checkpoint_pointer(folder, "configuration_a")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                checkpoint_pointer(folder, "configuration_b")
            checkpoint.write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                checkpoint_pointer(folder, "configuration_a")

    def test_checkpoint_pointer_cannot_escape_its_phase_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            atomic_json(
                folder / "checkpoint_latest.json",
                {"file": "../outside.pt", "sha256": "x", "fingerprint": "a"},
            )
            with self.assertRaises(ValueError):
                checkpoint_pointer(folder, "a")


if __name__ == "__main__":
    unittest.main()
