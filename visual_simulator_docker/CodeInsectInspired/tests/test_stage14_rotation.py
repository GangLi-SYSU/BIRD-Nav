"""20m-only scope, unchanged controls, paired outcomes and circular-model loading."""

import copy
from pathlib import Path
import sys
import unittest

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.experiment_state import read_json
from insect_utils.navigation_core import make_episodes, simulate_oracle
from insect_utils.rotation_protocol import PROTOCOL, validate_protocol
from insect_utils.rotation_report import matched_outcomes
from insect_utils.rotation_runtime import reference_predictor


class RotationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = read_json(PROJECT / PROTOCOL)
        cls.training = read_json(PROJECT / "runs/stage13_circular_01/frozen.json")
        cls.baseline = read_json(PROJECT / "runs/stage12_navigation_01/frozen.json")

    def test_only20m_and64_matched_initial_conditions_per_arm(self):
        validate_protocol(self.protocol, self.training, self.baseline)
        episodes = make_episodes(self.protocol["formal"], [0, 0, 40], "formal")
        self.assertEqual(len(episodes), 64)
        self.assertEqual({r["radius_m"] for r in episodes}, {20})
        old = [r for r in self.baseline["episodes"]["formal"] if r["radius_m"] == 20]
        self.assertEqual(episodes, old)

    def test_other_navigation_radii_are_rejected(self):
        for radii in ([5, 10, 20, 30, 40], [10], [20, 40]):
            changed = copy.deepcopy(self.protocol)
            changed["formal"]["radii_m"] = radii
            with self.assertRaises(ValueError):
                validate_protocol(changed, self.training, self.baseline)

    def test_controller_success_threshold_and_budget_cannot_silently_change(self):
        for key, value in (("success_radius_m", 1), ("max_steps", 100), ("fixed_yaw", True)):
            changed = copy.deepcopy(self.protocol)
            changed["controller"][key] = value
            with self.assertRaises(ValueError):
                validate_protocol(changed, self.training, self.baseline)

    def test_different_training_data_or_missing_arm_is_rejected(self):
        for key, value in (("training_fingerprint", "changed"), ("arms", ["mask_rotation"])):
            changed = copy.deepcopy(self.protocol)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_protocol(changed, self.training, self.baseline)

    def test20m_oracle_geometry_converges_at_all_fixed_starts(self):
        for episode in make_episodes(self.protocol["formal"], [0, 0, 40], "oracle"):
            result = simulate_oracle(
                episode, [0, 0, 40], self.protocol["controller"], self.baseline["budget"]
            )
            self.assertEqual(result["reason"], "success")
            self.assertLessEqual(result["final_distance_m"], 0.5)


class MatchedOutcomeTests(unittest.TestCase):
    def rows(self):
        return [
            dict(
                radius_m=20,
                bearing_deg=i * 90,
                initial_yaw_deg=0,
                initial_position_m=[i, 20, 40],
                reason="success" if i < 2 else "timeout",
                steps=20 + i,
                final_distance_m=0.4 if i < 2 else 2.0,
                path_length_m=30 + i,
            )
            for i in range(4)
        ]

    def test_paired_success_counts_keep_discordant_trials(self):
        a = self.rows()
        b = copy.deepcopy(a)
        b[1]["reason"] = "timeout"
        b[2]["reason"] = "success"
        result = matched_outcomes(a, list(reversed(b)))
        self.assertEqual(
            result["counts"],
            dict(both_success=1, only_mask_success=1, only_rotation_success=1, both_fail=1),
        )
        self.assertEqual(result["both_success_mean_step_difference_rotation_minus_mask"], 0)

    def test_unmatched_duplicate_or_changed_initial_positions_are_rejected(self):
        a = self.rows()
        for b in (a[:-1], a + [a[0]]):
            with self.assertRaises(ValueError):
                matched_outcomes(a, b)
        b = copy.deepcopy(a)
        b[0]["initial_position_m"][0] += 0.1
        with self.assertRaises(ValueError):
            matched_outcomes(a, b)

    def test_no_common_success_has_no_conditional_path_claim(self):
        a = self.rows()
        b = copy.deepcopy(a)
        for row in b:
            row["reason"] = "timeout"
        result = matched_outcomes(a, b)
        self.assertIsNone(result["both_success_mean_step_difference_rotation_minus_mask"])


class CircularNavigationModelTests(unittest.TestCase):
    def test_navigation_reference_loads_circle_weights_and_restores_meters_once(self):
        torch.set_num_threads(2)
        base = "runs/stage13_circular_acceptance_02/smoke/mask_rotation/"
        frozen = dict(model=dict(weights=base + "best_validation.pth", scale_m=44))
        model = reference_predictor(frozen)
        scripted = torch.jit.load(str(PROJECT / base / "deployment/home_vector_m.torchscript.pt"))
        raw = torch.rand(2, 3, 1024, 1024)
        with torch.inference_mode():
            actual, expected = model(raw), scripted(raw)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-4))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 10628)
        masked = model.network.preprocess(raw)
        self.assertTrue(torch.equal(masked[:, :, 0, 0], torch.zeros(2, 3)))


if __name__ == "__main__":
    unittest.main()
