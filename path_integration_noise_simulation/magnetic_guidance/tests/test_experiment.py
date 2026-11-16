import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_experiment import schedule, simulate, wilson  # noqa: E402


class MagneticIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())

    def test_nominal_distance_and_fractional_step(self):
        for route, multiplier in [("straight", 1), ("dogleg", 2)]:
            (length, angle, duration), split = schedule(251.13, route, self.config)
            vector = length[:, None] * np.stack([np.cos(angle), np.sin(angle)], -1)
            np.testing.assert_allclose(vector[:split].sum(0), [251.13, 0], atol=1e-10)
            np.testing.assert_allclose(vector.sum(0), [0, 0], atol=1e-10)
            self.assertAlmostEqual(length[:split].sum(), 251.13 * multiplier)
            self.assertTrue(np.all(duration <= self.config["delta_t_s"] + 1e-10))

    def test_perfect_compass_returns_home_without_distance_noise(self):
        for route in ["straight", "dogleg"]:
            result, _ = simulate(self.config, 251.13, route, 0, [0, 1, 2])
            np.testing.assert_allclose(result["magnetic"][:, -1], 0, atol=1e-9)
            self.assertGreater(np.linalg.norm(result["gyro_only"][:, -1]), 1)

    def test_zero_gyro_arms_identical(self):
        self.config["gyro_noise_deg_per_sqrt_s"] = 0
        result, _ = simulate(self.config, 250, "dogleg", 0.5, [0, 1, 2])
        np.testing.assert_array_equal(result["magnetic"], result["gyro_only"])

    def test_trials_can_be_extended_without_changing_existing_samples(self):
        first, _ = simulate(self.config, 250, "dogleg", 0.5, [0, 1, 2])
        later, _ = simulate(self.config, 250, "dogleg", 0.5, [2, 3])
        for arm in first:
            np.testing.assert_array_equal(first[arm][2], later[arm][0])

    def test_diffusion_variance_and_linear_sigma_scaling(self):
        # Straight compass path: signed endpoint variance = sigma^2 * 2D.
        result, _ = simulate(self.config, 20, "straight", 0.5, list(range(4000)))
        observed = np.mean(np.sum(result["magnetic"][:, -1] ** 2, axis=1))
        self.assertLess(abs(observed - 10), 0.7)
        twice, _ = simulate(self.config, 20, "straight", 1, list(range(10)))
        np.testing.assert_allclose(
            twice["magnetic"][:, -1], 2 * result["magnetic"][:10, -1], atol=1e-10
        )

    def test_confidence_interval(self):
        low, high = wilson(50, 50)
        self.assertAlmostEqual(low, 0.9286524, places=6)
        self.assertAlmostEqual(high, 1)


if __name__ == "__main__":
    unittest.main()
