import unittest
import numpy as np
from insect_utils.interactive_homing import DemoFlight
from insect_utils.demo_pi_comparison import compare_pi


class PIComparisonTests(unittest.TestCase):
    def route(self, sigma=0):
        f = DemoFlight(sigma)
        f.takeoff()
        f.altitude_tick(8)
        for _ in range(200):
            f.manual_tick(1, 0, 0, 0.5)
        for _ in range(100):
            f.manual_tick(0, 1, 0, 0.5)
        f.start_homing()
        return f

    def test_no_distance_or_heading_noise_both_return_to_origin(self):
        f = self.route()
        snapshot = f.odometry.snapshot()
        snapshot["gyro_deg_per_sqrt_s"] = 0
        for s in snapshot["outbound"]:
            s["gyro_estimate_xy_m"] = s["magnetic_estimate_xy_m"][:]
            s["gyro_bias_rad"] = 0
        result = compare_pi(snapshot)
        for stat in result["statistics"].values():
            self.assertEqual(stat["successes"], 50)
            np.testing.assert_allclose(stat["endpoints_m"], 0, atol=1e-8)

    def test_compass_removes_heading_drift_but_not_distance_noise(self):
        f = self.route(0.1)
        snapshot = f.odometry.snapshot()
        result = compare_pi(snapshot)
        self.assertEqual(result["compass_noise_deg"], 0)
        self.assertGreater(
            np.linalg.norm(
                np.array(snapshot["outbound"][-1]["gyro_estimate_xy_m"])
                - snapshot["outbound"][-1]["magnetic_estimate_xy_m"]
            ),
            1,
        )
        self.assertGreater(result["statistics"]["magnetic"]["mean_endpoint_error_m"], 0)
        self.assertEqual(result["statistics"]["magnetic"]["trials"], 50)
        self.assertEqual(result, compare_pi(snapshot))

    def test_actual_trace_retains_handoff_and_does_not_change_live_motion(self):
        f = self.route(0.5)
        for _ in range(500):
            f.pi_tick(0.5)
        snapshot = f.odometry.snapshot()
        self.assertEqual(f.phase, "VISUAL_RETURN")
        np.testing.assert_allclose(snapshot["actual_pi_return"][-1]["true_xy_m"], f.position[:2])
        before = f.rng.bit_generator.state
        compare_pi(snapshot)
        self.assertEqual(before, f.rng.bit_generator.state)


if __name__ == "__main__":
    unittest.main()
