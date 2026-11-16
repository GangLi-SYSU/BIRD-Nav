import unittest
import numpy as np
from insect_utils.interactive_homing import DemoFlight


class InteractiveFlightTests(unittest.TestCase):
    def flying(self, sigma=0):
        f = DemoFlight(sigma)
        f.takeoff()
        f.altitude_tick(8)
        self.assertEqual(f.phase, "MANUAL")
        return f

    def test_zero_noise_full_pi_return(self):
        f = self.flying()
        f.manual_tick(1, 0, 0, 40)
        f.manual_tick(0, 1, 0, 20)
        f.start_homing()
        for _ in range(100):
            f.pi_tick(0.5)
        self.assertEqual(f.phase, "VISUAL_RETURN")
        np.testing.assert_allclose(f.position, [0, 0, 40], atol=1e-10)

    def test_truth_does_not_trigger_handoff(self):
        f = self.flying()
        f.manual_tick(1, 0, 0, 40)
        f.position[:2] = [0, 0]
        f.start_homing()
        f.pi_tick(0.5)
        self.assertEqual(f.phase, "PI_RETURN")
        f.estimated_xy[:] = 0
        f.position[:2] = [100, 50]
        f.pi_tick(0.5)
        self.assertEqual(f.phase, "VISUAL_RETURN")
        self.assertGreater(f.handoff_distance, 15)

    def test_pause_freezes_manual_and_return(self):
        f = self.flying(0.1)
        f.paused = True
        f.manual_tick(1, 1, 1, 10)
        np.testing.assert_array_equal(f.position, [0, 0, 40])

    def test_world_heading_and_strafe(self):
        f = self.flying()
        f.yaw = 90
        f.manual_tick(1, 0, 0, 1)
        np.testing.assert_allclose(f.position[:2], [0, 2.5], atol=1e-10)
        f.manual_tick(0, 1, 0, 1)
        np.testing.assert_allclose(f.position[:2], [-2.5, 2.5], atol=1e-10)

    def test_visual_uses_supplied_motion_and_fifty_step_limit(self):
        f = self.flying()
        f.phase = "VISUAL_RETURN"
        f.position[:2] = [20, 0]
        for _ in range(50):
            f.apply_visual_motion(dict(displacement_world_m=[0, 0.1], next_yaw_deg=90))
        self.assertEqual(f.phase, "FAILED")
        np.testing.assert_allclose(f.position[:2], [20, 5])

    def test_landing_preserves_error_and_measures_horizontal_distance(self):
        f = self.flying()
        f.position[:] = [3, 4, 40]
        f.phase = "SUCCESS"
        self.assertTrue(f.start_landing())
        f.landing_tick(f.landing_duration / 2)
        np.testing.assert_allclose(f.position, [3, 4, 20.15])
        self.assertIsNone(f.landing_measurement())
        f.landing_tick(100)
        self.assertEqual(f.phase, "LANDED")
        np.testing.assert_array_equal(f.position, [3, 4, 0.3])
        self.assertEqual(f.landing_measurement()["horizontal_distance_m"], 5)
        self.assertEqual(f.landing_measurement()["homing_outcome"], "SUCCESS")

    def test_landing_pause_and_failed_homing_remain_distinct(self):
        f = self.flying()
        f.phase = "FAILED"
        f.reason = "Visual controller reached 50 steps"
        f.start_landing()
        f.paused = True
        f.landing_tick(100)
        self.assertEqual(f.position[2], 40)
        f.paused = False
        f.landing_tick(100)
        self.assertEqual(f.phase, "LANDED")
        self.assertEqual(f.landing_measurement()["homing_outcome"], "FAILED")
        self.assertEqual(f.reason, "Visual controller reached 50 steps")

    def test_cannot_land_before_homing_ends(self):
        f = self.flying()
        self.assertFalse(f.start_landing())
        self.assertEqual(f.phase, "MANUAL")


if __name__ == "__main__":
    unittest.main()
