import unittest
import numpy as np
from insect_utils.demo_animation import SmoothAction


class SmoothActionTests(unittest.TestCase):
    def test_intermediate_motion_and_exact_endpoint(self):
        m = dict(displacement_world_m=[2, 1], next_yaw_deg=10, step_size_m=2.236)
        a = SmoothAction([5, 3, 40], 350, m)
        p, y, done = a.advance(a.duration / 2)
        np.testing.assert_allclose(p, [6, 3.5, 40])
        self.assertAlmostEqual(y, 0)
        self.assertFalse(done)
        p, y, done = a.advance(99)
        np.testing.assert_allclose(p, [7, 4, 40])
        self.assertEqual(y, 10)
        self.assertTrue(done)

    def test_pause_keeps_pose(self):
        a = SmoothAction(
            [0, 0, 40], 0, dict(displacement_world_m=[1, 0], next_yaw_deg=30, step_size_m=1)
        )
        first = a.advance(0.3)
        second = a.advance(10, paused=True)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1:], second[1:])


if __name__ == "__main__":
    unittest.main()
