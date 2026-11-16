import unittest
import numpy as np
from insect_utils.demo_overlays import nose_tip


class HeadingOverlayTests(unittest.TestCase):
    def test_body_heading_in_north_up_image(self):
        for yaw, expected in [
            (0, [126, 100]),
            (90, [100, 74]),
            (180, [74, 100]),
            (270, [100, 126]),
            (450, [100, 74]),
        ]:
            np.testing.assert_array_equal(nose_tip([100, 100], yaw), expected)


if __name__ == "__main__":
    unittest.main()
