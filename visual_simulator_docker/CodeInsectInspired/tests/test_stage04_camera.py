"""Geometry invariants for approved nadir optics and body/image coordinates."""

import json
import math
from pathlib import Path
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.camera_geometry import body_to_world, body_yaw_from_quaternion
from insect_utils.camera_geometry import home_vector_body, optics, pixel_to_flat_world
from insect_utils.camera_geometry import usd_quaternion_wxyz, usd_world_matrix, world_to_pixel
from insect_utils.experiment_config import load_config, stage_parameters


class DownwardGeometryTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(
            (PROJECT / "config/experiments/downward_h10_hfov30_landmarks.json").read_text()
        )
        self.camera = self.spec["camera"]
        self.settings = optics(self.camera)

    def test_meter_and_centimeter_stages_use_physical_24mm_lens(self):
        self.assertAlmostEqual(optics(self.camera, focal_length_mm=24)["usd_focal_length"], 0.24)
        self.assertAlmostEqual(
            optics(self.camera, meters_per_unit=0.01, focal_length_mm=24)["usd_focal_length"], 24
        )
        self.assertAlmostEqual(
            optics(self.camera, focal_length_mm=24)["sensor_size_mm"][0], 12.8615612367
        )

    def test_equal_scaling_of_virtual_lens_and_filmback_preserves_projection(self):
        a = optics(self.camera, focal_length_mm=50)
        b = optics(self.camera, focal_length_mm=100)
        self.assertEqual(a["intrinsics"], b["intrinsics"])
        self.assertEqual(a["hfov_deg"], b["hfov_deg"])
        self.assertAlmostEqual(b["usd_aperture_wh"][0], 2 * a["usd_aperture_wh"][0])

    def test_square_sensor_has_equal_focal_pixels_and_30deg_vertical_fov(self):
        self.assertAlmostEqual(self.settings["intrinsics"][0][0], 1910.8100134753)
        self.assertAlmostEqual(self.settings["intrinsics"][1][1], 1910.8100134753)
        self.assertAlmostEqual(self.settings["vfov_deg"], 30)

    def test_unconfirmed_fov_and_panorama_crop_are_rejected(self):
        for change in ({"fov_definition_confirmed": False}, {"crop_box": [0, 312, 1024, 712]}):
            spec = dict(self.camera, **change)
            with self.assertRaises(ValueError):
                optics(spec)

    def test_camera_axes_are_right_body_minus_y_and_up_body_x(self):
        matrix = usd_world_matrix([12, -8, 10], 0)
        for actual, expected in zip(matrix[0][:3], [0, -1, 0]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(matrix[1][:3], [1, 0, 0]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(matrix[3], [12, -8, 10, 1])

    def test_known_cardinal_points_land_above_left_below_right(self):
        position = [12, -8, 10]
        for point, direction in (
            ([14, -8, 0], "up"),
            ([12, -6, 0], "left"),
            ([10, -8, 0], "down"),
            ([12, -10, 0], "right"),
        ):
            u, v = world_to_pixel(point, position, 0, self.settings)
            if direction == "up":
                self.assertLess(v, 512)
            elif direction == "left":
                self.assertLess(u, 512)
            elif direction == "down":
                self.assertGreater(v, 512)
            else:
                self.assertGreater(u, 512)

    def test_four_headings_keep_body_forward_at_image_top(self):
        position = [12, -8, 10]
        for yaw in (0, 90, 180, 270):
            forward = body_to_world([1, 0], yaw)
            u, v = world_to_pixel(
                [12 + forward[0], -8 + forward[1], 0], position, yaw, self.settings
            )
            self.assertAlmostEqual(u, 512)
            self.assertLess(v, 512)

    def test_pixel_projection_round_trip_includes_nonzero_ground_and_yaw(self):
        for yaw in (0, 37, 90, 180, 270):
            point = [12.7, -7.4, 2]
            pixel = world_to_pixel(point, [12, -8, 12], yaw, self.settings)
            restored = pixel_to_flat_world(pixel, [12, -8, 12], yaw, 2, self.settings)
            for a, b in zip(point, restored):
                self.assertAlmostEqual(a, b)

    def test_footprint_edges_are_5_359m_apart(self):
        left = pixel_to_flat_world([0, 512], [0, 0, 10], 0, 0, self.settings)
        right = pixel_to_flat_world([1024, 512], [0, 0, 10], 0, 0, self.settings)
        self.assertAlmostEqual(math.dist(left, right), 5.3589838486)

    def test_body_target_reconstructs_arbitrary_home_without_extra_minus_sign(self):
        position, home = [12, -8, 10], [4, 7, 10]
        for yaw in (0, 90, 180, 270, 32):
            target = home_vector_body(position, home, yaw)
            world = body_to_world(target, yaw)
            for i in range(2):
                self.assertAlmostEqual(position[i] + world[i], home[i])

    def test_sensor_usd_quaternion_is_not_the_body_quaternion(self):
        q = usd_quaternion_wxyz(0)
        self.assertAlmostEqual(q[0], math.sqrt(0.5))
        self.assertAlmostEqual(q[3], -math.sqrt(0.5))
        self.assertAlmostEqual(body_yaw_from_quaternion([math.sqrt(0.5), 0, 0, math.sqrt(0.5)]), 90)
        with self.assertRaises(ValueError):
            body_yaw_from_quaternion([0.5, 0.5, 0.5, 0.5])

    def test_both_pipeline_endpoints_receive_the_confirmed_camera(self):
        resolved = load_config(PROJECT / "config/experiments/downward_h10_hfov30_landmarks.json")
        stages = stage_parameters(resolved, PROJECT / "runs/example")
        self.assertEqual(stages["capture"]["camera"], self.camera)
        self.assertEqual(stages["evaluate"]["render_config"]["camera"], self.camera)
        self.assertEqual(resolved["derived"]["camera_contract"]["mount"], "body_fixed_nadir")

    def test_nonfinite_geometry_and_zero_depth_are_rejected(self):
        for value in (float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                usd_quaternion_wxyz(value)
        with self.assertRaises(ValueError):
            world_to_pixel([0, 0, 10], [0, 0, 10], 0, self.settings)


if __name__ == "__main__":
    unittest.main()
