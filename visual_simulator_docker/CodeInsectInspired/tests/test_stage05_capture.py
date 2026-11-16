"""Regression checks for missing samples, measured labels and rotated footprints."""

import copy
import json
from pathlib import Path
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.camera_geometry import optics, usd_world_matrix
from insect_utils.capture_geometry import exact_batches, footprint, intersection_area
from insect_utils.capture_geometry import make_samples, measured_record, polygon_area
from insect_utils.frame_stability import STABILITY_POLICY
from insect_utils.experiment_config import preflight, resolve_spec, stage_parameters, validate_spec


class CaptureGeometryTests(unittest.TestCase):
    def setUp(self):
        self.home = [12.0, -8.0, 12.0]
        self.sample = make_samples([[14, -7, 12]], [[90]], self.home, 2)[0]
        self.receipt = {
            "pose_id": self.sample["pose_id"],
            "camera_world_matrix": usd_world_matrix([14, -7, 12], 90),
            "rendering_frame": 12,
            "updates_waited": 4,
            "stability": {
                "policy": dict(STABILITY_POLICY),
                "consecutive_changes": 2,
                "history": [
                    {"frame": 10, "rgb_mae": None},
                    {"frame": 11, "rgb_mae": 0.4},
                    {"frame": 12, "rgb_mae": 0.3},
                ],
            },
        }
        self.settings = optics(
            {
                "mode": "downward_pinhole",
                "crop_box": None,
                "fov_definition_confirmed": True,
                "hfov_deg": 30,
                "raw_resolution_wh": [1024, 1024],
            }
        )

    def record(self):
        return measured_record(self.sample, self.receipt, self.home, 2, 10)

    def test_four_cameras_keep_all_147_including_last_three(self):
        batches = list(exact_batches(list(range(147)), 4))
        self.assertEqual(len(batches), 37)
        self.assertEqual(batches[-1], [(0, 144), (1, 145), (2, 146)])
        self.assertEqual([sample for batch in batches for _, sample in batch], list(range(147)))

    def test_single_camera_and_more_cameras_than_samples(self):
        self.assertEqual(list(exact_batches([3, 4], 1)), [[(0, 3)], [(0, 4)]])
        self.assertEqual(list(exact_batches([3, 4], 8)), [[(0, 3), (1, 4)]])

    def test_invalid_camera_count_rejected(self):
        for value in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                list(exact_batches([1], value))

    def test_headings_share_position_group_and_keep_unique_ids(self):
        samples = make_samples(
            [[0, 0, 10], [1, 0, 10]], [[0, 90, 180, 270], [0, 90, 180, 270]], [0, 0, 10], 0
        )
        self.assertEqual(len({s["sample_id"] for s in samples}), 8)
        self.assertEqual(len({s["image"] for s in samples}), 8)
        self.assertEqual(len({s["split_group"] for s in samples}), 2)
        self.assertTrue(all(s["role"] == "learning" for s in samples))

    def test_nonzero_home_ground_and_cardinal_yaws_use_actual_matrix(self):
        for yaw in [0, 90, 180, 270, 37.5]:
            sample = make_samples([[14, -7, 12]], [[yaw]], self.home, 2)[0]
            receipt = copy.deepcopy(self.receipt)
            receipt["camera_world_matrix"] = usd_world_matrix([14, -7, 12], yaw)
            result = measured_record(sample, receipt, self.home, 2, 10)
            self.assertAlmostEqual(result["yaw"], yaw)
            self.assertAlmostEqual(result["agl_m"], 10)
            if yaw == 90:
                self.assertAlmostEqual(result["target_x"], -1)
                self.assertAlmostEqual(result["target_y"], 2)

    def test_wrong_pose_id_or_measured_pose_rejected(self):
        self.receipt["pose_id"] = "old_pose"
        with self.assertRaisesRegex(ValueError, "ID mismatch"):
            self.record()
        self.receipt["pose_id"] = self.sample["pose_id"]
        self.receipt["camera_world_matrix"][3][0] += 1
        with self.assertRaisesRegex(ValueError, "differs from planned"):
            self.record()

    def test_small_measured_offset_changes_label_instead_of_copying_planned_label(self):
        self.receipt["camera_world_matrix"][3][0] += 0.00001
        row = self.record()
        self.assertAlmostEqual(row["x"], 14.00001)
        self.assertAlmostEqual(row["target_y"], 2.00001)

    def test_old_height_bug_cannot_silently_enter_labels(self):
        for z in [0.01, 1.5]:
            with self.assertRaises(ValueError):
                sample = make_samples([[14, -7, z]], [[90]], self.home, 0)[0]
                receipt = copy.deepcopy(self.receipt)
                receipt["camera_world_matrix"] = usd_world_matrix([14, -7, z], 90)
                measured_record(sample, receipt, self.home, 0, 10)

    def test_unsatisfied_stability_or_old_frame_rejected(self):
        self.receipt["stability"]["history"][-1]["rgb_mae"] = 0.6
        with self.assertRaisesRegex(ValueError, "settled-frame"):
            self.record()
        self.receipt["stability"]["history"][-1]["rgb_mae"] = 0.3
        self.receipt["rendering_frame"] = 11
        with self.assertRaisesRegex(ValueError, "settled-frame"):
            self.record()

    def test_square_overlap_includes_rotation_and_translation(self):
        a = footprint([0, 0, 10], 0, 0, self.settings)
        b = footprint([0, 0, 10], 90, 0, self.settings)
        c = footprint([0, 0, 10], 45, 0, self.settings)
        area = polygon_area(a)
        self.assertAlmostEqual(intersection_area(a, b), area)
        self.assertAlmostEqual(intersection_area(a, c) / area, 2 * (2 ** 0.5 - 1))
        self.assertEqual(intersection_area(a, footprint([20, 0, 10], 0, 0, self.settings)), 0)
        self.assertAlmostEqual(
            intersection_area(a, footprint([1, 0, 10], 0, 0, self.settings)) / area,
            1 - 1 / (area ** 0.5),
        )


class CaptureConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(
            (PROJECT / "config/experiments/downward_h10_hfov30_landmarks.json").read_text()
        )

    def test_camera_count_and_heading_count_have_independent_budgets(self):
        for cameras in [1, 4]:
            for headings in [1, 4, 8]:
                self.spec["capture"].update(number_cameras=cameras, headings_per_position=headings)
                resolved = resolve_spec(self.spec)
                self.assertEqual(resolved["derived"]["planned_positions"], 147)
                self.assertEqual(resolved["derived"]["expected_samples"], 147 * headings)
                params = stage_parameters(resolved, PROJECT / "runs/example")["capture"]
                self.assertEqual(params["headings_per_position"], headings)
                self.assertEqual(params["seed"], 10)

    def test_no_heading_choice_blocks_real_capture_before_kit(self):
        self.spec["capture"].pop("headings_per_position", None)
        with self.assertRaisesRegex(ValueError, "explicitly confirm"):
            preflight(resolve_spec(self.spec), "capture")
        self.assertNotIn("omni.isaac.kit", sys.modules)

    def test_heading_counts_must_be_valid_explicit_integers(self):
        for count in [True, 1.0, 0, 2, 5, -1]:
            self.spec["capture"]["headings_per_position"] = count
            with self.assertRaises(ValueError):
                validate_spec(self.spec)

    def test_training_and_closed_loop_stay_gated_after_capture_adapter_is_added(self):
        self.spec["capture"]["headings_per_position"] = 1
        resolved = resolve_spec(self.spec)
        with self.assertRaisesRegex(ValueError, "stage F"):
            preflight(resolved, "train")
        with self.assertRaisesRegex(ValueError, "stage H"):
            preflight(resolved, "evaluate")

    def test_selected_eight_heading_protocol_reaches_all_downward_configs(self):
        protocol = json.loads(
            (PROJECT / "config/experiments/stage05_sampling_protocol.json").read_text()
        )
        self.assertEqual(protocol["heading_coverage"]["headings_per_position"], 8)
        for path in (PROJECT / "config/experiments").glob("downward_h10_hfov30*.json"):
            spec = json.loads(path.read_text())
            self.assertEqual(spec["capture"]["headings_per_position"], 8)
            resolved = resolve_spec(spec)
            self.assertEqual(resolved["derived"]["planned_positions"], 147)
            self.assertEqual(resolved["derived"]["expected_samples"], 1176)
            self.assertTrue(resolved["provenance"]["sampling_protocol_sha256"])


if __name__ == "__main__":
    unittest.main()
