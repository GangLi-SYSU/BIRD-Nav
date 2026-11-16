"""Coordinate, termination, artifact and coupled-loop checks for navigation."""

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.camera_geometry import home_vector_body, usd_world_matrix
from insect_utils.experiment_state import read_json
from insect_utils.frame_stability import STABILITY_POLICY
from insect_utils.navigation_core import InvalidPrediction, decide_motion, endpoint
from insect_utils.navigation_core import horizontal_distance, make_episodes, safe_segment
from insect_utils.navigation_core import segment_min_distance, simulate_oracle, termination
from insect_utils.navigation_protocol import PROTOCOL
from insect_utils.navigation_results import saved_episodes, summarize


class NavigationGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = read_json(PROJECT / PROTOCOL)
        cls.controller = cls.protocol["controller"]

    def test_authorized_counts_and_all_initial_headings(self):
        for phase, count in (("rendered_oracle", 8), ("pilot", 48), ("formal", 320)):
            episodes = make_episodes(self.protocol[phase], [0, 0, 40], phase)
            self.assertEqual(len(episodes), count)
            self.assertEqual(len({e["episode_id"] for e in episodes}), count)
            for episode in episodes:
                self.assertAlmostEqual(
                    horizontal_distance(episode["initial_position_m"], [0, 0, 40]),
                    episode["radius_m"],
                )

    def test_body_vectors_at_all_yaws_produce_same_world_homeward_action(self):
        position, home = [18, -7, 40], [13, -8, 40]
        expected = None
        for yaw in (0, 45, 90, 180, 270, 359):
            motion = decide_motion(home_vector_body(position, home, yaw), yaw, self.controller)
            after = endpoint(position, motion)
            self.assertLess(horizontal_distance(after, home), horizontal_distance(position, home))
            if expected is None:
                expected = after
            for a, b in zip(expected, after):
                self.assertAlmostEqual(a, b)

    def test_no_legacy_extra_negative_sign(self):
        for position, expected_yaw in (
            ([5, 0, 40], 180),
            ([0, 5, 40], 270),
            ([-5, 0, 40], 0),
            ([0, -5, 40], 90),
        ):
            motion = decide_motion(home_vector_body(position, [0, 0, 40], 0), 0, self.controller)
            self.assertAlmostEqual(motion["next_yaw_deg"], expected_yaw)

    def test_original_dynamic_step_and_cap_are_retained(self):
        self.assertAlmostEqual(decide_motion([3, 0], 0, self.controller)["step_size_m"], 0.5)
        self.assertAlmostEqual(decide_motion([100, 0], 0, self.controller)["step_size_m"], 2.5)

    def test_prediction_does_not_use_true_distance_to_shorten_terminal_step(self):
        start = [0.6, 0, 40]
        motion = decide_motion([-30, 0], 0, self.controller)
        after = endpoint(start, motion)
        self.assertAlmostEqual(after[0], -1.9)
        self.assertLess(segment_min_distance(start, after, [0, 0, 40]), 0.5)
        self.assertIsNone(termination(after, [0, 0, 40], 1, self.controller))

    def test_success_is_horizontal_even_at40m_height(self):
        self.assertEqual(termination([0.3, 0.4, 40], [0, 0, 0], 1, self.controller), "success")
        self.assertIsNone(termination([0.501, 0, 40], [0, 0, 40], 1, self.controller))

    def test_initial_inside_is_separate_and_last_step_success_wins(self):
        self.assertEqual(
            termination([0, 0, 40], [0, 0, 40], 0, self.controller), "initial_within_success"
        )
        self.assertEqual(termination([0.49, 0, 40], [0, 0, 40], 50, self.controller), "success")
        self.assertEqual(termination([1, 0, 40], [0, 0, 40], 50, self.controller), "timeout")

    def test_invalid_predictions_fail_explicitly(self):
        for vector in ([0, 0], [1e-12, 0], [float("nan"), 1], [float("inf"), 0], [1, 2, 3]):
            with self.subTest(vector=vector), self.assertRaises(InvalidPrediction):
                decide_motion(vector, 0, self.controller)

    def test_boundary_guard_checks_entire_safe_convex_segment(self):
        budget = {
            "allowed_camera_center_xy_m": [[-10, -10], [10, 10]],
            "camera_z_m": 40,
            "minimum_surface_clearance_m": 20,
        }
        self.assertTrue(safe_segment([-9, -9, 40], [9, 9, 40], budget)["valid"])
        with self.assertRaises(ValueError):
            safe_segment([9, 0, 40], [11, 0, 40], budget)
        with self.assertRaises(ValueError):
            safe_segment([0, 0, 40], [1, 0, 39], budget)

    def test_oracle_converges_from_all_bearings_yaws_and_nonzero_homes(self):
        group = {
            "radii_m": [2, 3, 5, 40],
            "bearings_per_radius": 4,
            "initial_yaws_deg": [0, 90, 180, 270],
        }
        for home in ([0, 0, 40], [13, -8, 40]):
            for episode in make_episodes(group, home, "oracle"):
                result = simulate_oracle(episode, home, self.controller)
                self.assertEqual(result["reason"], "success")
                self.assertLessEqual(result["final_distance_m"], 0.5)
                self.assertLessEqual(result["steps"], 50)


class CoupledLoopTests(unittest.TestCase):
    def fixture(self, controller=None):
        protocol = read_json(PROJECT / PROTOCOL)
        if controller is not None:
            protocol["controller"] = controller
        protocol["home_camera_xyz_m"] = [7, -3, 40]
        group = {"radii_m": [2], "bearings_per_radius": 2, "initial_yaws_deg": [0, 90]}
        return {
            "fingerprint": "engineering_fixture",
            "protocol": protocol,
            "budget": {
                "allowed_camera_center_xy_m": [[-30, -30], [30, 30]],
                "camera_z_m": 40,
                "minimum_surface_clearance_m": 20,
            },
            "episodes": {
                "rendered_oracle": make_episodes(group, [7, -3, 40], "fixture"),
                "pilot": make_episodes(group, [7, -3, 40], "pilot"),
            },
        }

    @staticmethod
    def capture(requests):
        rows = []
        for camera, position, yaw, pose in requests:
            camera.frame += 10
            frame = camera.frame
            receipt = {
                "pose_id": pose,
                "backend": "replicator",
                "camera_prim": camera.path,
                "rendering_frame": frame,
                "camera_world_matrix": usd_world_matrix(position, yaw),
                "updates_waited": 4,
                "stability": {
                    "policy": STABILITY_POLICY,
                    "consecutive_changes": 2,
                    "history": [
                        {"frame": frame - 2, "rgb_mae": None},
                        {"frame": frame - 1, "rgb_mae": 0.0},
                        {"frame": frame, "rgb_mae": 0.0},
                    ],
                },
            }
            rows.append((np.full((4, 4, 3), 80, dtype=np.uint8), receipt))
        return rows

    @staticmethod
    def settings():
        values = {
            "/rtx/rendermode": "RayTracedLighting",
            "/rtx/post/aa/op": 0,
            "/rtx-transient/post/aa/limitedOps": False,
            "/rtx/post/histogram/enabled": False,
        }
        return types.SimpleNamespace(get=values.get)

    @staticmethod
    def pose(_camera, position, yaw):
        return list(position), yaw, usd_world_matrix(position, yaw)

    def test_four_independent_oracle_lanes_commit_pose_and_frame_evidence(self):
        from insect_utils.navigation_runtime import run_episodes

        cameras = [types.SimpleNamespace(path="/World/Fake{}".format(i), frame=0) for i in range(4)]
        frozen = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with patch(
                "insect_utils.downward_camera.capture_batch", side_effect=self.capture
            ), patch("insect_utils.navigation_runtime.apply_pose", side_effect=self.pose):
                result = run_episodes(
                    folder, "rendered_oracle", frozen, cameras, None, self.settings(), []
                )
            self.assertEqual(result["successes"], 4)
            for row in saved_episodes(folder, "rendered_oracle", frozen).values():
                self.assertLessEqual(row["final_distance_m"], 0.5)
                trace = read_json(
                    folder
                    / "rendered_oracle/episodes"
                    / row["episode_id"]
                    / row["attempt"]
                    / "trajectory.json"
                )
                self.assertEqual(len(trace["actions"]), row["steps"])
                self.assertGreater(row["steps"], 0)
                for step in trace["actions"]:
                    self.assertLess(step["executed_action_angle_error_deg"], 1e-5)
                    self.assertEqual(len(step["executed_camera_world_matrix"]), 4)
            image = next((folder / "rendered_oracle/episodes").rglob("*.png"))
            image.write_bytes(b"tampered image")
            with self.assertRaises(ValueError):
                saved_episodes(folder, "rendered_oracle", frozen)

    def test_neural_policy_is_not_corrected_using_true_home(self):
        from insect_utils.navigation_runtime import run_episodes

        controller = dict(read_json(PROJECT / PROTOCOL)["controller"], max_steps=1)
        frozen = self.fixture(controller)
        frozen["episodes"]["pilot"] = frozen["episodes"]["pilot"][:1]
        cameras = [types.SimpleNamespace(path="/World/Fake0", frame=0)]
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "insect_utils.downward_camera.capture_batch", side_effect=self.capture
            ), patch("insect_utils.navigation_runtime.apply_pose", side_effect=self.pose), patch(
                "insect_utils.navigation_runtime.predict_rgb", return_value=([[10, 0]], 0)
            ):
                summary = run_episodes(
                    Path(temporary), "pilot", frozen, cameras, object(), self.settings(), []
                )
            self.assertEqual(summary["successes"], 0)
            self.assertEqual(summary["termination_reasons"], {"timeout": 1})
            result = next(iter(saved_episodes(Path(temporary), "pilot", frozen).values()))
            self.assertGreater(result["final_distance_m"], result["initial_distance_m"])

    def test_invalid_camera_batch_preserves_attempt_without_committing_outcomes(self):
        from insect_utils.navigation_runtime import run_episodes

        frozen = self.fixture()
        cameras = [types.SimpleNamespace(path="/World/Fake0", frame=0)]
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "insect_utils.downward_camera.capture_batch",
                side_effect=RuntimeError("stale frame"),
            ):
                with self.assertRaisesRegex(RuntimeError, "Frame acquisition failed"):
                    run_episodes(
                        Path(temporary),
                        "rendered_oracle",
                        frozen,
                        cameras,
                        None,
                        self.settings(),
                        [],
                    )
            self.assertEqual(saved_episodes(Path(temporary), "rendered_oracle", frozen), {})
            self.assertEqual(len(list(Path(temporary).rglob("invalid_frame.json"))), 1)

    def test_initial_success_excluded_from_nontrivial_rate(self):
        rows = [
            {"reason": "initial_within_success", "radius_m": 0, "initial_yaw_deg": 0},
            {
                "reason": "success",
                "radius_m": 2,
                "initial_yaw_deg": 0,
                "path_length_m": 1.6,
                "initial_distance_m": 2,
                "success_radius_m": 0.5,
                "final_distance_m": 0.4,
                "minimum_endpoint_distance_m": 0.4,
                "steps": 7,
            },
        ]
        result = summarize(rows)
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["nontrivial_episodes"], 1)
        self.assertEqual(result["success_rate"], 1)


class PipelineReceiptTests(unittest.TestCase):
    def test_isaac_zero_exit_without_receipt_cannot_start_the_next_phase(self):
        import run_navigation_experiment as runner

        process = Mock(returncode=0)
        process.poll.return_value = 0
        lock = Mock()
        lock.fileno.return_value = 42
        frozen = {"fingerprint": "fixture", "protocol": {"controller": {"success_radius_m": 0.5}}}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with patch.object(runner, "PROJECT", folder), patch.object(
                runner, "PHASES", ("rendered_oracle", "pilot")
            ), patch.object(runner, "verify_navigation", return_value=frozen), patch.object(
                runner.subprocess, "Popen", return_value=process
            ) as launched:
                with self.assertRaisesRegex(RuntimeError, "completion receipt"):
                    runner.pipeline(folder, lock)
                self.assertEqual(launched.call_count, 1)


if __name__ == "__main__":
    unittest.main()
