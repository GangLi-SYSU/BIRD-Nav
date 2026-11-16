"""Batch acquisition must reject stale products and keep each camera's evidence."""

from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.camera_geometry import usd_world_matrix
from insect_utils.downward_camera import capture_batch


class BatchCaptureTests(unittest.TestCase):
    def setUp(self):
        self.clock = {"frame": 0, "steps": 0}
        self.omni = types.ModuleType("omni")
        parent = types.ModuleType("omni.replicator")
        core = types.ModuleType("omni.replicator.core")

        def step(**kwargs):
            self.assertEqual(kwargs, {"rt_subframes": 2, "pause_timeline": False})
            self.clock["frame"] += 1
            self.clock["steps"] += 1

        core.orchestrator = types.SimpleNamespace(step=step)
        self.omni.replicator = parent
        parent.core = core
        self.modules = patch.dict(
            sys.modules,
            {"omni": self.omni, "omni.replicator": parent, "omni.replicator.core": core},
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def camera(self, index, stale_until=0, wrong_pose=False):
        clock = self.clock

        class Camera:
            backend = "replicator"
            last_frame = -1
            settings = {
                "resolution_wh": [2, 2],
                "usd_focal_length": 1.0,
                "usd_aperture_wh": [1.0, 1.0],
            }
            path = "/World/Camera_" + str(index)
            product_path = "/Render/Product_" + str(index)

            def _frame(self):
                return clock["frame"]

            def set_body_pose(self, position, yaw):
                self.matrix = np.asarray(usd_world_matrix(position, yaw), dtype=float)
                return True

        camera = Camera()
        camera.reference = types.SimpleNamespace(
            get_data=lambda: {
                "referenceTimeNumerator": 0 if clock["frame"] <= stale_until else clock["frame"]
            }
        )

        def params():
            matrix = camera.matrix.copy()
            if wrong_pose:
                matrix[3, 0] += 1
            return {
                "cameraViewTransform": np.linalg.inv(matrix).ravel(),
                "cameraProjection": np.eye(4).ravel(),
            }

        camera.params = types.SimpleNamespace(get_data=params)
        camera.rgb = types.SimpleNamespace(
            get_data=lambda: np.full((2, 2, 3), 50 + index, dtype=np.uint8)
        )
        camera.prim = types.SimpleNamespace(
            GetAttribute=lambda name: types.SimpleNamespace(
                Get=lambda: 0.0 if name == "fStop" else 1.0
            )
        )
        return camera

    def test_all_cameras_share_steps_but_keep_distinct_images_and_receipts(self):
        cameras = [self.camera(0), self.camera(1)]
        result = capture_batch(
            [(cameras[0], [1, 2, 40], 0, "a"), (cameras[1], [3, 4, 40], 45, "b")]
        )
        self.assertEqual(self.clock["steps"], 4)
        self.assertEqual([item[1]["pose_id"] for item in result], ["a", "b"])
        self.assertEqual([int(item[0][0, 0, 0]) for item in result], [50, 51])
        self.assertEqual(result[0][1]["position_m"], [1.0, 2.0, 40.0])
        self.assertEqual(result[1][1]["position_m"], [3.0, 4.0, 40.0])
        self.assertTrue(all(item[1]["stability"]["consecutive_changes"] == 2 for item in result))

    def test_slow_camera_does_not_replace_an_earlier_accepted_camera_receipt(self):
        a, b = self.camera(0), self.camera(1, stale_until=2)
        result = capture_batch([(a, [0, 0, 40], 0, "a"), (b, [1, 1, 40], 0, "b")])
        self.assertEqual(self.clock["steps"], 6)
        self.assertEqual(result[0][1]["rendering_frame"], 4)
        self.assertEqual(result[1][1]["rendering_frame"], 6)

    def test_one_stale_or_wrong_pose_product_blocks_the_batch(self):
        for stale, wrong in [(100, False), (0, True)]:
            with self.subTest(stale=stale, wrong=wrong):
                self.clock.update(frame=0, steps=0)
                a, b = self.camera(0), self.camera(1, stale_until=stale, wrong_pose=wrong)
                with self.assertRaisesRegex(RuntimeError, "b"):
                    capture_batch([(a, [0, 0, 40], 0, "a"), (b, [1, 1, 40], 0, "b")], max_updates=8)
                self.assertIsNone(b.last_receipt)


if __name__ == "__main__":
    unittest.main()
