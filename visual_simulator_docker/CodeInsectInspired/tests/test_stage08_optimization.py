"""Scene clearance, asymmetric appearance and the square attention contract."""

import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from generate_identifiable_landmarks import layout, mesh
from tools.check_stage07_overfit import candidates


class IdentifiableSceneTests(unittest.TestCase):
    def test_layout_is_reproducible_with_diverse_independent_shape_colors(self):
        items = layout()
        self.assertEqual(items, layout())
        self.assertNotEqual(items, layout(seed=11))
        self.assertEqual(len({p["shape"] for p in items}), 8)
        self.assertGreater(len({(p["shape"], p["material_index"]) for p in items}), 100)

    def test_meshes_stand_on_ground_with_valid_color_faces_and_flight_clearance(self):
        for item in layout():
            points, faces, accents = mesh(item)
            self.assertAlmostEqual(min(p[2] for p in points) + item["height_m"] / 2, 0)
            self.assertAlmostEqual(
                max(p[2] for p in points) + item["height_m"] / 2, item["top_z_m"]
            )
            self.assertLessEqual(item["top_z_m"], 3)
            self.assertTrue(accents)
            self.assertTrue(all(0 <= n < len(faces) for n in accents))
            self.assertTrue(all(0 <= n < len(points) for face in faces for n in face))

    def test_enlarged_objects_remain_separated(self):
        items = layout()
        for i, a in enumerate(items):
            for b in items[i + 1 :]:
                self.assertGreaterEqual(
                    math.dist(a["center_xyz_m"][:2], b["center_xyz_m"][:2]),
                    a["bounding_radius_xy_m"] + b["bounding_radius_xy_m"] + 0.25,
                )

    def test_dense_grid_has_landmark_candidates_without_using_the_learning_path(self):
        items = layout()
        for x in [-25, -12.1, -1.3, 7.2, 21.9]:
            for y in [-25, -10.7, 0.4, 13.8, 24.2]:
                for yaw in [0, 22.5, 45, 67.5]:
                    self.assertGreater(
                        candidates({"x": x, "y": y, "yaw": yaw, "agl_m": 10}, items), 0
                    )


HAS_TORCH = all(importlib.util.find_spec(n) for n in ["torch", "onnx"])


@unittest.skipUnless(HAS_TORCH, "Square network runtime checks require torch and ONNX")
class SquareModelTests(unittest.TestCase):
    def test_input_is_square_and_two_dimensional_features_preserve_batch(self):
        import torch
        from insect_utils.attention_model import create_attention

        torch.set_num_threads(2)
        model = create_attention({"network_variant": "nadir384x384"}).eval()
        observed = {}
        h1 = model.inception1.register_forward_pre_hook(
            lambda _, a: observed.update(input=list(a[0].shape))
        )
        h2 = model.conv4.register_forward_hook(
            lambda _, a, out: observed.update(last=list(out.shape))
        )
        images = torch.rand(2, 3, 1024, 1024)
        result = model(images)
        h1.remove()
        h2.remove()
        self.assertEqual(observed, {"input": [2, 3, 384, 384], "last": [2, 4, 7, 7]})
        self.assertEqual(list(result.shape), [2, 2])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 10628)
        result.square().mean().backward()
        self.assertTrue(
            all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        )

    def test_legacy_initialization_and_square_backbone_remain_matched(self):
        import torch
        from simple_network import IncepAttentionCNN_rgb
        from insect_utils.attention_model import create_attention

        torch.manual_seed(10)
        original = IncepAttentionCNN_rgb()
        torch.manual_seed(10)
        legacy = create_attention({})
        self.assertTrue(
            all(torch.equal(a, b) for a, b in zip(original.parameters(), legacy.parameters()))
        )
        torch.manual_seed(10)
        square = create_attention({"network_variant": "nadir384x384"})
        for name, value in original.state_dict().items():
            if not name.startswith("fc."):
                self.assertTrue(torch.equal(value, square.state_dict()[name]), name)
        with self.assertRaises(ValueError):
            create_attention({"network_variant": "unapproved"})

    def test_square_weights_reload_and_onnx_use_the_same_variant(self):
        import torch
        from insect_utils.attention_model import create_attention, load_attention
        from insect_utils.downward_training import export_onnx_reference

        torch.set_num_threads(2)
        config = {"network_variant": "nadir384x384"}
        model = create_attention(config).eval()
        images = torch.rand(2, 3, 1024, 1024)
        protocol = {
            "training": config,
            "delivery": {
                "onnx_opset": 17,
                "onnx_reference_atol": 1e-5,
                "onnx_reference_rtol": 1e-4,
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pth"
            torch.save(model.state_dict(), path)
            with torch.inference_mode():
                self.assertTrue(torch.equal(model(images), load_attention(path, config)(images)))
            result = export_onnx_reference(path, images, Path(folder), protocol)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["network_contract"]["input_hw"], [384, 384])


if __name__ == "__main__":
    unittest.main()
