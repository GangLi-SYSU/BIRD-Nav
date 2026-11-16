"""Nadir rotation geometry, missing-corner isolation, split safety and deployment parity."""

from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.camera_geometry import home_vector_body, optics, world_to_pixel
from insect_utils.circular_augmentation import (
    CircularAttention,
    CircularDataset,
    CircularInput,
    circular_mask,
    make_circular_loader,
    rotate_nadir,
)
from insect_utils.coverage_model import CoverageDataset, PhysicalHomeVector
from insect_utils.downward_training import restore_rng, rng_snapshot, seed_all, train_epochs


class CircularGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_ccw_image_moves_forward_point_left_and_rotates_label(self):
        image = torch.zeros(1, 3, 384, 384)
        image[:, :, 100, 191] = 1
        rotated, target = rotate_nadir(image, torch.tensor([[10.0, 0.0]]), [90.0])
        self.assertTrue(torch.allclose(rotated, torch.rot90(image, 1, (2, 3)), atol=3e-5))
        self.assertTrue(torch.allclose(target, torch.tensor([[0.0, 10.0]]), atol=1e-5))

    def test_labels_reconstruct_same_world_home_for_arbitrary_angles(self):
        position, home = [3.0, -7.0, 40.0], [0.0, 0.0, 40.0]
        angles = [0, 1.7, 22.5, 45, 90, 179, 270, 359.9]
        for yaw in (0, 45, 180, 315):
            target = torch.tensor([home_vector_body(position, home, yaw)])
            for angle in angles:
                _, actual = rotate_nadir(torch.zeros(1, 3, 384, 384), target, [angle])
                expected = torch.tensor([home_vector_body(position, home, yaw - angle)])
                self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_rotation_matches_current_isaac_projection_at_nonzero_yaw(self):
        settings = optics(
            dict(
                mode="downward_pinhole",
                crop_box=None,
                fov_definition_confirmed=True,
                raw_resolution_wh=[384, 384],
                hfov_deg=70,
            )
        )
        axis = torch.arange(384, dtype=torch.float32)
        for height in (0, 12):
            # Projection coordinates measure from the image edge; tensor pixel
            # centres are indexed by integers, hence the half-pixel conversion.
            pixel = np.asarray(world_to_pixel([5, -2, height], [0, 0, 40], 73, settings)) - 0.5
            blob = torch.exp(
                -((axis[None, :] - pixel[0]).square() + (axis[:, None] - pixel[1]).square()) / 12.5
            )
            image = blob[None, None].expand(1, 3, 384, 384).contiguous()
            for angle in (12.3, 45, 90, 225):
                expected = (
                    np.asarray(world_to_pixel([5, -2, height], [0, 0, 40], 73 - angle, settings))
                    - 0.5
                )
                rotated, _ = rotate_nadir(image, torch.zeros(1, 2), [angle])
                weights = rotated[0, 0]
                observed = np.array(
                    [
                        float((weights * axis[None, :]).sum() / weights.sum()),
                        float((weights * axis[:, None]).sum() / weights.sum()),
                    ]
                )
                self.assertTrue(np.allclose(observed, expected, atol=0.02))

    def test_rotation_preserves_norm_and_zero_home_vector(self):
        values = torch.tensor([[3.0, 4.0], [0.0, 0.0], [-0.1, 0.2]])
        _, actual = rotate_nadir(torch.zeros(3, 3, 384, 384), values, [27.3, 52.7, 312.1])
        self.assertTrue(torch.allclose(values.norm(dim=1), actual.norm(dim=1), atol=1e-6))
        self.assertTrue(torch.equal(actual[1], values[1]))

    def test_mask_removes_all_unsupported_rotation_corners(self):
        source = torch.ones(1, 3, 384, 384)
        mask = circular_mask().expand_as(source)
        for angle in np.linspace(0, 360, 73):
            rotated, _ = rotate_nadir(source, torch.zeros(1, 2), [float(angle)])
            actual = CircularInput()(rotated)
            self.assertTrue(torch.allclose(actual, mask, atol=2e-5))
            self.assertTrue(torch.equal(actual * (1 - mask), torch.zeros_like(actual)))

    def test_bad_shapes_and_nonfinite_labels_or_angles_are_rejected(self):
        images, target = torch.zeros(1, 3, 384, 384), torch.ones(1, 2)
        for labels, angles in (
            (target, [float("nan")]),
            (target, []),
            (target * float("nan"), [20]),
        ):
            with self.assertRaises(ValueError):
                rotate_nadir(images, labels, angles)


class CircularDatasetTests(unittest.TestCase):
    def fake_read(self, _index):
        image = torch.zeros(3, 1024, 1024)
        image[:, 256:320, 480:544] = 1
        return image, torch.tensor([0.25, -0.125])

    def rows(self):
        return [
            dict(split=name, position_id=name)
            for name in ("train", "validation", "legacy_test", "new_test")
        ]

    def test_validation_is_deterministic_and_does_not_draw_rotation_rng(self):
        dataset = CircularDataset(self.rows(), "validation")
        with patch.object(CoverageDataset, "__getitem__", self.fake_read):
            with patch("numpy.random.uniform", side_effect=AssertionError("random eval")):
                first, label = dataset[0]
                second, other = dataset[0]
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(label, other))

    def test_random_rotation_rejected_on_every_holdout_and_test_access_guarded(self):
        for split in ("validation", "legacy_test", "new_test"):
            with self.assertRaises(ValueError):
                CircularDataset(self.rows(), split, random_rotation=True, allow_test=True)
        with self.assertRaises(ValueError):
            CircularDataset(self.rows(), "new_test")

    def test_online_augmentation_does_not_expand_or_move_position_groups(self):
        dataset = CircularDataset(self.rows(), "train", random_rotation=True)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.records, [self.rows()[0]])
        with patch.object(CoverageDataset, "__getitem__", self.fake_read):
            with patch("numpy.random.uniform", return_value=90.0):
                _, target = dataset[0]
        self.assertTrue(torch.allclose(target, torch.tensor([0.125, 0.25]), atol=1e-6))

    def test_checkpoint_rng_reproduces_next_rotation_and_label(self):
        dataset = CircularDataset(self.rows(), "train", random_rotation=True)
        loader = type("Loader", (), {"generator": torch.Generator().manual_seed(10)})()
        random.seed(10)
        np.random.seed(10)
        state = rng_snapshot([loader], False)
        with patch.object(CoverageDataset, "__getitem__", self.fake_read):
            first = dataset[0]
            restore_rng(state, [loader])
            second = dataset[0]
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_worker_generator_restore_reproduces_next_epoch_augmentations(self):
        rows = [dict(split="train", position_id="train_" + str(i)) for i in range(4)]
        config = dict(seed=10, scale_m=44, batch_size=2, num_workers=2)
        loader = make_circular_loader(rows, "train", config, random_rotation=True)
        with patch.object(CoverageDataset, "__getitem__", self.fake_read):
            list(loader)
            state = rng_snapshot([loader], False)
            first = list(loader)
            restore_rng(state, [loader])
            second = list(loader)
        self.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a[0], b[0]))
            self.assertTrue(torch.equal(a[1], b[1]))

    def test_resumed_rotation_training_matches_uninterrupted_parameters(self):
        config = dict(
            seed=10,
            scale_m=44,
            batch_size=2,
            num_workers=0,
            network_variant="nadir384x384",
            epochs=2,
            validation_interval_updates=0,
        )
        rows = [dict(split="train", position_id="train_" + str(i)) for i in range(4)]
        rows += [dict(split="validation", position_id="validation_" + str(i)) for i in range(2)]

        def run(folder, resume=False, stop=None):
            seed_all(10)
            model = CircularAttention(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
            train = make_circular_loader(rows, "train", config, random_rotation=True)
            val = make_circular_loader(rows, "validation", config)
            train_epochs(
                model,
                optimizer,
                train,
                val,
                config,
                folder,
                "circular-resume-fixture",
                resume=resume,
                stop_after_epoch=stop,
            )
            return model.state_dict()

        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(CoverageDataset, "__getitem__", self.fake_read):
                full = run(Path(temporary) / "full")
                run(Path(temporary) / "resumed", stop=1)
                resumed = run(Path(temporary) / "resumed", resume=True)
        self.assertTrue(all(torch.equal(full[name], resumed[name]) for name in full))


class CircularDeploymentTests(unittest.TestCase):
    def test_original_parameter_count_and_initial_values_unchanged(self):
        from insect_utils.attention_model import create_attention

        config = {"network_variant": "nadir384x384"}
        torch.manual_seed(10)
        old = create_attention(config)
        torch.manual_seed(10)
        new = CircularAttention(config)
        self.assertEqual(sum(p.numel() for p in new.parameters()), 10628)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(old.parameters(), new.parameters())))

    def test_deployment_owns_mask_and_restores_meters_once(self):
        config = {"network_variant": "nadir384x384"}
        circular = CircularAttention(config).eval()
        raw = torch.rand(2, 3, 1024, 1024)
        with torch.inference_mode():
            expected = circular.network(circular.preprocess(raw)) * 44
            actual = PhysicalHomeVector(circular, 44)(raw)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(circular.preprocess(raw)[:, :, 0, 0], torch.zeros(2, 3)))

    def test_saved_torchscript_masks_raw_rgb_and_handles_batch_sizes(self):
        model = PhysicalHomeVector(CircularAttention({"network_variant": "nadir384x384"}), 44)
        model.eval()
        raw = torch.rand(2, 3, 1024, 1024)
        with tempfile.TemporaryDirectory() as temporary, torch.inference_mode():
            scripted = torch.jit.trace(model, raw[:1], check_inputs=[(raw,)])
            path = str(Path(temporary) / "circular.pt")
            scripted.save(path)
            restored = torch.jit.load(path)
            self.assertTrue(torch.allclose(restored(raw), model(raw), atol=1e-5, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
