"""RGB and public loader/evaluation contracts; run with the Isaac Python packages."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.experiment_config import digest
from insect_utils.rgb_preprocessing import rgb_array, rgb_tensor

HAS_TORCH = all(importlib.util.find_spec(name) for name in ("torch", "torchvision", "numpy", "PIL"))


@unittest.skipUnless(HAS_TORCH, "Runtime tests need torch, torchvision, NumPy and Pillow")
class DownwardRuntimeTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        from test_stage06_data import example_records

        self.tmp = tempfile.TemporaryDirectory(dir=str(PROJECT / "runs"))
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        (self.folder / "Replicator/rgb").mkdir(parents=True)
        records = example_records(3)
        for i, row in enumerate(records):
            Image.new("RGB", (8, 8), (i + 1, 128, 255 - i)).save(self.folder / row["image"])
            row["sha256"] = digest(self.folder / row["image"])
        self.context = {
            "folder": self.folder,
            "capture_folder": self.folder,
            "records": records,
            "capture_config": {
                "spec": {"camera": {"raw_resolution_wh": [8, 8]}, "home": {"ground_xy_m": [12, -8]}}
            },
            "capture_manifest": {"headings_per_position": 8},
            "split": {
                "seed": 10,
                "ratio": 0.95,
                "indices": {
                    "train": list(range(8)),
                    "validation": list(range(8, 16)),
                    "test": list(range(16, 24)),
                },
            },
        }
        self.cfg = {
            "camera_mode": "downward_pinhole",
            "prepared_data_dir": str(self.folder),
            "augment": False,
            "seed": 10,
            "split_ratio": 0.95,
            "batch_size": 8,
            "batch_size_val": 8,
            "batch_size_eval": 8,
            "num_workers": 0,
            "dataset_folder": str(self.folder),
            "image_folder": str(self.folder / "Replicator/rgb"),
            "csv_filename": "dataset_navigation.csv",
        }

    def test_rgb_rgba_totensor_scaling_and_channel_order_match(self):
        import numpy as np
        import torch
        from torchvision.transforms import functional

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[:] = [255, 128, 0]
        rgba = np.concatenate([rgb, np.zeros((8, 8, 1), dtype=np.uint8)], axis=2)
        tensor = rgb_tensor(rgb)
        self.assertTrue(torch.equal(tensor, functional.to_tensor(rgb)))
        self.assertTrue(torch.equal(tensor, rgb_tensor(rgba)))
        self.assertEqual(list(tensor.shape), [3, 8, 8])
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor[0, 0, 0].item(), 1.0)
        self.assertAlmostEqual(tensor[1, 0, 0].item(), 128 / 255, places=7)
        self.assertEqual(tensor[2, 0, 0].item(), 0.0)

    def test_wrong_dtype_channels_or_dimensions_are_rejected(self):
        import numpy as np

        for image in [
            np.zeros((8, 8, 3), dtype=float),
            np.zeros((8, 8), dtype=np.uint8),
            np.zeros((8, 8, 2), dtype=np.uint8),
        ]:
            with self.assertRaises(ValueError):
                rgb_array(image)
        with self.assertRaises(ValueError):
            rgb_array(np.zeros((8, 8, 3), dtype=np.uint8), [1024, 1024])

    def test_dataset_does_not_crop_resize_or_change_labels(self):
        import torch
        from insect_utils.downward_data import DownwardDataset

        dataset = DownwardDataset(self.context, "validation", self.cfg)
        image, target = dataset[0]
        row = dataset.record(0)
        self.assertEqual(list(image.shape), [3, 8, 8])
        self.assertTrue(
            torch.equal(
                target, torch.tensor([row["target_x"], row["target_y"]], dtype=torch.float32)
            )
        )
        self.assertTrue(torch.equal(image, dataset[0][0]))

    def test_changed_png_is_rejected_at_read_time(self):
        from PIL import Image
        from insect_utils.downward_data import DownwardDataset

        dataset = DownwardDataset(self.context, "test", self.cfg)
        Image.new("RGB", (8, 8)).save(self.folder / dataset.record(0)["image"])
        with self.assertRaisesRegex(ValueError, "PNG hash changed"):
            dataset[0]

    def test_public_loader_uses_independent_datasets_and_ignores_legacy_random_split(self):
        import data_loader

        with patch("insect_utils.downward_data.load_prepared", return_value=self.context), patch(
            "data_loader.CustomDataset", side_effect=AssertionError("legacy dataset called")
        ):
            loaded = data_loader.get_data_loaders(self.cfg, load_indices=False)
        self.assertEqual(len({id(loader.dataset) for loader in loaded[:3]}), 3)
        self.assertEqual([len(loader.dataset) for loader in loaded[:3]], [8, 8, 8])

    def test_evaluation_is_repeatable_and_rejects_panorama_augmentation(self):
        import numpy as np
        import torch
        from insect_utils.train_utils import evaluate_on_dataset

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(2))

            def forward(self, x):
                return x.mean(dim=(2, 3))[:, :2] + self.bias

        model = Model()
        args = (
            model,
            self.cfg["dataset_folder"],
            self.cfg["image_folder"],
            "dataset_navigation.csv",
        )
        with patch("insect_utils.downward_data.load_prepared", return_value=self.context), patch(
            "insect_utils.train_utils.augment_image_vector",
            side_effect=AssertionError("panorama wrap called"),
        ):
            a = evaluate_on_dataset(*args, config=self.cfg, evaluation_set="validation")
            b = evaluate_on_dataset(*args, config=self.cfg, evaluation_set="validation")
            self.assertTrue(np.array_equal(a[0], b[0]))
            with self.assertRaises(ValueError):
                evaluate_on_dataset(*args, config=self.cfg, augment=True)

    def test_online_wrapper_uses_shared_preprocessing_without_loading_kit(self):
        import numpy as np
        import torch
        import virtual_homing_ANN

        rgb = np.full((8, 8, 3), 37, dtype=np.uint8)
        rgba = np.concatenate([rgb, np.zeros((8, 8, 1), dtype=np.uint8)], axis=2)
        self.assertTrue(
            torch.equal(virtual_homing_ANN.downward_rgb_tensor(rgba)[0], rgb_tensor(rgb))
        )
        self.assertFalse(any(n == "omni" or n.startswith("omni.") for n in sys.modules))


if __name__ == "__main__":
    unittest.main()
