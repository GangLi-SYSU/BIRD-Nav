"""Physical output adapter and complete-frame dataset for the normalization control."""

import math

import torch
from torch import nn
from torch.utils.data import Dataset

from insect_utils.coverage_plan import PARTITIONS
from insect_utils.experiment_config import PROJECT
from insect_utils.rgb_preprocessing import load_rgb_tensor


class CoverageDataset(Dataset):
    def __init__(self, records, split, scale=1.0, allow_test=False):
        if split not in PARTITIONS or ("test" in split and not allow_test):
            raise ValueError("Test data is restricted to final evaluation")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Target scale must be finite and positive")
        self.records = [r for r in records if r["split"] == split]
        self.scale = float(scale)
        if not self.records:
            raise ValueError("Empty dataset partition")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image = load_rgb_tensor(PROJECT / row["image"], [1024, 1024], row["sha256"])
        target = torch.tensor([row["target_x"], row["target_y"]], dtype=torch.float32)
        return image, target / self.scale


class PhysicalHomeVector(nn.Module):
    """Restore meters inside the deployment graph, with no new trainable parameters."""

    def __init__(self, network, scale):
        super().__init__()
        self.network = network
        self.scale = float(scale)

    def forward(self, rgb):
        return self.network(rgb) * self.scale
