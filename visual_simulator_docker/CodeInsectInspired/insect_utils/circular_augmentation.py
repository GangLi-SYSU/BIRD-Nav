"""Nadir yaw augmentation with coupled labels and one deployment-safe circular mask."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from insect_utils.attention_model import create_attention
from insect_utils.coverage_model import CoverageDataset
from insect_utils.downward_data import seed_worker


INPUT_SIZE = 384
# Pixel centres are 0..383; one pixel of support remains outside the retained disk.
MASK_RADIUS_PX = 190.0


def resize_rgb(images):
    return functional.interpolate(
        images, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False
    )


def circular_mask():
    coordinates = torch.arange(INPUT_SIZE, dtype=torch.float32) - (INPUT_SIZE - 1) / 2
    squared_radius = coordinates[:, None].square() + coordinates[None, :].square()
    return (squared_radius <= MASK_RADIUS_PX ** 2).float()[None, None]


def rotate_nadir(images, targets, angles_deg):
    """Rotate displayed images CCW and body-frame (forward, left) vectors by +angle.

    A positive image rotation represents virtual body yaw decreasing by that angle.
    The affine grid is an inverse map in screen (right, down) coordinates. Keep the
    square scale and centre; apply the fixed disk afterwards, inside the model.
    """
    if images.ndim != 4 or images.shape[1:] != (3, INPUT_SIZE, INPUT_SIZE):
        raise ValueError("Rotation expects NCHW RGB already resized to 384 square")
    if targets.shape != (len(images), 2) or not torch.isfinite(targets).all():
        raise ValueError("Expected finite paired two-component body vectors")
    angles = torch.as_tensor(angles_deg, dtype=images.dtype, device=images.device)
    if angles.shape != (len(images),) or not torch.isfinite(angles).all():
        raise ValueError("Expected one finite rotation angle per image")
    if targets.device != images.device:
        raise ValueError("Images and targets must share a device")
    radians = angles * (math.pi / 180)
    cosine, sine = radians.cos(), radians.sin()
    zero = torch.zeros_like(cosine)
    affine = torch.stack((cosine, -sine, zero, sine, cosine, zero), dim=1).reshape(-1, 2, 3)
    grid = functional.affine_grid(affine, images.shape, align_corners=False)
    rotated = functional.grid_sample(
        images, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    forward, left = targets[:, 0], targets[:, 1]
    vectors = torch.stack((cosine * forward - sine * left, sine * forward + cosine * left), 1)
    return rotated, vectors


class CircularInput(nn.Module):
    """Identical resize/mask for training, validation, TorchScript and ONNX."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mask", circular_mask(), persistent=False)

    def forward(self, rgb):
        return resize_rgb(rgb) * self.mask


class CircularAttention(nn.Module):
    """Keep all 10,628 original trainable parameters; add only fixed preprocessing."""

    def __init__(self, config):
        super().__init__()
        if config["network_variant"] != "nadir384x384":
            raise ValueError("Circular preprocessing is for the square nadir network only")
        self.preprocess = CircularInput()
        self.network = create_attention(config)

    def forward(self, rgb):
        return self.network(self.preprocess(rgb))


class CircularDataset(CoverageDataset):
    """Online rotation only on training records; preserve positions and split membership."""

    def __init__(self, records, split, scale=44.0, random_rotation=False, allow_test=False):
        if random_rotation and split != "train":
            raise ValueError("Random rotation must never run on validation or test data")
        super().__init__(records, split, scale, allow_test=allow_test)
        self.random_rotation = random_rotation

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        # Both arms use exactly this resize. The model always owns the final mask.
        image = resize_rgb(image[None])[0]
        if self.random_rotation:
            angle = float(np.random.uniform(0.0, 360.0))
            images, targets = rotate_nadir(image[None], target[None], [angle])
            image, target = images[0], targets[0]
        return image, target


def make_circular_loader(records, split, config, random_rotation=False, allow_test=False):
    dataset = CircularDataset(
        records, split, config["scale_m"], random_rotation, allow_test=allow_test
    )
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=split == "train",
        num_workers=config["num_workers"],
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
