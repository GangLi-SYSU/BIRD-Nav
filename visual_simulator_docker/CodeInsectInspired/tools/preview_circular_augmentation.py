"""Render actual training-image augmentation with label arrows, without opening test images."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.circular_augmentation import (
    CircularInput,
    circular_mask,
    resize_rgb,
    rotate_nadir,
)
from insect_utils.circular_training import load_circular_data, verify_circular
from insect_utils.experiment_config import digest
from insect_utils.experiment_state import atomic_json, now
from insect_utils.rgb_preprocessing import load_rgb_tensor


def make_preview(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    frozen = verify_circular(folder)
    rows = [
        r for r in load_circular_data(frozen) if r["split"] == "train" and r["heading_index"] == 0
    ]
    selected = []
    for radius in (0, 15, 40):
        selected.append(min(rows, key=lambda r: abs(np.hypot(r["x"], r["y"]) - radius)))
    destination = folder / "preview"
    if destination.exists():
        raise ValueError("Preview already exists; preserve it or use a new run directory")
    destination.mkdir(parents=True)
    angles = [0.0, 22.5, 45.0, 90.0, 137.0]
    figure, axes = plt.subplots(len(selected), len(angles) + 1, figsize=(18, 11))
    receipts = []
    for index, row in enumerate(selected):
        image = load_rgb_tensor(PROJECT / row["image"], [1024, 1024], row["sha256"])
        image = resize_rgb(image[None])
        target = torch.tensor([[row["target_x"], row["target_y"]]], dtype=torch.float32)
        for column, angle in enumerate([None] + angles):
            if angle is None:
                transformed, label = image, target
                shown = transformed
                name = "source_square"
            else:
                transformed, label = rotate_nadir(image, target, [angle])
                shown = CircularInput()(transformed)
                name = "ccw_" + str(angle).replace(".", "p")
            rgb = (shown[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
            filename = row["position_id"] + "_" + name + ".png"
            Image.fromarray(rgb).save(destination / filename)
            axis = axes[index, column]
            axis.imshow(rgb)
            vector = label[0].numpy()
            direction = vector / max(float(np.linalg.norm(vector)), 1e-12)
            # Body +X maps up, body +Y maps left. Arrows show direction, not pixel scale.
            axis.annotate(
                "",
                xy=(191.5 - 65 * direction[1], 191.5 - 65 * direction[0]),
                xytext=(191.5, 191.5),
                arrowprops=dict(color="red", width=2, headwidth=8),
            )
            title = "Original square" if angle is None else "Image CCW {:g} deg".format(angle)
            axis.set_title(title + "\nv=({:.2f}, {:.2f}) m".format(*vector), fontsize=10)
            axis.axis("off")
            receipts.append(
                dict(
                    image=filename,
                    image_sha256=digest(destination / filename),
                    source=row["image"],
                    source_sha256=row["sha256"],
                    position_id=row["position_id"],
                    source_split="train",
                    original_yaw_deg=row["yaw"],
                    image_ccw_deg=angle,
                    virtual_body_yaw_deg=row["yaw"] - (angle or 0),
                    target_body_m=vector.tolist(),
                    target_norm_m=float(np.linalg.norm(vector)),
                )
            )
    figure.suptitle(
        "Actual training inputs | fixed circle | red arrows: home direction (not scale)",
        fontsize=13,
    )
    figure.tight_layout(h_pad=2.5, rect=(0, 0, 1, 0.96))
    figure.savefig(destination / "augmentation_grid.png", dpi=130)
    plt.close(figure)
    summary = dict(
        status="completed",
        updated=now(),
        fingerprint=frozen["fingerprint"],
        source_split="train only",
        test_images_opened=0,
        training_updates=0,
        mask_retained_fraction=float(circular_mask().mean()),
        views=receipts,
        grid="preview/augmentation_grid.png",
    )
    atomic_json(destination / "completion.json", summary)
    return summary
