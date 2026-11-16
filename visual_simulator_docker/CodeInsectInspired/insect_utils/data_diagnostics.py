"""Scientific plots for grouped splits and downward label geometry."""

import math

from insect_utils.camera_geometry import body_to_world
from insect_utils.rgb_preprocessing import body_direction_image


def plot_split_map(records, split, destination):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9))
    first = {}
    for row in records:
        first.setdefault(row["position_id"], row)
    for name, color, marker in [
        ("train", "#3979b7", "."),
        ("validation", "#f49e29", "s"),
        ("test", "#329844", "^"),
    ]:
        rows = [first[g] for g in split["groups"][name]]
        ax.scatter(
            [r["x"] for r in rows],
            [r["y"] for r in rows],
            c=color,
            marker=marker,
            label="{}: {} positions / {} images".format(
                name, len(rows), len(split["indices"][name])
            ),
        )
    ax.set(
        xlabel="World X (m)", ylabel="World Y (m)", title="Position-group split (no spatial buffer)"
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def plot_world_vectors(records, home, destination, predictions=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9))
    for index, row in enumerate(records):
        vector = body_to_world([row["target_x"], row["target_y"]], row["yaw"])
        ax.quiver(
            row["x"],
            row["y"],
            *vector,
            color="#30934b",
            angles="xy",
            scale_units="xy",
            scale=1,
            alpha=0.45,
        )
        if predictions is not None:
            world = body_to_world([float(v) for v in predictions[index]], row["yaw"])
            ax.quiver(
                row["x"],
                row["y"],
                *world,
                color="#347bb5",
                angles="xy",
                scale_units="xy",
                scale=1,
                alpha=0.6,
            )
    ax.scatter(*home, marker="*", s=160, c="#edb62f", edgecolors="k", label="Configured home")
    ax.scatter([r["x"] for r in records], [r["y"] for r in records], s=8, c="k")
    ax.set(
        xlabel="World X (m)",
        ylabel="World Y (m)",
        title="World-frame home vectors (green); predictions blue if supplied",
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def plot_label_images(dataset, indices, destination, model=None):
    import matplotlib.pyplot as plt
    import torch

    if not indices:
        raise ValueError("No diagnostic examples requested")
    columns = min(4, len(indices))
    rows = math.ceil(len(indices) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 4.3), squeeze=False)
    for ax, index in zip(axes.ravel(), indices):
        image, target = dataset[index]
        row = dataset.record(index)
        ax.imshow(image.permute(1, 2, 0).numpy())
        height, width = image.shape[1:]
        length = min(width, height) * 0.3
        direction = body_direction_image(target.tolist())
        if direction is not None:
            ax.arrow(
                width / 2,
                height / 2,
                direction[0] * length,
                direction[1] * length,
                color="#16d935",
                width=width * 0.008,
                length_includes_head=True,
            )
        else:
            ax.plot(width / 2, height / 2, "go")
        if model is not None:
            model.eval()
            with torch.inference_mode():
                prediction = (
                    model(image.unsqueeze(0).to(next(model.parameters()).device))[0].cpu().tolist()
                )
            direction = body_direction_image(prediction)
            if direction is not None:
                ax.arrow(
                    width / 2,
                    height / 2,
                    direction[0] * length,
                    direction[1] * length,
                    color="#30a9ff",
                    width=width * 0.005,
                    length_includes_head=True,
                )
        ax.set_title(
            "yaw={:.1f} deg; body target=({:.2f},{:.2f}) m".format(row["yaw"], *target.tolist()),
            fontsize=9,
        )
        ax.set_xlabel(row["sample_id"] + " | direction only; home may be outside view", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(indices) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)
    return str(destination)


def plot_network_resize(image, destination):
    import matplotlib.pyplot as plt
    import torch.nn.functional as functional

    resized = functional.interpolate(
        image.unsqueeze(0), size=(192, 1800), mode="bilinear", align_corners=False
    )[0]
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [4, 1]})
    axes[0].imshow(image.permute(1, 2, 0).numpy())
    axes[0].set_title("Raw full frame: 1024 x 1024 (camera FOV unchanged)")
    axes[1].imshow(resized.permute(1, 2, 0).numpy())
    axes[1].set_title("Original network's internal bilinear resize: width1800 x height192")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
