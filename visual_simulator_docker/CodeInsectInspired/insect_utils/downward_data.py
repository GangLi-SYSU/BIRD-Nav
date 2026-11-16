"""Verified downward datasets and deterministic evaluation without Isaac/Kit."""

from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from insect_utils.data_preparation import load_prepared, project_path
from insect_utils.rgb_preprocessing import load_rgb_tensor, require_downward_config


class DownwardDataset(Dataset):
    """One fixed split, complete RGB and no image augmentation."""

    def __init__(self, context, split, config):
        require_downward_config(config)
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown downward dataset split")
        self.context, self.split, self.config = context, split, dict(config)
        self.indices = list(context["split"]["indices"][split])
        self.augmentation = False
        self.resolution = context["capture_config"]["spec"]["camera"]["raw_resolution_wh"]

    def __len__(self):
        return len(self.indices)

    def record(self, index):
        return self.context["records"][self.indices[index]]

    def __getitem__(self, index):
        row = self.record(index)
        image = load_rgb_tensor(
            self.context["capture_folder"] / row["image"], self.resolution, row["sha256"]
        )
        target = torch.tensor([row["target_x"], row["target_y"]], dtype=torch.float32)
        return image, target


def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def get_prepared_loaders(config):
    require_downward_config(config)
    context = load_prepared(config["prepared_data_dir"])
    split = context["split"]
    if config.get("seed", 10) != split["seed"] or config["split_ratio"] != split["ratio"]:
        raise ValueError("Requested seed/split ratio differs from the frozen preparation")
    loaders = []
    for name, batch_key in [
        ("train", "batch_size"),
        ("validation", "batch_size_val"),
        ("test", "batch_size_eval"),
    ]:
        dataset = DownwardDataset(context, name, config)
        loaders.append(
            DataLoader(
                dataset,
                batch_size=config[batch_key],
                shuffle=name == "train",
                num_workers=config.get("num_workers", 4),
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(split["seed"]),
            )
        )
    rows = context["records"]
    positions = {Path(r["image"]).name: np.array([r["x"], r["y"], r["z"]]) for r in rows}
    rotations = {Path(r["image"]).name: np.array([0, 0, r["yaw"]]) for r in rows}
    targets = {Path(r["image"]).name: np.array([r["target_x"], r["target_y"]]) for r in rows}
    return (
        *loaders,
        positions,
        rotations,
        None,
        targets,
        split["indices"]["train"],
        split["indices"]["validation"],
        split["indices"]["test"],
    )


def evaluate_prepared(model, config, split="test", graphics=False, limit=None):
    """Evaluate a frozen subset; return the legacy five-value evaluation result."""
    require_downward_config(config)
    name = "validation" if split == "val" else split
    loaded = get_prepared_loaders(config)
    number = {"train": 0, "validation": 1, "test": 2}
    if name not in number:
        raise ValueError("Unknown evaluation split")
    # Evaluation uses the fixed dataset order, including when inspecting training.
    dataset = loaded[number[name]].dataset
    count = len(dataset) if limit is None else min(limit, len(dataset))
    if count < 1:
        raise ValueError("Evaluation subset is empty")
    loader = DataLoader(
        torch.utils.data.Subset(dataset, list(range(count))),
        batch_size=config["batch_size_eval"],
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config.get("seed", 10)),
    )
    model.eval()
    device = next(model.parameters()).device
    predictions, targets = [], []
    with torch.inference_mode():
        for images, labels in loader:
            output = model(images.to(device))
            if not torch.isfinite(output).all():
                raise ValueError("Non-finite evaluation prediction")
            predictions.append(output.cpu())
            targets.append(labels)
    prediction = torch.cat(predictions).numpy()
    target = torch.cat(targets).numpy()
    rows = [dataset.record(i) for i in range(count)]
    positions = {Path(r["image"]).name: np.array([r["x"], r["y"], r["z"]]) for r in rows}
    rotations = {Path(r["image"]).name: np.array([0, 0, r["yaw"]]) for r in rows}
    if graphics:
        from insect_utils.data_diagnostics import plot_world_vectors

        folder = Path(config.get("diagnostic_dir", dataset.context["folder"] / "inspection"))
        folder.mkdir(parents=True, exist_ok=True)
        plot_world_vectors(
            rows,
            dataset.context["capture_config"]["spec"]["home"]["ground_xy_m"],
            folder / ("evaluation_" + name + ".png"),
            prediction,
        )
    return prediction, target, positions, rotations, None


def evaluate_dataset_arguments(
    model, data_folder, image_folder, csv_filename, config, evaluation_set, augment, graphics
):
    require_downward_config(config)
    if augment:
        raise ValueError("Panorama wrap evaluation is invalid for downward images")
    context = load_prepared(config["prepared_data_dir"])
    expected = context["capture_folder"].resolve()
    if (
        project_path(data_folder).resolve() != expected
        or project_path(image_folder).resolve() != expected / "Replicator/rgb"
        or csv_filename != "dataset_navigation.csv"
    ):
        raise ValueError("External data needs its own verified stage F preparation")
    return evaluate_prepared(model, config, evaluation_set, graphics)


def inspect_prepared(config, n_examples=10, model=None):
    from insect_utils.data_diagnostics import plot_label_images

    loaded = get_prepared_loaders(config)
    dataset = loaded[0].dataset
    folder = Path(config.get("diagnostic_dir", dataset.context["folder"] / "inspection"))
    folder.mkdir(parents=True, exist_ok=True)
    # Spread examples over distinct positions instead of showing eight near-duplicates.
    examples = list(
        range(0, len(dataset), dataset.context["capture_manifest"]["headings_per_position"])
    )[:n_examples]
    return plot_label_images(dataset, examples, folder / "label_directions.png", model)


def evaluate_checkpoint(filename, config):
    from simple_network import IncepAttentionCNN_rgb

    require_downward_config(config)
    if config["model_type"] != "attention":
        raise ValueError("Downward experiments preserve the original attention model")
    model = IncepAttentionCNN_rgb().eval()
    model.load_state_dict(torch.load(filename, map_location="cpu", weights_only=True))
    result = None
    for name in ["train", "validation", "test"]:
        result = evaluate_prepared(model, config, name)
    return result
