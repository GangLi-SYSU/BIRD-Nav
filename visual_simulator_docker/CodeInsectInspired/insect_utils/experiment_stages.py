"""Worker adapters. Heavy simulation imports occur only in Isaac stages."""

import csv
import math
import os
from pathlib import Path
import re
import sys

from insect_utils.experiment_config import preflight, resolve_spec, stage_parameters
from insect_utils.experiment_state import atomic_json, collect_artifacts, complete_stage, read_json
from insect_utils.experiment_state import stage_record, verify_completed, verify_run


def vector(text, length):
    values = [float(v) for v in re.split(r"[,\s]+", text.strip("[]() \t")) if v]
    if len(values) != length or not all(math.isfinite(v) for v in values):
        raise ValueError("Invalid dataset position/rotation/target")
    return values


def synthetic_capture(resolved, folder):
    """Explicit integration fixture; these images are not observations of a scene."""
    import numpy as np
    from PIL import Image
    from insect_utils.flight_path_functions import calculate_target_vectors, generate_bee_path

    spec, derived = resolved["spec"], resolved["derived"]
    positions = generate_bee_path(*spec["capture"]["shape_params"], derived["home_camera_xyz_m"])
    rotations = [[0.0, 0.0, float((i * 37) % 360)] for i in range(len(positions))]
    targets = calculate_target_vectors(positions, rotations, derived["home_camera_xyz_m"])
    image_dir = folder / "Replicator/rgb"
    image_dir.mkdir(parents=True)
    width, height = derived["processed_resolution_wh"]
    yy, xx = np.indices((height, width))
    with (folder / "dataset_navigation.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Filename", "Position", "Rotation", "Target"])
        for index, (position, rotation, target) in enumerate(zip(positions, rotations, targets)):
            image = np.stack(
                (
                    (xx // 4 + index * 17) % 256,
                    (yy // 2 + index * 29) % 256,
                    ((xx + yy) // 4 + index * 7) % 256,
                ),
                axis=2,
            ).astype(np.uint8)
            filename = "rgb_{:04d}.png".format(index)
            Image.fromarray(image).save(image_dir / filename)
            writer.writerow([filename, tuple(position), tuple(rotation), "[{} {}]".format(*target)])


def seal_capture(resolved, run_dir):
    if (
        resolved["spec"]["camera"]["mode"] == "downward_pinhole"
        and resolved["spec"]["capture"]["backend"] == "isaac"
    ):
        from insect_utils.downward_capture import seal_dataset

        return seal_dataset(resolved, run_dir, stage_parameters(resolved, run_dir)["capture"])
    from PIL import Image

    folder = run_dir / "capture"
    with (folder / "dataset_navigation.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["Filename", "Position", "Rotation", "Target"]:
            raise ValueError("Expected the no-noise legacy four-column dataset schema")
        rows = list(reader)
    if len(rows) != resolved["derived"]["expected_samples"]:
        raise ValueError("Dataset count differs from the declared sample count policy")
    names = set()
    for row in rows:
        name = row["Filename"]
        if Path(name).name != name or name in names:
            raise ValueError("Duplicate or unsafe image filename in dataset")
        names.add(name)
        position = vector(row["Position"], 3)
        rotation = vector(row["Rotation"], 3)
        target = vector(row["Target"], 2)
        yaw = math.radians(rotation[2])
        home = [
            position[0] + math.cos(yaw) * target[0] - math.sin(yaw) * target[1],
            position[1] + math.sin(yaw) * target[0] + math.cos(yaw) * target[1],
        ]
        if any(abs(a - b) > 1e-5 for a, b in zip(home, resolved["spec"]["home"]["ground_xy_m"])):
            raise ValueError("Dataset label does not reconstruct configured home")
        if abs(position[2] - resolved["derived"]["home_camera_xyz_m"][2]) > 1e-6:
            raise ValueError("Capture height differs from the unified flight height")
        with Image.open(folder / "Replicator/rgb" / name) as image:
            if list(image.size) != resolved["derived"]["processed_resolution_wh"]:
                raise ValueError("Unexpected processed image dimensions")
            image.verify()
    if not rows:
        raise ValueError("Dataset is empty")
    atomic_json(
        folder / "dataset_manifest.json",
        {
            "fingerprint": resolved["fingerprint"],
            "purpose": resolved["spec"]["purpose"],
            "producer": resolved["spec"]["capture"]["backend"],
            "sample_count": len(rows),
            "planned_samples": resolved["derived"]["planned_samples"],
            "camera_mode": resolved["spec"]["camera"]["mode"],
            "count_policy": resolved["spec"]["capture"]["sample_count_policy"],
            "files": collect_artifacts(folder, run_dir),
        },
    )


def train_stage(resolved, run_dir, parameters):
    import numpy as np
    import torch
    import train
    from simple_network import IncepAttentionCNN_rgb

    capture = verify_completed(run_dir, "capture", resolved["fingerprint"])
    manifest = read_json(run_dir / "capture/dataset_manifest.json")
    if (
        manifest["fingerprint"] != resolved["fingerprint"]
        or manifest["purpose"] != resolved["spec"]["purpose"]
    ):
        raise ValueError("Capture manifest does not match this experiment")
    size = manifest["sample_count"]
    train_size = int(parameters["split_ratio"] * size)
    val_size = int((1 - parameters["split_ratio"]) * train_size)
    if min(train_size - val_size, val_size, size - train_size) < 1:
        raise ValueError("Dataset is too small for the declared legacy train/validation/test split")
    filename = train.train_model(parameters, only_training=True)
    model = IncepAttentionCNN_rgb().eval()
    model.load_state_dict(torch.load(filename, map_location="cpu", weights_only=True))
    if sum(p.numel() for p in model.parameters()) != 10820:
        raise ValueError("Network contract changed")
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Saved model contains non-finite parameters")
    history = read_json(run_dir / "train/training_history.json")
    if not history["optimizer_steps"] or not history["parameters_changed"]:
        raise ValueError("Training smoke did not update the network")
    kit_modules = [name for name in sys.modules if name == "omni" or name.startswith("omni.")]
    if kit_modules:
        raise ValueError("Training process unexpectedly imported Isaac/Kit")
    atomic_json(
        run_dir / "train/summary.json",
        {
            "purpose": resolved["spec"]["purpose"],
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "network_parameters": 10820,
            "optimizer_steps": history["optimizer_steps"],
            "epochs_completed": history["epochs_completed"],
            "kit_modules": kit_modules,
            "capture_artifact_count": len(capture["artifacts"]),
            "note": "Engineering fixtures do not measure visual homing ability."
            if resolved["spec"]["purpose"] == "engineering_smoke"
            else None,
        },
    )


def evaluate_stage(resolved, run_dir, parameters, app):
    import numpy as np
    import virtual_homing_ANN

    result = virtual_homing_ANN.main(
        parameters, verbose=True, graphics=False, simulation_app=app, network_type="attention"
    )
    names = (
        "successful_runs",
        "total_distances",
        "straight_line_distances",
        "angular_errors",
        "distance_errors",
        "radius_scales",
        "success_rates",
        "trajectories",
        "successes",
    )

    def plain(value):
        if isinstance(value, np.ndarray):
            return plain(value.tolist())
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    atomic_json(
        run_dir / "evaluate/results.json", {key: plain(value) for key, value in zip(names, result)}
    )


def run_worker(stage, run_dir):
    stored = read_json(run_dir / "resolved_config.json")
    resolved = resolve_spec(stored["spec"])
    verify_run(run_dir, resolved)
    record = stage_record(run_dir, stage)
    if (
        not record
        or record["status"] != "running"
        or record["fingerprint"] != resolved["fingerprint"]
    ):
        raise ValueError("Worker must be launched by the prepared stage controller")
    preflight(resolved, stage)
    for dependency in {"capture": (), "train": ("capture",), "evaluate": ("capture", "train")}[
        stage
    ]:
        verify_completed(run_dir, dependency, resolved["fingerprint"])
    import random
    import numpy as np

    seed = resolved["spec"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    parameters = stage_parameters(resolved, run_dir)[stage]
    atomic_json(run_dir / stage / "effective_config.json", parameters)
    app = None
    try:
        needs_isaac = stage == "evaluate" or (
            stage == "capture" and resolved["spec"]["capture"]["backend"] == "isaac"
        )
        if needs_isaac:
            from omni.isaac.kit import SimulationApp

            app = SimulationApp({"headless": resolved["spec"]["runtime"]["headless"]})
        if stage == "capture":
            if resolved["spec"]["capture"]["backend"] == "synthetic_smoke":
                synthetic_capture(resolved, run_dir / "capture")
            else:
                import render_dataset

                render_dataset.generate_dataset(parameters, simulation_app=app)
            seal_capture(resolved, run_dir)
        elif stage == "train":
            import torch

            torch.manual_seed(seed)
            torch.set_num_threads(resolved["spec"]["runtime"]["torch_num_threads"])
            train_stage(resolved, run_dir, parameters)
        else:
            evaluate_stage(resolved, run_dir, parameters, app)
        verify_run(run_dir, resolve_spec(stored["spec"]))
        # Commit output validation before Kit closes; teardown can exit Python.
        complete_stage(run_dir, stage, os.getpid())
        print("Worker completed {} (pid {})".format(stage, os.getpid()), flush=True)
    finally:
        if app is not None:
            app.close()
