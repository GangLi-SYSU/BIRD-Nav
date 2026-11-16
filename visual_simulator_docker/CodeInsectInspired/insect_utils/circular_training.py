"""Isolated circular-input controls reusing the frozen data, network and epoch engine."""

from pathlib import Path

import torch

from insect_utils.circular_augmentation import CircularAttention, make_circular_loader
from insect_utils.coverage_data import load_data
from insect_utils.coverage_model import PhysicalHomeVector
from insect_utils.coverage_plan import verify_freeze as verify_source
from insect_utils.coverage_training import export_physical, metrics_by_radius
from insect_utils.downward_training import atomic_torch, runtime_signature, seed_all, train_epochs
from insect_utils.experiment_config import PROJECT, canonical_hash, digest
from insect_utils.experiment_state import atomic_json, now, read_json
from insect_utils.rgb_preprocessing import load_rgb_tensor
from insect_utils.training_state import checkpoint_pointer


PROTOCOL = "config/experiments/stage13_circular_protocol.json"


def prepare_circular(folder, arms=None):
    folder = Path(folder).resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    protocol = read_json(PROJECT / PROTOCOL)
    validate_preprocessing_contract(protocol)
    if arms is not None:
        if not arms or len(set(arms)) != len(arms) or not set(arms) <= set(protocol["arms"]):
            raise ValueError("Expected distinct known circular control arms")
        protocol["arms"] = {name: protocol["arms"][name] for name in sorted(arms)}
    parent_folder = PROJECT / protocol["source_run"]
    parent = verify_source(parent_folder)
    if parent["fingerprint"] != protocol["source_fingerprint"]:
        raise ValueError("Source data differs from the approved coverage experiment")
    source_completion = read_json(parent_folder / "completion.json")
    if source_completion["status"] != "completed":
        raise ValueError("Source coverage experiment is incomplete")
    records = load_data(parent_folder)
    counts = {
        split: sum(r["split"] == split for r in records)
        for split in ("train", "validation", "legacy_test", "new_test")
    }
    if counts != dict(train=4096, validation=320, legacy_test=64, new_test=320):
        raise ValueError("Frozen source split sizes changed")
    files = dict(parent["files"])
    extras = list((PROJECT / "insect_utils").glob("circular_*.py"))
    extras += [
        PROJECT / "run_circular_experiment.py",
        PROJECT / PROTOCOL,
        PROJECT / "tests/test_stage13_circular.py",
        PROJECT / "tools/preview_circular_augmentation.py",
        parent_folder / "dataset.json",
        parent_folder / "completion.json",
    ]
    files.update({str(path.relative_to(PROJECT)): digest(path) for path in extras})
    value = dict(
        protocol=protocol,
        files=files,
        split_counts=counts,
        source_fingerprint=parent["fingerprint"],
    )
    value["fingerprint"] = canonical_hash(value)
    destination = folder / "frozen.json"
    if destination.exists():
        if read_json(destination) != value:
            raise ValueError("Circular experiment is frozen; use a new run directory")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(destination, value)
        atomic_json(folder / "protocol.json", protocol)
    return value


def verify_circular(folder):
    value = read_json(Path(folder) / "frozen.json")
    if (
        canonical_hash({k: v for k, v in value.items() if k != "fingerprint"})
        != value["fingerprint"]
    ):
        raise ValueError("Circular experiment fingerprint changed")
    for name, expected in value["files"].items():
        if digest(PROJECT / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    return value


def load_circular_data(frozen):
    return load_data(PROJECT / frozen["protocol"]["source_run"])


def smoke_subset(records):
    """Two complete positions per split; held-out test images never enter smoke runs."""
    groups = {
        split: sorted({r["position_id"] for r in records if r["split"] == split})[:2]
        for split in ("train", "validation")
    }
    return [r for r in records if r["split"] in groups and r["position_id"] in groups[r["split"]]]


def build_model(config, device):
    seed_all(config["seed"], cuda=device.type == "cuda")
    model = CircularAttention(config).to(device)
    if sum(p.numel() for p in model.parameters()) != config["model_parameters"]:
        raise ValueError("Trainable network architecture changed")
    return model


def validate_preprocessing_contract(protocol):
    from insect_utils.circular_augmentation import INPUT_SIZE, MASK_RADIUS_PX

    spec = protocol["preprocessing"]
    if (
        spec["input_size"] != INPUT_SIZE
        or spec["mask_radius_px"] != MASK_RADIUS_PX
        or spec["mask_center_xy_px"] != [191.5, 191.5]
        or spec["mask_fill_rgb_float"] != [0.0, 0.0, 0.0]
    ):
        raise ValueError("Configured preprocessing differs from executable implementation")


def train_circular_arm(folder, arm, smoke=False):
    folder = Path(folder)
    frozen = verify_circular(folder)
    protocol = frozen["protocol"]
    validate_preprocessing_contract(protocol)
    if arm not in protocol["arms"]:
        raise ValueError("Arm was not included in this frozen experiment")
    config = dict(protocol["training"])
    if not torch.cuda.is_available():
        raise RuntimeError("Training and GPU smoke require the available project GPU")
    device = torch.device("cuda")
    torch.set_num_threads(config["torch_num_threads"])
    records = load_circular_data(frozen)
    if smoke:
        records = smoke_subset(records)
        config.update(epochs=2)
    destination = folder / ("smoke" if smoke else "arms") / arm
    destination.mkdir(parents=True, exist_ok=True)
    fingerprint = canonical_hash(
        dict(experiment=frozen["fingerprint"], arm=arm, smoke=smoke, config=config)
    )
    completion_file = destination / "completion.json"
    if completion_file.exists():
        previous = read_json(completion_file)
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Training completion belongs to another configuration")
        for name, expected in previous["files"].items():
            if digest(destination / name) != expected:
                raise ValueError("Completed circular artifact changed: " + name)
        return previous
    model = build_model(config, device)
    from insect_utils.train_utils import parameter_digest

    initial = parameter_digest(model)
    train_loader = make_circular_loader(
        records, "train", config, random_rotation=protocol["arms"][arm]
    )
    val_loader = make_circular_loader(records, "validation", config)
    contract = dict(
        fingerprint=fingerprint,
        arm=arm,
        smoke=smoke,
        random_rotation=protocol["arms"][arm],
        configuration=config,
        initial_parameter_sha256=initial,
        runtime=runtime_signature(device),
        train_images=len(train_loader.dataset),
        validation_images=len(val_loader.dataset),
        test_data_opened=False,
        preprocessing=protocol["preprocessing"],
        formal_initialization_from_smoke=False,
    )
    atomic_json(destination / "contract.json", contract)
    atomic_json(
        folder / "progress.json",
        dict(
            status="running",
            phase="smoke" if smoke else "train",
            arm=arm,
            updated=now(),
            engine=str((destination / "engine_scaled_units").relative_to(folder)),
        ),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], **config["optimizer_kwargs"]
    )
    engine = destination / "engine_scaled_units"
    result = train_epochs(
        model,
        optimizer,
        train_loader,
        val_loader,
        config,
        engine,
        fingerprint,
        resume=(engine / "checkpoint_latest.json").exists(),
    )
    if result["status"] != "completed":
        raise RuntimeError("Circular training did not reach its configured budget")
    checkpoint_file, _ = checkpoint_pointer(engine, fingerprint)
    final = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    best_row = min(final["history"], key=lambda r: (r["validation_component_mse_m2"], r["epoch"]))
    eligible = []
    for path in (engine / "checkpoints").glob("epoch_{:04d}_*.pt".format(best_row["epoch"])):
        candidate = torch.load(path, map_location="cpu", weights_only=True)
        if (
            candidate["fingerprint"] == fingerprint
            and candidate["history"] == final["history"][: best_row["epoch"]]
        ):
            eligible.append(candidate["model"])
    if not eligible or any(
        any(not torch.equal(a[k], eligible[0][k]) for k in a) for a in eligible[1:]
    ):
        raise ValueError("Validation-selected checkpoint missing or ambiguous")
    atomic_torch(destination / "best_validation.pth", eligible[0])
    atomic_torch(destination / "final.pth", final["model"])
    history = []
    for row in final["history"]:
        physical = dict(row)
        for key in ("train_component_mse_m2", "validation_component_mse_m2"):
            physical[key] *= config["scale_m"] ** 2
        history.append(physical)
    atomic_json(
        destination / "physical_history.json", dict(epochs=history, scale_m=config["scale_m"])
    )
    model.load_state_dict(eligible[0])
    example_rows = [r for r in records if r["split"] == "validation"][:2]
    raw = torch.stack(
        [load_rgb_tensor(PROJECT / r["image"], [1024, 1024], r["sha256"]) for r in example_rows]
    )
    physical_model = PhysicalHomeVector(model, config["scale_m"]).eval()
    export = export_physical(physical_model, raw, destination / "deployment")
    export.update(preprocessing=protocol["preprocessing"], random_rotation_at_inference=False)
    atomic_json(destination / "deployment/export_check.json", export)
    # CPU replay checks the complete raw-RGB path, including the model-owned mask.
    with torch.inference_mode():
        cpu = physical_model.cpu().eval()
        expected = cpu(raw)
        prepared = next(iter(val_loader))[0][:2]
        actual = cpu(prepared)
    if not torch.allclose(actual, expected, atol=1e-5, rtol=1e-4):
        raise ValueError("Dataset and raw-RGB deployment preprocessing disagree")
    best_physical = history[best_row["epoch"] - 1]
    summary = dict(
        status="completed",
        updated=now(),
        fingerprint=fingerprint,
        arm=arm,
        smoke=smoke,
        epochs=config["epochs"],
        optimizer_steps=result["optimizer_steps"],
        initial_parameter_sha256=initial,
        best_validation=best_physical,
        final_validation=history[-1],
        test_data_opened=False,
        dataset_deployment_max_abs_difference_m=float((actual - expected).abs().max()),
        resources=result["resources"],
    )
    names = [
        "best_validation.pth",
        "final.pth",
        "physical_history.json",
        "contract.json",
        "deployment/home_vector_m.torchscript.pt",
        "deployment/home_vector_m.onnx",
        "deployment/export_check.json",
    ]
    summary["files"] = {name: digest(destination / name) for name in names}
    atomic_json(completion_file, summary)
    verify_circular(folder)
    return summary


def freeze_circular_selection(folder):
    folder = Path(folder)
    frozen = verify_circular(folder)
    arms = {
        name: read_json(folder / "arms" / name / "completion.json")
        for name in frozen["protocol"]["arms"]
    }
    if any(
        r["status"] != "completed" or r["smoke"] or r["test_data_opened"] for r in arms.values()
    ):
        raise ValueError("Every requested formal arm must finish without test inference")
    if len({r["initial_parameter_sha256"] for r in arms.values()}) != 1:
        raise ValueError("Controlled arms did not share the same initialization")
    for arm, completion in arms.items():
        for name, expected in completion["files"].items():
            if digest(folder / "arms" / arm / name) != expected:
                raise ValueError("Selected training artifact changed")
    selected = min(
        arms, key=lambda name: (arms[name]["best_validation"]["validation_component_mse_m2"], name)
    )
    value = dict(
        fingerprint=frozen["fingerprint"],
        arms=arms,
        selected_arm=selected,
        selection=(
            "validation physical component MSE only; "
            "existing test sets are development comparisons"
        ),
    )
    path = folder / "selection_frozen_before_test.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("Frozen circular selection changed")
    atomic_json(path, value)
    return value


def evaluate_circular(folder):
    """Explicit post-training development comparison, never an automatic navigation run."""
    folder = Path(folder)
    frozen = verify_circular(folder)
    selection = freeze_circular_selection(folder)
    records = load_circular_data(frozen)
    config = dict(frozen["protocol"]["training"], scale_m=1.0)
    torch.set_num_threads(config["torch_num_threads"])
    results = {}
    for arm in frozen["protocol"]["arms"]:
        model = build_model(config, torch.device("cpu"))
        model.load_state_dict(
            torch.load(
                folder / "arms" / arm / "best_validation.pth", map_location="cpu", weights_only=True
            )
        )
        physical = PhysicalHomeVector(model, frozen["protocol"]["training"]["scale_m"]).eval()
        results[arm] = {}
        for split in ("validation", "legacy_test", "new_test"):
            loader = make_circular_loader(records, split, config, allow_test=True)
            predictions, targets = [], []
            with torch.inference_mode():
                for images, values in loader:
                    predictions.extend(physical(images).tolist())
                    targets.extend(values.tolist())
            source = read_json(PROJECT / frozen["protocol"]["source_run"] / "protocol.json")
            results[arm][split] = metrics_by_radius(
                loader.dataset.records, predictions, targets, source["radial_edges_m"]
            )
            atomic_json(
                folder / "evaluation" / arm / (split + ".json"),
                dict(
                    rows=[
                        dict(
                            position_id=r["position_id"],
                            heading=r["heading_index"],
                            prediction_m=p,
                            target_m=t,
                        )
                        for r, p, t in zip(loader.dataset.records, predictions, targets)
                    ],
                    metrics=results[arm][split],
                ),
            )
    output = dict(
        fingerprint=frozen["fingerprint"],
        updated=now(),
        results=results,
        selected_arm=selection["selected_arm"],
        limits=frozen["protocol"]["limits"],
        navigation_runs=0,
    )
    atomic_json(folder / "evaluation/results.json", output)
    atomic_json(
        folder / "completion.json",
        dict(
            status="completed",
            updated=now(),
            fingerprint=frozen["fingerprint"],
            results_sha256=digest(folder / "evaluation/results.json"),
        ),
    )
    return output
