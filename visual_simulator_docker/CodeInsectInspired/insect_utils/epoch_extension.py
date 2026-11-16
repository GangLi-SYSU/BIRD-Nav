"""Extend the completed300-epoch run without altering any frozen predecessor."""

from pathlib import Path

import torch

from insect_utils.circular_augmentation import make_circular_loader
from insect_utils.circular_training import build_model
from insect_utils.continuation_protocol import verify_continuation
from insect_utils.continuation_training import (
    import_checkpoint,
    require_runtime,
    select_checkpoint,
    states_equal,
)
from insect_utils.coverage_data import load_data
from insect_utils.coverage_model import PhysicalHomeVector
from insect_utils.coverage_training import export_physical
from insect_utils.downward_training import atomic_torch, runtime_signature, train_epochs
from insect_utils.experiment_config import PROJECT, canonical_hash, digest
from insect_utils.experiment_state import atomic_json, now, read_json
from insect_utils.navigation_protocol import verify_navigation
from insect_utils.rgb_preprocessing import load_rgb_tensor
from insect_utils.training_state import checkpoint_pointer


PROTOCOL = "config/experiments/stage16_epoch_extension_protocol.json"


def require_config(original, extended, target):
    if (
        original["epochs"] != 300
        or target not in (450, 600)
        or extended != dict(original, epochs=target)
    ):
        raise ValueError("Only total epochs may change from300 to the chosen450 or600")


def build_extension_plan(target):
    """Validate either proposed budget without writing a plan or starting training."""
    protocol = read_json(PROJECT / PROTOCOL)
    if target not in protocol["allowed_target_epochs"]:
        raise ValueError("Choose the explicitly approved target epoch budget")
    protocol["target_epochs"] = target
    protocol["target_optimizer_steps"] = target * 512
    parent_folder = PROJECT / protocol["source_run"]
    parent = verify_continuation(parent_folder)
    if parent["fingerprint"] != protocol["source_fingerprint"]:
        raise ValueError("The completed300-epoch source experiment changed")
    completed = read_json(parent_folder / "training/completion.json")
    if (
        completed["status"] != "completed"
        or completed["epochs_total"] != 300
        or completed["optimizer_steps_total"] != 153600
    ):
        raise ValueError("The source training run must have300 complete epochs")
    engine = parent_folder / "training/engine_scaled_units"
    source, pointer = checkpoint_pointer(engine, parent["fingerprint"])
    if (
        pointer["epoch"] != 300
        or pointer["optimizer_steps"] != 153600
        or pointer["sha256"] != protocol["source_checkpoint_sha256"]
    ):
        raise ValueError("Use the exact complete epoch300 optimizer/RNG checkpoint")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    best = completed["best_validation"]["epoch"]
    candidates = []
    selected_weights = parent_folder / "training/best_validation.pth"
    if digest(selected_weights) != completed["files"]["best_validation.pth"]:
        raise ValueError("Historical validation-selected weights changed")
    weights = torch.load(selected_weights, map_location="cpu", weights_only=True)
    for path in sorted((engine / "checkpoints").glob("epoch_{:04d}_*.pt".format(best))):
        candidate = torch.load(path, map_location="cpu", weights_only=True)
        if (
            candidate["fingerprint"] == parent["fingerprint"]
            and candidate["history"] == payload["history"][:best]
            and states_equal(candidate["model"], weights)
        ):
            candidates.append(path)
    if not candidates:
        raise ValueError("The historical validation-best checkpoint is missing")
    navigation = verify_navigation(parent_folder / "navigation")
    config = dict(parent["training_config"], epochs=target)
    require_config(parent["training_config"], config, target)
    data_file = PROJECT / protocol["dataset_run"] / "dataset.json"
    if parent["files"].get(str(data_file.relative_to(PROJECT))) != digest(data_file):
        raise ValueError("The original frozen dataset must be reused")
    records = load_data(PROJECT / protocol["dataset_run"])
    if (
        sum(r["split"] == "train" for r in records) != 4096
        or sum(r["split"] == "validation" for r in records) != 320
    ):
        raise ValueError("Original data partitions changed")
    if len(navigation["episodes"]["formal"]) != 64 or {
        e["radius_m"] for e in navigation["episodes"]["formal"]
    } != {20}:
        raise ValueError("The original64 trials at20m are required")
    files = dict(navigation["files"])
    extras = [
        PROJECT / PROTOCOL,
        PROJECT / "insect_utils/epoch_extension.py",
        PROJECT / "insect_utils/epoch_extension_report.py",
        PROJECT / "run_epoch_extension.py",
        PROJECT / "tests/test_stage16_epoch_extension.py",
        source,
        candidates[0],
        parent_folder / "training/engine_scaled_units/checkpoint_latest.json",
        parent_folder / "training/completion.json",
        parent_folder / "training/contract.json",
        parent_folder / "navigation/frozen.json",
        parent_folder / "navigation/formal/results.json",
        parent_folder / "navigation/formal/completion.json",
    ]
    files.update({str(p.relative_to(PROJECT)): digest(p) for p in extras})
    value = dict(
        protocol=protocol,
        training_config=config,
        preprocessing=parent["preprocessing"],
        navigation_protocol=navigation["protocol"],
        episodes=navigation["episodes"],
        budget=navigation["budget"],
        files=files,
        source_checkpoint=str(source.relative_to(PROJECT)),
        historical_best_checkpoint=str(candidates[0].relative_to(PROJECT)),
        historical_best_epoch=best,
        source_runtime=read_json(parent_folder / "training/contract.json")["runtime"],
    )
    value["fingerprint"] = canonical_hash(value)
    return value


def prepare_extension(folder, target):
    folder = Path(folder).resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    value = build_extension_plan(target)
    path = folder / "frozen.json"
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Extension already frozen; use another run directory")
    else:
        atomic_json(path, value)
        atomic_json(folder / "protocol.json", value["protocol"])
    return value


def verify_extension(folder):
    value = read_json(Path(folder) / "frozen.json")
    if (
        canonical_hash({k: v for k, v in value.items() if k != "fingerprint"})
        != value["fingerprint"]
    ):
        raise ValueError("Extension experiment fingerprint changed")
    for name, expected in value["files"].items():
        if digest(PROJECT / name) != expected:
            raise ValueError("Frozen extension input changed: " + name)
    return value


def preserve_best(source, engine, fingerprint, source_fingerprint, epoch):
    """Retain the earlier validation candidate without moving the latest resume pointer."""
    engine = Path(engine)
    original = torch.load(source, map_location="cpu", weights_only=True)
    if original["fingerprint"] != source_fingerprint or original["epoch"] != epoch:
        raise ValueError("Historical checkpoint identity changed")
    path = engine / "checkpoints" / "epoch_{:04d}_historical_best.pt".format(epoch)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch(path, dict(original, fingerprint=fingerprint))
    imported = torch.load(path, map_location="cpu", weights_only=True)
    if imported["fingerprint"] != fingerprint or not states_equal(
        {k: v for k, v in original.items() if k != "fingerprint"},
        {k: v for k, v in imported.items() if k != "fingerprint"},
    ):
        raise ValueError("Historical validation candidate was changed during import")
    receipt = dict(
        status="passed",
        source_sha256=digest(source),
        imported_sha256=digest(path),
        epoch=epoch,
        used_for_validation_selection_only=True,
        resume_pointer_unchanged=True,
    )
    atomic_json(engine.parent / "historical_best_import.json", receipt)
    return receipt


def train_extension(folder):
    folder = Path(folder)
    frozen = verify_extension(folder)
    config = frozen["training_config"]
    target = config["epochs"]
    destination = folder / "training"
    destination.mkdir(exist_ok=True)
    completed_file = destination / "completion.json"
    if completed_file.exists():
        value = read_json(completed_file)
        if value["fingerprint"] != frozen["fingerprint"]:
            raise ValueError("Wrong training completion identity")
        for name, expected in value["files"].items():
            if digest(destination / name) != expected:
                raise ValueError("Training artifact changed")
        return value
    if not torch.cuda.is_available():
        raise RuntimeError("Formal epoch extension requires the original GPU")
    torch.set_num_threads(config["torch_num_threads"])
    device = torch.device("cuda")
    runtime = runtime_signature(device)
    require_runtime(runtime, frozen["source_runtime"])
    records = load_data(PROJECT / frozen["protocol"]["dataset_run"])
    train = make_circular_loader(records, "train", config, random_rotation=True)
    val = make_circular_loader(records, "validation", config)
    if len(train) != 512:
        raise ValueError("Training update count per epoch changed")
    model = build_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], **config["optimizer_kwargs"]
    )
    source = PROJECT / frozen["source_checkpoint"]
    original = torch.load(source, map_location="cpu", weights_only=True)
    engine = destination / "engine_scaled_units"
    imported = import_checkpoint(
        source, engine, frozen["fingerprint"], frozen["protocol"]["source_fingerprint"], 300, 153600
    )
    preserve_best(
        PROJECT / frozen["historical_best_checkpoint"],
        engine,
        frozen["fingerprint"],
        frozen["protocol"]["source_fingerprint"],
        frozen["historical_best_epoch"],
    )
    if any(g["lr"] != config["learning_rate"] for g in original["optimizer"]["param_groups"]):
        raise ValueError("Original learning rate changed")
    atomic_json(
        destination / "contract.json",
        dict(
            fingerprint=frozen["fingerprint"],
            configuration=config,
            runtime=runtime,
            preprocessing=frozen["preprocessing"],
            start_epoch=300,
            target_epoch=target,
            checkpoint_import=imported,
            historical_best_epoch=frozen["historical_best_epoch"],
            train_images=len(train.dataset),
            validation_images=len(val.dataset),
            static_test_data_opened=False,
        ),
    )
    result = train_epochs(
        model, optimizer, train, val, config, engine, frozen["fingerprint"], resume=True
    )
    if result["status"] != "completed" or result["optimizer_steps"] != target * 512:
        raise RuntimeError("Extended training did not finish its chosen epoch budget")
    path, _ = checkpoint_pointer(engine, frozen["fingerprint"])
    final = torch.load(path, map_location="cpu", weights_only=True)
    if final["history"][:300] != original["history"]:
        raise ValueError("Inherited300-epoch history changed")
    best, state = select_checkpoint(engine, frozen["fingerprint"], final)
    atomic_torch(destination / "best_validation.pth", state)
    atomic_torch(destination / "final.pth", final["model"])
    history = []
    for row in final["history"]:
        physical = dict(row)
        for key in ("train_component_mse_m2", "validation_component_mse_m2"):
            physical[key] *= config["scale_m"] ** 2
        history.append(physical)
    atomic_json(
        destination / "physical_history.json",
        dict(epochs=history, scale_m=44, inherited_epochs=300, additional_epochs=target - 300),
    )
    atomic_json(
        destination / "selection_frozen_before_navigation.json",
        dict(
            fingerprint=frozen["fingerprint"],
            selected_epoch=best["epoch"],
            best_validation=history[best["epoch"] - 1],
            criterion="full-validation physical component MSE over all inherited and new epochs",
            selected_weights_sha256=digest(destination / "best_validation.pth"),
        ),
    )
    model.load_state_dict(state)
    physical = PhysicalHomeVector(model, 44).eval()
    examples = [r for r in records if r["split"] == "validation"][:2]
    raw = torch.stack(
        [load_rgb_tensor(PROJECT / r["image"], [1024, 1024], r["sha256"]) for r in examples]
    )
    exported = export_physical(physical, raw, destination / "deployment")
    exported.update(preprocessing=frozen["preprocessing"], random_rotation_at_inference=False)
    atomic_json(destination / "deployment/export_check.json", exported)
    names = [
        "best_validation.pth",
        "final.pth",
        "contract.json",
        "physical_history.json",
        "checkpoint_import.json",
        "historical_best_import.json",
        "selection_frozen_before_navigation.json",
        "deployment/home_vector_m.torchscript.pt",
        "deployment/home_vector_m.onnx",
        "deployment/export_check.json",
    ]
    value = dict(
        status="completed",
        updated=now(),
        fingerprint=frozen["fingerprint"],
        epochs_total=target,
        epochs_added=target - 300,
        optimizer_steps_total=target * 512,
        optimizer_steps_added=(target - 300) * 512,
        best_validation=history[best["epoch"] - 1],
        final_validation=history[-1],
        static_test_data_opened=False,
        resources=result["resources"],
        files={name: digest(destination / name) for name in names},
    )
    atomic_json(completed_file, value)
    verify_extension(folder)
    return value


def prepare_navigation(folder):
    folder = Path(folder)
    frozen = verify_extension(folder)
    completed = read_json(folder / "training/completion.json")
    target = frozen["training_config"]["epochs"]
    if (
        completed["status"] != "completed"
        or completed["epochs_total"] != target
        or completed["optimizer_steps_total"] != target * 512
    ):
        raise ValueError("Training budget is incomplete")
    files = dict(frozen["files"])
    for name, expected in completed["files"].items():
        path = folder / "training" / name
        if digest(path) != expected:
            raise ValueError("Selected training artifact changed")
        files[str(path.relative_to(PROJECT))] = expected
    for path in (
        folder / "frozen.json",
        folder / "training/completion.json",
        folder / "training/selection_frozen_before_navigation.json",
    ):
        files[str(path.relative_to(PROJECT))] = digest(path)
    base = folder / "training"
    value = dict(
        protocol=frozen["navigation_protocol"],
        episodes=frozen["episodes"],
        budget=frozen["budget"],
        files=files,
        model=dict(
            arm="mask_rotation_epoch_extension",
            epoch=completed["best_validation"]["epoch"],
            scale_m=44,
            weights=str((base / "best_validation.pth").relative_to(PROJECT)),
            torchscript=str(
                (base / "deployment/home_vector_m.torchscript.pt").relative_to(PROJECT)
            ),
            output_units="body-frame home vector in meters",
        ),
    )
    value["fingerprint"] = canonical_hash(value)
    path = folder / "navigation/frozen.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("Navigation model is already frozen")
    if not path.exists():
        atomic_json(path, value)
        atomic_json(path.with_name("protocol.json"), value["protocol"])
        atomic_json(path.with_name("episode_plan.json"), value["episodes"])
    return value
