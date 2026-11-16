"""Resume every optimizer/RNG state at epoch150 in a new immutable experiment lineage."""

import copy
from pathlib import Path

import torch

from insect_utils.circular_augmentation import make_circular_loader
from insect_utils.circular_training import build_model
from insect_utils.continuation_protocol import verify_continuation
from insect_utils.coverage_data import load_data
from insect_utils.coverage_model import PhysicalHomeVector
from insect_utils.coverage_training import export_physical
from insect_utils.downward_training import atomic_torch, runtime_signature, train_epochs
from insect_utils.experiment_config import PROJECT, digest
from insect_utils.experiment_state import atomic_json, now, read_json
from insect_utils.rgb_preprocessing import load_rgb_tensor
from insect_utils.training_state import checkpoint_pointer, save_checkpoint


def states_equal(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(states_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(states_equal(x, y) for x, y in zip(a, b))
        )
    return type(a) is type(b) and a == b


def import_checkpoint(
    source, engine, fingerprint, source_fingerprint, expected_epoch, expected_steps
):
    """Rebind only the run identity; prove all training state survives serialization."""
    engine = Path(engine)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if (
        payload["fingerprint"] != source_fingerprint
        or payload["epoch"] != expected_epoch
        or payload["optimizer_steps"] != expected_steps
        or len(payload["history"]) != expected_epoch
    ):
        raise ValueError("Source checkpoint epoch, update count or experiment differs")
    receipt_file = engine.parent / "checkpoint_import.json"
    if (engine / "checkpoint_latest.json").exists():
        path, pointer = checkpoint_pointer(engine, fingerprint)
        if receipt_file.exists():
            receipt = read_json(receipt_file)
            if receipt["source_sha256"] != digest(source) or receipt["fingerprint"] != fingerprint:
                raise ValueError("Imported checkpoint lineage changed")
            return receipt
        if pointer["epoch"] != expected_epoch:
            raise ValueError("A continued checkpoint is missing its import provenance")
    else:
        pointer = save_checkpoint(engine, copy.deepcopy(payload), fingerprint)
        path, _ = checkpoint_pointer(engine, fingerprint)
    imported = torch.load(path, map_location="cpu", weights_only=True)
    parent_state = {k: v for k, v in payload.items() if k != "fingerprint"}
    child_state = {k: v for k, v in imported.items() if k != "fingerprint"}
    if not states_equal(parent_state, child_state):
        raise ValueError("Checkpoint transfer changed model, optimizer, RNG or history")
    receipt = dict(
        status="passed",
        updated=now(),
        fingerprint=fingerprint,
        source_sha256=digest(source),
        imported_checkpoint_sha256=digest(path),
        source_fingerprint=source_fingerprint,
        imported_epoch=expected_epoch,
        imported_optimizer_steps=expected_steps,
        all_payload_except_experiment_fingerprint_exactly_equal=True,
    )
    atomic_json(receipt_file, receipt)
    return receipt


def require_runtime(actual, original):
    keys = (
        "python",
        "torch",
        "cuda_version",
        "cudnn_version",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cudnn_allow_tf32",
        "matmul_allow_tf32",
        "gpu_name",
    )
    for key in keys:
        if actual[key] != original[key]:
            raise ValueError("Continuation runtime changed: " + key)


def select_checkpoint(engine, fingerprint, final):
    best = min(final["history"], key=lambda r: (r["validation_component_mse_m2"], r["epoch"]))
    states = []
    for path in (engine / "checkpoints").glob("epoch_{:04d}_*.pt".format(best["epoch"])):
        candidate = torch.load(path, map_location="cpu", weights_only=True)
        if (
            candidate["fingerprint"] == fingerprint
            and candidate["history"] == final["history"][: best["epoch"]]
        ):
            states.append(candidate["model"])
    if not states or any(not states_equal(states[0], state) for state in states[1:]):
        raise ValueError("Validation-best continuation checkpoint missing or ambiguous")
    return best, states[0]


def train_continuation(folder):
    folder = Path(folder)
    frozen = verify_continuation(folder)
    config = frozen["training_config"]
    destination = folder / "training"
    destination.mkdir(exist_ok=True)
    completion_file = destination / "completion.json"
    if completion_file.exists():
        completed = read_json(completion_file)
        if completed["fingerprint"] != frozen["fingerprint"]:
            raise ValueError("Training completion belongs to another continuation")
        for name, expected in completed["files"].items():
            if digest(destination / name) != expected:
                raise ValueError("Completed continuation artifact changed")
        return completed
    if not torch.cuda.is_available():
        raise RuntimeError("The original GPU is required for formal continuation")
    torch.set_num_threads(config["torch_num_threads"])
    device = torch.device("cuda")
    runtime = runtime_signature(device)
    require_runtime(runtime, frozen["source_runtime"])
    parent = PROJECT / frozen["protocol"]["source_training_run"]
    parent_protocol = read_json(parent / "protocol.json")
    records = load_data(PROJECT / parent_protocol["source_run"])
    train_loader = make_circular_loader(records, "train", config, random_rotation=True)
    val_loader = make_circular_loader(records, "validation", config)
    model = build_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], **config["optimizer_kwargs"]
    )
    engine = destination / "engine_scaled_units"
    source = PROJECT / frozen["source_checkpoint"]
    receipt = import_checkpoint(
        source,
        engine,
        frozen["fingerprint"],
        frozen["protocol"]["source_arm_fingerprint"],
        150,
        76800,
    )
    original = torch.load(source, map_location="cpu", weights_only=True)
    for group in original["optimizer"]["param_groups"]:
        if group["lr"] != config["learning_rate"]:
            raise ValueError("Source optimizer learning rate differs from frozen configuration")
    if frozen["protocol"]["source_epoch"] != 150 or len(train_loader) != 512:
        raise ValueError("Original epoch/sample/update accounting changed")
    atomic_json(
        destination / "contract.json",
        dict(
            fingerprint=frozen["fingerprint"],
            configuration=config,
            runtime=runtime,
            preprocessing=frozen["preprocessing"],
            train_images=len(train_loader.dataset),
            validation_images=len(val_loader.dataset),
            source_checkpoint=frozen["source_checkpoint"],
            checkpoint_import=receipt,
            start_epoch=150,
            target_epoch=300,
            expected_additional_optimizer_steps=76800,
            weights_optimizer_and_rng_restored_by_engine=True,
            static_test_data_opened=False,
        ),
    )
    result = train_epochs(
        model,
        optimizer,
        train_loader,
        val_loader,
        config,
        engine,
        frozen["fingerprint"],
        resume=True,
    )
    if result["status"] != "completed" or result["optimizer_steps"] != 153600:
        raise ValueError("Continuation has not completed300 total epochs")
    checkpoint_file, _ = checkpoint_pointer(engine, frozen["fingerprint"])
    final = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    if final["history"][:150] != original["history"]:
        raise ValueError("Inherited first150 training epochs changed")
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
        dict(epochs=history, scale_m=44, inherited_epochs=150, additional_epochs=150),
    )
    selection = dict(
        fingerprint=frozen["fingerprint"],
        selected_epoch=best["epoch"],
        selected_by="full-validation physical component MSE over1..300; no navigation outcomes",
        best_validation=history[best["epoch"] - 1],
        static_test_data_opened=False,
        selected_weights_sha256=digest(destination / "best_validation.pth"),
    )
    atomic_json(destination / "selection_frozen_before_navigation.json", selection)
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
        "physical_history.json",
        "contract.json",
        "checkpoint_import.json",
        "selection_frozen_before_navigation.json",
        "deployment/home_vector_m.torchscript.pt",
        "deployment/home_vector_m.onnx",
        "deployment/export_check.json",
    ]
    summary = dict(
        status="completed",
        updated=now(),
        fingerprint=frozen["fingerprint"],
        epochs_total=300,
        epochs_added=150,
        optimizer_steps_total=153600,
        optimizer_steps_added=76800,
        best_validation=history[best["epoch"] - 1],
        final_validation=history[-1],
        static_test_data_opened=False,
        resources=result["resources"],
        files={name: digest(destination / name) for name in names},
    )
    atomic_json(completion_file, summary)
    verify_continuation(folder)
    return summary
