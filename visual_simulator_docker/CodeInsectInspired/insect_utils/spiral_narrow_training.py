"""Fresh600-epoch circular/rotation training on the newly rendered spiral only."""

from pathlib import Path

import torch

from insect_utils.circular_augmentation import make_circular_loader
from insect_utils.circular_training import (
    build_model,
    smoke_subset,
    validate_preprocessing_contract,
)
from insect_utils.coverage_model import PhysicalHomeVector
from insect_utils.coverage_plan import verify_freeze
from insect_utils.coverage_training import export_physical
from insect_utils.downward_training import atomic_torch, runtime_signature, train_epochs
from insect_utils.experiment_config import PROJECT, canonical_hash, digest
from insect_utils.experiment_state import atomic_json, now, read_json
from insect_utils.rgb_preprocessing import load_rgb_tensor
from insect_utils.spiral_narrow_protocol import load_data
from insect_utils.training_state import checkpoint_pointer


def train(folder, smoke=False):
    folder = Path(folder)
    frozen = verify_freeze(folder)
    protocol = frozen["protocol"]
    validate_preprocessing_contract(protocol)
    config = dict(protocol["training"])
    records = load_data(folder)
    if smoke:
        records = smoke_subset(records)
        config["epochs"] = 2
    destination = folder / ("smoke" if smoke else "training")
    destination.mkdir(parents=True, exist_ok=True)
    fingerprint = canonical_hash(
        dict(
            experiment=frozen["fingerprint"], config=config, smoke=smoke, fresh_initialization=True
        )
    )
    completion_file = destination / "completion.json"
    if completion_file.exists():
        result = read_json(completion_file)
        if result["fingerprint"] != fingerprint:
            raise ValueError("Completed training belongs to another protocol")
        for name, expected in result["files"].items():
            if digest(destination / name) != expected:
                raise ValueError("Completed training artifact changed")
        return result
    if not torch.cuda.is_available():
        raise RuntimeError("The authorized training requires the project GPU")
    torch.set_num_threads(config["torch_num_threads"])
    device = torch.device("cuda")
    model = build_model(config, device)
    from insect_utils.train_utils import parameter_digest

    initial = parameter_digest(model)
    train_loader = make_circular_loader(records, "train", config, random_rotation=True)
    val_loader = make_circular_loader(records, "validation", config)
    dataset = read_json(folder / "dataset.json")
    contract = dict(
        fingerprint=fingerprint,
        experiment_fingerprint=frozen["fingerprint"],
        configuration=config,
        initial_parameter_sha256=initial,
        initialization="fresh seeded Xavier; no source checkpoint and no smoke weights",
        runtime=runtime_signature(device),
        train_images=len(train_loader.dataset),
        validation_images=len(val_loader.dataset),
        dataset_records_sha256=dataset["records_sha256"],
        camera=protocol["camera"],
        preprocessing=protocol["preprocessing"],
        spiral_only=True,
        source_weights=None,
        test_data_opened=False,
        random_training_rotation=True,
        smoke=smoke,
    )
    contract_file = destination / "contract.json"
    if contract_file.exists() and read_json(contract_file) != contract:
        raise ValueError("Existing training contract differs; do not mix resumed experiments")
    atomic_json(contract_file, contract)
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
        raise ValueError("Training did not complete its frozen epoch budget")
    path, _ = checkpoint_pointer(engine, fingerprint)
    final = torch.load(path, map_location="cpu", weights_only=True)
    best = min(final["history"], key=lambda r: (r["validation_component_mse_m2"], r["epoch"]))
    candidates = []
    for path in (engine / "checkpoints").glob("epoch_{:04d}_*.pt".format(best["epoch"])):
        candidate = torch.load(path, map_location="cpu", weights_only=True)
        if (
            candidate["fingerprint"] == fingerprint
            and candidate["history"] == final["history"][: best["epoch"]]
        ):
            candidates.append(candidate["model"])
    if not candidates or any(
        any(not torch.equal(candidate[k], candidates[0][k]) for k in candidate)
        for candidate in candidates[1:]
    ):
        raise ValueError("Missing or ambiguous validation-selected weights")
    atomic_torch(destination / "best_validation.pth", candidates[0])
    atomic_torch(destination / "final.pth", final["model"])
    history = []
    for row in final["history"]:
        physical = dict(row)
        for key in ("train_component_mse_m2", "validation_component_mse_m2"):
            physical[key] *= config["scale_m"] ** 2
        history.append(physical)
    atomic_json(destination / "physical_history.json", dict(epochs=history, scale_m=44.0))
    if not smoke:
        model.load_state_dict(candidates[0])
        physical_model = PhysicalHomeVector(model, config["scale_m"]).eval()
        examples = [r for r in records if r["split"] == "validation"][:2]
        raw = torch.stack(
            [load_rgb_tensor(PROJECT / r["image"], [1024, 1024], r["sha256"]) for r in examples]
        )
        exported = export_physical(physical_model, raw, destination / "deployment")
        exported.update(preprocessing=protocol["preprocessing"], random_rotation_at_inference=False)
        atomic_json(destination / "deployment/export_check.json", exported)
        with torch.inference_mode():
            cpu = physical_model.cpu().eval()
            expected = cpu(raw)
            actual = cpu(next(iter(val_loader))[0][:2])
        if not torch.allclose(actual, expected, atol=1e-5, rtol=1e-4):
            raise ValueError("Raw RGB deployment and dataset preprocessing differ")
    names = ["contract.json", "best_validation.pth", "final.pth", "physical_history.json"]
    if not smoke:
        names += [
            "deployment/home_vector_m.torchscript.pt",
            "deployment/home_vector_m.onnx",
            "deployment/export_check.json",
        ]
    summary = dict(
        status="completed",
        updated=now(),
        fingerprint=fingerprint,
        experiment_fingerprint=frozen["fingerprint"],
        smoke=smoke,
        epochs=config["epochs"],
        optimizer_steps=result["optimizer_steps"],
        expected_optimizer_steps=config["epochs"] * len(train_loader),
        best_validation=history[best["epoch"] - 1],
        final_validation=history[-1],
        fresh_initial_parameter_sha256=initial,
        spiral_only=True,
        test_data_opened=False,
        resources=result["resources"],
        files={name: digest(destination / name) for name in names},
    )
    if summary["optimizer_steps"] != summary["expected_optimizer_steps"]:
        raise ValueError("Unexpected number of optimizer updates")
    verify_freeze(folder)
    atomic_json(completion_file, summary)
    return summary
