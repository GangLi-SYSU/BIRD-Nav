"""Predeclare an equal-update data control while the approved capture finishes."""

from pathlib import Path

from insect_utils.coverage_data import audit_records
from insect_utils.experiment_config import PROJECT, canonical_hash, digest
from insect_utils.experiment_state import atomic_json, read_json
from insect_utils.navigation_protocol import verify_navigation


PROTOCOL = "config/experiments/stage20_near_finetune_protocol.json"


def declare(folder):
    folder = Path(folder).resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    spec = read_json(PROJECT / PROTOCOL)
    data = verify_navigation(PROJECT / spec["data_run"])
    source = read_json(PROJECT / spec["source_training_run"] / "frozen.json")
    if (
        data["fingerprint"] != spec["data_fingerprint"]
        or source["fingerprint"] != spec["source_training_fingerprint"]
        or digest(PROJECT / spec["source_checkpoint"]) != spec["source_checkpoint_sha256"]
    ):
        raise ValueError("Declared data or source checkpoint changed")
    if (
        spec["blocks"] * spec["updates_per_block"] != 25600
        or spec["batch_size"] != 8
        or spec["learning_rate"] != 0.0005
        or spec["arms"] != ["original_data", "near_supplement"]
    ):
        raise ValueError("The equal-update recommended budget changed")
    config = dict(source["training_config"])
    for key in (
        "batch_size",
        "learning_rate",
        "num_workers",
        "torch_num_threads",
        "scale_m",
        "seed",
    ):
        if config[key] != spec[key]:
            raise ValueError("Original training setting changed: " + key)
    config.pop("epochs")
    files = dict(data["files"])
    extras = [
        PROJECT / PROTOCOL,
        PROJECT / spec["source_checkpoint"],
        PROJECT / spec["data_run"] / "frozen.json",
        PROJECT / spec["source_training_run"] / "training/contract.json",
        PROJECT / "run_near_finetune.py",
        PROJECT / "tests/test_stage20_near_finetune.py",
    ]
    extras += [
        PROJECT / "insect_utils" / n
        for n in (
            "near_finetune_protocol.py",
            "near_finetune_engine.py",
            "near_finetune_training.py",
            "near_finetune_report.py",
        )
    ]
    files.update({str(p.relative_to(PROJECT)): digest(p) for p in extras})
    value = dict(
        spec=spec,
        training_config=config,
        files=files,
        source_runtime=read_json(PROJECT / spec["source_training_run"] / "training/contract.json")[
            "runtime"
        ],
    )
    value["fingerprint"] = canonical_hash(value)
    path = folder / "predeclared.json"
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Fine-tuning is already declared; use a new run")
    else:
        atomic_json(path, value)
    return value


def verify_declared(folder):
    value = read_json(Path(folder) / "predeclared.json")
    if (
        canonical_hash({k: v for k, v in value.items() if k != "fingerprint"})
        != value["fingerprint"]
    ):
        raise ValueError("Fine-tuning declaration changed")
    for name, sha in value["files"].items():
        if digest(PROJECT / name) != sha:
            raise ValueError("Declared fine-tuning source changed: " + name)
    return value


def freeze_ready_data(folder):
    folder = Path(folder)
    declared = verify_declared(folder)
    base = PROJECT / declared["spec"]["data_run"]
    data = verify_navigation(base)
    completed = read_json(base / "completion.json")
    if completed["status"] != "completed" or completed["fingerprint"] != data["fingerprint"]:
        raise ValueError("Data capture and preparation must finish before training")
    for name, sha in completed["files"].items():
        if digest(base / name) != sha:
            raise ValueError("Completed near-data artifact changed")
    dataset = read_json(base / "dataset.json")
    if canonical_hash(dataset["records"]) != dataset["records_sha256"]:
        raise ValueError("Combined dataset records changed")
    audit_records(dataset["records"])
    counts = [sum(r["split"] == s for r in dataset["records"]) for s in ("train", "validation")]
    if counts != [5640, 448] or dataset["reserved_positions_captured"]:
        raise ValueError("Combined data partitions differ from the approved plan")
    files = dict(declared["files"])
    for p in (folder / "predeclared.json", base / "completion.json", base / "dataset.json"):
        files[str(p.relative_to(PROJECT))] = digest(p)
    value = dict(
        declared=declared,
        spec=declared["spec"],
        training_config=declared["training_config"],
        source_runtime=declared["source_runtime"],
        files=files,
    )
    value["fingerprint"] = canonical_hash(value)
    path = folder / "frozen.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("Fine-tuning data is already frozen")
    if not path.exists():
        atomic_json(path, value)
    return value


def verify_finetune(folder):
    return verify_navigation(folder)


def prepare_navigation(folder, arm):
    folder = Path(folder)
    frozen = verify_finetune(folder)
    if arm not in frozen["spec"]["arms"]:
        raise ValueError("Unknown fine-tuning arm")
    selection = read_json(folder / "selection_frozen_before_navigation.json")
    if selection["fingerprint"] != frozen["fingerprint"]:
        raise ValueError("Model selections changed")
    source = verify_navigation(PROJECT / frozen["spec"]["navigation_source"])
    files = dict(frozen["files"])
    for name in frozen["spec"]["arms"]:
        trained = folder / "arms" / name
        completed = read_json(trained / "completion.json")
        if (
            completed != selection["arms"][name]
            or completed["optimizer_steps_added"] != 25600
            or completed["status"] != "completed"
        ):
            raise ValueError("Both full fine-tuning budgets must finish before navigation")
        for path, sha in completed["files"].items():
            file = trained / path
            if digest(file) != sha:
                raise ValueError("Selected fine-tuning artifact changed")
            files[str(file.relative_to(PROJECT))] = sha
    for p in (folder / "frozen.json", folder / "selection_frozen_before_navigation.json"):
        files[str(p.relative_to(PROJECT))] = digest(p)
    base = folder / "arms" / arm
    value = dict(
        protocol=source["protocol"],
        episodes=source["episodes"],
        budget=source["budget"],
        files=files,
        model=dict(
            arm=arm,
            epoch=selection["arms"][arm]["selected_block"],
            scale_m=44,
            weights=str((base / "best_validation.pth").relative_to(PROJECT)),
            torchscript=str(
                (base / "deployment/home_vector_m.torchscript.pt").relative_to(PROJECT)
            ),
            output_units="body-frame home vector in meters",
        ),
    )
    value["fingerprint"] = canonical_hash(value)
    path = folder / "navigation" / arm / "frozen.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("Navigation selection is already frozen")
    if not path.exists():
        atomic_json(path, value)
    return value
