"""Matched-budget circular-model fine tuning, common validation and frozen exports."""

from pathlib import Path

import torch

from insect_utils.circular_augmentation import CircularDataset
from insect_utils.circular_training import build_model
from insect_utils.continuation_training import import_checkpoint, require_runtime, states_equal
from insect_utils.coverage_model import PhysicalHomeVector
from insect_utils.coverage_training import export_physical
from insect_utils.downward_data import seed_worker
from insect_utils.downward_training import atomic_torch, runtime_signature
from insect_utils.experiment_config import PROJECT, canonical_hash, digest
from insect_utils.experiment_state import atomic_json, now, read_json
from insect_utils.local_probe_runtime import verify_capture
from insect_utils.near_finetune_engine import train_blocks
from insect_utils.near_finetune_protocol import verify_finetune
from insect_utils.rgb_preprocessing import load_rgb_tensor
from insect_utils.training_state import checkpoint_pointer


def validation_metrics(model, loader, scale=44):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    targets = []
    with torch.inference_mode():
        for images, truth in loader:
            output = model(images.to(device)).cpu() * scale
            if not torch.isfinite(output).all():
                raise ValueError("Non-finite validation prediction")
            predictions.append(output)
            targets.append(truth * scale)
    model.train(was_training)
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    radius = torch.linalg.vector_norm(targets, dim=1)
    error = (predictions - targets).square().mean(dim=1)
    masks = {
        "0-0.5": radius <= 0.5,
        "0.5-2": (radius > 0.5) & (radius <= 2),
        "2-5": (radius > 2) & (radius <= 5),
        "above5": radius > 5,
    }
    groups = {}
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError("Every declared validation distance band needs samples")
        groups[name] = dict(
            images=int(mask.sum()),
            component_mse_m2=float(error[mask].mean()),
            vector_error_mean_m=float(
                torch.linalg.vector_norm((predictions - targets)[mask], dim=1).mean()
            ),
        )
    return dict(
        images=len(targets),
        component_mse_m2=float(error.mean()),
        bands=groups,
        selection_score=sum(g["component_mse_m2"] for g in groups.values()) / 4,
    )


def select_state(engine, fingerprint, final):
    best = min(final["history"], key=lambda r: (r["selection_score"], r["block"]))
    candidates = []
    for path in (engine / "checkpoints").glob("epoch_{:04d}_*.pt".format(best["block"])):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload["fingerprint"] == fingerprint
            and payload["history"] == final["history"][: best["block"] + 1]
        ):
            candidates.append(payload["model"])
    if not candidates or any(not states_equal(candidates[0], s) for s in candidates[1:]):
        raise ValueError("Selected fine-tuning checkpoint is missing or ambiguous")
    return best, candidates[0]


def replay_probe(physical, run):
    probe = verify_capture(PROJECT / "runs/stage18_local_cross13_01")
    rows = []
    with torch.inference_mode():
        for source in probe["rows"]:
            raw = load_rgb_tensor(PROJECT / source["image"], [1024, 1024], source["image_sha256"])
            prediction = physical(raw[None])[0].tolist()
            from insect_utils.local_probe import vector_diagnostic

            row = vector_diagnostic(
                source["position_m"], source["body_yaw_deg"], prediction, [0, 0, 40]
            )
            row.update(position_id=source["position_id"], sample_id=source["sample_id"])
            rows.append(row)
    from insect_utils.local_probe import summarize_rows

    value = dict(
        rows=rows,
        summary=summarize_rows(rows, [0, 0, 40]),
        used_for_checkpoint_selection=False,
        limits=(
            "Existing development probes; the home location is now in training, "
            "so this is not an independent test."
        ),
    )
    atomic_json(run / "probe_replay.json", value)
    return value


def train_arm(folder, arm):
    folder = Path(folder)
    frozen = verify_finetune(folder)
    if arm not in frozen["spec"]["arms"]:
        raise ValueError("Unknown training arm")
    fingerprint = canonical_hash(dict(experiment=frozen["fingerprint"], arm=arm))
    destination = folder / "arms" / arm
    destination.mkdir(parents=True, exist_ok=True)
    done = destination / "completion.json"
    if done.exists():
        value = read_json(done)
        if (
            value["fingerprint"] != fingerprint
            or value["experiment_fingerprint"] != frozen["fingerprint"]
            or value["arm"] != arm
            or value["optimizer_steps_added"] != 25600
            or value["status"] != "completed"
        ):
            raise ValueError("Completed training belongs to another arm or budget")
        for name, sha in value["files"].items():
            if digest(destination / name) != sha:
                raise ValueError("Completed fine-tuning output changed")
        return value
    if not torch.cuda.is_available():
        raise RuntimeError("Formal fine tuning requires the original GPU")
    config = dict(
        frozen["training_config"],
        blocks=frozen["spec"]["blocks"],
        updates_per_block=frozen["spec"]["updates_per_block"],
    )
    torch.set_num_threads(config["torch_num_threads"])
    device = torch.device("cuda")
    runtime = runtime_signature(device)
    require_runtime(runtime, frozen["source_runtime"])
    dataset = read_json(PROJECT / frozen["spec"]["data_run"] / "dataset.json")
    if canonical_hash(dataset["records"]) != dataset["records_sha256"]:
        raise ValueError("Dataset changed")
    records = dataset["records"]
    train_records = [
        r
        for r in records
        if r["split"] != "train" or arm == "near_supplement" or r["origin"] == "original"
    ]
    train = CircularDataset(train_records, "train", 44, random_rotation=True)
    val = CircularDataset(records, "validation", 44, random_rotation=False)
    expected_count = 4096 if arm == "original_data" else 5640
    if len(train) != expected_count or len(val) != 448:
        raise ValueError("Arm training/validation membership changed")
    val_loader = torch.utils.data.DataLoader(
        val,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(10),
    )
    source = PROJECT / frozen["spec"]["source_checkpoint"]
    origin = torch.load(source, map_location="cpu", weights_only=True)
    source_weights = (
        PROJECT / frozen["spec"]["source_training_run"] / "training/best_validation.pth"
    )
    if not states_equal(
        origin["model"], torch.load(source_weights, map_location="cpu", weights_only=True)
    ):
        raise ValueError("Full589 checkpoint is not the validation-selected model")
    imported = import_checkpoint(
        source,
        destination / "source_engine",
        fingerprint,
        frozen["spec"]["source_training_fingerprint"],
        589,
        301568,
    )
    model = build_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], **config["optimizer_kwargs"]
    )
    engine = destination / "engine"
    atomic_json(
        destination / "contract.json",
        dict(
            fingerprint=fingerprint,
            arm=arm,
            configuration=config,
            runtime=runtime,
            train_images=len(train),
            validation_images=len(val),
            source_import=imported,
            phase="equal-update fine tuning",
            sampler=frozen["spec"]["sampling"],
            selection=frozen["spec"]["selection"],
            reserved_test_images_opened=False,
        ),
    )
    result = train_blocks(
        model,
        optimizer,
        train,
        val_loader,
        config,
        engine,
        fingerprint,
        origin,
        validation_metrics,
        resume=(engine / "checkpoint_latest.json").exists(),
    )
    if result["status"] != "completed" or result["optimizer_steps_added"] != 25600:
        raise ValueError("Fine-tuning budget is incomplete")
    path, _ = checkpoint_pointer(engine, fingerprint)
    final = torch.load(path, map_location="cpu", weights_only=True)
    selected, state = select_state(engine, fingerprint, final)
    atomic_torch(destination / "best_validation.pth", state)
    atomic_torch(destination / "final.pth", final["model"])
    atomic_json(
        destination / "selection_before_navigation.json",
        dict(
            selected_block=selected["block"],
            selection_score=selected["selection_score"],
            validation=selected["validation"],
            selected_by=frozen["spec"]["selection"],
            weights_sha256=digest(destination / "best_validation.pth"),
        ),
    )
    model.load_state_dict(state)
    physical = PhysicalHomeVector(model.cpu(), 44).eval()
    examples = [r for r in records if r["split"] == "validation"][:2]
    raw = torch.stack(
        [load_rgb_tensor(PROJECT / r["image"], [1024, 1024], r["sha256"]) for r in examples]
    )
    export = export_physical(physical, raw, destination / "deployment")
    export.update(fixed_circle_inside_export=True, random_rotation_at_inference=False)
    atomic_json(destination / "deployment/export_check.json", export)
    probe = replay_probe(physical, destination)
    names = [
        "best_validation.pth",
        "final.pth",
        "contract.json",
        "checkpoint_import.json",
        "selection_before_navigation.json",
        "probe_replay.json",
        "engine/training_history.json",
        "deployment/home_vector_m.torchscript.pt",
        "deployment/home_vector_m.onnx",
        "deployment/export_check.json",
    ]
    value = dict(
        status="completed",
        updated=now(),
        fingerprint=fingerprint,
        experiment_fingerprint=frozen["fingerprint"],
        arm=arm,
        optimizer_steps_added=25600,
        optimizer_steps_total=result["optimizer_steps_total"],
        selected_block=selected["block"],
        best_validation=selected["validation"],
        source_validation=final["history"][0]["validation"],
        final_validation=final["history"][-1]["validation"],
        probe_summary=probe["summary"],
        reserved_test_images_opened=False,
        resources=result["resources"],
        files={name: digest(destination / name) for name in names},
    )
    atomic_json(done, value)
    verify_finetune(folder)
    return value
