"""Original-model training, resumable epoch checkpoints and isolated evaluation."""

import gc
import os
from pathlib import Path
import random
import time
import uuid

from insect_utils.experiment_config import canonical_hash, digest
from insect_utils.experiment_state import atomic_json
from insect_utils.training_state import ResourceWatch, assert_no_kit, checkpoint_pointer
from insect_utils.training_state import (
    memory_available,
    position_subset,
    save_checkpoint,
    vector_metrics,
)


def seed_all(seed, cuda=False):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def runtime_signature(device):
    import torch

    result = {
        "python": __import__("sys").version.split()[0],
        "torch": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(gpu_name=properties.name, gpu_total_memory_bytes=properties.total_memory)
    return result


def rng_snapshot(loaders, cuda):
    import numpy as np
    import torch

    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if cuda else [],
        "loaders": [loader.generator.get_state() for loader in loaders],
    }


def restore_rng(state, loaders):
    import numpy as np
    import torch

    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if len(state["loaders"]) != len(loaders):
        raise ValueError("Checkpoint loader count mismatch")
    for loader, value in zip(loaders, state["loaders"]):
        loader.generator.set_state(value.cpu())


def atomic_torch(path, value):
    import torch

    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def gpu_resources(device):
    import torch

    return {
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else 0,
    }


def validation_loss(model, loader, device):
    import torch

    training = model.training
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for images, targets in loader:
            loss = torch.nn.functional.mse_loss(model(images.to(device)), targets.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite validation loss")
            total += float(loss) * len(images)
            count += len(images)
    model.train(training)
    if not count:
        raise ValueError("Validation dataset is empty")
    return total / count


def train_epochs(
    model,
    optimizer,
    train_loader,
    val_loader,
    config,
    folder,
    fingerprint,
    resume=False,
    stop_after_epoch=None,
):
    """Generic epoch engine; formal callers enforce the original network/protocol."""
    import torch
    from insect_utils.train_utils import parameter_digest

    assert_no_kit()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    history = []
    validation_events = []
    steps = 0
    start_epoch = 0
    nonzero_gradients = 0
    initial_validation = None
    initial = parameter_digest(model)
    pointer_file = folder / "checkpoint_latest.json"
    if pointer_file.exists():
        if not resume:
            raise ValueError("Checkpoint exists; explicitly use --resume")
        path, pointer = checkpoint_pointer(folder, fingerprint)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["fingerprint"] != fingerprint or checkpoint["epoch"] != pointer["epoch"]:
            raise ValueError("Checkpoint payload fingerprint/epoch mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint["history"]
        validation_events = checkpoint["validation_events"]
        start_epoch = checkpoint["epoch"]
        steps = checkpoint["optimizer_steps"]
        nonzero_gradients = checkpoint["nonzero_gradient_updates"]
        initial = checkpoint["initial_parameter_sha256"]
        initial_validation = checkpoint["initial_validation_component_mse_m2"]
        if len(history) != start_epoch or steps != start_epoch * len(train_loader):
            raise ValueError("Checkpoint history/optimizer step count mismatch")
        restore_rng(checkpoint["rng"], [train_loader, val_loader])
    elif resume and any((folder / "checkpoints").glob("*.pt")):
        raise ValueError("No committed checkpoint pointer; preserved orphan checkpoint files")
    epochs = config["epochs"]
    if start_epoch > epochs:
        raise ValueError("Checkpoint exceeds configured epoch budget")
    limit = epochs if stop_after_epoch is None else min(epochs, stop_after_epoch)
    if initial_validation is None:
        initial_validation = validation_loss(model, val_loader, device)
    with ResourceWatch() as watch:
        for epoch in range(start_epoch, limit):
            model.train()
            started = time.perf_counter()
            total = 0.0
            seen = 0
            gradient_max = 0.0
            for images, targets in train_loader:
                if memory_available() < config.get("minimum_host_available_bytes", 0):
                    raise RuntimeError(
                        "Insufficient host memory; last complete epoch checkpoint retained"
                    )
                optimizer.zero_grad(set_to_none=True)
                output = model(images.to(device))
                loss = torch.nn.functional.mse_loss(output, targets.to(device))
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                gradients = torch.cat(
                    [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
                )
                if not torch.isfinite(gradients).all():
                    raise ValueError("Non-finite gradient")
                norm = float(torch.linalg.vector_norm(gradients))
                gradient_max = max(gradient_max, norm)
                nonzero_gradients += norm > 0
                optimizer.step()
                steps += 1
                seen += len(images)
                total += float(loss.detach()) * len(images)
                interval = config.get("validation_interval_updates", 25)
                if interval and steps % interval == 0:
                    value = validation_loss(model, val_loader, device)
                    validation_events.append(
                        {"optimizer_steps": steps, "validation_component_mse_m2": value}
                    )
                    atomic_json(
                        folder / "progress.json",
                        {
                            "status": "running",
                            "epoch": epoch + 1,
                            "epochs": epochs,
                            "optimizer_steps": steps,
                            "epoch_samples_seen": seen,
                            "validation_component_mse_m2": value,
                        },
                    )
                    print(
                        "UPDATE steps={} epoch={}/{} validation_mse={:.6f}".format(
                            steps, epoch + 1, epochs, value
                        ),
                        flush=True,
                    )
            if seen != len(train_loader.dataset):
                raise ValueError("Training loader dropped or duplicated samples")
            value = validation_loss(model, val_loader, device)
            item = {
                "epoch": epoch + 1,
                "optimizer_steps": steps,
                "training_samples": seen,
                "train_component_mse_m2": total / seen,
                "validation_component_mse_m2": value,
                "epoch_elapsed_s": time.perf_counter() - started,
                "maximum_gradient_l2_norm": gradient_max,
                **gpu_resources(device),
            }
            history.append(item)
            payload = {
                "epoch": epoch + 1,
                "optimizer_steps": steps,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "validation_events": validation_events,
                "initial_parameter_sha256": initial,
                "nonzero_gradient_updates": nonzero_gradients,
                "initial_validation_component_mse_m2": initial_validation,
                "rng": rng_snapshot([train_loader, val_loader], cuda),
            }
            save_checkpoint(folder, payload, fingerprint)
            atomic_json(
                folder / "training_history.json",
                {"epochs": history, "validation_events": validation_events},
            )
            atomic_json(folder / "progress.json", dict(item, status="running", epochs=epochs))
            print(
                "EPOCH {}/{} train_mse={:.6f} validation_mse={:.6f} steps={} seconds={:.2f}".format(
                    epoch + 1, epochs, total / seen, value, steps, item["epoch_elapsed_s"]
                ),
                flush=True,
            )
    completed = len(history) == epochs
    if completed and (not nonzero_gradients or parameter_digest(model) == initial):
        raise ValueError("No effective gradient/parameter update occurred")
    summary = {
        "status": "completed" if completed else "interrupted",
        "epochs_completed": len(history),
        "optimizer_steps": steps,
        "nonzero_gradient_updates": nonzero_gradients,
        "initial_parameter_sha256": initial,
        "final_parameter_sha256": parameter_digest(model),
        "initial_validation_component_mse_m2": initial_validation,
        "resources": dict(watch.summary(), **gpu_resources(device)),
        "final_train_component_mse_m2": history[-1]["train_component_mse_m2"] if history else None,
        "final_validation_component_mse_m2": history[-1]["validation_component_mse_m2"]
        if history
        else None,
    }
    atomic_json(folder / "engine_summary.json", summary)
    atomic_json(
        folder / "progress.json",
        {
            "status": summary["status"],
            "epoch": len(history),
            "epochs": epochs,
            "optimizer_steps": steps,
            "train_component_mse_m2": summary["final_train_component_mse_m2"],
            "validation_component_mse_m2": summary["final_validation_component_mse_m2"],
        },
    )
    return summary


def loaders_for(config, positions=None, workers=None):
    import torch
    from data_loader import get_data_loaders
    from insect_utils.downward_data import seed_worker

    cfg = dict(config)
    if workers is not None:
        cfg["num_workers"] = workers
    loaded = get_data_loaders(cfg)
    train_loader, val_loader = loaded[:2]
    groups = None
    if positions is not None:
        records = [train_loader.dataset.record(i) for i in range(len(train_loader.dataset))]
        indices, groups = position_subset(records, positions)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_loader.dataset, indices),
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(cfg["seed"]),
        )
    return train_loader, val_loader, groups


def train_from_config(config):
    import torch
    from insect_utils.attention_model import create_attention, model_contract
    from insect_utils.train_utils import parameter_digest

    required = ["training_run_dir", "training_fingerprint", "training_purpose"]
    if any(key not in config for key in required):
        raise ValueError("Use run_downward_training.py with verified stage F data")
    assert_no_kit()
    torch.set_num_threads(config["torch_num_threads"])
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training run")
    seed_all(config["seed"], device.type == "cuda")
    model = create_attention(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], **config["optimizer_kwargs"]
    )
    positions = config.get("position_subset_count")
    train_loader, val_loader, groups = loaders_for(config, positions)
    signature = canonical_hash(
        {"training": config["training_fingerprint"], "runtime": runtime_signature(device)}
    )
    folder = Path(config["training_run_dir"])
    atomic_json(
        folder / "effective_config.json",
        dict(config, runtime=runtime_signature(device), selected_position_groups=groups),
    )
    summary = train_epochs(
        model,
        optimizer,
        train_loader,
        val_loader,
        config,
        folder,
        signature,
        resume=config.get("resume_checkpoint", False),
    )
    if summary["status"] != "completed":
        raise ValueError("Training did not reach its epoch budget")
    # The delivered state_dict is always the final epoch; no test-based selection.
    weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    filename = folder / "model_attention.pth"
    atomic_torch(filename, weights)
    reference_images, _ = next(iter(val_loader))
    reference_images = reference_images.to(device)
    model.eval()
    with torch.inference_mode():
        before = model(reference_images).cpu()
    loaded = create_attention(config).to(device).eval()
    loaded.load_state_dict(torch.load(filename, map_location="cpu", weights_only=True))
    with torch.inference_mode():
        after = loaded(reference_images).cpu()
    maximum = float((before - after).abs().max())
    if not torch.allclose(before, after, rtol=1e-6, atol=1e-6):
        raise ValueError("Saved/reloaded model predictions differ")
    summary.update(
        purpose=config["training_purpose"],
        model_parameters=model_contract(config)["parameters"],
        network_contract=model_contract(config),
        model_selection="last_epoch",
        model_sha256=digest(filename),
        reload_max_absolute_difference=maximum,
        loaded_parameter_sha256=parameter_digest(loaded),
        runtime=runtime_signature(device),
        kit_modules=[],
    )
    atomic_json(folder / "summary.json", summary)
    assert_no_kit()
    return str(filename)


def probe_workers(config, protocol, folder):
    import hashlib
    import torch
    from insect_utils.attention_model import create_attention
    from insect_utils.train_utils import parameter_digest

    results = []
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    device = torch.device(config["device"])
    torch.set_num_threads(config["torch_num_threads"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Probe requires the training GPU")
    for workers in protocol["engineering"]["probe_workers"]:
        seed_all(config["seed"], True)
        model = create_attention(config).to(device)
        initial = parameter_digest(model)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"], **config["optimizer_kwargs"]
        )
        loader, _, groups = loaders_for(config, protocol["engineering"]["probe_positions"], workers)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        steps = 0
        order = hashlib.sha256()
        with ResourceWatch() as watch:
            for images, targets in loader:
                if memory_available() < protocol["engineering"]["minimum_host_available_bytes"]:
                    raise RuntimeError("Insufficient host memory during loader probe")
                order.update(targets.numpy().tobytes())
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(images.to(device)), targets.to(device))
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loader probe loss")
                loss.backward()
                optimizer.step()
                steps += 1
        result = {
            "num_workers": workers,
            "optimizer_steps": steps,
            "engineering_positions": groups,
            "initial_parameter_sha256": initial,
            "batch_target_order_sha256": order.hexdigest(),
            "elapsed_s": time.perf_counter() - started,
            "resources": dict(watch.summary(), **gpu_resources(device)),
        }
        results.append(result)
        atomic_json(folder / "probe_report.json", {"status": "running", "results": results})
        print(
            "PROBE workers={} steps={} seconds={:.2f}".format(workers, steps, result["elapsed_s"]),
            flush=True,
        )
        del loader, optimizer, model, images, targets, loss
        gc.collect()
        torch.cuda.empty_cache()
    if len({r["batch_target_order_sha256"] for r in results}) != 1:
        raise ValueError("Worker count changed training batch order")
    report = {
        "status": "passed",
        "results": results,
        "selected_workers": config["num_workers"],
        "selection": "Retain the original four workers after the resource probe passed",
        "engineering_optimizer_steps": sum(r["optimizer_steps"] for r in results),
        "kit_modules": [],
    }
    atomic_json(folder / "probe_report.json", report)
    assert_no_kit()
    return report


def export_onnx_reference(weights, images, folder, protocol):
    import numpy as np
    import onnx
    from onnx.reference import ReferenceEvaluator
    import torch
    from insect_utils.attention_model import create_attention, model_contract

    model_config = {
        "network_variant": protocol["training"].get("network_variant", "panorama192x1800")
    }
    model = create_attention(model_config).eval()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    images = images[:2].cpu()
    destination = Path(folder) / "model_attention.onnx"
    temporary = destination.with_name(destination.name + ".tmp")
    torch.onnx.export(
        model,
        images[:1],
        temporary,
        opset_version=protocol["delivery"]["onnx_opset"],
        input_names=["rgb"],
        output_names=["home_vector_body_m"],
        dynamic_axes={"rgb": {0: "batch"}, "home_vector_body_m": {0: "batch"}},
    )
    graph = onnx.load(str(temporary))
    onnx.checker.check_model(graph, full_check=True)
    with torch.inference_mode():
        expected = model(images).numpy()
    actual = ReferenceEvaluator(graph).run(None, {"rgb": images.numpy()})[0]
    atol = protocol["delivery"]["onnx_reference_atol"]
    rtol = protocol["delivery"]["onnx_reference_rtol"]
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError("ONNX/PyTorch CPU prediction mismatch")
    os.replace(temporary, destination)
    result = {
        "status": "passed",
        "evaluator": "onnx.reference.ReferenceEvaluator",
        "samples": len(images),
        "max_absolute_difference": float(np.abs(actual - expected).max()),
        "atol": atol,
        "rtol": rtol,
        "onnx_sha256": digest(destination),
        "input_shape": ["batch", 3, 1024, 1024],
        "network_contract": model_contract(model_config),
    }
    atomic_json(Path(folder) / "onnx_check.json", result)
    return result


def evaluate_final(config, weights, folder, protocol):
    import numpy as np
    import torch
    from insect_utils.attention_model import create_attention, model_contract
    from insect_utils.data_preparation import load_prepared
    from insect_utils.downward_data import evaluate_prepared
    from data_loader import get_data_loaders

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    device = torch.device(config["device"])
    torch.set_num_threads(config["torch_num_threads"])
    model = create_attention(config).to(device).eval()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    context = load_prepared(config["prepared_data_dir"])
    cfg = dict(config, diagnostic_dir=str(folder / "plots"))
    metrics = {}
    for split in ["train", "validation", "test"]:
        prediction, target, _, _, _ = evaluate_prepared(model, cfg, split, graphics=True)
        metrics[split] = vector_metrics(prediction.tolist(), target.tolist())
        indices = context["split"]["indices"][split]
        ids = np.asarray([context["records"][i]["sample_id"] for i in indices])
        np.savez_compressed(
            folder / ("predictions_" + split + ".npz"),
            sample_ids=ids,
            predictions=prediction,
            targets=target,
        )
    val_loader = get_data_loaders(config)[1]
    images, _ = next(iter(val_loader))
    conversion = export_onnx_reference(weights, images, folder, protocol)
    report = {
        "status": "passed",
        "model_sha256": digest(weights),
        "model_selection": "last_epoch150",
        "network_contract": model_contract(config),
        "metrics_within_learning_region": metrics,
        "spatial_holdout": {
            "status": "not_evaluated",
            "reason": "User-selected position groups have no spatial buffer",
        },
        "outside_learning_region": {"status": "not_evaluated", "next_stage": "H/I"},
        "onnx": conversion,
        "closed_loop_homing_run": False,
        "kit_modules": [],
    }
    atomic_json(folder / "evaluation.json", report)
    assert_no_kit()
    return report
