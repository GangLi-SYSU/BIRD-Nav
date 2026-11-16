"""Resumable fixed-update blocks with matched per-block sampling/augmentation seeds."""

from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from insect_utils.downward_data import seed_worker
from insect_utils.downward_training import restore_rng, rng_snapshot, gpu_resources
from insect_utils.experiment_state import atomic_json
from insect_utils.training_state import (
    ResourceWatch,
    assert_no_kit,
    checkpoint_pointer,
    memory_available,
    save_checkpoint,
)


def block_loader(dataset, config, block):
    count = config["updates_per_block"] * config["batch_size"]
    if len(dataset) < count:
        raise ValueError("A block needs enough distinct eligible images")
    seed = config["seed"] + 1000003 * block
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[
        :count
    ].tolist()
    return DataLoader(
        Subset(dataset, indices),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed + 1),
    )


def train_blocks(
    model,
    optimizer,
    dataset,
    val_loader,
    config,
    engine,
    fingerprint,
    origin,
    evaluate,
    resume=False,
    stop_after_block=None,
):
    assert_no_kit()
    engine = Path(engine)
    engine.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    first = 0
    steps = 0
    history = []
    bootstrap = block_loader(dataset, config, 1)
    pointer = engine / "checkpoint_latest.json"
    if pointer.exists():
        if not resume:
            raise ValueError("Existing fine-tuning checkpoint requires explicit resume")
        path, _ = checkpoint_pointer(engine, fingerprint)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        first = checkpoint["epoch"]
        steps = checkpoint["optimizer_steps"]
        history = checkpoint["history"]
        if (
            steps != first * config["updates_per_block"]
            or len(history) != first + 1
            or checkpoint["source_optimizer_steps"] != origin["optimizer_steps"]
        ):
            raise ValueError("Fine-tuning update accounting changed")
    else:
        if any((engine / "checkpoints").glob("*.pt")):
            raise ValueError("Uncommitted checkpoint files preserved; verify before resuming")
        checkpoint = origin
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng(checkpoint["rng"], [bootstrap, val_loader])

    def commit(block, loader):
        payload = dict(
            epoch=block,
            optimizer_steps=steps,
            source_epoch=origin["epoch"],
            source_optimizer_steps=origin["optimizer_steps"],
            total_optimizer_steps=origin["optimizer_steps"] + steps,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            history=history,
            rng=rng_snapshot([loader, val_loader], cuda),
        )
        save_checkpoint(engine, payload, fingerprint)
        atomic_json(engine / "training_history.json", dict(blocks=history))

    if not history:
        value = evaluate(model, val_loader)
        history.append(
            dict(
                block=0,
                optimizer_steps_added=0,
                train_component_mse_m2=None,
                validation_component_mse_m2=value["component_mse_m2"],
                selection_score=value["selection_score"],
                validation=value,
            )
        )
        commit(0, bootstrap)
    limit = (
        config["blocks"] if stop_after_block is None else min(config["blocks"], stop_after_block)
    )
    with ResourceWatch() as watch:
        for block in range(first + 1, limit + 1):
            loader = block_loader(dataset, config, block)
            model.train()
            started = time.perf_counter()
            total = 0.0
            seen = 0
            nonzero = 0
            max_gradient = 0.0
            for images, targets in loader:
                if memory_available() < config.get("minimum_host_available_bytes", 0):
                    raise RuntimeError("Low host memory; last complete block is retained")
                optimizer.zero_grad(set_to_none=True)
                output = model(images.to(device))
                loss = torch.nn.functional.mse_loss(output, targets.to(device))
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite fine-tuning loss")
                loss.backward()
                gradient = torch.cat(
                    [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
                )
                if not torch.isfinite(gradient).all():
                    raise ValueError("Non-finite gradient")
                norm = float(torch.linalg.vector_norm(gradient))
                max_gradient = max(max_gradient, norm)
                nonzero += norm > 0
                optimizer.step()
                steps += 1
                seen += len(images)
                total += float(loss.detach()) * len(images)
                if steps % 64 == 0:
                    atomic_json(
                        engine / "progress.json",
                        dict(
                            status="running",
                            block=block,
                            blocks=config["blocks"],
                            optimizer_steps_added=steps,
                            optimizer_steps_total=origin["optimizer_steps"] + steps,
                            block_images_seen=seen,
                        ),
                    )
            if seen != config["updates_per_block"] * config["batch_size"] or nonzero == 0:
                raise ValueError("Block has an incorrect sample count or no effective gradients")
            value = evaluate(model, val_loader)
            history.append(
                dict(
                    block=block,
                    optimizer_steps_added=steps,
                    train_component_mse_m2=total / seen * config["scale_m"] ** 2,
                    validation_component_mse_m2=value["component_mse_m2"],
                    selection_score=value["selection_score"],
                    validation=value,
                    elapsed_s=time.perf_counter() - started,
                    maximum_gradient_norm=max_gradient,
                )
            )
            commit(block, loader)
            atomic_json(
                engine / "progress.json",
                dict(history[-1], status="running", blocks=config["blocks"]),
            )
            print(
                (
                    "BLOCK {}/{} train_mse_m2={:.5f} val_mse_m2={:.5f} "
                    "selection={:.5f} added_updates={} seconds={:.2f}"
                ).format(
                    block,
                    config["blocks"],
                    history[-1]["train_component_mse_m2"],
                    value["component_mse_m2"],
                    value["selection_score"],
                    steps,
                    history[-1]["elapsed_s"],
                ),
                flush=True,
            )
    summary = dict(
        status="completed" if history[-1]["block"] == config["blocks"] else "interrupted",
        blocks_completed=history[-1]["block"],
        optimizer_steps_added=steps,
        optimizer_steps_total=origin["optimizer_steps"] + steps,
        resources=dict(watch.summary(), **gpu_resources(device)),
    )
    atomic_json(engine / "engine_summary.json", summary)
    atomic_json(engine / "progress.json", dict(summary, blocks=config["blocks"]))
    return summary
