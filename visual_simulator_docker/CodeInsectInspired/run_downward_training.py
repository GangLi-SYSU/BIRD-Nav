"""Run isolated loader probe, smoke training, formal training and final evaluation."""

import argparse
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys

from insect_utils.data_preparation import load_prepared, project_path
from insect_utils.experiment_config import PROJECT, canonical_hash, digest
from insect_utils.experiment_state import (
    RunLock,
    atomic_json,
    collect_artifacts,
    read_json,
    verify_artifacts,
)
from insect_utils.training_state import TRAINING_PROTOCOL


STAGES = ("probe", "smoke", "train", "evaluate")
DEPENDENCIES = {
    "probe": (),
    "smoke": ("probe",),
    "train": ("probe", "smoke"),
    "evaluate": ("train",),
}
SOURCES = (
    "run_downward_training.py",
    "insect_utils/downward_training.py",
    "insect_utils/training_state.py",
    "train.py",
    "data_loader.py",
    "insect_utils/train_utils.py",
    "simple_network.py",
    "insect_utils/data_preparation.py",
    "insect_utils/downward_data.py",
    "insect_utils/rgb_preprocessing.py",
    "insect_utils/attention_model.py",
)
DEFAULT_PREPARED = PROJECT / "runs/stage06_data/position_groups_01"


def configuration(prepared, protocol_path=None):
    context = load_prepared(prepared)
    report = read_json(prepared / "stage06_report.json")
    if (
        report["status"] != "passed"
        or report["preparation_fingerprint"] != context["manifest"]["fingerprint"]
    ):
        raise ValueError("Stage F data acceptance is incomplete")
    protocol_path = project_path(protocol_path or TRAINING_PROTOCOL).resolve()
    protocol = read_json(protocol_path)
    config = read_json(prepared / "data_config.json")
    for key in [
        "seed",
        "model_type",
        "criterion",
        "optimizer",
        "learning_rate",
        "optimizer_kwargs",
        "batch_size",
        "epochs",
        "num_workers",
    ]:
        if config[key] != protocol["training"][key]:
            raise ValueError("Original training setting changed: " + key)
    if config["augment"]:
        raise ValueError("Downward panorama augmentation remains disabled")
    config["network_variant"] = protocol["training"].get("network_variant", "panorama192x1800")
    versions = {
        name: importlib.metadata.version(name)
        for name in ["torch", "torchvision", "numpy", "Pillow", "onnx"]
    }
    payload = {
        "prepared_data_dir": str(prepared.relative_to(PROJECT)),
        "preparation_fingerprint": context["manifest"]["fingerprint"],
        "preparation_manifest_sha256": digest(prepared / "preparation_manifest.json"),
        "data_acceptance_sha256": digest(prepared / "stage06_report.json"),
        "training_config": config,
        "training_protocol": protocol,
        "training_protocol_sha256": digest(protocol_path),
        "training_protocol_file": str(protocol_path.relative_to(PROJECT)),
        "source_sha256": {name: digest(PROJECT / name) for name in SOURCES},
        "runtime_packages": versions,
        "device": "cuda:0",
    }
    return dict(payload, fingerprint=canonical_hash(payload))


def verified_manifest(run, current=None):
    stored = read_json(run / "run_manifest.json")
    if (
        canonical_hash({k: v for k, v in stored.items() if k != "fingerprint"})
        != stored["fingerprint"]
    ):
        raise ValueError("Training run manifest changed")
    if current is not None and current["fingerprint"] != stored["fingerprint"]:
        raise ValueError(
            "Training data,code,protocol or runtime changed; use a new output directory"
        )
    return stored


def verify_stage(run, stage):
    state = read_json(run / "state" / (stage + ".json"))
    manifest = verified_manifest(run)
    if state["status"] != "completed" or state["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Required training phase is not completed: " + stage)
    verify_artifacts(run, state["artifacts"])
    return state


def stage_config(run, stage, manifest, resume):
    protocol = manifest["training_protocol"]
    config = dict(manifest["training_config"])
    folder = run / stage
    config.update(
        training_run_dir=str(folder),
        models_folder=str(folder),
        training_purpose="engineering_smoke" if stage == "smoke" else "research",
        torch_num_threads=protocol["training"]["torch_num_threads"],
        device=manifest["device"],
        minimum_host_available_bytes=protocol["engineering"]["minimum_host_available_bytes"],
        validation_interval_updates=protocol["training"]["validation_interval_updates"],
        resume_checkpoint=resume,
    )
    if stage == "smoke":
        config.update(
            epochs=protocol["engineering"]["smoke_epochs"],
            position_subset_count=protocol["engineering"]["smoke_positions"],
        )
    fingerprint = canonical_hash(
        {
            "run": manifest["fingerprint"],
            "stage": stage,
            "epochs": config["epochs"],
            "positions": config.get("position_subset_count"),
        }
    )
    config["training_fingerprint"] = fingerprint
    return config


def worker(run, stage, resume):
    stored = verified_manifest(run)
    current = configuration(
        project_path(stored["prepared_data_dir"]), stored.get("training_protocol_file")
    )
    manifest = verified_manifest(run, current)
    state = read_json(run / "state" / (stage + ".json"))
    if state["status"] != "running" or state["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Worker must be started by the matching training controller")
    config = stage_config(run, stage, manifest, resume)
    protocol = manifest["training_protocol"]
    folder = run / stage
    from insect_utils.training_state import assert_no_kit
    from insect_utils.downward_training import probe_workers, evaluate_final, export_onnx_reference

    assert_no_kit()
    from insect_utils.attention_model import model_contract

    if model_contract(config)["parameters"] != protocol["training"]["model_parameters"]:
        raise ValueError("Model variant and declared parameter count differ")
    if stage == "probe":
        probe_workers(config, protocol, folder)
    elif stage in ("smoke", "train"):
        import train

        filename = train.train_model(config, only_training=True)
        if stage == "smoke":
            from data_loader import get_data_loaders

            images, _ = next(iter(get_data_loaders(config)[1]))
            export_onnx_reference(Path(filename), images, folder, protocol)
    else:
        evaluate_final(config, run / "train/model_attention.pth", folder, protocol)
    assert_no_kit()
    verified_manifest(
        run,
        configuration(
            project_path(manifest["prepared_data_dir"]), manifest.get("training_protocol_file")
        ),
    )
    print("STAGE07_PHASE_PASS " + stage, flush=True)


def execute(run, stage, manifest, resume, lock):
    state_path = run / "state" / (stage + ".json")
    previous = read_json(state_path) if state_path.exists() else None
    if previous and previous["status"] == "completed":
        if not resume:
            raise ValueError("Phase already completed; explicitly use --resume")
        verify_stage(run, stage)
        print("Verified/reused " + stage, flush=True)
        return
    if previous and not resume:
        raise ValueError("Interrupted phase; explicitly use --resume")
    for dependency in DEPENDENCIES[stage]:
        verify_stage(run, dependency)
    attempt = previous["attempt"] + 1 if previous else 1
    folder = run / stage
    if folder.exists() and (
        stage in ("probe", "evaluate") or not (folder / "checkpoint_latest.json").exists()
    ):
        archive = (
            run
            / "attempts"
            / (stage + "_" + str(attempt - 1) + "_" + __import__("uuid").uuid4().hex[:6])
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(folder), str(archive))
    folder.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "attempt": attempt, "fingerprint": manifest["fingerprint"]}
    atomic_json(state_path, state)
    log = run / "logs" / ("{}_{:02d}.log".format(stage, attempt))
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output",
        str(run),
        "--worker",
        stage,
    ]
    if resume:
        command.append("--resume")
    print("Running {} in an isolated process; log: {}".format(stage, log), flush=True)
    try:
        with log.open("x") as stream:
            result = subprocess.run(
                command,
                cwd=str(PROJECT),
                env=dict(
                    os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", MPLBACKEND="Agg"
                ),
                stdout=stream,
                stderr=subprocess.STDOUT,
                pass_fds=(lock.fileno(),),
            )
        if result.returncode:
            raise RuntimeError(
                "{} worker exited with {}; read {}".format(stage, result.returncode, log)
            )
        state.update(
            status="completed",
            artifacts=collect_artifacts(folder, run),
            log=str(log.relative_to(run)),
            log_sha256=digest(log),
        )
        atomic_json(state_path, state)
        print("Completed " + stage, flush=True)
    except BaseException as error:
        state.update(status="failed", error=str(error))
        atomic_json(state_path, state)
        raise


def finish(run):
    for stage in STAGES:
        verify_stage(run, stage)
    formal = read_json(run / "train/summary.json")
    smoke = read_json(run / "smoke/summary.json")
    probe = read_json(run / "probe/probe_report.json")
    evaluation = read_json(run / "evaluate/evaluation.json")
    if formal["epochs_completed"] != 150 or formal["optimizer_steps"] != 19950:
        raise ValueError("Formal training did not complete the original budget")
    if smoke["initial_parameter_sha256"] != formal["initial_parameter_sha256"]:
        raise ValueError("Smoke/formal training did not start from the same fresh initialization")
    report = {
        "status": "passed",
        "model_for_stage_h": "train/model_attention.pth",
        "model_sha256": digest(run / "train/model_attention.pth"),
        "epochs": 150,
        "optimizer_steps": 19950,
        "engineering_optimizer_steps": probe["engineering_optimizer_steps"]
        + smoke["optimizer_steps"],
        "model_parameters": formal["model_parameters"],
        "network_contract": formal["network_contract"],
        "model_selection": "last_epoch150",
        "smoke_weights_reused": False,
        "metrics_within_learning_region": evaluation["metrics_within_learning_region"],
        "onnx": evaluation["onnx"],
        "closed_loop_homing_run": False,
        "next_stage": "H: oracle and network closed-loop integration",
    }
    atomic_json(run / "result.json", report)
    print("STAGE07_TRAINING_PASS " + str(run), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--protocol", type=Path, help="Explicit approved training protocol")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "status") + STAGES, default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--worker", choices=STAGES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    run = project_path(args.output).resolve()
    if args.worker:
        worker(run, args.worker, args.resume)
        return
    if args.stage == "status":
        verified_manifest(run)
        for stage in STAGES:
            path = run / "state" / (stage + ".json")
            print(stage, read_json(path)["status"] if path.exists() else "pending")
        return
    if args.verify_only:
        for stage in STAGES:
            verify_stage(run, stage)
        result = read_json(run / "result.json")
        if result["status"] != "passed" or result["model_sha256"] != digest(
            run / "train/model_attention.pth"
        ):
            raise ValueError("Training acceptance/model handoff is incomplete or changed")
        print("STAGE07_VERIFIED " + str(run))
        return
    manifest = configuration(project_path(args.prepared_data).resolve(), args.protocol)
    if not run.exists():
        run.mkdir(parents=True, exist_ok=False)
        (run / "state").mkdir()
        (run / "logs").mkdir()
        atomic_json(run / "run_manifest.json", manifest)
    elif not args.resume:
        raise ValueError("Output exists; use a new directory or explicit --resume")
    with RunLock(run) as lock:
        verified_manifest(run, manifest)
        for stage in STAGES if args.stage == "all" else (args.stage,):
            execute(run, stage, manifest, args.resume, lock)
        if all(
            (run / "state" / (s + ".json")).exists()
            and read_json(run / "state" / (s + ".json"))["status"] == "completed"
            for s in STAGES
        ):
            finish(run)


if __name__ == "__main__":
    main()
