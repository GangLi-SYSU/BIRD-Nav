"""Run the chosen450/600 total-epoch budget, then the unchanged20m64-trial evaluation."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

from insect_utils.epoch_extension import prepare_extension, prepare_navigation, verify_extension
from insect_utils.experiment_config import PROJECT, digest
from insect_utils.experiment_state import RunLock, atomic_json, now, read_json
from insect_utils.rotation_protocol import geometry_check
from run_navigation_experiment import verify_phase


def worker_process(folder, lock, phase, frozen):
    log = folder / "logs" / (phase + "_" + uuid.uuid4().hex[:10] + ".log")
    log.parent.mkdir(exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-dir",
        str(folder),
        "--worker",
        phase,
    ]
    print("START " + phase + " " + str(log), flush=True)
    with log.open("x") as stream:
        process = subprocess.Popen(
            command,
            cwd=PROJECT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            pass_fds=(lock.fileno(),),
            env=dict(
                os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", MPLBACKEND="Agg"
            ),
        )
        while process.poll() is None:
            path = folder / (
                "training/engine_scaled_units/progress.json"
                if phase == "train"
                else "navigation/progress.json"
            )
            detail = (
                read_json(path)
                if path.exists()
                else dict(status="starting", epoch=300, epochs=frozen["training_config"]["epochs"])
            )
            atomic_json(
                folder / "progress.json",
                dict(
                    status="running",
                    phase=phase,
                    updated=now(),
                    detail=detail,
                    fingerprint=frozen["fingerprint"],
                    log=str(log.relative_to(PROJECT)),
                ),
            )
            time.sleep(5)
    if process.returncode:
        raise RuntimeError(phase + " failed; read " + str(log))


def pipeline(folder, lock):
    frozen = verify_extension(folder)
    target = frozen["training_config"]["epochs"]
    if not (folder / "training/completion.json").exists():
        worker_process(folder, lock, "train", frozen)
    trained = read_json(folder / "training/completion.json")
    if (
        trained["status"] != "completed"
        or trained["fingerprint"] != frozen["fingerprint"]
        or trained["epochs_total"] != target
        or trained["optimizer_steps_total"] != target * 512
    ):
        raise ValueError("Training worker exited without the required complete budget")
    for name, expected in trained["files"].items():
        if digest(folder / "training" / name) != expected:
            raise ValueError("Training artifact changed")
    nav = prepare_navigation(folder)
    base = folder / "navigation"
    if not (base / "geometry/completion.json").exists():
        geometry_check(base, nav)
    verify_phase(base, "geometry", nav)
    phases = ("rendered_oracle", "model_check", "formal")
    if not all((base / phase / "completion.json").exists() for phase in phases):
        worker_process(folder, lock, "navigation", frozen)
    for phase in phases:
        verify_phase(base, phase, nav)
    from insect_utils.epoch_extension_report import report_extension

    report_extension(folder)
    verify_extension(folder)
    atomic_json(
        folder / "progress.json", dict(read_json(folder / "completion.json"), phase="completed")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--target-epochs", type=int, choices=(450, 600))
    parser.add_argument("--worker", choices=("train", "navigation"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    if args.worker:
        if args.worker == "train":
            from insect_utils.epoch_extension import train_extension

            train_extension(folder)
        else:
            from insect_utils.rotation_runtime import run_arm

            run_arm(folder / "navigation")
        return
    folder.mkdir(parents=True, exist_ok=True)
    with RunLock(folder) as lock:
        if args.plan_only:
            if args.target_epochs is None:
                parser.error("--plan-only requires an approved --target-epochs")
            value = prepare_extension(folder, args.target_epochs)
            print("PREPARED_EPOCH_EXTENSION " + value["fingerprint"], flush=True)
            return
        if args.target_epochs is not None:
            parser.error(
                "The target is read from the frozen plan; omit --target-epochs when running"
            )
        try:
            pipeline(folder, lock)
        except BaseException as error:
            atomic_json(
                folder / ("failure_" + uuid.uuid4().hex[:10] + ".json"),
                dict(error=str(error), traceback=traceback.format_exc(), updated=now()),
            )
            atomic_json(
                folder / "progress.json", dict(status="failed", updated=now(), error=str(error))
            )
            raise


if __name__ == "__main__":
    main()
