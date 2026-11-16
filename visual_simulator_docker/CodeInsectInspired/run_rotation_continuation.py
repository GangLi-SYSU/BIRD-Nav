"""Train151..300 and automatically repeat exactly64 navigation trials at20m."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

from insect_utils.continuation_protocol import (
    prepare_continuation,
    prepare_navigation,
    verify_continuation,
)
from insect_utils.experiment_config import PROJECT, digest
from insect_utils.experiment_state import RunLock, atomic_json, now, read_json
from insect_utils.rotation_protocol import geometry_check
from run_navigation_experiment import verify_phase


def check_training(folder, frozen):
    path = folder / "training/completion.json"
    if not path.exists():
        raise RuntimeError("Training worker exited without a completion receipt")
    value = read_json(path)
    if (
        value["status"] != "completed"
        or value["fingerprint"] != frozen["fingerprint"]
        or value["epochs_total"] != 300
        or value["optimizer_steps_total"] != 153600
    ):
        raise ValueError("Training receipt does not match the300-epoch plan")
    for name, expected in value["files"].items():
        if digest(folder / "training" / name) != expected:
            raise ValueError("Continued training artifact changed")


def run_worker(folder, lock, phase, frozen):
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
            state_file = (
                folder / "training/engine_scaled_units/progress.json"
                if phase == "train"
                else folder / "navigation/progress.json"
            )
            detail = (
                read_json(state_file)
                if state_file.exists()
                else dict(status="starting", epoch=150, epochs=300)
            )
            atomic_json(
                folder / "progress.json",
                dict(
                    status="running",
                    phase=phase,
                    updated=now(),
                    detail=detail,
                    log=str(log.relative_to(PROJECT)),
                    fingerprint=frozen["fingerprint"],
                ),
            )
            time.sleep(5)
    if process.returncode:
        raise RuntimeError(phase + " worker failed; read " + str(log))
    return log


def pipeline(folder, lock):
    frozen = verify_continuation(folder)
    if not (folder / "training/completion.json").exists():
        run_worker(folder, lock, "train", frozen)
    check_training(folder, frozen)
    nav = prepare_navigation(folder)
    base = folder / "navigation"
    if not (base / "geometry/completion.json").exists():
        geometry_check(base, nav)
    verify_phase(base, "geometry", nav)
    phases = ("rendered_oracle", "model_check", "formal")
    if not all((base / phase / "completion.json").exists() for phase in phases):
        run_worker(folder, lock, "navigation", frozen)
    # Positive receipts are mandatory, including when Isaac swallows an exception with exit0.
    for phase in phases:
        verify_phase(base, phase, nav)
    from insect_utils.continuation_report import report_continuation

    report_continuation(folder)
    verify_continuation(folder)
    atomic_json(
        folder / "progress.json", dict(read_json(folder / "completion.json"), phase="completed")
    )
    print("CONTINUATION AND20M TEST COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=PROJECT / "runs/stage15_rotation_300_01")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--worker", choices=("train", "navigation"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    if args.worker:
        if args.worker == "train":
            from insect_utils.continuation_training import train_continuation

            train_continuation(folder)
        else:
            from insect_utils.rotation_runtime import run_arm

            run_arm(folder / "navigation")
        return
    folder.mkdir(parents=True, exist_ok=True)
    with RunLock(folder) as lock:
        if args.plan_only:
            value = prepare_continuation(folder)
            print("PREPARED300 " + value["fingerprint"], flush=True)
            return
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
