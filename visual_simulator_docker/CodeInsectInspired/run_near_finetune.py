"""Wait for near-home data, fine-tune both controls, then repeat matched5m/10m trials."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

from insect_utils.experiment_config import PROJECT
from insect_utils.experiment_state import RunLock, atomic_json, now, read_json
from insect_utils.near_finetune_protocol import (
    declare,
    verify_declared,
    freeze_ready_data,
    verify_finetune,
    prepare_navigation,
)
from insect_utils.near_finetune_report import plot_training, plot_navigation, report_final
from run_navigation_experiment import verify_phase


def worker(folder, lock, phase, arm):
    log = folder / "logs" / (phase + "_" + arm + "_" + uuid.uuid4().hex[:10] + ".log")
    log.parent.mkdir(exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-dir",
        str(folder),
        "--worker",
        phase,
        "--arm",
        arm,
    ]
    print("START " + phase + " " + arm + " " + str(log), flush=True)
    last_plot = 0
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
                "arms/" + arm + "/engine/progress.json"
                if phase == "train"
                else "navigation/" + arm + "/progress.json"
            )
            detail = read_json(path) if path.exists() else dict(status="starting")
            atomic_json(
                folder / "progress.json",
                dict(
                    status="running",
                    phase=phase,
                    arm=arm,
                    updated=now(),
                    detail=detail,
                    log=str(log.relative_to(PROJECT)),
                ),
            )
            if time.monotonic() - last_plot >= 30:
                if phase == "train":
                    plot_training(folder)
                else:
                    plot_navigation(folder, arm)
                last_plot = time.monotonic()
            time.sleep(5)
    if process.returncode:
        raise RuntimeError(phase + " " + arm + " failed; read " + str(log))


def pipeline(folder, lock):
    declared = verify_declared(folder)
    base = PROJECT / declared["spec"]["data_run"]
    while not (base / "completion.json").exists():
        progress = read_json(base / "progress.json") if (base / "progress.json").exists() else {}
        if progress.get("status") == "failed":
            raise RuntimeError("Near-data capture failed; training has not started")
        atomic_json(
            folder / "progress.json",
            dict(status="waiting", phase="waiting_for_data", updated=now(), detail=progress),
        )
        time.sleep(5)
    frozen = freeze_ready_data(folder)
    selection = {}
    for arm in frozen["spec"]["arms"]:
        if not (folder / "arms" / arm / "completion.json").exists():
            worker(folder, lock, "train", arm)
        from insect_utils.near_finetune_training import train_arm

        selection[arm] = train_arm(folder, arm)
        plot_training(folder)
    value = dict(fingerprint=frozen["fingerprint"], arms=selection, updated=now())
    target = folder / "selection_frozen_before_navigation.json"
    if target.exists():
        old = read_json(target)
        if old["fingerprint"] != value["fingerprint"] or old["arms"] != selection:
            raise ValueError("Validation selections changed")
    else:
        atomic_json(target, value)
    from insect_utils.rotation_protocol import geometry_check

    for arm in frozen["spec"]["arms"]:
        nav = prepare_navigation(folder, arm)
        path = folder / "navigation" / arm
        if not (path / "geometry/completion.json").exists():
            geometry_check(path, nav)
        verify_phase(path, "geometry", nav)
        if not all(
            (path / p / "completion.json").exists()
            for p in ("rendered_oracle", "model_check", "formal")
        ):
            worker(folder, lock, "navigation", arm)
        for phase in ("rendered_oracle", "model_check", "formal"):
            verify_phase(path, phase, nav)
        plot_navigation(folder, arm)
    report_final(folder)
    verify_finetune(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=PROJECT / "runs/stage20_near_finetune_01")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--worker", choices=("train", "navigation"), help=argparse.SUPPRESS)
    parser.add_argument(
        "--arm", choices=("original_data", "near_supplement"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    if args.worker:
        if args.arm is None:
            parser.error("Worker arm is required")
        if args.worker == "train":
            from insect_utils.near_finetune_training import train_arm

            train_arm(folder, args.arm)
        else:
            from insect_utils.rotation_runtime import run_arm

            run_arm(folder / "navigation" / args.arm)
        return
    folder.mkdir(parents=True, exist_ok=True)
    with RunLock(folder) as lock:
        if args.plan_only:
            value = declare(folder)
            print("DECLARED_NEAR_FINETUNE " + value["fingerprint"], flush=True)
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
