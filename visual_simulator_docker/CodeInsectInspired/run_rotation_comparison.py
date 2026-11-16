"""Persistently wait for both150-epoch controls, then run the approved20m comparison."""

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
from insect_utils.rotation_protocol import geometry_check, prepare_arm, prepare_comparison
from insect_utils.rotation_protocol import training_ready, verify_comparison
from run_navigation_experiment import verify_phase


def training_progress(declared):
    base = PROJECT / declared["protocol"]["training_run"]
    result = {}
    for arm in declared["protocol"]["arms"]:
        path = base / "arms" / arm / "engine_scaled_units/progress.json"
        value = read_json(path) if path.exists() else dict(epoch=0, epochs=150)
        result[arm] = {k: value[k] for k in ("epoch", "epochs", "optimizer_steps") if k in value}
    return result


def pipeline(folder, lock, wait):
    declared = verify_comparison(folder)
    while not training_ready(declared):
        if not wait:
            raise RuntimeError("Training is not complete; use --wait-for-training")
        base = PROJECT / declared["protocol"]["training_run"]
        if (base / "failure.json").exists():
            raise RuntimeError("Training failed; read " + str(base / "failure.json"))
        atomic_json(
            folder / "progress.json",
            dict(
                status="waiting",
                phase="training",
                updated=now(),
                training=training_progress(declared),
                initial_radii_m=[20],
            ),
        )
        time.sleep(10)
    # Freeze BOTH selected models before any neural navigation test is allowed.
    frozen_arms = {arm: prepare_arm(folder, arm) for arm in declared["protocol"]["arms"]}
    for arm, frozen in frozen_arms.items():
        verify_comparison(folder)
        arm_folder = folder / "arms" / arm
        if not (arm_folder / "geometry/completion.json").exists():
            geometry_check(arm_folder, frozen)
        verify_phase(arm_folder, "geometry", frozen)
        if not all(
            (arm_folder / phase / "completion.json").exists()
            for phase in ("rendered_oracle", "model_check", "formal")
        ):
            log = folder / "logs" / (arm + "_" + uuid.uuid4().hex[:10] + ".log")
            log.parent.mkdir(exist_ok=True)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--run-dir",
                str(folder),
                "--worker",
                arm,
            ]
            with log.open("x") as stream:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    pass_fds=(lock.fileno(),),
                    env=dict(
                        os.environ,
                        PYTHONDONTWRITEBYTECODE="1",
                        PYTHONUNBUFFERED="1",
                        MPLBACKEND="Agg",
                    ),
                )
                while process.poll() is None:
                    path = arm_folder / "progress.json"
                    state = read_json(path) if path.exists() else dict(status="starting")
                    atomic_json(
                        folder / "progress.json",
                        dict(
                            status="running",
                            phase="navigation",
                            arm=arm,
                            updated=now(),
                            detail=state,
                            log=str(log.relative_to(PROJECT)),
                            initial_radii_m=[20],
                        ),
                    )
                    time.sleep(5)
            if process.returncode:
                raise RuntimeError("20m navigation worker failed; read " + str(log))
        # Kit may swallow exceptions with an exit0: receipts and artifact integrity are mandatory.
        for phase in ("rendered_oracle", "model_check", "formal"):
            verify_phase(arm_folder, phase, frozen)
        print("VERIFIED20M " + arm, flush=True)
    from insect_utils.rotation_report import report_comparison

    report_comparison(folder)
    verify_comparison(folder)
    atomic_json(
        folder / "progress.json", dict(read_json(folder / "completion.json"), phase="completed")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=PROJECT / "runs/stage14_rotation_r20_01")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--wait-for-training", action="store_true")
    parser.add_argument("--worker", choices=("mask_only", "mask_rotation"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    if args.worker:
        from insect_utils.rotation_runtime import run_arm

        run_arm(folder / "arms" / args.worker)
        return
    folder.mkdir(parents=True, exist_ok=True)
    with RunLock(folder) as lock:
        if args.plan_only:
            value = prepare_comparison(folder)
            print("DECLARED20M " + value["fingerprint"], flush=True)
            return
        try:
            pipeline(folder, lock, args.wait_for_training)
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
