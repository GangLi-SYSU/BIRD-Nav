"""Run the frozen Stage11 acquisition and two offline training controls on this disk."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

from insect_utils.coverage_plan import prepare_plan, verify_freeze
from insect_utils.experiment_config import PROJECT
from insect_utils.experiment_state import RunLock, atomic_json, now, read_json


PHASES = (
    "capture",
    "prepare",
    "smoke_meters",
    "smoke_normalized",
    "train_meters",
    "train_normalized",
    "evaluate",
)


def execute_worker(phase, folder):
    if phase == "capture":
        from insect_utils.coverage_capture import capture

        capture(folder)
    elif phase == "prepare":
        from insect_utils.coverage_data import prepare_data

        prepare_data(folder)
    elif phase.startswith(("train_", "smoke_")):
        from insect_utils.coverage_training import train_arm

        prefix, arm = phase.split("_", 1)
        train_arm(folder, arm, smoke=prefix == "smoke")
    else:
        from insect_utils.coverage_training import evaluate

        evaluate(folder)


def pipeline(folder, lock):
    freeze = verify_freeze(folder)
    for phase in PHASES:
        verify_freeze(folder)
        log = folder / "logs" / (phase + "_" + uuid.uuid4().hex[:10] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        progress = {
            "phase": phase,
            "status": "starting",
            "updated": now(),
            "fingerprint": freeze["fingerprint"],
            "log": str(log.relative_to(PROJECT)),
        }
        atomic_json(folder / "progress.json", progress)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-dir",
            str(folder),
            "--worker",
            phase,
        ]
        print("START {} {}".format(phase, log), flush=True)
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
                if phase.startswith(("train_", "smoke_")):
                    prefix, arm = phase.split("_", 1)
                    engine = folder / ("arms" if prefix == "train" else "smoke") / arm
                    state = engine / "engine_scaled_units/progress.json"
                    if state.exists():
                        item = read_json(state)
                        scale = freeze["protocol"]["arms"][arm]
                        physical = {
                            key: value * scale ** 2
                            for key, value in item.items()
                            if key.endswith("component_mse_m2") and value is not None
                        }
                        progress.update(
                            status="running",
                            updated=now(),
                            epoch=item["epoch"],
                            epochs=item["epochs"],
                            optimizer_steps=item["optimizer_steps"],
                            physical_metrics=physical,
                        )
                        atomic_json(folder / "progress.json", progress)
                time.sleep(10)
        if process.returncode:
            raise RuntimeError("{} failed (exit{}); read {}".format(phase, process.returncode, log))
        if phase == "capture":
            completion_file = folder / "capture/completion.json"
            if not completion_file.exists():
                raise RuntimeError(
                    "Isaac exited without a complete capture receipt; resume capture"
                )
            receipt = read_json(completion_file)
            if (
                receipt["fingerprint"] != freeze["fingerprint"]
                or receipt["new_images"] != freeze["plan"]["audit"]["fresh_images"]
            ):
                raise RuntimeError("Capture completion receipt does not match the frozen plan")
        print("COMPLETED " + phase, flush=True)
    completion = read_json(folder / "completion.json")
    atomic_json(folder / "progress.json", dict(completion, phase="completed"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=PROJECT / "runs/stage11_coverage_02")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--worker", choices=PHASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    if args.worker:
        execute_worker(args.worker, folder)
        return
    folder.mkdir(parents=True, exist_ok=True)
    with RunLock(folder) as lock:
        if args.plan_only:
            result = prepare_plan(folder)
            print(result["plan"]["audit"], flush=True)
            return
        verify_freeze(folder)
        if (folder / "completion.json").exists():
            print("Experiment already completed; results are preserved", flush=True)
            return
        try:
            pipeline(folder, lock)
        except BaseException as error:
            atomic_json(
                folder / "failure_{}.json".format(uuid.uuid4().hex[:10]),
                {"updated": now(), "error": str(error), "traceback": traceback.format_exc()},
            )
            progress = read_json(folder / "progress.json")
            atomic_json(folder / "progress.json", dict(progress, status="failed", error=str(error)))
            raise


if __name__ == "__main__":
    main()
