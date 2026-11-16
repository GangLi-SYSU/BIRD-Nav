"""Acquire a fresh40m/30deg spiral,train600 epochs,and test10/20m with2m success."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

from insect_utils.coverage_plan import verify_freeze
from insect_utils.experiment_config import PROJECT
from insect_utils.experiment_state import RunLock, atomic_json, now, read_json
from insect_utils.spiral_narrow_protocol import prepare, prepare_navigation


def worker(folder, phase):
    if phase == "capture":
        from insect_utils.coverage_capture import capture

        capture(folder)
    elif phase == "data":
        from insect_utils.spiral_narrow_protocol import prepare_data
        from insect_utils.spiral_narrow_report import preview_training_images

        prepare_data(folder)
        preview_training_images(folder)
    elif phase in ("smoke", "train"):
        from insect_utils.spiral_narrow_training import train

        train(folder, smoke=phase == "smoke")
    else:
        from insect_utils.rotation_runtime import run_arm

        run_arm(folder / "navigation")


def detail_path(folder, phase):
    return (
        folder
        / {
            "capture": "progress.json",
            "data": "dataset.json",
            "smoke": "smoke/engine_scaled_units/progress.json",
            "train": "training/engine_scaled_units/progress.json",
            "navigation": "navigation/progress.json",
        }[phase]
    )


def run_worker(folder, phase, lock):
    from insect_utils.spiral_narrow_report import plot_live

    logs = folder / "logs"
    logs.mkdir(exist_ok=True)
    log = logs / (phase + "_" + uuid.uuid4().hex[:10] + ".log")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-dir",
        str(folder),
        "--worker",
        phase,
    ]
    print("START " + phase + " " + str(log.relative_to(PROJECT)), flush=True)
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
        last_plot = 0
        while process.poll() is None:
            path = detail_path(folder, phase)
            detail = read_json(path) if path.exists() and phase != "data" else {}
            atomic_json(
                folder / "pipeline_progress.json",
                dict(
                    status="running",
                    phase=phase,
                    updated=now(),
                    detail=detail,
                    worker_pid=process.pid,
                    log=str(log.relative_to(PROJECT)),
                ),
            )
            if time.monotonic() - last_plot >= 30 and phase in ("train", "navigation"):
                plot_live(folder)
                last_plot = time.monotonic()
            time.sleep(5)
    if process.returncode:
        raise RuntimeError(phase + " worker failed; read " + str(log))
    if phase in ("train", "navigation"):
        plot_live(folder)
    print("COMPLETED " + phase, flush=True)


def pipeline(folder, lock):
    from insect_utils.rotation_protocol import geometry_check
    from insect_utils.spiral_narrow_report import report_final
    from run_navigation_experiment import verify_phase

    if not (folder / "dataset.json").exists():
        run_worker(folder, "capture", lock)
        run_worker(folder, "data", lock)
    for phase, file in (("smoke", "smoke/completion.json"), ("train", "training/completion.json")):
        if not (folder / file).exists():
            run_worker(folder, phase, lock)
    nav = prepare_navigation(folder)
    destination = folder / "navigation"
    if not (destination / "geometry/completion.json").exists():
        geometry_check(destination, nav)
    verify_phase(destination, "geometry", nav)
    phases = ("rendered_oracle", "model_check", "formal")
    if not all((destination / phase / "completion.json").exists() for phase in phases):
        run_worker(folder, "navigation", lock)
    for phase in phases:
        verify_phase(destination, phase, nav)
    verify_freeze(folder)
    result = report_final(folder)
    atomic_json(folder / "pipeline_progress.json", dict(result, phase="completed"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--worker",
        choices=("capture", "data", "smoke", "train", "navigation"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    if args.worker:
        worker(folder, args.worker)
        return
    folder.mkdir(parents=True, exist_ok=True)
    with RunLock(folder) as lock:
        prepare(folder)
        from insect_utils.spiral_narrow_report import plan_figure

        plan_figure(folder)
        if args.plan_only:
            print("PREPARED " + str(folder), flush=True)
            return
        try:
            pipeline(folder, lock)
        except BaseException as error:
            detail = dict(
                status="failed", updated=now(), error=str(error), traceback=traceback.format_exc()
            )
            atomic_json(folder / ("failure_" + uuid.uuid4().hex[:10] + ".json"), detail)
            atomic_json(folder / "pipeline_progress.json", detail)
            raise


if __name__ == "__main__":
    main()
