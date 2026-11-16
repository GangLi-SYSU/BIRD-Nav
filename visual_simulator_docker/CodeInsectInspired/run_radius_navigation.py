"""Run128 homing trials at10m/5m with the frozen589-epoch model and controller."""

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
from insect_utils.navigation_protocol import verify_navigation
from insect_utils.radius_extension import prepare, report
from insect_utils.rotation_protocol import geometry_check
from run_navigation_experiment import verify_phase


def pipeline(folder, lock):
    frozen = verify_navigation(folder)
    if not (folder / "geometry/completion.json").exists():
        geometry_check(folder, frozen)
    verify_phase(folder, "geometry", frozen)
    phases = ("rendered_oracle", "model_check", "formal")
    if not all((folder / p / "completion.json").exists() for p in phases):
        log = folder / "logs" / ("navigation_" + uuid.uuid4().hex[:10] + ".log")
        log.parent.mkdir(exist_ok=True)
        atomic_json(
            folder / "progress.json",
            dict(
                status="starting",
                phase="rendered_oracle",
                updated=now(),
                fingerprint=frozen["fingerprint"],
                log=str(log.relative_to(PROJECT)),
            ),
        )
        print("START radius navigation " + str(log), flush=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-dir",
            str(folder),
            "--worker",
        ]
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
                time.sleep(5)
        if process.returncode:
            raise RuntimeError("Navigation worker failed; read " + str(log))
    for phase in phases:
        verify_phase(folder, phase, frozen)
    report(folder)
    verify_navigation(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=PROJECT / "runs/stage17_radius_5_10_01")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    if args.worker:
        from insect_utils.rotation_runtime import run_arm

        run_arm(folder)
        return
    folder.mkdir(parents=True, exist_ok=True)
    with RunLock(folder) as lock:
        if args.plan_only:
            value = prepare(folder)
            print("PREPARED_RADIUS_NAVIGATION " + value["fingerprint"], flush=True)
            return
        try:
            pipeline(folder, lock)
        except BaseException as error:
            atomic_json(
                folder / ("failure_" + uuid.uuid4().hex[:10] + ".json"),
                dict(error=str(error), traceback=traceback.format_exc(), updated=now()),
            )
            state = (
                read_json(folder / "progress.json") if (folder / "progress.json").exists() else {}
            )
            atomic_json(
                folder / "progress.json",
                dict(state, status="failed", error=str(error), updated=now()),
            )
            raise


if __name__ == "__main__":
    main()
