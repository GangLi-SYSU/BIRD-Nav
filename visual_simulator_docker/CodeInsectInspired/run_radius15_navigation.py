"""Run64 additional15m navigation trials with the unchanged epoch484 model."""

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
from insect_utils.radius15_extension import plot_live, prepare, report
from insect_utils.rotation_protocol import geometry_check
from run_navigation_experiment import verify_phase


def pipeline(folder, lock):
    frozen = verify_navigation(folder)
    # Require the verified inherited IO check; never silently run extra20m captures.
    verify_phase(folder, "model_check", frozen)
    if not (folder / "geometry/completion.json").exists():
        geometry_check(folder, frozen)
    verify_phase(folder, "geometry", frozen)
    phases = ("rendered_oracle", "model_check", "formal")
    if not all((folder / phase / "completion.json").exists() for phase in phases):
        log = folder / "logs" / ("navigation_" + uuid.uuid4().hex[:10] + ".log")
        log.parent.mkdir(exist_ok=True)
        atomic_json(
            folder / "progress.json",
            dict(status="starting", phase="rendered_oracle", updated=now()),
        )
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
            last_plot = 0
            while process.poll() is None:
                state = read_json(folder / "progress.json")
                atomic_json(
                    folder / "pipeline_progress.json",
                    dict(
                        status="running",
                        phase=state.get("phase", "starting"),
                        updated=now(),
                        detail=state,
                        log=str(log.relative_to(PROJECT)),
                    ),
                )
                if time.monotonic() - last_plot >= 30:
                    plot_live(folder)
                    last_plot = time.monotonic()
                time.sleep(5)
        if process.returncode:
            raise RuntimeError("15m navigation worker failed; read " + str(log))
    for phase in phases:
        verify_phase(folder, phase, frozen)
    report(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
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
        frozen = prepare(folder)
        plot_live(folder)
        if args.plan_only:
            print("PREPARED15M " + frozen["fingerprint"], flush=True)
            return
        try:
            pipeline(folder, lock)
        except BaseException as error:
            failure = dict(
                status="failed", updated=now(), error=str(error), traceback=traceback.format_exc()
            )
            atomic_json(folder / ("failure_" + uuid.uuid4().hex[:10] + ".json"), failure)
            atomic_json(folder / "pipeline_progress.json", failure)
            raise


if __name__ == "__main__":
    main()
