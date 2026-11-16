"""Run64 genuine50-action trials from15m and report final endpoint errors."""

import argparse
from pathlib import Path
import os
import subprocess
import sys
import time
import traceback
import uuid

from insect_utils.experiment_config import PROJECT, digest
from insect_utils.experiment_state import RunLock, atomic_json, now, read_json
from insect_utils.fixed50_metrics import COMPLETED, endpoint_metrics, fixed_termination
from insect_utils.fixed50_protocol import prepare
from insect_utils.fixed50_report import plot_live, report
from insect_utils.navigation_core import decide_motion, endpoint, safe_segment
from insect_utils.camera_geometry import home_vector_body
from run_navigation_experiment import verify_phase


def geometry_check(folder, frozen):
    controller = frozen["protocol"]["controller"]
    home = frozen["protocol"]["home_camera_xyz_m"]
    checks = []
    for episode in frozen["episodes"]["formal"]:
        positions = simulate_fixed_oracle(episode, home, controller, frozen["budget"])
        checks.append(
            dict(episode_id=episode["episode_id"], metrics=endpoint_metrics(positions, home))
        )
    path = folder / "geometry/results.json"
    atomic_json(path, dict(checks=checks, diagnostic_only=True))
    atomic_json(
        path.with_name("completion.json"),
        dict(
            status="completed",
            fingerprint=frozen["fingerprint"],
            oracle_checks=len(checks),
            results_sha256=digest(path),
        ),
    )


def simulate_fixed_oracle(episode, home, controller, budget=None):
    position = list(episode["initial_position_m"])
    yaw = episode["initial_yaw_deg"]
    positions = [position]
    for step in range(50):
        if fixed_termination(position, home, step, controller) is not None:
            raise ValueError("Fixed horizon stopped early in pure geometry")
        motion = decide_motion(home_vector_body(position, home, yaw), yaw, controller)
        next_position = endpoint(position, motion)
        if budget is not None:
            safe_segment(position, next_position, budget)
        position, yaw = next_position, motion["next_yaw_deg"]
        positions.append(position)
    if fixed_termination(position, home, 50, controller) != COMPLETED:
        raise ValueError("Fixed horizon did not finish at action50")
    return positions


def pipeline(folder, lock, frozen):
    if not (folder / "geometry/completion.json").exists():
        geometry_check(folder, frozen)
    verify_phase(folder, "geometry", frozen)
    if not (folder / "formal/completion.json").exists():
        log = folder / "logs" / ("fixed50_" + uuid.uuid4().hex[:10] + ".log")
        log.parent.mkdir(exist_ok=True)
        atomic_json(
            folder / "progress.json",
            dict(
                status="starting",
                phase="formal",
                updated=now(),
                episodes_completed=0,
                episodes_total=64,
                horizon_actions=50,
                distance_based_termination=False,
            ),
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
                        phase="formal",
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
            raise RuntimeError("Fixed50 worker failed; read " + str(log))
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
        from insect_utils.fixed50_runtime import run_fixed

        run_fixed(folder)
        return
    folder.mkdir(parents=True, exist_ok=True)
    with RunLock(folder) as lock:
        frozen = prepare(folder)
        plot_live(folder)
        if args.plan_only:
            print("PREPARED_FIXED50 " + frozen["fingerprint"], flush=True)
            return
        try:
            pipeline(folder, lock, frozen)
        except BaseException as error:
            failure = dict(
                status="failed", updated=now(), error=str(error), traceback=traceback.format_exc()
            )
            atomic_json(folder / ("failure_" + uuid.uuid4().hex[:10] + ".json"), failure)
            atomic_json(folder / "pipeline_progress.json", failure)
            raise


if __name__ == "__main__":
    main()
