"""Frozen Stage12: oracle/IO checks,48 pilot trials, then320 navigation trials."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

from insect_utils.experiment_config import PROJECT, canonical_hash, digest
from insect_utils.experiment_state import RunLock, atomic_json, now, read_json
from insect_utils.navigation_core import make_episodes, simulate_oracle
from insect_utils.navigation_protocol import PHASES, prepare_navigation, verify_navigation
from insect_utils.navigation_results import saved_episodes


def geometry_check(folder, frozen):
    protocol = frozen["protocol"]
    checks = []
    group = {
        "radii_m": [2, 3, 5, 40],
        "bearings_per_radius": 4,
        "initial_yaws_deg": [0, 90, 180, 270],
    }
    for home in ([0, 0, 40], [13, -8, 40]):
        for episode in make_episodes(group, home, "geometry"):
            result = simulate_oracle(episode, home, protocol["controller"], frozen["budget"])
            if result["reason"] != "success":
                raise RuntimeError("Oracle coordinate/action chain failed")
            checks.append({"home": home, "episode": episode, "result": result})
    atomic_json(Path(folder) / "geometry/results.json", {"checks": checks})
    atomic_json(
        Path(folder) / "geometry/completion.json",
        {
            "status": "completed",
            "fingerprint": frozen["fingerprint"],
            "oracle_checks": len(checks),
            "results_sha256": digest(Path(folder) / "geometry/results.json"),
        },
    )


def verify_phase(folder, phase, frozen):
    base = Path(folder) / phase
    file = base / "completion.json"
    if not file.exists():
        raise RuntimeError("Worker exited without a verified completion receipt for " + phase)
    value = read_json(file)
    if value["status"] != "completed" or value["fingerprint"] != frozen["fingerprint"]:
        raise ValueError("Phase completion does not match frozen navigation")
    if phase == "model_check":
        for name, expected in value["artifacts"].items():
            if digest(base / name) != expected:
                raise ValueError("Model/camera acceptance artifact changed")
    else:
        if digest(base / "results.json") != value["results_sha256"]:
            raise ValueError("Navigation phase results changed")
        if phase != "geometry":
            saved = saved_episodes(folder, phase, frozen)
            payload = read_json(base / "results.json")
            if canonical_hash(list(saved.values())) != canonical_hash(payload["episodes"]):
                raise ValueError("Episode files differ from committed phase results")
    return value


def worker(folder, phase):
    frozen = verify_navigation(folder)
    if phase == "geometry":
        geometry_check(folder, frozen)
    else:
        from insect_utils.navigation_runtime import run_rendered

        run_rendered(folder, phase)


def pipeline(folder, lock):
    frozen = verify_navigation(folder)
    for phase in PHASES:
        verify_navigation(folder)
        if (folder / phase / "completion.json").exists():
            verify_phase(folder, phase, frozen)
            print("VERIFIED COMPLETED " + phase, flush=True)
            continue
        log = folder / "logs" / (phase + "_" + uuid.uuid4().hex[:10] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(
            folder / "progress.json",
            {
                "phase": phase,
                "status": "starting",
                "updated": now(),
                "success_radius_m": frozen["protocol"]["controller"]["success_radius_m"],
                "log": str(log.relative_to(PROJECT)),
                "fingerprint": frozen["fingerprint"],
            },
        )
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
                time.sleep(5)
        if process.returncode:
            raise RuntimeError(
                "Navigation phase {} failed, exit{}; read {}".format(phase, process.returncode, log)
            )
        verify_phase(folder, phase, frozen)
        print("COMPLETED " + phase, flush=True)
    from insect_utils.navigation_report import report_navigation

    report_navigation(folder, frozen)
    verify_navigation(folder)
    atomic_json(
        folder / "progress.json", dict(read_json(folder / "completion.json"), phase="completed")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=PROJECT / "runs/stage12_navigation_01")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--worker", choices=PHASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    if args.worker:
        worker(folder, args.worker)
        return
    folder.mkdir(parents=True, exist_ok=True)
    with RunLock(folder) as lock:
        if args.plan_only:
            result = prepare_navigation(folder)
            print(
                {
                    "fingerprint": result["fingerprint"],
                    "episodes": {k: len(v) for k, v in result["episodes"].items()},
                    "success_radius_m": result["protocol"]["controller"]["success_radius_m"],
                    "budget": result["budget"],
                },
                flush=True,
            )
            return
        verify_navigation(folder)
        if (folder / "completion.json").exists():
            for phase in PHASES:
                verify_phase(folder, phase, verify_navigation(folder))
            print("Navigation already completed; outputs preserved", flush=True)
            return
        try:
            pipeline(folder, lock)
        except BaseException as error:
            atomic_json(
                folder / ("failure_" + uuid.uuid4().hex[:10] + ".json"),
                {"error": str(error), "updated": now(), "traceback": traceback.format_exc()},
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
