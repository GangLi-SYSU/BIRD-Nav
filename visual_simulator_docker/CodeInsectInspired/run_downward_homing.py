"""Plan or run isolated capture/train/evaluate stages from one configuration."""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from insect_utils.experiment_config import PROJECT, STAGES, load_config, preflight
from insect_utils.experiment_state import RunLock, begin_stage, fail_stage, prepare_run
from insect_utils.experiment_state import run_directory, stage_record, verify_completed, verify_run


def execute_stage(resolved, run_dir, stage, resume, lock):
    preflight(resolved, stage)
    attempt = begin_stage(run_dir, stage, resolved, resume)
    if attempt is None:
        print("Verified and reused completed stage: " + stage, flush=True)
        return
    log = run_dir / "logs" / ("{}_{:02d}.log".format(stage, attempt["attempt"]))
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        stage,
        "--run-dir",
        str(run_dir),
    ]
    print("Running {} in its own process; log: {}".format(stage, log), flush=True)
    try:
        with log.open("x") as stream:
            environment = dict(
                os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", MPLBACKEND="Agg"
            )
            result = subprocess.run(
                command,
                cwd=str(PROJECT),
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                pass_fds=(lock.fileno(),),
            )
        if result.returncode:
            raise ValueError(
                "{} worker exited with {}; read {}".format(stage, result.returncode, log)
            )
        verify_completed(run_dir, stage, resolved["fingerprint"])
    except Exception as error:
        fail_stage(run_dir, stage, error)
        raise
    print("Completed stage: " + stage, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "config/experiments/downward_h10_hfov30.json"
    )
    parser.add_argument(
        "--stage", choices=("plan", "prepare", "status", "all") + STAGES, default="plan"
    )
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", choices=STAGES, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.run_dir is None:
            parser.error("Internal worker needs --run-dir")
        from insect_utils.experiment_stages import run_worker

        run_worker(args.worker, args.run_dir)
        return
    resolved = load_config(args.config)
    if args.stage == "plan":
        blockers = {}
        for stage in STAGES:
            try:
                preflight(resolved, stage)
            except ValueError as error:
                blockers[stage] = str(error)
        print(
            json.dumps(
                {
                    "resolved_config": resolved,
                    "execution_blockers": blockers,
                    "note": "Training additionally requires a verified capture completion record.",
                },
                indent=2,
            )
        )
        return
    run_id = args.run_id
    if run_id is None:
        if args.stage not in ("prepare", "all"):
            parser.error("Use the same --run-id that was printed by prepare")
        run_id = (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
            + uuid.uuid4().hex[:8]
        )
    run_dir = run_directory(resolved, run_id, PROJECT)
    if args.stage == "all":
        for stage in STAGES:
            preflight(resolved, stage)
    if not run_dir.exists():
        if args.stage not in ("prepare", "all"):
            raise ValueError("Run not prepared; execute --stage prepare with this run ID first")
        prepare_run(run_dir, resolved)
    elif args.stage == "prepare" and not args.resume:
        raise ValueError("Run ID already exists; choose a new ID or verify it with --resume")
    if args.stage == "status":
        verify_run(run_dir, resolved)
        print(json.dumps({stage: stage_record(run_dir, stage) for stage in STAGES}, indent=2))
        return
    with RunLock(run_dir) as lock:
        verify_run(run_dir, resolved)
        print("Run ID: {}\nRun directory: {}".format(run_id, run_dir), flush=True)
        if args.stage != "prepare":
            stages = STAGES if args.stage == "all" else (args.stage,)
            for stage in stages:
                execute_stage(resolved, run_dir, stage, args.resume, lock)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print("Error: " + str(error), file=sys.stderr)
        sys.exit(2)
