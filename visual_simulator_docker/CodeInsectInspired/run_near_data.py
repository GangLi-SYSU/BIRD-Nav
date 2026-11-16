"""Capture the approved symmetric local data supplement without training/navigation."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

from insect_utils.experiment_config import PROJECT
from insect_utils.experiment_state import RunLock, atomic_json, now
from insect_utils.near_data import prepare, prepare_dataset
from insect_utils.local_probe_runtime import verify_capture
from insect_utils.navigation_protocol import verify_navigation


def pipeline(folder, lock):
    frozen = verify_navigation(folder)
    if not (folder / "capture/completion.json").exists():
        log = folder / "logs" / ("capture_" + uuid.uuid4().hex[:10] + ".log")
        log.parent.mkdir(exist_ok=True)
        atomic_json(
            folder / "progress.json",
            dict(
                status="starting",
                phase="capture",
                updated=now(),
                fingerprint=frozen["fingerprint"],
                images_completed=0,
                images_total=len(frozen["samples"]),
                log=str(log.relative_to(PROJECT)),
            ),
        )
        print("START local probe " + str(log), flush=True)
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
            raise RuntimeError("Local capture failed; read " + str(log))
    verify_capture(folder)
    from insect_utils.near_data_report import report_near_data

    prepare_dataset(folder)
    report_near_data(folder)
    verify_navigation(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=PROJECT / "runs/stage19_near_data_01")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    if args.worker:
        from insect_utils.local_probe_runtime import capture_probe

        capture_probe(folder)
        return
    folder.mkdir(parents=True, exist_ok=True)
    with RunLock(folder) as lock:
        if args.plan_only:
            value = prepare(folder)
            print("PREPARED_NEAR_DATA " + value["fingerprint"], flush=True)
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
