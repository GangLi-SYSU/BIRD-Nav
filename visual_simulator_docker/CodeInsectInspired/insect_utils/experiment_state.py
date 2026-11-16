"""Atomic stage records, artifact integrity and exclusive run ownership."""

import datetime
import fcntl
import json
import os
from pathlib import Path
import uuid

from insect_utils.experiment_config import canonical_hash, digest, identifier, relative_path


DEPENDENCIES = {"capture": (), "train": ("capture",), "evaluate": ("capture", "train")}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class RunLock:
    def __init__(self, run_dir):
        self.path = Path(run_dir) / ".run.lock"
        self.stream = None

    def __enter__(self):
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise ValueError(
                "Another process still owns this run; it cannot be resumed concurrently"
            )
        return self

    def fileno(self):
        return self.stream.fileno()

    def __exit__(self, *error):
        # The worker inherits this descriptor. A surviving worker retains the lock
        # even if its parent is interrupted; closing it must not explicitly unlock.
        self.stream.close()


def run_directory(resolved, run_id, project):
    root = Path(resolved["spec"]["output_root"])
    if not root.is_absolute():
        root = project / root
    return root / identifier(resolved["spec"]["name"]) / identifier(run_id)


def prepare_run(run_dir, resolved):
    run_dir.mkdir(parents=True, exist_ok=False)
    with RunLock(run_dir):
        atomic_json(run_dir / "resolved_config.json", resolved)
        atomic_json(
            run_dir / "run_manifest.json",
            {
                "created_at": now(),
                "purpose": resolved["spec"]["purpose"],
                "fingerprint": resolved["fingerprint"],
                "parent_pid": os.getpid(),
            },
        )
        (run_dir / "state").mkdir()
        (run_dir / "logs").mkdir()


def verify_run(run_dir, current):
    path = run_dir / "resolved_config.json"
    if not path.is_file():
        raise ValueError("Existing directory is not a prepared run; prepare a new run ID")
    stored = read_json(path)
    payload = {key: stored[key] for key in ("spec", "derived", "provenance")}
    if canonical_hash(payload) != stored.get("fingerprint"):
        raise ValueError("Saved resolved configuration was modified")
    if stored["fingerprint"] != current["fingerprint"]:
        raise ValueError("Config, source, baseline protocol or scene changed; use a new run ID")
    return stored


def artifact_record(path, run_dir):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular artifact: " + str(path))
    relative = str(path.resolve().relative_to(run_dir.resolve()))
    return relative, {"sha256": digest(path), "bytes": path.stat().st_size}


def collect_artifacts(folder, run_dir):
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    if not files:
        raise ValueError("A stage cannot complete without artifacts")
    return dict(artifact_record(path, run_dir) for path in files)


def verify_artifacts(run_dir, files):
    if not files:
        raise ValueError("Completion record has no artifacts")
    for relative, expected in files.items():
        path = run_dir / relative_path(relative)
        if not path.is_file() or path.is_symlink():
            raise ValueError("Missing or invalid artifact: " + relative)
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError("Artifact fingerprint mismatch: " + relative)


def stage_record(run_dir, stage):
    path = run_dir / "state" / (stage + ".json")
    return read_json(path) if path.is_file() else None


def verify_completed(run_dir, stage, fingerprint):
    record = stage_record(run_dir, stage)
    if not record or record["status"] != "completed":
        raise ValueError("Required stage is not completed: " + stage)
    if record["fingerprint"] != fingerprint:
        raise ValueError("Stage belongs to a different experiment: " + stage)
    verify_artifacts(run_dir, record["artifacts"])
    for dependency in DEPENDENCIES[stage]:
        previous = verify_completed(run_dir, dependency, fingerprint)
        if record["inputs"][dependency] != canonical_hash(previous["artifacts"]):
            raise ValueError("Stage input artifacts changed: " + dependency)
    return record


def begin_stage(run_dir, stage, resolved, resume):
    previous = stage_record(run_dir, stage)
    if previous and previous["status"] == "completed":
        if not resume:
            raise ValueError(stage + " already completed; --resume verifies and reuses it")
        verify_completed(run_dir, stage, resolved["fingerprint"])
        return None
    if previous and not resume:
        raise ValueError("Stage was interrupted or failed; use --resume to retry it")
    inputs = {}
    for dependency in DEPENDENCIES[stage]:
        value = verify_completed(run_dir, dependency, resolved["fingerprint"])
        inputs[dependency] = canonical_hash(value["artifacts"])
    attempt = previous["attempt"] + 1 if previous else 1
    folder = run_dir / stage
    if folder.exists():
        if not previous:
            raise ValueError("Unrecorded stage output exists; refusing to reuse it")
        archive = run_dir / "attempts" / stage / ("attempt_{:02d}".format(previous["attempt"]))
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            # A crash may have archived output before the new state was committed.
            archive = archive.with_name(archive.name + "_recovery_" + uuid.uuid4().hex[:8])
        folder.rename(archive)
    folder.mkdir()
    value = {
        "status": "running",
        "stage": stage,
        "attempt": attempt,
        "started_at": now(),
        "fingerprint": resolved["fingerprint"],
        "parent_pid": os.getpid(),
        "inputs": inputs,
        "artifacts": {},
    }
    atomic_json(run_dir / "state" / (stage + ".json"), value)
    return value


def complete_stage(run_dir, stage, worker_pid):
    value = stage_record(run_dir, stage)
    if not value or value["status"] != "running":
        raise ValueError("Worker has no running stage record")
    value.update(
        status="completed",
        completed_at=now(),
        worker_pid=worker_pid,
        artifacts=collect_artifacts(run_dir / stage, run_dir),
    )
    atomic_json(run_dir / "state" / (stage + ".json"), value)


def fail_stage(run_dir, stage, error):
    value = stage_record(run_dir, stage)
    value.update(status="failed", failed_at=now(), error=str(error))
    atomic_json(run_dir / "state" / (stage + ".json"), value)
