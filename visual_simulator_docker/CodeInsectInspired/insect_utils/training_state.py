"""Training metrics, resource observation and epoch checkpoint file integrity."""

import math
import os
from pathlib import Path
import threading
import uuid

from insect_utils.experiment_config import digest, relative_path
from insect_utils.experiment_state import atomic_json, read_json


TRAINING_PROTOCOL = "config/experiments/stage07_training_protocol.json"


def quantile(values, fraction):
    if not 0 <= fraction <= 1:
        raise ValueError("Quantile fraction must be between zero and one")
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    return {
        "mean": sum(values) / len(values) if values else None,
        "p50": quantile(values, 0.5),
        "p90": quantile(values, 0.9),
        "p95": quantile(values, 0.95),
    }


def vector_metrics(predictions, targets):
    if len(predictions) != len(targets) or len(predictions) == 0:
        raise ValueError("Empty/mismatched predictions")
    squared = []
    vector = []
    distance = []
    angles = []
    zero_target = zero_prediction = 0
    for prediction, target in zip(predictions, targets):
        if len(prediction) != 2 or len(target) != 2:
            raise ValueError("Expected two-dimensional vectors")
        px, py, tx, ty = [float(v) for v in [*prediction, *target]]
        if not all(math.isfinite(v) for v in [px, py, tx, ty]):
            raise ValueError("Non-finite regression output")
        error = (px - tx) ** 2 + (py - ty) ** 2
        squared.append(error)
        vector.append(math.sqrt(error))
        pn, tn = math.hypot(px, py), math.hypot(tx, ty)
        distance.append(abs(pn - tn))
        zero_target += tn <= 1e-8
        zero_prediction += pn <= 1e-8
        if pn > 1e-8 and tn > 1e-8:
            angles.append(abs(math.degrees(math.atan2(px * ty - py * tx, px * tx + py * ty))))
    return {
        "samples": len(squared),
        "component_mse_m2": sum(squared) / (2 * len(squared)),
        "mean_squared_vector_error_m2": sum(squared) / len(squared),
        "vector_error_m": distribution(vector),
        "absolute_distance_error_m": distribution(distance),
        "absolute_shortest_angle_error_deg": distribution(angles),
        "valid_angle_samples": len(angles),
        "excluded_angle_samples": len(squared) - len(angles),
        "zero_target_samples": zero_target,
        "zero_prediction_samples": zero_prediction,
    }


def position_subset(records, count):
    groups = sorted({r["position_id"] for r in records})
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(groups):
        raise ValueError("Invalid position subset size")
    selected = (
        [groups[round(i * (len(groups) - 1) / (count - 1))] for i in range(count)]
        if count > 1
        else groups[:1]
    )
    return [i for i, r in enumerate(records) if r["position_id"] in selected], selected


def memory_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot read available host memory")


def process_memory():
    pending = [os.getpid()]
    seen = set()
    rss = pss = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            children = Path("/proc/{}/task/{}/children".format(pid, pid)).read_text().split()
            pending.extend(int(p) for p in children)
            for line in Path("/proc/{}/smaps_rollup".format(pid)).read_text().splitlines():
                if line.startswith("Rss:"):
                    rss += int(line.split()[1]) * 1024
                if line.startswith("Pss:"):
                    pss += int(line.split()[1]) * 1024
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return rss, pss


class ResourceWatch:
    def __init__(self):
        self.stop = threading.Event()
        self.max_rss = 0
        self.max_pss = 0
        self.min_available = memory_available()
        self.thread = None

    def sample(self):
        rss, pss = process_memory()
        self.max_rss = max(self.max_rss, rss)
        self.max_pss = max(self.max_pss, pss)
        self.min_available = min(self.min_available, memory_available())

    def run(self):
        while not self.stop.wait(0.25):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=2)
        self.sample()

    def summary(self):
        return {
            "max_process_and_workers_rss_bytes": self.max_rss,
            "max_process_and_workers_pss_bytes": self.max_pss,
            "min_host_memory_available_bytes": self.min_available,
            "sampling_interval_s": 0.25,
        }


def checkpoint_pointer(folder, fingerprint):
    pointer = read_json(folder / "checkpoint_latest.json")
    if pointer["fingerprint"] != fingerprint:
        raise ValueError("Checkpoint configuration fingerprint mismatch")
    path = folder / relative_path(pointer["file"])
    if path.is_symlink() or not path.is_file() or digest(path) != pointer["sha256"]:
        raise ValueError("Checkpoint bytes changed or are missing")
    return path, pointer


def save_checkpoint(folder, payload, fingerprint):
    import torch

    folder = Path(folder)
    checkpoints = folder / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    name = "epoch_{:04d}_{}.pt".format(payload["epoch"], uuid.uuid4().hex[:10])
    path = checkpoints / name
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        torch.save(dict(payload, fingerprint=fingerprint), stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    pointer = {
        "file": str(path.relative_to(folder)),
        "sha256": digest(path),
        "fingerprint": fingerprint,
        "epoch": payload["epoch"],
        "optimizer_steps": payload["optimizer_steps"],
    }
    atomic_json(folder / "checkpoint_latest.json", pointer)
    return pointer


def assert_no_kit():
    import sys

    if any(name == "omni" or name.startswith("omni.") for name in sys.modules):
        raise RuntimeError("Offline training must not import Isaac/Kit")
