"""Run the engineering fixture through isolated capture and one real training epoch."""

import datetime
import json
from pathlib import Path
import subprocess
import sys
import uuid

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.experiment_config import load_config
from insect_utils.experiment_state import atomic_json, read_json, run_directory


def main():
    config = PROJECT / "config/experiments/stage02_pipeline_smoke.json"
    run_id = (
        "check_"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
        + uuid.uuid4().hex[:6]
    )
    resolved = load_config(config)
    run_dir = run_directory(resolved, run_id, PROJECT)

    def command(stage, resume=False, config_path=config, success=True):
        args = [
            sys.executable,
            str(PROJECT / "run_downward_homing.py"),
            "--config",
            str(config_path),
            "--stage",
            stage,
            "--run-id",
            run_id,
        ]
        if resume:
            args.append("--resume")
        result = subprocess.run(
            args, cwd=str(PROJECT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        print(result.stdout, end="", flush=True)
        if (result.returncode == 0) != success:
            raise AssertionError("Unexpected result from {}: {}".format(stage, result.returncode))
        return result.stdout

    for stage in ("prepare", "capture", "train"):
        command(stage)
    capture = read_json(run_dir / "state/capture.json")
    training = read_json(run_dir / "state/train.json")
    summary = read_json(run_dir / "train/summary.json")
    assert capture["worker_pid"] != training["worker_pid"]
    assert summary["kit_modules"] == [] and summary["optimizer_steps"] == 3
    assert summary["network_parameters"] == 10820 and summary["epochs_completed"] == 1
    command("capture", resume=True)
    command("train", resume=True)
    assert read_json(run_dir / "state/train.json")["attempt"] == 1

    history = run_dir / "train/training_history.json"
    original = history.read_bytes()
    try:
        history.write_bytes(original + b"\n")
        rejected = command("train", resume=True, success=False)
        assert "Artifact fingerprint mismatch" in rejected
    finally:
        history.write_bytes(original)
    changed = json.loads(config.read_text())
    changed["seed"] = 11
    alternate = run_dir / "checks/changed_config.json"
    atomic_json(alternate, changed)
    assert "use a new run ID" in command("train", resume=True, config_path=alternate, success=False)
    assert "cannot produce closed-loop" in command("evaluate", success=False)
    command("train", resume=True)
    import onnx

    onnx.checker.check_model(onnx.load(str(run_dir / "train/model_attention.onnx")))
    atomic_json(
        run_dir / "checks/integration_report.json",
        {
            "status": "passed",
            "purpose": "engineering_smoke_only",
            "run_id": run_id,
            "capture_worker_pid": capture["worker_pid"],
            "train_worker_pid": training["worker_pid"],
            "optimizer_steps": summary["optimizer_steps"],
            "epochs": 1,
            "kit_modules_in_training": [],
            "resume_cache_verified": True,
            "corrupted_artifact_rejected": True,
            "changed_config_rejected": True,
            "synthetic_homing_evaluation_rejected": True,
            "onnx_valid": True,
            "note": "Artificial fixture data, no physical rendering or homing evaluation.",
        },
    )
    print("STAGE02_INTEGRATION_PASS " + str(run_dir), flush=True)


if __name__ == "__main__":
    main()
