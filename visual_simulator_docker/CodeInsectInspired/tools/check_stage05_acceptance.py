"""Run a tail probe and two complete captures; never invoke training or homing."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.downward_capture import seal_dataset
from insect_utils.experiment_config import digest, load_config, stage_parameters
from insect_utils.experiment_state import atomic_json, run_directory, verify_completed, verify_run


def summarize(run, manifest):
    records = [
        json.loads(line) for line in (run / "capture/samples.jsonl").read_text().splitlines()
    ]
    return {
        "run": str(run),
        "sample_count": manifest["sample_count"],
        "position_count": manifest["position_count"],
        "headings_per_position": manifest["headings_per_position"],
        "camera_count": manifest["camera_count"],
        "pose_signature": manifest["pose_signature"],
        "manifest_sha256": digest(run / "capture/dataset_manifest.json"),
        "updates_waited_range": [
            min(r["receipt"]["updates_waited"] for r in records),
            max(r["receipt"]["updates_waited"] for r in records),
        ],
        "max_last_rgb_change_uint8": max(
            r["receipt"]["stability"]["last_rgb_mae"] for r in records
        ),
        "summed_capture_elapsed_s": sum(r["receipt"]["elapsed_s"] for r in records),
        "training_steps": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "config/experiments/downward_h10_hfov30_landmarks.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source = json.loads(args.config.read_text())
    if source["capture"].get("headings_per_position") != 8:
        raise ValueError("This acceptance implements the user-selected eight-heading protocol")
    output = args.output.resolve()
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not (output / "acceptance.json").is_file():
        raise ValueError("Resume requires a prepared acceptance directory")
    report_path = output / "acceptance.json"
    report = {"status": "running", "headings_per_position": 8, "runs": {}, "training_steps": 0}
    if args.resume:
        stored = json.loads(report_path.read_text())
        if stored.get("source_config_sha256") != digest(args.config):
            raise ValueError("Source configuration changed; use a new output directory")
    report["source_config_sha256"] = digest(args.config)
    report["acceptance_tool_sha256"] = digest(Path(__file__))
    atomic_json(report_path, report)
    try:
        for name, cameras, tail in [
            ("tail_probe", 4, True),
            ("single_camera", 1, False),
            ("four_cameras", 4, False),
        ]:
            spec = copy.deepcopy(source)
            spec["output_root"] = str(output / "runs")
            spec["capture"]["number_cameras"] = cameras
            if tail:
                spec["purpose"] = "engineering_smoke"
                spec["capture"].update(shape_params=[5, 8, 0.35], headings_per_position=1)
                spec["declared_changes"].append(
                    "Five-frame GPU tail probe only; excluded from the learning dataset."
                )
            config = output / (name + ".json")
            if config.exists():
                if json.loads(config.read_text()) != spec:
                    raise ValueError("Acceptance configuration changed; use a new output directory")
            else:
                atomic_json(config, spec)
            resolved = load_config(config)
            run = run_directory(resolved, name, PROJECT)
            for stage in ("prepare", "capture"):
                command = [
                    sys.executable,
                    str(PROJECT / "run_downward_homing.py"),
                    "--config",
                    str(config),
                    "--stage",
                    stage,
                    "--run-id",
                    name,
                ]
                if args.resume:
                    command.append("--resume")
                subprocess.run(command, cwd=str(PROJECT), check=True)
            verify_run(run, load_config(config))
            verify_completed(run, "capture", resolved["fingerprint"])
            manifest = seal_dataset(
                resolved, run, stage_parameters(resolved, run)["capture"], publish=False
            )
            report["runs"][name] = summarize(run, manifest)
            atomic_json(report_path, report)
            print("STAGE05_RUN_PASS " + name, flush=True)
        single, four = report["runs"]["single_camera"], report["runs"]["four_cameras"]
        if single["sample_count"] != 1176 or four["sample_count"] != 1176:
            raise ValueError("Incomplete eight-heading capture")
        if single["position_count"] != 147 or four["position_count"] != 147:
            raise ValueError("Learning positions changed")
        if single["pose_signature"] != four["pose_signature"]:
            raise ValueError("Camera allocation changed the requested poses")
        report.update(
            status="passed",
            total_accepted_images=2357,
            acceptance_runs_merged=False,
            dataset_for_stage_f=report["runs"]["four_cameras"]["run"] + "/capture",
            stage_f_ready=False,
            closed_loop_ready=False,
        )
        atomic_json(report_path, report)
        print("STAGE05_ACCEPTANCE_PASS " + str(output), flush=True)
    except BaseException as error:
        report.update(status="failed", error=str(error))
        atomic_json(report_path, report)
        raise


if __name__ == "__main__":
    main()
