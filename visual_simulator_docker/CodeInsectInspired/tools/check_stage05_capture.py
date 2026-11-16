"""Analyze a capture plan/proposal, or verify a completed real capture."""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.downward_capture import build_plan, save_plan_plots, seal_dataset
from insect_utils.experiment_config import resolve_spec, stage_parameters
from insect_utils.experiment_state import atomic_json, verify_completed, verify_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--proposed-headings", type=int, choices=(1, 4, 8))
    group.add_argument("--confirmed-plan", action="store_true")
    group.add_argument("--verify-run", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "config/experiments/downward_h10_hfov30_landmarks.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.verify_run:
        if args.output:
            parser.error("Verification is read-only; do not specify --output")
        run = args.verify_run.resolve()
        stored = json.loads((run / "resolved_config.json").read_text())
        resolved = resolve_spec(stored["spec"])
        verify_run(run, resolved)
        verify_completed(run, "capture", resolved["fingerprint"])
        result = seal_dataset(
            resolved, run, stage_parameters(resolved, run)["capture"], publish=False
        )
        print("STAGE05_VERIFIED {} images={}".format(run, result["sample_count"]))
        return
    if not args.output:
        parser.error("Proposal analysis needs a new --output directory")
    spec = json.loads(args.config.read_text())
    if args.proposed_headings is not None:
        spec["capture"]["headings_per_position"] = args.proposed_headings
    resolved = resolve_spec(spec)
    output = args.output.resolve()
    config = stage_parameters(resolved, output)["capture"]
    plan = build_plan(config)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(
        output / ("confirmed_config.json" if args.confirmed_plan else "proposed_config.json"), spec
    )
    atomic_json(output / "capture_plan.json", plan)
    coverage = save_plan_plots(plan, spec, output)
    report = {
        "status": "confirmed_plan_not_captured" if args.confirmed_plan else "proposal_only",
        "real_images_captured": 0,
        "training_steps": 0,
        "position_count": plan["position_count"],
        "headings_per_position": plan["headings_per_position"],
        "planned_sample_count": plan["sample_count"],
        "path_length_m": plan["path_length_m"],
        "max_radius_m": plan["max_radius_m"],
        "max_adjacent_step_m": max(s["distance_m"] for s in plan["adjacent_steps"]),
        "min_adjacent_area_overlap_fraction": min(
            s["footprint_area_overlap_fraction"] for s in plan["adjacent_steps"]
        ),
        "coverage": coverage,
        "pose_signature": plan["pose_signature"],
        "note": ("Plan analysis only: no real images, training, or navigation evaluation."),
    }
    atomic_json(
        output / ("plan_report.json" if args.confirmed_plan else "proposal_report.json"), report
    )
    print(
        ("STAGE05_PLAN " if args.confirmed_plan else "STAGE05_PROPOSAL ") + str(output), flush=True
    )
    print(json.dumps({key: value for key, value in report.items() if key != "coverage"}, indent=2))


if __name__ == "__main__":
    main()
