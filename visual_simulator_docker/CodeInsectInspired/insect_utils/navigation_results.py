"""Integrity-checked episode records and honest grouped navigation statistics."""

from collections import Counter
import math
from pathlib import Path

from insect_utils.experiment_config import digest
from insect_utils.experiment_state import atomic_json, read_json
from insect_utils.navigation_core import horizontal_distance
from insect_utils.training_state import distribution


def saved_episodes(folder, phase, frozen, verify_files=True):
    results = {}
    for file in sorted((Path(folder) / phase / "episodes").glob("*/summary.json")):
        result = read_json(file)
        if result["fingerprint"] != frozen["fingerprint"] or result["episode_id"] in results:
            raise ValueError("Episode fingerprint mismatch or duplicate ID")
        if result["episode_id"] != file.parent.name:
            raise ValueError("Episode summary is stored under another episode ID")
        if verify_files:
            for name, expected in result["artifacts"].items():
                path = file.parent / name
                path.resolve().relative_to(file.parent.resolve())
                if digest(path) != expected:
                    raise ValueError("Completed navigation artifact changed: " + str(path))
        results[result["episode_id"]] = result
    return results


def summarize(results):
    rows = list(results)
    eligible = [r for r in rows if r["reason"] != "initial_within_success"]
    successes = [r for r in eligible if r["reason"] == "success"]
    distances = [r["final_distance_m"] for r in eligible]
    travel_ratios = [
        r["path_length_m"] / max(r["initial_distance_m"] - r["success_radius_m"], 1e-8)
        for r in successes
    ]
    output = {
        "episodes": len(rows),
        "nontrivial_episodes": len(eligible),
        "initial_within_success": len(rows) - len(eligible),
        "successes": len(successes),
        "success_rate": len(successes) / len(eligible) if eligible else None,
        "termination_reasons": dict(Counter(r["reason"] for r in rows)),
        "final_distance_m": distribution(distances),
        "minimum_endpoint_distance_m": distribution(
            [r["minimum_endpoint_distance_m"] for r in eligible]
        ),
        "steps": distribution([r["steps"] for r in eligible]),
        "successful_path_ratio_to_success_circle": distribution(travel_ratios),
    }
    output["by_radius"] = {}
    for radius in sorted({r["radius_m"] for r in rows}):
        selected = [r for r in eligible if r["radius_m"] == radius]
        count = sum(r["reason"] == "success" for r in selected)
        output["by_radius"][str(radius)] = {
            "episodes": len(selected),
            "successes": count,
            "success_rate": count / len(selected) if selected else None,
            "final_distance_m": distribution([r["final_distance_m"] for r in selected]),
            "termination_reasons": dict(Counter(r["reason"] for r in selected)),
        }
    output["by_initial_yaw"] = {}
    for yaw in sorted({r["initial_yaw_deg"] for r in rows}):
        selected = [r for r in eligible if r["initial_yaw_deg"] == yaw]
        count = sum(r["reason"] == "success" for r in selected)
        output["by_initial_yaw"][str(yaw)] = {
            "episodes": len(selected),
            "successes": count,
            "success_rate": count / len(selected) if selected else None,
        }
    return output


def complete_episode(folder, phase, state, reason, frozen, detail=None):
    base = Path(folder) / phase / "episodes" / state["episode"]["episode_id"]
    path = state["attempt"]
    steps = len(state["actions"])
    positions = [state["episode"]["initial_position_m"]] + [
        r["executed_position_after_m"] for r in state["actions"]
    ]
    home = frozen["protocol"]["home_camera_xyz_m"]
    controller = frozen["protocol"]["controller"]
    distance = horizontal_distance(state["position"], home)
    if reason == "success" and distance > controller["success_radius_m"] + 1e-8:
        raise ValueError("Successful episode is outside the authorized success circle")
    if reason == "timeout" and steps != controller["max_steps"]:
        raise ValueError("Timeout has the wrong movement budget")
    atomic_json(path / "trajectory.json", {"positions_m": positions, "actions": state["actions"]})
    artifacts = {
        str(p.relative_to(base)): digest(p) for p in sorted(path.rglob("*")) if p.is_file()
    }
    summary = dict(
        state["episode"],
        fingerprint=frozen["fingerprint"],
        reason=reason,
        detail=detail,
        success_radius_m=controller["success_radius_m"],
        steps=steps,
        final_position_m=state["position"],
        final_yaw_deg=state["yaw"],
        initial_distance_m=horizontal_distance(positions[0], home),
        final_distance_m=distance,
        minimum_endpoint_distance_m=min(horizontal_distance(p, home) for p in positions),
        minimum_segment_distance_m=min(
            [r["segment_min_distance_m"] for r in state["actions"]]
            + [horizontal_distance(positions[0], home)]
        ),
        path_length_m=sum(math.dist(a[:2], b[:2]) for a, b in zip(positions, positions[1:])),
        attempt=str(path.relative_to(base)),
        artifacts=artifacts,
    )
    destination = base / "summary.json"
    if destination.exists():
        raise FileExistsError("Completed navigation episodes cannot be overwritten")
    atomic_json(destination, summary)
    return summary


def commit_phase(folder, phase, frozen):
    results = saved_episodes(folder, phase, frozen)
    expected = {e["episode_id"] for e in frozen["episodes"][phase]}
    if set(results) != expected:
        raise ValueError("Missing or extra navigation episodes in " + phase)
    summary = summarize(results.values())
    if phase == "rendered_oracle" and summary["successes"] != len(expected):
        raise ValueError("Rendered oracle must pass every episode before neural navigation")
    atomic_json(
        Path(folder) / phase / "results.json",
        {
            "fingerprint": frozen["fingerprint"],
            "summary": summary,
            "episodes": list(results.values()),
        },
    )
    atomic_json(
        Path(folder) / phase / "completion.json",
        {
            "status": "completed",
            "fingerprint": frozen["fingerprint"],
            "episodes": len(results),
            "results_sha256": digest(Path(folder) / phase / "results.json"),
        },
    )
    return summary
