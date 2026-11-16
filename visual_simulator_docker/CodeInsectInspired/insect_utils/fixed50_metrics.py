"""Distance-independent50-action termination and explicit endpoint statistics."""

from collections import Counter
import math
from pathlib import Path

from insect_utils.experiment_config import digest
from insect_utils.experiment_state import atomic_json
from insect_utils.navigation_core import horizontal_distance
from insect_utils.navigation_results import complete_episode, saved_episodes
from insect_utils.training_state import distribution


COMPLETED = "fixed50_complete"


def fixed_termination(position, home, steps, controller):
    """Keep the existing call signature; position and truth distance never stop a trial."""
    if controller["max_steps"] != 50:
        raise ValueError("The approved fixed horizon is exactly50 actions")
    if steps > 50 or steps < 0:
        raise ValueError("Invalid fixed-horizon action count")
    return COMPLETED if steps == 50 else None


def endpoint_metrics(positions, home):
    distances = [horizontal_distance(p, home) for p in positions]
    if not distances:
        raise ValueError("A trajectory needs its initial position")
    first = next((i for i, radius in enumerate(distances) if radius <= 2), None)
    tail = distances[-10:]
    return dict(
        actions=len(positions) - 1,
        final_at_step50_m=distances[-1] if len(positions) == 51 else None,
        first_entry_2m_step=first,
        ever_entered_2m=first is not None,
        final_within_2m=distances[-1] <= 2,
        entered_then_outside_2m_at_end=first is not None and distances[-1] > 2,
        maximum_after_first_entry_m=max(distances[first:]) if first is not None else None,
        last10_mean_distance_m=sum(tail) / len(tail),
        last10_min_distance_m=min(tail),
        last10_max_distance_m=max(tail),
        last10_std_distance_m=(sum((r - sum(tail) / len(tail)) ** 2 for r in tail) / len(tail))
        ** 0.5,
    )


def complete_fixed_episode(folder, phase, state, reason, frozen, detail=None):
    if reason == COMPLETED and len(state["actions"]) != 50:
        raise ValueError("Cannot commit a fixed50 trial before all50 actions execute")
    positions = [state["episode"]["initial_position_m"]] + [
        a["executed_position_after_m"] for a in state["actions"]
    ]
    metrics = endpoint_metrics(positions, frozen["protocol"]["home_camera_xyz_m"])
    return complete_episode(
        folder,
        phase,
        state,
        reason,
        frozen,
        detail=dict(fixed_horizon=metrics, engineering_detail=detail),
    )


def summarize_fixed(rows, scoring_radii):
    rows = list(rows)
    complete = [r for r in rows if r["reason"] == COMPLETED and r["steps"] == 50]
    final = [r["final_distance_m"] for r in complete]
    metrics = [r["detail"]["fixed_horizon"] for r in complete]
    distribution_final = dict(
        distribution(final),
        minimum=min(final) if final else None,
        maximum=max(final) if final else None,
        rms=math.sqrt(sum(r * r for r in final) / len(final)) if final else None,
    )
    return dict(
        episodes=len(rows),
        completed50=len(complete),
        aborted=len(rows) - len(complete),
        termination_reasons=dict(Counter(r["reason"] for r in rows)),
        horizon_actions=50,
        distance_based_termination=False,
        final_distance_m=distribution_final,
        minimum_endpoint_distance_m=distribution(
            [r["minimum_endpoint_distance_m"] for r in complete]
        ),
        ever_entered_2m=sum(m["ever_entered_2m"] for m in metrics),
        final_within_2m=sum(m["final_within_2m"] for m in metrics),
        entered_then_outside_2m_at_end=sum(m["entered_then_outside_2m_at_end"] for m in metrics),
        final_within_radius={
            str(radius): dict(
                count=sum(d <= radius for d in final),
                rate=sum(d <= radius for d in final) / len(final) if final else None,
            )
            for radius in scoring_radii
        },
        final_position_mean_xy_m=[
            sum(r["final_position_m"][i] for r in complete) / len(complete) if complete else None
            for i in (0, 1)
        ],
        last10_mean_distance_m=distribution([m["last10_mean_distance_m"] for m in metrics]),
        last10_max_distance_m=distribution([m["last10_max_distance_m"] for m in metrics]),
    )


def commit_fixed_phase(folder, phase, frozen):
    rows = saved_episodes(folder, phase, frozen)
    if set(rows) != {e["episode_id"] for e in frozen["episodes"][phase]}:
        raise ValueError("Missing or unexpected fixed50 episodes")
    summary = summarize_fixed(rows.values(), frozen["fixed50"]["scoring_radii_m"])
    if summary["completed50"] != 64 or summary["aborted"]:
        atomic_json(Path(folder) / phase / "incomplete_fixed_horizon.json", summary)
        raise RuntimeError(
            "Not all64 trials completed50 genuine actions; inspect exception records"
        )
    result = dict(fingerprint=frozen["fingerprint"], summary=summary, episodes=list(rows.values()))
    atomic_json(Path(folder) / phase / "results.json", result)
    atomic_json(
        Path(folder) / phase / "completion.json",
        dict(
            status="completed",
            fingerprint=frozen["fingerprint"],
            episodes=64,
            completed50=64,
            results_sha256=digest(Path(folder) / phase / "results.json"),
        ),
    )
    return summary
