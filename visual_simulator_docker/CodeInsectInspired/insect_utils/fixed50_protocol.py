"""Freeze a15m comparison that ends after50 actions,independent of home distance."""

import copy
from pathlib import Path

from insect_utils.experiment_config import PROJECT, canonical_hash, digest
from insect_utils.experiment_state import atomic_json, read_json
from insect_utils.navigation_core import make_episodes, safe_segment
from insect_utils.navigation_protocol import verify_navigation
from run_navigation_experiment import verify_phase


PROTOCOL = "config/experiments/stage25_fixed50_protocol.json"


def fixed_protocol(source, spec):
    protocol = copy.deepcopy(source["protocol"])
    if source["model"]["epoch"] != 484 or protocol["formal"] != dict(
        radii_m=[15], bearings_per_radius=16, initial_yaws_deg=[0, 90, 180, 270]
    ):
        raise ValueError("Use the unchanged epoch484 model and its64 original15m starts")
    if spec["horizon_actions"] != 50 or spec["distance_based_termination"] is not False:
        raise ValueError("User requested exactly50 actions without arrival stopping")
    if protocol["controller"]["max_steps"] != 50 or protocol["controller"]["success_radius_m"] != 2:
        raise ValueError("Preserve the50-action budget and2m evaluation radius")
    protocol["termination_policy"] = dict(
        kind="fixed_action_horizon", actions=50, distance_based=False
    )
    protocol["success_definition"] = (
        "Execute50 actions. Score the action50 endpoint and log earlier2m entry. "
        "The controller success_radius_m is evaluation metadata only."
    )
    return protocol


def prepare(folder):
    folder = Path(folder).resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    if (folder / "frozen.json").exists():
        return verify_navigation(folder)
    spec = read_json(PROJECT / PROTOCOL)
    parent = PROJECT / spec["source_navigation_run"]
    source = verify_navigation(parent)
    if source["fingerprint"] != spec["source_navigation_fingerprint"]:
        raise ValueError("Original15m navigation source changed")
    receipt = read_json(parent / "formal/completion.json")
    if receipt["status"] != "completed" or receipt["episodes"] != 64:
        raise ValueError("The arrival-stopped15m reference is incomplete")
    if digest(parent / "formal/results.json") != receipt["results_sha256"]:
        raise ValueError("Reference15m results changed")
    for phase in ("model_check", "rendered_oracle"):
        verify_phase(parent, phase, source)
    protocol = fixed_protocol(source, spec)
    episodes = dict(
        formal=make_episodes(protocol["formal"], protocol["home_camera_xyz_m"], "formal")
    )
    for e in episodes["formal"]:
        p = e["initial_position_m"]
        safe_segment(p, p, source["budget"])
    files = dict(source["files"])
    extras = list((PROJECT / "insect_utils").glob("fixed50_*.py")) + [
        PROJECT / PROTOCOL,
        PROJECT / "run_fixed50_navigation.py",
        PROJECT / "tests/test_stage25_fixed50.py",
        parent / "frozen.json",
        parent / "formal/results.json",
        parent / "formal/completion.json",
        parent / "rendered_oracle/completion.json",
        parent / "rendered_oracle/results.json",
        parent / "model_check/completion.json",
    ]
    files.update({str(p.relative_to(PROJECT)): digest(p) for p in extras})
    value = dict(
        protocol=protocol,
        episodes=episodes,
        model=source["model"],
        budget=source["budget"],
        fixed50=spec,
        files=files,
    )
    value["fingerprint"] = canonical_hash(value)
    atomic_json(folder / "frozen.json", value)
    atomic_json(folder / "protocol.json", protocol)
    atomic_json(folder / "episode_plan.json", episodes)
    return value
