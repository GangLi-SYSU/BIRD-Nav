"""Freeze navigation conditions, selected model, scene and exact executable sources."""

from pathlib import Path

from insect_utils.coverage_plan import verify_freeze as verify_training
from insect_utils.experiment_config import PROJECT, canonical_hash, digest
from insect_utils.experiment_state import atomic_json, read_json
from insect_utils.navigation_core import make_episodes, safe_segment
from insect_utils.scene_geometry import require_assets, spatial_budget


PROTOCOL = "config/experiments/stage12_navigation_protocol.json"
PHASES = ("geometry", "rendered_oracle", "model_check", "pilot", "formal")


def prepare_navigation(folder):
    protocol = read_json(PROJECT / PROTOCOL)
    parent_folder = PROJECT / protocol["training_run"]
    parent = verify_training(parent_folder)
    completion = read_json(parent_folder / "completion.json")
    if completion["status"] != "completed" or completion["fingerprint"] != parent["fingerprint"]:
        raise ValueError("Navigation requires a completed frozen training experiment")
    selection = read_json(parent_folder / "selection_frozen_before_test.json")
    if selection["selected_arm"] != "normalized":
        raise ValueError("Unexpected validation-selected model; review protocol explicitly")
    if (
        protocol["scene"] != parent["protocol"]["scene"]
        or protocol["camera"] != parent["protocol"]["camera"]
        or protocol["home_camera_xyz_m"] != [0, 0, parent["protocol"]["height_agl_m"]]
    ):
        raise ValueError("Navigation scene/height/optics differs from training")
    export = parent_folder / "arms/normalized/deployment"
    delivery = read_json(export / "export_check.json")
    scripted = export / "home_vector_m.torchscript.pt"
    if delivery["status"] != "passed" or digest(scripted) != delivery["torchscript_sha256"]:
        raise ValueError("Physical-unit deployment model failed integrity checks")
    weights = parent_folder / "arms/normalized/best_validation.pth"
    if digest(weights) != selection["arms"]["normalized"]["files"]["best_validation.pth"]:
        raise ValueError("Validation-selected source weights changed")
    manifest = require_assets(PROJECT / protocol["scene"])
    spec = {
        "camera": protocol["camera"],
        "scene": {"ground_z_m": protocol["ground_z_m"]},
        "flight": {
            "camera_height_agl_m": protocol["home_camera_xyz_m"][2] - protocol["ground_z_m"]
        },
        "home": {"ground_xy_m": protocol["home_camera_xyz_m"][:2]},
        "evaluation": {
            "radius_m": max(protocol["formal"]["radii_m"]),
            "radius_scales": [1],
            "max_steps": protocol["controller"]["max_steps"],
            "max_step_m": protocol["controller"]["max_step_m"],
            "pitch_roll_deg": 0,
        },
    }
    budget = spatial_budget(spec, manifest, protocol["runtime"]["safety_margin_m"])
    budget["boundary_guard_connected_to_controller"] = True
    if budget["minimum_surface_clearance_m"] < 1:
        raise ValueError("Flight plane does not clear every declared scene surface")
    episodes = {
        phase: make_episodes(protocol[phase], protocol["home_camera_xyz_m"], phase)
        for phase in ("rendered_oracle", "pilot", "formal")
    }
    if len(episodes["pilot"]) != 48 or len(episodes["formal"]) != 320:
        raise ValueError("Authorized pilot/formal counts changed")
    for items in episodes.values():
        for episode in items:
            safe_segment(episode["initial_position_m"], episode["initial_position_m"], budget)
    sources = dict(parent["files"])
    extras = list((PROJECT / "insect_utils").glob("navigation_*.py"))
    extras += list((PROJECT / "tests").glob("test_stage12*.py"))
    extras += [
        PROJECT / "run_navigation_experiment.py",
        PROJECT / PROTOCOL,
        scripted,
        weights,
        parent_folder / "selection_frozen_before_test.json",
        parent_folder / "completion.json",
        export / "export_check.json",
        PROJECT / protocol["scene"],
    ]
    for path in extras:
        sources[str(path.relative_to(PROJECT))] = digest(path)
    value = {
        "protocol": protocol,
        "episodes": episodes,
        "budget": budget,
        "files": sources,
        "training_fingerprint": parent["fingerprint"],
        "model": {
            "arm": "normalized",
            "epoch": 101,
            "scale_m": 44,
            "torchscript": str(scripted.relative_to(PROJECT)),
            "weights": str(weights.relative_to(PROJECT)),
            "output_units": "body-frame home vector in meters",
        },
    }
    value["fingerprint"] = canonical_hash(value)
    folder = Path(folder)
    destination = folder / "frozen.json"
    if destination.exists():
        if read_json(destination) != value:
            raise ValueError("Navigation is frozen; use another run directory after changes")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(destination, value)
        atomic_json(folder / "protocol.json", protocol)
        atomic_json(folder / "episode_plan.json", episodes)
    return value


def verify_navigation(folder):
    frozen = read_json(Path(folder) / "frozen.json")
    if (
        canonical_hash({k: v for k, v in frozen.items() if k != "fingerprint"})
        != frozen["fingerprint"]
    ):
        raise ValueError("Navigation fingerprint changed")
    for name, expected in frozen["files"].items():
        if digest(PROJECT / name) != expected:
            raise ValueError("Frozen navigation source/model/asset changed: " + name)
    return frozen
