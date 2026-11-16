"""Freeze the authorized epoch extension and preserve the original data/navigation protocol."""

from pathlib import Path

from insect_utils.circular_training import verify_circular
from insect_utils.experiment_config import PROJECT, canonical_hash, digest
from insect_utils.experiment_state import atomic_json, read_json
from insect_utils.navigation_protocol import verify_navigation
from insect_utils.training_state import checkpoint_pointer


PROTOCOL = "config/experiments/stage15_rotation_continuation_protocol.json"


def require_epoch_only_change(original, extended):
    expected = dict(original, epochs=300)
    if original["epochs"] != 150 or extended != expected:
        raise ValueError("Only the total epoch budget may change, from150 to300")


def prepare_continuation(folder):
    folder = Path(folder).resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    protocol = read_json(PROJECT / PROTOCOL)
    if (
        protocol["source_epoch"] != 150
        or protocol["target_epochs"] != 300
        or protocol["source_optimizer_steps"] != 76800
        or protocol["target_optimizer_steps"] != 153600
    ):
        raise ValueError("Expected exactly150 additional epochs and76800 additional updates")
    parent_folder = PROJECT / protocol["source_training_run"]
    parent = verify_circular(parent_folder)
    if parent["fingerprint"] != protocol["source_training_fingerprint"]:
        raise ValueError("Source circular training experiment changed")
    arm_folder = parent_folder / "arms" / protocol["source_arm"]
    completion = read_json(arm_folder / "completion.json")
    contract = read_json(arm_folder / "contract.json")
    if (
        completion["status"] != "completed"
        or completion["smoke"]
        or completion["fingerprint"] != protocol["source_arm_fingerprint"]
        or completion["epochs"] != protocol["source_epoch"]
        or completion["optimizer_steps"] != protocol["source_optimizer_steps"]
        or contract["configuration"] != parent["protocol"]["training"]
        or not contract["random_rotation"]
    ):
        raise ValueError("Expected the completed original augmentation arm at epoch150")
    checkpoint, pointer = checkpoint_pointer(
        arm_folder / "engine_scaled_units", completion["fingerprint"]
    )
    if (
        pointer["epoch"] != 150
        or pointer["optimizer_steps"] != 76800
        or pointer["sha256"] != protocol["source_checkpoint_sha256"]
    ):
        raise ValueError("The exact complete150-epoch source checkpoint is required")
    config = dict(contract["configuration"], epochs=protocol["target_epochs"])
    require_epoch_only_change(contract["configuration"], config)
    navigation_folder = PROJECT / protocol["source_navigation_run"]
    navigation = verify_navigation(navigation_folder)
    if navigation["model"]["arm"] != "mask_rotation" or navigation["model"]["epoch"] != 150:
        raise ValueError("Expected the historical150-epoch augmentation navigation model")
    nav_keys = (
        "scene",
        "home_camera_xyz_m",
        "ground_z_m",
        "camera",
        "controller",
        "formal",
        "rendered_oracle",
        "runtime",
        "success_definition",
        "supervision",
    )
    nav_protocol = {key: navigation["protocol"][key] for key in nav_keys}
    if (
        nav_protocol["formal"]
        != dict(radii_m=[20], bearings_per_radius=16, initial_yaws_deg=[0, 90, 180, 270])
        or nav_protocol["controller"]["success_radius_m"] != 0.5
        or nav_protocol["controller"]["max_steps"] != 50
    ):
        raise ValueError("The original64 trials at20m and original controller are required")
    files = dict(navigation["files"])
    extras = list((PROJECT / "insect_utils").glob("continuation_*.py"))
    extras += [
        PROJECT / PROTOCOL,
        PROJECT / "run_rotation_continuation.py",
        PROJECT / "tests/test_stage15_continuation.py",
        checkpoint,
        arm_folder / "engine_scaled_units/checkpoint_latest.json",
        arm_folder / "completion.json",
        arm_folder / "contract.json",
        arm_folder / "physical_history.json",
        navigation_folder / "frozen.json",
        navigation_folder / "formal/results.json",
        navigation_folder / "formal/completion.json",
    ]
    files.update({str(path.relative_to(PROJECT)): digest(path) for path in extras})
    value = dict(
        protocol=protocol,
        training_config=config,
        preprocessing=parent["protocol"]["preprocessing"],
        navigation_protocol=nav_protocol,
        episodes=navigation["episodes"],
        budget=navigation["budget"],
        files=files,
        source_checkpoint=str(checkpoint.relative_to(PROJECT)),
        source_runtime=contract["runtime"],
        source_navigation_fingerprint=navigation["fingerprint"],
    )
    value["fingerprint"] = canonical_hash(value)
    path = folder / "frozen.json"
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Continuation experiment is frozen; use a new run directory")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(path, value)
        atomic_json(folder / "protocol.json", protocol)
    return value


def verify_continuation(folder):
    value = read_json(Path(folder) / "frozen.json")
    if (
        canonical_hash({k: v for k, v in value.items() if k != "fingerprint"})
        != value["fingerprint"]
    ):
        raise ValueError("Continuation fingerprint changed")
    for name, expected in value["files"].items():
        if digest(PROJECT / name) != expected:
            raise ValueError("Frozen continuation source/data/model changed: " + name)
    return value


def prepare_navigation(folder):
    folder = Path(folder)
    frozen = verify_continuation(folder)
    trained = read_json(folder / "training/completion.json")
    if (
        trained["status"] != "completed"
        or trained["epochs_total"] != 300
        or trained["optimizer_steps_total"] != 153600
    ):
        raise ValueError("Complete300-epoch training is required before navigation")
    files = dict(frozen["files"])
    for name, expected in trained["files"].items():
        path = folder / "training" / name
        if digest(path) != expected:
            raise ValueError("Selected continued model or training artifact changed")
        files[str(path.relative_to(PROJECT))] = expected
    for path in (
        folder / "frozen.json",
        folder / "training/completion.json",
        folder / "training/selection_frozen_before_navigation.json",
    ):
        files[str(path.relative_to(PROJECT))] = digest(path)
    base = folder / "training"
    value = dict(
        protocol=frozen["navigation_protocol"],
        episodes=frozen["episodes"],
        budget=frozen["budget"],
        files=files,
        model=dict(
            arm="mask_rotation_continued",
            epoch=trained["best_validation"]["epoch"],
            scale_m=44,
            torchscript=str(
                (base / "deployment/home_vector_m.torchscript.pt").relative_to(PROJECT)
            ),
            weights=str((base / "best_validation.pth").relative_to(PROJECT)),
            output_units="body-frame home vector in meters",
        ),
    )
    value["fingerprint"] = canonical_hash(value)
    path = folder / "navigation/frozen.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("Navigation is already frozen to another model or protocol")
    if not path.exists():
        atomic_json(path, value)
        atomic_json(path.with_name("protocol.json"), value["protocol"])
        atomic_json(path.with_name("episode_plan.json"), value["episodes"])
    return value
