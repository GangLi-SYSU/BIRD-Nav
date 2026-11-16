"""Predeclare a 20m-only comparison before inspecting either navigation outcome."""

from pathlib import Path

from insect_utils.circular_training import verify_circular
from insect_utils.experiment_config import PROJECT, canonical_hash, digest
from insect_utils.experiment_state import atomic_json, read_json
from insect_utils.navigation_core import make_episodes, safe_segment, simulate_oracle
from insect_utils.navigation_protocol import verify_navigation


PROTOCOL = "config/experiments/stage14_rotation_r20_protocol.json"


def validate_protocol(protocol, training, baseline):
    if protocol["arms"] != ["mask_only", "mask_rotation"]:
        raise ValueError("Both approved circular controls are required")
    if protocol["training_fingerprint"] != training["fingerprint"]:
        raise ValueError("Circular training fingerprint changed")
    for key in ("scene", "camera", "controller", "home_camera_xyz_m", "ground_z_m", "runtime"):
        if protocol[key] != baseline["protocol"][key]:
            raise ValueError("Historical scene, camera, controller or runtime changed: " + key)
    if protocol["formal"] != dict(
        radii_m=[20], bearings_per_radius=16, initial_yaws_deg=[0, 90, 180, 270]
    ):
        raise ValueError("Only the approved64 initial conditions at20m may be tested")
    if protocol["rendered_oracle"]["radii_m"] != [20]:
        raise ValueError("Rendered engineering returns must also start at20m only")


def prepare_comparison(folder):
    folder = Path(folder).resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    protocol = read_json(PROJECT / PROTOCOL)
    training = verify_circular(PROJECT / protocol["training_run"])
    baseline = verify_navigation(PROJECT / "runs/stage12_navigation_01")
    validate_protocol(protocol, training, baseline)
    plans = {
        phase: make_episodes(protocol[phase], protocol["home_camera_xyz_m"], phase)
        for phase in ("rendered_oracle", "formal")
    }
    for episodes in plans.values():
        for episode in episodes:
            safe_segment(
                episode["initial_position_m"], episode["initial_position_m"], baseline["budget"]
            )
    files = dict(training["files"])
    extras = list((PROJECT / "insect_utils").glob("navigation_*.py"))
    extras += list((PROJECT / "insect_utils").glob("rotation_*.py"))
    extras += [
        PROJECT / PROTOCOL,
        PROJECT / "run_rotation_comparison.py",
        PROJECT / "run_navigation_experiment.py",
        PROJECT / "tests/test_stage14_rotation.py",
    ]
    files.update({str(p.relative_to(PROJECT)): digest(p) for p in extras})
    value = dict(
        protocol=protocol,
        files=files,
        episodes=plans,
        budget=baseline["budget"],
        training_fingerprint=training["fingerprint"],
    )
    value["fingerprint"] = canonical_hash(value)
    path = folder / "predeclared.json"
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Comparison already declared; use a new run directory after changes")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(path, value)
        atomic_json(folder / "protocol.json", protocol)
        atomic_json(folder / "episode_plan.json", plans)
    return value


def verify_comparison(folder):
    value = read_json(Path(folder) / "predeclared.json")
    if (
        canonical_hash({k: v for k, v in value.items() if k != "fingerprint"})
        != value["fingerprint"]
    ):
        raise ValueError("Predeclared comparison fingerprint changed")
    for name, expected in value["files"].items():
        if digest(PROJECT / name) != expected:
            raise ValueError("Predeclared comparison source changed: " + name)
    return value


def training_ready(declared):
    base = PROJECT / declared["protocol"]["training_run"]
    progress_file = base / "progress.json"
    selection = base / "selection_frozen_before_test.json"
    if not progress_file.exists() or not selection.exists():
        return False
    progress = read_json(progress_file)
    return progress.get("phase") == "train" and progress.get("status") == "completed"


def prepare_arm(folder, arm):
    folder = Path(folder)
    declared = verify_comparison(folder)
    if arm not in declared["protocol"]["arms"] or not training_ready(declared):
        raise ValueError("Both formal training arms and validation selections must finish first")
    training_folder = PROJECT / declared["protocol"]["training_run"]
    training = verify_circular(training_folder)
    selection = read_json(training_folder / "selection_frozen_before_test.json")
    if selection["fingerprint"] != training["fingerprint"]:
        raise ValueError("Validation selection has a different training fingerprint")
    for name in declared["protocol"]["arms"]:
        completed = read_json(training_folder / "arms" / name / "completion.json")
        if (
            completed["status"] != "completed"
            or completed["smoke"]
            or completed["epochs"] != 150
            or completed["optimizer_steps"] != 76800
            or completed != selection["arms"][name]
        ):
            raise ValueError("Formal training arm is incomplete or its selection changed")
    completed = selection["arms"][arm]
    base = training_folder / "arms" / arm
    files = dict(declared["files"])
    for name in (
        "best_validation.pth",
        "deployment/home_vector_m.torchscript.pt",
        "deployment/export_check.json",
    ):
        expected = completed["files"][name]
        if digest(base / name) != expected:
            raise ValueError("Selected circular model/export changed")
        files[str((base / name).relative_to(PROJECT))] = expected
    extra = [folder / "predeclared.json", training_folder / "selection_frozen_before_test.json"]
    files.update({str(p.relative_to(PROJECT)): digest(p) for p in extra})
    value = dict(
        protocol=declared["protocol"],
        episodes=declared["episodes"],
        budget=declared["budget"],
        files=files,
        comparison_fingerprint=declared["fingerprint"],
        model=dict(
            arm=arm,
            epoch=completed["best_validation"]["epoch"],
            scale_m=44,
            torchscript=str(
                (base / "deployment/home_vector_m.torchscript.pt").relative_to(PROJECT)
            ),
            weights=str((base / "best_validation.pth").relative_to(PROJECT)),
            output_units="body-frame home vector in meters",
        ),
    )
    value["fingerprint"] = canonical_hash(value)
    arm_folder = folder / "arms" / arm
    path = arm_folder / "frozen.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("Arm navigation model or protocol is already frozen")
    if not path.exists():
        atomic_json(path, value)
        atomic_json(arm_folder / "protocol.json", value["protocol"])
        atomic_json(arm_folder / "episode_plan.json", value["episodes"])
    return value


def geometry_check(folder, frozen):
    """Pure geometry checks also keep the approved20m initial radius."""
    checks = []
    for home in ([0, 0, 40], [13, -8, 40]):
        for episode in make_episodes(frozen["protocol"]["formal"], home, "geometry"):
            result = simulate_oracle(
                episode, home, frozen["protocol"]["controller"], frozen["budget"]
            )
            if result["reason"] != "success":
                raise ValueError("20m oracle geometry failed")
            checks.append(dict(home=home, episode=episode, result=result))
    path = Path(folder) / "geometry/results.json"
    atomic_json(path, dict(checks=checks))
    atomic_json(
        path.with_name("completion.json"),
        dict(
            status="completed",
            fingerprint=frozen["fingerprint"],
            oracle_checks=len(checks),
            results_sha256=digest(path),
        ),
    )
