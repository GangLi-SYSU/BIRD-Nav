"""Resolve one experiment specification without importing Isaac, NumPy or PyTorch."""

import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re


PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL = "config/experiments/stage01_protocol.json"
CAMERA_PROTOCOL = "config/experiments/stage04_camera_protocol.json"
SAMPLING_PROTOCOL = "config/experiments/stage05_sampling_protocol.json"
DATA_PROTOCOL = "config/experiments/stage06_data_protocol.json"
STAGES = ("capture", "train", "evaluate")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError("Use a run/experiment name containing letters, digits, _, - or .")
    return value


def finite_number(value, field, minimum=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(field + " must be a finite number")
    if integer and not isinstance(value, int):
        raise ValueError(field + " must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(field + " is below its minimum")


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Scene and artifact paths must be relative and cannot contain '..'")
    return path


def validate_spec(spec):
    required = {
        "schema_version",
        "name",
        "purpose",
        "seed",
        "output_root",
        "scene",
        "home",
        "flight",
        "camera",
        "capture",
        "training",
        "evaluation",
        "runtime",
        "declared_changes",
    }
    if set(spec) != required or spec["schema_version"] != 1:
        raise ValueError("Unknown/missing experiment fields or schema version")
    sections = {
        "scene": {"path", "ground_z_m"},
        "home": {"ground_xy_m"},
        "flight": {"camera_height_agl_m"},
        "camera": {"mode", "raw_resolution_wh", "hfov_deg", "fov_definition_confirmed", "crop_box"},
        "capture": {
            "backend",
            "shape_flight",
            "shape_params",
            "number_cameras",
            "sample_count_policy",
        },
        "training": {
            "model_type",
            "criterion",
            "optimizer",
            "learning_rate",
            "optimizer_kwargs",
            "batch_size",
            "batch_size_val",
            "batch_size_eval",
            "epochs",
            "split_ratio",
            "augment",
            "num_workers",
        },
        "evaluation": {
            "radius_m",
            "radius_scales",
            "points_per_radius",
            "max_steps",
            "success_radius_m",
            "step_rule",
            "max_step_m",
            "pitch_roll_deg",
            "north",
            "fixed_yaw",
        },
        "runtime": {"headless", "torch_num_threads"},
    }
    for name, fields in sections.items():
        optional = {"headings_per_position"} if name == "capture" else set()
        if (
            not isinstance(spec[name], dict)
            or not fields <= set(spec[name])
            or not set(spec[name]) <= fields | optional
        ):
            raise ValueError("Unknown/missing fields in " + name)
    if not isinstance(spec["output_root"], str) or not spec["output_root"].strip():
        raise ValueError("output_root must be a nonempty path string")
    if not isinstance(spec["declared_changes"], list) or not all(
        isinstance(v, str) for v in spec["declared_changes"]
    ):
        raise ValueError("declared_changes must be a list of descriptions")
    identifier(spec["name"])
    if spec["purpose"] not in ("research", "engineering_smoke"):
        raise ValueError("Unknown experiment purpose")
    finite_number(spec["seed"], "seed", 0, True)
    if spec["seed"] >= 2 ** 32:
        raise ValueError("seed must fit NumPy's unsigned 32-bit seed range")
    relative_path(spec["scene"]["path"])
    finite_number(spec["scene"]["ground_z_m"], "scene.ground_z_m")
    if len(spec["home"]["ground_xy_m"]) != 2:
        raise ValueError("home.ground_xy_m must contain two coordinates")
    for value in spec["home"]["ground_xy_m"]:
        finite_number(value, "home.ground_xy_m")
    finite_number(spec["flight"]["camera_height_agl_m"], "flight.camera_height_agl_m", 0.001)
    camera = spec["camera"]
    if camera["mode"] not in ("panorama_legacy", "downward_pinhole"):
        raise ValueError("Unknown camera mode")
    if len(camera["raw_resolution_wh"]) != 2:
        raise ValueError("Camera resolution must be [width, height]")
    for value in camera["raw_resolution_wh"]:
        finite_number(value, "camera.raw_resolution_wh", 1, True)
    if camera["mode"] == "panorama_legacy":
        if (
            camera["raw_resolution_wh"] != [1024, 1024]
            or camera["crop_box"] != [0, 312, 1024, 712]
            or camera["hfov_deg"] is not None
        ):
            raise ValueError("Stage B legacy bridge supports the original resolution and crop only")
    else:
        finite_number(camera["hfov_deg"], "camera.hfov_deg", 0.001)
        if camera["hfov_deg"] >= 179 or camera["crop_box"] is not None:
            raise ValueError("Downward pinhole needs HFOV <179 and no panorama crop")
        if spec["training"]["augment"]:
            raise ValueError("Panorama wrap augmentation is invalid for downward images")
    capture = spec["capture"]
    if capture["backend"] not in ("isaac", "synthetic_smoke"):
        raise ValueError("Unknown capture backend")
    if capture["backend"] == "synthetic_smoke" and spec["purpose"] != "engineering_smoke":
        raise ValueError("Synthetic fixtures cannot be used for research runs")
    if capture["shape_flight"] != "bee" or len(capture["shape_params"]) != 3:
        raise ValueError("Stage B supports the original bee path [n, m, b]")
    for index, value in enumerate(capture["shape_params"]):
        finite_number(value, "capture.shape_params", 4 if index == 1 else 0.001, index < 2)
    finite_number(capture["number_cameras"], "capture.number_cameras", 1, True)
    if "headings_per_position" in capture:
        finite_number(capture["headings_per_position"], "capture.headings_per_position", 1, True)
        if capture["headings_per_position"] not in (1, 4, 8):
            raise ValueError("Select 1, 4 or 8 real headings per learning position")
        if camera["mode"] != "downward_pinhole":
            raise ValueError("Explicit heading coverage belongs to downward capture only")
    if capture["sample_count_policy"] not in ("exact", "legacy_truncate"):
        raise ValueError("Unknown sample count policy")
    training = spec["training"]
    if (training["model_type"], training["criterion"], training["optimizer"]) != (
        "attention",
        "MSELoss",
        "AdamW",
    ):
        raise ValueError("This experiment preserves the attention/MSE/AdamW network contract")
    for key in ("epochs", "batch_size", "batch_size_val", "batch_size_eval"):
        finite_number(training[key], "training." + key, 1, True)
    finite_number(training["num_workers"], "training.num_workers", 0, True)
    finite_number(training["learning_rate"], "training.learning_rate", 1e-12)
    finite_number(training["split_ratio"], "training.split_ratio", 0.001)
    if training["split_ratio"] >= 1:
        raise ValueError("training.split_ratio must be below 1")
    if not isinstance(training["augment"], bool):
        raise ValueError("training.augment must be boolean")
    optimizer = training["optimizer_kwargs"]
    if (
        set(optimizer) != {"betas", "eps", "weight_decay", "amsgrad"}
        or len(optimizer["betas"]) != 2
    ):
        raise ValueError("Explicit AdamW options must be betas/eps/weight_decay/amsgrad")
    for beta in optimizer["betas"]:
        finite_number(beta, "optimizer.betas", 0)
        if beta >= 1:
            raise ValueError("AdamW betas must be below 1")
    finite_number(optimizer["eps"], "optimizer.eps", 1e-20)
    finite_number(optimizer["weight_decay"], "optimizer.weight_decay", 0)
    evaluation = spec["evaluation"]
    if evaluation["step_rule"] != "legacy_norm_v1":
        raise ValueError("Unknown movement rule; no silent controller replacement is allowed")
    for key in ("radius_m", "success_radius_m", "max_step_m"):
        finite_number(evaluation[key], "evaluation." + key, 0.001)
    for key in ("points_per_radius", "max_steps"):
        finite_number(evaluation[key], "evaluation." + key, 1, True)
    if not evaluation["radius_scales"]:
        raise ValueError("At least one evaluation radius is required")
    for value in evaluation["radius_scales"]:
        finite_number(value, "evaluation.radius_scales", 0.001)
    finite_number(evaluation["pitch_roll_deg"], "evaluation.pitch_roll_deg", 0)
    for value in (
        evaluation["north"],
        evaluation["fixed_yaw"],
        spec["runtime"]["headless"],
        optimizer["amsgrad"],
        camera["fov_definition_confirmed"],
    ):
        if not isinstance(value, bool):
            raise ValueError("Runtime and control flags must be boolean")
    finite_number(spec["runtime"]["torch_num_threads"], "runtime.torch_num_threads", 1, True)


def resolve_spec(spec, project=PROJECT):
    from insect_utils.scene_geometry import asset_provenance
    from insect_utils.camera_geometry import CONTRACT, optics
    from insect_utils.frame_stability import STABILITY_POLICY

    validate_spec(spec)
    spec = copy.deepcopy(spec)
    home_xy = spec["home"]["ground_xy_m"]
    ground = spec["scene"]["ground_z_m"]
    height = spec["flight"]["camera_height_agl_m"]
    position_count = spec["capture"]["shape_params"][1] - 3
    heading_count = spec["capture"].get("headings_per_position", 1)
    planned = position_count * heading_count
    cameras = spec["capture"]["number_cameras"]
    expected = (
        planned
        if spec["capture"]["sample_count_policy"] == "exact"
        else (planned // cameras) * cameras
    )
    width, image_height = spec["camera"]["raw_resolution_wh"]
    crop = spec["camera"]["crop_box"]
    derived = {
        "home_ground_xyz_m": home_xy + [ground],
        "home_camera_xyz_m": home_xy + [ground + height],
        "planned_samples": planned,
        "expected_samples": expected,
        "planned_positions": position_count,
        "headings_per_position": heading_count,
        "processed_resolution_wh": [crop[2] - crop[0], crop[3] - crop[1]]
        if crop
        else [width, image_height],
    }
    if spec["camera"]["mode"] == "downward_pinhole":
        tangent = math.tan(math.radians(spec["camera"]["hfov_deg"]) / 2)
        derived["vfov_deg"] = math.degrees(2 * math.atan(tangent * image_height / width))
        derived["flat_ground_footprint_wh_m"] = [
            2 * height * tangent,
            2 * height * tangent * image_height / width,
        ]
        derived["camera_contract"] = copy.deepcopy(CONTRACT)
        derived["camera_acquisition_policy"] = dict(STABILITY_POLICY)
        if spec["camera"]["fov_definition_confirmed"]:
            derived["camera_optics"] = optics(spec["camera"])
    scene = project / spec["scene"]["path"]
    sources = sorted(project.glob("*.py")) + sorted((project / "insect_utils").glob("*.py"))
    versions = {}
    for name in ("torch", "torchvision", "numpy", "scipy", "Pillow", "onnx"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    provenance = {
        "source_sha256": {str(p.relative_to(project)): digest(p) for p in sources},
        "scene_sha256": digest(scene) if scene.is_file() else None,
        "scene_bundle": asset_provenance(scene),
        "runtime_versions": {"python": platform.python_version(), "packages": versions},
        "baseline_protocol_sha256": digest(project / PROTOCOL),
        "camera_protocol_sha256": digest(project / CAMERA_PROTOCOL)
        if (project / CAMERA_PROTOCOL).is_file()
        else None,
        "sampling_protocol_sha256": digest(project / SAMPLING_PROTOCOL)
        if (project / SAMPLING_PROTOCOL).is_file()
        else None,
        "data_protocol_sha256": digest(project / DATA_PROTOCOL)
        if (project / DATA_PROTOCOL).is_file()
        else None,
    }
    payload = {"spec": spec, "derived": derived, "provenance": provenance}
    return dict(payload, fingerprint=canonical_hash(payload))


def load_config(path, project=PROJECT):
    with Path(path).open(encoding="utf-8") as stream:
        return resolve_spec(json.load(stream), project)


def preflight(resolved, stage, project=PROJECT):
    spec = resolved["spec"]
    if (
        stage == "train"
        and spec["camera"]["mode"] == "downward_pinhole"
        and spec["capture"]["backend"] == "isaac"
    ):
        raise ValueError(
            "Downward training uses stage F data via the isolated run_downward_training.py entry"
        )
    if stage in ("capture", "evaluate"):
        if spec["capture"]["backend"] == "synthetic_smoke":
            if stage == "evaluate":
                raise ValueError(
                    "Synthetic engineering data cannot produce closed-loop homing results"
                )
            return
        if spec["camera"]["mode"] == "downward_pinhole":
            if stage == "evaluate":
                raise ValueError("Downward closed-loop controller is not implemented yet (stage H)")
            if "headings_per_position" not in spec["capture"]:
                raise ValueError(
                    "Stage E: explicitly confirm and set capture.headings_per_position"
                )
            if spec["capture"]["sample_count_policy"] != "exact":
                raise ValueError("Downward capture requires exact sample accounting")
            from insect_utils.camera_geometry import optics
            from insect_utils.scene_geometry import require_assets, spatial_budget

            optics(spec["camera"])
            manifest = require_assets(project / spec["scene"]["path"])
            spatial_budget(spec, manifest)
            return
        if not (project / spec["scene"]["path"]).is_file():
            raise ValueError("Scene does not exist; complete stage C first")
        planned = resolved["derived"]["planned_samples"]
        if (
            stage == "capture"
            and spec["capture"]["sample_count_policy"] == "exact"
            and planned % spec["capture"]["number_cameras"]
        ):
            raise ValueError(
                "Legacy capture would drop trailing samples; "
                "complete stage E or declare legacy_truncate"
            )


def stage_parameters(resolved, run_dir, project=PROJECT):
    spec, derived = resolved["spec"], resolved["derived"]
    capture_dir = run_dir / "capture"
    train_dir = run_dir / "train"
    camera = spec["camera"]
    render = {
        "map": str(project / spec["scene"]["path"]),
        "output_folder": str(capture_dir),
        "home_position": derived["home_camera_xyz_m"],
        "shape_flight": spec["capture"]["shape_flight"],
        "shape_params": spec["capture"]["shape_params"],
        "number_cameras": spec["capture"]["number_cameras"],
        "n_positions": derived["planned_samples"],
        "use_noisy_spiral": False,
        "only_point_north": False,
        "csv_filename": "dataset_navigation.csv",
        "crop": camera["crop_box"] is not None,
        "crop_box_params": camera["crop_box"],
        "image_size": camera["raw_resolution_wh"][0],
        "headless": spec["runtime"]["headless"],
        "camera": copy.deepcopy(camera),
        "camera_contract": derived.get("camera_contract"),
        "camera_acquisition_policy": derived.get("camera_acquisition_policy"),
    }
    if camera["mode"] == "downward_pinhole":
        render.update(
            seed=spec["seed"],
            ground_z_m=spec["scene"]["ground_z_m"],
            camera_height_agl_m=spec["flight"]["camera_height_agl_m"],
            headings_per_position=spec["capture"].get("headings_per_position"),
            sample_count_policy=spec["capture"]["sample_count_policy"],
            experiment_spec=copy.deepcopy(spec),
        )
    training = dict(
        spec["training"],
        dataset_folder=str(capture_dir),
        image_folder=str(capture_dir / "Replicator/rgb"),
        csv_filename="dataset_navigation.csv",
        evaluation_dataset_folder=str(capture_dir),
        evaluation_image_folder=str(capture_dir / "Replicator/rgb"),
        evaluation_csv_filename="dataset_navigation.csv",
        models_folder=str(train_dir),
        model_name="model_attention",
        model_suffix="_attention",
        indices_file=str(train_dir / "indices.pkl"),
        training_log_path=str(train_dir / "training_history.json"),
    )
    evaluation = spec["evaluation"]
    homing = {
        "map": render["map"],
        "model": str(train_dir / "model_attention.pth"),
        "params": {
            "step_size": 0.5,
            "count_limit": evaluation["max_steps"],
            "threshold": evaluation["success_radius_m"],
            "max_step": evaluation["max_step_m"],
        },
        "start_location_pattern": "circle",
        "circle_params": {
            "radius": evaluation["radius_m"],
            "radius_scales": evaluation["radius_scales"],
            "num_points": evaluation["points_per_radius"],
        },
        "virtual_home_point": derived["home_camera_xyz_m"],
        "pitch_roll": evaluation["pitch_roll_deg"],
        "north": evaluation["north"],
        "fixed_yaw": evaluation["fixed_yaw"],
        "save_images": False,
        "render_config": render,
    }
    return {"capture": render, "train": training, "evaluate": homing}
