"""Check the frozen legacy protocol without importing or running the main experiment."""

import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT / "config/experiments/stage01_protocol.json"


def literal_assignments(nodes):
    """Read only literal assignments; never eval or import an experiment entrypoint."""
    values = {}
    for node in nodes:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return values


def parse_source(project, relative):
    return ast.parse((project / relative).read_text(encoding="utf-8"))


def function_node(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def subscript_key(node):
    # Python 3.8 wraps subscript values in ast.Index; Python 3.9+ does not.
    value = node.slice
    if isinstance(value, ast.Index):
        value = value.value
    return ast.literal_eval(value)


def apply_entry_overrides(tree, variable, config, names, fields):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == variable
            ):
                continue
            key = subscript_key(target)
            if key in fields:
                if isinstance(node.value, ast.Name):
                    config[key] = names[node.value.id]
                else:
                    config[key] = ast.literal_eval(node.value)


def sampling_branch(tree, pattern):
    for node in tree.body:
        while isinstance(node, ast.If):
            condition = node.test
            if (
                isinstance(condition, ast.Compare)
                and isinstance(condition.left, ast.Name)
                and condition.left.id == "shape_flight"
                and len(condition.ops) == 1
                and isinstance(condition.ops[0], ast.Eq)
                and ast.literal_eval(condition.comparators[0]) == pattern
            ):
                return literal_assignments(node.body)
            node = node.orelse[0] if len(node.orelse) == 1 else None
    raise ValueError("Cannot resolve the learning-flight hyperparameter branch")


def projection_value(tree, keyword=None):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if keyword:
            for item in node.keywords:
                if item.arg == keyword:
                    return ast.literal_eval(item.value)
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "set_projection_type":
            return ast.literal_eval(node.args[0])
    raise ValueError("Cannot locate camera projection")


def inspect_legacy(project):
    """Resolve values overridden by the legacy entry, including its bee branch."""
    entry = parse_source(project, "run_ANN_homing_simulations.py")
    settings = literal_assignments(entry.body)
    branch = sampling_branch(entry, settings["shape_flight"])
    configs = {}
    for name in ("render", "training", "virt_homing"):
        path = project / ("config/config_" + name + ".json")
        configs[name] = json.loads(path.read_text(encoding="utf-8"))
    render, training, homing = (configs[n] for n in ("render", "training", "virt_homing"))
    names = dict(settings, **branch)
    apply_entry_overrides(
        entry,
        "config_render",
        render,
        names,
        {
            "shape_flight",
            "shape_params",
            "n_positions",
            "only_point_north",
            "use_noisy_spiral",
            "home_position",
            "number_cameras",
            "crop",
            "crop_box_params",
        },
    )
    apply_entry_overrides(
        entry,
        "config_training",
        training,
        names,
        {
            "epochs",
            "learning_rate",
            "batch_size",
            "batch_size_val",
            "batch_size_eval",
            "split_ratio",
            "model_type",
            "augment",
        },
    )
    train_fn = function_node(parse_source(project, "train.py"), "train_model")
    calls = {
        n.targets[0].id: n.value
        for n in train_fn.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
    }
    eval_fn = function_node(parse_source(project, "virtual_homing_ANN.py"), "main")
    eval_constants = literal_assignments(eval_fn.body)
    step_expressions = [
        n.value
        for n in ast.walk(eval_fn)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "step_size" for t in n.targets)
        and isinstance(n.value, ast.BinOp)
    ]
    if len(step_expressions) != 1:
        raise ValueError("Expected exactly one effective dynamic step expression")
    loader_fn = function_node(parse_source(project, "data_loader.py"), "make_loader")
    loader_call = next(n.value for n in loader_fn.body if isinstance(n, ast.Return))
    workers = next(
        ast.literal_eval(k.value) for k in loader_call.keywords if k.arg == "num_workers"
    )
    image_size = render.get("image_size", render["crop_box_params"][2])
    effective = {
        "entrypoint": {
            k: settings[k] for k in ("mode", "all_environments", "seed_number", "train_models")
        },
        "sampling": {
            "shape_flight": render["shape_flight"],
            "shape_params": render["shape_params"],
            "n_positions": render["n_positions"],
            "use_noisy_spiral": render["use_noisy_spiral"],
            "only_point_north": render["only_point_north"],
            "home_position": render["home_position"],
            "number_cameras": render["number_cameras"],
            "projection": projection_value(
                parse_source(project, "render_dataset.py"), "projection_type"
            ),
            "raw_resolution_wh": [image_size, image_size],
            "crop": render["crop"],
            "crop_box_params": render["crop_box_params"],
        },
        "training": {
            "model_type": training["model_type"],
            "criterion": calls["criterion"].func.attr,
            "optimizer": calls["optimizer"].func.attr,
            **{k: training[k] for k in ("learning_rate", "batch_size", "epochs", "split_ratio")},
            "batch_size_val": training["batch_size_val"],
            "batch_size_eval": training["batch_size_eval"],
            "augment": training["augment"],
            "num_workers": workers,
        },
        "evaluation": {
            "virtual_home_point": homing["virtual_home_point"],
            "start_location_pattern": homing["start_location_pattern"],
            "circle_params": homing["circle_params"],
            "count_limit": homing["params"]["count_limit"],
            "threshold_m": homing["params"]["threshold"],
            "configured_step_size_m": homing["params"]["step_size"],
            "effective_step_expression": ast.dump(step_expressions[0]),
            "max_step_m": eval_constants["max_step"],
            **{k: homing[k] for k in ("pitch_roll", "north", "fixed_yaw")},
            "projection": projection_value(eval_fn),
        },
    }
    return effective


def differences(expected, actual, path="legacy_reference"):
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                result.append(path + "." + key + ": missing or unexpected key")
            else:
                result.extend(differences(expected[key], actual[key], path + "." + key))
        return result
    if expected != actual:
        return ["{}: expected {!r}, got {!r}".format(path, expected, actual)]
    return []


def check_protocol(project, protocol):
    expected = json.loads(json.dumps(protocol["legacy_reference"]))
    expression = expected["evaluation"]["effective_step_expression"]
    expected["evaluation"]["effective_step_expression"] = ast.dump(
        ast.parse(expression, mode="eval").body
    )
    actual = inspect_legacy(project)
    errors = differences(expected, actual)
    if protocol["target_experiment"]["execution_ready"] is not False:
        errors.append("Stage A protocol must not advertise an implemented downward experiment")
    if errors:
        raise ValueError("Baseline drift detected:\n" + "\n".join(errors))
    return actual


def write_new_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def runtime_check(project, protocol, output, render, capture_input=None, publish=None):
    """Exercise original helpers with fresh weights; never perform an optimizer step."""
    app = None
    if render:
        from omni.isaac.kit import SimulationApp

        app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    try:
        import importlib.metadata
        import random
        import numpy as np
        import torch

        sys.path.insert(0, str(project))
        from simple_network import IncepAttentionCNN_rgb
        from insect_utils.flight_path_functions import calculate_target_vectors, generate_bee_path

        seed = protocol["legacy_reference"]["entrypoint"]["seed_number"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(2)
        model = IncepAttentionCNN_rgb().eval()
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if count != protocol["network_contract"]["trainable_parameters"]:
            raise ValueError("Network parameter count changed: {}".format(count))
        reference = protocol["legacy_reference"]
        with torch.no_grad():
            prediction = model(torch.zeros(1, 3, 400, 1024))
        if tuple(prediction.shape) != (1, 2) or not torch.isfinite(prediction).all():
            raise ValueError("Invalid network forward result")
        optimizer = torch.optim.AdamW(model.parameters(), lr=reference["training"]["learning_rate"])
        path = generate_bee_path(
            *reference["sampling"]["shape_params"], reference["sampling"]["home_position"]
        )
        n = len(path)
        train_size = int(reference["training"]["split_ratio"] * n)
        val_size = int((1 - reference["training"]["split_ratio"]) * train_size)
        report = {
            "python": sys.version.split()[0],
            "packages": {
                n: importlib.metadata.version(n)
                for n in ("torch", "torchvision", "numpy", "scipy", "scikit-image", "onnx")
            },
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "network": {
                "trainable_parameters": count,
                "output_shape": list(prediction.shape),
                "optimizer_defaults": optimizer.defaults,
                "weights": "fresh_seed_10_untrained",
            },
            "learning_path": {
                "planned_positions": n,
                "max_radius_m": float(np.max(np.linalg.norm(np.asarray(path)[:, :2], axis=1))),
                "legacy_four_camera_frames": (n // 4) * 4,
                "split_if_all_positions_captured": {
                    "train": train_size - val_size,
                    "validation": val_size,
                    "test": n - train_size,
                },
            },
            "training_steps": 0,
            "renderer_exercised": render,
            "verified_existing_captures": capture_input is not None,
            "captures": [],
        }
        version = Path("/isaac-sim/VERSION")
        report["isaac_sim_version_file"] = (
            version.read_text().strip() if version.is_file() else None
        )
        if torch.cuda.is_available():
            report["cuda_operation"] = bool((torch.ones(2, device="cuda") + 1).sum().item() == 4)
        if render or capture_input is not None:
            from PIL import Image
            from torchvision.transforms import functional as TF

            plan = protocol["validation_plan"]["panorama_smoke"]
            scene = project / plan["scene"]
            if not scene.is_file():
                raise ValueError("Smoke scene missing: {}".format(scene))
            report["scene"] = plan["scene"]
            if render:
                import omni
                from omni.isaac.core import World
                from omni.isaac.core.utils.rotations import euler_angles_to_quat
                from omni.isaac.sensor import Camera
                from virtual_homing_ANN import get_image

                omni.usd.get_context().open_stage(str(scene))
                world = World(stage_units_in_meters=1.0)
                camera = Camera(prim_path="/World/Camera", name="camera1", resolution=(1024, 1024))
                camera.set_projection_type(reference["sampling"]["projection"])
                world.reset()
                camera.initialize()
                for _ in range(8):
                    world.step(render=True)
            else:
                report["capture_input"] = str(capture_input)
            previous = None
            for index, (x, y, yaw) in enumerate(plan["poses_xy_yaw_deg"]):
                position = [x, y, reference["sampling"]["home_position"][2]]
                stem = "panorama_{:02d}".format(index)
                if render:
                    orientation = euler_angles_to_quat([0.0, 0.0, yaw], degrees=True)
                    get_image(orientation, position, camera, world)
                    raw, tensor = get_image(orientation, position, camera, world)
                else:
                    with Image.open(capture_input / (stem + "_raw.png")) as image:
                        raw = np.array(image.convert("RGB"))
                    crop = Image.fromarray(raw).crop(reference["sampling"]["crop_box_params"])
                    with Image.open(capture_input / (stem + "_crop.png")) as image:
                        if not np.array_equal(np.array(crop), np.array(image.convert("RGB"))):
                            raise ValueError("Saved crop does not match original preprocessing")
                    tensor = TF.to_tensor(crop).unsqueeze(0)
                if raw.shape != (1024, 1024, 3) or tuple(tensor.shape) != (1, 3, 400, 1024):
                    raise ValueError("Unexpected panorama dimensions")
                std = float(tensor.std().item())
                if std < 0.005:
                    raise ValueError("Rendered strip has no usable image variation")
                target = calculate_target_vectors(
                    [position], [[0.0, 0.0, yaw]], reference["sampling"]["home_position"]
                )[0]
                angle = np.deg2rad(yaw)
                rotation = np.array(
                    [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                )
                reconstructed = np.asarray(position[:2]) + rotation @ target
                if not np.allclose(reconstructed, reference["sampling"]["home_position"][:2]):
                    raise ValueError("Body-frame label does not reconstruct the configured home")
                with torch.no_grad():
                    predicted = model(tensor)
                if not torch.isfinite(predicted).all():
                    raise ValueError("Non-finite prediction on rendered frame")
                if render:
                    Image.fromarray(raw).save(output / (stem + "_raw.png"))
                    Image.fromarray(raw).crop(reference["sampling"]["crop_box_params"]).save(
                        output / (stem + "_crop.png")
                    )
                delta = (
                    float(np.mean(np.abs(raw.astype(float) - previous)))
                    if previous is not None
                    else None
                )
                if delta is not None and delta < 0.01:
                    raise ValueError("Camera motion produced an unchanged frame")
                report["captures"].append(
                    {
                        "position": position,
                        "yaw_deg": yaw,
                        "target_body_xy_m": target.tolist(),
                        "strip_std": std,
                        "mean_raw_delta_from_previous": delta,
                        "raw_image": stem + "_raw.png",
                        "crop_image": stem + "_crop.png",
                        "untrained_prediction": predicted.tolist(),
                    }
                )
                previous = raw.astype(float)
                action = "Captured" if render else "Verified"
                print(
                    "{} panorama {} / {}".format(action, index + 1, len(plan["poses_xy_yaw_deg"])),
                    flush=True,
                )
        # Kit shutdown can terminate the process. Publish measurements before cleanup.
        if publish is not None:
            publish(report)
        return report
    finally:
        if app is not None:
            app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--runtime", action="store_true", help="Check installed Isaac Python and original network"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Also capture exactly three original panoramas; no training",
    )
    parser.add_argument(
        "--verify-captures",
        type=Path,
        help="Verify existing smoke PNGs without starting Kit or replacing images",
    )
    args = parser.parse_args()
    if args.render and args.verify_captures is not None:
        parser.error("--render and --verify-captures are mutually exclusive")
    runtime = args.runtime or args.render or args.verify_captures is not None
    if runtime and args.output is None:
        parser.error("Runtime checks require --output")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    effective = check_protocol(PROJECT, protocol)
    print("PASS: legacy effective settings match stage01_protocol.json", flush=True)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        name = "runtime_report.json" if runtime else "static_report.json"
        destination = args.output / name
        if destination.exists():
            raise FileExistsError("Refusing to replace existing report: {}".format(destination))
        report = {
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
            "effective_legacy_settings": effective,
        }
        if runtime:
            # Reserve the report path before any expensive rendering or file output.
            with destination.open("x", encoding="utf-8") as stream:
                stream.write('{"status": "in_progress"}\n')

            def publish(measurements):
                report["runtime"] = measurements
                report["status"] = "passed"
                destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print("Report: {}".format(destination), flush=True)

            try:
                runtime_check(
                    PROJECT, protocol, args.output, args.render, args.verify_captures, publish
                )
            except Exception as error:
                destination.write_text(
                    json.dumps({"status": "failed", "error": str(error)}, indent=2) + "\n"
                )
                raise
        else:
            report["status"] = "passed"
            write_new_json(destination, report)
            write_new_json(args.output / "protocol_snapshot.json", protocol)
        print("Report: {}".format(destination), flush=True)


if __name__ == "__main__":
    main()
