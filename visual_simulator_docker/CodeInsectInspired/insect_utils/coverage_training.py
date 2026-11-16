"""Paired training, validation-only selection and physical-unit deployment checks."""

import copy
from pathlib import Path
import time

import numpy as np
import torch

from insect_utils.attention_model import create_attention
from insect_utils.coverage_data import load_data, make_loader
from insect_utils.coverage_model import PhysicalHomeVector
from insect_utils.coverage_plan import radial_bin, verify_freeze
from insect_utils.downward_training import atomic_torch, runtime_signature, seed_all, train_epochs
from insect_utils.experiment_config import PROJECT, canonical_hash, digest
from insect_utils.experiment_state import atomic_json, now, read_json
from insect_utils.training_state import checkpoint_pointer, vector_metrics


def new_model(config, device):
    seed_all(config["seed"], cuda=device.type == "cuda")
    model = create_attention(config).to(device)
    if sum(p.numel() for p in model.parameters()) != 10628:
        raise ValueError("Network architecture changed")
    return model


def initialize_runtime(config):
    if not torch.cuda.is_available():
        raise RuntimeError("Formal paired experiment requires the configured GPU")
    torch.set_num_threads(config["torch_num_threads"])
    return torch.device("cuda")


def select_smoke_records(records, edges):
    selected = set()
    for bin_index in range(4):
        groups = sorted(
            {
                r["position_id"]
                for r in records
                if r["split"] == "train" and radial_bin(r["x"], r["y"], edges) == bin_index
            }
        )
        selected.update(groups[:4])
    return [r for r in records if r["split"] == "validation" or r["position_id"] in selected]


def train_arm(folder, arm, smoke=False):
    folder = Path(folder)
    freeze = verify_freeze(folder)
    protocol = freeze["protocol"]
    config = dict(protocol["training"])
    scale = protocol["arms"][arm]
    records = load_data(folder)
    if smoke:
        records = select_smoke_records(records, protocol["radial_edges_m"])
        config["epochs"] = 3
    destination = folder / ("smoke" if smoke else "arms") / arm
    destination.mkdir(parents=True, exist_ok=True)
    fingerprint = canonical_hash(
        {
            "experiment": freeze["fingerprint"],
            "arm": arm,
            "scale": scale,
            "smoke": smoke,
            "config": config,
        }
    )
    if (destination / "completion.json").exists():
        result = read_json(destination / "completion.json")
        if result["fingerprint"] != fingerprint:
            raise ValueError("Training completion belongs to another experiment")
        for name, expected in result["files"].items():
            if digest(destination / name) != expected:
                raise ValueError("Completed training artifact changed")
        return result
    device = initialize_runtime(config)
    train_loader = make_loader(records, "train", scale, config["num_workers"], shuffle=True)
    val_loader = make_loader(records, "validation", scale, config["num_workers"])
    model = new_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], **config["optimizer_kwargs"]
    )
    from insect_utils.train_utils import parameter_digest

    initial = parameter_digest(model)
    atomic_json(
        destination / "contract.json",
        {
            "fingerprint": fingerprint,
            "arm": arm,
            "scale_m": scale,
            "network_weight_output_units": "meters" if scale == 1 else "normalized (meters/44)",
            "deployment_output_units": "body-frame home vector in meters",
            "engine_loss_conversion": "physical component MSE(m2)=engine MSE*scale_m**2",
            "initial_parameter_sha256": initial,
            "runtime": runtime_signature(device),
            "configuration": config,
            "train_images": len(train_loader.dataset),
            "validation_images": len(val_loader.dataset),
            "test_data_opened": False,
            "formal_initialization_from_smoke": False,
        },
    )
    engine = destination / "engine_scaled_units"
    resume = (engine / "checkpoint_latest.json").exists()
    counter = [read_json(engine / "checkpoint_latest.json")["optimizer_steps"] if resume else 0]
    matched = destination / "matched_19950.pth"
    matched_steps = protocol["matched_optimizer_steps"]
    if not smoke and counter[0] >= matched_steps and not matched.exists():
        raise ValueError("Matched-budget snapshot missing from a resumed experiment")

    def record_matched_snapshot(_optimizer, _args, _kwargs):
        counter[0] += 1
        if not smoke and counter[0] == matched_steps:
            atomic_torch(matched, model.state_dict())
            atomic_json(
                destination / "matched_19950.json",
                {
                    "optimizer_steps": matched_steps,
                    "sha256": digest(matched),
                    "scale_m": scale,
                    "selection": "predeclared",
                },
            )

    hook = optimizer.register_step_post_hook(record_matched_snapshot)
    try:
        result = train_epochs(
            model, optimizer, train_loader, val_loader, config, engine, fingerprint, resume=resume
        )
    finally:
        hook.remove()
    if result["status"] != "completed":
        raise RuntimeError("Training did not reach the configured epoch budget")
    checkpoint_file, _ = checkpoint_pointer(engine, fingerprint)
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    physical_history = []
    for row in checkpoint["history"]:
        physical = dict(row)
        for key in ("train_component_mse_m2", "validation_component_mse_m2"):
            physical[key] *= scale ** 2
        physical_history.append(physical)
    best = min(physical_history, key=lambda r: (r["validation_component_mse_m2"], r["epoch"]))
    best_files = list((engine / "checkpoints").glob("epoch_{:04d}_*.pt".format(best["epoch"])))
    eligible = []
    for file in best_files:
        candidate = torch.load(file, map_location="cpu", weights_only=True)
        if (
            candidate["fingerprint"] == fingerprint
            and candidate["history"] == checkpoint["history"][: best["epoch"]]
        ):
            eligible.append(candidate["model"])
    if not eligible:
        raise ValueError("Validation-selected checkpoint is missing")
    if any(any(not torch.equal(state[k], eligible[0][k]) for k in state) for state in eligible[1:]):
        raise ValueError("Ambiguous epoch checkpoint payload")
    atomic_torch(destination / "final.pth", checkpoint["model"])
    atomic_torch(destination / "best_validation.pth", eligible[0])
    atomic_json(
        destination / "physical_history.json",
        {"epochs": physical_history, "unit": "component MSE in m2", "scale_m": scale},
    )
    summary = {
        "status": "completed",
        "updated": now(),
        "fingerprint": fingerprint,
        "arm": arm,
        "scale_m": scale,
        "epochs": config["epochs"],
        "optimizer_steps": result["optimizer_steps"],
        "initial_parameter_sha256": initial,
        "best_validation": best,
        "final_validation": physical_history[-1],
        "test_data_opened": False,
        "resources": result["resources"],
    }
    names = ["final.pth", "best_validation.pth", "physical_history.json", "contract.json"]
    if not smoke:
        names += ["matched_19950.pth", "matched_19950.json"]
    summary["files"] = {name: digest(destination / name) for name in names}
    atomic_json(destination / "completion.json", summary)
    return summary


def freeze_selection(folder):
    folder = Path(folder)
    freeze = verify_freeze(folder)
    arms = {
        name: read_json(folder / "arms" / name / "completion.json")
        for name in freeze["protocol"]["arms"]
    }
    if any(r["status"] != "completed" or r["test_data_opened"] for r in arms.values()):
        raise ValueError("Both arms must finish before opening test partitions")
    if len({r["initial_parameter_sha256"] for r in arms.values()}) != 1:
        raise ValueError("Controlled arms did not share identical initialization")
    selected = min(
        arms, key=lambda name: (arms[name]["best_validation"]["validation_component_mse_m2"], name)
    )
    value = {
        "fingerprint": freeze["fingerprint"],
        "arms": arms,
        "selected_arm": selected,
        "selection_criterion": "minimum balanced full-validation physical component MSE",
        "selection_data": "validation only; frozen before any test inference",
    }
    file = folder / "selection_frozen_before_test.json"
    if file.exists() and read_json(file) != value:
        raise ValueError("Previously frozen validation selection changed")
    atomic_json(file, value)
    return value


def infer(model, records, split):
    loader = make_loader(records, split, 1, workers=4, allow_test=True)
    predictions, targets = [], []
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for images, values in loader:
            predictions.extend(model(images.to(device)).cpu().tolist())
            targets.extend(values.tolist())
    return loader.dataset.records, predictions, targets


def metrics_by_radius(rows, predictions, targets, edges):
    result = {"overall": vector_metrics(predictions, targets), "radial_bins": {}}
    for index in range(4):
        selected = [i for i, r in enumerate(rows) if radial_bin(r["x"], r["y"], edges) == index]
        label = "{}-{}m".format(edges[index], edges[index + 1])
        result["radial_bins"][label] = (
            vector_metrics([predictions[i] for i in selected], [targets[i] for i in selected])
            if selected
            else None
        )
    if all(result["radial_bins"].values()):
        weights = [(b * b - a * a) / edges[-1] ** 2 for a, b in zip(edges, edges[1:])]
        result["area_weighted_mean_vector_error_m"] = sum(
            weight * value["vector_error_m"]["mean"]
            for weight, value in zip(weights, result["radial_bins"].values())
        )
    return result


def export_physical(model, example, destination):
    import onnx
    from onnx.reference import ReferenceEvaluator

    destination.mkdir(parents=True, exist_ok=True)
    model = copy.deepcopy(model).cpu().eval()
    example = example[:2].cpu()
    with torch.inference_mode():
        expected = model(example).numpy()
    scripted = torch.jit.trace(model, example[:1], check_inputs=[(example,)])
    script_path = destination / "home_vector_m.torchscript.pt"
    scripted.save(str(script_path))
    reloaded = torch.jit.load(str(script_path))
    with torch.inference_mode():
        actual_script = reloaded(example).numpy()
    if not np.allclose(actual_script, expected, atol=1e-5, rtol=1e-4):
        raise ValueError("TorchScript physical-unit output mismatch")
    onnx_path = destination / "home_vector_m.onnx"
    torch.onnx.export(
        model,
        example[:1],
        onnx_path,
        opset_version=17,
        input_names=["rgb"],
        output_names=["home_vector_body_m"],
        dynamic_axes={"rgb": {0: "batch"}, "home_vector_body_m": {0: "batch"}},
    )
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph, full_check=True)
    actual = ReferenceEvaluator(graph).run(None, {"rgb": example.numpy()})[0]
    if not np.allclose(actual, expected, atol=1e-5, rtol=1e-4):
        raise ValueError("ONNX physical-unit output mismatch")
    result = {
        "status": "passed",
        "input": "float32 NCHW full RGB/255, [N,3,1024,1024]",
        "network_resize_hw": [384, 384],
        "output": "body-frame home vector in meters",
        "samples": len(example),
        "onnx_max_abs_difference_m": float(abs(actual - expected).max()),
        "torchscript_max_abs_difference_m": float(abs(actual_script - expected).max()),
        "atol": 1e-5,
        "rtol": 1e-4,
        "onnx_sha256": digest(onnx_path),
        "torchscript_sha256": digest(script_path),
    }
    atomic_json(destination / "export_check.json", result)
    return result


def evaluate(folder):
    folder = Path(folder)
    freeze = verify_freeze(folder)
    protocol = freeze["protocol"]
    selection = freeze_selection(folder)
    records = load_data(folder)
    device = initialize_runtime(protocol["training"])
    results = {}
    started = time.monotonic()
    for arm, scale in protocol["arms"].items():
        destination = folder / "arms" / arm
        for checkpoint in ("best_validation", "final", "matched_19950"):
            name = arm + "/" + checkpoint
            model = new_model(protocol["training"], device)
            model.load_state_dict(
                torch.load(
                    destination / (checkpoint + ".pth"), map_location="cpu", weights_only=True
                )
            )
            physical = PhysicalHomeVector(model, scale).to(device).eval()
            results[name] = {}
            for split in ("validation", "legacy_test", "new_test"):
                rows, predictions, targets = infer(physical, records, split)
                results[name][split] = metrics_by_radius(
                    rows, predictions, targets, protocol["radial_edges_m"]
                )
                atomic_json(
                    folder / "evaluation" / arm / checkpoint / (split + ".json"),
                    {
                        "rows": [
                            {
                                "position_id": r["position_id"],
                                "heading": r["heading_index"],
                                "prediction_m": p,
                                "target_m": t,
                            }
                            for r, p, t in zip(rows, predictions, targets)
                        ],
                        "metrics": results[name][split],
                    },
                )
            if checkpoint == "best_validation":
                example = next(iter(make_loader(records, "validation", workers=0)))[0][:2]
                export_physical(physical, example, destination / "deployment")
    legacy = new_model(protocol["training"], device)
    legacy.load_state_dict(
        torch.load(PROJECT / protocol["legacy_weights"], map_location="cpu", weights_only=True)
    )
    results["stage10/final"] = {}
    for split in ("legacy_test", "new_test"):
        rows, predictions, targets = infer(legacy, records, split)
        results["stage10/final"][split] = metrics_by_radius(
            rows, predictions, targets, protocol["radial_edges_m"]
        )
    output = {
        "fingerprint": freeze["fingerprint"],
        "selection": selection["selected_arm"],
        "results": results,
        "elapsed_s": time.monotonic() - started,
        "updated": now(),
        "limits": protocol["limits"],
    }
    atomic_json(folder / "evaluation/results.json", output)
    write_report(folder, output)
    verify_freeze(folder)
    atomic_json(
        folder / "completion.json",
        {
            "status": "completed",
            "updated": now(),
            "fingerprint": freeze["fingerprint"],
            "results_sha256": digest(folder / "evaluation/results.json"),
            "report": "result_summary.md",
        },
    )


def write_report(folder, output):
    lines = [
        "# Stage11 覆盖增密与目标归一化对照",
        "",
        "两个训练组均为600位置方案中的512个训练位置，384×384原网络，150轮；" "按完整验证集的米制MSE选权重，两个组的选择在测试推理前冻结。",
        "",
        "| 模型 | 原测试平均向量误差(m) | 新测试平均向量误差(m) | 新测试角度误差(°) |",
        "|---|---:|---:|---:|",
    ]
    for name, result in output["results"].items():
        old = result["legacy_test"]["overall"]
        new = result["new_test"]["overall"]
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} |".format(
                name,
                old["vector_error_m"]["mean"],
                new["vector_error_m"]["mean"],
                new["absolute_shortest_angle_error_deg"]["mean"],
            )
        )
    lines += [
        "",
        "验证集选择的组：`{}`。".format(output["selection"]),
        "",
        "150轮的新训练组各有76,800次优化器更新，上一轮为19,950次。"
        "表中matched_19950用于相同更新预算的补充比较；final与best_validation分开报告。",
        "",
        "旧测试集已用于开发诊断；新测试集仅在两组训练与选择完成后打开。" "新测试按距离等量取样，分距离指标与面积加权误差见evaluation/results.json。",
        "",
        "同场景内相邻视野可能重叠。本结果不代表跨场景表现或闭环归巢成功率。",
        "",
    ]
    (folder / "result_summary.md").write_text("\n".join(lines), encoding="utf-8")
