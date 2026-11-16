"""Prepare and accept grouped downward data; no simulation or optimizer steps."""

import argparse
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.data_preparation import CORE_SOURCES, DATA_PROTOCOL, RECORD_FIELDS
from insect_utils.data_preparation import grouped_split, load_prepared, validate_records
from insect_utils.data_preparation import verify_capture_source, project_path
from insect_utils.experiment_config import canonical_hash, digest
from insect_utils.experiment_state import atomic_json, read_json


DEFAULT_SOURCE = (
    PROJECT
    / "runs/stage05_capture/eight_heading_validation_01/runs"
    / "downward_h10_hfov30_landmarks/four_cameras"
)


def prepare(source, output):
    import torch

    reference, stored, capture = verify_capture_source(source, full=True)
    if (
        capture["position_count"] != 147
        or capture["headings_per_position"] != 8
        or capture["camera_count"] != 4
    ):
        raise ValueError("Expected the accepted four-camera,147-position,eight-heading dataset")
    original = [json.loads(s) for s in (source / "capture/samples.jsonl").read_text().splitlines()]
    records = [{field: r[field] for field in RECORD_FIELDS} for r in original]
    groups = validate_records(records, stored["spec"]["home"]["ground_xy_m"], 8)
    protocol = read_json(PROJECT / DATA_PROTOCOL)
    seed = protocol["split"]["seed"]
    ratio = stored["spec"]["training"]["split_ratio"]
    if ratio != protocol["split"]["ratio"]:
        raise ValueError("Capture/training split ratio differs from approval")
    permutation = torch.randperm(
        len(groups), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    split = grouped_split(records, permutation, ratio)
    split["generator"] = protocol["split"]["generator"]
    if {name: len(ids) for name, ids in split["groups"].items()} != protocol["split"][
        "position_counts"
    ]:
        raise ValueError("Position split counts differ from the approved protocol")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "source_reference.json", reference)
    atomic_json(output / "records.json", records)
    atomic_json(output / "splits.json", split)
    source_capture = str((source / "capture").relative_to(PROJECT))
    models = Path("runs/stage07_training") / output.name
    config = dict(
        stored["spec"]["training"],
        camera_mode="downward_pinhole",
        seed=seed,
        prepared_data_dir=str(output.relative_to(PROJECT)),
        dataset_folder=source_capture,
        image_folder=source_capture + "/Replicator/rgb",
        csv_filename="dataset_navigation.csv",
        evaluation_dataset_folder=source_capture,
        evaluation_image_folder=source_capture + "/Replicator/rgb",
        evaluation_csv_filename="dataset_navigation.csv",
        models_folder=str(models),
        model_name="model_attention",
        model_suffix="_attention",
        training_log_path=str(models / "training_history.json"),
    )
    atomic_json(output / "data_config.json", config)
    names = ["source_reference.json", "records.json", "splits.json", "data_config.json"]
    manifest = {
        "schema_version": 1,
        "status": "prepared",
        "source_capture_fingerprint": reference["fingerprint"],
        "source_capture_manifest_sha256": reference["capture_manifest_sha256"],
        "data_protocol_sha256": digest(PROJECT / DATA_PROTOCOL),
        "files_sha256": {name: digest(output / name) for name in names},
        "core_source_sha256": {name: digest(PROJECT / name) for name in CORE_SOURCES},
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__},
        "group_counts": {name: len(ids) for name, ids in split["groups"].items()},
        "image_counts": {name: len(ids) for name, ids in split["indices"].items()},
        "augmentation": False,
        "training_steps": 0,
        "interpretation": protocol["split"]["interpretation"],
    }
    manifest["fingerprint"] = canonical_hash(manifest)
    atomic_json(output / "preparation_manifest.json", manifest)
    return config


def check_runtime(output, config):
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from PIL import Image
    from torchvision.transforms import functional as vision
    from unittest.mock import patch
    import data_loader
    import virtual_homing_ANN
    from insect_utils.data_diagnostics import (
        plot_network_resize,
        plot_split_map,
        plot_world_vectors,
    )
    from insect_utils.downward_camera import DownwardCamera
    from insect_utils.train_utils import evaluate_on_dataset, parameter_digest
    from simple_network import IncepAttentionCNN_rgb

    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    cfg = dict(config, num_workers=0, diagnostic_dir=str(output / "inspection"))
    context = load_prepared(output)
    order = torch.randperm(
        len(context["split"]["group_permutation"]),
        generator=torch.Generator().manual_seed(cfg["seed"]),
    ).tolist()
    repeated = grouped_split(context["records"], order, cfg["split_ratio"])
    if any(repeated[key] != context["split"][key] for key in ["groups", "indices"]):
        raise ValueError("Isolated seed did not reproduce the saved split")
    loaders = data_loader.get_data_loaders(cfg)[:3]
    assert len({id(loader.dataset) for loader in loaders}) == 3
    assert all(not loader.dataset.augmentation for loader in loaders)
    checked = 0
    for loader in loaders:
        dataset = loader.dataset
        for index in range(len(dataset)):
            image, target = dataset[index]
            row = dataset.record(index)
            with Image.open(context["capture_folder"] / row["image"]) as raw:
                array = np.asarray(raw).copy()
                old_tensor = vision.to_tensor(raw)
            rgba = np.concatenate([array, np.zeros((*array.shape[:2], 1), dtype=np.uint8)], axis=2)
            online = virtual_homing_ANN.downward_rgb_tensor(rgba)[0]
            if image.dtype != torch.float32 or list(image.shape) != [3, 1024, 1024]:
                raise ValueError("Wrong RGB tensor contract")
            if not torch.equal(image, online) or not torch.equal(image, old_tensor):
                raise ValueError("Disk/online/ToTensor pixel mismatch")
            if not torch.equal(
                target, torch.tensor([row["target_x"], row["target_y"]], dtype=torch.float32)
            ):
                raise ValueError("Loader label changed")
            checked += 1
        print("STAGE06_PIXELS " + dataset.split + " " + str(len(dataset)), flush=True)
    if checked != 1176:
        raise ValueError("Not all captured images were checked")

    # Exercise the actual get_image branch with a mock camera returning captured RGB plus alpha.
    first = loaders[2].dataset
    row = first.record(0)
    image, _ = first[0]
    with Image.open(context["capture_folder"] / row["image"]) as raw:
        array = np.asarray(raw).copy()
    rgba = np.concatenate([array, np.full((*array.shape[:2], 1), 255, dtype=np.uint8)], axis=2)

    class CameraProbe(DownwardCamera):
        def __init__(self):
            self.last_frame = 0

        def capture(self, *_):
            self.last_frame += 1
            return rgba, {}

    angle = math.radians(row["yaw"]) / 2
    _, online = virtual_homing_ANN.get_image(
        [math.cos(angle), 0, 0, math.sin(angle)],
        [row["x"], row["y"], row["z"]],
        CameraProbe(),
        None,
    )
    if not torch.equal(image, online[0]):
        raise ValueError("get_image bypassed the shared preprocessing")

    model = IncepAttentionCNN_rgb().eval()
    initial = parameter_digest(model)
    if sum(p.numel() for p in model.parameters()) != 10820:
        raise ValueError("Original network changed")
    images, _ = next(iter(loaders[0]))
    observed = {}

    def check_resize(_, args):
        expected = functional.interpolate(
            images, size=(192, 1800), mode="bilinear", align_corners=False
        )
        observed["shape"] = list(args[0].shape)
        observed["matches_original_bilinear"] = torch.equal(args[0], expected)

    hook = model.inception1.register_forward_pre_hook(check_resize)
    with torch.inference_mode():
        prediction = model(images)
    hook.remove()
    if not observed["matches_original_bilinear"] or observed["shape"][1:] != [3, 192, 1800]:
        raise ValueError("Network resize contract changed")
    if list(prediction.shape) != [8, 2] or not torch.isfinite(prediction).all():
        raise ValueError("Original forward failed")

    repeat = {}
    with patch(
        "data_loader.augment_image_vector",
        side_effect=AssertionError("Old panorama augmentation called"),
    ), patch(
        "insect_utils.train_utils.augment_image_vector",
        side_effect=AssertionError("Old panorama augmentation called"),
    ):
        for name in ["validation", "test"]:
            args = (model, cfg["dataset_folder"], cfg["image_folder"], cfg["csv_filename"])
            a = evaluate_on_dataset(*args, config=cfg, evaluation_set=name)
            b = evaluate_on_dataset(*args, config=cfg, evaluation_set=name)
            repeat[name] = bool(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
            if not repeat[name]:
                raise ValueError("Validation/test predictions changed between repeats")
        data_loader.inspect_dataset(cfg, n_examples=4)
        try:
            evaluate_on_dataset(*args, config=cfg, augment=True)
        except ValueError:
            pass
        else:
            raise ValueError("Panorama augmentation was not rejected")
    alt = data_loader.get_data_loaders(dict(cfg, num_workers=2))[1]
    alternate = iter(alt)
    batch = next(alternate)
    del alternate
    reference = next(iter(loaders[1]))
    if not torch.equal(batch[0], reference[0]) or not torch.equal(batch[1], reference[1]):
        raise ValueError("Worker count changed validation data")
    if parameter_digest(model) != initial:
        raise ValueError("Data acceptance updated model weights")
    diagnostics = output / "inspection"
    plot_split_map(context["records"], context["split"], diagnostics / "split_map.png")
    positions = {}
    for record in context["records"]:
        positions.setdefault(record["position_id"], record)
    plot_world_vectors(
        list(positions.values()),
        context["capture_config"]["spec"]["home"]["ground_xy_m"],
        diagnostics / "world_home_vectors.png",
    )
    plot_network_resize(images[0], diagnostics / "network_resize.png")
    neighbors = {}
    train = [positions[g] for g in context["split"]["groups"]["train"]]
    for name in ["validation", "test"]:
        neighbors[name] = min(
            math.hypot(positions[g]["x"] - r["x"], positions[g]["y"] - r["y"])
            for g in context["split"]["groups"][name]
            for r in train
        )
    if any(name == "omni" or name.startswith("omni.") for name in sys.modules):
        raise ValueError("Data preparation unexpectedly imported Kit")
    steps = len(loaders[0])
    return {
        "status": "passed",
        "checked_images": checked,
        "group_counts": context["manifest"]["group_counts"],
        "image_counts": context["manifest"]["image_counts"],
        "same_position_leakage": 0,
        "nearest_train_position_distance_m": neighbors,
        "spatial_buffer_m": 0.0,
        "validation_test_repeat_exact": repeat,
        "workers0_vs2_equal": True,
        "offline_online_rgba_totensor_equal": True,
        "get_image_mock_camera_equal": True,
        "network_parameters": 10820,
        "network_internal_resize": observed,
        "network_output_shape": [8, 2],
        "initial_parameter_sha256": initial,
        "final_parameter_sha256": parameter_digest(model),
        "training_steps": 0,
        "optimizer_steps_per_epoch_if_trained": steps,
        "optimizer_steps_for_configured_epochs_if_trained": steps * config["epochs"],
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "kit_modules": [],
        "model_weights": "fresh untrained seed10; forward checks only",
        "source_capture_unchanged": True,
        "preparation_fingerprint": context["manifest"]["fingerprint"],
        "diagnostics_sha256": {
            str(p.relative_to(output)): digest(p) for p in diagnostics.glob("*.png")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = project_path(args.output).resolve()
    if args.verify_only:
        context = load_prepared(output)
        report = read_json(output / "stage06_report.json")
        if (
            report["status"] != "passed"
            or report["preparation_fingerprint"] != context["manifest"]["fingerprint"]
        ):
            raise ValueError("Stage F runtime acceptance is not complete")
        verify_capture_source(context["source"]["run"], full=True)
        for name, expected in report["diagnostics_sha256"].items():
            if digest(output / name) != expected:
                raise ValueError("Changed diagnostic figure")
        print("STAGE06_VERIFIED " + str(output))
        return
    config = prepare(project_path(args.source_run).resolve(), output)
    atomic_json(output / "stage06_report.json", {"status": "running"})
    try:
        report = check_runtime(output, config)
        # Recheck immutable input lineage and prepared files after all reads.
        load_prepared(output)
        atomic_json(output / "stage06_report.json", report)
        print("STAGE06_DATA_PASS " + str(output), flush=True)
    except BaseException as error:
        atomic_json(output / "stage06_report.json", {"status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
