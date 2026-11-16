"""Read-only diagnostics of the frozen stage G run; never train or replace weights."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
RUN = PROJECT / "runs/stage07_training/attention_8h_01"
PREPARED = PROJECT / "runs/stage06_data/position_groups_01"


def read(path):
    return json.loads(path.read_text())


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def candidates(row, landmarks):
    """Conservative ground-frustum/bounding-circle overlap, not a visibility mask."""
    half = row["agl_m"] * math.tan(math.radians(15))
    yaw = math.radians(row["yaw"])
    count = 0
    for item in landmarks:
        dx = item["center_xyz_m"][0] - row["x"]
        dy = item["center_xyz_m"][1] - row["y"]
        forward = math.cos(yaw) * dx + math.sin(yaw) * dy
        left = -math.sin(yaw) * dx + math.cos(yaw) * dy
        outside_x = max(abs(forward) - half, 0)
        outside_y = max(abs(left) - half, 0)
        count += math.hypot(outside_x, outside_y) <= item["bounding_radius_xy_m"]
    return count


def diagnostics(output):
    records = read(PREPARED / "records.json")
    splits = read(PREPARED / "splits.json")["indices"]
    scene = read(PROJECT / "maps/downward_landmarks_v1/landmarks.manifest.json")
    train_rows = [records[i] for i in splits["train"]]
    positions = {r["position_id"]: (r["x"], r["y"]) for r in train_rows}
    train_positions = list(positions)
    train_xy = np.array(list(positions.values()))
    by_id = {r["sample_id"]: r for r in records}
    source = read(PREPARED / "source_reference.json")
    capture_rows = [
        json.loads(line)
        for line in (PROJECT / source["run"] / "capture/samples.jsonl").read_text().splitlines()
    ]
    if len(records) != len(capture_rows) or any(
        r["sample_id"] != c["sample_id"] for r, c in zip(records, capture_rows)
    ):
        raise ValueError("Prepared/capture sample order mismatch")
    maximum_target_error = 0.0
    maximum_planned_position_error = 0.0
    for row, capture in zip(records, capture_rows):
        yaw = math.radians(row["yaw"])
        expected = [
            -math.cos(yaw) * row["x"] - math.sin(yaw) * row["y"],
            math.sin(yaw) * row["x"] - math.cos(yaw) * row["y"],
        ]
        maximum_target_error = max(
            maximum_target_error,
            np.linalg.norm(np.array(expected) - [row["target_x"], row["target_y"]]),
        )
        maximum_planned_position_error = max(
            maximum_planned_position_error,
            np.linalg.norm(
                np.array(capture["planned_position_m"]) - [row["x"], row["y"], row["z"]]
            ),
        )
    visibility = {r["sample_id"]: candidates(r, scene["landmarks"]) for r in records}
    report = {
        "scope": "Post-training diagnostics only; no weights/labels/splits/production code changed",
        "model_sha256": hashlib.sha256(
            (RUN / "train/model_attention.pth").read_bytes()
        ).hexdigest(),
        "unique_png_sha256": len({r["sha256"] for r in records}),
        "images": len(records),
        "maximum_label_formula_error_m": float(maximum_target_error),
        "maximum_measured_vs_planned_position_error_m": float(maximum_planned_position_error),
        "ground_footprint_side_m": 20 * math.tan(math.radians(15)),
        "texture_texels_across_footprint": 20
        * math.tan(math.radians(15))
        / scene["texture"]["meters_per_texel"],
        "landmark_geometry": {
            "method": (
                "Bounding circle overlap with ground square; "
                "overestimates actual object visibility; shadows not classified"
            ),
            "all_images_candidate_count_histogram": dict(Counter(visibility.values())),
            "distinct_shape_color_pairs": len(
                {(p["shape"], tuple(p["color_rgb"])) for p in scene["landmarks"]}
            ),
            "total_landmarks": len(scene["landmarks"]),
        },
        "splits": {},
    }
    details = []
    for split in ["train", "validation", "test"]:
        data = np.load(RUN / "evaluate" / ("predictions_" + split + ".npz"))
        rows = [by_id[str(i)] for i in data["sample_ids"]]
        prediction, target = data["predictions"], data["targets"]
        cross = prediction[:, 0] * target[:, 1] - prediction[:, 1] * target[:, 0]
        dot = np.sum(prediction * target, axis=1)
        signed = np.arctan2(cross, dot)
        angles = np.abs(np.rad2deg(signed))
        distances = np.linalg.norm(prediction - target, axis=1)
        nearest_distance, nearest_yaw_gap = [], []
        for row, angle, error in zip(rows, angles, distances):
            near = np.linalg.norm(train_xy - [row["x"], row["y"]], axis=1).argmin()
            group = train_positions[int(near)]
            gap = min(
                abs((t["yaw"] - row["yaw"] + 180) % 360 - 180)
                for t in train_rows
                if t["position_id"] == group
            )
            nearest_distance.append(float(np.linalg.norm(train_xy[near] - [row["x"], row["y"]])))
            nearest_yaw_gap.append(gap)
            details.append(
                {
                    "sample_id": row["sample_id"],
                    "split": split,
                    "position_id": row["position_id"],
                    "candidate_landmarks": visibility[row["sample_id"]],
                    "angle_error_deg": float(angle),
                    "vector_error_m": float(error),
                }
            )
        groups = {}
        for label, mask in [
            ("zero_candidate_objects", np.array([visibility[r["sample_id"]] == 0 for r in rows])),
            (
                "one_or_more_candidate_objects",
                np.array([visibility[r["sample_id"]] > 0 for r in rows]),
            ),
        ]:
            groups[label] = {
                "images": int(mask.sum()),
                "mean_angle_error_deg": float(angles[mask].mean()) if mask.any() else None,
                "mean_vector_error_m": float(distances[mask].mean()) if mask.any() else None,
            }
        report["splits"][split] = {
            "positions": len({r["position_id"] for r in rows}),
            "nearest_training_position_distance_m": distribution(nearest_distance),
            "yaw_gap_at_nearest_training_position_deg": distribution(nearest_yaw_gap),
            "landmark_candidate_histogram": dict(Counter(visibility[r["sample_id"]] for r in rows)),
            "errors_by_landmark_candidates": groups,
            "fraction_angle_error_above_90_deg": float((angles > 90).mean()),
            "signed_angle_error_circular_concentration": float(abs(np.exp(1j * signed).mean())),
        }
    (output / "geometry_and_errors.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "sample_errors.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def network_probe(output):
    import sys

    sys.path.insert(0, str(PROJECT))
    import torch
    from simple_network import IncepAttentionCNN_rgb

    torch.set_num_threads(2)
    torch.manual_seed(10)
    model = IncepAttentionCNN_rgb().eval()
    parameters = sum(p.numel() for p in model.parameters())
    old_head_parameters = sum(p.numel() for p in model.fc.parameters())
    result = []
    with torch.inference_mode():
        for height, width in [(192, 1800), (192, 192), (256, 256), (384, 384), (512, 512)]:
            x = torch.zeros(1, 3, height, width)
            shapes = []
            for module in [model.inception1, model.inception2, model.conv3, model.conv4]:
                x = torch.tanh(module(x))
                shapes.append(list(x.shape))
            flat = x.flatten(1)
            head = torch.nn.Linear(flat.shape[1], 16)
            prediction = model.fc2(torch.tanh(head(flat)))
            result.append(
                {
                    "input_hw": [height, width],
                    "feature_shapes": shapes,
                    "flattened_features": flat.shape[1],
                    "output_shape": list(prediction.shape),
                    "parameters_with_matching_fc": parameters
                    - old_head_parameters
                    + sum(p.numel() for p in head.parameters()),
                }
            )
    payload = {
        "purpose": (
            "Shape probes only; no candidate model trained or saved; frozen weights untouched"
        ),
        "original_parameters": parameters,
        "torch": str(torch.__version__),
        "candidates": result,
    }
    (output / "network_shapes.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--network-probe", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else PROJECT / args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.network_probe:
        network_probe(output)
    else:
        diagnostics(output)


if __name__ == "__main__":
    main()
