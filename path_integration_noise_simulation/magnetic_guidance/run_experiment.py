"""Isolated, reproducible extension of the repository's PI diffusion model."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def leg(length, heading, speed, dt):
    """Include a fractional last step; never round away distance."""
    count = int(math.floor(length / (speed * dt)))
    lengths = [speed * dt] * count
    remainder = length - sum(lengths)
    if remainder > 1e-9:
        lengths.append(remainder)
    lengths = np.asarray(lengths)
    return lengths, np.full(len(lengths), heading), lengths / speed


def schedule(distance, route, config):
    if route == "straight":
        legs = [(distance, 0)]
    elif route == "dogleg":
        legs = [(distance / 2, angle) for angle in [0, np.pi / 2, 0, -np.pi / 2]]
    else:
        raise ValueError(route)
    outbound = [leg(d, a, config["outbound_speed_m_s"], config["delta_t_s"]) for d, a in legs]
    out = [np.concatenate([item[k] for item in outbound]) for k in range(3)]
    back = leg(distance, np.pi, config["return_speed_m_s"], config["delta_t_s"])
    return [np.concatenate([out[k], back[k]]) for k in range(3)], len(out[0])


def simulate(config, distance, route, sigma, trial_ids):
    """Nominal odometry and noisy physical motion, matching original code convention.

    Magnetic heading resets angular drift to zero every update. Actual position
    never enters the controller. Paired arms share signed distance increments.
    """
    (lengths, headings, durations), outbound_steps = schedule(distance, route, config)
    route_id = {"straight": 0, "dogleg": 1}[route]
    distance_errors, gyro_errors, bearings = [], [], []
    for trial_id in trial_ids:
        # Sigma and arm deliberately excluded: common random numbers for comparison.
        rng = np.random.default_rng(
            np.random.SeedSequence([config["seed"], route_id, int(distance * 1000), int(trial_id)])
        )
        bearings.append(rng.uniform(-np.pi, np.pi))
        distance_errors.append(rng.standard_normal(len(lengths)))
        gyro_errors.append(rng.standard_normal(len(lengths)))
    bearing = np.asarray(bearings)[:, None]
    steps = lengths + sigma * np.sqrt(lengths) * np.asarray(distance_errors)
    gyro = np.deg2rad(config["gyro_noise_deg_per_sqrt_s"]) * np.sqrt(durations)
    drift = np.cumsum(np.asarray(gyro_errors) * gyro, axis=1)
    result = {}
    for arm in ["magnetic", "gyro_only"]:
        angles = headings + bearing + (drift if arm == "gyro_only" else 0)
        increments = steps[:, :, None] * np.stack([np.cos(angles), np.sin(angles)], -1)
        position = np.concatenate(
            [np.zeros((len(trial_ids), 1, 2)), np.cumsum(increments, axis=1)], axis=1
        )
        result[arm] = position
    return (
        result,
        {
            "outbound_steps": outbound_steps,
            "negative_increment_count": np.sum(steps < 0, axis=1),
            "total_steps": len(lengths),
            "outbound_length_m": float(np.sum(lengths[:outbound_steps])),
            "return_length_m": float(np.sum(lengths[outbound_steps:])),
        },
    )


def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0, center - half), min(1, center + half)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plots(output, config, summaries, examples):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for r, route in enumerate(config["routes"]):
        for c, arm in enumerate(["magnetic", "gyro_only"]):
            ax = axes[r, c]
            values = np.array(
                [
                    [
                        next(
                            row["success_rate"]
                            for row in summaries
                            if row["route"] == route
                            and row["arm"] == arm
                            and row["distance_m"] == d
                            and row["sigma"] == s
                        )
                        for d in config["distances_m"]
                    ]
                    for s in config["distance_noise_m_per_sqrt_m"]
                ]
            )
            im = ax.imshow(values * 100, vmin=0, vmax=100, cmap="viridis", aspect="auto")
            ax.set_xticks(range(values.shape[1]), config["distances_m"])
            ax.set_yticks(range(values.shape[0]), config["distance_noise_m_per_sqrt_m"])
            ax.set(
                xlabel="Nominal distance from home (m)",
                ylabel="Distance sigma (m/sqrt(m))",
                title=f"{route} / {arm}: endpoint within 15 m (%)",
            )
            for i, j in np.ndindex(values.shape):
                ax.text(
                    j,
                    i,
                    f"{values[i,j]*100:.0f}",
                    ha="center",
                    va="center",
                    color="black" if values[i, j] > 0.6 else "white",
                )
    fig.colorbar(im, ax=axes, label="Success (%)")
    fig.savefig(output / "success_rates.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for ax, arm in zip(axes, ["magnetic", "gyro_only"]):
        points = examples[arm][:, -1]
        ax.scatter(points[:, 0], points[:, 1], s=15)
        ax.add_patch(Circle((0, 0), 15, fill=False, color="red", label="15 m"))
        ax.scatter([0], [0], marker="*", color="red")
        ax.set(
            aspect="equal",
            xlabel="East (m)",
            ylabel="North (m)",
            title=f"{arm}: 1000 m dogleg, sigma=0.5",
        )
        ax.legend()
        ax.grid(alpha=0.3)
    fig.savefig(output / "endpoint_example.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["trials"] < 1 or config["trial_start"] < 0:
        raise ValueError("Invalid trial count/index")
    for key in ["delta_t_s", "outbound_speed_m_s", "return_speed_m_s", "success_radius_m"]:
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(key)
    if any(d <= 0 or not np.isfinite(d) for d in config["distances_m"]):
        raise ValueError("Distances must be positive and finite")
    if any(s < 0 or not np.isfinite(s) for s in config["distance_noise_m_per_sqrt_m"]):
        raise ValueError("Noise must be nonnegative and finite")
    if config["routes"] != ["straight", "dogleg"]:
        raise ValueError("This report expects straight and dogleg routes")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    original = Path(__file__).resolve().parents[1] / "determine_LHA_odometry.py"
    inputs = [Path(__file__).resolve(), original, args.config.resolve()]
    manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (args.output / "input_hashes.json").write_text(json.dumps(manifest, indent=2))
    trial_ids = list(range(config["trial_start"], config["trial_start"] + config["trials"]))
    rows, summaries, examples = [], [], None
    for route in config["routes"]:
        for distance in config["distances_m"]:
            for sigma in config["distance_noise_m_per_sqrt_m"]:
                trajectories, metadata = simulate(config, distance, route, sigma, trial_ids)
                name = f"{route}_d{distance}_s{sigma}"
                np.savez_compressed(
                    args.output / (name + ".npz"), trial_ids=trial_ids, **trajectories
                )
                for arm, positions in trajectories.items():
                    errors = np.linalg.norm(positions[:, -1], axis=1)
                    successes = int(np.sum(errors <= config["success_radius_m"]))
                    low, high = wilson(successes, len(trial_ids))
                    summary = dict(
                        route=route,
                        arm=arm,
                        distance_m=distance,
                        sigma=sigma,
                        trials=len(trial_ids),
                        successes=successes,
                        success_rate=successes / len(trial_ids),
                        ci95_low=low,
                        ci95_high=high,
                        mean_error_m=float(errors.mean()),
                        median_error_m=float(np.median(errors)),
                        p95_error_m=float(np.percentile(errors, 95)),
                        nominal_outbound_length_m=metadata["outbound_length_m"],
                        nominal_return_length_m=metadata["return_length_m"],
                    )
                    summaries.append(summary)
                    for i, trial_id in enumerate(trial_ids):
                        rows.append(
                            dict(
                                route=route,
                                arm=arm,
                                distance_m=distance,
                                sigma=sigma,
                                trial_id=trial_id,
                                final_x_m=positions[i, -1, 0],
                                final_y_m=positions[i, -1, 1],
                                error_m=errors[i],
                                success=bool(errors[i] <= config["success_radius_m"]),
                                actual_outbound_radius_m=float(
                                    np.linalg.norm(positions[i, metadata["outbound_steps"]])
                                ),
                                negative_increments=int(metadata["negative_increment_count"][i]),
                                steps=metadata["total_steps"],
                            )
                        )
                if route == "dogleg" and distance == 1000 and sigma == 0.5:
                    examples = trajectories
                print(
                    f"Completed {name}: 50 paired trials"
                    if len(trial_ids) == 50
                    else f"Completed {name}: {len(trial_ids)} paired trials",
                    flush=True,
                )
    write_csv(args.output / "trials.csv", rows)
    write_csv(args.output / "summary.csv", summaries)
    if examples is not None:
        plots(args.output, config, summaries, examples)
    report = [
        "# 无噪声地磁辅助路径积分：首轮实验",
        "",
        "每格为成功次数/实验次数；真实终点距家 ≤15 m。停止依据为内部估计归零。",
        "距离是名义离家距离；直线外出航程 D，折线外出航程 2D，名义返程均 D。",
        "σ 单位 m/√m，单步距离增量噪声标准差 σ√Δs；不是百分比误差。",
        "",
    ]
    for route in config["routes"]:
        for arm in ["magnetic", "gyro_only"]:
            report += [
                f"## {route} / {arm}",
                "",
                "| σ | " + " | ".join(f"{d} m" for d in config["distances_m"]) + " |",
                "|---|" + "---|" * len(config["distances_m"]),
            ]
            for sigma in config["distance_noise_m_per_sqrt_m"]:
                group = [
                    next(
                        x
                        for x in summaries
                        if x["route"] == route
                        and x["arm"] == arm
                        and x["distance_m"] == d
                        and x["sigma"] == sigma
                    )
                    for d in config["distances_m"]
                ]
                report.append(
                    f"| {sigma} | "
                    + " | ".join(
                        f"{x['successes']}/{x['trials']} ({x['success_rate']:.0%})" for x in group
                    )
                    + " |"
                )
            report.append("")
    report += [
        "## 解释与限制",
        "",
        "地磁消除了累计航向漂移，但不能消除里程误差；因此距离越长仍可能错过视觉范围。",
        "这是原代码的二维运动学噪声模型扩展，不是 Isaac 场景飞行或真实磁力计验证。",
        "无地磁对照使用原机器人的 0.63°/√s 航向随机游走；初始方向准确。",
        "原始高斯模型允许有符号负步长；高噪声仅作压力测试，不解释为真实传感器标定。",
        "每条记录保存负步长次数。没有裁剪高斯分布，以免引入均值偏差。",
        "原代码在估计距家不足一个返程步长时停止；此实现补足最后一段至估计归零，",
        "避免把最高 2 m 的离散残差混入误差。真实位置只用于评分。",
        "各组共享随机数，便于配对比较；每组内部 50 个独立种子。",
        "summary.csv 给出 Wilson 95% 区间；50/50 对应约 92.9%–100%，不能声称必然成功。",
        "15 m 仅表示进入第二阶段候选范围，并不等同于视觉归巢最终成功。",
        "",
        "![成功率](success_rates.png)",
        "",
        "![终点示例](endpoint_example.png)",
    ]
    (args.output / "REPORT.md").write_text("\n".join(report) + "\n")
    (args.output / "completion.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "groups": len(summaries),
                "episodes": len(rows),
                "summary_sha256": hashlib.sha256(
                    (args.output / "summary.csv").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
