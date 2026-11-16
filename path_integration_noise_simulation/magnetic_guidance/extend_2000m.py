"""Extend the fixed dogleg condition; preserve the first 50 published trials."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from run_experiment import schedule, simulate, wilson, write_csv  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw_arm(output, arm, outbound, endpoints, example, split, stats):
    teal, orange = "#168d96", "#e88b28"
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    ax, zoom = axes
    ax.scatter(*outbound.T, s=7, color=teal, alpha=0.5, label="Outbound endpoints (n=1000)")
    ax.scatter(*endpoints.T, s=7, color=orange, alpha=0.6, label="Homing endpoints (n=1000)")
    ax.plot(*example[: split + 1].T, color=teal, lw=1.7, label="Outbound path: trial 0")
    ax.plot(*example[split:].T, color=orange, lw=1.7, label="Homing path: trial 0")
    ax.add_patch(
        Circle(
            (0, 0),
            stats["outbound_r99_m"],
            fill=False,
            color=teal,
            ls="--",
            lw=1.3,
            label="99% outbound radius",
        )
    )
    ax.add_patch(
        Circle(
            (0, 0),
            stats["return_r99_m"],
            fill=False,
            color="#555555",
            ls="--",
            lw=1.3,
            label="99% homing radius",
        )
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Flight overview (Fig. 2a-style)")
    good = np.linalg.norm(endpoints, axis=1) <= 15
    zoom.scatter(*endpoints[good].T, s=11, color=orange, alpha=0.55, label="Within 15 m")
    zoom.scatter(*endpoints[~good].T, s=22, marker="x", color="#b32836", label="Outside 15 m")
    zoom.add_patch(
        Circle((0, 0), 15, fill=False, color="#238a43", lw=1.8, label="Visual-stage radius: 15 m")
    )
    zoom.add_patch(
        Circle(
            (0, 0),
            stats["return_r99_m"],
            fill=False,
            color="#555555",
            ls="--",
            label=f"99% homing radius: {stats['return_r99_m']:.2f} m",
        )
    )
    limit = max(18, float(np.abs(endpoints).max()) * 1.1)
    zoom.set(xlim=(-limit, limit), ylim=(-limit, limit), title="All 1000 homing endpoints")
    zoom.legend(loc="upper left", fontsize=8)
    for axis in axes:
        axis.scatter([0], [0], marker="*", s=130, color="black", zorder=8)
        axis.set(xlabel="East (m)", ylabel="North (m)", aspect="equal")
        axis.grid(alpha=0.15)
    label = "Perfect magnetic heading" if arm == "magnetic" else "Gyroscope only (paired control)"
    fig.suptitle(
        f"{label} | D = 2000 m | sigma = 0.1 m/sqrt(m)\n"
        f"Within 15 m: {stats['successes']}/1000 ({stats['success_rate']:.1%})"
        f" | 99% area ratio: {stats['area_ratio_percent']:.5f}%"
    )
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"fig2a_{arm}.{suffix}", dpi=210)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cfg_path = root / "config_2000m_1000.json"
    config = json.loads(cfg_path.read_text())
    source = root / "runs/magnetic_01/dogleg_d2000_s0.1.npz"
    previous = json.loads((source.parent / "config.json").read_text())
    for key in [
        "seed",
        "delta_t_s",
        "outbound_speed_m_s",
        "return_speed_m_s",
        "gyro_noise_deg_per_sqrt_s",
        "success_radius_m",
    ]:
        assert config[key] == previous[key], key
    assert config["trials"] == 1000 and config["trial_start"] == 0
    assert config["distances_m"] == [2000] and config["routes"] == ["dogleg"]
    assert config["distance_noise_m_per_sqrt_m"] == [0.1]
    assert config["success_radius_m"] == 15
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in [Path(__file__), root / "run_experiment.py", cfg_path]:
        shutil.copy2(path, snapshot / path.name)
    shutil.copy2(cfg_path, output / "config.json")
    inputs = [Path(__file__), root / "run_experiment.py", cfg_path, source]
    (output / "input_hashes.json").write_text(
        json.dumps({str(p): sha(p) for p in inputs}, indent=2)
    )
    (_, _, _), split = schedule(2000, "dogleg", config)
    rows, all_out, all_end, examples = [], {}, {}, {}
    for start in range(0, 1000, 50):
        ids = list(range(start, start + 50))
        if start == 0:
            with np.load(source) as old:
                assert np.array_equal(old["trial_ids"], ids)
                paths = {arm: old[arm].copy() for arm in ["magnetic", "gyro_only"]}
            # Exact replay validates compatibility; formal data are reused, not duplicated.
            replay, _ = simulate(config, 2000, "dogleg", 0.1, ids)
            for arm in paths:
                np.testing.assert_array_equal(paths[arm], replay[arm])
        else:
            paths, _ = simulate(config, 2000, "dogleg", 0.1, ids)
        batch = output / f"trajectories_{start:04d}_{start+49:04d}.npz"
        np.savez_compressed(batch, trial_ids=ids, **paths)
        with np.load(batch) as saved:
            for arm, positions in paths.items():
                assert np.isfinite(positions).all()
                np.testing.assert_array_equal(saved[arm], positions)
                all_out.setdefault(arm, []).append(positions[:, split])
                all_end.setdefault(arm, []).append(positions[:, -1])
                if start == 0:
                    examples[arm] = positions[0]
                for i, trial_id in enumerate(ids):
                    endpoint, departure = positions[i, -1], positions[i, split]
                    error = float(np.linalg.norm(endpoint))
                    rows.append(
                        dict(
                            arm=arm,
                            trial_id=trial_id,
                            reused_prior_trial=trial_id < 50,
                            outbound_x_m=departure[0],
                            outbound_y_m=departure[1],
                            outbound_radius_m=float(np.linalg.norm(departure)),
                            final_x_m=endpoint[0],
                            final_y_m=endpoint[1],
                            final_error_m=error,
                            success=error <= 15,
                        )
                    )
        print(f"Completed and verified {start+50}/1000 paired trials", flush=True)
    write_csv(output / "trials.csv", rows)
    summaries = []
    for arm in ["magnetic", "gyro_only"]:
        outbound, endpoints = np.concatenate(all_out[arm]), np.concatenate(all_end[arm])
        errors = np.linalg.norm(endpoints, axis=1)
        success = int(np.sum(errors <= 15))
        lo, hi = wilson(success, 1000)
        r_out, r_in = np.percentile(np.linalg.norm(outbound, axis=1), 99), np.percentile(errors, 99)
        stats = dict(
            arm=arm,
            trials=1000,
            successes=success,
            success_rate=success / 1000,
            ci95_low=lo,
            ci95_high=hi,
            mean_error_m=float(errors.mean()),
            median_error_m=float(np.median(errors)),
            p95_error_m=float(np.percentile(errors, 95)),
            return_r99_m=float(r_in),
            maximum_error_m=float(errors.max()),
            outbound_r99_m=float(r_out),
            area_ratio_percent=float(100 * (r_in / r_out) ** 2),
        )
        summaries.append(stats)
        draw_arm(output, arm, outbound, endpoints, examples[arm], split, stats)
    write_csv(output / "summary.csv", summaries)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for arm, color in [("magnetic", "#168d96"), ("gyro_only", "#e88b28")]:
        errors = np.sort(np.linalg.norm(np.concatenate(all_end[arm]), axis=1))
        ax.step(errors, np.arange(1, 1001) / 1000, where="post", label=arm, color=color)
    ax.axvline(15, color="#238a43", ls="--", label="15 m visual-stage radius")
    ax.set(
        xscale="log",
        xlabel="Final distance from home (m, log scale)",
        ylabel="Cumulative fraction",
        title="2000 m dogleg / sigma = 0.1 / 1000 paired trials",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output / "endpoint_cdf.png", dpi=180)
    plt.close(fig)
    report = [
        "# 2000 m、σ=0.1：1000 次路径积分归巢实验",
        "",
        "沿用折线外出：名义离家 2000 m，外出航程 4000 m，返程 2000 m。",
        "无噪声地磁辅助 1000 次，另有相同随机数配对的无地磁对照 1000 次。",
        "每组复用上一轮 ID 0–49 的 50 次，新增 ID 50–999 的 950 次。",
        "停止于自身估计的家，真实终点 ≤15 m 仅用于事后评分。σ 的单位为 m/√m。",
        "",
        "| 指标 | 地磁辅助 | 无地磁对照 |",
        "|---|---:|---:|",
    ]
    a, b = summaries
    report += [
        f"| 成功次数 | {a['successes']}/1000 | {b['successes']}/1000 |",
        f"| 成功率 | {a['success_rate']:.1%} | {b['success_rate']:.1%} |",
        f"| Wilson 95% 区间 | {a['ci95_low']:.2%}–{a['ci95_high']:.2%} | "
        f"{b['ci95_low']:.2%}–{b['ci95_high']:.2%} |",
    ]
    for key, label in [
        ("mean_error_m", "平均终点误差/m"),
        ("median_error_m", "中位终点误差/m"),
        ("p95_error_m", "95% 终点半径/m"),
        ("return_r99_m", "99% 终点半径/m"),
        ("maximum_error_m", "最大误差/m"),
        ("outbound_r99_m", "99% 外出终点半径/m"),
        ("area_ratio_percent", "99% 归巢圆/外出圆面积比/%"),
    ]:
        report.append(f"| {label} | {a[key]:.5f} | {b[key]:.5f} |")
    report += [
        "",
        "## 可视化",
        "",
        "参照[论文 Fig. 2a 图注](https://www.nature.com/articles/s41586-026-10461-3/figures/2)：",
        "青色为全部外出终点，橙色为归巢终点，星形为家；虚线圆为以家为圆心的经验 99% 半径。",
        "轨迹示例固定使用 ID 0，未按成败挑选。没有旋转或缩放真实坐标。",
        "右侧增加局部放大：全部 1000 个终点均显示，红叉表示落在 15 m 圈外；绿色为固定 15 m 圈。",
        "99% 半径使用 NumPy 默认线性分位数；面积比是两个圆面积之比，不是成功率。",
        "外出点采用外出结束位置，与论文图注一致；不采用原脚本中的最远途经点。",
        "",
        "![地磁辅助](fig2a_magnetic.png)",
        "",
        "![无地磁对照](fig2a_gyro_only.png)",
        "",
        "![误差累计分布](endpoint_cdf.png)",
        "",
        "## 解释范围",
        "",
        "这是上一轮二维高斯运动学模型的扩展，不是重现论文原始参数或 Isaac 实机动力学。",
        "准确航向消除了方向随机游走，不能消除里程累积误差。15 m 内仅代表视觉阶段可接管的几何范围。",
        "各试验种子独立；两组配对。保存完整逐步轨迹、原始终点表、源代码快照和输入校验值。",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    (output / "completion.json").write_text(
        json.dumps(
            dict(
                status="complete",
                trials_per_arm=1000,
                arms=2,
                reused_per_arm=50,
                new_per_arm=950,
                first_50_exact_replay=True,
                all_saved_trajectories_verified=True,
                summary_sha256=sha(output / "summary.csv"),
            ),
            indent=2,
        )
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
