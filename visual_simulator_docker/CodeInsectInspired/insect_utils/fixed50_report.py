"""Live labelled concentric circles and final50th-action error reports."""

import csv
import json
import math
from pathlib import Path

from insect_utils.experiment_config import PROJECT, digest
from insect_utils.experiment_state import atomic_json, now, read_json
from insect_utils.fixed50_metrics import COMPLETED, summarize_fixed
from insect_utils.navigation_protocol import verify_navigation
from insect_utils.spiral_narrow_report import save_figure, timestamp
from run_navigation_experiment import verify_phase


def live_paths(folder):
    folder = Path(folder)
    completed = {}
    paths = []
    for file in sorted((folder / "formal/episodes").glob("*/summary.json")):
        row = read_json(file)
        completed[row["episode_id"]] = row
        trajectory = file.parent / row["attempt"] / "trajectory.json"
        positions = read_json(trajectory)["positions_m"]
        paths.append(
            dict(
                episode_id=row["episode_id"],
                positions=positions,
                complete=row["reason"] == COMPLETED and row["steps"] == 50,
                final_distance_m=row["final_distance_m"],
            )
        )
    progress_path = folder / "progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {}
    for identifier, attempt in progress.get("active_episode_attempts", {}).items():
        if identifier in completed:
            continue
        path = folder / attempt
        path.resolve().relative_to(folder.resolve())
        start = read_json(path / "start.json")["episode"]["initial_position_m"]
        file = path / "steps.jsonl"
        lines = file.read_text().splitlines(keepends=True) if file.exists() else []
        actions = [json.loads(line) for line in lines if line.endswith("\n")]
        positions = [start] + [a["executed_position_after_m"] for a in actions]
        paths.append(
            dict(episode_id=identifier, positions=positions, complete=False, final_distance_m=None)
        )
    return list(completed.values()), paths


def endpoint_color(distance):
    for radius, color in ((0.5, "#238443"), (1, "#36a8a0"), (2, "#2878b5"), (3, "#ef9b20")):
        if distance <= radius:
            return color
    return "#cf443c"


def draw_rings(ax, radii):
    from matplotlib.patches import Circle

    for radius in radii:
        color = "#997422" if radius == 2 else "#777777"
        ax.add_patch(
            Circle(
                (0, 0),
                radius,
                fill=False,
                ec=color,
                lw=1.2 if radius == 2 else 0.7,
                ls="-" if radius == 2 else "--",
                alpha=0.8,
                zorder=2,
            )
        )
        angle = math.radians(35)
        ax.text(
            radius * math.cos(angle),
            radius * math.sin(angle),
            "{:g}m".format(radius),
            fontsize=8,
            color="black",
            ha="center",
            va="center",
            zorder=10,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5),
        )
    ax.scatter(0, 0, marker="+", c="black", s=65, zorder=9)
    ax.set(aspect="equal", xlabel="World X(m)", ylabel="World Y(m)")
    ax.grid(alpha=0.15)


def plot_live(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    folder = Path(folder)
    frozen = read_json(folder / "frozen.json")
    rows, paths = live_paths(folder)
    summary = summarize_fixed(rows, frozen["fixed50"]["scoring_radii_m"])
    analysis = folder / "analysis"
    analysis.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    extent = max([20] + [max(abs(p[0]), abs(p[1])) * 1.08 for t in paths for p in t["positions"]])
    overview_rings = list(frozen["fixed50"]["overview_rings_m"])
    overview_rings += list(range(30, int(extent // 10) * 10 + 1, 10))
    for t in paths:
        positions = t["positions"]
        color = endpoint_color(t["final_distance_m"]) if t["complete"] else "#aaaaaa"
        for ax in (axes[0, 0], axes[0, 1]):
            ax.plot(
                [p[0] for p in positions],
                [p[1] for p in positions],
                color=color,
                lw=0.8,
                alpha=0.4 if t["complete"] else 0.65,
                ls="-" if t["complete"] else ":",
                zorder=1,
            )
            ax.scatter(
                positions[-1][0],
                positions[-1][1],
                color=color,
                s=15,
                marker="o" if t["complete"] else "x",
                zorder=6,
            )
        if t["complete"]:
            axes[1, 0].scatter(positions[-1][0], positions[-1][1], color=color, s=24, zorder=6)
            distances = [math.hypot(*p[:2]) for p in positions]
            axes[1, 1].plot(range(len(distances)), distances, c=color, alpha=0.12, lw=0.7)
        else:
            distances = [math.hypot(*p[:2]) for p in positions]
            axes[1, 1].plot(
                range(len(distances)), distances, c="#aaaaaa", alpha=0.5, lw=0.7, ls=":"
            )
    draw_rings(axes[0, 0], overview_rings)
    axes[0, 0].set(
        xlim=(-extent, extent),
        ylim=(-extent, extent),
        title="All trajectories |dots=action50, x=active",
    )
    for ax in (axes[0, 1], axes[1, 0]):
        draw_rings(ax, frozen["fixed50"]["near_home_rings_m"])
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
    axes[0, 1].set_title("Near-home trajectories |continued after2m entry")
    outside = sum(t["complete"] and any(abs(v) > 6 for v in t["positions"][-1][:2]) for t in paths)
    axes[1, 0].set_title("Action50 endpoints |{} outside this zoom".format(outside))
    finished = [t for t in paths if t["complete"]]
    if finished:
        radii = np.asarray([[math.hypot(*p[:2]) for p in t["positions"]] for t in finished])
        steps = np.arange(51)
        axes[1, 1].plot(
            steps, np.median(radii, axis=0), c="black", lw=1.7, label="Median of completed trials"
        )
        axes[1, 1].fill_between(
            steps,
            np.percentile(radii, 10, axis=0),
            np.percentile(radii, 90, axis=0),
            color="#6b9dc2",
            alpha=0.18,
            label="P10-P90",
        )
    axes[1, 1].axhline(2, c="#997422", ls="--", label="2m reference (no stop)")
    axes[1, 1].set(
        xlim=(0, 50),
        xlabel="Executed action",
        ylabel="Distance to home(m)",
        title="Distance vs action; no arrival stopping",
    )
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.15)
    fig.suptitle(
        "15m starts |epoch484 |40m/30deg |exactly50 actions\nCompleted50: {}/64 |{}".format(
            summary["completed50"], timestamp()
        )
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.text(
        0.5,
        0.015,
        "Endpoint colors: <=0.5m green; <=1m teal; <=2m blue; <=3m orange; >3m red. "
        "Gray dotted=active.",
        ha="center",
        fontsize=9,
    )
    save_figure(fig, analysis / "fixed50_trajectories.png")
    plt.close(fig)
    atomic_json(
        analysis / "live_summary.json",
        dict(updated=now(), partial=summary["completed50"] < 64, summary=summary),
    )
    (analysis / "live_results.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>15m fixed50</title>'
        "<h1>15m |50 actions |no arrival stop</h1><p>Dashed rings labelled in meters. "
        "Endpoint statistics include completed50-action trials only. "
        "Gray dotted trajectories are active.</p>"
        '<img id="plot" style="max-width:100%"><script>function refresh(){'
        'document.getElementById("plot").src="fixed50_trajectories.png?t="+Date.now()}'
        "refresh();setInterval(refresh,30000)</script>",
        encoding="utf-8",
    )


def report(folder):
    folder = Path(folder)
    frozen = verify_navigation(folder)
    verify_phase(folder, "formal", frozen)
    results = read_json(folder / "formal/results.json")
    rows = results["episodes"]
    summary = results["summary"]
    if summary["completed50"] != 64 or summary["aborted"] or any(r["steps"] != 50 for r in rows):
        raise ValueError("Final report requires64 genuine50-action trials")
    plot_live(folder)
    with (folder / "analysis/endpoint_distances.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "episode_id",
                "bearing_deg",
                "initial_yaw_deg",
                "final_x_m",
                "final_y_m",
                "action50_distance_m",
                "minimum_endpoint_distance_m",
                "first_entry_2m_step",
                "last10_mean_distance_m",
                "last10_max_distance_m",
            ]
        )
        for r in rows:
            m = r["detail"]["fixed_horizon"]
            writer.writerow(
                [
                    r["episode_id"],
                    r["bearing_deg"],
                    r["initial_yaw_deg"],
                    *r["final_position_m"][:2],
                    r["final_distance_m"],
                    r["minimum_endpoint_distance_m"],
                    m["first_entry_2m_step"],
                    m["last10_mean_distance_m"],
                    m["last10_max_distance_m"],
                ]
            )
    parent = PROJECT / frozen["fixed50"]["source_navigation_run"]
    old = read_json(parent / "formal/results.json")["summary"]
    d = summary["final_distance_m"]
    lines = [
        "# 15m起点、固定50步：终点误差实验",
        "",
        "第484轮模型、40m高度、30°下视相机，使用同样16方位×4航向共64个起点。"
        "每条轨迹均执行50次原网络决策，进入2m后继续导航；动态步长和航向更新保持不变，"
        "没有加入悬停或真值修正。",
        "",
        "## 第50步终点距离",
        "",
        "均值{:.3f}m；中位数{:.3f}m；P90={:.3f}m；P95={:.3f}m；最大{:.3f}m。".format(
            d["mean"], d["p50"], d["p90"], d["p95"], d["maximum"]
        ),
        "",
        "| 最终距离范围 | 次数 | 比例 |",
        "|---|---:|---:|",
    ]
    for radius, row in summary["final_within_radius"].items():
        lines.append("| ≤{}m | {}/64 | {:.2%} |".format(radius, row["count"], row["rate"]))
    lines += [
        "",
        "途中曾进入2m：{}/64；第50步仍在2m内：{}/64；曾进入但最终在2m外：{}次。".format(
            summary["ever_entered_2m"],
            summary["final_within_2m"],
            summary["entered_then_outside_2m_at_end"],
        ),
        "最终位置平均偏移(X,Y)=({:.3f},{:.3f})m。".format(*summary["final_position_mean_xy_m"]),
        "最后10个端点的平均距离：跨轨迹均值{:.3f}m；各轨迹最后10步最大距离的中位数{:.3f}m。".format(
            summary["last10_mean_distance_m"]["mean"], summary["last10_max_distance_m"]["p50"]
        ),
        "",
        "## 与原15m实验的区别",
        "",
        (
            "原实验在首次进入2m时停止，成功{}/64（{:.2%}），平均最终距离{:.3f}m；"
            "本次衡量第50步终点，途中进入2m不会提前停止。两次渲染运行共享起点和模型，"
            "轨迹前缀可能存在数值差异。"
        ).format(old["successes"], old["success_rate"], old["final_distance_m"]["mean"]),
        "",
        "[标注圆环的轨迹图](analysis/fixed50_trajectories.png) · "
        "[逐条终点误差CSV](analysis/endpoint_distances.csv)",
        "",
        "圆环：近家0.5、1、2、3、5m；整体5、10、15、20m，必要时增加更大圆环。灰色虚线为尚未跑完的轨迹，完成后的彩色圆点为第50步端点。",
        "",
        "使用运动学相机代理；第50步位置不等于实机悬停稳定点。最后10步统计用于观察持续误差或振荡。未重新训练、未改变模型权重。",
        "",
    ]
    (folder / "result_summary.md").write_text("\n".join(lines), encoding="utf-8")
    completion = dict(
        status="completed",
        updated=now(),
        fingerprint=frozen["fingerprint"],
        selected_epoch=484,
        initial_radii_m=[15],
        neural_trials=64,
        horizon_actions=50,
        distance_based_termination=False,
        summary=summary,
        results_sha256=digest(folder / "formal/results.json"),
        report="result_summary.md",
    )
    atomic_json(folder / "completion.json", completion)
    atomic_json(folder / "progress.json", dict(completion, phase="completed"))
    atomic_json(folder / "pipeline_progress.json", dict(completion, phase="completed"))
    return completion
