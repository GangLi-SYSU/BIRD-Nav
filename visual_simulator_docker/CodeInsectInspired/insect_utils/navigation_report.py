"""Plot every recorded trajectory and report fixed-protocol navigation outcomes."""

from pathlib import Path

from insect_utils.experiment_config import digest
from insect_utils.experiment_state import atomic_json, now, read_json


def report_navigation(folder, frozen):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    folder = Path(folder)
    result = {phase: read_json(folder / phase / "results.json") for phase in ("pilot", "formal")}
    home = frozen["protocol"]["home_camera_xyz_m"]
    for phase, value in result.items():
        radii = frozen["protocol"][phase]["radii_m"]
        fig, axes = plt.subplots(1, len(radii), figsize=(4.2 * len(radii), 4.3), squeeze=False)
        for ax, radius in zip(axes[0], radii):
            selected = [r for r in value["episodes"] if r["radius_m"] == radius]
            for episode in selected:
                path = (
                    folder
                    / phase
                    / "episodes"
                    / episode["episode_id"]
                    / episode["attempt"]
                    / "trajectory.json"
                )
                positions = read_json(path)["positions_m"]
                success = episode["reason"] == "success"
                color = "#238846" if success else "#ce4a49"
                ax.plot(
                    [p[0] for p in positions],
                    [p[1] for p in positions],
                    color=color,
                    alpha=0.4,
                    lw=0.9,
                )
                ax.scatter(positions[0][0], positions[0][1], s=9, c="#888888")
                ax.scatter(positions[-1][0], positions[-1][1], s=9, c=color)
            ax.add_patch(Circle(home[:2], 0.5, facecolor="#ffd368", edgecolor="black", zorder=10))
            ax.scatter(home[0], home[1], marker="+", c="black", s=25, zorder=11)
            count = sum(r["reason"] == "success" for r in selected)
            ax.set(
                aspect="equal",
                xlabel="World X (m)",
                ylabel="World Y (m)",
                title="R={}m: {}/{} success".format(radius, count, len(selected)),
            )
            ax.grid(alpha=0.2)
        fig.suptitle(
            "{}: green=success, red=failure; true horizontal success radius0.5m".format(phase)
        )
        fig.tight_layout()
        fig.savefig(folder / (phase + "_trajectories.png"), dpi=170)
        plt.close(fig)
    formal, pilot = result["formal"]["summary"], result["pilot"]["summary"]
    lines = [
        "# 第十二阶段：0.5m成功半径的视觉归巢测试",
        "",
        "模型：Stage11验证集预选归一化组第101轮；40m离地高度、70°下视相机，" "完整1024×1024 RGB输入、网络内部384×384、部署图已将输出恢复为米。",
        "",
        "成功定义：每步完成位姿更新后，仿真真值水平距离≤0.5m即终止并记成功。" "保持50步预算及原动态步长min(0.1+0.4×预测距离/3,2.5)m；没有用真值缩短最后一步。",
        "",
        "## 结果",
        "",
        "- 小规模测试：{}/{}成功（{:.1%}）。".format(
            pilot["successes"], pilot["nontrivial_episodes"], pilot["success_rate"]
        ),
        "- 正式测试：{}/{}成功（{:.1%}）。".format(
            formal["successes"], formal["nontrivial_episodes"], formal["success_rate"]
        ),
        "- 正式测试结束原因：`{}`。".format(formal["termination_reasons"]),
        "",
        "| 起始半径(m) | 成功/总数 | 成功率 | 所有试验最终距离均值(m) |",
        "|---|---:|---:|---:|",
    ]
    for radius, value in formal["by_radius"].items():
        lines.append(
            "| {} | {}/{} | {:.1%} | {:.3f} |".format(
                radius,
                value["successes"],
                value["episodes"],
                value["success_rate"],
                value["final_distance_m"]["mean"],
            )
        )
    lines += [
        "",
        "## 解释范围",
        "",
        "这是原仓库风格的离散相机/机器人代理位姿闭环，未仿真四旋翼动力学、制动、悬停或降落。"
        "真实位置仅用于成功判断、误差统计、边界监督和位姿执行核验；网络策略只读取RGB。"
        "真值向量只用于独立oracle工程验收。",
        "",
        "评测使用训练过的同一个场景，视野可能与训练图重叠；不同初始航向也不是独立场景重复。" "不把所有轨迹帧当作独立统计样本，不据导航结果改选模型或调整阈值。",
        "",
        "## 文件",
        "",
        "- `formal/results.json`：全部320次结果及按半径、初始航向统计。",
        "- `pilot/results.json`：全部48次小规模结果。",
        "- `formal/episodes/`：每次的照片、帧/相机凭据、预测、执行位姿及完整轨迹。",
        "- `frozen.json`、`protocol.json`、`episode_plan.json`：冻结的模型、代码、场景和试验条件。",
        "- [正式轨迹](formal_trajectories.png)；[小规模轨迹](pilot_trajectories.png)。",
        "",
    ]
    (folder / "result_summary.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_json(
        folder / "completion.json",
        {
            "status": "completed",
            "updated": now(),
            "fingerprint": frozen["fingerprint"],
            "formal_episodes": formal["episodes"],
            "formal_successes": formal["successes"],
            "formal_success_rate": formal["success_rate"],
            "success_radius_m": 0.5,
            "formal_results_sha256": digest(folder / "formal/results.json"),
            "report": "result_summary.md",
        },
    )
