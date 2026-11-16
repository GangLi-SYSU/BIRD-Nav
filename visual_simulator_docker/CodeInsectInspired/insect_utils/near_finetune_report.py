"""Automatically refreshed physical loss/trajectory plots and the paired final report."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import os
import uuid

from insect_utils.experiment_config import PROJECT, digest
from insect_utils.experiment_state import atomic_json, now, read_json
from insect_utils.near_finetune_protocol import verify_finetune
from insect_utils.rotation_report import matched_outcomes


def plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    fig.savefig(temporary, format="png", dpi=145)
    os.replace(temporary, path)


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).strftime("%m-%d %H:%M:%S CST")


def plot_training(folder):
    plt = plotting()
    folder = Path(folder)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    found = False
    for ax, arm in zip(axes, ("original_data", "near_supplement")):
        file = folder / "arms" / arm / "engine/training_history.json"
        if not file.exists():
            ax.set_title(arm + " | waiting")
            continue
        rows = read_json(file)["blocks"]
        found = True
        training = [r for r in rows if r["train_component_mse_m2"] is not None]
        ax.plot(
            [r["block"] for r in training],
            [r["train_component_mse_m2"] for r in training],
            label="train component MSE",
        )
        ax.plot(
            [r["block"] for r in rows],
            [r["validation_component_mse_m2"] for r in rows],
            label="full-validation MSE",
        )
        ax.plot(
            [r["block"] for r in rows],
            [r["selection_score"] for r in rows],
            label="band-macro MSE (selection)",
        )
        ax.set(
            title=arm + " | complete block{}/50".format(rows[-1]["block"]),
            xlabel="Block (512 updates)",
            ylabel="Physical MSE (m2)",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    if found:
        fig.suptitle("Matched25600-update fine tuning | " + stamp())
        fig.tight_layout()
        save_figure(fig, folder / "analysis/training_live.png")
    plt.close(fig)


def plot_navigation(folder, arm):
    plt = plotting()
    from matplotlib.patches import Circle

    folder = Path(folder)
    base = folder / "navigation" / arm
    if not (base / "frozen.json").exists():
        return
    frozen = read_json(base / "frozen.json")
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for row, radius in enumerate((10, 5)):
        completed = successes = active = 0
        for episode in frozen["episodes"]["formal"]:
            if episode["radius_m"] != radius:
                continue
            path = base / "formal/episodes" / episode["episode_id"]
            if (path / "summary.json").exists():
                summary = read_json(path / "summary.json")
                completed += 1
                successes += summary["reason"] == "success"
                positions = read_json(path / summary["attempt"] / "trajectory.json")["positions_m"]
                color = "#238846" if summary["reason"] == "success" else "#ce4a49"
            else:
                attempts = sorted(
                    path.glob("attempts/*/start.json"), key=lambda p: p.stat().st_mtime
                )
                if not attempts:
                    continue
                positions = [episode["initial_position_m"]]
                active += 1
                color = "#3578bb"
                steps = attempts[-1].parent / "steps.jsonl"
                if steps.exists():
                    for line in steps.read_text().splitlines():
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            break
                        positions.append(record["executed_position_after_m"])
            for ax in axes[row]:
                ax.plot(
                    [p[0] for p in positions], [p[1] for p in positions], c=color, alpha=0.5, lw=0.8
                )
                ax.scatter(positions[-1][0], positions[-1][1], c=color, s=10)
        for col, ax in enumerate(axes[row]):
            ax.add_patch(Circle((0, 0), 0.5, fc="#ffd368", ec="black", zorder=10))
            ax.set(
                aspect="equal",
                xlabel="World X(m)",
                ylabel="World Y(m)",
                title="{}m | complete{}/64 success{} active{} | {}".format(
                    radius, completed, successes, active, "near home" if col else "full path"
                ),
            )
            extent = 3.5 if col else radius + 2
            ax.set_xlim(-extent, extent)
            ax.set_ylim(-extent, extent)
            ax.grid(alpha=0.2)
    fig.suptitle(arm + " | green=success,red=failure,blue=active | " + stamp())
    fig.tight_layout()
    save_figure(fig, folder / "analysis" / ("navigation_" + arm + "_live.png"))
    plt.close(fig)


def report_final(folder):
    folder = Path(folder)
    frozen = verify_finetune(folder)
    arms = frozen["spec"]["arms"]
    training = {a: read_json(folder / "arms" / a / "completion.json") for a in arms}
    navigation = {a: read_json(folder / "navigation" / a / "formal/results.json") for a in arms}
    if any(n["summary"]["episodes"] != 128 for n in navigation.values()):
        raise ValueError("Incomplete matched navigation budget")
    paired = {}
    for radius in (10, 5):
        paired[str(radius)] = matched_outcomes(
            *[[e for e in navigation[a]["episodes"] if e["radius_m"] == radius] for a in arms]
        )
    source = read_json(PROJECT / frozen["spec"]["navigation_source"] / "formal/results.json")
    comparison = dict(
        fingerprint=frozen["fingerprint"],
        training=training,
        navigation={a: n["summary"] for a, n in navigation.items()},
        paired=paired,
        historical589=source["summary"],
        historical589_rerun=False,
        limits=frozen["spec"]["limits"],
    )
    atomic_json(folder / "analysis/comparison.json", comparison)
    plot_training(folder)
    for arm in arms:
        plot_navigation(folder, arm)
    lines = [
        "# 第三步：近家补数的等更新次数对照",
        "",
        "两组从第589轮同一完整断点开始，各新增25,600次更新。原数据组只使用原4096张训练图；"
        "补数组使用5640张训练图。共享448张验证图、距离分组平均MSE选模及原控制器。",
        "",
        "| 组别 | 最佳微调段 | 验证分组平均MSE(m²) | 原点8航向预测长度均值(m) |",
        "|---|---:|---:|---:|",
    ]
    for arm, value in training.items():
        lines.append(
            "| {} | {} | {:.4f} | {:.3f} |".format(
                arm,
                value["selected_block"],
                value["best_validation"]["selection_score"],
                value["probe_summary"]["home_output_norm_m"]["mean"],
            )
        )
    lines += ["", "| 组别 | 起始半径 | 成功/64 | 成功率 | 平均最终距离(m) |", "|---|---:|---:|---:|---:|"]
    for arm, value in navigation.items():
        for radius in ("10", "5"):
            s = value["summary"]["by_radius"][radius]
            lines.append(
                "| {} | {}m | {} | {:.2%} | {:.3f} |".format(
                    arm, radius, s["successes"], s["success_rate"], s["final_distance_m"]["mean"]
                )
            )
    lines += [
        "",
        "历史589轮原模型为10m40/64、5m46/64，本轮未重新运行该原始权重；" "本轮original_data是增加相同训练预算后的对照。",
        "",
        "原点诊断位置现已是训练位置，104张重放只用于开发诊断，不作为独立盲测或选模依据。" "新增局部验证为0.15m隔离的同场景插值；18个预留位置未采图、未评估。",
        "",
        "[训练曲线](analysis/training_live.png) · "
        "[原数据轨迹](analysis/navigation_original_data_live.png) · "
        "[近家补数轨迹](analysis/navigation_near_supplement_live.png)",
        "",
    ]
    (folder / "result_summary.md").write_text("\n".join(lines), encoding="utf-8")
    names = [
        "result_summary.md",
        "analysis/comparison.json",
        "analysis/training_live.png",
        "analysis/navigation_original_data_live.png",
        "analysis/navigation_near_supplement_live.png",
    ]
    completed = dict(
        status="completed",
        updated=now(),
        fingerprint=frozen["fingerprint"],
        optimizer_steps_added_per_arm=25600,
        neural_trials_per_arm=128,
        summaries=comparison["navigation"],
        report="result_summary.md",
        files={n: digest(folder / n) for n in names},
    )
    atomic_json(folder / "completion.json", completed)
    atomic_json(folder / "progress.json", dict(completed, phase="completed"))
    return completed
