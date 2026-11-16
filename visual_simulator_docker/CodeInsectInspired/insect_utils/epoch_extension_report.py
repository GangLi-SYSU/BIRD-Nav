"""Report the chosen extra epoch budget against the frozen300-epoch baseline."""

from pathlib import Path

from insect_utils.epoch_extension import verify_extension
from insect_utils.experiment_config import PROJECT, digest
from insect_utils.experiment_state import atomic_json, now, read_json
from insect_utils.rotation_report import matched_outcomes


def report_extension(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    folder = Path(folder)
    frozen = verify_extension(folder)
    target = frozen["training_config"]["epochs"]
    source = PROJECT / frozen["protocol"]["source_run"]
    bases = {"source300": source / "navigation", "extended" + str(target): folder / "navigation"}
    results = {name: read_json(base / "formal/results.json") for name, base in bases.items()}
    if any(
        r["summary"]["episodes"] != 64 or set(r["summary"]["by_radius"]) != {"20"}
        for r in results.values()
    ):
        raise ValueError("The report requires64 trials at20m for each budget")
    old, new = list(results)
    previous = read_json(source / "training/completion.json")
    completed = read_json(folder / "training/completion.json")
    paired = matched_outcomes(results[old]["episodes"], results[new]["episodes"])
    raw = paired.pop("counts")
    paired["counts"] = dict(
        both_success=raw["both_success"],
        only_source_success=raw["only_mask_success"],
        only_extended_success=raw["only_rotation_success"],
        both_fail=raw["both_fail"],
    )
    renamed = {
        "mask_success": "source_success",
        "rotation_success": "extended_success",
        "mask_steps": "source_steps",
        "rotation_steps": "extended_steps",
        "mask_final_distance_m": "source_final_distance_m",
        "rotation_final_distance_m": "extended_final_distance_m",
    }
    paired["pairs"] = [{renamed.get(k, k): v for k, v in r.items()} for r in paired["pairs"]]
    paired = {
        k.replace("rotation_minus_mask", "extended_minus_source"): v for k, v in paired.items()
    }
    delta = results[new]["summary"]["success_rate"] - results[old]["summary"]["success_rate"]
    analysis = folder / "analysis"
    analysis.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for row, (name, base) in enumerate(bases.items()):
        for episode in results[name]["episodes"]:
            positions = read_json(
                base
                / "formal/episodes"
                / episode["episode_id"]
                / episode["attempt"]
                / "trajectory.json"
            )["positions_m"]
            color = "#238846" if episode["reason"] == "success" else "#ce4a49"
            for ax in axes[row]:
                ax.plot(
                    [p[0] for p in positions], [p[1] for p in positions], c=color, alpha=0.5, lw=0.9
                )
                ax.scatter(positions[-1][0], positions[-1][1], c=color, s=10)
        for col, ax in enumerate(axes[row]):
            ax.add_patch(Circle((0, 0), 0.5, fc="#ffd368", ec="black", zorder=10))
            ax.set(
                aspect="equal",
                xlabel="World X(m)",
                ylabel="World Y(m)",
                title="{} | {}/64 | {}".format(
                    name,
                    results[name]["summary"]["successes"],
                    "near home" if col else "all trajectories",
                ),
            )
            ax.grid(alpha=0.2)
            if col:
                ax.set_xlim(-3.5, 3.5)
                ax.set_ylim(-3.5, 3.5)
    fig.tight_layout()
    fig.savefig(analysis / "epoch_budget_trajectories.png", dpi=160)
    plt.close(fig)
    history = read_json(folder / "training/physical_history.json")["epochs"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, rows in ((axes[0], history), (axes[1], history[299:])):
        ax.plot(
            [r["epoch"] for r in rows],
            [r["train_component_mse_m2"] for r in rows],
            label="training",
        )
        ax.plot(
            [r["epoch"] for r in rows],
            [r["validation_component_mse_m2"] for r in rows],
            label="validation",
        )
        ax.axvline(300, c="gray", ls="--", label="resume at300")
        ax.set(xlabel="Total epoch", ylabel="Physical component MSE(m2)")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_yscale("log")
    axes[0].set_title("All inherited and new epochs")
    axes[1].set_title("Additional training segment")
    fig.tight_layout()
    fig.savefig(analysis / "epoch_budget_learning_curves.png", dpi=160)
    plt.close(fig)
    selected = completed["best_validation"]["epoch"]
    output = dict(
        updated=now(),
        fingerprint=frozen["fingerprint"],
        target_epochs=target,
        source_selected_epoch=previous["best_validation"]["epoch"],
        selected_epoch=selected,
        summaries={n: r["summary"] for n, r in results.items()},
        paired=paired,
        success_rate_difference_extended_minus_source=delta,
        training=completed,
        limits=frozen["protocol"]["limits"],
    )
    atomic_json(analysis / "comparison.json", output)
    lines = [
        "# 增强组300→{}轮续训与20m归巢对照".format(target),
        "",
        "从第300轮完整断点恢复，新增{}轮；模型、优化器、随机数、加载器和历史状态延续，" "其余训练及导航设置不变。".format(target - 300),
        "",
        "按完整验证集选择第{}轮权重，在导航前冻结。历史最佳第298轮也被保留为候选，" "没有通过导航结果反选模型。".format(selected),
        "",
        "| 训练预算 | 选中轮次 | 最佳验证MSE(m²) | 20m成功/总数 | 成功率 | 平均最终距离(m) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, record in ((old, previous), (new, completed)):
        s = results[name]["summary"]
        lines.append(
            "| {} | {} | {:.4f} | {}/64 | {:.2%} | {:.3f} |".format(
                name,
                record["best_validation"]["epoch"],
                record["best_validation"]["validation_component_mse_m2"],
                s["successes"],
                s["success_rate"],
                s["final_distance_m"]["mean"],
            )
        )
    lines += [
        "",
        "相对300轮基线的成功率差：{:+.2f}个百分点。".format(delta * 100),
        "",
        "若仍选中历史298轮，本次未得到更优验证权重，导航差异属于相同模型的重复测量，不能归因为新训练收益。"
        "初始半径仅20m，成功半径0.5m，最多50步；同场景、同训练种子，航向及视野相关且渲染存在波动。"
        "该比较增加了计算预算，不是等预算的方法比较，也不验证悬停或自主到家确认。",
        "",
        "[轨迹对照](analysis/epoch_budget_trajectories.png) · "
        "[训练曲线](analysis/epoch_budget_learning_curves.png)",
        "",
    ]
    (folder / "result_summary.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_json(
        folder / "completion.json",
        dict(
            status="completed",
            updated=now(),
            fingerprint=frozen["fingerprint"],
            epochs_total=target,
            epochs_added=target - 300,
            neural_trials=64,
            initial_radii_m=[20],
            selected_epoch=selected,
            summaries=output["summaries"],
            success_rate_difference_extended_minus_source=delta,
            report="result_summary.md",
            comparison_sha256=digest(analysis / "comparison.json"),
        ),
    )
    return output
