"""Compare the fixed150-epoch augmentation result with the continued training result."""

from pathlib import Path

from insect_utils.continuation_protocol import verify_continuation
from insect_utils.experiment_config import PROJECT, digest
from insect_utils.experiment_state import atomic_json, now, read_json
from insect_utils.rotation_report import matched_outcomes


def report_continuation(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    folder = Path(folder)
    frozen = verify_continuation(folder)
    original = PROJECT / frozen["protocol"]["source_navigation_run"]
    sources = {"epoch150": original, "continued300": folder / "navigation"}
    results = {name: read_json(base / "formal/results.json") for name, base in sources.items()}
    if any(
        r["summary"]["episodes"] != 64 or set(r["summary"]["by_radius"]) != {"20"}
        for r in results.values()
    ):
        raise ValueError("Only the same64 trials at20m may enter this report")
    trained = read_json(folder / "training/completion.json")
    selected = trained["best_validation"]["epoch"]
    paired = matched_outcomes(results["epoch150"]["episodes"], results["continued300"]["episodes"])
    counts = paired.pop("counts")
    paired["counts"] = dict(
        both_success=counts["both_success"],
        only150_success=counts["only_mask_success"],
        only_continued_success=counts["only_rotation_success"],
        both_fail=counts["both_fail"],
    )
    # Rename the shared helper's generic arm labels for this epoch-budget comparison.
    paired["pairs"] = [
        dict(
            radius_m=r["radius_m"],
            bearing_deg=r["bearing_deg"],
            initial_yaw_deg=r["initial_yaw_deg"],
            epoch150_success=r["mask_success"],
            continued_success=r["rotation_success"],
            epoch150_steps=r["mask_steps"],
            continued_steps=r["rotation_steps"],
            epoch150_final_distance_m=r["mask_final_distance_m"],
            continued_final_distance_m=r["rotation_final_distance_m"],
        )
        for r in paired["pairs"]
    ]
    for old, new in (
        (
            "both_success_mean_step_difference_rotation_minus_mask",
            "both_success_mean_step_difference_continued_minus150",
        ),
        (
            "both_success_mean_path_difference_m_rotation_minus_mask",
            "both_success_mean_path_difference_m_continued_minus150",
        ),
    ):
        paired[new] = paired.pop(old)
    delta = (
        results["continued300"]["summary"]["success_rate"]
        - results["epoch150"]["summary"]["success_rate"]
    )
    analysis = folder / "analysis"
    analysis.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for index, (name, base) in enumerate(sources.items()):
        for episode in results[name]["episodes"]:
            trace = read_json(
                base
                / "formal/episodes"
                / episode["episode_id"]
                / episode["attempt"]
                / "trajectory.json"
            )["positions_m"]
            color = "#238846" if episode["reason"] == "success" else "#ce4a49"
            for ax in axes[index]:
                ax.plot(
                    [p[0] for p in trace], [p[1] for p in trace], color=color, alpha=0.5, lw=0.9
                )
                ax.scatter(trace[-1][0], trace[-1][1], color=color, s=10)
        for col, ax in enumerate(axes[index]):
            ax.add_patch(Circle((0, 0), 0.5, facecolor="#ffd368", edgecolor="black", zorder=10))
            ax.scatter(0, 0, marker="+", c="black", zorder=11)
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
    fig.suptitle(
        "20m comparison | continued validation-selected epoch{} | success radius0.5m".format(
            selected
        )
    )
    fig.tight_layout()
    fig.savefig(analysis / "r20_epoch_comparison.png", dpi=160)
    plt.close(fig)
    history = read_json(folder / "training/physical_history.json")["epochs"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(
        [r["epoch"] for r in history],
        [r["train_component_mse_m2"] for r in history],
        label="training",
    )
    axes[0].plot(
        [r["epoch"] for r in history],
        [r["validation_component_mse_m2"] for r in history],
        label="validation",
    )
    axes[0].set_yscale("log")
    axes[0].set(title="Same augmentation method | inherited1..150, continued151..300")
    axes[1].plot(
        [r["epoch"] for r in history[149:]],
        [r["validation_component_mse_m2"] for r in history[149:]],
        label="validation",
    )
    axes[1].set(title="Continuation validation trend")
    for ax in axes:
        ax.axvline(150, c="gray", ls="--", label="checkpoint import at150")
        ax.set(xlabel="Epoch", ylabel="Physical component MSE(m2)")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(analysis / "continuation_learning_curves.png", dpi=160)
    plt.close(fig)
    output = dict(
        updated=now(),
        fingerprint=frozen["fingerprint"],
        selected_epoch=selected,
        summaries={k: v["summary"] for k, v in results.items()},
        paired=paired,
        success_rate_difference_continued_minus150=delta,
        training=trained,
        limits=frozen["protocol"]["limits"],
        static_test_images_evaluated=0,
    )
    atomic_json(analysis / "comparison.json", output)
    lines = [
        "# 增强组150→300轮续训与20m归巢对照",
        "",
        "使用原增强组第150轮完整断点继续训练至总计300轮；恢复模型、AdamW、随机数和加载器状态。"
        "数据、学习率0.0005、旋转顺序、圆形掩膜、网络和控制器均保持原设置。新增150轮/76,800次更新，累计153,600次。",
        "",
        "按照1–300轮相同完整验证集的米制MSE选模，本次选中第{}轮；模型在导航前冻结。".format(selected),
        "",
        "| 模型 | 选中轮次 | 20m成功/总数 | 成功率 | 所有试验最终距离均值(m) | 步数均值 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, epoch in (("epoch150", 150), ("continued300", selected)):
        s = results[name]["summary"]
        lines.append(
            "| {} | {} | {}/64 | {:.2%} | {:.3f} | {:.2f} |".format(
                name,
                epoch,
                s["successes"],
                s["success_rate"],
                s["final_distance_m"]["mean"],
                s["steps"]["mean"],
            )
        )
    lines += [
        "",
        "续训后成功率差：{:+.2f}个百分点。".format(delta * 100),
        "",
        "本次验证最佳MSE为{:.4f}m²，最终300轮MSE为{:.4f}m²。".format(
            trained["best_validation"]["validation_component_mse_m2"],
            trained["final_validation"]["validation_component_mse_m2"],
        ),
        "",
        "配对结果：两次均成功{}次，仅150轮成功{}次，仅续训模型成功{}次，两次均失败{}次。".format(
            paired["counts"]["both_success"],
            paired["counts"]["only150_success"],
            paired["counts"]["only_continued_success"],
            paired["counts"]["both_fail"],
        ),
        "",
        "若最优轮次仍为150，则没有因导航结果改选其他权重；重复渲染/闭环本身可能使结果变化。"
        "这是同一增强策略增加训练预算的比较，不能当作与无增强组等计算量的对照。"
        "仅评估20m、同场景、同训练种子、0.5m落点成功圈及50步；未验证自主到家确认、动力学或悬停。",
        "",
        "[20m轨迹对照](analysis/r20_epoch_comparison.png) · "
        "[续训曲线](analysis/continuation_learning_curves.png)",
        "",
        "- `training/checkpoint_import.json`：完整训练状态移交一致性核验。",
        "- `training/physical_history.json`：保留原150轮并追加151–300轮。",
        "- `training/selection_frozen_before_navigation.json`：导航前的验证集选模。",
        "- `navigation/formal/episodes/`：64次逐步图像、相机凭据和轨迹。",
        "",
    ]
    (folder / "result_summary.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_json(
        folder / "completion.json",
        dict(
            status="completed",
            updated=now(),
            fingerprint=frozen["fingerprint"],
            epochs_total=300,
            epochs_added=150,
            neural_trials=64,
            initial_radii_m=[20],
            selected_epoch=selected,
            success_rate_difference_continued_minus150=delta,
            summaries=output["summaries"],
            report="result_summary.md",
            comparison_sha256=digest(analysis / "comparison.json"),
        ),
    )
    return output
