"""Matched20m outcomes and trajectories without treating correlated frames as trials."""

from pathlib import Path
import statistics

from insect_utils.experiment_config import PROJECT, digest
from insect_utils.experiment_state import atomic_json, now, read_json
from insect_utils.rotation_protocol import verify_comparison


def episode_key(row):
    return row["radius_m"], row["bearing_deg"], row["initial_yaw_deg"]


def matched_outcomes(first, second):
    a = {episode_key(r): r for r in first}
    b = {episode_key(r): r for r in second}
    if len(a) != len(first) or len(b) != len(second) or a.keys() != b.keys():
        raise ValueError("Comparison trials do not have identical unique initial conditions")
    counts = dict(both_success=0, only_mask_success=0, only_rotation_success=0, both_fail=0)
    rows, common_step_differences, common_path_differences = [], [], []
    for key in sorted(a):
        x, y = a[key], b[key]
        if x["initial_position_m"] != y["initial_position_m"]:
            raise ValueError("Matched navigation positions differ")
        sa, sb = x["reason"] == "success", y["reason"] == "success"
        category = (
            "both_success"
            if sa and sb
            else "only_mask_success"
            if sa
            else "only_rotation_success"
            if sb
            else "both_fail"
        )
        counts[category] += 1
        rows.append(
            dict(
                radius_m=key[0],
                bearing_deg=key[1],
                initial_yaw_deg=key[2],
                mask_success=sa,
                rotation_success=sb,
                mask_steps=x["steps"],
                rotation_steps=y["steps"],
                mask_final_distance_m=x["final_distance_m"],
                rotation_final_distance_m=y["final_distance_m"],
            )
        )
        if sa and sb:
            common_step_differences.append(y["steps"] - x["steps"])
            common_path_differences.append(y["path_length_m"] - x["path_length_m"])
    return dict(
        counts=counts,
        pairs=rows,
        both_success_mean_step_difference_rotation_minus_mask=(
            statistics.mean(common_step_differences) if common_step_differences else None
        ),
        both_success_mean_path_difference_m_rotation_minus_mask=(
            statistics.mean(common_path_differences) if common_path_differences else None
        ),
    )


def report_comparison(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    folder = Path(folder)
    declared = verify_comparison(folder)
    training = PROJECT / declared["protocol"]["training_run"]
    arms = declared["protocol"]["arms"]
    results = {arm: read_json(folder / "arms" / arm / "formal/results.json") for arm in arms}
    contracts = {arm: read_json(training / "arms" / arm / "completion.json") for arm in arms}
    paired = matched_outcomes(results[arms[0]]["episodes"], results[arms[1]]["episodes"])
    summaries = {arm: results[arm]["summary"] for arm in arms}
    if any(s["episodes"] != 64 or set(s["by_radius"]) != {"20"} for s in summaries.values()):
        raise ValueError("Comparison report requires64 completed20m trials per arm")
    difference = summaries[arms[1]]["success_rate"] - summaries[arms[0]]["success_rate"]
    figures = folder / "analysis"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for row, arm in enumerate(arms):
        for episode in results[arm]["episodes"]:
            path = (
                folder
                / "arms"
                / arm
                / "formal/episodes"
                / episode["episode_id"]
                / episode["attempt"]
                / "trajectory.json"
            )
            trace = read_json(path)["positions_m"]
            color = "#238846" if episode["reason"] == "success" else "#ce4a49"
            for ax in axes[row, :2]:
                ax.plot(
                    [p[0] for p in trace], [p[1] for p in trace], color=color, alpha=0.5, lw=0.9
                )
                ax.scatter(trace[-1][0], trace[-1][1], c=color, s=10)
        for col in (0, 1):
            ax = axes[row, col]
            ax.add_patch(Circle((0, 0), 0.5, facecolor="#ffd368", edgecolor="black", zorder=10))
            ax.scatter(0, 0, marker="+", c="black", zorder=11)
            ax.set(aspect="equal", xlabel="World X (m)", ylabel="World Y (m)")
            ax.grid(alpha=0.2)
            if col == 1:
                ax.set_xlim(-3, 3)
                ax.set_ylim(-3, 3)
            ax.set_title(
                "{} | {} | {}/64 success".format(
                    arm, "near home" if col else "all trajectories", summaries[arm]["successes"]
                )
            )
        yaws = summaries[arm]["by_initial_yaw"]
        keys = sorted(yaws, key=float)
        axes[row, 2].bar(keys, [100 * yaws[k]["success_rate"] for k in keys], color="#4385be")
        axes[row, 2].set(
            ylim=(0, 105),
            xlabel="Initial yaw (deg)",
            ylabel="Success (%)",
            title=arm + " | 16 trials per initial yaw",
        )
    fig.suptitle("20m-only paired comparison | green=success, red=failure | success radius0.5m")
    fig.tight_layout()
    fig.savefig(figures / "r20_comparison_trajectories.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for arm in arms:
        history = read_json(training / "arms" / arm / "physical_history.json")["epochs"]
        for ax, key in zip(axes, ("train_component_mse_m2", "validation_component_mse_m2")):
            ax.plot([r["epoch"] for r in history], [r[key] for r in history], label=arm)
            ax.set(xlabel="Epoch", ylabel="Physical component MSE (m2)")
            ax.grid(alpha=0.2)
            ax.legend()
    axes[0].set_title("Training loss (rotation arm uses augmented images)")
    axes[1].set_title("Same fixed validation images for both arms")
    fig.tight_layout()
    fig.savefig(figures / "training_curves.png", dpi=160)
    plt.close(fig)
    output = dict(
        updated=now(),
        fingerprint=declared["fingerprint"],
        summaries=summaries,
        paired=paired,
        success_rate_difference_rotation_minus_mask=difference,
        training={arm: contracts[arm]["best_validation"] for arm in arms},
        limits=declared["protocol"]["limits"],
        static_test_images_evaluated=0,
    )
    atomic_json(figures / "comparison.json", output)
    conclusion = (
        "本轮20m测试观察到旋转增强组成功率提高"
        if difference > 0
        else "本轮20m测试未观察到旋转增强提高成功率"
        if difference == 0
        else "本轮20m测试中旋转增强组成功率更低"
    )
    lines = [
        "# 圆形掩膜与旋转增强：20m归巢对照",
        "",
        conclusion + "。",
        "",
        "两组均为原384×384网络、512训练位置/4096图、150轮/76,800次更新、相同初始化；"
        "仅随机旋转及对应标签变换不同。均按完整验证集米制MSE选择各自最佳轮次后冻结。",
        "",
        "每组20m圆周16个方位×4初始航向=64次；没有进行其他半径归巢或额外静态测试集评估。" "成功条件为执行落点距起点≤0.5m，最多50步；沿用原动态步长和机头转向。",
        "",
        "| 模式 | 验证最佳轮次 | 验证MSE(m²) | 20m成功/总数 | 成功率 | 最终距离均值(m) | 步数均值 |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        s, c = summaries[arm], contracts[arm]["best_validation"]
        lines.append(
            "| {} | {} | {:.3f} | {}/64 | {:.2%} | {:.3f} | {:.2f} |".format(
                arm,
                c["epoch"],
                c["validation_component_mse_m2"],
                s["successes"],
                s["success_rate"],
                s["final_distance_m"]["mean"],
                s["steps"]["mean"],
            )
        )
    counts = paired["counts"]
    lines += [
        "",
        "旋转增强组相对无旋转组的成功率差：{:+.2f}个百分点。".format(100 * difference),
        "",
        "同一起点与航向配对：两组均成功{}次，仅无旋转成功{}次，仅旋转成功{}次，两组均失败{}次。".format(
            counts["both_success"],
            counts["only_mask_success"],
            counts["only_rotation_success"],
            counts["both_fail"],
        ),
        "",
        "成功后由真值评测器立即终止，最终距离不能解释为自主悬停误差。"
        "同场景、单训练种子、航向/视野相关且闭环渲染观测可能变化；当前收益仅是本轮20m的初步证据。"
        "未增加步数、放宽成功圈、用真值修正策略或根据导航结果重选模型。",
        "",
        "[20m轨迹与近家放大图](analysis/r20_comparison_trajectories.png) · "
        "[训练曲线](analysis/training_curves.png)",
        "",
        "原始数据：`arms/<模式>/formal/episodes/`；逐步照片、稳定性凭据、预测和执行轨迹均保存。",
        "",
    ]
    (folder / "result_summary.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_json(
        folder / "completion.json",
        dict(
            status="completed",
            updated=now(),
            fingerprint=declared["fingerprint"],
            neural_trials=128,
            initial_radii_m=[20],
            summaries=summaries,
            success_rate_difference_rotation_minus_mask=difference,
            report="result_summary.md",
            comparison_sha256=digest(figures / "comparison.json"),
        ),
    )
    return output
