"""Extend the frozen589-epoch navigation model to the requested10m and5m radii."""

import copy
from pathlib import Path

from insect_utils.experiment_config import PROJECT, canonical_hash, digest
from insect_utils.experiment_state import atomic_json, now, read_json
from insect_utils.navigation_core import make_episodes, safe_segment
from insect_utils.navigation_protocol import verify_navigation
from run_navigation_experiment import verify_phase


PROTOCOL = "config/experiments/stage17_radius_navigation_protocol.json"


def extended_protocol(source, radii):
    if radii != [10, 5]:
        raise ValueError("Only the requested10m and5m starting radii are permitted")
    if source["model"]["epoch"] != 589 or source["model"]["scale_m"] != 44:
        raise ValueError("Use the unchanged validation-selected589-epoch model")
    expected = dict(radii_m=[20], bearings_per_radius=16, initial_yaws_deg=[0, 90, 180, 270])
    if source["protocol"]["formal"] != expected:
        raise ValueError("The historical64-condition navigation source changed")
    protocol = copy.deepcopy(source["protocol"])
    for phase in ("formal", "rendered_oracle"):
        protocol[phase]["radii_m"] = list(radii)
    return protocol


def build_plan():
    extension = read_json(PROJECT / PROTOCOL)
    source_folder = PROJECT / extension["source_navigation_run"]
    source = verify_navigation(source_folder)
    if source["fingerprint"] != extension["source_navigation_fingerprint"]:
        raise ValueError("The selected model's source navigation experiment changed")
    if extension["selected_epoch"] != source["model"]["epoch"]:
        raise ValueError("Selected model identity changed")
    # The historical results are read only for the eventual reference table.
    receipt = read_json(source_folder / "formal/completion.json")
    if (
        receipt["status"] != "completed"
        or receipt["fingerprint"] != source["fingerprint"]
        or receipt["episodes"] != 64
        or digest(source_folder / "formal/results.json") != receipt["results_sha256"]
    ):
        raise ValueError("Historical20m reference result changed")
    protocol = extended_protocol(source, extension["formal_radii_m"])
    episodes = {
        phase: make_episodes(protocol[phase], protocol["home_camera_xyz_m"], phase)
        for phase in ("rendered_oracle", "formal")
    }
    for group in episodes.values():
        for episode in group:
            safe_segment(
                episode["initial_position_m"], episode["initial_position_m"], source["budget"]
            )
    extras = [
        PROJECT / PROTOCOL,
        PROJECT / "insect_utils/radius_extension.py",
        PROJECT / "run_radius_navigation.py",
        PROJECT / "tests/test_stage17_radius_navigation.py",
        source_folder / "frozen.json",
        source_folder / "formal/results.json",
        source_folder / "formal/completion.json",
    ]
    files = dict(source["files"])
    files.update({str(p.relative_to(PROJECT)): digest(p) for p in extras})
    value = dict(
        protocol=protocol,
        episodes=episodes,
        model=source["model"],
        budget=source["budget"],
        files=files,
        radius_extension=extension,
    )
    value["fingerprint"] = canonical_hash(value)
    return value


def prepare(folder):
    folder = Path(folder).resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    value = build_plan()
    path = folder / "frozen.json"
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Radius experiment is already frozen; use a new run directory")
    else:
        atomic_json(path, value)
        atomic_json(folder / "protocol.json", value["protocol"])
        atomic_json(folder / "episode_plan.json", value["episodes"])
    return value


def report(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    folder = Path(folder)
    frozen = verify_navigation(folder)
    for phase in ("geometry", "rendered_oracle", "model_check", "formal"):
        verify_phase(folder, phase, frozen)
    results = read_json(folder / "formal/results.json")
    radii = frozen["protocol"]["formal"]["radii_m"]
    if results["summary"]["episodes"] != 128 or set(results["summary"]["by_radius"]) != {"5", "10"}:
        raise ValueError("Exactly64 new trials at each requested radius must be complete")
    source = PROJECT / frozen["radius_extension"]["source_navigation_run"]
    old = read_json(source / "formal/results.json")
    analysis = folder / "analysis"
    analysis.mkdir(exist_ok=True)
    home = frozen["protocol"]["home_camera_xyz_m"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    diagnostics = {}
    for row, radius in enumerate(radii):
        selected = [e for e in results["episodes"] if e["radius_m"] == radius]
        if len(selected) != 64:
            raise ValueError("A radius has missing or duplicate trials")
        for episode in selected:
            path = (
                folder
                / "formal/episodes"
                / episode["episode_id"]
                / episode["attempt"]
                / "trajectory.json"
            )
            positions = read_json(path)["positions_m"]
            color = "#238846" if episode["reason"] == "success" else "#ce4a49"
            for ax in axes[row]:
                ax.plot(
                    [p[0] for p in positions], [p[1] for p in positions], c=color, alpha=0.5, lw=0.9
                )
                ax.scatter(positions[-1][0], positions[-1][1], c=color, s=12)
        count = sum(e["reason"] == "success" for e in selected)
        for col, ax in enumerate(axes[row]):
            ax.add_patch(Circle(home[:2], 0.5, fc="#ffd368", ec="black", zorder=10))
            ax.scatter(home[0], home[1], c="black", marker="+", zorder=11)
            ax.set(
                aspect="equal",
                xlabel="World X (m)",
                ylabel="World Y (m)",
                title="R={}m | {}/64 success | {}".format(
                    radius, count, "near home" if col else "all trajectories"
                ),
            )
            ax.grid(alpha=0.2)
            if col:
                ax.set_xlim(home[0] - 3.5, home[0] + 3.5)
                ax.set_ylim(home[1] - 3.5, home[1] + 3.5)
        diagnostics[str(radius)] = {
            "entered_{}m".format(d): sum(e["minimum_endpoint_distance_m"] <= d for e in selected)
            for d in (0.5, 1, 2)
        }
    fig.suptitle("Epoch589 | green=success, red=failure | success radius0.5m | max50 steps")
    fig.tight_layout()
    figure = analysis / "r10_r5_trajectories.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    comparison = dict(
        fingerprint=frozen["fingerprint"],
        updated=now(),
        model=frozen["model"],
        new_results=results["summary"],
        near_home=diagnostics,
        historical20m=old["summary"],
        historical20m_rerun=False,
        limits=frozen["radius_extension"]["limits"],
    )
    atomic_json(analysis / "radius_comparison.json", comparison)
    lines = [
        "# 第589轮模型：10m与5m归巢实验",
        "",
        "同一验证最优模型、40m飞行高度、70°下视相机、384×384圆形输入。" "每个半径16方位×4航向=64次；没有新增训练或修改控制器。",
        "",
        "动态步长min(0.1+0.4×预测距离/3,2.5)m，成功半径0.5m，最多50步。" "每步完成后用真值水平距离计成功，不用真值控制或缩短末步。",
        "",
        "| 起始半径 | 成功/总数 | 成功率 | 平均最终距离(m) | 平均步数 |",
        "|---|---:|---:|---:|---:|",
    ]
    summaries = [
        (
            str(radius) + "m",
            results["summary"]["by_radius"][str(radius)],
            [e for e in results["episodes"] if e["radius_m"] == radius],
        )
        for radius in radii
    ]
    summaries.append(("20m（历史参考，本次未重跑）", old["summary"], old["episodes"]))
    for label, summary, episodes in summaries:
        lines.append(
            "| {} | {}/64 | {:.2%} | {:.3f} | {:.2f} |".format(
                label,
                summary["successes"],
                summary["success_rate"],
                summary["final_distance_m"]["mean"],
                sum(e["steps"] for e in episodes) / 64,
            )
        )
    lines += [
        "",
        "[完整轨迹与近家放大图](analysis/r10_r5_trajectories.png)",
        "",
        "工程流程先做纯几何检查和5m/10m真值向量验收；部署一致性验收沿用原运行器的" "8个20m静态拍照位姿，这些不计入归巢试验。正式神经导航仅运行本次128次。",
        "",
        "同一场景与训练种子，视野及航向相关；结果用于开发对照，未验证动力学、悬停或自主到家确认。",
        "",
    ]
    (folder / "result_summary.md").write_text("\n".join(lines), encoding="utf-8")
    completion = dict(
        status="completed",
        updated=now(),
        fingerprint=frozen["fingerprint"],
        selected_epoch=589,
        formal_episodes=128,
        initial_radii_m=radii,
        summary=results["summary"],
        near_home=diagnostics,
        formal_results_sha256=digest(folder / "formal/results.json"),
        comparison_sha256=digest(analysis / "radius_comparison.json"),
        trajectory_plot_sha256=digest(figure),
        report="result_summary.md",
    )
    atomic_json(folder / "completion.json", completion)
    atomic_json(folder / "progress.json", dict(completion, phase="completed"))
    return completion
