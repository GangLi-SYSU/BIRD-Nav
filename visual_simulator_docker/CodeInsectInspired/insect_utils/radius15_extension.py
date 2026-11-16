"""Add64 trials at15m using the already selected compact-spiral model."""

import copy
import datetime
from pathlib import Path

from insect_utils.experiment_config import PROJECT, canonical_hash, digest
from insect_utils.experiment_state import atomic_json, now, read_json
from insect_utils.navigation_core import make_episodes, safe_segment
from insect_utils.navigation_protocol import verify_navigation
from run_navigation_experiment import verify_phase


PROTOCOL = "config/experiments/stage24_radius15_protocol.json"


def extend_protocol(source, radii):
    if radii != [15]:
        raise ValueError("This additional experiment only permits15m starts")
    if source["model"]["epoch"] != 484 or source["model"]["scale_m"] != 44:
        raise ValueError("Use the unchanged validation-selected epoch484 model")
    if source["protocol"]["formal"] != dict(
        radii_m=[10, 20], bearings_per_radius=16, initial_yaws_deg=[0, 90, 180, 270]
    ):
        raise ValueError("The source10m/20m comparison protocol changed")
    if source["protocol"]["controller"]["success_radius_m"] != 2:
        raise ValueError("Preserve the2m success condition")
    protocol = copy.deepcopy(source["protocol"])
    for phase in ("formal", "rendered_oracle"):
        protocol[phase]["radii_m"] = list(radii)
    return protocol


def prepare(folder):
    folder = Path(folder).resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    if (folder / "frozen.json").exists():
        return verify_navigation(folder)
    extension = read_json(PROJECT / PROTOCOL)
    parent = PROJECT / extension["source_navigation_run"]
    source = verify_navigation(parent)
    if source["fingerprint"] != extension["source_navigation_fingerprint"]:
        raise ValueError("Source navigation experiment identity changed")
    if extension["selected_epoch"] != source["model"]["epoch"]:
        raise ValueError("Requested model differs from the source selection")
    formal_receipt = read_json(parent / "formal/completion.json")
    if (
        formal_receipt["status"] != "completed"
        or formal_receipt["fingerprint"] != source["fingerprint"]
        or formal_receipt["episodes"] != 128
        or digest(parent / "formal/results.json") != formal_receipt["results_sha256"]
    ):
        raise ValueError("Historical10m/20m result is incomplete or changed")
    acceptance = verify_phase(parent, "model_check", source)
    protocol = extend_protocol(source, extension["formal_radii_m"])
    episodes = {
        phase: make_episodes(protocol[phase], protocol["home_camera_xyz_m"], phase)
        for phase in ("rendered_oracle", "formal")
    }
    if len(episodes["formal"]) != extension["formal_trials"] or extension["formal_trials"] != 64:
        raise ValueError("Expected exactly64 neural trials at15m")
    for group in episodes.values():
        for episode in group:
            p = episode["initial_position_m"]
            safe_segment(p, p, source["budget"])
    files = dict(source["files"])
    extras = [
        PROJECT / PROTOCOL,
        PROJECT / "insect_utils/radius15_extension.py",
        PROJECT / "run_radius15_navigation.py",
        PROJECT / "tests/test_stage24_radius15.py",
        parent / "frozen.json",
        parent / "formal/completion.json",
        parent / "formal/results.json",
        parent / "model_check/completion.json",
    ]
    extras += [parent / "model_check" / name for name in acceptance["artifacts"]]
    files.update({str(p.relative_to(PROJECT)): digest(p) for p in extras})
    value = dict(
        protocol=protocol,
        episodes=episodes,
        model=source["model"],
        budget=source["budget"],
        radius_extension=extension,
        files=files,
    )
    value["fingerprint"] = canonical_hash(value)
    atomic_json(folder / "frozen.json", value)
    atomic_json(folder / "protocol.json", protocol)
    atomic_json(folder / "episode_plan.json", episodes)
    atomic_json(
        folder / "model_check/completion.json",
        dict(
            status="completed",
            fingerprint=value["fingerprint"],
            updated=now(),
            images=0,
            reused_verified_images=acceptance["images"],
            additional_renderings=False,
            source_receipt=str((parent / "model_check/completion.json").relative_to(PROJECT)),
            source_receipt_sha256=digest(parent / "model_check/completion.json"),
            reason="Identical model/camera pipeline; source receipt and images are frozen inputs",
            artifacts={},
        ),
    )
    return value


def plot_live(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from insect_utils.spiral_narrow_report import save_figure

    folder = Path(folder)
    frozen = read_json(folder / "frozen.json")
    episodes = [read_json(p) for p in sorted((folder / "formal/episodes").glob("*/summary.json"))]
    if any(e["fingerprint"] != frozen["fingerprint"] for e in episodes):
        raise ValueError("Live trajectory plot mixes experiments")
    analysis = folder / "analysis"
    analysis.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for episode in episodes:
        trajectory = (
            folder
            / "formal/episodes"
            / episode["episode_id"]
            / episode["attempt"]
            / "trajectory.json"
        )
        ps = read_json(trajectory)["positions_m"]
        color = "#258a46" if episode["reason"] == "success" else "#cc4a49"
        for ax in axes:
            ax.plot([p[0] for p in ps], [p[1] for p in ps], c=color, alpha=0.5, lw=0.9)
            ax.scatter(ps[-1][0], ps[-1][1], c=color, s=15, zorder=5)
    for index, ax in enumerate(axes):
        ax.add_patch(Circle((0, 0), 2, fc="#fff2cc", ec="black", zorder=1))
        ax.scatter(0, 0, marker="+", c="black", zorder=6)
        ax.set(
            aspect="equal",
            xlabel="World X(m)",
            ylabel="World Y(m)",
            title="Near home" if index else "All15m trajectories",
        )
        ax.grid(alpha=0.2)
        if index:
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
    count = sum(e["reason"] == "success" for e in episodes)
    zone = datetime.timezone(datetime.timedelta(hours=8))
    stamp = datetime.datetime.now(zone).strftime("%Y-%m-%d %H:%M:%S CST")
    fig.suptitle(
        "Epoch484 |40m/30deg |15m starts |arrival<=2m |completed{}/64 successes{}\n{}".format(
            len(episodes), count, stamp
        )
    )
    fig.tight_layout()
    save_figure(fig, analysis / "r15_trajectories.png")
    plt.close(fig)
    (analysis / "live_results.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>15m navigation</title>'
        "<h1>15m navigation | epoch484 | success radius2m</h1>"
        "<p>Counts include completed formal episodes only. Refresh30 seconds.</p>"
        '<img id="plot" style="max-width:100%"><script>function refresh(){'
        'document.getElementById("plot").src="r15_trajectories.png?t="+Date.now()}'
        "refresh();setInterval(refresh,30000)</script>",
        encoding="utf-8",
    )


def report(folder):
    folder = Path(folder)
    frozen = verify_navigation(folder)
    for phase in ("geometry", "rendered_oracle", "model_check", "formal"):
        verify_phase(folder, phase, frozen)
    results = read_json(folder / "formal/results.json")
    if results["summary"]["episodes"] != 64 or set(results["summary"]["by_radius"]) != {"15"}:
        raise ValueError("Final report requires all64 formal15m trials")
    source = PROJECT / frozen["radius_extension"]["source_navigation_run"]
    old = read_json(source / "formal/results.json")
    plot_live(folder)
    lines = [
        "# 第484轮模型：15m归巢补充测试",
        "",
        "同一20m密集螺旋训练模型，40m高度、30°下视相机，384×384圆形输入，未重新训练。",
        "本次仅新增15m的16方位×4初始航向=64次正式神经导航，成功半径2m、最多50步，动态步长保持不变。",
        "原8张相机/部署一致性检查已核验并复用；另执行15m的4次渲染真值向量检查，不计为神经导航。",
        "",
        "| 初始半径 | 成功/次数 | 成功率 | 平均最终距离(m) |",
        "|---|---:|---:|---:|---:|",
    ]
    for radius in ("10", "15", "20"):
        row = results["summary"] if radius == "15" else old["summary"]["by_radius"][radius]
        label = radius + "m" + ("（本次）" if radius == "15" else "（历史结果，本次未重跑）")
        lines.append(
            "| {} | {}/64 | {:.2%} | {:.3f} |".format(
                label, row["successes"], row["success_rate"], row["final_distance_m"]["mean"]
            )
        )
    lines += [
        "",
        "[15m完整轨迹与近家放大图](analysis/r15_trajectories.png)",
        "",
        "成功距离为每步执行后真值水平距离≤2m；首次进入即停止。相同场景、训练种子及相关起始方位，仅为开发测试，未评估2m内精度或实机自主到家识别。",
        "",
    ]
    (folder / "result_summary.md").write_text("\n".join(lines), encoding="utf-8")
    result = dict(
        status="completed",
        updated=now(),
        fingerprint=frozen["fingerprint"],
        selected_epoch=frozen["model"]["epoch"],
        neural_trials=64,
        initial_radii_m=[15],
        success_radius_m=2,
        summary=results["summary"],
        historical_radii_retested=False,
        results_sha256=digest(folder / "formal/results.json"),
        report="result_summary.md",
    )
    atomic_json(folder / "completion.json", result)
    atomic_json(folder / "progress.json", dict(result, phase="completed"))
    atomic_json(folder / "pipeline_progress.json", dict(result, phase="completed"))
    return result
