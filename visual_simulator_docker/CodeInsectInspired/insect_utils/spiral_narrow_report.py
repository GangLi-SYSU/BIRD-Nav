"""Atomic live loss/trajectory plots and final reports for the spiral30deg run."""

import datetime
import os
from pathlib import Path
import uuid

from insect_utils.experiment_config import PROJECT, digest
from insect_utils.experiment_state import atomic_json, now, read_json


def timestamp():
    zone = datetime.timezone(datetime.timedelta(hours=8))
    return datetime.datetime.now(zone).strftime("%Y-%m-%d %H:%M:%S CST")


def save_figure(fig, path):
    temporary = path.with_name(path.stem + ".tmp-" + uuid.uuid4().hex)
    fig.savefig(temporary, format="png", dpi=150)
    os.replace(str(temporary), str(path))


def plot_live(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    folder = Path(folder)
    analysis = folder / "analysis"
    analysis.mkdir(exist_ok=True)
    engine = folder / "training/engine_scaled_units/training_history.json"
    if engine.exists():
        rows = read_json(engine)["epochs"]
        if rows:
            epochs = [r["epoch"] for r in rows]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for ax in axes:
                for key, label in (("train", "Train"), ("validation", "Validation")):
                    values = [r[key + "_component_mse_m2"] * 44 ** 2 for r in rows]
                    ax.plot(epochs, values, label=label, lw=1)
                ax.set(xlabel="Epoch", ylabel="Physical component MSE (m2)")
                ax.grid(alpha=0.2)
                ax.legend()
            axes[1].set_xlim(max(1, epochs[-1] - 99), max(2, epochs[-1]))
            axes[1].set_title("Last100 epochs (same units)")
            fig.suptitle(
                "Spiral only |40m/30deg | epoch{}/600 | {}".format(epochs[-1], timestamp())
            )
            fig.tight_layout()
            save_figure(fig, analysis / "training_live.png")
            plt.close(fig)
    nav = folder / "navigation"
    summaries = sorted((nav / "formal/episodes").glob("*/summary.json"))
    if not summaries:
        return
    frozen = read_json(nav / "frozen.json")
    episodes = [read_json(p) for p in summaries]
    if any(e["fingerprint"] != frozen["fingerprint"] for e in episodes):
        raise ValueError("Live plot contains another experiment")
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for row, radius in enumerate((10, 20)):
        selected = [e for e in episodes if e["radius_m"] == radius]
        successes = sum(e["reason"] == "success" for e in selected)
        for episode in selected:
            trajectory = nav / "formal/episodes" / episode["episode_id"] / episode["attempt"]
            positions = read_json(trajectory / "trajectory.json")["positions_m"]
            color = "#258a46" if episode["reason"] == "success" else "#cc4a49"
            for ax in axes[row]:
                ax.plot([p[0] for p in positions], [p[1] for p in positions], c=color, alpha=0.5)
                ax.scatter(positions[-1][0], positions[-1][1], c=color, s=12)
        for col, ax in enumerate(axes[row]):
            ax.add_patch(Circle((0, 0), 2, fc="#ffd368", ec="black", zorder=10))
            ax.scatter(0, 0, c="black", marker="+", zorder=11)
            ax.set(
                aspect="equal",
                xlabel="World X(m)",
                ylabel="World Y(m)",
                title="R={}m | completed{}/64 | successes{} | {}".format(
                    radius, len(selected), successes, "near home" if col else "all trajectories"
                ),
            )
            ax.grid(alpha=0.2)
            if col:
                ax.set_xlim(-4, 4)
                ax.set_ylim(-4, 4)
    fig.suptitle("40m/30deg | arrival<=2m | {}".format(timestamp()))
    fig.tight_layout()
    save_figure(fig, analysis / "navigation_live.png")
    plt.close(fig)


def plan_figure(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    folder = Path(folder)
    frozen = read_json(folder / "frozen.json")
    positions = frozen["plan"]["positions"]
    analysis = folder / "analysis"
    analysis.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([p["xyz_m"][0] for p in positions], [p["xyz_m"][1] for p in positions], "k-", lw=0.7)
    for split, color in (("train", "#348abd"), ("validation", "#ff8c00"), ("test", "#9932cc")):
        points = [p["xyz_m"] for p in positions if p["split"] == split]
        ax.scatter([p[0] for p in points], [p[1] for p in points], s=20, c=color, label=split)
    for radius in (10, 20):
        ax.add_patch(Circle((0, 0), radius, fill=False, linestyle="--", ec="#888888"))
    ax.add_patch(Circle((0, 0), 2, fc="#ffd368", ec="black", label="2m success area"))
    ax.set(aspect="equal", xlabel="World X(m)", ylabel="World Y(m)")
    ax.set_title("Original147-position spiral |8 headings/position |40m/30deg")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, analysis / "spiral_plan.png")
    plt.close(fig)
    html = """<!doctype html><meta charset="utf-8"><title>Spiral30 experiment</title>
<style>body{font-family:sans-serif;max-width:1200px;margin:24px auto}img{max-width:100%}</style>
<h1>40m /30deg | Spiral only | Navigation success radius2m</h1>
<p>Training600 epochs. Navigation10m and20m,64 trials each. Updates every30 seconds.</p>
<p>Partial plots include only completed episodes; counts are displayed separately.</p>
<h2>Training and validation loss</h2><img data-source="training_live.png">
<h2>Navigation trajectories</h2><img data-source="navigation_live.png">
<h2>Spiral acquisition plan</h2><img src="spiral_plan.png">
<script>function refresh(){document.querySelectorAll('[data-source]').forEach(
e=>e.src=e.dataset.source+'?t='+Date.now())}refresh();setInterval(refresh,30000)</script>
"""
    (analysis / "live_results.html").write_text(html, encoding="utf-8")


def preview_training_images(folder):
    from PIL import Image, ImageDraw

    folder = Path(folder)
    records = read_json(folder / "dataset.json")["records"]
    candidates = [r for r in records if r["split"] == "train" and r["heading_index"] == 0]
    chosen = [candidates[int(i * (len(candidates) - 1) / 7)] for i in range(8)]
    canvas = Image.new("RGB", (4 * 256, 2 * 286), "white")
    draw = ImageDraw.Draw(canvas)
    for index, record in enumerate(chosen):
        with Image.open(PROJECT / record["image"]) as opened:
            image = opened.resize((256, 256))
        x, y = index % 4 * 256, index // 4 * 286
        canvas.paste(image, (x, y))
        draw.text(
            (x + 4, y + 258),
            "x={:.1f} y={:.1f} yaw={:.0f}".format(record["x"], record["y"], record["yaw"]),
            fill="black",
        )
    canvas.save(folder / "analysis/training_camera_preview.png")


def report_final(folder):
    from insect_utils.coverage_plan import verify_freeze
    from insect_utils.navigation_protocol import verify_navigation
    from run_navigation_experiment import verify_phase

    folder = Path(folder)
    frozen = verify_freeze(folder)
    nav = verify_navigation(folder / "navigation")
    for phase in ("geometry", "rendered_oracle", "model_check", "formal"):
        verify_phase(folder / "navigation", phase, nav)
    results = read_json(folder / "navigation/formal/results.json")
    summary = results["summary"]
    if summary["episodes"] != 128 or set(summary["by_radius"]) != {"10", "20"}:
        raise ValueError("Final report requires all128 authorized neural trials")
    trained = read_json(folder / "training/completion.json")
    plot_live(folder)
    lines = [
        "# 40m高度、30°视场角、2m成功半径：螺旋数据训练与归巢测试",
        "",
        "仅沿原始bee螺旋轨迹147个位置采集，每位置8个航向，共1176张新图。",
        "按位置划分：训练133/1064张，验证6/48张，保留测试8/64张。没有二维或近家补采，没有使用旧照片、旧模型或工程检查权重。",
        "",
        "384×384、固定圆形掩膜、训练随机旋转及同步标签变换；10628参数网络，目标/44，部署输出恢复米。",
        "AdamW，学习率0.0005，batch8，600轮共79800次更新。按验证误差选择第{}轮，验证分量MSE={:.4f}m²；未用导航或保留测试选择模型。".format(
            trained["best_validation"]["epoch"],
            trained["best_validation"]["validation_component_mse_m2"],
        ),
        "",
        "正式测试每半径16个方位×4个航向=64次。每步端点真值水平距离≤2m即成功，最多50步；动态步长min(0.1+预测距离×0.4/3,2.5)m。图像出现归巢点不触发成功。",
        "",
        "| 起始半径 | 成功/次数 | 成功率 | 平均最终距离(m) |",
        "|---|---:|---:|---:|",
    ]
    for radius in ("10", "20"):
        row = summary["by_radius"][radius]
        if row["episodes"] != 64:
            raise ValueError("Missing trials at a requested radius")
        lines.append(
            "| {}m | {}/64 | {:.2%} | {:.3f} |".format(
                radius, row["successes"], row["success_rate"], row["final_distance_m"]["mean"]
            )
        )
    lines += [
        "",
        "[损失曲线](analysis/training_live.png) · [归巢轨迹](analysis/navigation_live.png) · "
        "[训练照片预览](analysis/training_camera_preview.png)",
        "",
        "限制：" + frozen["protocol"]["limits"],
        "",
        "2m成功判定使用仿真真值，仅用于仿真停止/评分，尚未构成实机自主到家识别器。",
        "",
    ]
    (folder / "result_summary.md").write_text("\n".join(lines), encoding="utf-8")
    completion = dict(
        status="completed",
        updated=now(),
        fingerprint=frozen["fingerprint"],
        epochs=600,
        selected_epoch=trained["best_validation"]["epoch"],
        neural_trials=128,
        success_radius_m=2,
        initial_radii_m=[10, 20],
        summary=summary,
        report="result_summary.md",
        results_sha256=digest(folder / "navigation/formal/results.json"),
    )
    atomic_json(folder / "completion.json", completion)
    return completion
