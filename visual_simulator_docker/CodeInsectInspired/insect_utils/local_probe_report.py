"""Visualize local prediction fields and distinguish heading variation from bias."""

from pathlib import Path

from insect_utils.experiment_config import PROJECT, digest
from insect_utils.experiment_state import atomic_json, now
from insect_utils.local_probe_runtime import verify_capture
from insect_utils.navigation_protocol import verify_navigation


def report_probe(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import numpy as np
    from PIL import Image

    folder = Path(folder)
    frozen = verify_navigation(folder)
    result = verify_capture(folder)
    rows, summary = result["rows"], result["summary"]
    home = frozen["protocol"]["home_camera_xyz_m"]
    headings = frozen["spec"]["headings_deg"]
    analysis = folder / "analysis"
    analysis.mkdir(exist_ok=True)
    extent = max(
        2.5,
        max(
            abs(r[k][i] - home[i])
            for r in rows
            for k in ("position_m", "predicted_home_world_xy_m")
            for i in (0, 1)
        )
        + 0.4,
    )

    def decorate(ax):
        ax.add_patch(Circle(home[:2], 0.5, fill=False, ec="gray", ls="--", lw=0.9))
        ax.scatter(home[0], home[1], marker="+", c="black", s=45, zorder=10)
        ax.set(aspect="equal", xlabel="World X (m)", ylabel="World Y (m)")
        ax.grid(alpha=0.2)

    colors = plt.cm.twilight(np.linspace(0, 1, len(headings), endpoint=False))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    for ax, yaw, color in zip(axes.flat, headings, colors):
        group = [r for r in rows if r["planned_body_yaw_deg"] == yaw]
        xy = np.array([r["position_m"][:2] for r in group])
        vectors = np.array([r["prediction_world_m"] for r in group])
        ax.quiver(
            xy[:, 0],
            xy[:, 1],
            home[0] - xy[:, 0],
            home[1] - xy[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="gray",
            alpha=0.25,
        )
        ax.quiver(
            xy[:, 0],
            xy[:, 1],
            vectors[:, 0],
            vectors[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=color,
            alpha=0.85,
        )
        ax.scatter(xy[:, 0], xy[:, 1], s=8, c=[color])
        decorate(ax)
        ax.set_xlim(home[0] - extent, home[0] + extent)
        ax.set_ylim(home[1] - extent, home[1] + extent)
        ax.set_title(
            "Yaw{}deg | mean vector error{:.2f}m".format(
                yaw, sum(r["vector_error_m"] for r in group) / len(group)
            )
        )
    fig.suptitle(
        "Epoch589 local fields | colored=prediction, gray=true home vector | arrows in meters"
    )
    fig.tight_layout()
    fig.savefig(analysis / "local_vector_fields.png", dpi=160)
    plt.close(fig)

    positions = summary["per_position"]
    xy = np.array([p["position_m"][:2] for p in positions])
    mean = np.array([p["predicted_home_mean_xy_m"] for p in positions])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].quiver(
        xy[:, 0],
        xy[:, 1],
        mean[:, 0] - xy[:, 0],
        mean[:, 1] - xy[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#3578bb",
    )
    axes[0].scatter(xy[:, 0], xy[:, 1], s=12, c="#3578bb")
    axes[0].set_title("Mean world prediction across8 headings")
    plot = axes[1].scatter(
        xy[:, 0],
        xy[:, 1],
        s=65,
        c=[p["heading_dispersion_rms_m"] for p in positions],
        cmap="viridis",
    )
    fig.colorbar(plot, ax=axes[1], label="Heading dispersion RMS (m)", shrink=0.75)
    axes[1].set_title("Same position, different headings")
    home_rows = sorted(
        [r for r in rows if r["true_distance_m"] <= 1e-8], key=lambda r: r["planned_body_yaw_deg"]
    )
    for row, color in zip(home_rows, colors):
        v = row["prediction_world_m"]
        axes[2].quiver(
            home[0],
            home[1],
            *v,
            angles="xy",
            scale_units="xy",
            scale=1,
            color=color,
            label="{:g}deg: {:.2f}m".format(
                row["planned_body_yaw_deg"], row["predicted_distance_m"]
            ),
        )
    axes[2].set_title("True home: correct output is zero")
    axes[2].legend(fontsize=8, loc="best", ncol=2)
    for ax in axes:
        decorate(ax)
        ax.set_xlim(home[0] - extent, home[0] + extent)
        ax.set_ylim(home[1] - extent, home[1] + extent)
    fig.suptitle(
        "Fixed-position diagnostic | original model/camera/controller | no training or navigation"
    )
    fig.tight_layout()
    fig.savefig(analysis / "local_summary.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(12, 7))
    for ax, row in zip(axes.flat, home_rows):
        with Image.open(PROJECT / row["image"]) as image:
            ax.imshow(np.asarray(image))
        ax.axis("off")
        ax.set_title(
            "Yaw{:g} | predicted distance{:.2f}m".format(
                row["planned_body_yaw_deg"], row["predicted_distance_m"]
            ),
            fontsize=10,
        )
    fig.suptitle("Fresh real-home observations | 40m AGL,70deg HFOV | target=(0,0)")
    fig.tight_layout()
    fig.savefig(analysis / "home_views.png", dpi=150)
    plt.close(fig)

    historical = []
    for old in frozen["historical_home"]:
        fresh = next(
            r
            for r in home_rows
            if abs((r["planned_body_yaw_deg"] - old["yaw_deg"] + 180) % 360 - 180) < 1e-5
        )
        with Image.open(PROJECT / old["image"]) as image:
            first = np.array(image, dtype=np.float32)
        with Image.open(PROJECT / fresh["image"]) as image:
            second = np.array(image, dtype=np.float32)
        historical.append(
            dict(
                yaw_deg=old["yaw_deg"],
                rgb_mae_255=float(np.abs(first - second).mean()),
                old_prediction_norm_m=old["vector_error_m"],
                fresh_prediction_norm_m=fresh["predicted_distance_m"],
                limit="Historical/fresh capture comparison, not a training intervention",
            )
        )
    atomic_json(
        analysis / "diagnostics.json", dict(summary=summary, historical_home_comparison=historical)
    )
    lines = [
        "# 第一步：原点周围多航向诊断",
        "",
        "设计：{}，{}个固定位置×8航向={}张新图。第589轮模型、40m高度、70°下视视场、"
        "384²圆形输入及成像稳定策略保持原样。没有训练或执行闭环导航。".format(
            frozen["design"], summary["positions"], summary["images"]
        ),
        "",
        "真实原点8个航向的预测长度：均值{:.3f}m，最大{:.3f}m；正确输出均应为零。".format(
            summary["home_output_norm_m"]["mean"], summary["home_output_norm_m"]["maximum"]
        ),
        "",
        "| 航向(°) | 预测机体系向量(m) | 反推的家世界坐标(m) | 预测长度(m) |",
        "|---:|---|---|---:|",
    ]
    for row in home_rows:
        lines.append(
            "| {:g} | ({:.3f},{:.3f}) | ({:.3f},{:.3f}) | {:.3f} |".format(
                row["planned_body_yaw_deg"],
                *row["prediction_body_m"],
                *row["predicted_home_world_xy_m"],
                row["predicted_distance_m"],
            )
        )
    lines += [
        "",
        "[8航向向量场](analysis/local_vector_fields.png) · [局部汇总图](analysis/local_summary.png) · "
        "[原点相机图像](analysis/home_views.png)",
        "",
        "箭头表示预测的世界坐标向量，箭头终点是模型反推的家；灰色箭头为真值，仅用于诊断。"
        "同位置航向离散度衡量反推的家随朝向变化的程度。"
        "原点的方向角无定义，不用它计算方向正确率。",
        "",
        "新的原点0°/90°照片还与已有工程照片比较像素差及输出差，见analysis/diagnostics.json。"
        "这些是开发探针，不是独立盲测，也没有自动并入训练集。"
        "下一步采样密度、近家占比、位置留出规则和训练预算仍需先确定。",
        "",
    ]
    (folder / "result_summary.md").write_text("\n".join(lines), encoding="utf-8")
    names = [
        "capture/results.json",
        "capture/completion.json",
        "result_summary.md",
        "analysis/diagnostics.json",
        "analysis/local_vector_fields.png",
        "analysis/local_summary.png",
        "analysis/home_views.png",
    ]
    completed = dict(
        status="completed",
        updated=now(),
        fingerprint=frozen["fingerprint"],
        design=frozen["design"],
        images=summary["images"],
        positions=summary["positions"],
        selected_epoch=589,
        new_training_updates=0,
        new_navigation_episodes=0,
        summary=summary,
        report="result_summary.md",
        files={name: digest(folder / name) for name in names},
    )
    atomic_json(folder / "completion.json", completed)
    atomic_json(folder / "progress.json", dict(completed, phase="completed"))
    return completed
