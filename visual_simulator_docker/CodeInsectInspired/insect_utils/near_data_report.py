"""Report local coverage and the revised, explicitly stated position separation."""

import math
from pathlib import Path

from insect_utils.experiment_config import digest
from insect_utils.experiment_state import atomic_json, now, read_json
from insect_utils.navigation_protocol import verify_navigation


def report_near_data(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    folder = Path(folder)
    frozen = verify_navigation(folder)
    data = read_json(folder / "dataset.json")
    records = data["records"]
    analysis = folder / "analysis"
    analysis.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    colors = {"train": "#3578bb", "validation": "#e49b22", "reserved_test": "#a855a5"}
    for split, color in colors.items():
        group = [p for p in frozen["positions"] if p["split"] == split]
        for ax in axes:
            ax.scatter(
                [p["xyz_m"][0] for p in group],
                [p["xyz_m"][1] for p in group],
                s=12 if split == "train" else 32,
                c=color,
                alpha=0.75,
                label="{}({})".format(split, len(group)),
            )
    for ax, extent in zip(axes, (5.5, 0.65)):
        ax.add_patch(Circle((0, 0), 0.5, fill=False, ec="black", ls="--"))
        ax.scatter(0, 0, c="black", marker="*", s=70, label="Home training anchor")
        ax.set(
            aspect="equal",
            xlabel="World X (m)",
            ylabel="World Y (m)",
            xlim=(-extent, extent),
            ylim=(-extent, extent),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_title("New local positions | antipodal symmetry")
    axes[1].set_title("Near-home detail | local holdout gap0.15m")
    fig.suptitle("193 new train +16 validation |18 reserved positions not captured")
    fig.tight_layout()
    fig.savefig(analysis / "near_position_splits.png", dpi=160)
    plt.close(fig)

    def counts(origin, split):
        points = {
            r["position_id"]: r
            for r in records
            if r["split"] == split and (origin is None or r["origin"] == origin)
        }
        radii = [math.hypot(r["x"], r["y"]) for r in points.values()]
        return [
            sum(r <= 0.5 for r in radii),
            sum(0.5 < r <= 2 for r in radii),
            sum(2 < r <= 5 for r in radii),
            sum(r > 5 for r in radii),
        ]

    coverage = dict(
        original_train=counts("original", "train"),
        combined_train=counts(None, "train"),
        original_validation=counts("original", "validation"),
        combined_validation=counts(None, "validation"),
    )
    atomic_json(analysis / "coverage.json", coverage)
    lines = [
        "# 第二步：补齐近家对称数据",
        "",
        "新采集193个训练位置及16个验证位置，每处8航向，共1672张图；另18个位置仅预留，未采图、未评估。"
        "新增训练位置包含原点，其他点成对关于原点对称，覆盖0–0.5m、0.5–2m及2–5m。",
        "",
        "保留原4096张训练图、320张验证图；组合后5640张训练图、448张验证图。"
        "5m内训练图比例约{:.2%}。采集使用原场景、40m高度、70°下视相机和稳定检查，"
        "此阶段没有训练或导航。".format(data["training_near_5m_fraction"]),
        "",
        "| 位置数 | 0–0.5m | 0.5–2m | 2–5m | >5m |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in coverage.items():
        lines.append("| {} | {} | {} | {} | {} |".format(name, *values))
    lines += [
        "",
        "[新增位置划分图](analysis/near_position_splits.png)",
        "",
        "原留出点与新训练点保持至少1m；新增局部验证/预留点与训练点保持至少0.15m。" "所有航向及后续增强按位置分组。这是同场景局部插值评估，不能沿用原1m隔离泛化的表述。",
        "",
        "193个新训练位置、16个新验证位置和18个预留位置在采图前按几何固定，未参考模型误差选点。" "104张诊断图片没有直接并入数据；原点训练锚点重新拍摄并明确分配为训练。",
        "",
        "下一阶段两组从同一个589轮断点开始，原数据组与补近家数据组采用相同更新次数。",
        "",
    ]
    (folder / "result_summary.md").write_text("\n".join(lines), encoding="utf-8")
    names = [
        "dataset.json",
        "capture/results.json",
        "capture/completion.json",
        "result_summary.md",
        "position_plan.json",
        "reserved_positions.json",
        "analysis/coverage.json",
        "analysis/near_position_splits.png",
    ]
    completed = dict(
        status="completed",
        updated=now(),
        fingerprint=frozen["fingerprint"],
        new_images=1672,
        train_images=5640,
        validation_images=448,
        reserved_positions_captured=False,
        new_training_updates=0,
        new_navigation_episodes=0,
        report="result_summary.md",
        files={n: digest(folder / n) for n in names},
    )
    atomic_json(folder / "completion.json", completed)
    atomic_json(folder / "progress.json", dict(completed, phase="completed"))
    return completed
