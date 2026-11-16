"""Compress the original147-position spiral to20m without changing other settings."""

import math
from pathlib import Path

from insect_utils.coverage_plan import verify_freeze
from insect_utils.experiment_config import PROJECT, canonical_hash, digest
from insect_utils.experiment_state import atomic_json, read_json
from insect_utils.scene_geometry import check_positions_and_segments
from insect_utils.spiral_narrow_protocol import build_plan, scene_budget


PROTOCOL = "config/experiments/stage23_compact_spiral_protocol.json"
REFERENCE_PROTOCOL = "config/experiments/stage22_spiral_h40_fov30_protocol.json"


def build_compact_plan(protocol, permutation=None):
    reference = read_json(PROJECT / REFERENCE_PROTOCOL)
    for key in (
        "seed",
        "scene",
        "height_agl_m",
        "camera",
        "training",
        "preprocessing",
        "initialization",
        "selection",
        "navigation",
    ):
        if protocol[key] != reference[key]:
            raise ValueError("Only the sampling range may change; changed: " + key)
    learning = protocol["learning"]
    for key in reference["learning"]:
        if key not in ("shape_params", "learning_radius_m", "collection_constraint"):
            if learning[key] != reference["learning"][key]:
                raise ValueError("Unchanged sampling setting was modified: " + key)
    n, count, b = learning["shape_params"]
    if [n, count] != [5, 150] or learning["learning_radius_m"] != 20:
        raise ValueError("Expected the same5-turn147-position spiral within20m")
    if not math.isclose(b, 20 / (2 * math.pi * n), rel_tol=0, abs_tol=1e-12):
        raise ValueError("Spiral scale does not produce the requested20m radius")
    plan = build_plan(protocol, permutation=permutation)
    original = build_plan(reference, permutation=permutation)
    scale = b / reference["learning"]["shape_params"][2]
    for new, old in zip(plan["positions"], original["positions"]):
        if any(new[k] != old[k] for k in ("position_id", "split", "yaws_deg", "trajectory_index")):
            raise ValueError("Compression changed sample membership or heading coverage")
        if any(abs(new["xyz_m"][i] - old["xyz_m"][i] * scale) > 1e-9 for i in (0, 1)):
            raise ValueError("Positions are not an exact uniform compression of the original path")
        if math.hypot(*new["xyz_m"][:2]) > 20 + 1e-9:
            raise ValueError("A compact training position is outside20m")
    if not math.isclose(plan["audit"]["maximum_radius_m"], 20, abs_tol=1e-9):
        raise ValueError("The compact spiral does not reach its declared outer radius")
    plan["radial_band_counts"] = {
        split: {
            "{}-{}m".format(low, high): sum(
                low < math.hypot(*p["xyz_m"][:2]) <= high + 1e-9
                for p in plan["positions"]
                if p["split"] == split
            )
            for low, high in ((0, 5), (5, 10), (10, 15), (15, 20))
        }
        for split in ("train", "validation", "test")
    }
    plan["compression"] = dict(
        reference_maximum_radius_m=original["audit"]["maximum_radius_m"],
        maximum_radius_m=20.0,
        xy_scale=scale,
        nominal_turns=5,
        nominal_radial_pitch_m=2 * math.pi * b,
        nominal_turns_between10_and20m=10 / (2 * math.pi * b),
        reference_radial_pitch_m=2 * math.pi * reference["learning"]["shape_params"][2],
        position_count_unchanged=True,
        headings_and_split_groups_unchanged=True,
        target_normalization_scale_unchanged_m=protocol["training"]["scale_m"],
    )
    return plan


def prepare(folder):
    folder = Path(folder).resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    if (folder / "frozen.json").exists():
        return verify_freeze(folder)
    protocol = read_json(PROJECT / PROTOCOL)
    plan = build_compact_plan(protocol)
    budget = scene_budget(protocol)
    check_positions_and_segments([p["xyz_m"] for p in plan["positions"]], budget)
    paths = list(PROJECT.glob("*.py")) + list((PROJECT / "insect_utils").glob("*.py"))
    paths += list((PROJECT / protocol["scene"]).parent.rglob("*"))
    paths += [
        PROJECT / PROTOCOL,
        PROJECT / REFERENCE_PROTOCOL,
        PROJECT / "tests/test_stage23_compact_spiral.py",
    ]
    value = dict(
        protocol=protocol,
        plan=plan,
        budget=budget,
        supersedes="runs/stage22_spiral_h40_fov30_01",
        files={str(p.relative_to(PROJECT)): digest(p) for p in sorted(paths) if p.is_file()},
    )
    value["fingerprint"] = canonical_hash(value)
    atomic_json(folder / "frozen.json", value)
    atomic_json(folder / "protocol.json", protocol)
    atomic_json(folder / "plan.json", plan)
    return value


def comparison_plot(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    folder = Path(folder)
    frozen = read_json(folder / "frozen.json")
    compact = frozen["plan"]
    scale = compact["compression"]["xy_scale"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, divisor, title in zip(axes, (scale, 1), ("Previous: radius43.98m", "New: radius20m")):
        rows = compact["positions"]
        xs = [p["xyz_m"][0] / divisor for p in rows]
        ys = [p["xyz_m"][1] / divisor for p in rows]
        ax.plot(xs, ys, color="#777777", lw=0.7)
        for split, color in (("train", "#348abd"), ("validation", "#ff8c00"), ("test", "#9932cc")):
            chosen = [i for i, p in enumerate(rows) if p["split"] == split]
            ax.scatter([xs[i] for i in chosen], [ys[i] for i in chosen], c=color, s=16, label=split)
        for radius in (10, 20):
            ax.add_patch(Circle((0, 0), radius, fill=False, ec="black", ls="--", lw=0.8))
        ax.add_patch(Circle((0, 0), 2, fc="#ffd368", ec="black"))
        ax.set(
            aspect="equal",
            xlim=(-46, 46),
            ylim=(-46, 46),
            xlabel="World X(m)",
            ylabel="World Y(m)",
            title=title,
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Same147 positions x8 headings |5-turn generator | dashed radii10m and20m")
    fig.tight_layout()
    from insect_utils.spiral_narrow_report import save_figure

    save_figure(fig, folder / "analysis/compact_vs_previous.png")
    plt.close(fig)
