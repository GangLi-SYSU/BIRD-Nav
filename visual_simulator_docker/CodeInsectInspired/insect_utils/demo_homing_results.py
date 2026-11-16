"""One final window: two selected PI panels and one visual-stage trajectory."""
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from insect_utils.demo_pi_comparison import _return_pair
from insect_utils.demo_visual_trace import export_trace


PANEL_IDS = ["outbound_odometry", "pi_full_return", "visual_homing"]


def export_homing_results(folder, episode, pi_trace, visual):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Circle

    folder = Path(folder)
    prefix = folder / f"episode_{episode:03d}_homing_results"
    samples = pi_trace["outbound"]
    if samples:
        last = samples[-1]
        true = np.asarray(last["true_xy_m"])
        mag = np.asarray(last["magnetic_estimate_xy_m"])
        gyro = np.asarray(last["gyro_estimate_xy_m"])
        bias = last["gyro_bias_rad"]
    else:
        true = mag = gyro = np.zeros(2)
        bias = 0.0
    _, prediction = _return_pair(
        true[None, :],
        mag[None, :],
        gyro[None, :],
        np.array([bias]),
        pi_trace["sigma"],
        pi_trace["gyro_deg_per_sqrt_s"],
        np.random.SeedSequence([pi_trace["seed"], 881]),
        keep_paths=True,
    )
    # Keep the existing action CSV/JSON, without producing another standalone figure.
    export_trace(folder, visual, render=False)
    payload = dict(
        episode=episode,
        panel_ids=PANEL_IDS,
        pi_recorded=pi_trace,
        visual=visual,
        forecast_magnetic_path_m=prediction[:, 0, 0].tolist(),
        forecast_gyro_path_m=prediction[:, 1, 0].tolist(),
        note="Forecasts use fixed 0.5 s updates. Actual PI and visual paths are "
        "recorded flight positions. No Monte Carlo panel is generated.",
    )
    temporary = prefix.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    os.replace(temporary, prefix.with_suffix(".json"))
    rows = []
    for phase, items in [("outbound", samples), ("pi_return", pi_trace["actual_pi_return"])]:
        for i, sample in enumerate(items):
            estimate = sample.get("gyro_estimate_xy_m", [None, None])
            rows.append(
                dict(
                    phase=phase,
                    sample=i,
                    true_x_m=sample["true_xy_m"][0],
                    true_y_m=sample["true_xy_m"][1],
                    magnetic_est_x_m=sample["magnetic_estimate_xy_m"][0],
                    magnetic_est_y_m=sample["magnetic_estimate_xy_m"][1],
                    gyro_est_x_m=estimate[0],
                    gyro_est_y_m=estimate[1],
                )
            )
    if rows:
        raw = folder / f"episode_{episode:03d}_pi_trace.csv"
        with raw.with_suffix(".csv.tmp").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(raw.with_suffix(".csv.tmp"), raw)

    figure = Figure(figsize=(13, 8), constrained_layout=True)
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(2, 2, width_ratios=[1, 1.35])
    outbound = figure.add_subplot(grid[0, 0])
    return_ax = figure.add_subplot(grid[1, 0])
    visual_ax = figure.add_subplot(grid[:, 1])
    colors = ["#555555", "#137fb4", "#e58524"]
    for key, label, color in zip(
        ["true_xy_m", "magnetic_estimate_xy_m", "gyro_estimate_xy_m"],
        ["True outbound", "Odometry + ideal compass", "Odometry / gyro only"],
        colors,
    ):
        points = np.asarray([[0.0, 0.0]] + [s[key] for s in samples])
        outbound.plot(*points.T, color=color, lw=1.5, label=label)
    outbound.set_title("Stage 1: outbound odometry", fontsize=11)
    for i, label, color in [
        (0, "Compass return forecast", colors[1]),
        (1, "Gyro-only return forecast", colors[2]),
    ]:
        points = prediction[:, i, 0]
        return_ax.plot(*points.T, "--", color=color, lw=1.3, label=label)
        return_ax.scatter(*points[-1], marker="x", s=35, color=color)
    if pi_trace["actual_pi_return"]:
        points = np.array([s["true_xy_m"] for s in pi_trace["actual_pi_return"]])
        return_ax.plot(*points.T, lw=1.8, color="#103762", label="Actual compass-assisted return")
        return_ax.scatter(*points[-1], s=35, color="#103762", zorder=6)
    return_ax.add_patch(
        Circle((0, 0), 15, fill=False, color="#29954a", lw=1.2, label="15 m visual range")
    )
    return_ax.set_title("Stage 1: full PI homing comparison")

    points = visual["positions"]
    xy = np.array([[p["x_m"], p["y_m"]] for p in points])
    for radius, color, style in [(2, "#29954a", "-"), (15, "#6c88aa", "--")]:
        visual_ax.add_patch(
            Circle(
                (0, 0),
                radius,
                fill=False,
                color=color,
                linestyle=style,
                lw=1.2,
                label=f"{radius} m radius",
            )
        )
    visual_ax.plot(*xy.T, "-o", lw=1.7, ms=3, color="#2878b5", label="Executed visual path")
    visual_ax.scatter(
        *xy[0], marker="^", s=90, color="#2878b5", zorder=7, label="Visual start (step 0)"
    )
    visual_ax.scatter(
        *xy[-1], marker="X", s=90, color="#d54c81", zorder=9, label="Visual endpoint / landing"
    )
    if visual["landing"]:
        landing = visual["landing"]["landing_position_m"][:2]
        visual_ax.scatter(*landing, s=165, facecolors="none", edgecolors="#d54c81", zorder=8)
    for start, end in zip(xy[:-1], xy[1:]):
        if np.linalg.norm(end - start) > 1e-8:
            visual_ax.annotate(
                "", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#2878b5", lw=1.1)
            )
    pad = max(0.7, float(np.ptp(xy, axis=0).max()) * 0.15)
    low = np.minimum(xy.min(0), [-2.3, -2.3]) - pad
    high = np.maximum(xy.max(0), [2.3, 2.3]) + pad
    spacing = float(np.linalg.norm(high - low)) * 0.08
    stride = max(1, math.ceil(len(points) / 14))
    last_xy = None
    last_label = None
    for i, point in enumerate(xy):
        final = i == len(points) - 1
        if i % stride and not final:
            continue
        distance = math.inf if last_xy is None else float(np.linalg.norm(point - last_xy))
        if not final and distance < spacing:
            continue
        if final and distance < spacing * 0.6 and last_label is not None:
            last_label.remove()
        last_label = visual_ax.annotate(
            str(i), point, xytext=(5, 5), textcoords="offset points", fontsize=8
        )
        last_xy = point
    visual_ax.set(
        xlim=(low[0], high[0]),
        ylim=(low[1], high[1]),
        title=f"Stage 2: visual homing | {len(points)-1} neural steps",
    )
    if len(points) == 1:
        visual_ax.text(
            0.03,
            0.04,
            "No visual movement: already within 2 m at handoff",
            transform=visual_ax.transAxes,
            fontsize=8,
        )
    for ax in [outbound, return_ax, visual_ax]:
        ax.scatter(0, 0, marker="*", s=100, color="#c79800", zorder=8, label="Home")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.22)
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.legend(loc="best", fontsize=7)
    outbound.set_aspect("equal", adjustable="datalim")
    return_ax.set_aspect("equal", adjustable="datalim")
    radial = np.linalg.norm(xy, axis=1)
    figure.suptitle(
        f"Episode {episode}: two-stage homing | {visual['outcome']}\n"
        f"Visual start {radial[0]:.2f} m -> endpoint {radial[-1]:.2f} m | "
        f"Distance noise {pi_trace['sigma']} m/sqrt(m)",
        fontsize=12,
    )
    png = prefix.with_suffix(".png")
    figure.savefig(png.with_suffix(".png.tmp"), format="png", dpi=140)
    os.replace(png.with_suffix(".png.tmp"), png)
    return png
