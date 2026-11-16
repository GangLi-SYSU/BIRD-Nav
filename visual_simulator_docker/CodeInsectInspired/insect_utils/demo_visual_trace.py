"""Visual-stage action endpoints and an independently exported trajectory figure."""
import csv
import json
import math
import os
from pathlib import Path

import numpy as np


class VisualTrace:
    def __init__(self):
        self.points = []

    def begin(self, position, yaw):
        self.points = [self._point(0, position, yaw)]

    @staticmethod
    def _point(step, position, yaw):
        return dict(
            step=int(step),
            x_m=float(position[0]),
            y_m=float(position[1]),
            z_m=float(position[2]),
            yaw_deg=float(yaw),
            distance_m=math.hypot(position[0], position[1]),
        )

    def completed_action(self, step, position, yaw):
        if not self.points or step != len(self.points):
            raise ValueError("Visual trace must contain one endpoint per completed action")
        self.points.append(self._point(step, position, yaw))

    def snapshot(self, episode, outcome, reason, landing=None):
        if not self.points:
            raise ValueError("Missing visual handoff position")
        return dict(
            episode=episode,
            outcome=outcome,
            reason=reason,
            positions=[dict(p) for p in self.points],
            landing=landing,
            point_definition="Step 0 is PI-to-vision handoff; subsequent points are "
            "executed neural action endpoints. Vertical descent is excluded.",
        )


def recover_trace(events, episode):
    """Reconstruct a completed user flight without introducing PI/motion samples."""
    trace = VisualTrace()
    final = None
    for event in events:
        if event.get("episode") != episode:
            continue
        state = event["state"]
        if event["event"] == "stage_transition" and event.get("to_stage") == "VISUAL_RETURN":
            trace.begin(state["position_m"], state["yaw_deg"])
        elif event["event"] == "neural_action_completed":
            if state["visual_steps"] == len(trace.points):
                trace.completed_action(state["visual_steps"], state["position_m"], state["yaw_deg"])
        elif event["event"] == "landing_complete":
            final = state
    if final is None or not trace.points:
        raise ValueError("No complete visual handoff and landing for this episode")
    if len(trace.points) - 1 != final["visual_steps"]:
        raise ValueError("Missing committed visual actions in event log")
    return trace.snapshot(
        episode, final["homing_outcome"], final["reason"], final["landing_measurement"]
    )


def export_trace(folder, snapshot, render=True):
    """CPU-only worker; never access Kit, USD or the interactive UI here."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Circle

    folder = Path(folder)
    prefix = folder / f"episode_{snapshot['episode']:03d}_visual_trajectory"
    points = snapshot["positions"]
    xy = np.array([[p["x_m"], p["y_m"]] for p in points])
    radial = np.linalg.norm(xy, axis=1)
    report = dict(
        snapshot,
        completed_actions=len(points) - 1,
        handoff_distance_m=float(radial[0]),
        final_distance_m=float(radial[-1]),
        closest_recorded_distance_m=float(radial.min()),
        horizontal_path_length_m=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
    )
    json_path = prefix.with_suffix(".json")
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2))
    os.replace(temporary, json_path)
    csv_path = prefix.with_suffix(".csv")
    with csv_path.with_suffix(".csv.tmp").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    os.replace(csv_path.with_suffix(".csv.tmp"), csv_path)
    if not render:
        return json_path

    figure = Figure(figsize=(12, 6), constrained_layout=True)
    FigureCanvasAgg(figure)
    context, detail = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.4]})
    final_label = "Final / landing" if snapshot["landing"] else "Visual endpoint"
    for ax in [context, detail]:
        for radius, color, style in [(2, "#29954a", "-"), (15, "#6c88aa", "--")]:
            ax.add_patch(
                Circle(
                    (0, 0),
                    radius,
                    fill=False,
                    color=color,
                    linestyle=style,
                    linewidth=1.3,
                    label=f"{radius} m radius",
                )
            )
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            "-o",
            color="#2878b5",
            lw=1.7,
            markersize=3,
            label="Executed visual path",
        )
        ax.scatter(
            [0], [0], marker="*", s=150, color="#d69b00", zorder=8, label="Home / flight origin"
        )
        ax.scatter(
            *xy[0], marker="^", s=85, color="#2878b5", zorder=7, label="Visual start (step 0)"
        )
        ax.scatter(*xy[-1], marker="X", s=85, color="#d54c81", zorder=9, label=final_label)
        if snapshot["landing"]:
            landing = np.asarray(snapshot["landing"]["landing_position_m"][:2])
            ax.scatter(
                *landing, marker="o", s=160, facecolors="none", edgecolors="#d54c81", zorder=8
            )
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.set_xlabel("East from home (m)")
        ax.set_ylabel("North from home (m)")
    context_limit = max(16.5, float(np.max(np.abs(xy))) * 1.15)
    context.set(
        xlim=(-context_limit, context_limit),
        ylim=(-context_limit, context_limit),
        title="15 m handoff area and 2 m success circle",
    )
    context.legend(loc="lower left", fontsize=7)
    padding = max(0.7, float(np.ptp(xy, axis=0).max()) * 0.15)
    low = np.minimum(xy.min(0), [-2.2, -2.2]) - padding
    high = np.maximum(xy.max(0), [2.2, 2.2]) + padding
    detail.set(
        xlim=(low[0], high[0]), ylim=(low[1], high[1]), title="Visual homing: local trajectory"
    )
    stride = max(1, math.ceil(len(points) / 18))
    label_spacing = 0.075 * float(np.linalg.norm(high - low))
    last_label = None
    last_label_xy = None
    for i, (x, y) in enumerate(xy):
        if i % stride == 0 or i == len(points) - 1:
            distance = (
                math.inf
                if last_label_xy is None
                else np.linalg.norm([x - last_label_xy[0], y - last_label_xy[1]])
            )
            if i != len(points) - 1 and distance < label_spacing:
                continue
            if i == len(points) - 1 and distance < label_spacing * 0.6 and last_label is not None:
                last_label.remove()
            last_label = detail.annotate(
                str(i), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8
            )
            last_label_xy = (x, y)
    for before, after in zip(xy[:-1], xy[1:]):
        if np.linalg.norm(after - before) > 1e-8:
            detail.annotate(
                "",
                xy=after,
                xytext=before,
                arrowprops=dict(arrowstyle="->", color="#2878b5", lw=1.2),
            )
    outcome = snapshot["outcome"]
    phase = "Landed" if snapshot["landing"] else "Visual stage complete; descending"
    figure.suptitle(
        f"Episode {snapshot['episode']} | Stage 2 visual homing | {outcome}\n"
        f"{len(points)-1} neural steps | Start {radial[0]:.2f} m -> "
        f"Final {radial[-1]:.2f} m | {phase}",
        fontsize=12,
    )
    if len(points) == 1:
        detail.text(
            0.02,
            0.03,
            "No neural movement: handoff already within success radius",
            transform=detail.transAxes,
            fontsize=7,
        )
    png = prefix.with_suffix(".png")
    figure.savefig(png.with_suffix(".png.tmp"), format="png", dpi=140)
    os.replace(png.with_suffix(".png.tmp"), png)
    return png
