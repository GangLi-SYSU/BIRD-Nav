"""Refresh closest-recorded-position plots without changing the running experiment."""

import argparse
import csv
import math
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insect_utils.experiment_config import PROJECT, digest
from insect_utils.experiment_state import atomic_json, now, read_json
from insect_utils.fixed50_report import draw_rings, endpoint_color, live_paths
from insect_utils.navigation_protocol import verify_navigation
from insect_utils.spiral_narrow_report import save_figure, timestamp
from insect_utils.training_state import distribution


def closest_recorded(positions, home):
    if not positions:
        raise ValueError("Cannot mark an empty trajectory")
    distances = [math.hypot(p[0] - home[0], p[1] - home[1]) for p in positions]
    step = min(range(len(distances)), key=distances.__getitem__)
    return dict(step=step, position_m=positions[step], distance_m=distances[step])


def render(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    frozen = read_json(folder / "frozen.json")
    home = frozen["protocol"]["home_camera_xyz_m"]
    rows, paths = live_paths(folder)
    summaries = {r["episode_id"]: r for r in rows}
    records = []
    finished = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    extent = max([20] + [max(abs(p[0]), abs(p[1])) * 1.08 for t in paths for p in t["positions"]])
    for path in paths:
        ps = path["positions"]
        point = closest_recorded(ps, home)
        full = path["complete"]
        color = endpoint_color(point["distance_m"]) if full else "#999999"
        mark = point["position_m"]
        summary = summaries.get(path["episode_id"])
        if full and not math.isclose(
            point["distance_m"], summary["minimum_endpoint_distance_m"], abs_tol=1e-8
        ):
            raise ValueError("Closest-point marker differs from the committed trajectory minimum")
        records.append(
            dict(
                episode_id=path["episode_id"],
                completed50=full,
                recorded_actions=len(ps) - 1,
                closest_step=point["step"],
                closest_x_m=mark[0],
                closest_y_m=mark[1],
                closest_distance_m=point["distance_m"],
                action50_distance_m=summary["final_distance_m"] if full else None,
            )
        )
        for ax in (axes[0, 0], axes[0, 1]):
            ax.plot(
                [p[0] for p in ps],
                [p[1] for p in ps],
                c=color,
                lw=0.8,
                alpha=0.4 if full else 0.6,
                ls="-" if full else ":",
                zorder=1,
            )
            ax.scatter(
                mark[0],
                mark[1],
                s=23 if full else 26,
                marker="o" if full else "D",
                facecolors=color if full else "none",
                edgecolors=color,
                zorder=6,
            )
        distances = [math.hypot(p[0] - home[0], p[1] - home[1]) for p in ps]
        axes[1, 1].plot(
            range(len(ps)),
            distances,
            c=color,
            alpha=0.15 if full else 0.5,
            lw=0.7,
            ls="-" if full else ":",
        )
        axes[1, 1].scatter(point["step"], point["distance_m"], c=color, s=9, zorder=4)
        if full:
            axes[1, 0].scatter(mark[0], mark[1], c=color, s=27, zorder=6)
            finished.append(distances)
    rings = frozen["fixed50"]["overview_rings_m"] + list(range(30, int(extent // 10) * 10 + 1, 10))
    draw_rings(axes[0, 0], rings)
    axes[0, 0].set(
        xlim=(-extent, extent),
        ylim=(-extent, extent),
        title="All trajectories |dots=closest recorded positions",
    )
    for ax in (axes[0, 1], axes[1, 0]):
        draw_rings(ax, frozen["fixed50"]["near_home_rings_m"])
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
    axes[0, 1].set_title("Near home |no action50 endpoint markers")
    outside = sum(
        r["completed50"] and (abs(r["closest_x_m"]) > 6 or abs(r["closest_y_m"]) > 6)
        for r in records
    )
    axes[1, 0].set_title("Closest recorded positions |{} outside this zoom".format(outside))
    if finished:
        data = np.asarray(finished)
        steps = np.arange(51)
        axes[1, 1].plot(
            steps, np.median(data, axis=0), c="black", lw=1.7, label="Median of completed trials"
        )
        axes[1, 1].fill_between(
            steps,
            np.percentile(data, 10, axis=0),
            np.percentile(data, 90, axis=0),
            color="#6b9dc2",
            alpha=0.18,
            label="P10-P90",
        )
    axes[1, 1].axhline(2, c="#997422", ls="--", label="2m reference (no stop)")
    axes[1, 1].set(
        xlim=(0, 50),
        xlabel="Executed action",
        ylabel="Distance to home(m)",
        title="Distance vs action |dots mark each trajectory minimum",
    )
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.15)
    fig.suptitle(
        "15m starts |epoch484 |40m/30deg |50 actions\nClosest recorded point markers |completed{}/64 |{}".format(
            len(finished), timestamp()
        )
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.text(
        0.5,
        0.015,
        "Closest-distance colors: <=0.5m green; <=1m teal; <=2m blue; <=3m orange; >3m red. Hollow diamonds=active minima so far.",
        ha="center",
        fontsize=8,
    )
    analysis = folder / "analysis"
    save_figure(fig, analysis / "closest_trajectories.png")
    plt.close(fig)
    with (analysis / "closest_points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "episode_id",
                "completed50",
                "recorded_actions",
                "closest_step",
                "closest_x_m",
                "closest_y_m",
                "closest_distance_m",
                "action50_distance_m",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    atomic_json(
        analysis / "closest_points.json",
        dict(
            updated=now(),
            simulation_fingerprint=frozen["fingerprint"],
            completed50=len(finished),
            marker_definition="Minimum horizontal distance among recorded positions including step0; earliest step breaks ties",
            closest_distance_m=distribution(
                [r["closest_distance_m"] for r in records if r["completed50"]]
            ),
            records=records,
            action50_statistics_modified=False,
        ),
    )
    html = (
        '<!doctype html><meta charset="utf-8"><title>Closest trajectory positions</title>'
        "<h1>15m /50 actions: closest recorded positions</h1>"
        "<p>Filled dots mark the closest recorded point over all50 actions. Hollow diamonds "
        "mark the closest point so far in active trials. Distance circles are labelled in meters. "
        "The action50 endpoint statistics remain available separately. Closest steps: "
        '<a href="closest_points.csv">CSV</a>.</p><img id="plot" style="max-width:100%">'
        '<script>function refresh(){document.getElementById("plot").src='
        '"closest_trajectories.png?t="+Date.now()}refresh();setInterval(refresh,30000)</script>'
    )
    (analysis / "live_closest.html").write_text(html, encoding="utf-8")
    return len(finished), html


def publish_final_caption(folder, html):
    # The simulation has completed, so its original plot writer is no longer active.
    # Publish the closest-point view to the original image path as well as the new one.
    analysis = folder / "analysis"
    primary = analysis / "fixed50_trajectories.png"
    backup = analysis / "fixed50_endpoints_before_closest_revision.png"
    if primary.exists() and not backup.exists():
        shutil.copyfile(primary, backup)
    temporary = primary.with_name(primary.name + ".tmp-" + uuid.uuid4().hex)
    shutil.copyfile(analysis / "closest_trajectories.png", temporary)
    os.replace(str(temporary), str(primary))
    report = folder / "result_summary.md"
    if report.exists():
        original = report.read_text(encoding="utf-8")
        backup = folder / "analysis/report_before_closest_marker_revision.md"
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        text = original.replace(
            "analysis/fixed50_trajectories.png", "analysis/closest_trajectories.png"
        )
        text = text.replace("完成后的彩色圆点为第50步端点", "完成后的彩色圆点为全程已记录位置中距归巢点最近的一点")
        text = text.replace("灰色虚线为尚未跑完的轨迹", "灰色虚线为尚未跑完的轨迹，空心菱形为目前最近点")
        note = (
            "图中圆点按每条轨迹的最近已记录位置绘制，最近步数见[最近点CSV](analysis/closest_points.csv)；下文第50步终点误差统计保持原定义。\n\n"
        )
        if note not in text:
            header, rest = text.split("\n", 1)
            text = header + "\n\n" + note + rest.lstrip("\n")
        report.write_text(text, encoding="utf-8")
    (folder / "analysis/live_results.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    frozen = verify_navigation(folder)
    atomic_json(
        folder / "analysis/closest_view_manifest.json",
        dict(
            started=now(),
            source_fingerprint=frozen["fingerprint"],
            script_sha256=digest(Path(__file__)),
            marker_definition="Closest recorded position, including initial pose; earliest step on ties",
            simulation_or_action50_metrics_changed=False,
        ),
    )
    deadline = time.monotonic() + 12 * 3600
    while True:
        complete, html = render(folder)
        print("CLOSEST_PLOT completed50={}/64".format(complete), flush=True)
        if (folder / "completion.json").exists():
            publish_final_caption(folder, html)
            atomic_json(
                folder / "analysis/closest_view_completion.json",
                dict(status="completed", updated=now(), completed50=complete),
            )
            return
        if not args.watch:
            return
        state = read_json(folder / "pipeline_progress.json")
        if state.get("status") == "failed":
            return
        if time.monotonic() >= deadline:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
