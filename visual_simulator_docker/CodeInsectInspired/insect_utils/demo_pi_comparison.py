"""Recorded odometry and paired compass / gyro-only PI counterfactuals."""
import copy
import csv
import json
import math
import os
from pathlib import Path

import numpy as np


class OdometryTrace:
    def __init__(self, sigma, seed, gyro_deg=0.63):
        self.sigma, self.seed, self.gyro_deg = sigma, int(seed), gyro_deg
        self.gyro_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 7341]))
        self.bias = 0.0
        self.gyro_xy = np.zeros(2)
        self.samples = []
        self.pi = []

    def outbound(self, nominal, actual_xy, estimated_xy, dt):
        nominal = np.asarray(nominal)
        self.bias += math.radians(self.gyro_deg) * math.sqrt(dt) * self.gyro_rng.standard_normal()
        c, s = math.cos(self.bias), math.sin(self.bias)
        self.gyro_xy += np.array([[c, -s], [s, c]]) @ nominal
        self.samples.append(
            dict(
                nominal_xy_m=nominal.tolist(),
                dt_s=float(dt),
                true_xy_m=np.asarray(actual_xy).tolist(),
                magnetic_estimate_xy_m=np.asarray(estimated_xy).tolist(),
                gyro_estimate_xy_m=self.gyro_xy.tolist(),
                gyro_bias_rad=self.bias,
            )
        )

    def start_return(self, actual_xy, estimated_xy):
        self.pi = [
            dict(
                true_xy_m=np.asarray(actual_xy).tolist(),
                magnetic_estimate_xy_m=np.asarray(estimated_xy).tolist(),
            )
        ]

    def return_step(self, actual_xy, estimated_xy):
        self.pi.append(
            dict(
                true_xy_m=np.asarray(actual_xy).tolist(),
                magnetic_estimate_xy_m=np.asarray(estimated_xy).tolist(),
            )
        )

    def snapshot(self):
        return copy.deepcopy(
            dict(
                sigma=self.sigma,
                seed=self.seed,
                gyro_deg_per_sqrt_s=self.gyro_deg,
                outbound=self.samples,
                actual_pi_return=self.pi,
            )
        )


def _return_pair(
    true_start, mag_estimate, gyro_estimate, bias, sigma, gyro_deg, seed, keep_paths=False
):
    """Same starting physical positions; navigation uses estimates, never truth."""
    true_start = np.asarray(true_start, dtype=float)
    estimated = np.stack([mag_estimate, gyro_estimate]).astype(float).copy()
    actual = np.stack([true_start, true_start]).copy()
    drift = np.asarray(bias, dtype=float).copy()
    count = len(true_start)
    rng = np.random.default_rng(seed)
    history = [actual.copy()]
    budget = int(np.ceil(np.linalg.norm(estimated, axis=-1).max() / 2)) + 2
    for _ in range(budget):
        distance = np.linalg.norm(estimated, axis=-1)
        if np.max(distance) < 1e-8:
            break
        length = np.minimum(distance, 2.0)
        direction = np.divide(
            -estimated,
            distance[:, :, None],
            out=np.zeros_like(estimated),
            where=distance[:, :, None] > 1e-8,
        )
        distance_draw = rng.standard_normal(count)
        gyro_draw = rng.standard_normal(count)
        drift += math.radians(gyro_deg) * np.sqrt(length[1] / 4) * gyro_draw
        c, s = np.cos(-drift), np.sin(-drift)
        world = direction.copy()
        world[1, :, 0] = c * direction[1, :, 0] - s * direction[1, :, 1]
        world[1, :, 1] = s * direction[1, :, 0] + c * direction[1, :, 1]
        actual += world * (length + sigma * np.sqrt(length) * distance_draw[None, :])[:, :, None]
        estimated += direction * length[:, :, None]
        if keep_paths:
            history.append(actual.copy())
    if np.linalg.norm(estimated, axis=-1).max() > 1e-7:
        raise RuntimeError("Counterfactual return failed to exhaust its own estimated home vector")
    return actual, np.asarray(history)


def compare_pi(trace, trials=50):
    samples = trace["outbound"]
    if not samples:
        nominal = np.zeros((1, 2))
        dt = np.zeros(1)
        actual_end = np.zeros(2)
        mag_end = np.zeros(2)
        gyro_end = np.zeros(2)
        bias = 0.0
    else:
        nominal = np.array([s["nominal_xy_m"] for s in samples])
        dt = np.array([s["dt_s"] for s in samples])
        actual_end = np.array(samples[-1]["true_xy_m"])
        mag_end = np.array(samples[-1]["magnetic_estimate_xy_m"])
        gyro_end = np.array(samples[-1]["gyro_estimate_xy_m"])
        bias = samples[-1]["gyro_bias_rad"]
    sigma = trace["sigma"]
    gyro = trace["gyro_deg_per_sqrt_s"]
    seed = trace["seed"]
    single, history = _return_pair(
        actual_end[None, :],
        mag_end[None, :],
        gyro_end[None, :],
        np.array([bias]),
        sigma,
        gyro,
        np.random.SeedSequence([seed, 881]),
        True,
    )
    rng = np.random.default_rng(np.random.SeedSequence([seed, 882]))
    length = np.linalg.norm(nominal, axis=1)
    direction = np.divide(
        nominal, length[:, None], out=np.zeros_like(nominal), where=length[:, None] > 1e-10
    )
    distance = length + sigma * np.sqrt(length) * rng.standard_normal((trials, len(length)))
    true_end = distance @ direction
    biases = np.cumsum(
        math.radians(gyro) * np.sqrt(dt) * rng.standard_normal((trials, len(length))), axis=1
    )
    c, s = np.cos(biases), np.sin(biases)
    gyro_est = np.stack(
        [
            (c * nominal[:, 0] - s * nominal[:, 1]).sum(1),
            (s * nominal[:, 0] + c * nominal[:, 1]).sum(1),
        ],
        axis=1,
    )
    mag_est = np.repeat(nominal.sum(0)[None, :], trials, axis=0)
    endpoints, _ = _return_pair(
        true_end, mag_est, gyro_est, biases[:, -1], sigma, gyro, np.random.SeedSequence([seed, 883])
    )
    stats = {}
    for arm, positions in zip(["magnetic", "gyro_only"], endpoints):
        error = np.linalg.norm(positions, axis=1)
        successes = int((error <= 15).sum())
        p = successes / trials
        z = 1.959963984540054
        denom = 1 + z * z / trials
        center = (p + z * z / (2 * trials)) / denom
        half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
        stats[arm] = dict(
            successes=successes,
            trials=trials,
            success_rate=p,
            wilson95=[max(0, center - half), min(1, center + half)],
            mean_endpoint_error_m=float(error.mean()),
            endpoints_m=positions.tolist(),
        )
    return dict(
        model="Shared manual flight commands and distance draws; gyro-only odometry "
        "accumulates heading error. Both return until their own PI estimate reaches zero.",
        compass_noise_deg=0.0,
        return_step_m=2.0,
        return_speed_m_s=4.0,
        success_radius_m=15.0,
        trials=trials,
        seed=seed,
        statistics=stats,
        forecast_magnetic_path_m=history[:, 0, 0].tolist(),
        forecast_gyro_path_m=history[:, 1, 0].tolist(),
        forecast_magnetic_error_m=float(np.linalg.norm(single[0, 0])),
        forecast_gyro_error_m=float(np.linalg.norm(single[1, 0])),
    )


def export_pi_comparison(folder, episode, trace):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Circle

    result = compare_pi(trace)
    prefix = Path(folder) / f"episode_{episode:03d}_pi_comparison"
    payload = dict(
        recorded=trace,
        comparison=result,
        note=(
            "Paired Monte Carlo replay is a model-based counterfactual, "
            "not 50 user-flown flights. Blue actual return is the live magnetic-assisted "
            "trajectory. Forecasts use fixed 0.5 s updates."
        ),
    )
    temporary = prefix.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    os.replace(temporary, prefix.with_suffix(".json"))
    rows = []
    for phase, items in [("outbound", trace["outbound"]), ("pi_return", trace["actual_pi_return"])]:
        for i, item in enumerate(items):
            g = item.get("gyro_estimate_xy_m", [None, None])
            rows.append(
                dict(
                    phase=phase,
                    sample=i,
                    true_x_m=item["true_xy_m"][0],
                    true_y_m=item["true_xy_m"][1],
                    magnetic_est_x_m=item["magnetic_estimate_xy_m"][0],
                    magnetic_est_y_m=item["magnetic_estimate_xy_m"][1],
                    gyro_est_x_m=g[0],
                    gyro_est_y_m=g[1],
                )
            )
    if rows:
        with prefix.with_suffix(".csv.tmp").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(prefix.with_suffix(".csv.tmp"), prefix.with_suffix(".csv"))
    fig = Figure(figsize=(12, 9), constrained_layout=True)
    FigureCanvasAgg(fig)
    outbound, home, zoom, distribution = fig.subplots(2, 2).flatten()
    colors = dict(true="#555555", magnetic="#137fb4", gyro="#e58524")
    for key, label, color in [
        ("true_xy_m", "True outbound", colors["true"]),
        ("magnetic_estimate_xy_m", "Odometry + ideal compass", colors["magnetic"]),
        ("gyro_estimate_xy_m", "Odometry / gyro only", colors["gyro"]),
    ]:
        points = np.array([[0.0, 0.0]] + [s[key] for s in trace["outbound"]])
        outbound.plot(*points.T, color=color, label=label, lw=1.4)
    for ax in [home, zoom]:
        for key, label, color in [
            ("forecast_magnetic_path_m", "Compass return forecast", colors["magnetic"]),
            ("forecast_gyro_path_m", "Gyro-only return forecast", colors["gyro"]),
        ]:
            points = np.array(result[key])
            ax.plot(*points.T, "--", color=color, label=label, lw=1.3)
            ax.scatter(*points[-1], marker="x", color=color, s=45)
        if len(trace["actual_pi_return"]) > 1:
            points = np.array([s["true_xy_m"] for s in trace["actual_pi_return"]])
            ax.plot(*points.T, color="#103762", lw=2, label="Actual compass-assisted return")
            ax.scatter(*points[-1], color="#103762", s=35)
        ax.add_patch(
            Circle((0, 0), 15, fill=False, color="#29954a", label="15 m visual handoff area")
        )
    zoom.set(xlim=(-25, 25), ylim=(-25, 25))
    for arm, color, label in [
        ("magnetic", colors["magnetic"], "Compass"),
        ("gyro_only", colors["gyro"], "Gyro only"),
    ]:
        stat = result["statistics"][arm]
        end = np.array(stat["endpoints_m"])
        distribution.scatter(
            *end.T,
            s=15,
            alpha=0.65,
            color=color,
            label=f"{label}: {stat['successes']}/{stat['trials']} ({stat['success_rate']:.0%})",
        )
    distribution.add_patch(
        Circle((0, 0), 15, fill=False, color="#29954a", label="Success radius: 15 m")
    )
    for ax, title in zip(
        [outbound, home, zoom, distribution],
        [
            "Recorded outbound: true vs odometry estimates",
            "Stage 1: PI return comparison",
            "Near-home return detail",
            "50 paired command replays: PI endpoints",
        ],
    ):
        ax.scatter(0, 0, marker="*", s=85, color="#c79800", zorder=9)
        ax.set(title=title, xlabel="East (m)", ylabel="North (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="best")
    # Preserve the fixed near-home viewing window rather than expand to far paths.
    zoom.set_aspect("equal", adjustable="box")
    zoom.set_xlim(-25, 25)
    zoom.set_ylim(-25, 25)
    fig.suptitle(
        f"Episode {episode}: odometry and PI | distance sigma {trace['sigma']} m/sqrt(m)\n"
        f"Ideal compass (zero noise) vs gyro drift {trace['gyro_deg_per_sqrt_s']} deg/sqrt(s)",
        fontsize=12,
    )
    fig.savefig(prefix.with_suffix(".png.tmp"), format="png", dpi=130)
    os.replace(prefix.with_suffix(".png.tmp"), prefix.with_suffix(".png"))
    return prefix.with_suffix(".png")
