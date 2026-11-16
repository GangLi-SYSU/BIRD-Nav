"""Read-only live training/validation curves for the epoch300 continuation run."""

import argparse
import json
import os
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def load_history(folder, normalized=False):
    config = json.loads((folder / "frozen.json").read_text())["training_config"]
    source = folder / "training/engine_scaled_units/training_history.json"
    if not source.exists():
        return [], config
    rows = json.loads(source.read_text())["epochs"]
    factor = 1.0 if normalized else config["scale_m"] ** 2
    return (
        [
            dict(
                epoch=r["epoch"],
                train=r["train_component_mse_m2"] * factor,
                validation=r["validation_component_mse_m2"] * factor,
            )
            for r in rows
        ],
        config,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=PROJECT / "runs/stage15_rotation_300_01")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds")
    parser.add_argument(
        "--normalized", action="store_true", help="Show original normalized engine loss"
    )
    parser.add_argument(
        "--save-only", action="store_true", help="Save one static PNG without opening a window"
    )
    parser.add_argument("--output", type=Path, help="PNG path for --save-only")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be at least1 second")
    folder = args.run_dir.resolve()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/bee-nav-live-loss")
    import matplotlib

    if args.save_only:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    try:
        figure.canvas.manager.set_window_title("Bee-Nav: live training and validation loss")
    except AttributeError:
        pass
    state = {"epoch": None}

    def refresh(_frame=None):
        rows, config = load_history(folder, args.normalized)
        epoch = rows[-1]["epoch"] if rows else 0
        if epoch == state["epoch"]:
            return
        state["epoch"] = epoch
        for axis in axes:
            axis.clear()
        if not rows:
            axes[0].text(0.5, 0.5, "Waiting for first completed epoch", ha="center")
        else:
            best = min(rows, key=lambda r: r["validation"])
            for axis, values, label in (
                (axes[0], rows, "Full history (log scale)"),
                (axes[1], rows[-50:], "Latest50 completed epochs"),
            ):
                axis.plot(
                    [r["epoch"] for r in values], [r["train"] for r in values], label="Training"
                )
                axis.plot(
                    [r["epoch"] for r in values],
                    [r["validation"] for r in values],
                    label="Validation",
                )
                axis.set(
                    title=label,
                    xlabel="Epoch",
                    ylabel="Normalized MSE" if args.normalized else "Physical component MSE (m2)",
                )
                axis.grid(alpha=0.2)
                axis.legend()
            if all(min(r["train"], r["validation"]) > 0 for r in rows):
                axes[0].set_yscale("log")
            figure.suptitle(
                "Epoch {}/{} | train {:.5f} | validation {:.5f} | best {:.5f} @ epoch {}".format(
                    epoch,
                    config["epochs"],
                    rows[-1]["train"],
                    rows[-1]["validation"],
                    best["validation"],
                    best["epoch"],
                )
            )
        figure.tight_layout()

    refresh()
    if args.save_only:
        output = args.output or folder / "monitor/loss_snapshot.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150)
        plt.close(figure)
        print(str(output))
        return
    if matplotlib.get_backend().lower() == "agg":
        raise RuntimeError(
            "No GUI backend. Run in a desktop terminal with DISPLAY set, or use --save-only."
        )
    figure._loss_animation = FuncAnimation(
        figure, refresh, interval=args.interval * 1000, cache_frame_data=False
    )
    plt.show()


if __name__ == "__main__":
    main()
