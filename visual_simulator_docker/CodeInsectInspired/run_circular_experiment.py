"""Prepare, preview, GPU-smoke or train isolated circular nadir controls."""

import argparse
from pathlib import Path
import traceback

from insect_utils.circular_training import (
    evaluate_circular,
    freeze_circular_selection,
    prepare_circular,
    train_circular_arm,
    verify_circular,
)
from insect_utils.experiment_config import PROJECT
from insect_utils.experiment_state import RunLock, atomic_json, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("prepare", "preview", "smoke", "train", "select", "evaluate")
    )
    parser.add_argument("--run-dir", type=Path, default=PROJECT / "runs/stage13_circular_01")
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("mask_only", "mask_rotation"),
        help="Choose controls when preparing a new run",
    )
    parser.add_argument(
        "--arm",
        choices=("mask_only", "mask_rotation"),
        help="Train/smoke one frozen arm; omit to run all requested arms sequentially",
    )
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    folder.mkdir(parents=True, exist_ok=True)
    with RunLock(folder):
        try:
            if args.phase == "prepare":
                frozen = prepare_circular(folder, args.arms)
                print("PREPARED " + frozen["fingerprint"], flush=True)
                return
            if args.arms:
                raise ValueError("--arms is only valid during preparation")
            frozen = verify_circular(folder)
            if args.phase == "preview":
                from tools.preview_circular_augmentation import make_preview

                result = make_preview(folder)
            elif args.phase in ("smoke", "train"):
                arms = [args.arm] if args.arm else list(frozen["protocol"]["arms"])
                result = {
                    arm: train_circular_arm(folder, arm, smoke=args.phase == "smoke")
                    for arm in arms
                }
                if args.phase == "smoke":
                    atomic_json(
                        folder / "smoke/completion.json",
                        dict(
                            status="completed",
                            updated=now(),
                            arms=result,
                            formal_training_updates=0,
                            formal_initialization_from_smoke=False,
                            test_data_opened=False,
                        ),
                    )
                elif all(
                    (folder / "arms" / arm / "completion.json").exists()
                    for arm in frozen["protocol"]["arms"]
                ):
                    freeze_circular_selection(folder)
            elif args.phase == "select":
                result = freeze_circular_selection(folder)
            else:
                result = evaluate_circular(folder)
            atomic_json(
                folder / "progress.json",
                dict(
                    status="completed",
                    phase=args.phase,
                    updated=now(),
                    fingerprint=frozen["fingerprint"],
                ),
            )
            print("COMPLETED " + args.phase, flush=True)
            if args.phase == "preview":
                print(result["grid"], flush=True)
        except BaseException as error:
            atomic_json(
                folder / "failure.json",
                dict(
                    updated=now(),
                    phase=args.phase,
                    error=str(error),
                    traceback=traceback.format_exc(),
                ),
            )
            raise


if __name__ == "__main__":
    main()
