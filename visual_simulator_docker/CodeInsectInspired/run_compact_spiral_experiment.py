"""Run the existing audited pipeline with the original spiral compressed to20m."""

import argparse
from pathlib import Path
import traceback
import uuid

from insect_utils.compact_spiral import comparison_plot, prepare
from insect_utils.experiment_config import PROJECT
from insect_utils.experiment_state import RunLock, atomic_json, now, read_json
from insect_utils.spiral_narrow_report import plan_figure
from run_spiral_narrow_experiment import pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    folder.relative_to((PROJECT / "runs").resolve())
    folder.mkdir(parents=True, exist_ok=True)
    with RunLock(folder) as lock:
        frozen = prepare(folder)
        plan_figure(folder)
        comparison_plot(folder)
        if args.plan_only:
            print("PREPARED_COMPACT_SPIRAL " + frozen["fingerprint"], flush=True)
            return
        try:
            # Worker entry points read the new frozen folder; no old data/weights are imported.
            pipeline(folder, lock)
            report = folder / "result_summary.md"
            marker = "# 半径20m密集螺旋实验"
            text = report.read_text(encoding="utf-8")
            if not text.startswith(marker):
                report.write_text(
                    marker + "\n\n采样点数147不变，原螺旋半径从43.98m压缩至20m，" "径向圈距4m。其余训练与测试设置保持不变。\n\n" + text,
                    encoding="utf-8",
                )
            result = read_json(folder / "completion.json")
            result.update(learning_radius_m=20, spiral_positions=147, nominal_radial_pitch_m=4)
            atomic_json(folder / "completion.json", result)
            atomic_json(folder / "pipeline_progress.json", dict(result, phase="completed"))
        except BaseException as error:
            detail = dict(
                status="failed", updated=now(), error=str(error), traceback=traceback.format_exc()
            )
            atomic_json(folder / ("failure_" + uuid.uuid4().hex[:10] + ".json"), detail)
            atomic_json(folder / "pipeline_progress.json", detail)
            raise


if __name__ == "__main__":
    main()
