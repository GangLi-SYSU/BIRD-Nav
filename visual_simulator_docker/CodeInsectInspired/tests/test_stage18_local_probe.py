"""Stationary probe coverage, unchanged sources and coordinate/error diagnostics."""

import copy
import math
from pathlib import Path
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.camera_geometry import body_to_world, home_vector_body
from insect_utils.experiment_state import read_json
from insect_utils.local_probe import build_plan, probe_samples, summarize_rows, vector_diagnostic


class LocalProbeTests(unittest.TestCase):
    def test_cross_has104_samples_with_origin_and_opposite_directions(self):
        samples = probe_samples("cross13", [0, 0, 40])
        self.assertEqual(len(samples), 104)
        positions = {tuple(s["planned_position_m"]) for s in samples}
        self.assertEqual(len(positions), 13)
        for x, y, z in positions:
            self.assertIn((-x, -y, z), positions)
        self.assertEqual(len({s["sample_id"] for s in samples}), 104)
        self.assertEqual({s["planned_body_yaw_deg"] for s in samples}, set(range(0, 360, 45)))
        home = [s for s in samples if s["planned_position_m"] == [0, 0, 40]]
        self.assertEqual(len(home), 8)
        self.assertTrue(all(math.hypot(*s["planned_target_body_m"]) == 0 for s in home))

    def test_grid_has648_samples_and_preserves_translated_home(self):
        samples = probe_samples("grid81", [13, -8, 40])
        self.assertEqual(len(samples), 648)
        self.assertEqual(len({s["position_id"] for s in samples}), 81)
        self.assertEqual(samples[0]["planned_position_m"], [13, -8, 40])
        self.assertEqual(
            {s["planned_position_m"][0] for s in samples}, {11 + i * 0.5 for i in range(9)}
        )
        for s in samples:
            target = s["planned_target_body_m"]
            world = body_to_world(target, s["planned_body_yaw_deg"])
            p = s["planned_position_m"]
            self.assertAlmostEqual(p[0] + world[0], 13)
            self.assertAlmostEqual(p[1] + world[1], -8)

    def test_ideal_predictor_has_no_bias_or_heading_dispersion(self):
        rows = []
        for sample in probe_samples("cross13", [0, 0, 40]):
            position = sample["planned_position_m"]
            yaw = sample["planned_body_yaw_deg"]
            row = vector_diagnostic(
                position, yaw, home_vector_body(position, [0, 0, 40], yaw), [0, 0, 40]
            )
            row.update(position_id=sample["position_id"])
            rows.append(row)
        summary = summarize_rows(rows, [0, 0, 40])
        self.assertAlmostEqual(summary["vector_error_m"]["mean"], 0)
        self.assertEqual(summary["home_output_norm_m"]["maximum"], 0)
        self.assertTrue(all(p["heading_dispersion_rms_m"] < 1e-12 for p in summary["per_position"]))
        self.assertTrue(
            all(r["angle_error_deg"] is None for r in rows if r["true_distance_m"] == 0)
        )

    def test_fixed_world_bias_differs_from_fixed_body_bias_across_headings(self):
        world_rows = []
        body_rows = []
        for yaw in range(0, 360, 45):
            for rows, prediction in (
                (world_rows, body_to_world([1, -0.25], -yaw)),
                (body_rows, [1, -0.25]),
            ):
                row = vector_diagnostic([0, 0, 40], yaw, prediction, [0, 0, 40])
                row["position_id"] = "home"
                rows.append(row)
        fixed_world = summarize_rows(world_rows, [0, 0, 40])["per_position"][0]
        fixed_body = summarize_rows(body_rows, [0, 0, 40])["per_position"][0]
        self.assertAlmostEqual(fixed_world["predicted_home_mean_xy_m"][0], 1)
        self.assertAlmostEqual(fixed_world["heading_dispersion_rms_m"], 0)
        self.assertAlmostEqual(fixed_body["predicted_home_mean_xy_m"][0], 0)
        self.assertAlmostEqual(fixed_body["heading_dispersion_rms_m"], math.hypot(1, 0.25))

    def test_invalid_predictions_unknown_design_and_zero_direction_handling(self):
        with self.assertRaises(ValueError):
            probe_samples("unconfirmed", [0, 0, 40])
        for prediction in ([float("nan"), 0], [1, 2, 3]):
            with self.assertRaises(ValueError):
                vector_diagnostic([1, 0, 40], 0, prediction, [0, 0, 40])
        row = vector_diagnostic([1, 0, 40], 0, [0, 0], [0, 0, 40])
        self.assertIsNone(row["angle_error_deg"])
        self.assertEqual(row["vector_error_m"], 1)
        self.assertEqual(
            vector_diagnostic([1, 0, 40], 0, [1, 0], [0, 0, 40])["angle_error_deg"], 180
        )

    def test_both_candidate_plans_keep_model_camera_controller_and_source_hashes(self):
        source = read_json(PROJECT / "runs/stage17_radius_5_10_01/frozen.json")
        original = copy.deepcopy(source)
        for design, count in (("cross13", 104), ("grid81", 648)):
            plan = build_plan(design)
            self.assertEqual(len(plan["samples"]), count)
            self.assertEqual(plan["model"], source["model"])
            for key in ("scene", "camera", "home_camera_xyz_m", "runtime", "controller"):
                self.assertEqual(plan["protocol"][key], source["protocol"][key])
            for name, sha in source["files"].items():
                self.assertEqual(plan["files"][name], sha)
            self.assertNotIn("formal", plan["protocol"])
        self.assertEqual(source, original)


if __name__ == "__main__":
    unittest.main()
