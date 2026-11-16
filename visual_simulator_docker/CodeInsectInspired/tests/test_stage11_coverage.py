"""Regression checks for coverage grouping, replay safety and target unit conversion."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.coverage_capture import check_observation, committed_records, planned_samples
from insect_utils.coverage_data import audit_records
from insect_utils.coverage_plan import PROTOCOL, audit_plan, build_plan, radial_bin
from insect_utils.experiment_config import canonical_hash
from insect_utils.experiment_state import atomic_json, read_json


class CoveragePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = read_json(PROJECT / PROTOCOL)
        prepared = PROJECT / cls.protocol["legacy_prepared"]
        cls.rows = read_json(prepared / "records.json")
        cls.split = read_json(prepared / "splits.json")
        cls.plan = build_plan(cls.protocol, cls.rows, cls.split)

    def test_exact_600_position_plan_and_balanced_holdouts(self):
        audit = self.plan["audit"]
        self.assertEqual((audit["positions"], audit["images"]), (600, 4800))
        for split in ("validation", "new_test"):
            self.assertEqual(audit["counts"][split]["bins"], [10, 10, 10, 10])
        self.assertEqual(audit["counts"]["train"]["positions"], 512)
        self.assertGreaterEqual(audit["minimum_train_holdout_center_distance_m"], 1)

    def test_every_radial_holdout_bin_covers_all_four_quadrants(self):
        for split, expected in (("validation", [3, 2, 3, 2]), ("new_test", [2, 3, 2, 3])):
            self.assertEqual(
                self.plan["audit"]["counts"][split]["quadrants_per_radial_bin"], [expected] * 4
            )

    def test_all_legacy_test_positions_and_headings_are_preserved(self):
        expected = {(r["position_id"], r["heading_index"]): r for r in self.rows}
        for position in self.plan["positions"]:
            if position["split"] != "legacy_test":
                continue
            self.assertTrue(position["legacy"])
            for heading, yaw in enumerate(position["yaws_deg"]):
                self.assertEqual(expected[position["position_id"], heading]["yaw"], yaw)

    def test_eight_physical_observations_per_fresh_position(self):
        samples = planned_samples(self.plan)
        self.assertEqual(len(samples), self.plan["audit"]["fresh_images"])
        self.assertEqual(len({s["sample_id"] for s in samples}), len(samples))
        groups = {}
        for sample in samples:
            groups.setdefault(sample["position_id"], []).append(sample)
        for rows in groups.values():
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({r["split"] for r in rows}), 1)
            self.assertEqual(len({r["planned_body_yaw_deg"] for r in rows}), 8)

    def test_deterministic_plan_without_predictions(self):
        second = build_plan(self.protocol, self.rows, self.split)
        self.assertEqual(canonical_hash(self.plan), canonical_hash(second))

    def test_cross_partition_duplicate_and_missing_headings_rejected(self):
        rows = [dict(r, split="train") for r in self.rows[:8]]
        audit_records(rows)
        changed = copy.deepcopy(rows)
        changed[0]["split"] = "new_test"
        with self.assertRaises(ValueError):
            audit_records(changed)
        with self.assertRaises(ValueError):
            audit_records(rows[:-1])
        with self.assertRaises(ValueError):
            audit_records(rows + rows)

    def test_invalid_radius_and_tampered_legacy_partition_rejected(self):
        with self.assertRaises(ValueError):
            radial_bin(45, 0, self.protocol["radial_edges_m"])
        bad = copy.deepcopy(self.plan)
        p = next(p for p in bad["positions"] if p["split"] == "legacy_test")
        p["position_id"] = "changed_test"
        with self.assertRaises(ValueError):
            audit_plan(bad, self.protocol, self.split)

    def test_camera_receipt_detects_optics_and_rendering_changes(self):
        source = PROJECT / self.protocol["legacy_capture"] / "capture/samples.jsonl"
        import json

        with source.open() as stream:
            row = json.loads(stream.readline())
        sample = {
            key: row[key]
            for key in (
                "sample_id",
                "pose_id",
                "position_id",
                "heading_index",
                "planned_position_m",
                "planned_body_yaw_deg",
                "planned_target_body_m",
                "split_group",
                "role",
            )
        }
        row["camera_id"] = "camera_0"
        row["receipt"]["camera_prim"] = "/World/CoverageCamera_00"
        check_observation(sample, row, self.protocol)
        bad = copy.deepcopy(row)
        bad["receipt"]["observed_usd_optics"]["horizontalAperture"] *= 2
        with self.assertRaises(ValueError):
            check_observation(sample, bad, self.protocol)
        bad = copy.deepcopy(row)
        bad["receipt"]["observed_render_settings"]["antialiasing"] = 3
        with self.assertRaises(ValueError):
            check_observation(sample, bad, self.protocol)

    def test_atomic_batch_resume_ignores_orphans_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = root / "capture/sessions/session_a"
            session.mkdir(parents=True)
            (session / "uncommitted.png").write_bytes(b"orphan")
            self.assertEqual(committed_records(root), {})
            row = {
                "sample_id": "sample",
                "session_id": "session_a",
                "camera_id": "camera_0",
                "frame_id": 40,
            }
            atomic_json(session / "batch_00001.json", [row])
            self.assertEqual(list(committed_records(root, verify_images=False)), ["sample"])
            atomic_json(session / "batch_00002.json", [row])
            with self.assertRaises(ValueError):
                committed_records(root, verify_images=False)


class PipelineCompletionTests(unittest.TestCase):
    def test_zero_exit_without_complete_capture_never_starts_preparation(self):
        import run_coverage_experiment as runner

        process = Mock(returncode=0)
        process.poll.return_value = 0
        lock = Mock()
        lock.fileno.return_value = 42
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with patch.object(runner, "PROJECT", folder), patch.object(
                runner, "PHASES", ("capture", "prepare")
            ), patch.object(
                runner, "verify_freeze", return_value={"fingerprint": "fixture"}
            ), patch.object(
                runner.subprocess, "Popen", return_value=process
            ) as launched:
                with self.assertRaisesRegex(RuntimeError, "complete capture receipt"):
                    runner.pipeline(folder, lock)
                self.assertEqual(launched.call_count, 1)


class PhysicalUnitsTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch runtime checks run in the project Docker image")
        self.torch = torch

    def test_scale_restores_meters_and_has_no_new_parameters(self):
        from insect_utils.coverage_model import PhysicalHomeVector

        torch = self.torch
        linear = torch.nn.Linear(3, 2)
        wrapper = PhysicalHomeVector(linear, 44)
        x = torch.randn(2, 3)
        self.assertTrue(torch.allclose(wrapper(x), linear(x) * 44))
        self.assertEqual(sum(p.numel() for p in wrapper.parameters()), 8)
        target = torch.randn(2, 2) * 40
        raw_mse = torch.nn.functional.mse_loss(wrapper(x), target)
        scaled_mse = torch.nn.functional.mse_loss(linear(x), target / 44)
        self.assertTrue(torch.allclose(raw_mse, scaled_mse * 44 ** 2))

    def test_test_partition_cannot_be_opened_for_training(self):
        from insect_utils.coverage_model import CoverageDataset

        with self.assertRaises(ValueError):
            CoverageDataset([], "new_test")
        with self.assertRaises(ValueError):
            CoverageDataset([], "legacy_test")

    def test_captured_tensor_and_targets_match_across_scales(self):
        from insect_utils.coverage_model import CoverageDataset

        protocol = read_json(PROJECT / PROTOCOL)
        row = dict(read_json(PROJECT / protocol["legacy_prepared"] / "records.json")[0])
        row.update(split="train", image=protocol["legacy_capture"] + "/capture/" + row["image"])
        raw_image, raw = CoverageDataset([row], "train", 1)[0]
        norm_image, norm = CoverageDataset([row], "train", 44)[0]
        self.assertTrue(self.torch.equal(raw_image, norm_image))
        self.assertTrue(self.torch.allclose(raw, norm * 44))
        self.assertEqual(tuple(raw_image.shape), (3, 1024, 1024))

    def test_optimizer_hook_saves_the_actual_post_update_state(self):
        torch = self.torch
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        saved = []
        hook = optimizer.register_step_post_hook(
            lambda _o, _a, _k: saved.append(model.weight.detach().clone())
        )
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        hook.remove()
        self.assertTrue(torch.equal(saved[0], model.weight))

    def test_network_initialization_is_identical_and_script_scales_output(self):
        from insect_utils.attention_model import create_attention
        from insect_utils.coverage_model import PhysicalHomeVector
        from insect_utils.downward_training import seed_all

        torch = self.torch
        torch.set_num_threads(2)
        seed_all(10)
        first = create_attention({"network_variant": "nadir384x384"}).eval()
        seed_all(10)
        second = create_attention({"network_variant": "nadir384x384"}).eval()
        self.assertEqual(sum(p.numel() for p in first.parameters()), 10628)
        for a, b in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(a, b))
        wrapper = PhysicalHomeVector(second, 44).eval()
        image = torch.rand(2, 3, 1024, 1024)
        with torch.no_grad():
            expected = first(image) * 44
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "physical.pt"
            torch.jit.trace(wrapper, image[:1]).save(str(path))
            with torch.no_grad():
                actual = torch.jit.load(str(path))(image)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
