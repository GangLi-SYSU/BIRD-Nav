"""Configuration and recovery failures that must never reuse wrong experiment data."""

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.experiment_config import identifier, load_config, preflight, resolve_spec
from insect_utils.experiment_config import stage_parameters
from insect_utils.experiment_state import RunLock, atomic_json, begin_stage, complete_stage
from insect_utils.experiment_state import (
    fail_stage,
    prepare_run,
    read_json,
    verify_completed,
    verify_run,
)


class ExperimentPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.project = Path(self.temporary.name)
        (self.project / "config/experiments").mkdir(parents=True)
        shutil.copyfile(
            PROJECT / "config/experiments/stage01_protocol.json",
            self.project / "config/experiments/stage01_protocol.json",
        )
        (self.project / "example.py").write_text("value = 1\n")
        self.spec = json.loads(
            (PROJECT / "config/experiments/downward_h10_hfov30.json").read_text()
        )
        scene = self.project / self.spec["scene"]["path"]
        scene.parent.mkdir(parents=True)
        scene.write_text("scene version one")
        self.run = self.project / "run"

    def tearDown(self):
        self.temporary.cleanup()

    def resolved(self):
        return resolve_spec(self.spec, self.project)

    def completed_capture(self):
        resolved = self.resolved()
        prepare_run(self.run, resolved)
        begin_stage(self.run, "capture", resolved, False)
        (self.run / "capture/sample.txt").write_text("recorded sample")
        complete_stage(self.run, "capture", 1234)
        return resolved

    def test_nonzero_home_and_ground_height_reach_both_stage_configs(self):
        self.spec["home"]["ground_xy_m"] = [12.0, -8.0]
        self.spec["scene"]["ground_z_m"] = 2.0
        resolved = self.resolved()
        params = stage_parameters(resolved, self.run, self.project)
        self.assertEqual(resolved["derived"]["home_ground_xyz_m"], [12.0, -8.0, 2.0])
        self.assertEqual(params["capture"]["home_position"], [12.0, -8.0, 12.0])
        self.assertEqual(params["evaluate"]["virtual_home_point"], [12.0, -8.0, 12.0])
        self.assertAlmostEqual(resolved["derived"]["flat_ground_footprint_wh_m"][0], 5.3589838486)

    def test_models_indices_and_images_are_owned_by_this_run(self):
        params = stage_parameters(self.resolved(), self.run, self.project)
        self.assertEqual(params["train"]["models_folder"], str(self.run / "train"))
        self.assertEqual(params["train"]["indices_file"], str(self.run / "train/indices.pkl"))
        self.assertEqual(params["capture"]["output_folder"], str(self.run / "capture"))

    def test_invalid_height_and_nan_are_rejected(self):
        for height in (-1, float("nan"), True):
            with self.subTest(height=height):
                self.spec["flight"]["camera_height_agl_m"] = height
                with self.assertRaises(ValueError):
                    self.resolved()

    def test_downward_panorama_augmentation_is_rejected(self):
        self.spec["training"]["augment"] = True
        with self.assertRaisesRegex(ValueError, "invalid for downward"):
            self.resolved()

    def test_unknown_top_level_knob_is_rejected(self):
        self.spec["epohcs"] = 1
        with self.assertRaisesRegex(ValueError, "Unknown/missing"):
            self.resolved()

    def test_misspelled_nested_knob_cannot_be_silently_ignored(self):
        self.spec["training"]["epohcs"] = 1
        with self.assertRaisesRegex(ValueError, "fields in training"):
            self.resolved()

    def test_synthetic_data_cannot_be_declared_research(self):
        self.spec["capture"]["backend"] = "synthetic_smoke"
        with self.assertRaisesRegex(ValueError, "cannot be used for research"):
            self.resolved()

    def test_downward_capture_stops_before_importing_kit(self):
        self.spec["capture"].pop("headings_per_position", None)
        with self.assertRaisesRegex(ValueError, "explicitly confirm"):
            preflight(self.resolved(), "capture", self.project)
        self.assertNotIn("omni.isaac.kit", sys.modules)

    def test_synthetic_data_cannot_run_closed_loop_evaluation(self):
        self.spec["capture"]["backend"] = "synthetic_smoke"
        self.spec["purpose"] = "engineering_smoke"
        with self.assertRaisesRegex(ValueError, "cannot produce closed-loop"):
            preflight(self.resolved(), "evaluate", self.project)

    def test_folder_existence_is_not_a_completion_marker(self):
        self.run.mkdir()
        with self.assertRaisesRegex(ValueError, "not a prepared run"):
            verify_run(self.run, self.resolved())

    def test_completed_data_is_reused_only_with_explicit_resume(self):
        resolved = self.completed_capture()
        with self.assertRaisesRegex(ValueError, "already completed"):
            begin_stage(self.run, "capture", resolved, False)
        self.assertIsNone(begin_stage(self.run, "capture", resolved, True))

    def test_changed_image_or_model_bytes_block_resume(self):
        resolved = self.completed_capture()
        (self.run / "capture/sample.txt").write_text("changed sample")
        with self.assertRaisesRegex(ValueError, "Artifact fingerprint mismatch"):
            begin_stage(self.run, "capture", resolved, True)

    def test_missing_artifact_blocks_completion_reuse(self):
        resolved = self.completed_capture()
        (self.run / "capture/sample.txt").unlink()
        with self.assertRaisesRegex(ValueError, "Missing or invalid artifact"):
            verify_completed(self.run, "capture", resolved["fingerprint"])

    def test_failed_output_is_archived_before_retry(self):
        resolved = self.resolved()
        prepare_run(self.run, resolved)
        begin_stage(self.run, "capture", resolved, False)
        (self.run / "capture/partial.txt").write_text("partial")
        fail_stage(self.run, "capture", "interrupted")
        with self.assertRaisesRegex(ValueError, "interrupted or failed"):
            begin_stage(self.run, "capture", resolved, False)
        record = begin_stage(self.run, "capture", resolved, True)
        self.assertEqual(record["attempt"], 2)
        self.assertTrue((self.run / "attempts/capture/attempt_01/partial.txt").is_file())
        self.assertFalse((self.run / "capture/partial.txt").exists())

    def test_training_requires_completed_capture(self):
        resolved = self.resolved()
        prepare_run(self.run, resolved)
        with self.assertRaisesRegex(ValueError, "not completed: capture"):
            begin_stage(self.run, "train", resolved, False)
        self.assertFalse((self.run / "train").exists())

    def test_retry_preserves_archive_from_interrupted_state_commit(self):
        resolved = self.resolved()
        prepare_run(self.run, resolved)
        begin_stage(self.run, "capture", resolved, False)
        fail_stage(self.run, "capture", "interrupted state commit")
        archive = self.run / "attempts/capture/attempt_01"
        archive.mkdir(parents=True)
        (archive / "old.txt").write_text("preserved")
        (self.run / "capture/new_partial.txt").write_text("new partial")
        begin_stage(self.run, "capture", resolved, True)
        self.assertEqual((archive / "old.txt").read_text(), "preserved")
        recovery = list(archive.parent.glob("attempt_01_recovery_*"))
        self.assertEqual(len(recovery), 1)
        self.assertTrue((recovery[0] / "new_partial.txt").is_file())

    def test_changed_package_versions_block_existing_run(self):
        resolved = self.resolved()
        prepare_run(self.run, resolved)
        with patch(
            "insect_utils.experiment_config.importlib.metadata.version",
            return_value="changed-runtime",
        ):
            with self.assertRaisesRegex(ValueError, "use a new run ID"):
                verify_run(self.run, self.resolved())

    def test_scene_source_or_configuration_drift_requires_new_run(self):
        for kind in ("scene", "source", "config"):
            with self.subTest(kind=kind):
                spec = copy.deepcopy(self.spec)
                resolved = resolve_spec(spec, self.project)
                folder = self.project / ("run_" + kind)
                prepare_run(folder, resolved)
                if kind == "scene":
                    (self.project / spec["scene"]["path"]).write_text("new scene")
                elif kind == "source":
                    (self.project / "example.py").write_text("value = 2\n")
                else:
                    spec["seed"] = 11
                with self.assertRaisesRegex(ValueError, "changed; use a new run"):
                    verify_run(folder, resolve_spec(spec, self.project))

    def test_modified_saved_configuration_is_rejected(self):
        resolved = self.resolved()
        prepare_run(self.run, resolved)
        changed = read_json(self.run / "resolved_config.json")
        changed["derived"]["home_camera_xyz_m"][2] = 1.5
        atomic_json(self.run / "resolved_config.json", changed)
        with self.assertRaisesRegex(ValueError, "was modified"):
            verify_run(self.run, resolved)

    def test_concurrent_controller_cannot_acquire_run_lock(self):
        self.run.mkdir()
        with RunLock(self.run):
            with self.assertRaisesRegex(ValueError, "Another process"):
                with RunLock(self.run):
                    pass

    def test_run_id_cannot_escape_output_root(self):
        for value in ("../other", "/tmp/other", "..", "a/b"):
            with self.assertRaises(ValueError):
                identifier(value)

    def test_checked_in_default_can_be_resolved_without_heavy_imports(self):
        resolved = load_config(PROJECT / "config/experiments/downward_h10_hfov30.json")
        self.assertEqual(resolved["derived"]["home_camera_xyz_m"][2], 10)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
