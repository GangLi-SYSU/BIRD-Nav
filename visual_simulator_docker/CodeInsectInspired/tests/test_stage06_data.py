"""Position leakage and historical capture integrity regressions; standard library."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.camera_geometry import home_vector_body
from insect_utils.data_preparation import (
    grouped_split,
    validate_records,
    verify_split,
    verify_capture_source,
)
from insect_utils.experiment_config import canonical_hash, digest
from insect_utils.experiment_state import atomic_json
from insect_utils.rgb_preprocessing import body_direction_image, require_downward_config


def example_records(count=147):
    result = []
    for position in range(count):
        point = [12 + position * 0.1, -8 + position * 0.03, 12]
        group = "position_{:06d}".format(position)
        for heading in range(8):
            index = len(result)
            yaw = (position * 11.3 + heading * 45) % 360
            target = home_vector_body(point, [12, -8, 12], yaw)
            result.append(
                {
                    "sample_id": "sample_{:06d}".format(index),
                    "pose_id": "pose_{:06d}".format(index),
                    "frame_id": index + 10,
                    "image": "Replicator/rgb/rgb_{:06d}.png".format(index),
                    "position_id": group,
                    "heading_index": heading,
                    "x": point[0],
                    "y": point[1],
                    "z": 12.0,
                    "yaw": yaw,
                    "target_x": target[0],
                    "target_y": target[1],
                    "camera_id": "camera_00",
                    "split_group": group,
                    "ground_z_m": 2.0,
                    "agl_m": 10.0,
                    "role": "learning",
                    "sha256": "f" * 64,
                }
            )
    return result


class GroupedSplitTests(unittest.TestCase):
    def setUp(self):
        self.rows = example_records()
        self.order = list(reversed(range(147)))

    def test_approved_group_and_image_counts_and_all_headings_preserved(self):
        self.assertEqual(len(validate_records(self.rows, [12, -8])), 147)
        split = grouped_split(self.rows, self.order)
        self.assertEqual(
            {k: len(v) for k, v in split["groups"].items()},
            {"train": 133, "validation": 6, "test": 8},
        )
        self.assertEqual(
            {k: len(v) for k, v in split["indices"].items()},
            {"train": 1064, "validation": 48, "test": 64},
        )
        self.assertEqual(split, grouped_split(self.rows, self.order))

    def test_single_heading_moved_between_splits_is_rejected(self):
        split = grouped_split(self.rows, self.order)
        index = split["indices"]["train"].pop()
        split["indices"]["test"].append(index)
        with self.assertRaises(ValueError):
            verify_split(self.rows, split)

    def test_duplicate_or_missing_group_permutation_is_rejected(self):
        for order in [self.order[:-1], self.order[:-1] + [self.order[0]]]:
            with self.assertRaises(ValueError):
                grouped_split(self.rows, order)

    def test_empty_validation_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Too few positions"):
            grouped_split(example_records(10), list(range(10)))

    def test_missing_heading_and_duplicate_image_are_rejected(self):
        rows = self.rows[:-1]
        with self.assertRaisesRegex(ValueError, "headings"):
            validate_records(rows, [12, -8])
        rows = copy.deepcopy(self.rows)
        rows[1]["image"] = rows[0]["image"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_records(rows, [12, -8])

    def test_incorrect_label_and_nonfinite_values_are_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[4]["target_x"] += 1
        with self.assertRaisesRegex(ValueError, "reconstruct home"):
            validate_records(rows, [12, -8])
        rows = copy.deepcopy(self.rows)
        rows[4]["yaw"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            validate_records(rows, [12, -8])

    def test_different_locations_cannot_share_a_position_group(self):
        rows = copy.deepcopy(self.rows)
        row = rows[1]
        row["x"] += 0.1
        row["target_x"], row["target_y"] = home_vector_body(
            [row["x"], row["y"], row["z"]], [12, -8, 12], row["yaw"]
        )
        with self.assertRaisesRegex(ValueError, "different camera locations"):
            validate_records(rows, [12, -8])

    def test_changed_heading_spacing_is_rejected_even_with_valid_home_label(self):
        rows = copy.deepcopy(self.rows)
        row = rows[1]
        row["yaw"] += 1
        row["target_x"], row["target_y"] = home_vector_body(
            [row["x"], row["y"], row["z"]], [12, -8, 12], row["yaw"]
        )
        with self.assertRaisesRegex(ValueError, "heading spacing"):
            validate_records(rows, [12, -8])

    def test_direction_arrow_uses_body_axes_and_handles_zero(self):
        for vector, expected in [
            ([1, 0], [0, -1]),
            ([0, 1], [-1, 0]),
            ([-1, 0], [0, 1]),
            ([0, -1], [1, 0]),
        ]:
            self.assertEqual(body_direction_image(vector), expected)
        self.assertIsNone(body_direction_image([0, 0]))
        with self.assertRaises(ValueError):
            body_direction_image([float("nan"), 1])

    def test_downward_augmentation_and_unprepared_configuration_are_rejected(self):
        with self.assertRaises(ValueError):
            require_downward_config({"camera_mode": "downward_pinhole", "augment": True})
        with self.assertRaises(ValueError):
            require_downward_config({"camera_mode": "downward_pinhole"})


class HistoricalCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name)
        self.run = self.project / "runs/source"
        (self.run / "capture").mkdir(parents=True)
        (self.run / "state").mkdir()
        payload = {
            "spec": {"home": {"ground_xy_m": [0, 0]}},
            "derived": {},
            "provenance": {"source_sha256": {"old.py": "original historical hash"}},
        }
        self.fingerprint = canonical_hash(payload)
        atomic_json(self.run / "resolved_config.json", dict(payload, fingerprint=self.fingerprint))
        atomic_json(self.run / "run_manifest.json", {"fingerprint": self.fingerprint})
        (self.run / "capture/sample.png").write_bytes(b"fixture image bytes")
        (self.run / "capture/samples.jsonl").write_text("{}\n")
        atomic_json(
            self.run / "capture/dataset_manifest.json",
            {
                "fingerprint": self.fingerprint,
                "camera_mode": "downward_pinhole",
                "producer": "isaac",
                "purpose": "research",
                "role": "learning_only",
            },
        )
        self.save_state()

    def save_state(self):
        files = {
            str(p.relative_to(self.run)): {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in (self.run / "capture").iterdir()
        }
        atomic_json(
            self.run / "state/capture.json",
            {"status": "completed", "fingerprint": self.fingerprint, "artifacts": files},
        )

    def verify(self):
        return verify_capture_source(self.run, self.project, full=True)

    def test_new_source_code_does_not_invalidate_a_frozen_verified_capture(self):
        (self.project / "old.py").write_text("a different current implementation")
        reference, _, _ = self.verify()
        self.assertEqual(reference["fingerprint"], self.fingerprint)

    def test_capture_byte_change_is_rejected(self):
        (self.run / "capture/sample.png").write_bytes(b"modified image")
        with self.assertRaisesRegex(ValueError, "Artifact fingerprint mismatch"):
            self.verify()

    def test_historical_configuration_mutation_is_rejected(self):
        path = self.run / "resolved_config.json"
        data = json.loads(path.read_text())
        data["spec"]["home"]["ground_xy_m"] = [1, 0]
        atomic_json(path, data)
        with self.assertRaisesRegex(ValueError, "configuration was modified"):
            self.verify()

    def test_incomplete_capture_is_rejected(self):
        path = self.run / "state/capture.json"
        data = json.loads(path.read_text())
        data["status"] = "running"
        atomic_json(path, data)
        with self.assertRaisesRegex(ValueError, "not completed"):
            self.verify()

    def test_engineering_fixture_cannot_be_used_as_research_learning_data(self):
        path = self.run / "capture/dataset_manifest.json"
        data = json.loads(path.read_text())
        data["purpose"] = "engineering_smoke"
        atomic_json(path, data)
        self.save_state()
        with self.assertRaisesRegex(ValueError, "not a learning dataset"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
