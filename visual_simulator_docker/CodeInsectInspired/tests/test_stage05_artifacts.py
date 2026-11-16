"""Exercise real PNG/CSV readers on fixtures; no Isaac or real observations."""

import csv
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.camera_geometry import usd_world_matrix
from insect_utils.capture_geometry import measured_record
from insect_utils.downward_capture import build_plan, seal_dataset, write_tables
from insect_utils.experiment_config import digest, resolve_spec, stage_parameters
from insect_utils.experiment_state import atomic_json
from insect_utils.frame_stability import STABILITY_POLICY

HAS_READERS = all(importlib.util.find_spec(name) for name in ("numpy", "PIL"))


@unittest.skipUnless(HAS_READERS, "PNG/CSV fixtures need NumPy and Pillow")
class CaptureArtifactTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        project = Path(self.temporary.name)
        (project / "config/experiments").mkdir(parents=True)
        (project / "config/experiments/stage01_protocol.json").write_text("{}")
        spec = json.loads(
            (PROJECT / "config/experiments/downward_h10_hfov30_landmarks.json").read_text()
        )
        spec["purpose"] = "engineering_smoke"
        spec["home"]["ground_xy_m"] = [12, -8]
        spec["scene"] = {"path": "maps/fixture.usda", "ground_z_m": 2.0}
        spec["camera"]["raw_resolution_wh"] = [8, 8]
        spec["capture"].update(shape_params=[5, 8, 0.35], headings_per_position=1)
        scene = project / spec["scene"]["path"]
        scene.parent.mkdir()
        scene.write_text("Fixture bytes only, no actual USD scene")
        atomic_json(
            scene.with_suffix(".manifest.json"),
            {
                "schema_version": 1,
                "entry": scene.name,
                "asset_sha256": {scene.name: digest(scene)},
                "ground_z_m": 2.0,
                "bounds_xy_m": [[-200, -200], [200, 200]],
                "max_surface_height_agl_m": 0,
            },
        )
        self.resolved = resolve_spec(spec, project)
        self.run = project / "run"
        self.output = self.run / "capture"
        self.config = stage_parameters(self.resolved, self.run, project)["capture"]
        self.plan = build_plan(self.config)
        (self.output / "Replicator/rgb").mkdir(parents=True)
        atomic_json(self.output / "capture_plan.json", self.plan)
        self.records = []
        for index, sample in enumerate(self.plan["samples"]):
            frame = 10 + index * 3
            receipt = {
                "pose_id": sample["pose_id"],
                "backend": "replicator",
                "camera_prim": "/World/LearningCamera_{:02d}".format(index % 4),
                "camera_world_matrix": usd_world_matrix(
                    sample["planned_position_m"], sample["planned_body_yaw_deg"]
                ),
                "rendering_frame": frame,
                "updates_waited": 4,
                "stability": {
                    "policy": dict(STABILITY_POLICY),
                    "consecutive_changes": 2,
                    "history": [
                        {"frame": frame - 2, "rgb_mae": None},
                        {"frame": frame - 1, "rgb_mae": 0.3},
                        {"frame": frame, "rgb_mae": 0.2},
                    ],
                },
            }
            record = measured_record(sample, receipt, self.config["home_position"], 2, 10)
            Image.new("RGB", (8, 8), (index + 1, 42, 17)).save(self.output / sample["image"])
            record.update(
                camera_id="camera_{:02d}".format(index % 4),
                receipt=receipt,
                sha256=digest(self.output / sample["image"]),
            )
            self.records.append(record)
            atomic_json(self.output / (record["image"] + ".json"), record)
        self.save_log()
        write_tables(self.output, self.records)

    def save_log(self):
        (self.output / "samples.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in self.records)
        )

    def seal(self, publish=True):
        return seal_dataset(self.resolved, self.run, self.config, publish)

    def test_nondivisible_capture_and_legacy_parser_roundtrip(self):
        manifest = self.seal()
        self.assertEqual(manifest["sample_count"], 5)
        self.assertEqual(self.plan["batch_sizes"], [4, 1])
        self.assertEqual(len({r["split_group"] for r in self.records}), 5)
        self.assertEqual(manifest["training_steps"], 0)
        self.assertEqual(self.seal(publish=False), manifest)
        with self.assertRaises(FileExistsError):
            self.seal()

    def test_dropped_last_sample_rejected(self):
        self.records.pop()
        self.save_log()
        with self.assertRaisesRegex(ValueError, "Missing or extra capture sample"):
            self.seal()

    def test_camera_allocation_does_not_change_path_or_random_headings(self):
        config = copy.deepcopy(self.config)
        config["number_cameras"] = 1
        plan = build_plan(config)
        self.assertEqual(plan["samples"], self.plan["samples"])
        self.assertEqual(plan["pose_signature"], self.plan["pose_signature"])
        self.assertEqual(plan["batch_sizes"], [1] * 5)

    def test_multiple_real_headings_expand_budget_and_keep_the_first_heading(self):
        for count in [4, 8]:
            config = copy.deepcopy(self.config)
            config["headings_per_position"] = count
            plan = build_plan(config)
            self.assertEqual(plan["sample_count"], 5 * count)
            for index, original in enumerate(self.plan["samples"]):
                first = plan["samples"][index * count]
                self.assertEqual(first["planned_position_m"], original["planned_position_m"])
                self.assertEqual(first["planned_body_yaw_deg"], original["planned_body_yaw_deg"])
                group = plan["samples"][index * count : (index + 1) * count]
                self.assertEqual(len({s["planned_body_yaw_deg"] for s in group}), count)
                self.assertEqual(len({s["split_group"] for s in group}), 1)

    def test_reordered_image_pose_records_rejected(self):
        self.records[0], self.records[1] = self.records[1], self.records[0]
        self.save_log()
        with self.assertRaisesRegex(ValueError, "pose ID mismatch"):
            self.seal()

    def test_wrong_capture_camera_rejected(self):
        self.records[-1]["receipt"]["camera_prim"] = "/World/LearningCamera_03"
        self.save_log()
        with self.assertRaisesRegex(ValueError, "different capture camera"):
            self.seal()

    def test_changed_image_and_unexpected_extra_image_rejected(self):
        from PIL import Image

        path = self.output / self.records[0]["image"]
        original = path.read_bytes()
        Image.new("RGB", (8, 8), (255, 255, 255)).save(path)
        with self.assertRaisesRegex(ValueError, "Changed capture image"):
            self.seal()
        path.write_bytes(original)
        Image.new("RGB", (8, 8)).save(path.parent / "extra.png")
        with self.assertRaisesRegex(ValueError, "Missing or extra image file"):
            self.seal()

    def test_typed_csv_and_legacy_csv_mismatches_rejected(self):
        path = self.output / "samples.csv"
        original = path.read_text()
        path.write_text(original.replace("position_000000", "position_999999", 1))
        with self.assertRaisesRegex(ValueError, "Typed CSV differs"):
            self.seal()
        path.write_text(original)
        path = self.output / "dataset_navigation.csv"
        with path.open(newline="") as stream:
            rows = list(csv.reader(stream))
        rows[1][-1] = "[999 999]"
        with path.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with self.assertRaisesRegex(ValueError, "Legacy CSV label parse mismatch"):
            self.seal()


if __name__ == "__main__":
    unittest.main()
