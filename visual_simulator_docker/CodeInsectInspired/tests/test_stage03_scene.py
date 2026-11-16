"""Scene integrity and flight-volume regressions; standard library only."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from insect_utils.experiment_config import digest, resolve_spec
from insect_utils.experiment_state import prepare_run, verify_run
from insect_utils.scene_geometry import asset_provenance, check_positions_and_segments
from insect_utils.scene_geometry import require_assets, spatial_budget


class Stage03SceneTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "config/experiments").mkdir(parents=True)
        (self.root / "config/experiments/stage01_protocol.json").write_text("{}")
        self.spec = json.loads(
            (PROJECT / "config/experiments/downward_h10_hfov30.json").read_text()
        )
        self.scene = self.root / self.spec["scene"]["path"]
        self.scene.parent.mkdir(parents=True)
        self.scene.write_text("scene bytes")
        self.texture = self.scene.parent / "terrain.png"
        self.texture.write_bytes(b"texture bytes")
        self.manifest = {
            "schema_version": 1,
            "entry": self.scene.name,
            "asset_sha256": {
                self.scene.name: digest(self.scene),
                "terrain.png": digest(self.texture),
            },
            "bounds_xy_m": [[-50, -50], [50, 50]],
            "ground_z_m": 0,
            "max_surface_height_agl_m": 0,
        }
        self.save_manifest()

    def tearDown(self):
        self.temporary.cleanup()

    def save_manifest(self):
        self.scene.with_suffix(".manifest.json").write_text(json.dumps(self.manifest))

    def test_all_yaws_use_footprint_diagonal_not_half_width(self):
        budget = spatial_budget(self.spec, self.manifest)
        self.assertAlmostEqual(budget["footprint_circumradius_m"], 3.7893738196)
        self.assertAlmostEqual(budget["allowed_camera_center_xy_m"][1][0], 41.2106261804)
        self.assertTrue(budget["boundary_guard_required"])
        self.assertFalse(budget["boundary_guard_connected_to_controller"])

    def test_full_worst_case_actions_need_more_than_100m_scene(self):
        budget = spatial_budget(self.spec, self.manifest)
        self.assertGreater(budget["required_side_m_without_boundary_guard"], 317)

    def test_valid_endpoint_pair_proves_segment_within_convex_flat_ground(self):
        budget = spatial_budget(self.spec, self.manifest)
        result = check_positions_and_segments([[-40, 40, 10], [40, -40, 10]], budget)
        self.assertEqual(result["segments_checked"], 1)

    def test_out_of_bounds_move_rejected_even_with_center_over_ground(self):
        budget = spatial_budget(self.spec, self.manifest)
        with self.assertRaisesRegex(ValueError, "leaves valid ground"):
            check_positions_and_segments([[40, 0, 10], [45, 0, 10]], budget)

    def test_nan_or_wrong_camera_height_rejected(self):
        budget = spatial_budget(self.spec, self.manifest)
        for point in ([0, float("nan"), 10], [0, 0, 1.5]):
            with self.assertRaises(ValueError):
                check_positions_and_segments([point], budget)

    def test_roof_too_close_to_flight_plane_rejected(self):
        self.manifest["max_surface_height_agl_m"] = 9.5
        budget = spatial_budget(self.spec, self.manifest)
        with self.assertRaisesRegex(ValueError, "vertical clearance"):
            check_positions_and_segments([[0, 0, 10]], budget)

    def test_ground_height_disagreement_rejected(self):
        self.manifest["ground_z_m"] = 2
        with self.assertRaisesRegex(ValueError, "ground height"):
            spatial_budget(self.spec, self.manifest)

    def test_start_circle_and_footprint_must_fit_for_nonzero_home(self):
        spec = copy.deepcopy(self.spec)
        spec["home"]["ground_xy_m"] = [20, 0]
        with self.assertRaisesRegex(ValueError, "Evaluation starts"):
            spatial_budget(spec, self.manifest)

    def test_tilted_camera_does_not_use_level_footprint_budget(self):
        self.spec["evaluation"]["pitch_roll_deg"] = 10
        with self.assertRaisesRegex(ValueError, "level cameras"):
            spatial_budget(self.spec, self.manifest)

    def test_texture_change_changes_run_fingerprint_and_blocks_resume(self):
        first = resolve_spec(self.spec, self.root)
        run = self.root / "run"
        prepare_run(run, first)
        self.texture.write_bytes(b"changed texture")
        second = resolve_spec(self.spec, self.root)
        self.assertNotEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(second["provenance"]["scene_bundle"]["changed"], ["terrain.png"])
        with self.assertRaisesRegex(ValueError, "changed; use a new run"):
            verify_run(run, second)
        with self.assertRaisesRegex(ValueError, "missing/changed"):
            require_assets(self.scene)

    def test_missing_texture_detected_without_importing_usd(self):
        self.texture.unlink()
        self.assertEqual(asset_provenance(self.scene)["missing"], ["terrain.png"])
        with self.assertRaises(ValueError):
            require_assets(self.scene)
        self.assertNotIn("pxr", sys.modules)

    def test_manifest_cannot_hide_changed_or_missing_entry(self):
        self.manifest["entry"] = "other.usda"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "entry mismatch"):
            require_assets(self.scene)

    def test_manifest_cannot_reference_outside_bundle(self):
        self.manifest["asset_sha256"]["../outside.png"] = "abc"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "relative"):
            asset_provenance(self.scene)


if __name__ == "__main__":
    unittest.main()
