"""Flat-scene asset integrity and conservative geometry checks without Kit."""

import json
import math
from pathlib import Path

from insect_utils.experiment_config import digest, relative_path


def asset_provenance(scene):
    """Rehash every declared dependency, rather than trusting the manifest."""
    scene = Path(scene)
    manifest_path = scene.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("entry") != scene.name:
        raise ValueError("Scene manifest schema/entry mismatch")
    expected = manifest["asset_sha256"]
    if scene.name not in expected or not expected:
        raise ValueError("Scene manifest does not include its entry USD")
    actual, missing, changed = {}, [], []
    for name, recorded in sorted(expected.items()):
        path = scene.parent / relative_path(name)
        try:
            path.resolve().relative_to(scene.parent.resolve())
        except ValueError:
            raise ValueError("Scene dependency escapes its bundle")
        actual[name] = digest(path) if path.is_file() else None
        if actual[name] is None:
            missing.append(name)
        elif actual[name] != recorded:
            changed.append(name)
    return {
        "manifest_sha256": digest(manifest_path),
        "asset_sha256": actual,
        "missing": missing,
        "changed": changed,
    }


def require_assets(scene):
    provenance = asset_provenance(scene)
    if provenance is None:
        raise ValueError("Scene dependency manifest is missing")
    if provenance["missing"] or provenance["changed"]:
        raise ValueError("Scene dependencies missing/changed: " + json.dumps(provenance))
    return json.loads(Path(scene).with_suffix(".manifest.json").read_text())


def spatial_budget(spec, manifest, safety_margin_m=5.0):
    """Account for every yaw of a rectangular footprint using its circumradius."""
    if spec["camera"]["mode"] != "downward_pinhole":
        raise ValueError("A finite flat footprint requires the downward pinhole mode")
    if spec["evaluation"]["pitch_roll_deg"] != 0:
        raise ValueError("Stage C flat-scene bounds support level cameras only")
    height = spec["flight"]["camera_height_agl_m"]
    width_px, height_px = spec["camera"]["raw_resolution_wh"]
    width = 2 * height * math.tan(math.radians(spec["camera"]["hfov_deg"]) / 2)
    footprint = [width, width * height_px / width_px]
    radius = math.hypot(*footprint) / 2
    inset = radius + safety_margin_m
    low, high = manifest["bounds_xy_m"]
    allowed = [[v + inset for v in low], [v - inset for v in high]]
    if any(a >= b for a, b in zip(*allowed)):
        raise ValueError("Scene is too small for footprint and safety margin")
    if not math.isclose(spec["scene"]["ground_z_m"], manifest["ground_z_m"], abs_tol=1e-6):
        raise ValueError("Configured ground height disagrees with scene")
    home = spec["home"]["ground_xy_m"]
    starts = spec["evaluation"]["radius_m"] * max(spec["evaluation"]["radius_scales"])
    available = min(home[i] - low[i] for i in range(2))
    available = min(available, *(high[i] - home[i] for i in range(2)))
    if starts + inset > available:
        raise ValueError("Evaluation starts plus footprint extend beyond valid ground")
    worst_radius = starts + spec["evaluation"]["max_steps"] * spec["evaluation"]["max_step_m"]
    return {
        "footprint_wh_m": footprint,
        "footprint_circumradius_m": radius,
        "safety_margin_m": safety_margin_m,
        "allowed_camera_center_xy_m": allowed,
        "max_evaluation_start_radius_m": starts,
        "worst_case_start_plus_actions_radius_m": worst_radius,
        "required_side_m_without_boundary_guard": 2 * (worst_radius + inset),
        "boundary_guard_required": worst_radius + inset > available,
        "boundary_guard_connected_to_controller": False,
        "camera_z_m": manifest["ground_z_m"] + height,
        "minimum_surface_clearance_m": height - manifest["max_surface_height_agl_m"],
    }


def check_positions_and_segments(positions, budget):
    """An axis-aligned rectangle is convex: valid endpoints imply valid segments."""
    low, high = budget["allowed_camera_center_xy_m"]
    points = list(positions)
    if not points:
        raise ValueError("No flight positions supplied")
    for index, point in enumerate(points):
        if len(point) != 3 or not all(math.isfinite(float(v)) for v in point):
            raise ValueError("Non-finite or invalid flight position")
        if not all(low[i] <= point[i] <= high[i] for i in range(2)):
            raise ValueError("Flight point/segment leaves valid ground at index " + str(index))
        if not math.isclose(point[2], budget["camera_z_m"], abs_tol=1e-6):
            raise ValueError("Flight position is not on the configured camera plane")
    if budget["minimum_surface_clearance_m"] < 1.0:
        raise ValueError("Less than 1 m vertical clearance above scene surfaces")
    return {"positions_checked": len(points), "segments_checked": len(points) - 1, "valid": True}
