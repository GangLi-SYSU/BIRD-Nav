"""Exact capture scheduling, measured-pose labels and flat-plane coverage math."""

import math

from insect_utils.camera_geometry import (
    body_to_world,
    home_vector_body,
    number,
    pixel_to_flat_world,
    usd_world_matrix,
    xyz,
)
from insect_utils.frame_stability import STABILITY_POLICY


def exact_batches(samples, camera_count):
    """Yield only occupied camera slots, including a partially filled final batch."""
    if isinstance(camera_count, bool) or not isinstance(camera_count, int) or camera_count < 1:
        raise ValueError("camera_count must be a positive integer")
    for start in range(0, len(samples), camera_count):
        yield list(enumerate(samples[start : start + camera_count]))


def make_samples(positions, headings, home, ground_z):
    """Pair each position with its real requested headings, independent of cameras."""
    home, ground_z = xyz(home), number(ground_z)
    if len(positions) != len(headings) or not positions:
        raise ValueError("Every position needs a nonempty list of headings")
    samples = []
    for position_index, (position, yaws) in enumerate(zip(positions, headings)):
        position = xyz(position)
        if not yaws or position[2] <= ground_z:
            raise ValueError("Expected headings and an optical center above ground")
        group = "position_{:06d}".format(position_index)
        for heading_index, yaw in enumerate(yaws):
            index = len(samples)
            yaw = number(yaw) % 360
            samples.append(
                {
                    "sample_id": "sample_{:06d}".format(index),
                    "pose_id": "pose_{:06d}".format(index),
                    "position_id": group,
                    "heading_index": heading_index,
                    "image": "Replicator/rgb/rgb_{:06d}.png".format(index),
                    "planned_position_m": position,
                    "planned_body_yaw_deg": yaw,
                    "planned_target_body_m": home_vector_body(position, home, yaw),
                    "split_group": group,
                    "role": "learning",
                }
            )
    return samples


def measured_record(sample, receipt, home, ground_z, required_agl):
    """Derive labels from the renderer's measured matrix and reject mismatched frames."""
    if receipt["pose_id"] != sample["pose_id"]:
        raise ValueError("Image/pose ID mismatch")
    stability = receipt["stability"]
    history = stability["history"]
    if (
        stability["policy"] != STABILITY_POLICY
        or stability["consecutive_changes"] < 2
        or len(history) < 3
        or any(a["frame"] >= b["frame"] for a, b in zip(history, history[1:]))
        or any(
            item["rgb_mae"] is None or not 0 <= number(item["rgb_mae"]) <= 0.5
            for item in history[-2:]
        )
        or receipt["rendering_frame"] != history[-1]["frame"]
        or not 1 <= receipt["updates_waited"] <= 90
    ):
        raise ValueError("Image has no valid settled-frame evidence")
    matrix = receipt["camera_world_matrix"]
    expected = usd_world_matrix(sample["planned_position_m"], sample["planned_body_yaw_deg"])
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("Invalid measured camera matrix")
    if any(
        abs(number(a) - b) > 1e-4
        for row, wanted in zip(matrix, expected)
        for a, b in zip(row, wanted)
    ):
        raise ValueError("Measured camera pose differs from planned pose")
    position = xyz(matrix[3][:3])
    yaw = (math.degrees(math.atan2(matrix[0][1], matrix[0][0])) + 90) % 360
    agl = position[2] - number(ground_z)
    if not math.isclose(agl, number(required_agl), abs_tol=1e-5, rel_tol=0):
        raise ValueError("Camera AGL differs from the required flight height")
    target = home_vector_body(position, home, yaw)
    restored = body_to_world(target, yaw)
    if any(abs(position[i] + restored[i] - home[i]) > 1e-6 for i in range(2)):
        raise ValueError("Measured label does not reconstruct home")
    return dict(
        sample,
        frame_id=receipt["rendering_frame"],
        x=position[0],
        y=position[1],
        z=position[2],
        yaw=yaw,
        target_x=target[0],
        target_y=target[1],
        ground_z_m=ground_z,
        agl_m=agl,
    )


def footprint(position, yaw, ground_z, settings):
    width, height = settings["resolution_wh"]
    return [
        pixel_to_flat_world(pixel, position, yaw, ground_z, settings)[:2]
        # Reversing image winding produces counterclockwise world-plane corners.
        for pixel in [(0, 0), (0, height), (width, height), (width, 0)]
    ]


def polygon_area(polygon):
    return (
        abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(polygon, polygon[1:] + polygon[:1]))) / 2
    )


def cross(a, b, point):
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


def contains(polygon, point):
    signs = [cross(a, b, point) for a, b in zip(polygon, polygon[1:] + polygon[:1])]
    return all(s >= -1e-9 for s in signs) or all(s <= 1e-9 for s in signs)


def intersection_area(subject, clip):
    """Clip convex, counterclockwise ground footprints without a geometry dependency."""
    output = list(subject)
    for a, b in zip(clip, clip[1:] + clip[:1]):
        points, output = output, []
        if not points:
            break
        previous = points[-1]
        for current in points:
            before, after = cross(a, b, previous), cross(a, b, current)
            if (before >= 0) != (after >= 0):
                fraction = before / (before - after)
                output.append(
                    [previous[i] + fraction * (current[i] - previous[i]) for i in range(2)]
                )
            if after >= 0:
                output.append(current)
            previous = current
    return polygon_area(output) if output else 0.0


def overlap_report(samples, ground_z, settings):
    polygons = [
        footprint(s["planned_position_m"], s["planned_body_yaw_deg"], ground_z, settings)
        for s in samples
    ]
    # make_samples groups headings by position: use the first heading for path steps.
    first = [i for i, sample in enumerate(samples) if sample["heading_index"] == 0]
    adjacent = []
    for i, j in zip(first, first[1:]):
        distance = math.dist(
            samples[i]["planned_position_m"][:2], samples[j]["planned_position_m"][:2]
        )
        adjacent.append(
            {
                "from": samples[i]["position_id"],
                "to": samples[j]["position_id"],
                "distance_m": distance,
                "footprint_area_overlap_fraction": intersection_area(polygons[i], polygons[j])
                / polygon_area(polygons[i]),
            }
        )
    return polygons, adjacent
