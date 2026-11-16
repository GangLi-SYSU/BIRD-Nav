"""Approved ideal nadir camera contract and coordinate math, without Isaac imports."""

import math


CONTRACT = {
    "optical_model": "ideal_pinhole_no_distortion",
    "mount": "body_fixed_nadir",
    "body_axes": "X forward, Y left, Z up; positive yaw from world X toward world Y",
    "image_axes": "right = body -Y; up = body +X; optical axis = body -Z",
    "height_reference": "camera optical center above flat ground",
    "mount_offset_body_m": [0.0, 0.0, 0.0],
    "roll_pitch_deg": [0.0, 0.0],
}


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Camera geometry requires finite numeric values")
    return float(value)


def xyz(value):
    if len(value) != 3:
        raise ValueError("Camera position must contain three coordinates")
    return [number(v) for v in value]


def optics(camera, meters_per_unit=1.0, focal_length_mm=100.0):
    if camera["mode"] != "downward_pinhole" or camera["crop_box"] is not None:
        raise ValueError("Expected full-frame downward pinhole camera")
    if not camera["fov_definition_confirmed"]:
        raise ValueError("Camera HFOV definition must be confirmed before calibration/capture")
    width, height = camera["raw_resolution_wh"]
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, height)):
        raise ValueError("Resolution must contain two positive integers")
    hfov = number(camera["hfov_deg"])
    units, focal_mm = number(meters_per_unit), number(focal_length_mm)
    if not 0 < hfov < 179 or units <= 0 or focal_mm <= 0:
        raise ValueError("Invalid HFOV, stage units or focal length")
    aperture_mm = 2 * focal_mm * math.tan(math.radians(hfov) / 2)
    fx = width / (2 * math.tan(math.radians(hfov) / 2))
    # USD filmback/lens values are tenths of world units, not millimeters.
    return {
        "resolution_wh": [width, height],
        "hfov_deg": hfov,
        "vfov_deg": math.degrees(2 * math.atan(math.tan(math.radians(hfov) / 2) * height / width)),
        "focal_length_mm": focal_mm,
        "sensor_size_mm": [aperture_mm, aperture_mm * height / width],
        "usd_focal_length": focal_mm / (100 * units),
        "usd_aperture_wh": [
            aperture_mm / (100 * units),
            aperture_mm * height / width / (100 * units),
        ],
        "intrinsics": [[fx, 0.0, width / 2], [0.0, fx, height / 2], [0.0, 0.0, 1.0]],
        "meters_per_unit": units,
        "clipping_range_m": [0.05, 1000.0],
        "depth_of_field": False,
        "distortion": None,
        "lens_scale_purpose": "virtual projection only; not a real hardware lens calibration",
    }


def usd_quaternion_wxyz(yaw_deg):
    angle = math.radians(number(yaw_deg) - 90) / 2
    return [math.cos(angle), 0.0, 0.0, math.sin(angle)]


def body_yaw_from_quaternion(quaternion):
    if len(quaternion) != 4:
        raise ValueError("Expected body quaternion in wxyz order")
    w, x, y, z = [number(v) for v in quaternion]
    if abs(x) > 1e-6 or abs(y) > 1e-6 or not math.isclose(w * w + z * z, 1, abs_tol=1e-5):
        raise ValueError("Approved camera adapter supports level bodies only")
    return math.degrees(2 * math.atan2(z, w))


def usd_world_matrix(position, yaw_deg):
    x, y, z = xyz(position)
    angle = math.radians(number(yaw_deg) - 90)
    c, s = math.cos(angle), math.sin(angle)
    # Gf uses row vectors: rows 0/1/2 are camera basis vectors in world coordinates.
    return [[c, s, 0, 0], [-s, c, 0, 0], [0, 0, 1, 0], [x, y, z, 1]]


def body_to_world(vector, yaw_deg):
    x, y = [number(v) for v in vector]
    angle = math.radians(number(yaw_deg))
    c, s = math.cos(angle), math.sin(angle)
    return [c * x - s * y, s * x + c * y]


def home_vector_body(position, home, yaw_deg):
    position, home = xyz(position), xyz(home)
    return body_to_world([home[0] - position[0], home[1] - position[1]], -number(yaw_deg))


def world_to_pixel(point, position, yaw_deg, settings):
    point, position = xyz(point), xyz(position)
    depth = position[2] - point[2]
    if depth <= 0:
        raise ValueError("Point is behind the nadir camera")
    matrix = usd_world_matrix(position, yaw_deg)
    delta = [a - b for a, b in zip(point, position)]
    right = sum(a * b for a, b in zip(delta, matrix[0][:3]))
    up = sum(a * b for a, b in zip(delta, matrix[1][:3]))
    k = settings["intrinsics"]
    return [k[0][2] + k[0][0] * right / depth, k[1][2] - k[1][1] * up / depth]


def pixel_to_flat_world(pixel, position, yaw_deg, ground_z, settings):
    position = xyz(position)
    depth = position[2] - number(ground_z)
    if depth <= 0:
        raise ValueError("Camera must be above the projected plane")
    u, v = [number(x) for x in pixel]
    k = settings["intrinsics"]
    right, up = (u - k[0][2]) * depth / k[0][0], -(v - k[1][2]) * depth / k[1][1]
    matrix = usd_world_matrix(position, yaw_deg)
    return [position[i] + right * matrix[0][i] + up * matrix[1][i] for i in range(2)] + [ground_z]
