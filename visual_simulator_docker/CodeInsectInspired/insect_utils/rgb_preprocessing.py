"""Shared full-frame downward RGB contract for disk images and online RGBA."""


def rgb_array(image, expected_wh=None):
    import numpy as np

    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError("Downward RGB input must be uint8")
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("Downward RGB input needs three RGB or four RGBA channels")
    if expected_wh is not None and [array.shape[1], array.shape[0]] != list(expected_wh):
        raise ValueError("Downward RGB dimensions differ from the camera contract")
    return np.ascontiguousarray(array[:, :, :3])


def rgb_tensor(image, expected_wh=None):
    """Return float32 CHW without cropping, resizing, wrapping or alpha blending."""
    import torch

    array = rgb_array(image, expected_wh)
    return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous().to(torch.float32).div_(255)


def load_rgb_tensor(path, expected_wh=None, expected_sha256=None):
    import hashlib
    from io import BytesIO
    from pathlib import Path
    from PIL import Image

    path = Path(path)
    if path.is_symlink():
        raise ValueError("Expected a regular captured PNG")
    data = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Captured PNG hash changed: " + path.name)
    with Image.open(BytesIO(data)) as image:
        if image.mode not in ("RGB", "RGBA"):
            raise ValueError("Captured PNG is not RGB/RGBA")
        return rgb_tensor(image, expected_wh)


def require_downward_config(config):
    if config.get("camera_mode") != "downward_pinhole":
        raise ValueError("Expected explicit downward_pinhole data configuration")
    if config.get("augment", False):
        raise ValueError(
            "Downward data uses real headings; panorama/rotation augmentation is disabled"
        )
    if not config.get("prepared_data_dir"):
        raise ValueError("Downward data requires a verified stage F preparation directory")


def body_direction_image(vector):
    """Image right=-body Y and image down=-body X; return a direction, not range."""
    import math

    x, y = [float(v) for v in vector]
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Direction vector must be finite")
    norm = math.hypot(x, y)
    return None if norm <= 1e-8 else [-y / norm, -x / norm]
