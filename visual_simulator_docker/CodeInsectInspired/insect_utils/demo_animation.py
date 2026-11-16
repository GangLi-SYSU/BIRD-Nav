"""Display interpolation; neural action endpoints and success rules stay unchanged."""
import math
import numpy as np


class SmoothAction:
    def __init__(self, position, yaw, motion, speed=1.2):
        self.start = np.asarray(position, dtype=float).copy()
        self.target = self.start.copy()
        self.target[:2] += motion["displacement_world_m"]
        self.yaw = float(yaw)
        self.turn = (motion["next_yaw_deg"] - yaw + 180) % 360 - 180
        self.motion = motion
        self.duration = max(1.2, motion["step_size_m"] / speed, abs(self.turn) / 65)
        self.elapsed = 0.0

    def advance(self, dt, paused=False):
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("Animation dt must be finite and nonnegative")
        if not paused:
            self.elapsed = min(self.duration, self.elapsed + dt)
        t = self.elapsed / self.duration
        weight = t * t * (3 - 2 * t)
        position = (1 - weight) * self.start + weight * self.target
        return position, (self.yaw + weight * self.turn) % 360, t >= 1


class LiveSettler:
    """Poll one live RGB frame at a time without a nested render loop."""

    def __init__(self, sensor, position, yaw):
        from insect_utils.camera_geometry import usd_world_matrix
        from insect_utils.frame_stability import FrameStability

        self.expected = np.asarray(usd_world_matrix(position, yaw))
        self.initial_frame = sensor._frame()
        self.last_frame = self.initial_frame
        self.pose_frame = None
        self.previous = None
        self.stability = FrameStability()
        self.updates = 0

    def poll(self, sensor):
        frame = sensor._frame()
        if frame <= self.last_frame:
            return None
        self.last_frame = frame
        self.updates += 1
        if self.updates > 90:
            raise RuntimeError("Live sensor did not settle within 90 fresh frames")
        params = sensor.params.get_data()
        reference = sensor.reference.get_data()
        if not params or not reference or int(reference["referenceTimeNumerator"]) != frame:
            self.previous = None
            self.stability.reset()
            return None
        matrix = np.linalg.inv(np.asarray(params["cameraViewTransform"]).reshape(4, 4))
        if not np.allclose(matrix, self.expected, atol=1e-4, rtol=0):
            self.previous = None
            self.pose_frame = None
            self.stability.reset()
            return None
        if self.pose_frame is None:
            self.pose_frame = frame
            return None
        raw = np.asarray(sensor.rgb.get_data()).copy()
        if raw.shape != (1024, 1024, 4) or raw.dtype != np.uint8 or sensor._frame() != frame:
            return None
        rgb = np.ascontiguousarray(raw[:, :, :3])
        if not np.any(rgb):
            return None
        mae = (
            None
            if self.previous is None
            else float(np.abs(rgb.astype(float) - self.previous).mean())
        )
        self.previous = rgb.astype(float)
        if not self.stability.observe(frame, mae):
            return None
        for key, value in dict(
            focalLength=sensor.settings["usd_focal_length"],
            horizontalAperture=sensor.settings["usd_aperture_wh"][0],
            verticalAperture=sensor.settings["usd_aperture_wh"][1],
            fStop=0,
        ).items():
            if abs(float(sensor.prim.GetAttribute(key).Get()) - value) > 1e-6:
                raise RuntimeError("Sensor optics changed: " + key)
        return (
            rgb,
            dict(
                rendering_frame=frame,
                stability=self.stability.summary(),
                pose_matched=True,
                live_updates=self.updates,
            ),
        )
