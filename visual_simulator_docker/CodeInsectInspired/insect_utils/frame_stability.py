"""Approved RGB settling policy and a monotonic-frame acceptance counter."""

import math


STABILITY_POLICY = {
    "rgb_mae_limit": 0.5,
    "consecutive_changes": 2,
    "max_render_updates": 90,
    "render_subframes": 2,
}


class FrameStability:
    """Count consecutive small RGB changes; stale frames cannot advance the count."""

    def __init__(self):
        self.last_frame = -1
        self.history = []
        self.reset()

    def reset(self):
        self.has_baseline = False
        self.consecutive = 0
        self.last_mae = None

    def observe(self, frame, mae):
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError("Stability requires a nonnegative integer frame ID")
        if frame <= self.last_frame:
            return False
        if mae is not None and (
            isinstance(mae, bool)
            or not isinstance(mae, (int, float))
            or not math.isfinite(mae)
            or mae < 0
        ):
            raise ValueError("RGB mean absolute change must be finite and nonnegative")
        self.last_frame = frame
        if not self.has_baseline or mae is None:
            self.has_baseline = True
            self.consecutive = 0
            self.last_mae = None
        else:
            self.last_mae = float(mae)
            self.consecutive = (
                self.consecutive + 1 if mae <= STABILITY_POLICY["rgb_mae_limit"] else 0
            )
        self.history.append(
            {"frame": frame, "rgb_mae": self.last_mae, "consecutive_changes": self.consecutive}
        )
        return self.consecutive >= STABILITY_POLICY["consecutive_changes"]

    def summary(self):
        return {
            "policy": dict(STABILITY_POLICY),
            "consecutive_changes": self.consecutive,
            "last_rgb_mae": self.last_mae,
            "history": list(self.history),
        }
