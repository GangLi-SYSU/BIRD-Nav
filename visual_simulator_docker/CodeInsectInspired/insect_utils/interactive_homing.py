"""Pure state machine for manual outbound -> PI return -> visual return."""
import math

import numpy as np
from insect_utils.demo_pi_comparison import OdometryTrace


class DemoFlight:
    def __init__(self, sigma=0.1, seed=20261106):
        if not math.isfinite(sigma) or not 0 <= sigma <= 2:
            raise ValueError("Sigma must be between 0 and 2 m/sqrt(m)")
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)
        self.odometry = OdometryTrace(self.sigma, seed)
        self.position = np.array([0.0, 0.0, 0.3])
        self.estimated_xy = np.zeros(2)
        self.yaw = 0.0
        self.phase = "GROUND"
        self.paused = False
        self.visual_steps = 0
        self.outbound_length = 0.0
        self.pi_length = 0.0
        self.takeoff_elapsed = 0.0
        self.handoff_distance = None
        self.reason = ""
        self.homing_outcome = None
        self.landing_start = None
        self.landing_elapsed = 0.0
        self.landing_duration = 0.0
        self.landing_position = None

    def start_landing(self):
        """Descend at the homing endpoint; never correct horizontal error."""
        if self.phase not in ("SUCCESS", "FAILED"):
            return False
        self.homing_outcome = self.phase
        self.landing_start = self.position.copy()
        self.landing_elapsed = 0.0
        self.landing_duration = max(6.0, (self.position[2] - 0.3) / 2.0)
        self.phase = "LANDING"
        return True

    def landing_tick(self, dt):
        if self.phase != "LANDING" or self.paused:
            return
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("Landing dt must be finite and nonnegative")
        self.landing_elapsed = min(self.landing_duration, self.landing_elapsed + dt)
        fraction = self.landing_elapsed / self.landing_duration
        weight = fraction * fraction * (3 - 2 * fraction)
        self.position[2] = (1 - weight) * self.landing_start[2] + weight * 0.3
        if fraction >= 1:
            self.position[2] = 0.3
            self.landing_position = self.position.copy()
            self.phase = "LANDED"

    def landing_measurement(self):
        if self.landing_position is None:
            return None
        return dict(
            start_position_m=[0.0, 0.0, 0.3],
            landing_position_m=self.landing_position.tolist(),
            horizontal_distance_m=float(np.linalg.norm(self.landing_position[:2])),
            homing_outcome=self.homing_outcome,
            homing_reason=self.reason,
            visual_steps=self.visual_steps,
        )

    def takeoff(self):
        if self.phase == "GROUND":
            self.phase = "TAKEOFF"

    def altitude_tick(self, dt):
        if self.phase != "TAKEOFF" or self.paused:
            return
        self.takeoff_elapsed += dt
        fraction = min(1.0, self.takeoff_elapsed / 8)
        self.position[2] = 0.3 + 39.7 * fraction * fraction * (3 - 2 * fraction)
        if fraction == 1:
            self.phase = "MANUAL"

    def _move(self, vector, duration):
        vector = np.asarray(vector, dtype=float)
        length = float(np.linalg.norm(vector))
        if length < 1e-10:
            return
        # Same effective motion/odometry mismatch as original PI experiment:
        # integrated nominal displacement, uncertain actual displacement.
        actual = length + self.sigma * math.sqrt(length) * self.rng.standard_normal()
        target = self.position[:2] + vector / length * actual
        if np.max(np.abs(target)) > 3200:
            self.paused = True
            self.reason = "Scene boundary: paused. Return or reset."
            return
        self.position[:2] = target
        self.estimated_xy += vector
        if self.phase == "MANUAL":
            self.outbound_length += length
            self.odometry.outbound(vector, self.position[:2], self.estimated_xy, duration)
        else:
            self.pi_length += length
            self.odometry.return_step(self.position[:2], self.estimated_xy)

    def manual_tick(self, forward, left, turn, dt):
        if self.phase != "MANUAL" or self.paused:
            return
        self.yaw = (self.yaw + turn * 45 * dt) % 360
        command = np.array([forward, left], dtype=float)
        norm = np.linalg.norm(command)
        if norm == 0:
            self.odometry.outbound([0, 0], self.position[:2], self.estimated_xy, dt)
            return
        command /= max(1, norm)
        angle = math.radians(self.yaw)
        rotation = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        self._move(rotation @ command * 2.5 * dt, dt)

    def start_homing(self):
        if self.phase == "MANUAL":
            self.phase = "PI_RETURN"
            self.paused = False
            self.reason = ""
            self.odometry.start_return(self.position[:2], self.estimated_xy)

    def pi_tick(self, dt):
        if self.phase != "PI_RETURN" or self.paused:
            return
        distance = float(np.linalg.norm(self.estimated_xy))
        if distance > 1e-8:
            direction = -self.estimated_xy / distance
            self.yaw = math.degrees(math.atan2(direction[1], direction[0])) % 360
            length = min(4 * dt, distance)
            self._move(direction * length, length / 4)
        if np.linalg.norm(self.estimated_xy) <= 1e-8 and not self.paused:
            self.phase = "VISUAL_RETURN"
            self.handoff_distance = float(np.linalg.norm(self.position[:2]))

    def visual_status(self):
        if self.phase != "VISUAL_RETURN":
            return self.phase
        # Simulator scoring only: no truth vector enters the neural action.
        if np.linalg.norm(self.position[:2]) <= 2:
            self.phase, self.reason = "SUCCESS", "Simulator arrival radius <= 2 m"
        elif self.visual_steps >= 50:
            self.phase, self.reason = "FAILED", "Visual controller reached 50 steps"
        return self.phase

    def apply_visual_motion(self, motion):
        if self.phase != "VISUAL_RETURN" or self.paused:
            return
        target = self.position[:2] + motion["displacement_world_m"]
        if np.max(np.abs(target)) > 190:
            self.phase, self.reason = "FAILED", "Visual scene boundary reached"
            return
        self.position[:2] = target
        self.yaw = motion["next_yaw_deg"]
        self.visual_steps += 1
        self.visual_status()

    def state(self):
        return dict(
            phase=self.phase,
            paused=self.paused,
            position_m=self.position.tolist(),
            yaw_deg=self.yaw,
            estimated_xy_m=self.estimated_xy.tolist(),
            true_distance_m=float(np.linalg.norm(self.position[:2])),
            estimated_distance_m=float(np.linalg.norm(self.estimated_xy)),
            sigma=self.sigma,
            visual_steps=self.visual_steps,
            outbound_length_m=self.outbound_length,
            pi_length_m=self.pi_length,
            handoff_distance_m=self.handoff_distance,
            reason=self.reason,
            homing_outcome=self.homing_outcome,
            landing_measurement=self.landing_measurement(),
        )
