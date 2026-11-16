"""Pure geometry and termination rules for the fixed0.5m navigation protocol."""

import math

from insect_utils.camera_geometry import body_to_world, home_vector_body, xyz


class InvalidPrediction(ValueError):
    pass


def horizontal_distance(position, home):
    return math.hypot(position[0] - home[0], position[1] - home[1])


def make_episodes(group, home, prefix):
    episodes = []
    for radius in group["radii_m"]:
        for bearing_index in range(group["bearings_per_radius"]):
            bearing = bearing_index * 360 / group["bearings_per_radius"]
            angle = math.radians(bearing)
            for heading_index, yaw in enumerate(group["initial_yaws_deg"]):
                episodes.append(
                    {
                        "episode_id": "{}_r{:g}_b{:02d}_h{}".format(
                            prefix, radius, bearing_index, heading_index
                        ),
                        "radius_m": radius,
                        "bearing_deg": bearing,
                        "initial_yaw_deg": yaw,
                        "heading_index": heading_index,
                        "initial_position_m": [
                            home[0] + radius * math.cos(angle),
                            home[1] + radius * math.sin(angle),
                            home[2],
                        ],
                    }
                )
    return episodes


def decide_motion(prediction_body_m, yaw_deg, controller):
    """The neural decision uses only the predicted vector and current heading."""
    if len(prediction_body_m) != 2:
        raise InvalidPrediction("invalid_prediction_shape")
    prediction = [float(v) for v in prediction_body_m]
    if not all(math.isfinite(v) for v in prediction):
        raise InvalidPrediction("invalid_prediction_nonfinite")
    magnitude = math.hypot(*prediction)
    if not math.isfinite(magnitude):
        raise InvalidPrediction("invalid_prediction_nonfinite")
    if magnitude <= controller["minimum_prediction_norm_m"]:
        raise InvalidPrediction("invalid_prediction_zero")
    world = body_to_world(prediction, yaw_deg)
    step = min(
        controller["step_offset_m"] + controller["step_gain"] * magnitude, controller["max_step_m"]
    )
    move = [v / magnitude * step for v in world]
    new_yaw = (
        yaw_deg if controller["fixed_yaw"] else math.degrees(math.atan2(world[1], world[0])) % 360
    )
    return {
        "prediction_body_m": prediction,
        "prediction_world_m": world,
        "predicted_distance_m": magnitude,
        "step_size_m": step,
        "displacement_world_m": move,
        "next_yaw_deg": new_yaw,
    }


def endpoint(position, motion):
    return [
        position[0] + motion["displacement_world_m"][0],
        position[1] + motion["displacement_world_m"][1],
        position[2],
    ]


def termination(position, home, steps, controller):
    if horizontal_distance(position, home) <= controller["success_radius_m"]:
        return "initial_within_success" if steps == 0 else "success"
    if steps >= controller["max_steps"]:
        return "timeout"
    return None


def segment_min_distance(start, end, home):
    dx, dy = end[0] - start[0], end[1] - start[1]
    square = dx * dx + dy * dy
    ratio = (
        0
        if square == 0
        else max(0, min(1, ((home[0] - start[0]) * dx + (home[1] - start[1]) * dy) / square))
    )
    return math.hypot(start[0] + ratio * dx - home[0], start[1] + ratio * dy - home[1])


def shortest_angle_error(prediction, truth):
    if min(math.hypot(*prediction), math.hypot(*truth)) <= 1e-8:
        return None
    return abs(
        math.degrees(
            math.atan2(
                prediction[0] * truth[1] - prediction[1] * truth[0],
                prediction[0] * truth[0] + prediction[1] * truth[1],
            )
        )
    )


def safe_segment(start, end, budget):
    from insect_utils.scene_geometry import check_positions_and_segments

    return check_positions_and_segments([start, end], budget)


def simulate_oracle(episode, home, controller, budget=None):
    position = xyz(episode["initial_position_m"])
    yaw = episode["initial_yaw_deg"]
    positions = [position]
    while True:
        reason = termination(position, home, len(positions) - 1, controller)
        if reason:
            return {
                "reason": reason,
                "positions": positions,
                "steps": len(positions) - 1,
                "final_distance_m": horizontal_distance(position, home),
            }
        motion = decide_motion(home_vector_body(position, home, yaw), yaw, controller)
        next_position = endpoint(position, motion)
        if budget is not None:
            safe_segment(position, next_position, budget)
        position, yaw = next_position, motion["next_yaw_deg"]
        positions.append(position)
