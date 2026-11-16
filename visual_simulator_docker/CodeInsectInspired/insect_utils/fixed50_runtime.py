"""Isolated50-action adapter of the frozen navigation loop.

The observation,motion,and logging path is preserved. Arrival termination and
phase summaries are replaced explicitly; historical source files stay unchanged.
"""

import json
import math
import os
from pathlib import Path
import signal
import time
import uuid

from insect_utils.camera_geometry import home_vector_body
from insect_utils.experiment_config import PROJECT, digest
from insect_utils.experiment_state import atomic_json, now
from insect_utils.fixed50_metrics import (
    COMPLETED,
    commit_fixed_phase,
    complete_fixed_episode,
    fixed_termination,
)
from insect_utils.navigation_core import (
    InvalidPrediction,
    decide_motion,
    endpoint,
    horizontal_distance,
    safe_segment,
    segment_min_distance,
    shortest_angle_error,
)
from insect_utils.navigation_protocol import verify_navigation
from insect_utils.navigation_results import saved_episodes
from insect_utils.navigation_runtime import (
    apply_pose,
    load_predictor,
    new_episode,
    observation_receipt,
    predict_rgb,
    save_observation,
)

BASE_RUNTIME_SHA256 = "577dcecd28aea78901542d04ac7743d91c5fd64ca9e1eb58160afee2f9d6f45d"


def run_fixed_episodes(folder, phase, frozen, cameras, model, settings, stop):
    from insect_utils.downward_camera import capture_batch

    protocol, budget = frozen["protocol"], frozen["budget"]
    home, controller = protocol["home_camera_xyz_m"], protocol["controller"]
    completed = saved_episodes(folder, phase, frozen)
    planned = frozen["episodes"][phase]
    expected = {e["episode_id"] for e in planned}
    if not set(completed) <= expected:
        raise ValueError("Previously saved episodes do not match the frozen test plan")
    pending = iter([e for e in planned if e["episode_id"] not in completed])
    states = [None] * len(cameras)
    exhausted = False
    session = uuid.uuid4().hex
    started = time.perf_counter()
    observations_seen = 0
    while True:
        if stop:
            raise InterruptedError("Navigation stopped; incomplete attempts are preserved")
        for slot in range(len(cameras)):
            if states[slot] is None and not exhausted:
                episode = next(pending, None)
                if episode is None:
                    exhausted = True
                    break
                state = new_episode(folder, phase, episode, session, slot)
                states[slot] = state
                reason = fixed_termination(state["position"], home, 0, controller)
                if reason:
                    result = complete_fixed_episode(folder, phase, state, reason, frozen)
                    completed[result["episode_id"]] = result
                    states[slot] = None
        active = [(slot, s) for slot, s in enumerate(states) if s is not None]
        if not active:
            if exhausted:
                break
            continue
        requests = [
            (
                cameras[slot],
                state["position"],
                state["yaw"],
                "{}_step_{:03d}".format(state["episode"]["episode_id"], len(state["actions"])),
            )
            for slot, state in active
        ]
        try:
            observations = capture_batch(requests)
        except Exception as error:
            for slot, state in active:
                atomic_json(
                    state["attempt"] / "invalid_frame.json",
                    {
                        "reason": "invalid_frame",
                        "error": str(error),
                        "completed_steps": len(state["actions"]),
                        "episode_incomplete": True,
                    },
                )
            atomic_json(
                Path(folder) / phase / "frame_failure.json",
                {
                    "updated": now(),
                    "error": str(error),
                    "engineering_failure": True,
                    "episodes_affected": [s["episode"]["episode_id"] for _, s in active],
                },
            )
            raise RuntimeError(
                "Frame acquisition failed; preserve attempts and stop for diagnosis"
            ) from error
        predictions, infer_seconds = (
            predict_rgb(model, observations)
            if model is not None
            else ([home_vector_body(s["position"], home, s["yaw"]) for _, s in active], 0.0)
        )
        for (slot, state), request, (rgb, receipt), prediction in zip(
            active, requests, observations, predictions
        ):
            measured = observation_receipt(
                cameras[slot],
                state["position"],
                state["yaw"],
                request[3],
                receipt,
                protocol,
                settings,
            )
            before = [measured[k] for k in ("x", "y", "z")]
            yaw_before = measured["yaw"]
            image = state["attempt"] / "step_{:03d}.png".format(len(state["actions"]))
            sha = save_observation(image, rgb, receipt)
            try:
                motion = decide_motion(prediction, yaw_before, controller)
            except InvalidPrediction as error:
                atomic_json(
                    state["attempt"] / "invalid_prediction.json",
                    {
                        "reason": str(error),
                        "prediction_repr": repr(prediction),
                        "image": image.name,
                        "sha256": sha,
                    },
                )
                result = complete_fixed_episode(folder, phase, state, str(error), frozen)
                completed[result["episode_id"]] = result
                states[slot] = None
                continue
            requested = endpoint(before, motion)
            try:
                safe_segment(before, requested, budget)
            except ValueError as error:
                atomic_json(
                    state["attempt"] / "rejected_motion.json",
                    {"motion": motion, "requested_endpoint_m": requested, "reason": str(error)},
                )
                result = complete_fixed_episode(
                    folder, phase, state, "out_of_bounds", frozen, str(error)
                )
                completed[result["episode_id"]] = result
                states[slot] = None
                continue
            after, yaw_after, actual_matrix = apply_pose(
                cameras[slot], requested, motion["next_yaw_deg"]
            )
            # Truth is computed for logging after the action has already been chosen.
            truth = home_vector_body(before, home, yaw_before)
            truth_world = [home[0] - before[0], home[1] - before[1]]
            move = [after[0] - before[0], after[1] - before[1]]
            record = dict(
                motion,
                step=len(state["actions"]) + 1,
                session_id=session,
                camera_slot=slot,
                image=str(image.relative_to(Path(folder))),
                image_sha256=sha,
                pose_id=request[3],
                rendering_frame=receipt["rendering_frame"],
                position_before_m=before,
                yaw_before_deg=yaw_before,
                executed_position_after_m=after,
                executed_yaw_after_deg=yaw_after,
                executed_camera_world_matrix=actual_matrix,
                executed_displacement_world_m=move,
                truth_body_m_evaluation_only=truth,
                distance_before_m=horizontal_distance(before, home),
                distance_after_m=horizontal_distance(after, home),
                segment_min_distance_m=segment_min_distance(before, after, home),
                prediction_vector_error_m=math.hypot(
                    prediction[0] - truth[0], prediction[1] - truth[1]
                ),
                prediction_angle_error_deg=shortest_angle_error(prediction, truth),
                executed_action_angle_error_deg=shortest_angle_error(move, truth_world),
                batch_inference_seconds=infer_seconds,
            )
            with (state["attempt"] / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            state["actions"].append(record)
            state.update(position=after, yaw=yaw_after)
            reason = fixed_termination(after, home, len(state["actions"]), controller)
            if reason:
                result = complete_fixed_episode(folder, phase, state, reason, frozen)
                completed[result["episode_id"]] = result
                states[slot] = None
                print(
                    "EPISODE {} reason={} steps={} final_distance={:.4f}".format(
                        result["episode_id"], reason, result["steps"], result["final_distance_m"]
                    ),
                    flush=True,
                )
        observations_seen += len(observations)
        progress = {
            "phase": phase,
            "status": "running",
            "updated": now(),
            "episodes_completed": len(completed),
            "episodes_total": len(planned),
            "completed50": sum(r["reason"] == COMPLETED for r in completed.values()),
            "ever_entered_2m": sum(
                r["detail"]["fixed_horizon"]["ever_entered_2m"] for r in completed.values()
            ),
            "final_within_2m": sum(
                r["reason"] == COMPLETED and r["final_distance_m"] <= 2 for r in completed.values()
            ),
            "active_episode_steps": {
                s["episode"]["episode_id"]: len(s["actions"]) for s in states if s is not None
            },
            "active_episode_attempts": {
                s["episode"]["episode_id"]: str(s["attempt"].relative_to(Path(folder)))
                for s in states
                if s is not None
            },
            "distance_based_termination": False,
            "horizon_actions": 50,
            "session_observations": observations_seen,
            "session_elapsed_s": time.perf_counter() - started,
            "success_radius_m": controller["success_radius_m"],
        }
        atomic_json(Path(folder) / "progress.json", progress)
    return commit_fixed_phase(folder, phase, frozen)


def run_fixed(folder):
    folder = Path(folder)
    frozen = verify_navigation(folder)
    if digest(PROJECT / "insect_utils/navigation_runtime.py") != BASE_RUNTIME_SHA256:
        raise ValueError("The shared navigation implementation changed")
    if frozen["protocol"]["termination_policy"]["distance_based"] is not False:
        raise ValueError("Fixed50 rendering must disable distance-based stopping")
    from omni.isaac.kit import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "anti_aliasing": 0})
    cameras, stop = [], []
    signal.signal(signal.SIGTERM, lambda signum, _: stop.append(signum))
    signal.signal(signal.SIGINT, lambda signum, _: stop.append(signum))
    try:
        import carb
        import omni.usd
        import omni.replicator.core as rep
        from omni.isaac.core import World
        from insect_utils.downward_camera import DownwardCamera
        from insect_utils.downward_capture import configure_capture_rendering

        omni.usd.get_context().open_stage(str(PROJECT / frozen["protocol"]["scene"]))
        world = World(stage_units_in_meters=1.0)
        world.reset()
        for index in range(frozen["protocol"]["runtime"]["camera_count"]):
            cameras.append(
                DownwardCamera(
                    frozen["protocol"]["camera"],
                    world,
                    "replicator",
                    name="NavigationCamera_{:02d}".format(index),
                )
            )
        settings = carb.settings.get_settings()
        configure_capture_rendering(settings, rep.settings)
        run_fixed_episodes(
            folder, "formal", frozen, cameras, load_predictor(frozen), settings, stop
        )
        verify_navigation(folder)
    except BaseException as error:
        atomic_json(
            folder / ("worker_error_" + uuid.uuid4().hex[:10] + ".json"),
            dict(error=str(error), updated=now()),
        )
        raise
    finally:
        for camera in reversed(cameras):
            camera.close()
        app.close()
