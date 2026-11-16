"""Real settled-frame navigation using four independent kinematic camera proxies."""

import json
import math
import os
from pathlib import Path
import signal
import time
import uuid

from insect_utils.camera_geometry import home_vector_body, usd_world_matrix
from insect_utils.capture_geometry import measured_record
from insect_utils.experiment_config import PROJECT, digest
from insect_utils.experiment_state import atomic_json, now
from insect_utils.navigation_core import InvalidPrediction, decide_motion, endpoint
from insect_utils.navigation_core import horizontal_distance, safe_segment, segment_min_distance
from insect_utils.navigation_core import shortest_angle_error, termination
from insect_utils.navigation_protocol import verify_navigation
from insect_utils.navigation_results import commit_phase, complete_episode, saved_episodes


def observation_receipt(camera, position, yaw, pose_id, receipt, protocol, settings):
    from insect_utils.downward_capture import require_capture_rendering

    sample = {"pose_id": pose_id, "planned_position_m": position, "planned_body_yaw_deg": yaw}
    measured = measured_record(
        sample,
        receipt,
        protocol["home_camera_xyz_m"],
        protocol["ground_z_m"],
        protocol["home_camera_xyz_m"][2] - protocol["ground_z_m"],
    )
    if receipt["backend"] != "replicator" or receipt["camera_prim"] != camera.path:
        raise ValueError("Navigation frame came from the wrong camera")
    receipt["observed_render_settings"] = require_capture_rendering(settings)
    return measured


def apply_pose(camera, position, yaw):
    """Execute a kinematic state update and measure its actual USD world transform."""
    import numpy as np
    from pxr import UsdGeom

    camera.set_body_pose(position, yaw)
    actual = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(camera.prim), dtype=float)
    if not np.allclose(actual, usd_world_matrix(position, yaw), atol=1e-5, rtol=0):
        raise RuntimeError("Executed camera pose differs from the requested motion")
    measured_yaw = (math.degrees(math.atan2(actual[0, 1], actual[0, 0])) + 90) % 360
    return actual[3, :3].tolist(), measured_yaw, actual.tolist()


def save_observation(path, rgb, receipt):
    from PIL import Image

    with path.open("xb") as stream:
        Image.fromarray(rgb).save(stream, format="PNG")
        stream.flush()
        os.fsync(stream.fileno())
    atomic_json(path.with_suffix(".receipt.json"), receipt)
    return digest(path)


def new_episode(folder, phase, episode, session, slot):
    attempt = (
        Path(folder) / phase / "episodes" / episode["episode_id"] / "attempts" / uuid.uuid4().hex
    )
    attempt.mkdir(parents=True, exist_ok=False)
    atomic_json(
        attempt / "start.json",
        {"episode": episode, "session_id": session, "camera_slot": slot, "started": now()},
    )
    return {
        "episode": episode,
        "position": list(episode["initial_position_m"]),
        "yaw": episode["initial_yaw_deg"],
        "actions": [],
        "attempt": attempt,
        "session_id": session,
        "slot": slot,
    }


def load_predictor(frozen):
    import torch

    torch.set_num_threads(frozen["protocol"]["runtime"]["torch_num_threads"])
    model = torch.jit.load(str(PROJECT / frozen["model"]["torchscript"]), map_location="cpu").eval()
    return model


def predict_rgb(model, observations):
    import torch
    from insect_utils.rgb_preprocessing import rgb_tensor

    images = torch.stack([rgb_tensor(rgb, [1024, 1024]) for rgb, _ in observations])
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(images)
    if list(output.shape) != [len(observations), 2]:
        raise RuntimeError("Physical deployment graph returned an invalid batch shape")
    return output.cpu().tolist(), time.perf_counter() - started


def run_episodes(folder, phase, frozen, cameras, model, settings, stop):
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
                reason = termination(state["position"], home, 0, controller)
                if reason:
                    result = complete_episode(folder, phase, state, reason, frozen)
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
                result = complete_episode(folder, phase, state, str(error), frozen)
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
                result = complete_episode(folder, phase, state, "out_of_bounds", frozen, str(error))
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
            reason = termination(after, home, len(state["actions"]), controller)
            if reason:
                result = complete_episode(folder, phase, state, reason, frozen)
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
            "successes": sum(r["reason"] == "success" for r in completed.values()),
            "active_episode_steps": {
                s["episode"]["episode_id"]: len(s["actions"]) for s in states if s is not None
            },
            "session_observations": observations_seen,
            "session_elapsed_s": time.perf_counter() - started,
            "success_radius_m": controller["success_radius_m"],
        }
        atomic_json(Path(folder) / "progress.json", progress)
    return commit_phase(folder, phase, frozen)


def model_check(folder, frozen, cameras, settings):
    """Check live RGB/disk preprocessing, scripted units and batch-size consistency."""
    import torch
    from insect_utils.attention_model import create_attention
    from insect_utils.coverage_model import PhysicalHomeVector
    from insect_utils.downward_camera import capture_batch
    from insect_utils.rgb_preprocessing import load_rgb_tensor, rgb_tensor

    model = load_predictor(frozen)
    reference = create_attention({"network_variant": "nadir384x384"}).cpu().eval()
    reference.load_state_dict(
        torch.load(PROJECT / frozen["model"]["weights"], map_location="cpu", weights_only=True)
    )
    reference = PhysicalHomeVector(reference, frozen["model"]["scale_m"]).eval()
    home = frozen["protocol"]["home_camera_xyz_m"]
    poses = [
        ([home[0] + x, home[1] + y, home[2]], yaw)
        for x, y in ((0, 0), (5, 0), (0, 5), (-5, 0))
        for yaw in (0, 90)
    ]
    destination = Path(folder) / "model_check" / "sessions" / uuid.uuid4().hex
    destination.mkdir(parents=True, exist_ok=False)
    checks = []
    for start in range(0, len(poses), len(cameras)):
        batch = poses[start : start + len(cameras)]
        requests = [
            (cameras[i], position, yaw, "modelcheck_{:02d}".format(start + i))
            for i, (position, yaw) in enumerate(batch)
        ]
        observations = capture_batch(requests)
        tensors = []
        for index, ((rgb, receipt), request) in enumerate(zip(observations, requests)):
            observation_receipt(
                cameras[index],
                request[1],
                request[2],
                request[3],
                receipt,
                frozen["protocol"],
                settings,
            )
            image = destination / (request[3] + ".png")
            sha = save_observation(image, rgb, receipt)
            tensor = rgb_tensor(rgb, [1024, 1024])
            if not torch.equal(tensor, load_rgb_tensor(image, [1024, 1024], sha)):
                raise ValueError("Online RGB preprocessing differs from training/disk input")
            tensors.append(tensor)
        stacked = torch.stack(tensors)
        with torch.inference_mode():
            actual = model(stacked)
            expected = reference(stacked)
            singles = torch.cat([model(t.unsqueeze(0)) for t in tensors])
        if not torch.isfinite(actual).all() or not torch.allclose(
            actual, expected, atol=1e-5, rtol=1e-4
        ):
            raise ValueError("Live deployment outputs are not the selected meter-scale model")
        if not torch.allclose(actual, singles, atol=1e-5, rtol=1e-4):
            raise ValueError("Navigation prediction depends on batched episode assignment")
        checks.append(
            {
                "samples": len(batch),
                "script_reference_max_abs_difference_m": float((actual - expected).abs().max()),
                "batch_single_max_abs_difference_m": float((actual - singles).abs().max()),
            }
        )
    artifacts = {
        str(p.relative_to(Path(folder) / "model_check")): digest(p)
        for p in destination.iterdir()
        if p.is_file()
    }
    atomic_json(
        Path(folder) / "model_check/completion.json",
        {
            "status": "completed",
            "fingerprint": frozen["fingerprint"],
            "images": len(poses),
            "online_disk_equal": True,
            "checks": checks,
            "output_unit": "meters",
            "extra_scale_applied": False,
            "artifacts": artifacts,
        },
    )


def run_rendered(folder, phase):
    frozen = verify_navigation(folder)
    from omni.isaac.kit import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "anti_aliasing": 0})
    cameras = []
    stop = []
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
        if phase == "model_check":
            model_check(folder, frozen, cameras, settings)
        else:
            model = None if phase == "rendered_oracle" else load_predictor(frozen)
            run_episodes(folder, phase, frozen, cameras, model, settings, stop)
        verify_navigation(folder)
        print("NAVIGATION PHASE COMPLETED " + phase, flush=True)
    except BaseException as error:
        atomic_json(
            Path(folder) / phase / ("worker_error_" + uuid.uuid4().hex[:10] + ".json"),
            {"error": str(error), "updated": now(), "phase": phase},
        )
        raise
    finally:
        for camera in reversed(cameras):
            camera.close()
        app.close()
