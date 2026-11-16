"""Manual outbound and real two-stage automatic homing in Isaac Sim."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless-smoke", action="store_true")
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from omni.isaac.kit import SimulationApp

    app = SimulationApp(
        dict(
            headless=args.headless_smoke,
            width=1280,
            height=800,
            renderer="RayTracedLighting",
            anti_aliasing=0,
            multi_gpu=False,
        )
    )
    sensor = None
    log = (args.output / "events.jsonl").open("w")
    try:
        import carb
        import numpy as np
        import torch
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom
        from omni.isaac.core import World
        from insect_utils.interactive_homing import DemoFlight
        from insect_utils.interactive_scene import create_demo_scene
        from insect_utils.downward_camera import DownwardCamera, capture_batch
        from insect_utils.downward_capture import configure_capture_rendering
        from insect_utils.navigation_runtime import predict_rgb
        from insect_utils.navigation_core import decide_motion

        frozen = json.loads(
            (
                PROJECT / "runs/stage23_compact_spiral_h40_fov30_01/navigation/frozen.json"
            ).read_text()
        )
        protocol = frozen["protocol"]
        model_path = PROJECT / frozen["model"]["torchscript"]
        model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if model_sha != "31ded9f200a9538d1358cd7e331c001dbd746d97263a532793f0e633f5f48119":
            raise RuntimeError("Unexpected visual model; refusing silent model replacement")
        torch.set_num_threads(2)
        model = torch.jit.load(str(model_path), map_location="cpu").eval()
        omni.usd.get_context().open_stage(str(PROJECT / protocol["scene"]))
        for _ in range(8):
            app.update()
        world = World(stage_units_in_meters=1.0)
        world.reset()
        stage = omni.usd.get_context().get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        bird, bird_position, bird_yaw, wings, ring = create_demo_scene(stage)
        camera = UsdGeom.Camera.Define(stage, "/World/DemoDisplayCamera")
        camera.GetFocalLengthAttr().Set(26)
        camera.GetHorizontalApertureAttr().Set(36)
        camera.GetVerticalApertureAttr().Set(22.5)
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 20000))
        camera_pose = camera.AddTransformOp()
        settings = carb.settings.get_settings()
        settings.set("/app/runLoops/main/rateLimitEnabled", True)
        settings.set("/app/runLoops/main/rateLimitFrequency", 30)
        configure_capture_rendering(settings, rep.settings)
        # Separate sensor: never make it the interactive viewport camera.
        sensor = DownwardCamera(protocol["camera"], world, "replicator", name="DemoNeuralSensor")
        configure_capture_rendering(settings, rep.settings)
        current = [DemoFlight(0.5 if args.headless_smoke else args.sigma)]
        controls = dict(
            keys=set(), view="CHASE", speed=1.0, sigma=args.sigma, episode=0, quit=False
        )
        pending = []
        trail = [current[0].position[:2].copy()]
        labels = {}

        def record(event, **extra):
            row = dict(
                event=event,
                time=time.time(),
                episode=controls["episode"],
                state=current[0].state(),
                **extra,
            )
            log.write(json.dumps(row) + "\n")
            log.flush()
            (args.output / "status.json").write_text(json.dumps(row, indent=2))

        def command(name):
            pending.append(name)

        def process_commands():
            for name in pending[:]:
                f = current[0]
                if name == "takeoff":
                    f.takeoff()
                elif name == "home":
                    f.start_homing()
                elif name == "pause":
                    f.paused = not f.paused
                    controls["keys"].clear()
                elif name == "view":
                    controls["view"] = "MAP" if controls["view"] == "CHASE" else "CHASE"
                elif name == "reset":
                    controls["episode"] += 1
                    current[0] = DemoFlight(controls["sigma"], 20261106 + controls["episode"])
                    controls["keys"].clear()
                    trail[:] = [current[0].position[:2].copy()]
                elif name == "quit":
                    controls["quit"] = True
                record("command", command=name)
            pending.clear()

        def update_display():
            f = current[0]
            pos = f.position
            bird_position.Set(Gf.Vec3d(*pos))
            bird_yaw.Set(float(f.yaw))
            for op, sign in wings:
                angle = (
                    8
                    if f.phase == "GROUND" or f.paused
                    else 8 + 18 * math.sin(time.monotonic() * 7.5)
                )
                op.Set(sign * angle)
            a = math.radians(f.yaw)
            forward = np.array([math.cos(a), math.sin(a), 0])
            side = np.array([-math.sin(a), math.cos(a), 0])
            if controls["view"] == "CHASE":
                eye = pos - 7 * forward + 3.5 * side + np.array([0, 0, 4])
                target = pos + 0.5 * forward - np.array([0, 0, 0.5])
                up = Gf.Vec3d(0, 0, 1)
            else:
                points = np.asarray(trail)
                center = (points.min(0) + points.max(0)) / 2
                width = max(120, float(np.ptp(points, axis=0).max()) * 2.2)
                eye = np.array([*center, 40 + width * 26 / 36])
                target = np.array([*center, 0])
                up = Gf.Vec3d(0, 1, 0)
            camera_pose.Set(
                Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), up).GetInverse()
            )
            if not args.headless_smoke:
                from omni.kit.viewport.utility import get_active_viewport

                get_active_viewport().camera_path = str(camera.GetPath())
                s = f.state()
                labels["state"].text = (
                    f"Stage: {f.phase} {'[PAUSED]' if f.paused else ''}\n"
                    f"Height: {pos[2]:.1f} m | Heading: {f.yaw:.1f} deg\n"
                    f"TRUE distance: {s['true_distance_m']:.2f} m (display/scoring)\n"
                    f"PI estimated distance: {s['estimated_distance_m']:.2f} m\n"
                    f"Outbound length: {f.outbound_length:.1f} m\n"
                    f"Active sigma: {f.sigma:.2f} m/sqrt(m)\n"
                    f"Visual steps: {f.visual_steps}/50\n"
                    f"PI handoff distance: {f.handoff_distance}\n{f.reason}"
                )

        def visual_step():
            f = current[0]
            if f.visual_status() != "VISUAL_RETURN" or f.paused:
                return
            if max(abs(f.position[:2])) > 190:
                f.phase, f.reason = "FAILED", "PI ended outside the supported visual scene"
                record("unsupported_visual_start")
                return
            # Remove display geometry AND its shadows from the neural image.
            for prim in [bird, ring]:
                UsdGeom.Imageable(prim).MakeInvisible()
            try:
                observations = capture_batch(
                    [
                        (
                            sensor,
                            f.position.tolist(),
                            f.yaw,
                            f"demo_{controls['episode']}_{f.visual_steps}",
                        )
                    ]
                )
                predictions, _ = predict_rgb(model, observations)
                motion = decide_motion(predictions[0], f.yaw, protocol["controller"])
                rgb, receipt = observations[0]
                from PIL import Image

                image_name = f"episode_{controls['episode']:03d}_visual_{f.visual_steps:03d}.png"
                Image.fromarray(rgb).save(args.output / image_name)
                # Inputs can arrive while Kit advances render frames; honor pause/reset
                # before applying a previously captured action.
                if pending:
                    record("visual_action_discarded_for_pending_input")
                    return
                f.apply_visual_motion(motion)
                record(
                    "neural_action",
                    motion=motion,
                    image=image_name,
                    frame=receipt["rendering_frame"],
                    stability=receipt["stability"],
                )
            finally:
                for prim in [bird, ring]:
                    UsdGeom.Imageable(prim).MakeVisible()

        if not args.headless_smoke:
            import omni.ui as ui
            import omni.appwindow

            keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            input_interface = carb.input.acquire_input_interface()

            def key_event(event, *_):
                key = event.input.name
                if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                    controls["keys"].add(key)
                    action = {
                        "T": "takeoff",
                        "H": "home",
                        "SPACE": "pause",
                        "R": "reset",
                        "V": "view",
                    }
                    if key in action:
                        command(action[key])
                elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                    controls["keys"].discard(key)
                return True

            subscription = input_interface.subscribe_to_keyboard_events(keyboard, key_event)
            window = ui.Window("Two-stage homing demo", width=420, height=600)
            with window.frame:
                with ui.VStack(spacing=5):
                    ui.Label("T: takeoff | W/S: forward/back | A/D: strafe", height=22)
                    ui.Label("Q/E: turn | H: auto home | Space: pause", height=22)
                    ui.Label("R: new flight | V: chase/map | Click viewport to fly", height=22)
                    with ui.HStack(height=30):
                        ui.Button("Take off", clicked_fn=lambda: command("takeoff"))
                        ui.Button("Auto home", clicked_fn=lambda: command("home"))
                        ui.Button("Pause / resume", clicked_fn=lambda: command("pause"))
                    with ui.HStack(height=30):
                        ui.Button("New flight", clicked_fn=lambda: command("reset"))
                        ui.Button("Chase / map", clicked_fn=lambda: command("view"))
                        ui.Button("Exit", clicked_fn=lambda: command("quit"))
                    ui.Label("Simulation speed (manual + PI only)", height=22)
                    speed = ui.FloatSlider(min=1, max=30, height=22)
                    speed.model.set_value(1.0)
                    speed.model.add_value_changed_fn(lambda m: controls.update(speed=m.as_float))
                    ui.Label("PI sigma for NEXT flight: change then New flight", height=22)
                    noise = ui.FloatSlider(min=0, max=2, height=22)
                    noise.model.set_value(args.sigma)
                    noise.model.add_value_changed_fn(lambda m: controls.update(sigma=m.as_float))
                    labels["state"] = ui.Label("", word_wrap=True, height=230)
                    ui.Label(
                        "Compass: perfect. PI -> vision at estimated home.\n"
                        "Visual stop: simulator distance <=2 m or 50 steps.\n"
                        "No collision / aerodynamic model.",
                        word_wrap=True,
                        height=55,
                    )
        (args.output / "protocol.json").write_text(
            json.dumps(
                dict(
                    model=frozen["model"],
                    model_sha256=model_sha,
                    visual_controller=protocol["controller"],
                    camera=protocol["camera"],
                    pi_sigma=args.sigma,
                    smoke_sigma=0.5 if args.headless_smoke else None,
                    handoff="PI estimate zero; true 15m radius is display only",
                    visual_stop="simulator truth <=2m or 50 neural steps",
                    display_geometry_hidden_for_inference=True,
                ),
                indent=2,
            )
        )
        record("ready")
        print("INTERACTIVE_DEMO_READY", flush=True)
        if args.headless_smoke:
            f = current[0]
            f.takeoff()
            f.altitude_tick(8)
            for _ in range(400):
                f.manual_tick(1, 0, 0, 0.5)
            for _ in range(400):
                f.manual_tick(0, 1, 0, 0.5)
            f.start_homing()
            for _ in range(1000):
                f.pi_tick(0.5)
            assert f.phase == "VISUAL_RETURN" and 2 < f.handoff_distance < 15
            record("smoke_handoff")
            update_display()
            while f.phase == "VISUAL_RETURN":
                visual_step()
                update_display()
            assert f.visual_steps > 0
            record("smoke_complete")
            (args.output / "completion.json").write_text(
                json.dumps(
                    dict(
                        status="passed",
                        real_neural_actions=f.visual_steps,
                        model_sha256=model_sha,
                        result=f.state(),
                    ),
                    indent=2,
                )
            )
            print("DEMO_SMOKE_COMPLETE " + json.dumps(f.state()), flush=True)
        else:
            tick = 0
            while app.is_running() and not controls["quit"]:
                app.update()
                process_commands()
                f = current[0]
                before = f.position.copy()
                dt = controls["speed"] / 30
                if f.phase == "TAKEOFF":
                    f.altitude_tick(1 / 30)
                elif f.phase == "MANUAL":
                    keys = controls["keys"]
                    f.manual_tick(
                        int("W" in keys) - int("S" in keys),
                        int("A" in keys) - int("D" in keys),
                        int("Q" in keys) - int("E" in keys),
                        dt,
                    )
                elif f.phase == "PI_RETURN":
                    f.pi_tick(dt)
                elif f.phase == "VISUAL_RETURN" and not f.paused:
                    try:
                        visual_step()
                    except Exception as error:
                        f.phase, f.reason = "ERROR", str(error)
                        record("visual_error", traceback=traceback.format_exc())
                if not np.array_equal(before, f.position):
                    trail.append(f.position[:2].copy())
                    record("motion")
                elif tick % 30 == 0:
                    record("heartbeat")
                update_display()
                tick += 1
            input_interface.unsubscribe_to_keyboard_events(keyboard, subscription)
            record("closed")
    except Exception:
        (args.output / "error.log").write_text(traceback.format_exc())
        raise
    finally:
        log.close()
        if sensor is not None:
            sensor.close()
        app.close()


if __name__ == "__main__":
    main()
