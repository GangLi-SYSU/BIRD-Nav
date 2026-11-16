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
    parser.add_argument(
        "--demo-config", type=Path, default=PROJECT / "config/experiments/demo_release.json"
    )
    parser.add_argument(
        "--show-trajectory",
        type=Path,
        help="Open a previously exported trajectory in the result window",
    )
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
    plot_worker = None
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
        from insect_utils.demo_animation import SmoothAction, LiveSettler
        from insect_utils.demo_overlays import annotate_overview, create_landing_markers
        from insect_utils.demo_visual_trace import VisualTrace
        from insect_utils.demo_homing_results import export_homing_results
        from concurrent.futures import ThreadPoolExecutor
        from insect_utils.downward_capture import configure_capture_rendering
        from insect_utils.navigation_runtime import predict_rgb
        from insect_utils.navigation_core import decide_motion

        frozen = json.loads(args.demo_config.read_text())
        protocol = frozen["protocol"]
        model_path = PROJECT / frozen["model"]["torchscript"]
        model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        expected_sha = frozen["model_sha256"][model_path.name]
        if model_sha != expected_sha:
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
        # The neural camera sees an unadorned copy, never a hidden display bird.
        sensor_offset = np.array([20000.0, 0.0, 0.0])
        clean_scene = UsdGeom.Xform.Define(stage, "/SensorEnvironment")
        clean_scene.GetPrim().GetReferences().AddReference(str(PROJECT / protocol["scene"]))
        clean_scene.AddTranslateOp().Set(Gf.Vec3d(*sensor_offset))
        for name in ["Sky", "Sun"]:
            light = stage.GetPrimAtPath("/SensorEnvironment/" + name)
            if light:
                light.SetActive(False)
        clean_ground = UsdGeom.Mesh(stage.GetPrimAtPath("/SensorEnvironment/Ground"))
        clean_ground.GetPointsAttr().Set(
            [(-3500, -3500, 0), (3500, -3500, 0), (3500, 3500, 0), (-3500, 3500, 0)]
        )
        clean_ground.GetPrim().GetAttribute("primvars:st").Set(
            [(0, 0), (1750, 0), (1750, 1750), (0, 1750)]
        )
        camera = UsdGeom.Camera.Define(stage, "/World/DemoDisplayCamera")
        camera.GetFocalLengthAttr().Set(26)
        camera.GetHorizontalApertureAttr().Set(36)
        camera.GetVerticalApertureAttr().Set(22.5)
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 20000))
        camera_pose = camera.AddTransformOp()
        overview = UsdGeom.Camera.Define(stage, "/World/DemoOverviewCamera")
        overview.GetFocalLengthAttr().Set(26)
        overview.GetHorizontalApertureAttr().Set(36)
        overview.GetVerticalApertureAttr().Set(27)
        overview.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 15000))
        overview_pose = overview.AddTransformOp()
        overview_product = rep.create.render_product(str(overview.GetPath()), (480, 360))
        overview_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        overview_rgb.attach([overview_product])
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
        providers = {}
        animation = [None]
        capture = [None]
        display_pose = [current[0].position.copy(), current[0].yaw]
        transition_left = [0.0]
        last_phase = [current[0].phase]
        view_state = dict(center=np.zeros(2), width=120.0, last_sensor_frame=-1)
        counts = dict(
            animation_frames=0, sensor_frames=0, overview_frames=0, handoffs=0, model_actions=0
        )
        primary_viewport = None
        marker_update = [None]
        last_overview = [None]
        visual_trace = [VisualTrace()]
        plot_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="visual_trajectory")
        plot_jobs = []
        last_plot = [None]

        def submit_plot(landing):
            f = current[0]
            if f.phase != "LANDED":
                raise RuntimeError("Final results require completed homing and landing")
            snapshot = visual_trace[0].snapshot(
                controls["episode"], f.homing_outcome, f.reason, landing
            )
            plot_jobs.append(
                (
                    controls["episode"],
                    plot_worker.submit(
                        export_homing_results,
                        args.output,
                        controls["episode"],
                        f.odometry.snapshot(),
                        snapshot,
                    ),
                )
            )

        def poll_plots():
            for episode, job in plot_jobs[:]:
                if not job.done():
                    continue
                plot_jobs.remove((episode, job))
                try:
                    path = job.result()
                    record("homing_results_saved", trajectory_episode=episode, path=str(path))
                    if episode != controls["episode"]:
                        continue
                    last_plot[0] = path
                    if "results" in providers:
                        from PIL import Image

                        image = np.asarray(Image.open(path).convert("RGBA")).copy()
                        providers["results"].set_data_array(image, [image.shape[1], image.shape[0]])
                except Exception as error:
                    record("results_export_error", error=str(error))
            if (
                last_plot[0] is not None
                and current[0].phase == "LANDED"
                and "results_window" in labels
            ):
                if labels.get("results_opened_episode") != controls["episode"]:
                    labels["results_window"].visible = True
                    labels["results_opened_episode"] = controls["episode"]

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
                    was_manual = f.phase == "MANUAL"
                    f.start_homing()
                    if was_manual:
                        record("outbound_complete", odometry=f.odometry.snapshot())
                elif name == "pause":
                    f.paused = not f.paused
                    controls["keys"].clear()
                elif name == "view":
                    # Main view remains third person; V restores the subwindows.
                    for key in ["overview_window", "sensor_window"]:
                        if key in labels:
                            labels[key].visible = True
                    if last_plot[0] is not None and "results_window" in labels:
                        labels["results_window"].visible = True
                elif name == "reset":
                    controls["episode"] += 1
                    current[0] = DemoFlight(controls["sigma"], 20261106 + controls["episode"])
                    controls["keys"].clear()
                    trail[:] = [current[0].position[:2].copy()]
                    animation[0] = None
                    capture[0] = None
                    transition_left[0] = 0.0
                    display_pose[:] = [current[0].position.copy(), current[0].yaw]
                    last_phase[0] = current[0].phase
                    visual_trace[0] = VisualTrace()
                    last_plot[0] = None
                    if "results_window" in labels:
                        labels["results_window"].visible = False
                elif name == "quit":
                    controls["quit"] = True
                record("command", command=name)
            pending.clear()

        def update_display():
            f = current[0]
            pos, yaw = display_pose
            bird_position.Set(Gf.Vec3d(*pos))
            bird_yaw.Set(float(yaw))
            if hasattr(wings, "update"):
                wings.update(time.monotonic(), f.phase, f.paused)
            else:
                for op, sign in wings:
                    angle = (
                        8
                        if f.phase in ("GROUND", "LANDED") or f.paused
                        else 8 + 18 * math.sin(time.monotonic() * 7.5)
                    )
                    op.Set(sign * angle)
            a = math.radians(yaw)
            forward = np.array([math.cos(a), math.sin(a), 0])
            side = np.array([-math.sin(a), math.cos(a), 0])
            eye = pos - 7 * forward + 3.5 * side + np.array([0, 0, 4])
            target = pos + 0.5 * forward - np.array([0, 0, 0.5])
            up = Gf.Vec3d(0, 0, 1)
            camera_pose.Set(
                Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), up).GetInverse()
            )
            if np.linalg.norm(pos[:2] - trail[-1]) > 0.1:
                trail.append(pos[:2].copy())
            points = np.asarray(trail)
            view_state["center"] = (points.min(0) + points.max(0)) / 2
            view_state["width"] = max(120, float(np.ptp(points, axis=0).max()) * 1.7)
            center, width = view_state["center"], view_state["width"]
            overview_pose.Set(
                Gf.Matrix4d()
                .SetLookAt(
                    Gf.Vec3d(*center, 40 + width * 26 / 36),
                    Gf.Vec3d(*center, 40),
                    Gf.Vec3d(0, 1, 0),
                )
                .GetInverse()
            )
            sensor.set_body_pose((pos + sensor_offset).tolist(), yaw)
            if marker_update[0] is not None:
                marker_update[0](f.landing_measurement())
            if primary_viewport is not None:
                primary_viewport.camera_path = str(camera.GetPath())
                s = f.state()
                labels["state"].text = (
                    f"Stage: {f.phase} {'[PAUSED]' if f.paused else ''}\n"
                    f"Height: {pos[2]:.1f} m | Heading: {yaw:.1f} deg\n"
                    f"TRUE distance: {np.linalg.norm(pos[:2]):.2f} m (display/scoring)\n"
                    f"PI estimated distance: {s['estimated_distance_m']:.2f} m\n"
                    f"Outbound length: {f.outbound_length:.1f} m\n"
                    f"Active sigma: {f.sigma:.2f} m/sqrt(m)\n"
                    f"Visual steps: {f.visual_steps}/50\n"
                    f"PI handoff distance: {f.handoff_distance}\n{f.reason}"
                )
                if transition_left[0] > 0:
                    text = (
                        "STAGE 1 COMPLETE  ->  STAGE 2: VISUAL HOMING\n"
                        f"Actual handoff: {f.handoff_distance:.2f} m | Preparing camera"
                    )
                    color = 0xFF00DDFF
                elif f.phase == "VISUAL_RETURN":
                    action = "MOVING" if animation[0] else "HOVER: acquiring stable image"
                    text = (
                        f"STAGE 2: VISUAL HOMING | {action}\n"
                        f"Completed neural steps: {f.visual_steps}/50"
                    )
                    color = 0xFF80EF80
                elif f.phase == "PI_RETURN":
                    text = (
                        "STAGE 1: PATH INTEGRATION HOMING\n"
                        "Following the internally estimated home vector"
                    )
                    color = 0xFFFFCC66
                elif f.phase == "LANDING":
                    text = (
                        f"HOMING {f.homing_outcome}  ->  LANDING\n"
                        f"Descending at current position | Height {pos[2]:.1f} m"
                    )
                    color = 0xFF00DDFF
                elif f.phase == "LANDED":
                    distance = f.landing_measurement()["horizontal_distance_m"]
                    text = (
                        f"LANDED | Homing result: {f.homing_outcome}\n"
                        f"START to LAND: {distance:.2f} m (horizontal)"
                    )
                    color = 0xFF80EF80 if f.homing_outcome == "SUCCESS" else 0xFF80A0FF
                else:
                    text = f"{f.phase}\n{f.reason}"
                    color = 0xFFFFFFFF
                labels["banner"].text = text
                labels["banner"].style = {"color": color, "font_size": 21}

        def update_images():
            frame = sensor._frame()
            if frame > view_state["last_sensor_frame"]:
                raw = np.asarray(sensor.rgb.get_data()).copy()
                if raw.shape == (1024, 1024, 4):
                    view_state["last_sensor_frame"] = frame
                    counts["sensor_frames"] += 1
                    if "sensor" in providers:
                        providers["sensor"].set_data_array(
                            np.ascontiguousarray(raw[::2, ::2]), [512, 512]
                        )
                        labels[
                            "sensor_info"
                        ].text = f"LIVE RGB | frame {frame} | 30 deg | source 1024 x 1024"
            raw = np.asarray(overview_rgb.get_data()).copy()
            if raw.shape == (360, 480, 4):
                counts["overview_frames"] += 1
                raw = annotate_overview(
                    raw,
                    trail,
                    display_pose[0],
                    display_pose[1],
                    view_state["center"],
                    view_state["width"],
                    current[0].landing_measurement(),
                )
                last_overview[0] = raw
                if "overview" in providers:
                    providers["overview"].set_data_array(raw, [480, 360])

        def visual_tick(dt):
            f = current[0]
            if f.paused:
                return
            if transition_left[0] > 0:
                transition_left[0] = max(0, transition_left[0] - dt)
                return
            if animation[0] is not None:
                pos, yaw, done = animation[0].advance(min(dt, animation[0].duration / 36))
                display_pose[:] = [pos, yaw]
                counts["animation_frames"] += 1
                if done:
                    motion = animation[0].motion
                    previous_steps = f.visual_steps
                    f.apply_visual_motion(motion)
                    if f.visual_steps > previous_steps:
                        visual_trace[0].completed_action(f.visual_steps, f.position, f.yaw)
                    animation[0] = None
                    capture[0] = None
                    display_pose[:] = [f.position.copy(), f.yaw]
                    record("neural_action_completed", motion=motion)
                return
            if f.visual_status() != "VISUAL_RETURN":
                return
            if max(abs(f.position[:2])) > 190:
                f.phase, f.reason = "FAILED", "PI ended outside the supported visual scene"
                record("unsupported_visual_start")
                return
            if capture[0] is None:
                capture[0] = LiveSettler(sensor, (f.position + sensor_offset).tolist(), f.yaw)
                return
            observation = capture[0].poll(sensor)
            if observation is None:
                return
            rgb, receipt = observation
            predictions, _ = predict_rgb(model, [observation])
            motion = decide_motion(predictions[0], f.yaw, protocol["controller"])
            from PIL import Image

            image_name = f"episode_{controls['episode']:03d}_visual_{f.visual_steps:03d}.png"
            Image.fromarray(rgb).save(args.output / image_name)
            animation[0] = SmoothAction(f.position, f.yaw, motion)
            capture[0] = None
            counts["model_actions"] += 1
            record(
                "neural_action_planned",
                motion=motion,
                image=image_name,
                frame=receipt["rendering_frame"],
                stability=receipt["stability"],
                animation_duration_s=animation[0].duration,
            )

        if not args.headless_smoke:
            import omni.ui as ui
            import omni.appwindow
            from omni.kit.viewport.utility import get_active_viewport

            primary_viewport = get_active_viewport()
            providers["overview"] = ui.ByteImageProvider()
            providers["sensor"] = ui.ByteImageProvider()
            providers["results"] = ui.ByteImageProvider()
            labels["results_window"] = ui.Window(
                "Two-stage homing results",
                width=1000,
                height=650,
                position_x=250,
                position_y=100,
                visible=False,
            )
            with labels["results_window"].frame:
                with ui.VStack():
                    ui.Label(
                        "Stage 1: outbound odometry + full PI return | "
                        "Stage 2: visual trajectory",
                        height=24,
                    )
                    ui.ImageWithProvider(providers["results"])
            if args.show_trajectory is not None:
                from PIL import Image

                image = np.asarray(Image.open(args.show_trajectory).convert("RGBA")).copy()
                providers["results"].set_data_array(image, [image.shape[1], image.shape[0]])
                labels["results_window"].visible = True
                last_plot[0] = args.show_trajectory
            labels["overview_window"] = ui.Window(
                "Overview - live top view", width=360, height=300, position_x=910, position_y=80
            )
            with labels["overview_window"].frame:
                with ui.VStack():
                    ui.Label("Green: START / 15m | Yellow arrow: NOSE | Pink: LAND", height=24)
                    ui.ImageWithProvider(providers["overview"])
            labels["sensor_window"] = ui.Window(
                "Downward camera - actual sensor RGB",
                width=360,
                height=405,
                position_x=910,
                position_y=390,
            )
            with labels["sensor_window"].frame:
                with ui.VStack():
                    labels["sensor_info"] = ui.Label("Waiting for live RGB...", height=28)
                    ui.ImageWithProvider(providers["sensor"])
            banner_window = ui.Window(
                "Homing phase", width=570, height=95, position_x=340, position_y=80
            )
            with banner_window.frame:
                labels["banner"] = ui.Label("Ready", word_wrap=True, style={"font_size": 21})

            keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            input_interface = carb.input.acquire_input_interface()

            def key_event(event, *_):
                if event.type not in (
                    carb.input.KeyboardEventType.KEY_PRESS,
                    carb.input.KeyboardEventType.KEY_RELEASE,
                ):
                    return True
                key = getattr(event.input, "name", str(event.input)).rsplit(".", 1)[-1]
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
            window = ui.Window(
                "Flight controls", width=330, height=590, position_x=8, position_y=185
            )
            with window.frame:
                with ui.VStack(spacing=5):
                    ui.Label("T: takeoff | W/S: forward/back | A/D: strafe", height=22)
                    ui.Label("Q/E: turn | H: auto home | Space: pause", height=22)
                    ui.Label("R: new flight | V: show panels | Click viewport to fly", height=22)
                    with ui.HStack(height=30):
                        ui.Button("Take off", clicked_fn=lambda: command("takeoff"))
                        ui.Button("Auto home", clicked_fn=lambda: command("home"))
                        ui.Button("Pause / resume", clicked_fn=lambda: command("pause"))
                    with ui.HStack(height=30):
                        ui.Button("New flight", clicked_fn=lambda: command("reset"))
                        ui.Button("Show panels", clicked_fn=lambda: command("view"))
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
                    display_geometry_hidden_for_inference=False,
                    independent_sensor_scene_offset_m=sensor_offset.tolist(),
                    visual_action_animation_min_s=1.2,
                    handoff_banner_duration_s=3.0,
                    automatic_vertical_landing=True,
                    landing_measurement="horizontal START-to-LAND distance in meters",
                    visual_trajectory_export="per-flight PNG, CSV, JSON; action endpoints only",
                    pi_comparison="Outbound odometry and full PI return only",
                    result_layout=["outbound_odometry", "pi_full_return", "visual_homing"],
                    result_popup="One window after both homing stages and landing",
                    pi_gyro_comparison_noise_deg_per_sqrt_s=0.63,
                ),
                indent=2,
            )
        )
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
            display_pose[:] = [f.position.copy(), f.yaw]
            trail[:] = [
                np.zeros(2),
                np.array([500.0, 0]),
                np.array([500.0, 500.0]),
                f.position[:2].copy(),
            ]
            update_display()
            # Engineering-only parity check, before starting the live demo loop.
            reference = DownwardCamera(
                protocol["camera"], world, "replicator", name="ParityReference"
            )
            for prim in [bird, ring]:
                UsdGeom.Imageable(prim).MakeInvisible()
            try:
                pair = capture_batch(
                    [
                        (reference, f.position.tolist(), f.yaw, "reference"),
                        (sensor, (f.position + sensor_offset).tolist(), f.yaw, "clean_copy"),
                    ]
                )
                mae = float(np.abs(pair[0][0].astype(float) - pair[1][0].astype(float)).mean())
                preds, _ = predict_rgb(model, pair)
                delta = float(np.linalg.norm(np.asarray(preds[0]) - preds[1]))
                record("sensor_copy_parity", rgb_mae=mae, prediction_difference_m=delta)
                assert mae < 5 and delta < 0.5, (mae, delta)
            finally:
                reference.close()
                for prim in [bird, ring]:
                    UsdGeom.Imageable(prim).MakeVisible()
            transition_left[0] = 3.0
            counts["handoffs"] += 1
            last_phase[0] = "VISUAL_RETURN"
            visual_trace[0].begin(f.position, f.yaw)
        marker_update[0] = create_landing_markers(stage)
        update_display()
        rep.orchestrator.run(start_timeline=True)
        record("ready")
        print("INTERACTIVE_MULTIVIEW_READY", flush=True)
        previous_time = time.monotonic()
        start_time = previous_time
        tick = 0
        while app.is_running() and not controls["quit"]:
            app.update()
            now = time.monotonic()
            dt = min(0.1, max(0.001, now - previous_time))
            previous_time = now
            process_commands()
            f = current[0]
            before = f.position.copy()
            if f.phase == "TAKEOFF":
                f.altitude_tick(dt)
            elif f.phase == "MANUAL":
                keys = controls["keys"]
                f.manual_tick(
                    int("W" in keys) - int("S" in keys),
                    int("A" in keys) - int("D" in keys),
                    int("Q" in keys) - int("E" in keys),
                    dt * controls["speed"],
                )
            elif f.phase == "PI_RETURN":
                f.pi_tick(dt * controls["speed"])
            elif f.phase == "LANDING":
                f.landing_tick(dt)
                if f.phase == "LANDED":
                    measurement = f.landing_measurement()
                    (args.output / f"episode_{controls['episode']:03d}_landing.json").write_text(
                        json.dumps(measurement, indent=2)
                    )
                    record("landing_complete", measurement=measurement)
                    submit_plot(measurement)
            if f.phase == "VISUAL_RETURN" and last_phase[0] != "VISUAL_RETURN":
                transition_left[0] = 3.0
                counts["handoffs"] += 1
                capture[0] = None
                visual_trace[0].begin(f.position, f.yaw)
                record(
                    "stage_transition",
                    from_stage="PI_RETURN",
                    to_stage="VISUAL_RETURN",
                    actual_inside_15m=f.handoff_distance <= 15,
                )
            last_phase[0] = f.phase
            if animation[0] is None:
                display_pose[:] = [f.position.copy(), f.yaw]
            if f.phase == "VISUAL_RETURN":
                try:
                    visual_tick(dt)
                except Exception as error:
                    f.phase, f.reason = "ERROR", str(error)
                    record("visual_error", traceback=traceback.format_exc())
                    if args.headless_smoke:
                        raise
            if f.phase in ("SUCCESS", "FAILED"):
                f.start_landing()
                capture[0] = None
                animation[0] = None
                record("landing_started", fixed_horizontal_position_m=f.position[:2].tolist())
            if not np.array_equal(before, f.position):
                record("motion")
            elif tick % 30 == 0:
                record("heartbeat", display_position_m=display_pose[0].tolist(), counters=counts)
            update_display()
            update_images()
            poll_plots()
            assert UsdGeom.Imageable(bird).ComputeVisibility() != UsdGeom.Tokens.invisible
            if args.headless_smoke:
                if f.phase != "LANDED":
                    assert not plot_jobs and last_plot[0] is None
                if f.phase == "LANDED":
                    assert f.visual_steps > 0 and counts["animation_frames"] > f.visual_steps * 10
                    assert counts["sensor_frames"] > 20 and counts["overview_frames"] > 20
                    np.testing.assert_array_equal(f.position[:2], f.landing_start[:2])
                    assert f.position[2] == 0.3
                    for _, job in plot_jobs:
                        job.result(timeout=60)
                    poll_plots()
                    combined = json.loads(
                        (
                            args.output / f"episode_{controls['episode']:03d}_homing_results.json"
                        ).read_text()
                    )
                    assert combined["panel_ids"] == [
                        "outbound_odometry",
                        "pi_full_return",
                        "visual_homing",
                    ]
                    np.testing.assert_allclose(
                        combined["pi_recorded"]["actual_pi_return"][-1]["true_xy_m"],
                        [visual_trace[0].points[0]["x_m"], visual_trace[0].points[0]["y_m"]],
                    )
                    trajectory_json = (
                        args.output / f"episode_{controls['episode']:03d}_visual_trajectory.json"
                    )
                    trace_result = json.loads(trajectory_json.read_text())
                    assert trace_result["completed_actions"] == f.visual_steps
                    assert len(trace_result["positions"]) == f.visual_steps + 1
                    assert (
                        abs(
                            trace_result["final_distance_m"]
                            - f.landing_measurement()["horizontal_distance_m"]
                        )
                        < 1e-9
                    )
                    from PIL import Image

                    Image.fromarray(last_overview[0]).save(args.output / "overview_landed.png")
                    record("smoke_complete", counters=counts)
                    (args.output / "completion.json").write_text(
                        json.dumps(
                            dict(
                                status="passed",
                                result=f.state(),
                                counters=counts,
                                sensor_copy_rgb_mae=mae,
                                sensor_copy_prediction_difference_m=delta,
                                bird_visible_every_loop=True,
                                live_sensor=True,
                                animated_motion=True,
                                visual_trajectory_verified=True,
                                homing_results_verified=True,
                                result_panels=combined["panel_ids"],
                                homing_results_png=str(last_plot[0]),
                                no_results_before_landing=True,
                            ),
                            indent=2,
                        )
                    )
                    print("MULTIVIEW_SMOKE_COMPLETE " + json.dumps(counts), flush=True)
                    break
                if now - start_time > 240:
                    raise RuntimeError("Live rendering smoke timeout: " + json.dumps(counts))
            tick += 1
        if not args.headless_smoke:
            input_interface.unsubscribe_to_keyboard_events(keyboard, subscription)
        rep.orchestrator.stop()
        overview_rgb.detach([overview_product.path])
        overview_product.destroy()
        record("closed")
    except Exception:
        (args.output / "error.log").write_text(traceback.format_exc())
        raise
    finally:
        log.close()
        if sensor is not None:
            sensor.close()
        if plot_worker is not None:
            plot_worker.shutdown(wait=True)
        app.close()


if __name__ == "__main__":
    main()
