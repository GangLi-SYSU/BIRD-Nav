"""Shared Isaac 4.2 camera setup and pose-checked fresh RGB acquisition."""

from insect_utils.camera_geometry import optics, usd_quaternion_wxyz, usd_world_matrix, xyz
from insect_utils.frame_stability import FrameStability, STABILITY_POLICY


def configure_prim(prim, settings):
    from pxr import Gf, Sdf, UsdGeom

    camera = UsdGeom.Camera(prim)
    camera.GetProjectionAttr().Set(UsdGeom.Tokens.perspective)
    camera.GetFocalLengthAttr().Set(settings["usd_focal_length"])
    camera.GetHorizontalApertureAttr().Set(settings["usd_aperture_wh"][0])
    camera.GetVerticalApertureAttr().Set(settings["usd_aperture_wh"][1])
    camera.GetHorizontalApertureOffsetAttr().Set(0)
    camera.GetVerticalApertureOffsetAttr().Set(0)
    camera.GetFStopAttr().Set(0)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(*settings["clipping_range_m"]))
    prim.CreateAttribute("cameraProjectionType", Sdf.ValueTypeNames.Token).Set("pinhole")


def set_usd_pose(prim, position, yaw_deg):
    from pxr import Gf, UsdGeom

    matrix = usd_world_matrix(position, yaw_deg)
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(*[v for row in matrix for v in row]))


def capture_batch(requests, max_updates=90):
    """Advance all render products together, retaining independent acceptance state.

    Each request is (camera, position, yaw, pose_id). RGB, frame and pose checks
    match single-camera capture; one slow or stale product cannot pass for another.
    """
    import time
    import numpy as np
    import omni.replicator.core as rep

    if (
        isinstance(max_updates, bool)
        or not isinstance(max_updates, int)
        or not 1 <= max_updates <= STABILITY_POLICY["max_render_updates"]
    ):
        raise ValueError("Capture budget must be an integer from 1 to 90 render updates")
    if not requests or len({id(item[0]) for item in requests}) != len(requests):
        raise ValueError("Batch requires distinct cameras and at least one request")
    started = time.perf_counter()
    states = []
    for camera, position, yaw, pose_id in requests:
        if camera.backend != "replicator":
            raise ValueError("Batch capture supports Replicator products only")
        position = xyz(position)
        previous = max(camera.last_frame, camera._frame())
        updated = camera.set_body_pose(position, yaw)
        camera.last_receipt = None
        camera.last_failure = None
        states.append(
            dict(
                camera=camera,
                position=position,
                yaw=yaw,
                pose_id=str(pose_id),
                previous=previous,
                updated=updated,
                expected=np.asarray(usd_world_matrix(position, yaw), dtype=float),
                pose_frame=None,
                previous_rgb=None,
                settling=FrameStability(),
                result=None,
            )
        )
    for update in range(max_updates):
        rep.orchestrator.step(
            rt_subframes=STABILITY_POLICY["render_subframes"], pause_timeline=False
        )
        for state in states:
            if state["result"] is not None:
                continue
            camera, settling = state["camera"], state["settling"]
            frame = camera._frame()
            if frame <= state["previous"]:
                continue
            reference, params = camera.reference.get_data(), camera.params.get_data()
            if not reference or not params or int(reference["referenceTimeNumerator"]) != frame:
                state.update(previous_rgb=None, pose_frame=None)
                settling.reset()
                continue
            matrix = np.linalg.inv(np.asarray(params["cameraViewTransform"]).reshape(4, 4))
            if not np.allclose(matrix, state["expected"], atol=1e-4, rtol=0):
                state.update(previous_rgb=None, pose_frame=None)
                settling.reset()
                continue
            if state["pose_frame"] is None:
                state["pose_frame"] = frame
                continue
            if frame <= state["pose_frame"] or frame <= settling.last_frame:
                continue
            raw = np.asarray(camera.rgb.get_data()).copy()
            width, height = camera.settings["resolution_wh"]
            if (
                raw.ndim != 3
                or raw.shape[:2] != (height, width)
                or raw.shape[2] < 3
                or camera._frame() != frame
            ):
                state["previous_rgb"] = None
                settling.reset()
                continue
            if raw.dtype != np.uint8:
                raise ValueError("Expected uint8 RGB from the renderer")
            rgb = np.ascontiguousarray(raw[:, :, :3])
            if not np.any(rgb):
                state["previous_rgb"] = None
                settling.reset()
                continue
            actual = {
                name: float(camera.prim.GetAttribute(name).Get())
                for name in ("focalLength", "horizontalAperture", "verticalAperture", "fStop")
            }
            expected_optics = dict(
                focalLength=camera.settings["usd_focal_length"],
                horizontalAperture=camera.settings["usd_aperture_wh"][0],
                verticalAperture=camera.settings["usd_aperture_wh"][1],
                fStop=0,
            )
            if any(abs(actual[k] - v) > 1e-6 for k, v in expected_optics.items()):
                raise ValueError("Renderer overwrote camera optics: " + str(actual))
            previous_rgb = state["previous_rgb"]
            mae = (
                float(np.abs(rgb.astype(np.float32) - previous_rgb).mean())
                if previous_rgb is not None
                else None
            )
            state["previous_rgb"] = rgb.astype(np.float32)
            if not settling.observe(frame, mae):
                continue
            receipt = dict(
                pose_id=state["pose_id"],
                backend=camera.backend,
                rendering_frame=frame,
                reference_time_numerator=int(reference["referenceTimeNumerator"]),
                first_pose_matched_frame=state["pose_frame"],
                camera_prim=camera.path,
                render_product=camera.product_path,
                position_m=state["position"],
                body_yaw_deg=float(state["yaw"]),
                pose_was_updated=state["updated"],
                usd_quaternion_wxyz=usd_quaternion_wxyz(state["yaw"]),
                camera_world_matrix=matrix.tolist(),
                updates_waited=update + 1,
                elapsed_s=time.perf_counter() - started,
                stability=settling.summary(),
                resolution_wh=[width, height],
                observed_usd_optics=actual,
                render_step="replicator_synchronized",
                acquisition="independent_checks_shared_batch_render",
                render_camera_projection=np.asarray(params["cameraProjection"])
                .reshape(4, 4)
                .tolist(),
            )
            camera.last_frame, camera.last_receipt = frame, receipt
            state["result"] = (rgb, dict(receipt))
        if all(state["result"] is not None for state in states):
            return [state["result"] for state in states]
    failed = []
    for state in states:
        if state["result"] is None:
            state["camera"].last_failure = dict(
                pose_id=state["pose_id"],
                updates_waited=max_updates,
                stability=state["settling"].summary(),
            )
            failed.append(state["pose_id"])
    raise RuntimeError("Batch has no settled pose-matched RGB for: " + str(failed))


class DownwardCamera:
    """One real Replicator or Camera API render product; no network/preprocessing."""

    def __init__(self, camera_spec, world, backend, name="DownwardCamera"):
        import omni.usd
        import omni.replicator.core as rep
        from pxr import UsdGeom

        if backend not in ("replicator", "sensor"):
            raise ValueError("Unknown downward camera backend")
        stage = omni.usd.get_context().get_stage()
        if UsdGeom.GetStageMetersPerUnit(stage) != 1 or UsdGeom.GetStageUpAxis(stage) != "Z":
            raise ValueError("Downward adapter requires a meter-scale Z-up stage")
        self.settings = optics(camera_spec)
        self.world, self.backend = world, backend
        self.sensor = None
        self.product = None
        self.item = None
        if backend == "replicator":
            # Author the camera directly: a Replicator creation graph can overwrite
            # poses during its first evaluation. Rendering remains Replicator RGB.
            self.path = "/World/" + name
            UsdGeom.Camera.Define(stage, self.path)
            self.product = rep.create.render_product(
                self.path, tuple(self.settings["resolution_wh"]), name=name, force_new=True
            )
            self.product_path = self.product.path
        else:
            from omni.isaac.sensor import Camera

            self.path = "/World/" + name
            self.sensor = Camera(
                prim_path=self.path, name=name, resolution=tuple(self.settings["resolution_wh"])
            )
            self.sensor.initialize()
            self.sensor.set_projection_type("pinhole")
            self.product_path = self.sensor.get_render_product_path()
        self.prim = stage.GetPrimAtPath(self.path)
        configure_prim(self.prim, self.settings)
        # Camera.initialize already owns RGB and ReferenceTime on the sensor path.
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb") if self.sensor is None else None
        self.reference = (
            rep.AnnotatorRegistry.get_annotator("ReferenceTime") if self.sensor is None else None
        )
        self.params = rep.AnnotatorRegistry.get_annotator("CameraParams")
        for annotator in (self.rgb, self.reference, self.params):
            if annotator is not None:
                annotator.attach([self.product_path])
        self.last_frame = -1
        self.last_receipt = None
        self._closed = False

    def _frame(self):
        if self.sensor is not None:
            return int(self.sensor.get_current_frame()["rendering_frame"])
        data = self.reference.get_data()
        return int(data["referenceTimeNumerator"]) if data else -1

    def set_body_pose(self, position, yaw_deg):
        """Apply a requested world pose; equivalent commands keep renderer history."""
        import numpy as np
        from pxr import UsdGeom

        position = xyz(position)
        expected = np.asarray(usd_world_matrix(position, yaw_deg), dtype=float)
        authored = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(self.prim))
        if np.allclose(authored, expected, atol=1e-6, rtol=0):
            return False
        if self.sensor is not None:
            self.sensor.set_world_pose(
                np.asarray(position), np.asarray(usd_quaternion_wxyz(yaw_deg)), camera_axes="usd"
            )
        else:
            set_usd_pose(self.prim, position, yaw_deg)
        return True

    def capture(self, position, yaw_deg, pose_id, max_updates=90):
        import time
        import numpy as np
        import omni.replicator.core as rep

        if (
            isinstance(max_updates, bool)
            or not isinstance(max_updates, int)
            or not 1 <= max_updates <= STABILITY_POLICY["max_render_updates"]
        ):
            raise ValueError("Capture budget must be an integer from 1 to 90 render updates")
        started = time.perf_counter()
        self.last_receipt = None
        self.last_failure = None
        position = xyz(position)
        previous = max(self.last_frame, self._frame())
        expected = np.asarray(usd_world_matrix(position, yaw_deg), dtype=float)
        pose_updated = self.set_body_pose(position, yaw_deg)
        last_matrix = None
        pose_frame = None
        settling = FrameStability()
        previous_rgb = None
        for update in range(max_updates):
            # Pump the whole synthetic-data graph on both paths. World.step alone
            # does not refresh every auxiliary annotator after orchestrator capture.
            rep.orchestrator.step(
                rt_subframes=STABILITY_POLICY["render_subframes"], pause_timeline=False
            )
            frame = self._frame()
            if frame <= previous:
                continue
            reference = (
                self.reference.get_data() if self.reference else {"referenceTimeNumerator": frame}
            )
            params = self.params.get_data()
            if not reference or not params or int(reference["referenceTimeNumerator"]) != frame:
                previous_rgb = None
                settling.reset()
                pose_frame = None
                continue
            view = np.asarray(params["cameraViewTransform"]).reshape(4, 4)
            last_matrix = np.linalg.inv(view)
            if not np.allclose(last_matrix, expected, atol=1e-4, rtol=0):
                previous_rgb = None
                settling.reset()
                pose_frame = None
                continue
            # A pose-matched header can arrive before the first color buffer.
            # Require a subsequent advancing frame with that same pose.
            if pose_frame is None:
                pose_frame = frame
                continue
            if frame <= pose_frame or frame <= settling.last_frame:
                continue
            raw = self.sensor.get_current_frame()["rgba"] if self.sensor else self.rgb.get_data()
            raw = np.asarray(raw).copy()
            width, height = self.settings["resolution_wh"]
            if raw.shape[:2] != (height, width) or raw.ndim != 3 or raw.shape[2] < 3:
                previous_rgb = None
                settling.reset()
                continue
            if self._frame() != frame:
                previous_rgb = None
                settling.reset()
                continue
            if raw.dtype != np.uint8:
                raise ValueError("Expected uint8 RGB from the renderer")
            rgb = np.ascontiguousarray(raw[:, :, :3])
            if not np.any(rgb):
                previous_rgb = None
                settling.reset()
                continue
            actual = {
                name: float(self.prim.GetAttribute(name).Get())
                for name in ("focalLength", "horizontalAperture", "verticalAperture", "fStop")
            }
            expected_optics = {
                "focalLength": self.settings["usd_focal_length"],
                "horizontalAperture": self.settings["usd_aperture_wh"][0],
                "verticalAperture": self.settings["usd_aperture_wh"][1],
                "fStop": 0.0,
            }
            if any(abs(actual[name] - value) > 1e-6 for name, value in expected_optics.items()):
                raise ValueError("Renderer overwrote camera optics: " + str(actual))
            if self.sensor is not None and not np.allclose(
                self.sensor.get_intrinsics_matrix(), self.settings["intrinsics"], atol=0.001, rtol=0
            ):
                raise ValueError("Camera API intrinsics disagree with the shared pinhole model")
            mae = (
                float(np.abs(rgb.astype(np.float32) - previous_rgb).mean())
                if previous_rgb is not None
                else None
            )
            previous_rgb = rgb.astype(np.float32)
            if not settling.observe(frame, mae):
                continue
            self.last_frame = frame
            self.last_receipt = {
                "pose_id": str(pose_id),
                "backend": self.backend,
                "rendering_frame": frame,
                "reference_time_numerator": int(reference["referenceTimeNumerator"]),
                "first_pose_matched_frame": pose_frame,
                "camera_prim": self.path,
                "render_product": self.product_path,
                "position_m": position,
                "body_yaw_deg": float(yaw_deg),
                "pose_was_updated": pose_updated,
                "usd_quaternion_wxyz": usd_quaternion_wxyz(yaw_deg),
                "camera_world_matrix": last_matrix.tolist(),
                "updates_waited": update + 1,
                "elapsed_s": time.perf_counter() - started,
                "stability": settling.summary(),
                "resolution_wh": [width, height],
                "observed_usd_optics": actual,
                "render_step": "replicator_synchronized",
                "render_camera_projection": np.asarray(params["cameraProjection"])
                .reshape(4, 4)
                .tolist(),
            }
            return rgb, dict(self.last_receipt)
        from pxr import UsdGeom

        authored = UsdGeom.XformCache().GetLocalToWorldTransform(self.prim)
        self.last_failure = {
            "pose_id": str(pose_id),
            "updates_waited": max_updates,
            "stability": settling.summary(),
            "elapsed_s": time.perf_counter() - started,
        }
        message = (
            "No settled pose-matched RGB for {} on {} / {}: "
            "observed matrix {}, expected {}, USD {}; settling {}"
        )
        raise RuntimeError(
            message.format(
                pose_id,
                self.path,
                self.product_path,
                last_matrix,
                expected,
                authored,
                self.last_failure,
            )
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.sensor is not None:
            self.sensor.pause()
        for annotator in (self.params, self.reference, self.rgb):
            if annotator is not None:
                annotator.detach([self.product_path])
        if self.product is not None:
            self.product.destroy()
