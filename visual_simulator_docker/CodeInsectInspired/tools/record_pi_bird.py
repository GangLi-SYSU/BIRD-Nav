"""Render a third-person bird replay without changing frozen scene/experiments."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if not args.smoke and (out / 'completion.json').exists():
        raise FileExistsError('Completed recording already exists')
    protocol = json.loads((out / 'protocol.json').read_text())
    from omni.isaac.kit import SimulationApp
    app = SimulationApp(dict(headless=True, width=1280, height=720,
                             renderer='RayTracedLighting', anti_aliasing=0, multi_gpu=False))
    writer = None
    try:
        import cv2
        import numpy as np
        import omni.usd
        import omni.replicator.core as rep
        from pxr import UsdGeom, UsdShade, Gf, Sdf
        import carb
        project = Path(__file__).resolve().parents[1]
        scene = project / 'maps/downward_real_h40_v1/real_scene.usda'
        omni.usd.get_context().open_stage(str(scene))
        for _ in range(12):
            app.update()
        stage = omni.usd.get_context().get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        settings = carb.settings.get_settings()
        settings.set('/rtx/post/aa/op', 0)
        settings.set('/rtx/post/histogram/enabled', False)
        ground = UsdGeom.Mesh(stage.GetPrimAtPath('/World/Ground'))
        ground.GetPointsAttr().Set([(-3500,-3500,0),(3500,-3500,0),
                                    (3500,3500,0),(-3500,3500,0)])
        ground.GetPrim().GetAttribute('primvars:st').Set([(0,0),(1750,0),(1750,1750),(0,1750)])

        def material(name, color):
            mat = UsdShade.Material.Define(stage, '/World/BirdLooks/' + name)
            shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + '/Surface')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(color)
            shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.7)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
            return mat
        white = material('Ivory', (.86,.83,.72))
        dark = material('Feathers', (.055,.075,.10))
        gold = material('Beak', (.95,.48,.055))
        red = material('Home', (.15,.7,.25))

        bird = UsdGeom.Xform.Define(stage, '/World/Bird')
        translate = bird.AddTranslateOp()
        heading_op = bird.AddRotateZOp()

        def ellipsoid(path, center, scale, mat):
            obj = UsdGeom.Sphere.Define(stage, path)
            obj.GetRadiusAttr().Set(1)
            obj.AddTranslateOp().Set(Gf.Vec3d(*center))
            obj.AddScaleOp().Set(Gf.Vec3f(*scale))
            UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(mat)
            return obj

        def mesh(path, points, faces, mat):
            obj = UsdGeom.Mesh.Define(stage, path)
            obj.GetPointsAttr().Set(points)
            obj.GetFaceVertexCountsAttr().Set([len(f) for f in faces])
            obj.GetFaceVertexIndicesAttr().Set([v for f in faces for v in f])
            obj.GetSubdivisionSchemeAttr().Set('none')
            obj.GetDoubleSidedAttr().Set(True)
            UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(mat)
            return obj

        ellipsoid('/World/Bird/Body', (0,0,0), (.67,.25,.24), white)
        ellipsoid('/World/Bird/Neck', (.48,0,.15), (.28,.18,.22), white)
        ellipsoid('/World/Bird/Head', (.68,0,.24), (.24,.18,.19), white)
        for sign in [-1,1]:
            ellipsoid('/World/Bird/Eye'+str(sign).replace('-','M'),
                      (.76,sign*.157,.29), (.035,.025,.035), dark)
        mesh('/World/Bird/Beak', [(.81,-.10,.23),(.81,.10,.23),(.83,0,.33),
                                (.83,0,.17),(1.18,0,.23)],
             [[0,2,4],[2,1,4],[1,3,4],[3,0,4]], gold)
        mesh('/World/Bird/Tail', [(-.43,-.14,0),(-.43,.14,0),(-1.12,.38,-.04),
                                (-.98,0,-.06),(-1.12,-.38,-.04)], [[0,1,2,3,4]], dark)
        wing_ops = []
        for side, sign in [('Left',1),('Right',-1)]:
            wing = UsdGeom.Xform.Define(stage, '/World/Bird/'+side)
            wing.AddTranslateOp().Set(Gf.Vec3d(.04,sign*.15,.05))
            wing_ops.append((wing.AddRotateXOp(),sign))
            pts = [(.34,0,0),(.18,sign*.65,.01),(-.12,sign*1.3,0),
                   (-.49,sign*1.15,0),(-.52,sign*.55,-.04),(-.38,0,-.04)]
            mesh(str(wing.GetPath())+'/Wing', pts, [[0,1,2,3,4,5]], white)
            for j in range(6):
                y = .76+j*.085
                x = -.14-j*.065
                mesh(str(wing.GetPath())+'/Feather'+str(j),
                     [(x+.12,sign*y,0),(x+.04,sign*(y+.34),-.02),
                      (x-.055,sign*(y+.37),-.025),(x-.13,sign*y,-.025)],
                     [[0,1,2,3]], dark)

        # A visible 15 m handoff circle is explanatory geometry, not a new landmark dataset.
        ring = UsdGeom.BasisCurves.Define(stage, '/World/VisualHandoffCircle')
        theta = np.linspace(0,2*np.pi,129)
        ring.GetPointsAttr().Set([(15*math.cos(t),15*math.sin(t),.12) for t in theta])
        ring.GetCurveVertexCountsAttr().Set([129])
        ring.GetTypeAttr().Set('linear')
        ring.GetWidthsAttr().Set([.18] * 129)
        UsdShade.MaterialBindingAPI.Apply(ring.GetPrim()).Bind(red)
        camera = UsdGeom.Camera.Define(stage, '/World/ThirdPersonCamera')
        camera.GetFocalLengthAttr().Set(26)
        camera.GetHorizontalApertureAttr().Set(36)
        camera.GetVerticalApertureAttr().Set(20.25)
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(.1,10000))
        camera_op = camera.AddTransformOp()
        product = rep.create.render_product(str(camera.GetPath()), (1280,720))
        annotator = rep.AnnotatorRegistry.get_annotator('rgb')
        annotator.attach([product])
        positions = np.load(out/'trajectory.npy')
        # Slow departure/arrival, accelerate the otherwise 35-minute journey.
        keys_video = np.array([0,4,12,42,54,59.958333333])
        keys_step = np.array([0,0,80,4120,4200,4200])
        fps = 24
        count = 1440
        indices = [0, 200, 900, 1300, 1439] if args.smoke else range(count)
        if not args.smoke:
            writer = cv2.VideoWriter(str(out/'pi_bird_third_person.mp4'),
                                     cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,720))
            if not writer.isOpened():
                raise RuntimeError('MP4 encoder unavailable')
        log = []
        previous_yaw = None
        for frame in indices:
            vt = frame/fps
            index = float(np.interp(vt,keys_video,keys_step))
            i = min(int(index),len(positions)-2)
            fraction = index-i
            xy = positions[i]*(1-fraction)+positions[i+1]*fraction
            # Use a short look-ahead for smooth cornering; positional path stays exact.
            lo,hi = max(0,i-8), min(len(positions)-1,i+8)
            direction = positions[hi]-positions[lo]
            yaw = math.atan2(direction[1],direction[0])
            if previous_yaw is not None:
                yaw = previous_yaw + .20*math.atan2(math.sin(yaw-previous_yaw),
                                                  math.cos(yaw-previous_yaw))
            previous_yaw = yaw
            pos = np.array([xy[0],xy[1],40.])
            translate.Set(Gf.Vec3d(*pos))
            heading_op.Set(math.degrees(yaw))
            for op,sign in wing_ops:
                op.Set(sign*(8+18*math.sin(2*math.pi*1.2*vt)))
            forward = np.array([math.cos(yaw),math.sin(yaw),0])
            side = np.array([-math.sin(yaw),math.cos(yaw),0])
            eye = pos - 8*forward + 4*side + np.array([0,0,4])
            target = pos + 1.5*forward - np.array([0,0,1.2])
            camera_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),
                                                Gf.Vec3d(0,0,1)).GetInverse())
            for _ in range(20 if frame == 0 else 2):
                rep.orchestrator.step(rt_subframes=1,pause_timeline=False)
            rgb = np.asarray(annotator.get_data()).copy()
            if rgb.shape != (720,1280,4) or rgb.dtype != np.uint8 or not np.any(rgb[:,:,:3]):
                raise RuntimeError('Invalid rendered frame: '+str(rgb.shape))
            bgr = cv2.cvtColor(rgb[:,:,:3],cv2.COLOR_RGB2BGR)
            phase = 'OUTBOUND' if index < 3200 else 'PATH-INTEGRATION HOMING'
            if index >= 4200:
                phase = 'PI COMPLETE - VISUAL STAGE HANDOFF'
            next_index = float(np.interp(min(vt+1/fps,keys_video[-1]),keys_video,keys_step))
            speedup = (next_index-index)*.5*fps
            cv2.rectangle(bgr,(0,0),(1280,104),(22,27,31),-1)
            cv2.putText(bgr,phase,(24,34),cv2.FONT_HERSHEY_SIMPLEX,.80,(220,240,240),2)
            text = f'Height 40 m | Home {np.linalg.norm(xy):.1f} m | Flight time {index*.5:.1f} s | Playback {speedup:.1f}x'
            cv2.putText(bgr,text,(24,72),cv2.FONT_HERSHEY_SIMPLEX,.60,(220,220,220),1)
            cv2.rectangle(bgr,(0,680),(1280,720),(22,27,31),-1)
            cv2.putText(bgr,'ISAAC SIM | Ideal magnetic heading | sigma=0.1 | Trial 0 | Kinematic replay; animated bird',
                        (20,706),cv2.FONT_HERSHEY_SIMPLEX,.52,(220,220,220),1)
            # Top-right minimap, computed from recorded coordinates without modifying motion.
            origin=np.array([1140,235]); scale=.047
            path_points=(positions* np.array([scale,-scale])+origin).astype(np.int32)
            cv2.rectangle(bgr,(1015,110),(1270,365),(240,240,235),-1)
            cv2.polylines(bgr,[path_points[:3201]],False,(140,150,30),1)
            cv2.polylines(bgr,[path_points[3200:]],False,(30,150,230),1)
            here=(xy*np.array([scale,-scale])+origin).astype(int)
            cv2.circle(bgr,tuple(origin),4,(40,120,30),-1)
            cv2.circle(bgr,tuple(here),5,(30,30,220),-1)
            cv2.putText(bgr,'Route overview',(1030,352),cv2.FONT_HERSHEY_SIMPLEX,.45,(40,40,40),1)
            if args.smoke or frame in [0,200,900,1300,1439]:
                cv2.imwrite(str(out/f'preview_{frame:04d}.png'),bgr)
            if writer is not None:
                writer.write(bgr)
            log.append(dict(frame=frame,video_time_s=vt,trajectory_index=index,
                            position_m=pos.tolist(),home_distance_m=float(np.linalg.norm(xy))))
            if frame % 48 == 0 or args.smoke:
                (out/'progress.json').write_text(json.dumps(dict(frame=frame,total=count,
                                                                smoke=args.smoke)))
                print(f'RENDERED {frame+1}/{count}',flush=True)
        if writer is not None:
            writer.release()
            writer=None
            capture=cv2.VideoCapture(str(out/'pi_bird_third_person.mp4'))
            frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            assert frames == count, frames
            capture.set(cv2.CAP_PROP_POS_FRAMES,count-1)
            ok,last=capture.read()
            assert ok and last.shape == (720,1280,3)
            capture.release()
            (out/'frames.json').write_text(json.dumps(log))
            (out/'completion.json').write_text(json.dumps(dict(status='complete',frames=frames,
                fps=fps,duration_s=frames/fps,scene=str(scene),altitude_m=40,
                bird='procedural animated mesh, 3 m wingspan',terrain_size_m=7000,
                original_scene_modified=False,final_distance_m=protocol['final_distance_m']),indent=2))
        annotator.detach([product.path])
        product.destroy()
        print('RECORDING_DONE',flush=True)
    finally:
        if writer is not None:
            writer.release()
        app.close()


if __name__ == '__main__':
    main()
