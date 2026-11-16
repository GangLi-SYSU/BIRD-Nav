"""Takeoff, full-area overhead PI replay, and third-person visual handoff."""
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
        timestamps = np.load(out/'timestamps.npy')
        split = protocol['outbound_steps']
        exit_index = protocol['exit_index']
        reentry_index = protocol['reentry_index']
        keys_video = np.array([0,8,13,14.5,60,61.5,67,74.958333333])
        keys_step = np.array([0,0,exit_index,exit_index,reentry_index,reentry_index,
                              len(positions)-1,len(positions)-1])
        fps = 24
        count = 1800
        indices = [0,120,300,600,1200,1440,1608,1799] if args.smoke else range(count)
        center = (positions.min(axis=0)+positions.max(axis=0))/2
        extent = np.ptp(positions,axis=0)
        overview_width = max(extent[0]*1.28, extent[1]*1280/520*1.30)
        overview_eye = np.array([*center,40+overview_width*26/36])
        overview_target = np.array([*center,40.])
        def smooth(value):
            t = min(1.,max(0.,value))
            return t*t*(3-2*t)
        if not args.smoke:
            writer = cv2.VideoWriter(str(out/'pi_bird_multiview.mp4'),
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
            altitude = .30 + 39.70*smooth(vt/8)
            pos = np.array([xy[0],xy[1],altitude])
            translate.Set(Gf.Vec3d(*pos))
            heading_op.Set(math.degrees(yaw))
            for op,sign in wing_ops:
                op.Set(sign*(8+18*min(1.,vt)*math.sin(2*math.pi*1.2*vt)))
            forward = np.array([math.cos(yaw),math.sin(yaw),0])
            side = np.array([-math.sin(yaw),math.cos(yaw),0])
            chase_eye = pos - 5.5*forward + 2.8*side + np.array([0,0,3.3])
            chase_target = pos + .5*forward - np.array([0,0,.3])
            if vt < 13:
                blend = 0.
            elif vt < 14.5:
                blend = smooth((vt-13)/1.5)
            elif vt < 60:
                blend = 1.
            else:
                blend = 1-smooth((vt-60)/1.5)
            eye = (1-blend)*chase_eye + blend*overview_eye
            target = (1-blend)*chase_target + blend*overview_target
            up = np.array([0,blend,1-blend])
            camera_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),
                                                Gf.Vec3d(*up)).GetInverse())
            for _ in range(20 if frame == 0 else 2):
                rep.orchestrator.step(rt_subframes=1,pause_timeline=False)
            rgb = np.asarray(annotator.get_data()).copy()
            if rgb.shape != (720,1280,4) or rgb.dtype != np.uint8 or not np.any(rgb[:,:,:3]):
                raise RuntimeError('Invalid rendered frame: '+str(rgb.shape))
            bgr = cv2.cvtColor(rgb[:,:,:3],cv2.COLOR_RGB2BGR)
            phase = 'OUTBOUND SEARCH' if index < split else 'PATH-INTEGRATION HOMING'
            if vt < 8:
                phase = 'TAKEOFF FROM GROUND (DISPLAY ANIMATION)'
            elif index >= len(positions)-1:
                phase = 'PI COMPLETE - INSIDE 15 m VISUAL RANGE'
            mode = 'FULL-AREA OVERHEAD' if blend >= .999 else ('THIRD PERSON' if blend <= .001 else 'CAMERA TRANSITION')
            next_index = float(np.interp(min(vt+1/fps,keys_video[-1]),keys_video,keys_step))
            simulation_time = float(np.interp(index,np.arange(len(timestamps)),timestamps))
            next_time = float(np.interp(next_index,np.arange(len(timestamps)),timestamps))
            speedup = (next_time-simulation_time)*fps
            cv2.rectangle(bgr,(0,0),(1280,104),(22,27,31),-1)
            cv2.putText(bgr,phase,(24,34),cv2.FONT_HERSHEY_SIMPLEX,.80,(220,240,240),2)
            text = f'Height {altitude:.1f} m | Home {np.linalg.norm(xy):.1f} m | PI time {simulation_time:.1f} s | Playback {speedup:.1f}x'
            cv2.putText(bgr,text,(24,72),cv2.FONT_HERSHEY_SIMPLEX,.60,(220,220,220),1)
            cv2.rectangle(bgr,(0,680),(1280,720),(22,27,31),-1)
            cv2.putText(bgr,'ISAAC SIM | Ideal compass | sigma=0.1 | New search route | Kinematic replay | '+mode,
                        (20,706),cv2.FONT_HERSHEY_SIMPLEX,.52,(220,220,220),1)
            if blend >= .999:
                scale = 1280/overview_width
                origin = np.array([640,360])
                projected = ((positions-center)*np.array([scale,-scale])+origin).astype(np.int32)
                last = min(i+1,len(positions)-1)
                cv2.polylines(bgr,[projected[:min(last,split)+1]],False,(210,110,30),3,cv2.LINE_AA)
                if last > split:
                    cv2.polylines(bgr,[projected[split:last+1]],False,(30,150,245),3,cv2.LINE_AA)
                home = ((-center)*np.array([scale,-scale])+origin).astype(int)
                current = ((xy-center)*np.array([scale,-scale])+origin).astype(int)
                cv2.circle(bgr,tuple(home),max(2,round(15*scale)),(40,180,40),2,cv2.LINE_AA)
                cv2.drawMarker(bgr,tuple(home),(30,240,30),cv2.MARKER_STAR,18,2)
                cv2.putText(bgr,'HOME / 15 m',(int(home[0])+12,int(home[1])+25),
                            cv2.FONT_HERSHEY_SIMPLEX,.6,(30,240,30),2,cv2.LINE_AA)
                cv2.circle(bgr,tuple(current),9,(0,255,255),2,cv2.LINE_AA)
                cv2.circle(bgr,tuple(current),3,(0,255,255),-1)
                cv2.putText(bgr,'Blue: outbound | Orange: homing | Yellow: bird',
                            (24,647),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA)
                scale_length=round(500*scale)
                cv2.line(bgr,(970,630),(970+scale_length,630),(255,255,255),3)
                cv2.putText(bgr,'500 m',(970,620),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
            if args.smoke or frame in [0,120,300,600,1200,1440,1608,1799]:
                cv2.imwrite(str(out/f'preview_{frame:04d}.png'),bgr)
            if writer is not None:
                writer.write(bgr)
            log.append(dict(frame=frame,video_time_s=vt,trajectory_index=index,
                            position_m=pos.tolist(),home_distance_m=float(np.linalg.norm(xy)),
                            overhead_blend=blend,simulation_time_s=simulation_time))
            if frame % 48 == 0 or args.smoke:
                (out/'progress.json').write_text(json.dumps(dict(frame=frame,total=count,
                                                                smoke=args.smoke)))
                print(f'RENDERED {frame+1}/{count}',flush=True)
        if writer is not None:
            writer.release()
            writer=None
            capture=cv2.VideoCapture(str(out/'pi_bird_multiview.mp4'))
            frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            assert frames == count, frames
            capture.set(cv2.CAP_PROP_POS_FRAMES,count-1)
            ok,last=capture.read()
            assert ok and last.shape == (720,1280,3)
            capture.release()
            (out/'frames.json').write_text(json.dumps(log))
            (out/'completion.json').write_text(json.dumps(dict(status='complete',frames=frames,
                fps=fps,duration_s=frames/fps,scene=str(scene),cruise_altitude_m=40,
                camera='third-person -> full-area overhead -> third-person',
                exit_index=exit_index,reentry_index=reentry_index,
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
