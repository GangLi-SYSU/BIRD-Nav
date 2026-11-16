"""Render the imported eagle in the actual scene before enabling it in the demo."""
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from omni.isaac.kit import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RayTracedLighting','width':1280,'height':720,'anti_aliasing':0})
try:
 import omni.usd
 import omni.replicator.core as rep
 import numpy as np
 from PIL import Image
 from pxr import Gf,UsdGeom
 from insect_utils.interactive_scene import create_demo_scene
 omni.usd.get_context().open_stage(str(PROJECT/'maps/downward_real_h40_v1/real_scene.usda'))
 for _ in range(8):app.update()
 stage=omni.usd.get_context().get_stage();stage.SetEditTarget(stage.GetSessionLayer())
 bird,pos,yaw,anim,_=create_demo_scene(stage);pos.Set(Gf.Vec3d(0,0,40));yaw.Set(0)
 camera=UsdGeom.Camera.Define(stage,'/World/EaglePreview')
 camera.GetFocalLengthAttr().Set(26);camera.GetHorizontalApertureAttr().Set(36);camera.GetVerticalApertureAttr().Set(20.25)
 camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(-5,3,43),Gf.Vec3d(0,0,40),Gf.Vec3d(0,0,1)).GetInverse())
 product=rep.create.render_product(str(camera.GetPath()),(1280,720));rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product])
 out=PROJECT/'runs/eagle_preview_01';out.mkdir(exist_ok=True)
 for i,t in enumerate([0,.4,.8,1.2]):
  anim.update(t,'MANUAL')
  for _ in range(10):rep.orchestrator.step(rt_subframes=2,pause_timeline=False)
  image=np.asarray(rgb.get_data()).copy();Image.fromarray(image).save(out/f'eagle_{i}.png')
 print('EAGLE_PREVIEW_DONE',flush=True)
except Exception:
 import traceback
 (PROJECT/'runs/eagle_preview_error.log').write_text(traceback.format_exc())
 raise
finally:app.close()
