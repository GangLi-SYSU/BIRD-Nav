from omni.isaac.kit import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RayTracedLighting'})
try:
 from pxr import Usd,UsdGeom,UsdSkel
 from pathlib import Path
 s=Usd.Stage.Open('/isaac-sim/CodeInsectInspired/assets/birds/quaternius_eagle/eagle.usd')
 report=['axis '+str(UsdGeom.GetStageUpAxis(s))+' unit '+str(UsdGeom.GetStageMetersPerUnit(s)),str(UsdGeom.BBoxCache(0,[UsdGeom.Tokens.default_]).ComputeWorldBound(s.GetDefaultPrim()).ComputeAlignedRange())]
 for p in s.Traverse():
  if p.GetTypeName()=='SkelAnimation':
   a=UsdSkel.Animation(p);report.append(str(p.GetPath())+' '+str(a.GetRotationsAttr().GetTimeSamples())+' joints '+str(len(a.GetJointsAttr().Get())))
 report.append(s.GetRootLayer().ExportToString()[:8000])
 s.Flatten().Export('/isaac-sim/CodeInsectInspired/assets/birds/quaternius_eagle/eagle_flatten.usda')
 for p in s.Traverse():
  if p.GetTypeName()=='SkelAnimation':
   a=UsdSkel.Animation(p); report.append(str(a.GetJointsAttr().Get())); report.append(str([(x.GetName(),len(x.GetTimeSamples())) for x in p.GetAttributes()]))
 Path('/isaac-sim/CodeInsectInspired/assets/birds/quaternius_eagle/inspection.txt').write_text('\n'.join(report))
 print('EAGLE_INSPECTED',flush=True)
finally:app.close()
