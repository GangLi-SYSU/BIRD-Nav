"""Convert the licensed Quaternius eagle FBX with its authored flight animation."""
from pathlib import Path
from omni.isaac.kit import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RayTracedLighting','anti_aliasing':0})
try:
    import asyncio
    import omni.kit.app
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('omni.kit.asset_converter',True)
    import omni.kit.asset_converter as converter
    asset=Path('/isaac-sim/CodeInsectInspired/assets/birds/quaternius_eagle')
    async def convert():
        context=converter.AssetConverterContext()
        context.ignore_animations=False
        task=converter.get_instance().create_converter_task(str(asset/'Eagle.fbx'),str(asset/'eagle.usd'),None,context)
        if not await task.wait_until_finished():
            raise RuntimeError(task.get_error_message())
        from pxr import Usd,UsdGeom
        stage=Usd.Stage.Open(str(asset/'eagle.usd'))
        print('EAGLE_CONVERTED', stage.GetStartTimeCode(),stage.GetEndTimeCode(),stage.GetTimeCodesPerSecond(),flush=True)
        for prim in stage.Traverse():
            if prim.GetTypeName() in ['SkelAnimation','Skeleton','Mesh']:
                print(prim.GetPath(),prim.GetTypeName(),flush=True)
    future=asyncio.ensure_future(convert())
    while not future.done():app.update()
    future.result()
finally:app.close()
