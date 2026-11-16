"""Licensed eagle and display markers in the interactive scene."""
import math


def create_demo_scene(stage):
    import numpy as np
    from pxr import Sdf, UsdGeom, UsdShade

    ground = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Ground"))
    ground.GetPointsAttr().Set(
        [(-3500, -3500, 0), (3500, -3500, 0), (3500, 3500, 0), (-3500, 3500, 0)]
    )
    ground.GetPrim().GetAttribute("primvars:st").Set([(0, 0), (1750, 0), (1750, 1750), (0, 1750)])

    def material(name, color):
        mat = UsdShade.Material.Define(stage, "/World/BirdLooks/" + name)
        shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    from insect_utils.demo_eagle import create_eagle

    bird, translate, heading_op, wing_ops = create_eagle(stage)
    red = material("Home", (0.15, 0.7, 0.25))

    # A visible 15 m handoff circle is explanatory geometry, not a new landmark dataset.
    ring = UsdGeom.BasisCurves.Define(stage, "/World/VisualHandoffCircle")
    theta = np.linspace(0, 2 * np.pi, 129)
    ring.GetPointsAttr().Set([(15 * math.cos(t), 15 * math.sin(t), 0.12) for t in theta])
    ring.GetCurveVertexCountsAttr().Set([129])
    ring.GetTypeAttr().Set("linear")
    ring.GetWidthsAttr().Set([0.18] * 129)
    UsdShade.MaterialBindingAPI.Apply(ring.GetPrim()).Bind(red)
    return bird, translate, heading_op, wing_ops, ring.GetPrim()
