"""Licensed Quaternius eagle with its rig, authored poses and display animation."""
import math
from pathlib import Path


def create_eagle(stage):
    from pxr import Gf, UsdGeom, UsdSkel, UsdShade, Sdf, Vt

    asset = Path(__file__).resolve().parents[1] / "assets/birds/quaternius_eagle/eagle.usd"
    if not asset.exists():
        raise FileNotFoundError("Converted eagle asset missing: " + str(asset))
    root = UsdGeom.Xform.Define(stage, "/World/Bird")
    position = root.AddTranslateOp()
    yaw = root.AddRotateZOp()
    model = UsdGeom.Xform.Define(stage, "/World/Bird/Model")
    model.GetPrim().GetReferences().AddReference(str(asset))
    model.GetPrim().GetVariantSets().GetVariantSet("animation_track").SetVariantSelection(
        "eaglearmature_flying"
    )
    model.GetPrim().GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3f(90, 0, -90))
    model.GetPrim().GetAttribute("xformOp:scale").Set(Gf.Vec3f(0.0023))
    # The FBX's material names survive import; restore a natural eagle palette.
    palette = {
        "Head": (0.68, 0.63, 0.48),
        "Wings": (0.13, 0.075, 0.035),
        "Beak": (0.75, 0.41, 0.025),
        "Claws": (0.55, 0.30, 0.025),
    }
    for name, color in palette.items():
        path = "/World/Bird/Model/Looks/" + name
        prim = stage.GetPrimAtPath(path)
        if not prim:
            continue
        material = UsdShade.Material(prim)
        mdl = UsdShade.Shader(stage.GetPrimAtPath(path + "/" + name))
        if mdl:
            mdl.GetInput("diffuse_color_constant").Set(Gf.Vec3f(*color))
        surface = UsdShade.Shader.Define(stage, path + "/DisplaySurface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    arm = "/World/Bird/Model/EagleArmature"
    skeleton = UsdSkel.Skeleton(stage.GetPrimAtPath(arm + "/Skeleton"))
    joint_names = list(skeleton.GetJointsAttr().Get())
    rest = skeleton.GetRestTransformsAttr().Get()
    indices = {str(name): i for i, name in enumerate(joint_names)}
    parent = [
        indices.get(str(name).rsplit("/", 1)[0], -1) if "/" in str(name) else -1
        for name in joint_names
    ]
    world = []
    for i, local in enumerate(rest):
        world.append(local if parent[i] < 0 else local * world[parent[i]])
    xs = [float(m[3][0]) for m in world]
    flap_roots = []
    for side in [-1, 1]:
        tip = max(range(len(xs)), key=lambda i: side * xs[i])
        threshold = abs(xs[tip]) * 0.2
        root_index = tip
        while parent[root_index] >= 0 and abs(xs[parent[root_index]]) > threshold:
            root_index = parent[root_index]
        axis = world[root_index].GetInverse().TransformDir(Gf.Vec3d(0, 0, 1)).GetNormalized()
        flap_roots.append((root_index, side, Gf.Vec3f(*axis)))
    fly = UsdSkel.Animation(stage.GetPrimAtPath(arm + "/EagleArmature_Flying"))
    idle = UsdSkel.Animation(stage.GetPrimAtPath(arm + "/EagleArmature_Idle"))
    player = UsdSkel.Animation.Define(stage, arm + "/DisplayPlayback")
    player.CreateJointsAttr().Set(fly.GetJointsAttr().Get())
    UsdSkel.BindingAPI.Apply(
        stage.GetPrimAtPath(arm + "/Skeleton")
    ).CreateAnimationSourceRel().SetTargets([player.GetPath()])

    class EagleAnimator:
        def __init__(self):
            self.last = None
            self.clock = 0.0
            self.weight = 0.0
            self.fly_rot = fly.GetRotationsAttr().Get(0)
            self.idle_rot = idle.GetRotationsAttr().Get(0)
            player.CreateTranslationsAttr().Set(fly.GetTranslationsAttr().Get(0))
            player.CreateScalesAttr().Set(fly.GetScalesAttr().Get(0))

        def update(self, now, phase, paused=False):
            dt = 0 if self.last is None else max(0, min(0.1, now - self.last))
            self.last = now
            if not paused:
                self.clock += dt
            target = 0 if phase in ("GROUND", "LANDED") else 0.5 + 0.5 * math.sin(self.clock * 7.5)
            if not paused:
                self.weight += max(-dt * 4, min(dt * 4, target - self.weight))
            values = [Gf.Slerp(self.weight, a, b) for a, b in zip(self.idle_rot, self.fly_rot)]
            flap = 0 if phase in ("GROUND", "LANDED") else 22 * math.sin(self.clock * 7.5)
            for index, side, axis in flap_roots:
                half = math.radians(side * flap) / 2
                rotation = Gf.Quatf(math.cos(half), axis * math.sin(half))
                values[index] = Gf.Quatf(values[index]) * rotation
            quats = Vt.QuatfArray(values)
            player.CreateRotationsAttr().Set(quats)

    animator = EagleAnimator()
    animator.update(0, "GROUND")
    return root.GetPrim(), position, yaw, animator
