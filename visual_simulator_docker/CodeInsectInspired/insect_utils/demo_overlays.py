"""North-up overview symbols; headings use the actual body orientation."""
import math
import numpy as np


def nose_tip(pixel, yaw_deg, length=26):
    angle = math.radians(yaw_deg)
    return np.rint(
        np.asarray(pixel) + length * np.array([math.cos(angle), -math.sin(angle)])
    ).astype(int)


def annotate_overview(rgba, trail, position, yaw, center, width, landing=None):
    import cv2

    image = rgba.copy()
    scale = 480 / width

    def project(xy):
        return np.rint((np.asarray(xy) - center) * [scale, -scale] + [240, 180]).astype(int)

    pixels = project(trail)
    cv2.polylines(image, [pixels], False, (30, 160, 255, 255), 2, cv2.LINE_AA)
    home = project([0, 0])
    here = project(position[:2])
    cv2.circle(image, tuple(home), max(3, int(15 * scale)), (50, 230, 50, 255), 1)
    cv2.drawMarker(image, tuple(home), (50, 255, 50, 255), cv2.MARKER_STAR, 12, 2)
    tip = nose_tip(here, yaw)
    cv2.circle(image, tuple(here), 5, (255, 230, 0, 255), 2)
    cv2.arrowedLine(
        image, tuple(here), tuple(tip), (255, 230, 0, 255), 2, cv2.LINE_AA, tipLength=0.35
    )
    cv2.putText(
        image,
        f"Nose {yaw%360:.0f} deg | N up",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if landing is not None:
        point = np.asarray(landing["landing_position_m"][:2])
        end = project(point)
        cv2.line(image, tuple(home), tuple(end), (255, 90, 210, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(image, tuple(end), (255, 90, 210, 255), cv2.MARKER_DIAMOND, 12, 2)
        distance = landing["horizontal_distance_m"]
        # A local inset keeps small landing errors visible after a kilometre-scale flight.
        cv2.rectangle(image, (264, 199), (479, 359), (24, 29, 34, 255), -1)
        mid = point / 2
        local_scale = 150 / max(6.0, float(np.linalg.norm(point)) * 1.6)
        origin = np.array([369, 279])
        a = np.rint((-mid) * [local_scale, -local_scale] + origin).astype(int)
        b = np.rint((point - mid) * [local_scale, -local_scale] + origin).astype(int)
        cv2.line(image, tuple(a), tuple(b), (255, 90, 210, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(image, tuple(a), (50, 255, 50, 255), cv2.MARKER_STAR, 12, 2)
        cv2.drawMarker(image, tuple(b), (255, 90, 210, 255), cv2.MARKER_DIAMOND, 10, 2)
        cv2.putText(
            image,
            "START",
            (int(a[0]) - 43, int(a[1]) - 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (50, 255, 50, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "LAND",
            (int(b[0]) + 8, int(b[1]) + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 90, 210, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Landing error: {distance:.2f} m",
            (274, 223),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "Horizontal distance",
            (274, 346),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (220, 220, 220, 255),
            1,
            cv2.LINE_AA,
        )
    return image


def create_landing_markers(stage):
    from pxr import UsdGeom, UsdShade, Gf, Sdf

    def material(name, color):
        mat = UsdShade.Material.Define(stage, "/World/LandingDisplay/" + name)
        shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color) * 0.3)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    green = material("StartMaterial", (0.05, 0.8, 0.12))
    pink = material("LandMaterial", (0.9, 0.05, 0.55))
    start = UsdGeom.Cylinder.Define(stage, "/World/LandingDisplay/Start")
    start.GetRadiusAttr().Set(0.6)
    start.GetHeightAttr().Set(0.04)
    start.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.09))
    UsdShade.MaterialBindingAPI.Apply(start.GetPrim()).Bind(green)
    end = UsdGeom.Cylinder.Define(stage, "/World/LandingDisplay/Touchdown")
    end.GetRadiusAttr().Set(0.35)
    end.GetHeightAttr().Set(0.04)
    end_pose = end.AddTranslateOp()
    UsdShade.MaterialBindingAPI.Apply(end.GetPrim()).Bind(pink)
    line = UsdGeom.BasisCurves.Define(stage, "/World/LandingDisplay/Distance")
    line.GetTypeAttr().Set("linear")
    line.GetCurveVertexCountsAttr().Set([2])
    line.GetWidthsAttr().Set([0.06, 0.06])
    UsdShade.MaterialBindingAPI.Apply(line.GetPrim()).Bind(pink)
    end.MakeInvisible()
    line.MakeInvisible()

    def update(landing):
        if landing is None:
            end.MakeInvisible()
            line.MakeInvisible()
        else:
            x, y = landing["landing_position_m"][:2]
            end_pose.Set(Gf.Vec3d(x, y, 0.10))
            line.GetPointsAttr().Set([(0, 0, 0.12), (x, y, 0.12)])
            end.MakeVisible()
            line.MakeVisible()

    return update
