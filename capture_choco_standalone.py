#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np

# ----------------------------
# Config loader
# ----------------------------
def load_config(path: str) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    try:
        import yaml  # pip install pyyaml
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("YAML root must be a mapping/dict")
        return cfg
    except ImportError as e:
        raise RuntimeError("PyYAML not installed. Install it or switch to JSON config.") from e


# ----------------------------
# Isaac Sim bootstrap
# ----------------------------
from isaacsim import SimulationApp  # noqa: E402

CFG_PATH = "configs/capture_choco.yaml"
cfg = load_config(CFG_PATH)

app_cfg = cfg.get("app", {})
simulation_app = SimulationApp(
    {
        "headless": bool(app_cfg.get("headless", False)),
        "renderer": str(app_cfg.get("renderer", "RayTracedLighting")),
    }
)

# After SimulationApp is created, omni.* is importable.
import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

# Isaac Sim 5.x physics bootstrap (fixes "No USD stage attached")
from isaacsim.core.api import SimulationContext  # noqa: E402


# ----------------------------
# Enable required extensions
# ----------------------------
def _enable_ext(ext_name: str) -> bool:
    try:
        em = omni.kit.app.get_app().get_extension_manager()
        if em.is_extension_enabled(ext_name):
            return True
        em.set_extension_enabled_immediate(ext_name, True)
        return em.is_extension_enabled(ext_name)
    except Exception as e:
        print(f"[ext] failed enabling {ext_name}: {e}")
        return False


if not _enable_ext("omni.physx.vehicle"):
    raise RuntimeError("Could not enable omni.physx.vehicle extension")

# Let Kit settle / register python modules
simulation_app.update()
simulation_app.update()

# Import your builder AFTER enabling extensions
from src.chocolate_waymo_builder import ChocolateBarConstructor, GridLayout  # noqa: E402
from src.chocolate_vehicle_controller import ChocolateWorldVehicleController

# ----------------------------
# Helpers
# ----------------------------
def meters_per_unit(stage: Usd.Stage) -> float:
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    return float(mpu) if mpu and float(mpu) > 0 else 0.01


def ensure_world_default_prim(stage: Usd.Stage) -> None:
    root = stage.GetPrimAtPath("/World")
    if not root.IsValid():
        root = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    if not stage.GetDefaultPrim().IsValid():
        stage.SetDefaultPrim(root)


def ensure_physics_scene(stage: Usd.Stage, scene_path: str = "/World/PhysicsScene") -> str:
    """
    Create a UsdPhysics.Scene if missing (still good practice),
    but the important part is we will start physics via SimulationContext.
    """
    scene_prim = stage.GetPrimAtPath(scene_path)
    if not scene_prim.IsValid():
        scene = UsdPhysics.Scene.Define(stage, scene_path)
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        print("[phys] created physics scene:", scene_path)
    else:
        print("[phys] physics scene exists:", scene_path)
    return scene_path


def create_invisible_ground_plane(
    stage: Usd.Stage,
    *,
    prim_path: str = "/World/__GroundPlane",
    size_m: Tuple[float, float] = (2000.0, 2000.0),
    thickness_m: float = 0.2,
    z_m: float = 0.0,
    invisible: bool = True,
) -> str:
    mpu = meters_per_unit(stage)

    w_m, l_m = float(size_m[0]), float(size_m[1])
    h_m = float(thickness_m)

    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.GetSizeAttr().Set(1.0)

    api = UsdGeom.XformCommonAPI(cube)
    api.SetTranslate(Gf.Vec3d(0.0, 0.0, (z_m - 0.5 * h_m) / mpu))
    api.SetScale(Gf.Vec3f(w_m / mpu, l_m / mpu, h_m / mpu))

    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if invisible:
        UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()

    return prim_path

def ensure_viewport_window() -> bool:
    """Make sure a viewport window exists in standalone GUI runs."""
    try:
        import omni.kit.viewport.utility as vutil
        vw = vutil.get_active_viewport_window()
        if vw is not None:
            return True

        # Try to create one (works in many Isaac Sim builds)
        try:
            import omni.kit.window.viewport as vpwin
            vpwin.ViewportWindow()  # create a viewport window
        except Exception:
            pass

        # Give Kit a couple frames to actually spawn UI
        simulation_app.update()
        simulation_app.update()

        vw = vutil.get_active_viewport_window()
        return vw is not None
    except Exception as e:
        print("[viewport] ensure_viewport_window failed:", e)
        return False


def force_set_active_camera(cam_path: str) -> None:
    """Force active camera on the active viewport (debug prints included)."""
    import omni.kit.viewport.utility as vutil
    vw = vutil.get_active_viewport_window()
    if vw is None:
        print("[cam] No active viewport window (cannot set camera).")
        return
    vp = vw.viewport_api

    # Print what the viewport thinks before/after
    try:
        print("[cam] viewport active before:", vp.get_active_camera())
    except Exception:
        pass

    try:
        vp.set_active_camera(cam_path)  # some versions
    except Exception:
        try:
            vp.set_active_camera_path(cam_path)  # other versions
        except Exception as e:
            print("[cam] failed to set active camera:", e)
            return

    try:
        print("[cam] viewport active after:", vp.get_active_camera())
    except Exception:
        pass


def print_cam_pose(stage: Usd.Stage, cam_path: str, tag: str = "") -> None:
    prim = stage.GetPrimAtPath(cam_path)
    if not prim.IsValid():
        print(f"[cam] {tag} prim invalid:", cam_path)
        return
    xf = UsdGeom.Xformable(prim)
    M = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = M.ExtractTranslation()
    r = M.ExtractRotation()
    axis = r.GetAxis()
    ang = r.GetAngle()
    print(
        f"[cam] {tag} pos_units=({float(t[0]):.3f},{float(t[1]):.3f},{float(t[2]):.3f}) "
        f"axis=({float(axis[0]):.3f},{float(axis[1]):.3f},{float(axis[2]):.3f}) deg={float(ang):.3f}"
    )

def dump_active_viewport_camera_pose_yaml() -> None:
    import omni.kit.viewport.utility as vutil
    vw = vutil.get_active_viewport_window()
    if vw is None:
        print("[cam] No active viewport window.")
        return
    vp = vw.viewport_api
    cam_path = vp.get_active_camera()
    print("[cam] active:", cam_path)

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(cam_path)
    xf = UsdGeom.Xformable(prim)
    M = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = M.ExtractTranslation()
    r = M.ExtractRotation()

    axis = r.GetAxis()
    ang = r.GetAngle()

    print("camera:")
    print(f'  prim_path: "{cam_path}"')
    print("  pose:")
    print(f"    pos_units: [{float(t[0])}, {float(t[1])}, {float(t[2])}]")
    print(f"    rot_axis: [{float(axis[0])}, {float(axis[1])}, {float(axis[2])}]")
    print(f"    rot_deg: {float(ang)}")

def _set_camera_world_matrix(cam_prim: Usd.Prim, M_world_units: Gf.Matrix4d):
    xf = UsdGeom.Xformable(cam_prim)
    ops = xf.GetOrderedXformOps()
    op = ops[0] if ops else xf.AddTransformOp()
    op.Set(M_world_units)

def apply_camera_pose_from_cfg(stage: Usd.Stage, cam_cfg: Dict[str, Any]) -> str:
    cam_path = str(cam_cfg.get("prim_path", "/World/CaptureCam"))
    pose = cam_cfg.get("pose", {}) or {}

    pos = pose.get("pos_units", [0.0, 0.0, 100.0])
    axis = pose.get("rot_axis", [0.0, 0.0, 1.0])
    deg = float(pose.get("rot_deg", 0.0))

    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam.CreateFocalLengthAttr(float(cam_cfg.get("focal_length", 24.0)))
    cam.CreateHorizontalApertureAttr(float(cam_cfg.get("horiz_aperture", 20.955)))
    cam.CreateVerticalApertureAttr(float(cam_cfg.get("vert_aperture", 15.2908)))

    R = Gf.Rotation(Gf.Vec3d(*map(float, axis)), deg)
    M = Gf.Matrix4d(1.0)
    M.SetRotate(R)
    M.SetTranslateOnly(Gf.Vec3d(*map(float, pos)))

    _set_camera_world_matrix(cam.GetPrim(), M)
    return cam_path

def compute_grid_center_m(
    *,
    world_count: int,
    grid_cols: int,
    world_size_m: Tuple[float, float],
    padding_m: float,
) -> Tuple[float, float]:
    cols = int(grid_cols)
    sx, sy = float(world_size_m[0]), float(world_size_m[1])
    pitch_x = sx + float(padding_m)
    pitch_y = sy + float(padding_m)

    rows = int(math.ceil(float(world_count) / float(cols))) if world_count > 0 else 1

    min_x = 0.0 - 0.5 * sx
    max_x = (cols - 1) * pitch_x + 0.5 * sx
    min_y = 0.0 - 0.5 * sy
    max_y = (rows - 1) * pitch_y + 0.5 * sy

    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    return (cx, cy)

def compute_center_extent_from_world_roots(
    stage: Usd.Stage,
    *,
    root_container: str,
    world_count: int,
    bounds_size_m: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Robust camera framing without UsdGeom.BBoxCache (avoids native crash).
    Uses each world root's world-space translation plus known bounds_size_m.

    Returns (center_m, extent_m).
    """
    mpu = meters_per_unit(stage)
    half = 0.5 * float(bounds_size_m)

    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")

    found = 0
    for i in range(int(world_count)):
        wp = f"{root_container}/world_{i:03d}"
        prim = stage.GetPrimAtPath(wp)
        if not prim.IsValid():
            continue

        xf = UsdGeom.Xformable(prim)
        M = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t_u = M.ExtractTranslation()

        tx_m = float(t_u[0] * mpu)
        ty_m = float(t_u[1] * mpu)
        tz_m = float(t_u[2] * mpu)

        min_x = min(min_x, tx_m - half)
        max_x = max(max_x, tx_m + half)
        min_y = min(min_y, ty_m - half)
        max_y = max(max_y, ty_m + half)
        min_z = min(min_z, tz_m - half)
        max_z = max(max_z, tz_m + half)

        found += 1

    if found == 0:
        raise RuntimeError("No valid world roots found under root_container; cannot place camera.")

    center_m = (0.5 * (min_x + max_x), 0.5 * (min_y + max_y), 0.5 * (min_z + max_z))
    extent_m = (max_x - min_x, max_y - min_y, max_z - min_z)
    return center_m, extent_m


def compute_world_bbox_center_extent_m(stage: Usd.Stage, root_path: str) -> tuple[tuple[float,float,float], tuple[float,float,float]]:
    """
    Returns (center_m, extent_m) for everything under root_path using world-space bounds.
    extent_m is full size (dx,dy,dz) in meters.
    """
    print('[chk] Reached the compute world bbox method')
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage) or 0.01)

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"Root prim not found: {root_path}")

    bbox_cache = UsdGeom.BBoxCache(
        timeCode=Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    print('[chk]stored bbox cache')
    bbox = bbox_cache.ComputeWorldBound(root)
    box = bbox.ComputeAlignedBox()

    mn = box.GetMin()
    mx = box.GetMax()

    center_u = 0.5 * (mn + mx)
    extent_u = (mx - mn)

    center_m = (float(center_u[0] * mpu), float(center_u[1] * mpu), float(center_u[2] * mpu))
    extent_m = (float(extent_u[0] * mpu), float(extent_u[1] * mpu), float(extent_u[2] * mpu))
    return center_m, extent_m


def place_camera_for_bbox(
    stage: Usd.Stage,
    cam_path: str,
    center_m: tuple[float,float,float],
    extent_m: tuple[float,float,float],
    *,
    tilt_deg: float = 35.0,
    yaw_deg: float = 225.0,
    margin: float = 1.25,
    look_at_z_offset_m: float = 0.0,
) -> None:
    """
    Deterministic orbit camera around bbox center.
    yaw: rotates around Z. 225deg = looking from (-x,-y) toward center.
    tilt: pitch down from horizontal.
    """
    cx, cy, cz = center_m
    dx, dy, dz = extent_m
    span = max(dx, dy, dz, 1.0)

    # distance scales with span
    dist = margin * span

    yaw = math.radians(yaw_deg)
    tilt = math.radians(tilt_deg)

    # spherical-ish placement: back around Z with a tilt downward
    ex = cx + dist * math.cos(yaw) * math.cos(tilt)
    ey = cy + dist * math.sin(yaw) * math.cos(tilt)
    ez = cz + dist * math.sin(tilt) + 0.10 * span  # little extra up

    target = (cx, cy, cz + look_at_z_offset_m)

    ensure_camera_lookat(
        stage,
        cam_path=cam_path,
        eye_m=(ex, ey, ez),
        target_m=target,
        up=(0.0, 0.0, 1.0),
        focal_length=24.0,
        horiz_ap=20.955,
        vert_ap=15.2908,
    )

    # clip range generous
    mpu = meters_per_unit(stage)
    cam_prim = stage.GetPrimAtPath(cam_path)
    cam = UsdGeom.Camera(cam_prim)
    cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.1, float((dist + 2.0 * span) / max(mpu, 1e-6))))


def ensure_camera_lookat(
    stage: Usd.Stage,
    *,
    cam_path: str,
    eye_m: Tuple[float, float, float],
    target_m: Tuple[float, float, float],
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    focal_length: float = 24.0,
    horiz_ap: float = 20.955,
    vert_ap: float = 15.2908,
) -> str:
    mpu = meters_per_unit(stage)

    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam.CreateFocalLengthAttr(float(focal_length))
    cam.CreateHorizontalApertureAttr(float(horiz_ap))
    cam.CreateVerticalApertureAttr(float(vert_ap))

    eye_u = Gf.Vec3d(eye_m[0] / mpu, eye_m[1] / mpu, eye_m[2] / mpu)
    tgt_u = Gf.Vec3d(target_m[0] / mpu, target_m[1] / mpu, target_m[2] / mpu)

    M = Gf.Matrix4d(1.0)
    M.SetLookAt(eye_u, tgt_u, Gf.Vec3d(*up))

    prim = cam.GetPrim()
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    op = ops[0] if ops else xf.AddTransformOp()
    op.Set(M)

    return cam_path


def set_viewport_camera(cam_path: str) -> None:
    try:
        import omni.kit.viewport.utility as vutil
        vw = vutil.get_active_viewport_window()
        if vw is None:
            print("[cam] No active viewport window.")
            return
        vw.viewport_api.set_active_camera(cam_path)
        print("[cam] viewport camera set to", cam_path)
    except Exception as e:
        print("[cam] failed to set viewport camera:", e)


def capture_active_viewport_png(filepath: str) -> bool:
    """
    Best-effort viewport capture. Requires GUI (headless=False).
    """
    try:
        import omni.kit.viewport.utility as vutil
        vw = vutil.get_active_viewport_window()
        if vw is None:
            print('[cam] Viewport Window not found')
            return False
        vp = vw.viewport_api
        if hasattr(vutil, "capture_viewport_to_file_async"):
            vutil.capture_viewport_to_file_async(vp, filepath)
            return True
        if hasattr(vutil, "capture_viewport_to_file"):
            vutil.capture_viewport_to_file(vp, filepath)
            return True
        
    except Exception:
        pass

    try:
        import omni.kit.capture.viewport as cap
        print('capture with capture view interface')
        cap.capture_viewport_to_file(filepath)
        return True
    except Exception:
        pass

    return False

from pxr import UsdLux, UsdGeom, Gf

def create_dome_light(
    stage,
    path="/World/__DomeLight",
    intensity=3000.0,
    exposure=0.0,
    texture_file=None,   # e.g. "/path/to/studio.hdr" (optional)
    rotation_deg_y=0.0,  # rotate HDR around Y if you want
):
    dome = UsdLux.DomeLight.Define(stage, path)
    prim = dome.GetPrim()

    # Basic lighting controls
    dome.CreateIntensityAttr(float(intensity))
    dome.CreateExposureAttr(float(exposure))

    # Optional HDR environment map
    if texture_file:
        dome.CreateTextureFileAttr(texture_file)

    # Optional rotation (useful when texture_file is set)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    rot_op = xform.AddRotateYOp()
    rot_op.Set(float(rotation_deg_y))

    print(f"[light] created dome light {path} intensity={intensity} exposure={exposure} tex={texture_file}")
    return prim

# ----------------------------
# Main
# ----------------------------
def main() -> None:
    try:
        # --- stage ---
        usd_ctx = omni.usd.get_context()
        usd_ctx.new_stage()
        stage = usd_ctx.get_stage()
        ensure_world_default_prim(stage)

        # Create physics scene prim (good practice)
        ensure_physics_scene(stage, "/World/PhysicsScene")

        # --- choose scenes ---
        io_cfg = cfg["io"]
        scene_dir = Path(io_cfg["scene_json_dir"]).expanduser().resolve()
        if not scene_dir.exists():
            raise FileNotFoundError(f"scene_json_dir does not exist: {scene_dir}")

        all_jsons = sorted(scene_dir.glob("scene_*.json"))
        k = int(io_cfg.get("take_first_k_scenes", 10))
        json_paths = all_jsons[:k]
        if len(json_paths) < k:
            raise RuntimeError(f"Found only {len(json_paths)} scene_*.json files in {scene_dir}, wanted {k}")

        # --- build chocolate worlds ---
        wcfg = cfg["world"]
        layout = GridLayout(
            world_size_m=tuple(map(float, wcfg["world_size_m"])),
            padding_m=float(wcfg["padding_m"]),
            grid_cols=int(wcfg["grid_cols"]),
            base_z_m=float(wcfg["base_z_m"]),
        )

        ctor = ChocolateBarConstructor(
            stage=stage,
            root_container=str(wcfg["root_container"]),
            layout=layout,
            origin_mode=str(wcfg.get("origin_mode", "center")),
        )

        road = cfg.get("road", {})
        agents = cfg.get("agents", {})

        ctor.build(
            json_paths=json_paths,
            world_count=int(wcfg["world_count"]),
            bounds_size_m=float(wcfg["bounds_size_m"]),
            max_agents_per_world=int(wcfg["max_agents_per_world"]),
            # road
            jump_break_m=float(road.get("jump_break_m", 3.0)),
            seg_width=float(road.get("seg_width", 0.10)),
            seg_height=float(road.get("seg_height", 0.10)),
            z_lift=float(road.get("z_lift", 0.02)),
            flatten_road_z=bool(road.get("flatten_road", True)),
            road_z_m=float(road.get("road_z_m", 0.0)),
            polyline_reduction_area=float(road.get("polyline_reduction_area", 0.0)),
            min_points_for_reduction=int(road.get("min_points_for_reduction", 10)),
            # agents
            spawn_z_m=float(agents.get("spawn_z_m", 1.0)),
            goal_radius_m=float(agents.get("goal_radius_m", 3.0)),
            parked_if_start_in_goal=bool(agents.get("parked_if_start_in_goal", True)),
            start_goal_thresh_m=float(agents.get("start_goal_thresh_m", 10.0)),
            parked_ground_z_m=float(agents.get("parked_ground_z_m", 0.0)),
            parked_chassis_size_m=tuple(map(float, agents.get("parked_chassis_size_m", [4.0, 2.0, 1.0]))),
            parked_wheel_radius_m=float(agents.get("parked_wheel_radius_m", 0.35)),
            parked_wheel_thickness_m=float(agents.get("parked_wheel_thickness_m", 0.15)),
            parked_wheel_inset_x_m=float(agents.get("parked_wheel_inset_x_m", 0.6)),
            parked_wheel_inset_y_m=float(agents.get("parked_wheel_inset_y_m", 0.05)),
            parked_ground_clearance_m=float(agents.get("parked_ground_clearance_m", 0.25)),
            goal_ring_z_m=float(agents.get("goal_ring_z_m", 0.0)),
            goal_ring_tube_radius_m=float(agents.get("goal_ring_tube_radius_m", 0.12)),
            goal_trigger_height_m=float(agents.get("goal_trigger_height_m", 0.6)),
        )
        # --- light source ---
        create_dome_light(stage, "/World/__DomeLight", intensity=3000.0, exposure=0.0)
        # --- ground plane ---
        gcfg = cfg.get("ground", {})
        if bool(gcfg.get("enable", True)):
            create_invisible_ground_plane(
                stage,
                prim_path="/World/__GroundPlane",
                size_m=tuple(map(float, gcfg.get("size_m", [2000.0, 2000.0]))),
                thickness_m=float(gcfg.get("thickness_m", 0.2)),
                z_m=float(gcfg.get("z_m", 0.0)),
                invisible=bool(gcfg.get("invisible", True)),
            )
            print("[ground] created /World/__GroundPlane")
        print('[chk] Start creating camera')
        # --- camera at center (angled) ---
        # Compute bbox of the actual built content
        center_m, extent_m = compute_center_extent_from_world_roots(
            stage,
            root_container=str(wcfg["root_container"]),   # "/World/MiniWorlds"
            world_count=int(wcfg["world_count"]),
            bounds_size_m=float(wcfg["bounds_size_m"]),
        )
        print("[center/ext]", center_m, extent_m)

        # ---- viewport first ----
        ok_vp = ensure_viewport_window()
        print("[viewport] exists =", ok_vp)

        # ---- choose camera mode from config ----
        cam_cfg = cfg.get("camera", {}) or {}
        cam_mode = str(cam_cfg.get("mode", "bbox"))  # "bbox" or "pose"

        cam_path = str(cam_cfg.get("prim_path", "/World/CaptureCam"))

        if cam_mode == "pose":
            # Uses your YAML-provided axis-angle + position (stage units)
            cam_path = apply_camera_pose_from_cfg(stage, cam_cfg)
            print("[cam] applied pose from config:", cam_path)
        else:
            # Default: bbox/orbit camera
            center_m, extent_m = compute_center_extent_from_world_roots(
                stage,
                root_container=str(wcfg["root_container"]),
                world_count=int(wcfg["world_count"]),
                bounds_size_m=float(wcfg["bounds_size_m"]),
            )
            print("[center/ext]", center_m, extent_m)
            place_camera_for_bbox(
                stage,
                cam_path=cam_path,
                center_m=center_m,
                extent_m=extent_m,
                tilt_deg=float(cam_cfg.get("tilt_deg", 35.0)),
                yaw_deg=float(cam_cfg.get("yaw_deg", 225.0)),
                margin=float(cam_cfg.get("margin", 1.10)),
                look_at_z_offset_m=float(cam_cfg.get("look_at_z_offset_m", 0.0)),
            )
            print("[cam] placed bbox camera:", cam_path)

        # ---- force viewport to use it ----
        force_set_active_camera(cam_path)
        print_cam_pose(stage, cam_path, tag="after_set")

        # optional: huge clip so you don't get black from clipping
        try:
            cam = UsdGeom.Camera(stage.GetPrimAtPath(cam_path))
            cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.1, 100000.0))
        except Exception:
            pass

        print_cam_pose(stage, cam_path, tag="after_set")


        # Let USD/Kit settle so physics can attach cleanly
        simulation_app.update()
        simulation_app.update()

        # --- physics + stepping (THIS fixes "No USD stage attached") ---
        mpu = meters_per_unit(stage)
        phys_cfg = cfg.get("physics", {})

        physics_dt = float(phys_cfg.get("physics_dt", 1.0 / 60.0))
        render_dt = float(phys_cfg.get("rendering_dt", physics_dt))

        sim = SimulationContext(
            stage_units_in_meters=mpu,
            physics_dt=physics_dt,
            rendering_dt=render_dt,
        )
        sim.initialize_physics()
        # ---- vehicle controller registry (after physics init) ----
        # one small step helps ensure any late-bound controller prims are present
        sim.step(render=False)
        ctrl_cfg = cfg.get("control", {})
        ACTION_REPEAT = int(ctrl_cfg.get("action_repeat", 1))
        ACTION_REPEAT = max(1, ACTION_REPEAT)
        print(f"[control] action_repeat={ACTION_REPEAT}")
        ctrl_suffix_candidates = ["", "/Vehicle", "/VehicleController"]
        ctrl = None
        for suf in ctrl_suffix_candidates:
            _ctrl = ChocolateWorldVehicleController(
                stage=stage,
                root_container=str(wcfg["root_container"]),
                world_count=int(wcfg["world_count"]),
                ctrl_suffix=suf,
                verbose=True,
            )
            _ctrl.refresh()
            if len(_ctrl.keys()) > 0:
                print(f"[ChocoCtrl] using ctrl_suffix='{suf}'")
                ctrl = _ctrl
                break

        if ctrl is None:
            print("[ChocoCtrl] ERROR: found 0 controllable vehicles with any ctrl_suffix candidate.")
            # you can raise here if you want:
            # raise RuntimeError("No controllable vehicles found; controller attrs not located.")


        warmup_frames = int(phys_cfg.get("warmup_frames", 30))
        capture_frames = int(phys_cfg.get("capture_frames", 300))

        out_dir = Path(io_cfg["out_dir"]).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = str(io_cfg.get("image_prefix", "frame_"))
        ext = str(io_cfg.get("image_ext", "png"))

        print(f"[run] warmup={warmup_frames} frames, capture={capture_frames} frames, out={out_dir}")

        total = warmup_frames + capture_frames


        #==================================MAIN LOOP===================================
        #==================================MAIN LOOP===================================
        #==================================MAIN LOOP===================================
        #==================================MAIN LOOP===================================
        #==================================MAIN LOOP===================================
        #==================================MAIN LOOP===================================  
        for t in range(total):
            if ctrl is not None and ctrl.keys():
                if (t % ACTION_REPEAT) == 0 or U_last is None:
                    # compute / sample a new action here
                    Ks = ctrl.keys()
                    U_last = np.zeros((len(Ks), 3), np.float32)
                    U_last[:, 0] = 0.25  # throttle example
                    U_last[:, 1] = 0.0
                    U_last[:, 2] = 0.0
                    print(f"[ctrl] apply new action at t={t}")
                    ctrl.apply_all(U_last)
                else:
                    # optional: re-apply held action (usually not necessary; attrs persist)
                    pass
            # Step physics + render a frame
            sim.step(render=True)

            if t < warmup_frames:
                continue
            if (t % 10) == 0:
                print_cam_pose(stage, cam_path, tag=f"t={t}")
            cap_idx = t - warmup_frames
            out_path = out_dir / f"{prefix}{cap_idx:06d}.{ext}"
            # ok = capture_active_viewport_png(str(out_path))
            # if not ok:
            #     raise RuntimeError(
            #         "Viewport capture failed. "
            #         "Make sure app.headless=false and a viewport window exists."
            #     )

            # Optional tiny sleep so UI stays responsive
            # time.sleep(0.001)

        print(f"[done] saved {capture_frames} frames to {out_dir}")
        #==================================MAIN LOOP===================================
        #==================================MAIN LOOP===================================
        #==================================MAIN LOOP===================================
        #==================================MAIN LOOP===================================
        #==================================MAIN LOOP===================================
        #==================================MAIN LOOP=================================== 
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
