from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML not installed. Install it or switch to JSON config.") from exc

    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping/dict")
    return cfg


def meters_per_unit(stage) -> float:
    from pxr import UsdGeom

    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    return float(mpu) if mpu and float(mpu) > 0 else 0.01


def ensure_world_default_prim(stage) -> None:
    from pxr import UsdGeom

    root = stage.GetPrimAtPath("/World")
    if not root.IsValid():
        root = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    if not stage.GetDefaultPrim().IsValid():
        stage.SetDefaultPrim(root)


def ensure_physics_scene(stage, scene_path: str = "/World/PhysicsScene") -> str:
    from pxr import Gf, UsdPhysics
    try:
        from pxr import PhysxSchema
    except Exception:
        PhysxSchema = None

    scene_prim = stage.GetPrimAtPath(scene_path)
    if not scene_prim.IsValid():
        scene = UsdPhysics.Scene.Define(stage, scene_path)
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        if PhysxSchema is not None:
            try:
                physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
                physx_scene.CreateEnableContactReportAttr(True)
            except Exception:
                pass
    return scene_path


def create_invisible_ground_plane(
    stage,
    *,
    prim_path: str = "/World/__GroundPlane",
    size_m: Tuple[float, float] = (2000.0, 2000.0),
    thickness_m: float = 0.2,
    z_m: float = 0.0,
    invisible: bool = True,
) -> str:
    from pxr import Gf, UsdGeom, UsdPhysics

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


def _enable_ext(ext_name: str) -> bool:
    import omni.kit.app

    try:
        em = omni.kit.app.get_app().get_extension_manager()
        if em.is_extension_enabled(ext_name):
            return True
        em.set_extension_enabled_immediate(ext_name, True)
        return em.is_extension_enabled(ext_name)
    except Exception as exc:
        print(f"[ext] failed enabling {ext_name}: {exc}")
        return False


def ensure_viewport_window(simulation_app) -> bool:
    try:
        import omni.kit.viewport.utility as vutil
        vw = vutil.get_active_viewport_window()
        if vw is not None:
            return True

        try:
            import omni.kit.window.viewport as vpwin
            vpwin.ViewportWindow()
        except Exception:
            pass

        simulation_app.update()
        simulation_app.update()

        vw = vutil.get_active_viewport_window()
        return vw is not None
    except Exception as exc:
        print("[viewport] ensure_viewport_window failed:", exc)
        return False


def capture_active_viewport_png(filepath: str) -> bool:
    try:
        import omni.kit.viewport.utility as vutil
        vw = vutil.get_active_viewport_window()
        if vw is None:
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
        cap.capture_viewport_to_file(filepath)
        return True
    except Exception:
        return False


def create_dome_light(
    stage,
    path="/World/__DomeLight",
    intensity=3000.0,
    exposure=0.0,
    texture_file=None,
    rotation_deg_y=0.0,
):
    from pxr import UsdLux, UsdGeom

    dome = UsdLux.DomeLight.Define(stage, path)
    prim = dome.GetPrim()

    dome.CreateIntensityAttr(float(intensity))
    dome.CreateExposureAttr(float(exposure))

    if texture_file:
        dome.CreateTextureFileAttr(texture_file)

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    rot_op = xform.AddRotateYOp()
    rot_op.Set(float(rotation_deg_y))

    print(
        f"[light] created dome light {path} intensity={intensity} exposure={exposure} tex={texture_file}"
    )
    return prim


def _set_camera_world_matrix(cam_prim, M_world_units):
    from pxr import UsdGeom

    xf = UsdGeom.Xformable(cam_prim)
    ops = xf.GetOrderedXformOps()
    op = ops[0] if ops else xf.AddTransformOp()
    op.Set(M_world_units)


def apply_camera_pose_from_cfg(stage, cam_cfg: Dict[str, Any]) -> str:
    from pxr import Gf, UsdGeom

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


def compute_center_extent_from_world_roots(
    stage,
    *,
    root_container: str,
    world_count: int,
    bounds_size_m: float,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    from pxr import UsdGeom, Usd

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
        raise RuntimeError("No valid world roots found; cannot place camera.")

    center_m = (0.5 * (min_x + max_x), 0.5 * (min_y + max_y), 0.5 * (min_z + max_z))
    extent_m = (max_x - min_x, max_y - min_y, max_z - min_z)
    return center_m, extent_m


def ensure_camera_lookat(
    stage,
    *,
    cam_path: str,
    eye_m: Tuple[float, float, float],
    target_m: Tuple[float, float, float],
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    focal_length: float = 24.0,
    horiz_ap: float = 20.955,
    vert_ap: float = 15.2908,
) -> str:
    from pxr import Gf, UsdGeom

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


def place_camera_for_bbox(
    stage,
    cam_path: str,
    center_m: Tuple[float, float, float],
    extent_m: Tuple[float, float, float],
    *,
    tilt_deg: float = 35.0,
    yaw_deg: float = 225.0,
    margin: float = 1.25,
    look_at_z_offset_m: float = 0.0,
) -> None:
    import math
    from pxr import Gf, UsdGeom

    cx, cy, cz = center_m
    dx, dy, dz = extent_m
    span = max(dx, dy, dz, 1.0)

    dist = margin * span

    yaw = math.radians(yaw_deg)
    tilt = math.radians(tilt_deg)

    ex = cx + dist * math.cos(yaw) * math.cos(tilt)
    ey = cy + dist * math.sin(yaw) * math.cos(tilt)
    ez = cz + dist * math.sin(tilt) + 0.10 * span

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

    mpu = meters_per_unit(stage)
    cam_prim = stage.GetPrimAtPath(cam_path)
    cam = UsdGeom.Camera(cam_prim)
    cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.1, float((dist + 2.0 * span) / max(mpu, 1e-6))))


def force_set_active_camera(cam_path: str) -> None:
    try:
        import omni.kit.viewport.utility as vutil
        vw = vutil.get_active_viewport_window()
        if vw is None:
            return
        vp = vw.viewport_api
        try:
            vp.set_active_camera(cam_path)
        except Exception:
            try:
                vp.set_active_camera_path(cam_path)
            except Exception:
                return
    except Exception:
        return


class ChocoWorldBuilder:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.simulation_app = None
        self.sim = None
        self.stage = None

    def start(self):
        from isaacsim import SimulationApp

        app_cfg = self.cfg.get("app", {})
        self.simulation_app = SimulationApp(
            {
                "headless": bool(app_cfg.get("headless", True)),
                "renderer": str(app_cfg.get("renderer", "RayTracedLighting")),
            }
        )

        import omni.usd
        from isaacsim.core.api import SimulationContext

        if not _enable_ext("omni.physx.vehicle"):
            raise RuntimeError("Could not enable omni.physx.vehicle extension")

        self.simulation_app.update()
        self.simulation_app.update()

        from src.chocolate_waymo_builder import ChocolateBarConstructor, GridLayout
        from src.chocolate_vehicle_controller import ChocolateWorldVehicleController
        from src.chocolate_obs_builder import ChocolateObsBuilder

        usd_ctx = omni.usd.get_context()
        usd_ctx.new_stage()
        self.stage = usd_ctx.get_stage()
        ensure_world_default_prim(self.stage)
        ensure_physics_scene(self.stage, "/World/PhysicsScene")

        io_cfg = self.cfg["io"]
        scene_dir = Path(io_cfg["scene_json_dir"]).expanduser().resolve()
        if not scene_dir.exists():
            raise FileNotFoundError(f"scene_json_dir does not exist: {scene_dir}")

        all_jsons = sorted(scene_dir.glob("scene_*.json"))
        k = int(io_cfg.get("take_first_k_scenes", 10))
        json_paths = all_jsons[:k]
        if len(json_paths) < k:
            raise RuntimeError(
                f"Found only {len(json_paths)} scene_*.json files in {scene_dir}, wanted {k}"
            )

        wcfg = self.cfg["world"]
        layout = GridLayout(
            world_size_m=tuple(map(float, wcfg["world_size_m"])),
            padding_m=float(wcfg["padding_m"]),
            grid_cols=int(wcfg["grid_cols"]),
            base_z_m=float(wcfg["base_z_m"]),
        )

        ctor = ChocolateBarConstructor(
            stage=self.stage,
            root_container=str(wcfg["root_container"]),
            layout=layout,
            origin_mode=str(wcfg.get("origin_mode", "center")),
        )

        road = self.cfg.get("road", {})
        agents = self.cfg.get("agents", {})

        ctor.build(
            json_paths=json_paths,
            world_count=int(wcfg["world_count"]),
            bounds_size_m=float(wcfg["bounds_size_m"]),
            max_agents_per_world=int(wcfg["max_agents_per_world"]),
            jump_break_m=float(road.get("jump_break_m", 3.0)),
            seg_width=float(road.get("seg_width", 0.10)),
            seg_height=float(road.get("seg_height", 0.10)),
            z_lift=float(road.get("z_lift", 0.02)),
            flatten_road_z=bool(road.get("flatten_road", True)),
            road_z_m=float(road.get("road_z_m", 0.0)),
            polyline_reduction_area=float(road.get("polyline_reduction_area", 0.0)),
            min_points_for_reduction=int(road.get("min_points_for_reduction", 10)),
            enable_segment_collision=bool(road.get("enable_segment_collision", False)),
            trigger_enable=bool(road.get("trigger_enable", False)),
            trigger_height_m=float(road.get("trigger_height_m", 1.0)),
            trigger_width_scale=float(road.get("trigger_width_scale", 1.0)),
            trigger_offset_z_m=float(road.get("trigger_offset_z_m", 0.5)),
            trigger_match_segment=bool(road.get("trigger_match_segment", True)),
            trigger_script_enable=bool(road.get("trigger_script_enable", True)),
            spawn_z_m=float(agents.get("spawn_z_m", 1.0)),
            goal_radius_m=float(agents.get("goal_radius_m", 3.0)),
            parked_if_start_in_goal=bool(agents.get("parked_if_start_in_goal", True)),
            start_goal_thresh_m=float(agents.get("start_goal_thresh_m", 10.0)),
            parked_ground_z_m=float(agents.get("parked_ground_z_m", 0.0)),
            parked_chassis_size_m=tuple(
                map(float, agents.get("parked_chassis_size_m", [4.0, 2.0, 1.0]))
            ),
            parked_wheel_radius_m=float(agents.get("parked_wheel_radius_m", 0.35)),
            parked_wheel_thickness_m=float(
                agents.get("parked_wheel_thickness_m", 0.15)
            ),
            parked_wheel_inset_x_m=float(agents.get("parked_wheel_inset_x_m", 0.6)),
            parked_wheel_inset_y_m=float(agents.get("parked_wheel_inset_y_m", 0.05)),
            parked_ground_clearance_m=float(
                agents.get("parked_ground_clearance_m", 0.25)
            ),
            goal_ring_z_m=float(agents.get("goal_ring_z_m", 0.0)),
            goal_ring_tube_radius_m=float(
                agents.get("goal_ring_tube_radius_m", 0.12)
            ),
            goal_trigger_height_m=float(
                agents.get("goal_trigger_height_m", 0.6)
            ),
        )

        gcfg = self.cfg.get("ground", {})
        if bool(gcfg.get("enable", True)):
            create_invisible_ground_plane(
                self.stage,
                prim_path="/World/__GroundPlane",
                size_m=tuple(map(float, gcfg.get("size_m", [2000.0, 2000.0]))),
                thickness_m=float(gcfg.get("thickness_m", 0.2)),
                z_m=float(gcfg.get("z_m", 0.0)),
                invisible=bool(gcfg.get("invisible", True)),
            )

        light_cfg = self.cfg.get("light", {}) or {}
        if bool(light_cfg.get("enable", True)):
            create_dome_light(
                self.stage,
                path=str(light_cfg.get("path", "/World/__DomeLight")),
                intensity=float(light_cfg.get("intensity", 3000.0)),
                exposure=float(light_cfg.get("exposure", 0.0)),
                texture_file=light_cfg.get("texture_file", None),
                rotation_deg_y=float(light_cfg.get("rotation_deg_y", 0.0)),
            )

        cam_cfg = self.cfg.get("camera", {}) or {}
        cam_mode = str(cam_cfg.get("mode", "bbox"))
        cam_path = str(cam_cfg.get("prim_path", "/World/CaptureCam"))

        if cam_mode == "pose":
            cam_path = apply_camera_pose_from_cfg(self.stage, cam_cfg)
        else:
            center_m, extent_m = compute_center_extent_from_world_roots(
                self.stage,
                root_container=str(wcfg["root_container"]),
                world_count=int(wcfg["world_count"]),
                bounds_size_m=float(wcfg["bounds_size_m"]),
            )
            place_camera_for_bbox(
                self.stage,
                cam_path=cam_path,
                center_m=center_m,
                extent_m=extent_m,
                tilt_deg=float(cam_cfg.get("tilt_deg", 35.0)),
                yaw_deg=float(cam_cfg.get("yaw_deg", 225.0)),
                margin=float(cam_cfg.get("margin", 1.10)),
                look_at_z_offset_m=float(cam_cfg.get("look_at_z_offset_m", 0.0)),
            )

        ensure_viewport_window(self.simulation_app)
        force_set_active_camera(cam_path)

        self.simulation_app.update()
        self.simulation_app.update()

        phys_cfg = self.cfg.get("physics", {})
        physics_dt = float(phys_cfg.get("physics_dt", 1.0 / 60.0))
        render_dt = float(phys_cfg.get("rendering_dt", physics_dt))

        mpu = meters_per_unit(self.stage)
        self.sim = SimulationContext(
            stage_units_in_meters=mpu,
            physics_dt=physics_dt,
            rendering_dt=render_dt,
        )
        self.sim.initialize_physics()
        self.sim.step(render=False)

        ctrl_cfg = self.cfg.get("control", {})
        ctrl_suffix_candidates = ["", "/Vehicle", "/VehicleController"]
        ctrl = None
        for suf in ctrl_suffix_candidates:
            _ctrl = ChocolateWorldVehicleController(
                stage=self.stage,
                root_container=str(wcfg["root_container"]),
                world_count=int(wcfg["world_count"]),
                ctrl_suffix=suf,
                verbose=True,
            )
            _ctrl.refresh()
            if len(_ctrl.keys()) > 0:
                ctrl = _ctrl
                break

        if ctrl is None:
            raise RuntimeError("No controllable vehicles found; controller attrs not located.")

        obs_builder = ChocolateObsBuilder()

        return {
            "sim": self.sim,
            "stage": self.stage,
            "ctrl": ctrl,
            "obs_builder": obs_builder,
            "physics_dt": physics_dt,
            "action_repeat": int(ctrl_cfg.get("action_repeat", 4)),
        }

    def close(self) -> None:
        if self.simulation_app is not None:
            self.simulation_app.close()
            self.simulation_app = None

    def capture_frame(self, filepath: str) -> bool:
        if self.simulation_app is None:
            return False
        ensure_viewport_window(self.simulation_app)
        return capture_active_viewport_png(filepath)
