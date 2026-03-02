# chocolate_env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np

from pxr import Gf, Usd, UsdGeom, UsdPhysics, Sdf, Vt

from src.trfc import weather_context_dim


@dataclass
class StepInfo:
    keys: List[object]
    mask: np.ndarray          # (N,) bool
    dist_m: np.ndarray        # (N,) float32  (note: your obs builder currently returns DIST IN METERS, not normalized)
    success: np.ndarray       # (N,) bool     (latched)
    collided: np.ndarray      # (N,) bool
    off_road: np.ndarray      # (N,) bool
    timeout: bool
    t_env: int


def _yaw_from_xform(M: Gf.Matrix4d) -> float:
    """Yaw (rad) from world transform, using +X as forward (matches your obs builder)."""
    fwd = M.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
    fx, fy = float(fwd[0]), float(fwd[1])
    return math.atan2(fy, fx)


def _find_rigid_body_prim(start_prim: Usd.Prim) -> Optional[Usd.Prim]:
    """Same idea as your obs builder helper. :contentReference[oaicite:5]{index=5}"""
    p = start_prim
    while p and p.IsValid():
        try:
            rb = UsdPhysics.RigidBodyAPI(p)
            if rb.GetVelocityAttr().IsValid():
                return p
        except Exception:
            pass
        p = p.GetParent()
    return None


def _zero_rb_vel(rb_prim: Usd.Prim) -> None:
    rb = UsdPhysics.RigidBodyAPI(rb_prim)
    try:
        rb.GetVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    except Exception:
        pass
    try:
        rb.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    except Exception:
        pass


def _is_vehicle_trigger_prim(prim: Usd.Prim) -> bool:
    if prim is None or not prim.IsValid():
        return False
    try:
        cd = prim.GetCustomData()
    except Exception:
        cd = {}
    if isinstance(cd, dict) and bool(cd.get("vehicle_trigger", False)):
        return True
    return "/VehicleTrigger" in prim.GetPath().pathString


def _set_collision_enabled_recursive(root_prim: Usd.Prim, enabled: bool) -> None:
    if root_prim is None or not root_prim.IsValid():
        return
    for prim in Usd.PrimRange(root_prim):
        if _is_vehicle_trigger_prim(prim):
            continue
        try:
            api = UsdPhysics.CollisionAPI(prim)
            attr = api.GetCollisionEnabledAttr()
            if attr and attr.IsValid():
                attr.Set(bool(enabled))
        except Exception:
            continue


class ChocolateEnv:
    """
    Batched multi-agent RL-style environment on top of IsaacSim:
      - Works with your ChocolateWorldVehicleController and ChocolateObsBuilder.
      - Shared policy is natural: compute actions for obs[mask] and write into U[mask].

    Notes:
      - Your obs builder's obs[:,4] is *dist_to_goal_m* (meters), despite the 'dist_n' name in your older prints.
        We keep using obs[:,4] but treat it as meters.
      - We do NOT rely on builder deleting cars/goals (you already fixed that).
      - We support per-agent reset (teleport + zero velocity) without rebuilding the stage.
    """

    def __init__(
        self,
        *,
        sim,
        stage,
        ctrl,
        obs_builder,
        bounds_size_m: float,
        physics_dt: float,
        action_repeat: int = 4,
        max_steps: int = 600,
        clear_on_done: bool = False,
        goal_success_dist_m: float = 2.0,   # SUCCESS when dist_to_goal_m <= this
        reward_scale: float = 1.0,
        success_bonus: float = 10.0,
        action_l2_penalty: float = 0.0,
        collision_penalty: float = 0.0,
        min_vehicle_z_m: Optional[float] = None,
        collision_penalty_types: Optional[List[int]] = None,
        collision_debug: bool = False,
        road_contact_done_types: Optional[List[int]] = None,
        road_contact_done_penalty: float = -1.0,
        lane_center_reward_enable: bool = False,
        lane_center_reward_type: int = 2,
        lane_center_reward_per_step: float = 0.05,
        survival_reward_per_step: float = 0.0,
        idle_penalty_enable: bool = False,
        idle_penalty_per_step: float = 0.05,
        idle_speed_threshold_mps: float = 0.5,
        vehicle_contact_done: bool = False,
        vehicle_contact_done_penalty: float = -5.0,
        vehicle_contact_done_mark_both: bool = True,
        road_contact_debug: bool = False,
        road_contact_debug_every: int = 100,
        road_points_enable: bool = False,
        road_points_k: int = 16,
        road_points_radius_m: float = 50.0,
        road_points_type_norm: float = 1.0,
        road_points_mode: str = "knn",
        vehicle_obs_enable: bool = False,
        vehicle_obs_k: int = 63,
        ttc_penalty_enable: bool = False,
        ttc_penalty_alpha: float = 1.0,
        ttc_penalty_max: float = 1.0,
        ttc_penalty_min_ttc: float = 0.2,
        obs_viz_enable: bool = False,
        obs_viz_world_idx: int = 0,
        obs_viz_agent_rank: int = 0,
        render: bool = False,
        root_container: str = "/World/MiniWorlds",
        world_prefix: str = "world_",
        warmup_on_reset_steps: int = 1,
        respawn_on_reset: bool = False,
        respawn_params: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ):
        self.sim = sim
        self.stage = stage
        self.ctrl = ctrl
        self.obs_builder = obs_builder
        print('in constructor now')
        self.bounds_size_m = float(bounds_size_m)
        self.physics_dt = float(physics_dt)
        self.action_repeat = max(1, int(action_repeat))
        self.max_steps = int(max_steps)

        self.clear_on_done = bool(clear_on_done)
        self.goal_success_dist_m = float(goal_success_dist_m)

        self.reward_scale = float(reward_scale)
        self.success_bonus = float(success_bonus)
        self.action_l2_penalty = float(action_l2_penalty)
        self.collision_penalty = float(collision_penalty)
        self.min_vehicle_z_m = None if min_vehicle_z_m is None else float(min_vehicle_z_m)
        self.collision_penalty_types = set(int(x) for x in (collision_penalty_types or []))
        self.collision_debug = bool(collision_debug)
        self.road_contact_done_types = set(int(x) for x in (road_contact_done_types or []))
        self.road_contact_done_penalty = float(road_contact_done_penalty)
        self.lane_center_reward_enable = bool(lane_center_reward_enable)
        if isinstance(lane_center_reward_type, (list, tuple, set)):
            self.lane_center_reward_types = set(int(x) for x in lane_center_reward_type)
        else:
            self.lane_center_reward_types = {int(lane_center_reward_type)}
        self.lane_center_reward_per_step = float(lane_center_reward_per_step)
        self.survival_reward_per_step = float(survival_reward_per_step)
        self.idle_penalty_enable = bool(idle_penalty_enable)
        self.idle_penalty_per_step = float(idle_penalty_per_step)
        self.idle_speed_threshold_mps = float(idle_speed_threshold_mps)
        self.vehicle_contact_done = bool(vehicle_contact_done)
        self.vehicle_contact_done_penalty = float(vehicle_contact_done_penalty)
        self.vehicle_contact_done_mark_both = bool(vehicle_contact_done_mark_both)
        self.road_contact_debug = bool(road_contact_debug)
        self.road_contact_debug_every = max(1, int(road_contact_debug_every))
        self.road_points_enable = bool(road_points_enable)
        self.road_points_k = int(road_points_k)
        self.road_points_radius_m = float(road_points_radius_m)
        self.road_points_type_norm = float(road_points_type_norm)
        self.road_points_mode = str(road_points_mode)
        self.vehicle_obs_enable = bool(vehicle_obs_enable)
        self.vehicle_obs_k = int(vehicle_obs_k)
        self.ttc_penalty_enable = bool(ttc_penalty_enable)
        self.ttc_penalty_alpha = float(ttc_penalty_alpha)
        self.ttc_penalty_max = float(ttc_penalty_max)
        self.ttc_penalty_min_ttc = float(ttc_penalty_min_ttc)
        self.obs_viz_enable = bool(obs_viz_enable)
        self.obs_viz_world_idx = int(obs_viz_world_idx)
        self.obs_viz_agent_rank = int(obs_viz_agent_rank)

        self.render = bool(render)
        self.root_container = str(root_container)
        self.world_prefix = str(world_prefix)
        self.warmup_on_reset_steps = max(0, int(warmup_on_reset_steps))
        self.respawn_on_reset = bool(respawn_on_reset)
        self.respawn_params = respawn_params or {}
        self.respawn_hold_radius_m = float(self.respawn_params.get("respawn_hold_radius_m", 3.0))
        self.verbose = bool(verbose)

        # --- episode state (per "row" = per AgentKey) ---
        self.t = 0
        self._keys: List[object] = []
        self._mask: np.ndarray = np.zeros((0,), dtype=bool)

        self._prev_dist_m: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._done: np.ndarray = np.zeros((0,), dtype=bool)
        self._success_latched: np.ndarray = np.zeros((0,), dtype=bool)

        # --- per-agent cached reset pose ---
        self._start_local_translate: Dict[object, Tuple[float, float, float]] = {}
        self._start_local_yaw_deg: Dict[object, float] = {}
        self._start_world_xy_m: Dict[object, Tuple[float, float]] = {}
        self._spawn_pos_units: dict = {}   # key -> (x_u, y_u, z_u)
        self._spawn_quat: dict = {}        # key -> (w, x, y, z)  (world orientation)
        self._pending_respawns: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._mpu = float(getattr(self.sim, "meters_per_unit", 1.0))  # IsaacSim usually has this
        self._collision_tracker = None
        if self.collision_penalty_types or self.collision_debug:
            self._collision_tracker = _RoadCollisionTracker(self.stage, self.ctrl, self.collision_penalty_types)
        self._collision_debug_printed = False


    # -------------------------
    # Internal helpers
    # -------------------------




    def _find_rb_prim(self, pose_prim):
        # walk UP until we find a prim with UsdPhysics.RigidBodyAPI velocity attr
        p = pose_prim
        while p and p.IsValid():
            try:
                rb = UsdPhysics.RigidBodyAPI(p)
                if rb.GetVelocityAttr().IsValid():
                    return p
            except Exception:
                pass
            p = p.GetParent()
        return None

    def _get_rb_linear_velocity_world(self, rb_prim: Usd.Prim) -> Tuple[float, float, float]:
        rb = UsdPhysics.RigidBodyAPI(rb_prim)
        v_attr = rb.GetVelocityAttr()
        if not v_attr or not v_attr.IsValid():
            return 0.0, 0.0, 0.0
        v = v_attr.Get(Usd.TimeCode.Default())
        if v is None:
            return 0.0, 0.0, 0.0
        return float(v[0]), float(v[1]), float(v[2])

    def _collect_all_vehicle_states(self):
        world_count = int(getattr(self.ctrl, "world_count", 0))
        out = {}
        tc = Usd.TimeCode.Default()
        for wi in range(world_count):
            agents_root = f"{self.root_container}/{self.world_prefix}{wi:03d}/Agents"
            root = self.stage.GetPrimAtPath(agents_root)
            if not root.IsValid():
                continue
            states = []
            for agent in root.GetAllChildren():
                # Find a descendant with 'controllable' customData (vehicle root)
                stack = [agent]
                vehicle_prim = None
                while stack:
                    p = stack.pop()
                    try:
                        cd = p.GetCustomData()
                    except Exception:
                        cd = {}
                    if isinstance(cd, dict) and "controllable" in cd and "agent_id" in cd:
                        vehicle_prim = p
                        break
                    stack.extend(p.GetAllChildren())
                if vehicle_prim is None:
                    continue
                try:
                    cd = vehicle_prim.GetCustomData()
                except Exception:
                    cd = {}
                if not isinstance(cd, dict):
                    continue
                agent_id = cd.get("agent_id", None)
                if agent_id is None:
                    continue
                controllable = bool(cd.get("controllable", False))

                xform = UsdGeom.Xformable(vehicle_prim)
                try:
                    M = xform.ComputeLocalToWorldTransform(tc)
                    p = M.ExtractTranslation()
                    px, py = float(p[0]), float(p[1])
                except Exception:
                    continue

                vx, vy = 0.0, 0.0
                if controllable:
                    h = self.ctrl.get(wi, int(agent_id))
                    if h is not None and h.pose_prim:
                        rb_prim = self._find_rb_prim(h.pose_prim)
                        if rb_prim is not None:
                            vx, vy, _ = self._get_rb_linear_velocity_world(rb_prim)

                states.append(
                    {
                        "agent_id": int(agent_id),
                        "pos": (px, py),
                        "vel": (vx, vy),
                        "controllable": controllable,
                    }
                )
            if states:
                out[wi] = states
        return out

    def _compute_ttc_penalty(self, keys: List[object], active: np.ndarray) -> np.ndarray:
        if not self.ttc_penalty_enable:
            return np.zeros((len(keys),), dtype=np.float32)

        states_by_world = self._collect_all_vehicle_states()
        penalties = np.zeros((len(keys),), dtype=np.float32)

        for i, k in enumerate(keys):
            if not active[i]:
                continue
            wi = int(getattr(k, "world_idx", -1))
            if wi not in states_by_world:
                continue
            ego_state = None
            for s in states_by_world[wi]:
                if s["agent_id"] == int(k.agent_id):
                    ego_state = s
                    break
            if ego_state is None:
                continue
            ex, ey = ego_state["pos"]
            evx, evy = ego_state["vel"]
            min_ttc = None
            for s in states_by_world[wi]:
                if s["agent_id"] == int(k.agent_id):
                    continue
                ox, oy = s["pos"]
                ovx, ovy = s["vel"]
                rx = ox - ex
                ry = oy - ey
                rvx = ovx - evx
                rvy = ovy - evy
                v2 = rvx * rvx + rvy * rvy
                if v2 < 1e-6:
                    continue
                rdotv = rx * rvx + ry * rvy
                if rdotv >= 0.0:
                    continue
                ttc = -rdotv / v2
                if ttc <= 0.0:
                    continue
                if min_ttc is None or ttc < min_ttc:
                    min_ttc = ttc
            if min_ttc is None:
                continue
            denom = max(float(min_ttc), float(self.ttc_penalty_min_ttc))
            penalty = float(self.ttc_penalty_alpha) / denom
            penalty = min(float(self.ttc_penalty_max), penalty)
            penalties[i] = -float(penalty)
        return penalties
    def _cache_spawn_if_missing(self):
        # cache LOCAL pose from the stable Vehicle_Parent xform
        for k in self._keys:
            if k in self._start_local_translate:
                continue
            h = self.ctrl.get(k.world_idx, k.agent_id)
            if h is None:
                continue
            self._cache_spawn_pose(k, h)


    def _physx_teleport_rb(self, rb_prim, pos_units, quat_wxyz):
        """
        Teleport a PhysX rigid body using PhysX interface (not USD xform ops).
        pos_units: (x,y,z) in stage units
        quat_wxyz: (w,x,y,z) world quaternion
        """
        import omni.physx
        from pxr import Gf

        rb_path = rb_prim.GetPath().pathString
        print('hererer')
        # PhysX sim interface name differs slightly across Isaac Sim builds,
        # so we try common variants.
        sim_iface = None
        if hasattr(omni.physx, "get_physx_simulation_interface"):
            sim_iface = omni.physx.get_physx_simulation_interface()
        elif hasattr(omni.physx, "get_physx_interface"):
            sim_iface = omni.physx.get_physx_interface()
        if sim_iface is None:
            print('no omni??')
            raise RuntimeError("No omni.physx simulation interface found.")
        print('passed?')
        px, py, pz = pos_units
        w, x, y, z = quat_wxyz

        # build Gf types (float versions)
        p = Gf.Vec3f(float(px), float(py), float(pz))
        q = Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))
        print('passed?')
        # --- pose setters (try a few common names) ---
        pose_setters = [
            "set_rigid_body_pose",
            "setRigidBodyPose",
            "set_rigid_body_global_pose",
            "setRigidBodyGlobalPose",
        ]
        ok = False
        for fn in pose_setters:
            if hasattr(sim_iface, fn):
                getattr(sim_iface, fn)(rb_path, p, q)
                ok = True
                break
        if not ok:
            # help you debug quickly
            print('not OK')
            cand = [m for m in dir(sim_iface) if ("rigid" in m.lower() and "pose" in m.lower())]
            raise RuntimeError(f"Couldn't find pose setter on physx iface. Candidates: {cand[:30]}")
        print('passed? WOW')
        # --- zero velocities (again try common names) ---
        vel_fns = [
            ("set_rigid_body_linear_velocity", "setRigidBodyLinearVelocity"),
            ("set_rigid_body_angular_velocity", "setRigidBodyAngularVelocity"),
        ]
        for a, b in vel_fns:
            if hasattr(sim_iface, a):
                getattr(sim_iface, a)(rb_path, Gf.Vec3f(0.0, 0.0, 0.0))
            elif hasattr(sim_iface, b):
                getattr(sim_iface, b)(rb_path, Gf.Vec3f(0.0, 0.0, 0.0))


    def _build_obs(self) -> Tuple[np.ndarray, np.ndarray, List[object]]:
        obs, mask, keys = self.obs_builder.build_obs_all_controlled(
            stage=self.stage,
            bounds_size_m=self.bounds_size_m,
            ctrl=self.ctrl,
            dt=self.physics_dt,
            root_container=self.root_container,
            world_prefix=self.world_prefix,
            road_points_enable=self.road_points_enable,
            road_points_k=self.road_points_k,
            road_points_radius_m=self.road_points_radius_m,
            road_points_type_norm=self.road_points_type_norm,
            road_points_mode=self.road_points_mode,
            vehicle_obs_enable=self.vehicle_obs_enable,
            vehicle_obs_k=self.vehicle_obs_k,
        )
        return obs, mask, keys

    def _debug_viz_obs_first_agent(self, obs: np.ndarray, keys: List[object]) -> None:
        # Visualize a selected agent using its observation.
        stage = self.stage
        root = "/World/ObsViz"
        if not stage.GetPrimAtPath(root).IsValid():
            UsdGeom.Xform.Define(stage, root)

        target_idx = None
        target_key = None
        rank = 0
        for i, k in enumerate(keys):
            if int(getattr(k, "world_idx", -1)) != self.obs_viz_world_idx:
                continue
            if rank == self.obs_viz_agent_rank:
                target_idx = i
                target_key = k
                break
            rank += 1
        if target_idx is None or target_key is None:
            return

        h = self.ctrl.get(target_key.world_idx, target_key.agent_id)
        if h is None:
            return

        try:
            start_prim = h.xform.GetPrim() if hasattr(h.xform, "GetPrim") else None
            if start_prim is None:
                start_prim = h.pose_prim if hasattr(h, "pose_prim") else None
            rb_prim = self._find_rb_prim(start_prim) if start_prim is not None else None
            if rb_prim is not None:
                M = UsdGeom.Xformable(rb_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            else:
                M = h.xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            p = M.ExtractTranslation()
            yaw = _yaw_from_xform(M)
        except Exception:
            return

        agent_root = f"{root}/W000_A{int(target_key.agent_id)}"
        if stage.GetPrimAtPath(agent_root).IsValid():
            stage.RemovePrim(agent_root)
        mpu = self._mpu
        agent_xf = UsdGeom.Xform.Define(stage, agent_root)
        xapi = UsdGeom.XformCommonAPI(agent_xf)
        z_lift_m = 2.5
        xapi.SetTranslate(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]) + z_lift_m / mpu))
        xapi.SetRotate(Gf.Vec3f(0.0, 0.0, float(math.degrees(yaw))), UsdGeom.XformCommonAPI.RotationOrderXYZ)

        mpu = self._mpu
        obs_vec = obs[target_idx]

        # Ego marker
        ego_cube = UsdGeom.Cube.Define(stage, f"{agent_root}/Ego")
        ego_cube.GetSizeAttr().Set(1.0)
        ego_x = UsdGeom.XformCommonAPI(ego_cube)
        ego_x.SetScale(Gf.Vec3f(1.2 / mpu, 0.6 / mpu, 0.3 / mpu))
        UsdGeom.Gprim(ego_cube.GetPrim()).CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.9, 0.2)])

        # Forward arrow
        arrow = UsdGeom.Cube.Define(stage, f"{agent_root}/Forward")
        arrow.GetSizeAttr().Set(1.0)
        arrow_x = UsdGeom.XformCommonAPI(arrow)
        arrow_x.SetTranslate(Gf.Vec3d(2.5 / mpu, 0.0, 0.0))
        arrow_x.SetScale(Gf.Vec3f(5.0 / mpu, 0.25 / mpu, 0.25 / mpu))
        UsdGeom.Gprim(arrow.GetPrim()).CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.3, 0.3)])

        # Goal marker (rel goal in ego frame)
        L = float(self.bounds_size_m)
        goal_x = float(obs_vec[0]) * L
        goal_y = float(obs_vec[1]) * L
        goal = UsdGeom.Sphere.Define(stage, f"{agent_root}/Goal")
        goal.GetRadiusAttr().Set(float(0.8 / mpu))
        goal_xf = UsdGeom.XformCommonAPI(goal)
        goal_xf.SetTranslate(Gf.Vec3d(goal_x / mpu, goal_y / mpu, 0.3 / mpu))
        UsdGeom.Gprim(goal.GetPrim()).CreateDisplayColorAttr().Set([Gf.Vec3f(0.2, 1.0, 0.2)])

        # Road points visualization
        if self.road_points_enable:
            rp_root = UsdGeom.Xform.Define(stage, f"{agent_root}/RoadPoints")
            base = 7 + weather_context_dim()
            k = int(self.road_points_k)
            radius = float(self.road_points_radius_m)
            for j in range(k):
                off = base + 3 * j
                rx = float(obs_vec[off + 0]) * radius
                ry = float(obs_vec[off + 1]) * radius
                t = float(obs_vec[off + 2])
                if rx == 0.0 and ry == 0.0 and t == 0.0:
                    continue
                s = UsdGeom.Sphere.Define(stage, f"{agent_root}/RoadPoints/P{j:03d}")
                s.GetRadiusAttr().Set(float(0.25 / mpu))
                sx = UsdGeom.XformCommonAPI(s)
                sx.SetTranslate(Gf.Vec3d(rx / mpu, ry / mpu, 0.2 / mpu))
                color = Gf.Vec3f(0.2, 0.4 + 0.6 * max(0.0, min(1.0, t)), 1.0)
                UsdGeom.Gprim(s.GetPrim()).CreateDisplayColorAttr().Set([color])

        # Vehicle observations visualization
        if self.vehicle_obs_enable:
            veh_root = UsdGeom.Xform.Define(stage, f"{agent_root}/Vehicles")
            base = 7 + weather_context_dim() + (
                int(self.road_points_k) * 3 if self.road_points_enable else 0
            )
            k = int(self.vehicle_obs_k)
            feat = 6
            for j in range(k):
                off = base + feat * j
                rx = float(obs_vec[off + 0]) * L
                ry = float(obs_vec[off + 1]) * L
                length = float(obs_vec[off + 2]) * L
                width = float(obs_vec[off + 3]) * L
                rel_yaw = float(obs_vec[off + 4]) * math.pi
                speed = float(obs_vec[off + 5])
                if rx == 0.0 and ry == 0.0 and length == 0.0 and width == 0.0:
                    continue
                c = UsdGeom.Cube.Define(stage, f"{agent_root}/Vehicles/V{j:03d}")
                c.GetSizeAttr().Set(1.0)
                cx = UsdGeom.XformCommonAPI(c)
                cx.SetTranslate(Gf.Vec3d(rx / mpu, ry / mpu, 0.1 / mpu))
                cx.SetRotate(Gf.Vec3f(0.0, 0.0, float(math.degrees(rel_yaw))), UsdGeom.XformCommonAPI.RotationOrderXYZ)
                cx.SetScale(Gf.Vec3f(max(0.6, length) / mpu, max(0.4, width) / mpu, 0.3 / mpu))
                sp = max(0.0, min(1.0, speed))
                color = Gf.Vec3f(1.0 - sp, 0.2 + 0.8 * sp, 0.3)
                UsdGeom.Gprim(c.GetPrim()).CreateDisplayColorAttr().Set([color])

    def _cache_start_pose_for_keys(self, keys: List[object]) -> None:
        """
        Cache local translate + yaw (deg) from the stable Vehicle_Parent xform.
        """
        for k in keys:
            if k in self._start_local_translate:
                continue
            h = self.ctrl.get(k.world_idx, k.agent_id)
            if h is None:
                continue
            self._cache_spawn_pose(k, h)

    def _agent_token(self, world_idx: int, agent_id: int) -> Tuple[int, int]:
        return (int(world_idx), int(agent_id))

    def _agent_token_from_key(self, key: object) -> Tuple[int, int]:
        return self._agent_token(getattr(key, "world_idx"), getattr(key, "agent_id"))

    def _get_agent_handle(self, world_idx: int, agent_id: int):
        return self.ctrl.get(int(world_idx), int(agent_id))

    def _get_agent_root_prim(self, h) -> Optional[Usd.Prim]:
        if h is None:
            return None
        root_path = getattr(h, "vehicle_root_path", None)
        if isinstance(root_path, str):
            prim = self.stage.GetPrimAtPath(root_path)
            if prim.IsValid():
                return prim
        pose_prim = getattr(h, "pose_prim", None)
        if pose_prim is not None and pose_prim.IsValid():
            return pose_prim
        return None

    def _get_spawn_xform_prim(self, h) -> Optional[Usd.Prim]:
        if h is None:
            return None
        root_path = getattr(h, "vehicle_root_path", None)
        if isinstance(root_path, str):
            root_prim = self.stage.GetPrimAtPath(root_path)
            if root_prim.IsValid():
                parent = root_prim.GetParent()
                if parent is not None and parent.IsValid():
                    return parent
                return root_prim
        pose_prim = getattr(h, "pose_prim", None)
        if pose_prim is not None and pose_prim.IsValid():
            parent = pose_prim.GetParent()
            if parent is not None and parent.IsValid():
                return parent
            return pose_prim
        return None

    def _cache_spawn_pose(self, key: object, h) -> bool:
        if key in self._start_local_translate:
            return True
        spawn_prim = self._get_spawn_xform_prim(h)
        if spawn_prim is None or not spawn_prim.IsValid():
            return False
        try:
            capi = UsdGeom.XformCommonAPI(spawn_prim)
            t = capi.GetTranslate()
            r = capi.GetRotate()
            self._start_local_translate[key] = (float(t[0]), float(t[1]), float(t[2]))
            self._start_local_yaw_deg[key] = float(r[2])
        except Exception:
            return False
        try:
            M = UsdGeom.Xformable(spawn_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            p = M.ExtractTranslation()
            self._start_world_xy_m[key] = (float(p[0]) * self._mpu, float(p[1]) * self._mpu)
        except Exception:
            pass
        return True

    def _set_agent_visible(self, h, visible: bool) -> None:
        if h is None:
            return
        try:
            prim = self._get_spawn_xform_prim(h)
            if prim is None or not prim.IsValid():
                prim = self._get_agent_root_prim(h)
            if prim is None or not prim.IsValid():
                return
            imageable = UsdGeom.Imageable(prim)
            if visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        except Exception:
            pass

    def _set_agent_collision_enabled(self, h, enabled: bool) -> None:
        root_prim = self._get_agent_root_prim(h)
        if root_prim is None or not root_prim.IsValid():
            return
        _set_collision_enabled_recursive(root_prim, enabled)

    def _reset_agent_contact_state(self, h) -> None:
        root_prim = self._get_agent_root_prim(h)
        if root_prim is None or not root_prim.IsValid():
            return
        try:
            root_prim.SetCustomDataByKey("road_contact_types", Vt.IntArray())
            root_prim.SetCustomDataByKey("vehicle_contact_ids", Vt.IntArray())
            root_prim.SetCustomDataByKey("vehicle_collided", False)
        except Exception:
            pass

    def _set_agent_local_pose(self, h, translate: Tuple[float, float, float], yaw_deg: float) -> bool:
        spawn_prim = self._get_spawn_xform_prim(h)
        if spawn_prim is None or not spawn_prim.IsValid():
            return False
        try:
            capi = UsdGeom.XformCommonAPI(spawn_prim)
            capi.SetTranslate(Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2])))
            capi.SetRotate(
                Gf.Vec3f(0.0, 0.0, float(yaw_deg)),
                UsdGeom.XformCommonAPI.RotationOrderXYZ,
            )
        except Exception:
            return False

        pose_prim = getattr(h, "pose_prim", None)
        if pose_prim is not None and pose_prim.IsValid():
            rb_prim = _find_rigid_body_prim(pose_prim)
            if rb_prim is not None:
                _zero_rb_vel(rb_prim)
        return True

    def _get_agent_world_xy_m(self, h) -> Optional[Tuple[float, float]]:
        if h is None:
            return None
        try:
            start_prim = h.xform.GetPrim() if hasattr(h.xform, "GetPrim") else None
            if start_prim is None:
                start_prim = h.pose_prim if hasattr(h, "pose_prim") else None
            if start_prim is None:
                return None
            rb_prim = _find_rigid_body_prim(start_prim)
            if rb_prim is not None:
                M = UsdGeom.Xformable(rb_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            else:
                M = h.xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            p = M.ExtractTranslation()
            return (float(p[0]) * self._mpu, float(p[1]) * self._mpu)
        except Exception:
            return None

    def _teleport_agent_to_spawn(self, key: object) -> bool:
        self._cache_spawn_if_missing()
        h = self.ctrl.get(key.world_idx, key.agent_id)
        if h is None:
            return False

        t0 = self._start_local_translate.get(key, None)
        yaw0 = self._start_local_yaw_deg.get(key, 0.0)
        if t0 is None:
            return False
        return self._set_agent_local_pose(h, t0, yaw0)

    def _move_agent_to_respawn_hold(self, key: object) -> bool:
        self._cache_spawn_if_missing()
        h = self.ctrl.get(key.world_idx, key.agent_id)
        if h is None:
            return False

        t0 = self._start_local_translate.get(key, None)
        yaw0 = self._start_local_yaw_deg.get(key, 0.0)
        if t0 is None:
            return False

        hold_depth_units = 50.0 / max(self._mpu, 1e-6)
        hold_translate = (float(t0[0]), float(t0[1]), float(t0[2] - hold_depth_units))
        return self._set_agent_local_pose(h, hold_translate, yaw0)

    def _spawn_area_is_clear(self, token: Tuple[int, int], radius_m: float) -> bool:
        if radius_m <= 0.0:
            return True

        pending = self._pending_respawns.get(token, {})
        spawn_xy = pending.get("spawn_xy_m", None)
        if spawn_xy is None:
            for key, cached_xy in self._start_world_xy_m.items():
                if self._agent_token_from_key(key) == token:
                    spawn_xy = cached_xy
                    break
        if spawn_xy is None:
            return False

        world_idx, agent_id = token
        radius2 = float(radius_m) * float(radius_m)
        states = self._collect_all_vehicle_states().get(int(world_idx), [])
        for state in states:
            other_agent_id = int(state.get("agent_id", -1))
            other_token = self._agent_token(world_idx, other_agent_id)
            if other_token == token or other_token in self._pending_respawns:
                continue
            pos = state.get("pos", None)
            if pos is None or len(pos) < 2:
                continue
            other_x = float(pos[0]) * self._mpu
            other_y = float(pos[1]) * self._mpu
            dx = float(other_x - spawn_xy[0])
            dy = float(other_y - spawn_xy[1])
            if dx * dx + dy * dy < radius2:
                return False
        return True

    def _queue_respawn(self, key: object) -> bool:
        self._cache_spawn_if_missing()
        token = self._agent_token_from_key(key)
        h = self.ctrl.get(key.world_idx, key.agent_id)
        if h is None:
            return False
        spawn_xy = self._start_world_xy_m.get(key, None)
        if spawn_xy is None:
            return False

        self._set_agent_visible(h, False)
        self._set_agent_collision_enabled(h, False)
        if not self._move_agent_to_respawn_hold(key):
            self._set_agent_collision_enabled(h, True)
            self._set_agent_visible(h, True)
            return False
        self._pending_respawns[token] = {
            "key": key,
            "radius_m": float(self.respawn_hold_radius_m),
            "spawn_xy_m": spawn_xy,
        }
        return True

    def _release_pending_respawns(self) -> None:
        if not self._pending_respawns:
            return

        released: List[Tuple[int, int]] = []
        for token, pending in list(self._pending_respawns.items()):
            radius_m = float(pending.get("radius_m", self.respawn_hold_radius_m))
            if not self._spawn_area_is_clear(token, radius_m):
                continue
            world_idx, agent_id = token
            h = self._get_agent_handle(world_idx, agent_id)
            if h is None:
                continue
            key = pending.get("key", None)
            if key is None or not self._teleport_agent_to_spawn(key):
                continue
            self._reset_agent_contact_state(h)
            self._set_agent_collision_enabled(h, True)
            self._set_agent_visible(h, True)
            if getattr(h, "pose_prim", None) is not None and h.pose_prim.IsValid():
                rb_prim = _find_rigid_body_prim(h.pose_prim)
                if rb_prim is not None:
                    _zero_rb_vel(rb_prim)
            released.append(token)

        for token in released:
            self._pending_respawns.pop(token, None)
            if self.verbose:
                print(f"[respawn] released world={token[0]} agent={token[1]}")

    def _freeze_agents(self, keys: List[object], which: np.ndarray) -> None:
        """
        "Freeze" = command brake and zero rigid-body velocities.
        """
        idx = np.where(which)[0]
        if idx.size == 0:
            return

        # Command brake
        U = np.zeros((len(keys), 3), dtype=np.float32)
        U[idx, 2] = 1.0
        try:
            self.ctrl.apply_all(U)
        except Exception:
            pass

        # Zero velocities
        for i in idx:
            k = keys[i]
            h = self.ctrl.get(k.world_idx, k.agent_id)
            if h is None:
                continue
            prim = h.pose_prim
            if not prim or not prim.IsValid():
                continue
            rb_prim = _find_rigid_body_prim(prim)
            if rb_prim is not None:
                _zero_rb_vel(rb_prim)

    def _pending_mask_for_keys(self, keys: List[object]) -> np.ndarray:
        if not self._pending_respawns:
            return np.zeros((len(keys),), dtype=bool)
        return np.asarray(
            [self._agent_token_from_key(k) in self._pending_respawns for k in keys],
            dtype=bool,
        )

    def _hide_agents(self, keys: List[object], which: np.ndarray) -> None:
        idx = np.where(which)[0]
        if idx.size == 0:
            return
        for i in idx:
            k = keys[i]
            h = self.ctrl.get(k.world_idx, k.agent_id)
            if h is None:
                continue
            try:
                UsdGeom.Imageable(h.pose_prim).MakeInvisible()
            except Exception:
                pass

    def _get_contact_types(self, h) -> List[int]:
        if h is None:
            return []
        prim = self.stage.GetPrimAtPath(h.vehicle_root_path)
        if not prim.IsValid():
            return []
        try:
            cd = prim.GetCustomData()
        except Exception:
            cd = {}
        if not isinstance(cd, dict):
            return []
        types = cd.get("road_contact_types", [])
        out = []
        try:
            for v in types:
                out.append(int(v))
        except Exception:
            return []
        return out

    def _get_vehicle_contact_ids(self, h) -> List[int]:
        if h is None:
            return []
        prim = self.stage.GetPrimAtPath(h.vehicle_root_path)
        if not prim.IsValid():
            return []
        try:
            cd = prim.GetCustomData()
        except Exception:
            cd = {}
        if not isinstance(cd, dict):
            return []
        ids = cd.get("vehicle_contact_ids", [])
        out = []
        try:
            for v in ids:
                out.append(int(v))
        except Exception:
            return []
        return out

    def _get_vehicle_collided(self, h) -> bool:
        if h is None:
            return False
        prim = self.stage.GetPrimAtPath(h.vehicle_root_path)
        if not prim.IsValid():
            return False
        try:
            cd = prim.GetCustomData()
        except Exception:
            cd = {}
        if not isinstance(cd, dict):
            return False
        return bool(cd.get("vehicle_collided", False))

    def _get_vehicle_world_z_m(self, h) -> Optional[float]:
        if h is None:
            return None
        try:
            start_prim = h.xform.GetPrim() if hasattr(h.xform, "GetPrim") else None
            if start_prim is None:
                start_prim = h.pose_prim if hasattr(h, "pose_prim") else None
            if start_prim is None:
                return None
            rb_prim = _find_rigid_body_prim(start_prim)
            if rb_prim is not None:
                M = UsdGeom.Xformable(rb_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            else:
                M = h.xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            p = M.ExtractTranslation()
            return float(p[2]) * self._mpu
        except Exception:
            return None

    def _find_agent_prim_path(self, world_idx: int, agent_id: int) -> Optional[str]:
        world_root = f"{self.root_container}/{self.world_prefix}{int(world_idx):03d}"
        agents_root = f"{world_root}/Agents"
        agents_prim = self.stage.GetPrimAtPath(agents_root)
        if not agents_prim.IsValid():
            return None
        for agent_prim in agents_prim.GetAllChildren():
            try:
                cd = agent_prim.GetCustomData()
            except Exception:
                cd = {}
            if isinstance(cd, dict) and int(cd.get("agent_id", -1)) == int(agent_id):
                return agent_prim.GetPath().pathString
        return None

    def _respawn_agent_from_metadata(self, world_idx: int, agent_id: int) -> bool:
        from src.chocolate_waymo_builder import WaymoJsonMiniWorldBuilder, LocalBounds

        agent_path = self._find_agent_prim_path(world_idx, agent_id)
        if agent_path is None:
            return False
        agent_prim = self.stage.GetPrimAtPath(agent_path)
        if not agent_prim.IsValid():
            return False
        try:
            cd = agent_prim.GetCustomData()
        except Exception:
            cd = {}
        if not isinstance(cd, dict):
            return False

        kept_idx = int(cd.get("kept_idx", -1))
        start_local = cd.get("start_local_m", None)
        goal_local = cd.get("goal_local_m", None)
        start_yaw_deg = float(cd.get("start_yaw_deg", 0.0))
        start_in_goal = bool(cd.get("start_in_goal", False))

        if kept_idx < 0 or start_local is None or goal_local is None:
            return False

        world_root = f"{self.root_container}/{self.world_prefix}{int(world_idx):03d}"
        bounds = LocalBounds(
            width_m=float(self.bounds_size_m),
            length_m=float(self.bounds_size_m),
            origin_xy=(0.0, 0.0),
        )
        builder = WaymoJsonMiniWorldBuilder(
            stage=self.stage,
            world_root=world_root,
            bounds=bounds,
            origin_mode="center",
        )

        goal_path = f"{world_root}/Goals/Goal_{kept_idx:04d}_id{int(agent_id)}"
        self.stage.RemovePrim(goal_path)
        self.stage.RemovePrim(agent_path)

        builder.respawn_agent_with_goal(
            kept_idx=int(kept_idx),
            agent_id=int(agent_id),
            start_local_m=(float(start_local[0]), float(start_local[1]), float(start_local[2])),
            start_yaw_deg=float(start_yaw_deg),
            goal_local_m=(float(goal_local[0]), float(goal_local[1]), float(goal_local[2])),
            start_in_goal=bool(start_in_goal),
            parked_ground_z_m=float(self.respawn_params.get("parked_ground_z_m", 0.0)),
            parked_chassis_size_m=tuple(self.respawn_params.get("parked_chassis_size_m", (4.0, 2.0, 1.0))),
            parked_wheel_radius_m=float(self.respawn_params.get("parked_wheel_radius_m", 0.35)),
            parked_wheel_thickness_m=float(self.respawn_params.get("parked_wheel_thickness_m", 0.15)),
            parked_wheel_inset_x_m=float(self.respawn_params.get("parked_wheel_inset_x_m", 0.35)),
            parked_wheel_inset_y_m=float(self.respawn_params.get("parked_wheel_inset_y_m", 0.25)),
            parked_ground_clearance_m=float(self.respawn_params.get("parked_ground_clearance_m", 0.25)),
            goal_radius_m=float(self.respawn_params.get("goal_radius_m", self.goal_success_dist_m)),
            goal_ring_z_m=float(self.respawn_params.get("goal_ring_z_m", 0.0)),
            goal_ring_tube_radius_m=float(self.respawn_params.get("goal_ring_tube_radius_m", 0.12)),
            goal_trigger_height_m=float(self.respawn_params.get("goal_trigger_height_m", 0.6)),
            vehicle_trigger_enable=bool(self.respawn_params.get("vehicle_trigger_enable", False)),
            vehicle_trigger_offset_m=tuple(
                self.respawn_params.get("vehicle_trigger_offset_m", (0.0, 0.0, 0.0))
            ),
            vehicle_trigger_size_m=tuple(
                self.respawn_params.get("vehicle_trigger_size_m", (1.0, 1.0, 1.0))
            ),
            vehicle_trigger_script_enable=bool(
                self.respawn_params.get("vehicle_trigger_script_enable", True)
            ),
        )

        return True

    def reset_done(self, done_mask: np.ndarray) -> None:
        if done_mask is None or not np.any(done_mask):
            return

        idx = np.where(done_mask)[0]

        if self.respawn_on_reset:
            used_builder_respawn = False
            for i in idx:
                k = self._keys[i]
                if not self._queue_respawn(k):
                    self._respawn_agent_from_metadata(k.world_idx, k.agent_id)
                    used_builder_respawn = True
            if used_builder_respawn:
                self.ctrl.refresh()
            self._release_pending_respawns()
            self._done[idx] = False
            self._success_latched[idx] = False
            self.sim.step(render=False)
            obs, mask, keys2 = self._build_obs()
            self._keys = keys2
            dist_n = obs[:, 4].astype(np.float32)
            dist_m = dist_n * (self.bounds_size_m * math.sqrt(2.0))
            self._mask = mask.copy()
            self._prev_dist_m[idx] = dist_m[idx]
            return

        # make sure spawn cache exists
        self._cache_spawn_if_missing()

        for i in idx:
            k = self._keys[i]
            self._teleport_agent_to_spawn(k)

        # clear episode bookkeeping
        self._done[idx] = False
        self._success_latched[idx] = False

        # IMPORTANT: clear obs-builder velocity memory for those keys
        st = getattr(self.obs_builder, "state", None)
        prev_map = getattr(st, "prev_pos_xy_m", None) if st is not None else None
        if isinstance(prev_map, dict):
            for i in idx:
                prev_map.pop(self._keys[i], None)

        # sync 1 physics step so teleport takes effect visually/physically
        self.sim.step(render=False)

        # refresh prev distance (meters)
        obs, mask, _ = self._build_obs()
        dist_n = obs[:, 4].astype(np.float32)
        dist_m = dist_n * (self.bounds_size_m * math.sqrt(2.0))
        self._mask = mask.copy()
        self._prev_dist_m[idx] = dist_m[idx]

    def reset_timeout(self) -> None:
        if self.respawn_on_reset and self._keys:
            done_mask = np.ones((len(self._keys),), dtype=bool)
            self.reset_done(done_mask)
            self.t = 0
        else:
            self.reset()


    # -------------------------
    # Core API
    # -------------------------
    def reset(self) -> Tuple[np.ndarray, np.ndarray, List[object]]:
        self.t = 0
        self._pending_respawns.clear()

        # Refresh controller registry
        self.ctrl.refresh()
        keys = self.ctrl.keys()
        print('after ctrl refresh')
        # Warmup step helps IsaacSim settle controller prims (you already do similar). :contentReference[oaicite:8]{index=8}
        for _ in range(self.warmup_on_reset_steps):
            self.sim.step(render=False)

        # Build obs/mask
        obs, mask, keys2 = self._build_obs()
        keys = keys2

        N = len(keys)
        self._keys = keys
        self._mask = mask.copy()

        # cache start poses once
        self._cache_start_pose_for_keys(keys)
        print('then it must be you')
        # init per-agent episode state
        self._done = np.zeros((N,), dtype=bool)
        self._success_latched = np.zeros((N,), dtype=bool)

        dist_n = obs[:, 4].astype(np.float32)
        dist_m = dist_n * (self.bounds_size_m * math.sqrt(2.0))
        self._prev_dist_m = dist_m.copy()


        if self.verbose:
            print(f"[env.reset] N={N} active={int(mask.sum())}")

        return obs, mask, keys

    def step(self, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StepInfo]:
        if self.t == 0 and (self._keys is None or len(self._keys) == 0):
            if self.verbose:
                print("[env.step] called before reset(); auto-resetting ...")
            self.reset()
        
        U = np.asarray(U, dtype=np.float32)
        keys = self._keys
        N = len(keys)

        if U.ndim != 2 or U.shape[0] != N or U.shape[1] not in (2, 3):
            raise ValueError(f"Action must be shape (N,2) or (N,3). got {U.shape}, N={N}")

        # Convert 2D -> 3D controller action: [thr, steer, brake]
        if U.shape[1] == 2:
            a_long = np.clip(U[:, 0], -1.0, 1.0)
            steer  = np.clip(U[:, 1], -1.0, 1.0)
            thr   = np.clip(a_long,  0.0, 1.0)
            brake = np.clip(-a_long, 0.0, 1.0)
            U3 = np.stack([thr, steer, brake], axis=-1).astype(np.float32)
        else:
            U3 = U

        # Don’t apply actions to agents already done (keeps them parked until reset_done)
        if self._done is not None and self._done.shape[0] == N and self._done.any():
            U3 = U3.copy()
            U3[self._done, :] = 0.0
            U3[self._done, 2] = 1.0  # brake
        self._release_pending_respawns()
        if self._pending_respawns:
            if U3 is U:
                U3 = U3.copy()
            token_to_idx = {
                self._agent_token_from_key(k): i
                for i, k in enumerate(keys)
            }
            for token in self._pending_respawns:
                idx = token_to_idx.get(token)
                if idx is None:
                    continue
                U3[idx, :] = 0.0
                U3[idx, 2] = 1.0
        # Apply controls once per env step
        self.ctrl.apply_all(U3)

        # Step physics action_repeat times
        for _ in range(self.action_repeat):
            self.sim.step(render=self.render)

        self.t += 1

        # Observe
        obs, mask, keys2 = self._build_obs()
        if self.obs_viz_enable:
            self._debug_viz_obs_first_agent(obs, keys2)

        # If keys changed, re-init state (should not happen if you stopped deleting prims).
        if len(keys2) != N:
            if self.verbose:
                print(f"[env.step] WARNING key count changed {N}->{len(keys2)}. Re-init state.")
            self._keys = keys2
            self._mask = mask.copy()
            N = len(keys2)
            self._done = np.zeros((N,), dtype=bool)
            self._success_latched = np.zeros((N,), dtype=bool)
            self._prev_dist_m = obs[:, 4].astype(np.float32).copy()
            keys = keys2

        dist_n = obs[:, 4].astype(np.float32)  # normalized
        dist_m = dist_n * (self.bounds_size_m * math.sqrt(2.0))

        pending_mask = self._pending_mask_for_keys(keys)
        active = mask & (~self._done) & (~pending_mask)

        # SUCCESS: distance threshold in meters
        success_now = (dist_m <= self.goal_success_dist_m) & active
        newly_success = success_now & (~self._success_latched)

        if newly_success.any():
            self._success_latched[newly_success] = True
            self._done[newly_success] = True
            self._freeze_agents(keys, newly_success)
            if self.clear_on_done:
                self._hide_agents(keys, newly_success)
        # Reward: progress toward goal (meters)
        progress = (self._prev_dist_m - dist_m) * self.reward_scale
        reward = np.zeros((N,), dtype=np.float32)
        reward[active] = progress[active]

        # Success bonus only on first reach
        if newly_success.any():
            reward[newly_success] += self.success_bonus

        # Action penalty (only for active rows)
        if self.action_l2_penalty > 0:
            l2 = (U3[:, 0] ** 2 + U3[:, 1] ** 2 + U3[:, 2] ** 2).astype(np.float32)
            reward[active] -= self.action_l2_penalty * l2[active]

        # Idle penalty (encourage movement)
        if self.idle_penalty_enable:
            vx_n = obs[:, 5].astype(np.float32)
            vy_n = obs[:, 6].astype(np.float32)
            speed_mps = np.sqrt(vx_n * vx_n + vy_n * vy_n) * 10.0
            idle = (speed_mps < self.idle_speed_threshold_mps) & active
            if idle.any():
                reward[idle] -= float(self.idle_penalty_per_step)

        # Dense TTC penalty (only for active rows)
        if self.ttc_penalty_enable:
            ttc_pen = self._compute_ttc_penalty(keys, active)
            reward[active] += ttc_pen[active]
        # Per-step survival reward (only for active rows)
        if self.survival_reward_per_step != 0.0:
            reward[active] += float(self.survival_reward_per_step)
        # Collision penalty with selected road types
        road_collided = np.zeros((N,), dtype=bool)
        if self._collision_tracker is not None:
            road_collided = self._collision_tracker.consume_collisions(keys)
            if road_collided.any() and self.collision_penalty != 0.0:
                reward[road_collided] += float(self.collision_penalty)
            if self.collision_debug:
                if not self._collision_debug_printed:
                    summary = self._collision_tracker.debug_summary()
                    print(f"[collision-debug] trigger_summary={summary}")
                    self._collision_debug_printed = True
                pairs = self._collision_tracker.consume_pairs()
                if pairs:
                    print(f"[collision-debug] t={self.t} pairs={pairs}")
                debug_hits = self._collision_tracker.consume_debug()
                if debug_hits:
                    print(f"[collision] t={self.t} hits={debug_hits}")

        below_min_z = np.zeros((N,), dtype=bool)
        if self.min_vehicle_z_m is not None:
            for i, k in enumerate(keys):
                if not active[i]:
                    continue
                h = self.ctrl.get(k.world_idx, k.agent_id)
                if h is None:
                    continue
                z_m = self._get_vehicle_world_z_m(h)
                if z_m is not None and z_m < self.min_vehicle_z_m:
                    below_min_z[i] = True
            if below_min_z.any():
                if self.collision_penalty != 0.0:
                    reward[below_min_z] += float(self.collision_penalty)
                self._done[below_min_z] = True

        # Road-contact termination based on trigger contact list
        if self.road_contact_done_types:
            hit_contact = np.zeros((N,), dtype=bool)
            for i, k in enumerate(keys):
                if not active[i]:
                    continue
                h = self.ctrl.get(k.world_idx, k.agent_id)
                if h is None:
                    continue
                contact_types = self._get_contact_types(h)
                if any(t in self.road_contact_done_types for t in contact_types):
                    hit_contact[i] = True
            if hit_contact.any():
                reward[hit_contact] += float(self.road_contact_done_penalty)
                self._done[hit_contact] = True

        # Lane-center per-step reward (based on road contact types)
        lane_hit = np.zeros((N,), dtype=bool)
        if self.lane_center_reward_enable:
            for i, k in enumerate(keys):
                if not active[i]:
                    continue
                h = self.ctrl.get(k.world_idx, k.agent_id)
                if h is None:
                    continue
                contact_types = self._get_contact_types(h)
                if any(t in self.lane_center_reward_types for t in contact_types):
                    lane_hit[i] = True
            if lane_hit.any():
                reward[lane_hit] += float(self.lane_center_reward_per_step)

        # Vehicle-trigger termination based on vehicle contact list
        if self.vehicle_contact_done:
            hit_contact = np.zeros((N,), dtype=bool)
            agent_id_to_idx = {int(k.agent_id): i for i, k in enumerate(keys)}
            for i, k in enumerate(keys):
                if not active[i]:
                    continue
                h = self.ctrl.get(k.world_idx, k.agent_id)
                if h is None:
                    continue
                contact_ids = self._get_vehicle_contact_ids(h)
                if not contact_ids:
                    continue
                hit_contact[i] = True
                if self.vehicle_contact_done_mark_both:
                    for other_id in contact_ids:
                        j = agent_id_to_idx.get(int(other_id))
                        if j is not None:
                            hit_contact[j] = True
            if hit_contact.any():
                reward[hit_contact] += float(self.vehicle_contact_done_penalty)
                self._done[hit_contact] = True

        # Collision flags (for reward shaping / logging)
        vehicle_collided = np.zeros((N,), dtype=bool)
        for i, k in enumerate(keys):
            if not active[i]:
                continue
            h = self.ctrl.get(k.world_idx, k.agent_id)
            if h is None:
                continue
            if self._get_vehicle_collided(h):
                vehicle_collided[i] = True
        collided_flags = road_collided | vehicle_collided | below_min_z
        collided_flags[pending_mask] = False

        # Timeout
        timeout = (self.t >= self.max_steps)

        # Done if already done, or timeout (for active rows), or invalid (no goal/pose)
        done = self._done.copy()
        if timeout:
            done[active] = True
        done |= (~mask)  # consistent with your old env logic :contentReference[oaicite:9]{index=9}
        if self.t % 10 == 0:
            print(f"[env] t={self.t} timeout={timeout} done_any={done.any()} active={active.sum()}")

        # update prev dist for next step
        self._prev_dist_m = dist_m.copy()
        self._mask = mask.copy()

        off_road = np.zeros((N,), dtype=bool)
        if self.lane_center_reward_enable:
            off_road = active & (~lane_hit)

        if self.road_contact_debug and (self.t % self.road_contact_debug_every == 0):
            # Print one example agent's contact types and off_road flag
            sample_idx = None
            for i, k in enumerate(keys):
                if not active[i]:
                    continue
                sample_idx = i
                h = self.ctrl.get(k.world_idx, k.agent_id)
                types = self._get_contact_types(h) if h is not None else []
                print(
                    f"[road-contact] t={self.t} agent={int(k.agent_id)} "
                    f"types={types} off_road={bool(off_road[i])}"
                )
                break
            if sample_idx is None:
                print(f"[road-contact] t={self.t} no active agents")

        info = StepInfo(
            keys=keys,
            mask=mask,
            dist_m=dist_m,                  # kept name for compatibility with your prints
            success=self._success_latched.copy(),
            collided=collided_flags.copy(),
            off_road=off_road,
            timeout=bool(timeout),
            t_env=int(self.t),
        )
        return obs, reward, done, info


class _RoadCollisionTracker:
    def __init__(self, stage: Usd.Stage, ctrl, road_types: set):
        self.stage = stage
        self.ctrl = ctrl
        self.road_types = set(int(x) for x in road_types)
        self._collided_keys = set()
        self._debug_hits = []
        self._pairs = []
        self._trigger_instancers = []
        self._trigger_counts = {}
        self._sub = None
        self._sub_trigger = None
        self._scan_trigger_instancers()
        self._subscribe()

    def _subscribe(self) -> None:
        try:
            import omni.physx

            if hasattr(omni.physx, "get_physx_simulation_interface"):
                sim_iface = omni.physx.get_physx_simulation_interface()
            else:
                sim_iface = omni.physx.get_physx_interface()

            if hasattr(sim_iface, "subscribe_contact_report_events"):
                self._sub = sim_iface.subscribe_contact_report_events(self._on_contact)
            if hasattr(sim_iface, "subscribe_trigger_report_events"):
                self._sub_trigger = sim_iface.subscribe_trigger_report_events(self._on_trigger)
        except Exception:
            self._sub = None
            self._sub_trigger = None

    def _scan_trigger_instancers(self) -> None:
        self._trigger_instancers = []
        self._trigger_counts = {}
        try:
            for prim in self.stage.TraverseAll():
                if not prim.IsValid():
                    continue
                path = prim.GetPath().pathString
                if not path.endswith("/Triggers"):
                    continue
                if prim.GetTypeName() != "PointInstancer":
                    continue
                rt = None
                try:
                    cd = prim.GetCustomData()
                except Exception:
                    cd = {}
                if isinstance(cd, dict) and "road_type" in cd:
                    try:
                        rt = int(cd["road_type"])
                    except Exception:
                        rt = None
                self._trigger_instancers.append(path)
                if rt is not None:
                    self._trigger_counts[rt] = self._trigger_counts.get(rt, 0) + 1
        except Exception:
            pass

    def debug_summary(self) -> dict:
        return {
            "trigger_instancers": len(self._trigger_instancers),
            "trigger_types": dict(self._trigger_counts),
        }

    def _find_road_type(self, prim_path: str) -> Optional[int]:
        if not prim_path:
            return None
        prim = self.stage.GetPrimAtPath(prim_path)
        while prim and prim.IsValid():
            try:
                cd = prim.GetCustomData()
            except Exception:
                cd = {}
            if isinstance(cd, dict) and "road_type" in cd:
                try:
                    return int(cd["road_type"])
                except Exception:
                    return None
            prim = prim.GetParent()
        return None

    def _find_key_for_path(self, prim_path: str):
        if not prim_path:
            return None
        for k in self.ctrl.keys():
            h = self.ctrl.get(k.world_idx, k.agent_id)
            if h is None:
                continue
            if prim_path.startswith(h.vehicle_root_path):
                return k
            if prim_path.startswith(h.pose_path):
                return k
        return None

    def _handle_pair(self, a_path: str, b_path: str) -> None:
        a_type = self._find_road_type(a_path)
        b_type = self._find_road_type(b_path)

        self._pairs.append((a_path, b_path, a_type, b_type))

        if a_type is not None and a_type in self.road_types:
            key = self._find_key_for_path(b_path)
            if key is not None:
                self._collided_keys.add(key)
                self._debug_hits.append((key, int(a_type), a_path, b_path))
        if b_type is not None and b_type in self.road_types:
            key = self._find_key_for_path(a_path)
            if key is not None:
                self._collided_keys.add(key)
                self._debug_hits.append((key, int(b_type), b_path, a_path))

    def _on_contact(self, contact) -> None:
        try:
            a_path = getattr(contact, "actor0", None)
            b_path = getattr(contact, "actor1", None)
            print(f"[collision-debug] contact actor0={a_path} actor1={b_path}")
            if a_path or b_path:
                self._handle_pair(str(a_path), str(b_path))
                return
        except Exception:
            pass

        try:
            a_path = contact.get("actor0", None)
            b_path = contact.get("actor1", None)
            print(f"[collision-debug] contact dict actor0={a_path} actor1={b_path}")
            if a_path or b_path:
                self._handle_pair(str(a_path), str(b_path))
        except Exception:
            pass

    def _on_trigger(self, trigger) -> None:
        try:
            a_path = getattr(trigger, "trigger", None)
            b_path = getattr(trigger, "other", None)
            print(f"[collision-debug] trigger trigger={a_path} other={b_path}")
            if a_path or b_path:
                self._handle_pair(str(a_path), str(b_path))
                return
        except Exception:
            pass
        try:
            a_path = trigger.get("trigger", None)
            b_path = trigger.get("other", None)
            print(f"[collision-debug] trigger dict trigger={a_path} other={b_path}")
            if a_path or b_path:
                self._handle_pair(str(a_path), str(b_path))
        except Exception:
            pass

    def consume_collisions(self, keys: List[object]) -> np.ndarray:
        mask = np.zeros((len(keys),), dtype=bool)
        if not self._collided_keys:
            return mask
        for i, k in enumerate(keys):
            if k in self._collided_keys:
                mask[i] = True
        self._collided_keys.clear()
        return mask

    def consume_debug(self) -> List[Tuple[object, int, str, str]]:
        if not self._debug_hits:
            return []
        hits = list(self._debug_hits)
        self._debug_hits.clear()
        return hits

    def consume_pairs(self) -> List[Tuple[str, str, Optional[int], Optional[int]]]:
        if not self._pairs:
            return []
        pairs = list(self._pairs)
        self._pairs.clear()
        return pairs
