# chocolate_env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np

from pxr import Gf, Usd, UsdGeom, UsdPhysics, Sdf


@dataclass
class StepInfo:
    keys: List[object]
    mask: np.ndarray          # (N,) bool
    dist_m: np.ndarray        # (N,) float32  (note: your obs builder currently returns DIST IN METERS, not normalized)
    success: np.ndarray       # (N,) bool     (latched)
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
        collision_penalty_types: Optional[List[int]] = None,
        collision_debug: bool = False,
        road_contact_done_types: Optional[List[int]] = None,
        road_contact_done_penalty: float = -1.0,
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
        self.collision_penalty_types = set(int(x) for x in (collision_penalty_types or []))
        self.collision_debug = bool(collision_debug)
        self.road_contact_done_types = set(int(x) for x in (road_contact_done_types or []))
        self.road_contact_done_penalty = float(road_contact_done_penalty)

        self.render = bool(render)
        self.root_container = str(root_container)
        self.world_prefix = str(world_prefix)
        self.warmup_on_reset_steps = max(0, int(warmup_on_reset_steps))
        self.respawn_on_reset = bool(respawn_on_reset)
        self.respawn_params = respawn_params or {}
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
        self._spawn_pos_units: dict = {}   # key -> (x_u, y_u, z_u)
        self._spawn_quat: dict = {}        # key -> (w, x, y, z)  (world orientation)
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
    def _cache_spawn_if_missing(self):
        # cache LOCAL pose from pose_prim (NOT rigid body prim)
        for k in self._keys:
            if k in self._start_local_translate:
                continue
            h = self.ctrl.get(k.world_idx, k.agent_id)
            if h is None or (not h.pose_prim) or (not h.pose_prim.IsValid()):
                continue
            capi = UsdGeom.XformCommonAPI(h.pose_prim)
            t = capi.GetTranslate()
            r = capi.GetRotate()  # XYZ degrees
            self._start_local_translate[k] = (float(t[0]), float(t[1]), float(t[2]))
            self._start_local_yaw_deg[k] = float(r[2])


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
        )
        return obs, mask, keys

    def _cache_start_pose_for_keys(self, keys: List[object]) -> None:
        """
        Cache local translate + yaw (deg) using XformCommonAPI on pose_prim.
        This is robust for resets because we're restoring local ops on the moving prim (pose_prim). :contentReference[oaicite:6]{index=6}
        """
        for k in keys:
            if k in self._start_local_translate:
                continue
            h = self.ctrl.get(k.world_idx, k.agent_id)
            if h is None:
                continue
            pose_prim = h.pose_prim
            if not pose_prim or not pose_prim.IsValid():
                continue

            capi = UsdGeom.XformCommonAPI(pose_prim)
            try:
                # local translate
                t = capi.GetTranslate()
                tx, ty, tz = float(t[0]), float(t[1]), float(t[2])

                # local rotation: we only restore Z (yaw) in XYZ order
                r = capi.GetRotate()
                yaw_deg = float(r[2])

                self._start_local_translate[k] = (tx, ty, tz)
                self._start_local_yaw_deg[k] = yaw_deg
            except Exception:
                # fallback: derive yaw from world transform (best-effort)
                try:
                    M = h.xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    p = M.ExtractTranslation()
                    self._start_local_translate[k] = (float(p[0]), float(p[1]), float(p[2]))
                    self._start_local_yaw_deg[k] = float(_yaw_from_xform(M) * 180.0 / math.pi)
                except Exception:
                    continue

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
        )

        return True

    def reset_done(self, done_mask: np.ndarray) -> None:
        if done_mask is None or not np.any(done_mask):
            return

        idx = np.where(done_mask)[0]

        if self.respawn_on_reset:
            for i in idx:
                k = self._keys[i]
                self._respawn_agent_from_metadata(k.world_idx, k.agent_id)
            self.ctrl.refresh()
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
            h = self.ctrl.get(k.world_idx, k.agent_id)
            if h is None or (not h.pose_prim) or (not h.pose_prim.IsValid()):
                continue

            t0 = self._start_local_translate.get(k, None)
            yaw0 = self._start_local_yaw_deg.get(k, 0.0)
            if t0 is None:
                continue

            # 1) teleport by USD local ops (on pose_prim = .../Vehicle)
            capi = UsdGeom.XformCommonAPI(h.pose_prim)
            capi.SetTranslate(Gf.Vec3d(t0[0], t0[1], t0[2]))
            capi.SetRotate(
                Gf.Vec3f(0.0, 0.0, float(yaw0)),
                UsdGeom.XformCommonAPI.RotationOrderXYZ
            )

            # 2) zero rigidbody velocities (find RB by walking UP)
            rb_prim = _find_rigid_body_prim(h.pose_prim)
            if rb_prim is not None:
                _zero_rb_vel(rb_prim)

            # 3) make sure it’s not stuck braking forever (optional but helps)
            #    briefly release controls next step by clearing done flags below

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
        # Apply controls once per env step
        self.ctrl.apply_all(U3)

        # Step physics action_repeat times
        for _ in range(self.action_repeat):
            self.sim.step(render=self.render)

        self.t += 1

        # Observe
        obs, mask, keys2 = self._build_obs()

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

        active = mask & (~self._done)

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
        print('progress', progress)
        reward = np.zeros((N,), dtype=np.float32)
        reward[active] = progress[active]

        # Success bonus only on first reach
        if newly_success.any():
            reward[newly_success] += self.success_bonus

        # Action penalty (only for active rows)
        if self.action_l2_penalty > 0:
            l2 = (U3[:, 0] ** 2 + U3[:, 1] ** 2 + U3[:, 2] ** 2).astype(np.float32)
            reward[active] -= self.action_l2_penalty * l2[active]
        print(self._collision_tracker)
        # Collision penalty with selected road types
        if self._collision_tracker is not None:
            collided = self._collision_tracker.consume_collisions(keys)
            print('collided', collided)
            if collided.any() and self.collision_penalty != 0.0:
                reward[collided] += float(self.collision_penalty)
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

        info = StepInfo(
            keys=keys,
            mask=mask,
            dist_m=dist_m,                  # kept name for compatibility with your prints
            success=self._success_latched.copy(),
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
            if a_path or b_path:
                print(f"[collision-debug] contact actor0={a_path} actor1={b_path}")
                self._handle_pair(str(a_path), str(b_path))
                return
        except Exception:
            pass

        try:
            a_path = contact.get("actor0", None)
            b_path = contact.get("actor1", None)
            if a_path or b_path:
                print(f"[collision-debug] contact dict actor0={a_path} actor1={b_path}")
                self._handle_pair(str(a_path), str(b_path))
        except Exception:
            pass

    def _on_trigger(self, trigger) -> None:
        try:
            a_path = getattr(trigger, "trigger", None)
            b_path = getattr(trigger, "other", None)
            if a_path or b_path:
                print(f"[collision-debug] trigger trigger={a_path} other={b_path}")
                self._handle_pair(str(a_path), str(b_path))
                return
        except Exception:
            pass
        try:
            a_path = trigger.get("trigger", None)
            b_path = trigger.get("other", None)
            if a_path or b_path:
                print(f"[collision-debug] trigger dict trigger={a_path} other={b_path}")
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
