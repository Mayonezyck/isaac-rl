from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from pxr import Usd, UsdGeom, Gf
from pxr import UsdPhysics

from src.trfc import encode_weather_context, weather_context_dim


# ----------------------------
# Small math helpers
# ----------------------------
def _wrap_pi(a: float) -> float:
    # wrap to [-pi, pi]
    a = (a + math.pi) % (2.0 * math.pi) - math.pi
    return a

def _yaw_from_xform(M: Gf.Matrix4d) -> float:
    """
    Extract planar yaw assuming Z-up and vehicle is mostly flat.
    USD local +X is treated as "forward".
    """
    # Transform local forward axis into world
    fwd_world = M.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
    fx, fy = float(fwd_world[0]), float(fwd_world[1])
    return math.atan2(fy, fx)

def _world_to_ego_xy(dx: float, dy: float, yaw: float) -> Tuple[float, float]:
    """
    world delta -> ego delta (2D), where ego x is forward and ego y is left.
    yaw is ego->world rotation about +Z.
    """
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    x_ego =  cy * dx + sy * dy
    y_ego = -sy * dx + cy * dy
    return x_ego, y_ego

def _get_rb_linear_velocity_world(rb_prim: Usd.Prim, tc: Usd.TimeCode) -> Tuple[float, float, float]:
    """
    Returns linear velocity in WORLD frame (stage units/sec), (vx, vy, vz).
    """
    rb = UsdPhysics.RigidBodyAPI(rb_prim)
    v_attr = rb.GetVelocityAttr()
    if not v_attr or not v_attr.IsValid():
        return 0.0, 0.0, 0.0
    v = v_attr.Get(tc)
    if v is None:
        return 0.0, 0.0, 0.0
    return float(v[0]), float(v[1]), float(v[2])


def _meters_per_unit(stage: Usd.Stage) -> float:
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    return float(mpu) if mpu and float(mpu) > 0 else 0.01

def _get_vehicle_size_local(prim: Usd.Prim, bbox_cache: UsdGeom.BBoxCache) -> Tuple[float, float]:
    # Returns (length_x, width_y) in stage units from local bounds.
    if prim is None or not prim.IsValid():
        return 0.0, 0.0
    try:
        box = bbox_cache.ComputeLocalBound(prim)
        rng = box.GetRange()
        size = rng.GetSize()
        return abs(float(size[0])), abs(float(size[1]))
    except Exception:
        return 0.0, 0.0

def _pick_vehicle_size_prim(stage: Usd.Stage, start_prim: Usd.Prim) -> Optional[Usd.Prim]:
    if start_prim is None or not start_prim.IsValid():
        return None
    try:
        child = stage.GetPrimAtPath(f"{start_prim.GetPath()}/Vehicle")
        if child.IsValid():
            return child
    except Exception:
        pass
    return start_prim

def _find_rigid_body_prim(start_prim: Usd.Prim) -> Optional[Usd.Prim]:
    """
    Walk up parents to find a prim with UsdPhysics.RigidBodyAPI applied.
    (Commonly the chassis prim for a PhysX vehicle.)
    """
    p = start_prim
    while p and p.IsValid():
        try:
            rb = UsdPhysics.RigidBodyAPI(p)
            # The API object always "constructs", but we need to confirm it has velocity attrs
            if rb.GetVelocityAttr().IsValid():
                return p
        except Exception:
            pass
        p = p.GetParent()
    return None

# ----------------------------
# Goal lookup (from your builder’s customData)
# ----------------------------
def _build_goal_map_for_world(stage: Usd.Stage, goals_root_path: str) -> Dict[int, Tuple[float, float, float]]:
    """
    Returns {agent_id: (gx, gy, gz)} in METERS (local coords of the miniworld).
    This uses goal prim customData: goal_center_m (written by _spawn_goal_ring_with_trigger).
    """
    out: Dict[int, Tuple[float, float, float]] = {}
    goals_prim = stage.GetPrimAtPath(goals_root_path)
    if not goals_prim.IsValid():
        return out

    for gprim in goals_prim.GetAllChildren():
        # Optional check: is_goal tag
        try:
            cd = gprim.GetCustomData()
        except Exception:
            cd = {}

        # Skip non-goal prims
        if isinstance(cd, dict) and cd.get("is_goal", False) is not True:
            # Many prims might not have the tag; name-based fallback below could be added if needed
            pass

        name = gprim.GetName()  # e.g. Goal_0028_id3163
        j = name.rfind("_id")
        if j < 0:
            continue
        digits = []
        for ch in name[j + 3 :]:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if not digits:
            continue
        agent_id = int("".join(digits))

        center = None
        if isinstance(cd, dict):
            center = cd.get("goal_center_m", None)

        if center is None:
            continue

        try:
            gx, gy, gz = center
            out[agent_id] = (float(gx), float(gy), float(gz))
        except Exception:
            continue

    return out

# ----------------------------
# Observation builder
# ----------------------------
# @dataclass
# class ObsState:
#     # store last position per AgentKey to estimate velocity
#     prev_pos_xy_m: Dict[object, Tuple[float, float]]

class ChocolateObsBuilder:
    """
    First-iteration observation:
      obs[i] = [
        rel_goal_x_ego_m,
        rel_goal_y_ego_m,
        heading_error_sin,
        heading_error_cos,
        dist_to_goal_m,
        vx_ego_mps,
        vy_ego_mps,
        water_film_norm,
        road_is_ac,
        road_is_sma,
        road_is_ogfc,
        (optional) road points ...
        road points use either:
          [rel_x, rel_y, road_type]
        or, when road-point directions are enabled:
          [rel_x, rel_y, road_type, dir_x_ego, dir_y_ego]
        (always) nearest vehicle features (63 * 6):
          [rel_x, rel_y, length, width, rel_yaw, speed]
      ]
    """
    def __init__(self):
        pass
        #self.state = ObsState(prev_pos_xy_m={})

    def _get_road_points_for_world(self, stage: Usd.Stage, world_root: str):
        prim = stage.GetPrimAtPath(world_root)
        if not prim.IsValid():
            return None, None, None
        try:
            cd = prim.GetCustomData()
        except Exception:
            cd = {}
        if not isinstance(cd, dict):
            return None, None, None
        pts = cd.get("road_points_m", None)
        dirs = cd.get("road_point_dirs", None)
        types = cd.get("road_point_types", None)
        if pts is None or types is None:
            return None, None, None
        try:
            pts_np = np.asarray(pts, dtype=np.float32)
            dirs_np = np.asarray(dirs, dtype=np.float32) if dirs is not None else None
            types_np = np.asarray(types, dtype=np.int32)
        except Exception:
            return None, None, None
        if pts_np.ndim != 2 or pts_np.shape[1] < 2:
            return None, None, None
        if dirs_np is None or dirs_np.ndim != 2 or dirs_np.shape[0] != pts_np.shape[0] or dirs_np.shape[1] < 2:
            dirs_np = np.zeros((pts_np.shape[0], 3), dtype=np.float32)
        mpu = _meters_per_unit(stage)
        # Stored road points are local to world_root; convert to world frame so they
        # are in the same frame as agent poses from ComputeLocalToWorldTransform.
        try:
            M = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = M.ExtractTranslation()
            pts_np = pts_np.copy()
            pts_np[:, 0] += float(t[0])
            pts_np[:, 1] += float(t[1])
            if pts_np.shape[1] >= 3:
                pts_np[:, 2] += float(t[2])
        except Exception:
            pass
        # Convert stage units to meters for consistency with reward/obs config values.
        pts_np[:, :2] *= float(mpu)
        if pts_np.shape[1] >= 3:
            pts_np[:, 2] *= float(mpu)
        dirs_xy = dirs_np[:, :2].astype(np.float32, copy=True)
        norms = np.linalg.norm(dirs_xy, axis=1, keepdims=True)
        dirs_xy = np.divide(dirs_xy, np.maximum(norms, 1e-6), out=np.zeros_like(dirs_xy), where=norms > 1e-6)
        return pts_np, types_np, dirs_xy

    def _get_world_weather_context(self, stage: Usd.Stage, world_root: str) -> np.ndarray:
        prim = stage.GetPrimAtPath(world_root)
        if not prim.IsValid():
            return np.zeros((weather_context_dim(),), dtype=np.float32)

        try:
            custom_data = prim.GetCustomData()
        except Exception:
            custom_data = {}
        if not isinstance(custom_data, dict):
            custom_data = {}

        metadata = custom_data.get("ground_friction_metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        water_film_mm = metadata.get(
            "water_film_mm",
            custom_data.get("ground_water_film_mm", 0.0),
        )
        road_type = metadata.get(
            "road_type",
            custom_data.get("ground_road_type", None),
        )
        return np.asarray(
            encode_weather_context(
                water_film_mm=water_film_mm,
                road_type=road_type,
            ),
            dtype=np.float32,
        )

    def build_obs_all_controlled(
        self,
        *,
        stage: Usd.Stage,
        bounds_size_m,
        ctrl,  # ChocolateWorldVehicleController
        root_container: str = "/World/MiniWorlds",
        world_prefix: str = "world_",
        dt: float = 1.0 / 60.0,
        use_world_count_from_ctrl: bool = True,
        road_points_enable: bool,
        road_points_k: int,
        road_points_radius_m: float,
        road_points_type_norm: float,
        road_points_mode: str = "knn",
        road_points_include_dirs: bool = False,
        vehicle_obs_enable: bool,
        vehicle_obs_k: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[object]]:
        """
        Returns:
          obs:  (N, 11 + road_points*(3|5) + vehicle_obs_k*6) float32
          mask: (N,) bool  (True if goal + pose valid)
          keys: length N (AgentKey list aligned with obs rows)
        """
        keys = ctrl.keys()
        N = len(keys)
        mpu = _meters_per_unit(stage)
        base_dim = 7 + weather_context_dim()
        road_point_feat_dim = 5 if road_points_include_dirs else 3
        vehicle_feat_dim = 6
        extra_dim = 0
        if road_points_enable:
            extra_dim = int(road_points_k) * int(road_point_feat_dim)
        vehicle_dim = int(vehicle_obs_k) * int(vehicle_feat_dim) if vehicle_obs_enable else 0
        obs = np.zeros((N, base_dim + extra_dim), dtype=np.float32)
        if vehicle_dim > 0:
            obs = np.pad(obs, ((0, 0), (0, vehicle_dim)), mode="constant")
        mask = np.zeros((N,), dtype=bool)
        #print('im in the builder')
        # Build per-world goal maps once
        world_count = ctrl.world_count if use_world_count_from_ctrl else max([k.world_idx for k in keys], default=-1) + 1
        goals_by_world: List[Dict[int, Tuple[float, float, float]]] = []
        weather_contexts_by_world: List[np.ndarray] = []
        road_points_by_world: List[Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]] = []
        for wi in range(int(world_count)):
            world_root = f"{root_container}/{world_prefix}{wi:03d}"
            goals_root = f"{world_root}/Goals"
            goals_by_world.append(_build_goal_map_for_world(stage, goals_root))
            weather_contexts_by_world.append(
                self._get_world_weather_context(stage, world_root)
            )
            if road_points_enable:
                road_points_by_world.append(self._get_road_points_for_world(stage, world_root))
            else:
                road_points_by_world.append((None, None, None))

        # Precompute per-agent state for neighbor observations
        per_agent = {}
        world_to_indices: Dict[int, List[int]] = {}
        if vehicle_obs_enable and vehicle_dim > 0:
            tc = Usd.TimeCode.Default()
            bbox_cache = UsdGeom.BBoxCache(tc, [UsdGeom.Tokens.default_], useExtentsHint=True)
            for i, k in enumerate(keys):
                h = ctrl.get(k.world_idx, k.agent_id)
                if h is None:
                    continue
                try:
                    start_prim = h.xform.GetPrim() if hasattr(h.xform, "GetPrim") else None
                    if start_prim is None:
                        start_prim = h.pose_prim if hasattr(h, "pose_prim") else None
                    rb_prim = _find_rigid_body_prim(start_prim) if start_prim is not None else None
                    if rb_prim is not None:
                        M = UsdGeom.Xformable(rb_prim).ComputeLocalToWorldTransform(tc)
                    else:
                        M = h.xform.ComputeLocalToWorldTransform(tc)
                    p = M.ExtractTranslation()
                    px, py, _pz = float(p[0]) * mpu, float(p[1]) * mpu, float(p[2]) * mpu
                    yaw = _yaw_from_xform(M)
                except Exception:
                    continue

                vx_w, vy_w = 0.0, 0.0
                try:
                    start_prim = h.xform.GetPrim() if hasattr(h.xform, "GetPrim") else None
                    if start_prim is None:
                        start_prim = h.pose_prim if hasattr(h, "pose_prim") else None
                    rb_prim = _find_rigid_body_prim(start_prim) if start_prim is not None else None
                    if rb_prim is not None:
                        vx_w, vy_w, _vz_w = _get_rb_linear_velocity_world(rb_prim, tc)
                except Exception:
                    vx_w, vy_w = 0.0, 0.0
                vx_w *= float(mpu)
                vy_w *= float(mpu)

                length_m, width_m = 0.0, 0.0
                try:
                    size_prim = _pick_vehicle_size_prim(stage, start_prim) if start_prim is not None else None
                    length_m, width_m = _get_vehicle_size_local(size_prim, bbox_cache)
                except Exception:
                    length_m, width_m = 0.0, 0.0
                length_m *= float(mpu)
                width_m *= float(mpu)

                if length_m <= 0.0:
                    length_m = 4.0
                if width_m <= 0.0:
                    width_m = 2.0

                per_agent[i] = (k.world_idx, px, py, yaw, vx_w, vy_w, length_m, width_m)
                world_to_indices.setdefault(k.world_idx, []).append(i)

        for i, k in enumerate(keys):
            h = ctrl.get(k.world_idx, k.agent_id)
            if h is None:
                continue

            # Pose
            try:
                start_prim = h.xform.GetPrim() if hasattr(h.xform, "GetPrim") else None
                if start_prim is None:
                    start_prim = h.pose_prim if hasattr(h, "pose_prim") else None
                rb_prim = _find_rigid_body_prim(start_prim) if start_prim is not None else None
                if rb_prim is not None:
                    M = UsdGeom.Xformable(rb_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                else:
                    M = h.xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                p = M.ExtractTranslation()
                px, py, pz = float(p[0]) * mpu, float(p[1]) * mpu, float(p[2]) * mpu
            except Exception:
                continue

            # Goal center in this world
            gmap = goals_by_world[k.world_idx] if 0 <= k.world_idx < len(goals_by_world) else {}
            g = gmap.get(k.agent_id, None)
            if g is None:
                continue
            gx, gy, gz = g

            dx = gx - px
            dy = gy - py
            dist = math.sqrt(dx * dx + dy * dy)

            # Yaw + ego-frame goal
            try:
                yaw = _yaw_from_xform(M)
            except Exception:
                yaw = 0.0
            relx, rely = _world_to_ego_xy(dx, dy, yaw)
            # Heading error
            goal_dir = math.atan2(dy, dx)
            he = _wrap_pi(goal_dir - yaw)
            he_s = math.sin(he)
            he_c = math.cos(he)

            # Finite-diff velocity in ego frame
            # prev = self.state.prev_pos_xy_m.get(k, None)
            # if prev is None or dt <= 1e-9:
            #     vx_ego, vy_ego = 0.0, 0.0
            # else:
            #     vx_w = (px - prev[0]) / dt
            #     vy_w = (py - prev[1]) / dt
            #     vx_ego, vy_ego = _world_to_ego_xy(vx_w, vy_w, yaw)

            # self.state.prev_pos_xy_m[k] = (px, py)
            # PhysX rigid-body velocity -> ego frame
            tc = Usd.TimeCode.Default()

            vx_ego, vy_ego = 0.0, 0.0
            try:
                # Start from something you have: h.xform is an Xform prim wrapper, so use its prim
                start_prim = h.xform.GetPrim() if hasattr(h.xform, "GetPrim") else None
                if start_prim is None:
                    start_prim = h.pose_prim if hasattr(h, "pose_prim") else None

                if start_prim is not None:
                    rb_prim = _find_rigid_body_prim(start_prim)
                    if rb_prim is not None:
                        vx_w, vy_w, _vz_w = _get_rb_linear_velocity_world(rb_prim, tc)
                        vx_w *= float(mpu)
                        vy_w *= float(mpu)
                        vx_ego, vy_ego = _world_to_ego_xy(vx_w, vy_w, yaw)
            except Exception:
                # keep zeros if anything fails
                vx_ego, vy_ego = 0.0, 0.0


            # normalization
            L = float(bounds_size_m)              # e.g. 200.0
            D = float(bounds_size_m) * math.sqrt(2.0)
            v_scale = float(10)      # e.g. 10.0
            relx_n = relx / L
            rely_n = rely / L
            dist_n = dist / D
            vx_n   = vx_ego / v_scale
            vy_n   = vy_ego / v_scale

            obs[i, 0] = float(relx_n)
            obs[i, 1] = float(rely_n)
            obs[i, 2] = float(he_s)
            obs[i, 3] = float(he_c)
            obs[i, 4] = float(dist_n)
            obs[i, 5] = float(vx_n)
            obs[i, 6] = float(vy_n)
            weather_context = (
                weather_contexts_by_world[k.world_idx]
                if 0 <= k.world_idx < len(weather_contexts_by_world)
                else np.zeros((weather_context_dim(),), dtype=np.float32)
            )
            obs[i, 7:11] = weather_context
            if road_points_enable:
                pts, types, dirs = (
                    road_points_by_world[k.world_idx]
                    if 0 <= k.world_idx < len(road_points_by_world)
                    else (None, None, None)
                )
                if pts is not None and types is not None and dirs is not None and pts.shape[0] > 0:
                    dx_all = pts[:, 0] - px
                    dy_all = pts[:, 1] - py
                    dist2 = dx_all * dx_all + dy_all * dy_all
                    if road_points_radius_m > 0:
                        keep = dist2 <= float(road_points_radius_m * road_points_radius_m)
                    else:
                        keep = np.ones_like(dist2, dtype=bool)
                    idxs = np.where(keep)[0]
                    if idxs.size > 0:
                        mode = str(road_points_mode).strip().lower().replace("_", "-")
                        if mode == "knn":
                            idxs = idxs[np.argsort(dist2[idxs])]
                        elif mode == "road-running":
                            # Keep original map insertion order from road_points_m.
                            pass
                        else:
                            # Fallback for unknown mode.
                            idxs = idxs[np.argsort(dist2[idxs])]
                        idxs = idxs[: int(road_points_k)]
                        off = base_dim
                        for j, idx in enumerate(idxs):
                            dx = float(dx_all[idx])
                            dy = float(dy_all[idx])
                            x_e, y_e = _world_to_ego_xy(dx, dy, yaw)
                            dir_x_e, dir_y_e = _world_to_ego_xy(float(dirs[idx, 0]), float(dirs[idx, 1]), yaw)
                            norm = float(road_points_radius_m) if road_points_radius_m > 0 else 1.0
                            base_idx = off + road_point_feat_dim * j
                            obs[i, base_idx + 0] = float(x_e / norm)
                            obs[i, base_idx + 1] = float(y_e / norm)
                            t_val = float(types[idx])
                            obs[i, base_idx + 2] = (
                                t_val / float(road_points_type_norm)
                                if road_points_type_norm > 0
                                else t_val
                            )
                            if road_points_include_dirs:
                                obs[i, base_idx + 3] = float(dir_x_e)
                                obs[i, base_idx + 4] = float(dir_y_e)

            # Nearest vehicle features (always appended)
            if vehicle_obs_enable and vehicle_dim > 0 and i in per_agent:
                off = base_dim + (
                    int(road_points_k) * int(road_point_feat_dim) if road_points_enable else 0
                )
                world_idx, px_i, py_i, yaw_i, _vx_i, _vy_i, _len_i, _wid_i = per_agent[i]
                candidates = []
                for j in world_to_indices.get(world_idx, []):
                    if j == i:
                        continue
                    _, px_j, py_j, yaw_j, vx_j, vy_j, len_j, wid_j = per_agent.get(j, (None,) * 8)
                    if px_j is None:
                        continue
                    dx = px_j - px_i
                    dy = py_j - py_i
                    dist2 = dx * dx + dy * dy
                    candidates.append((dist2, j, dx, dy, yaw_j, vx_j, vy_j, len_j, wid_j))
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    L = float(bounds_size_m)
                    max_yaw = math.pi
                    speed_scale = 10.0
                    for n, (_, j, dx, dy, yaw_j, vx_j, vy_j, len_j, wid_j) in enumerate(candidates[:vehicle_obs_k]):
                        relx, rely = _world_to_ego_xy(dx, dy, yaw_i)
                        rel_yaw = _wrap_pi(yaw_j - yaw_i)
                        speed = math.sqrt(vx_j * vx_j + vy_j * vy_j)
                        relx_n = relx / L
                        rely_n = rely / L
                        len_n = len_j / L
                        wid_n = wid_j / L
                        yaw_n = rel_yaw / max_yaw
                        speed_n = speed / speed_scale
                        idx = off + n * vehicle_feat_dim
                        obs[i, idx + 0] = float(relx_n)
                        obs[i, idx + 1] = float(rely_n)
                        obs[i, idx + 2] = float(len_n)
                        obs[i, idx + 3] = float(wid_n)
                        obs[i, idx + 4] = float(yaw_n)
                        obs[i, idx + 5] = float(speed_n)
            mask[i] = True
        return obs, mask, keys
