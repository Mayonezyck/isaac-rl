from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from pxr import Usd, UsdGeom, Gf

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
@dataclass
class ObsState:
    # store last position per AgentKey to estimate velocity
    prev_pos_xy_m: Dict[object, Tuple[float, float]]

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
      ]
    """
    def __init__(self):
        self.state = ObsState(prev_pos_xy_m={})

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
    ) -> Tuple[np.ndarray, np.ndarray, List[object]]:
        """
        Returns:
          obs:  (N,6) float32
          mask: (N,) bool  (True if goal + pose valid)
          keys: length N (AgentKey list aligned with obs rows)
        """
        keys = ctrl.keys()
        N = len(keys)
        obs = np.zeros((N, 7), dtype=np.float32)
        mask = np.zeros((N,), dtype=bool)
        #print('im in the builder')
        # Build per-world goal maps once
        world_count = ctrl.world_count if use_world_count_from_ctrl else max([k.world_idx for k in keys], default=-1) + 1
        goals_by_world: List[Dict[int, Tuple[float, float, float]]] = []
        for wi in range(int(world_count)):
            world_root = f"{root_container}/{world_prefix}{wi:03d}"
            goals_root = f"{world_root}/Goals"
            goals_by_world.append(_build_goal_map_for_world(stage, goals_root))

        for i, k in enumerate(keys):
            h = ctrl.get(k.world_idx, k.agent_id)
            if h is None:
                continue

            # Pose
            try:
                M = h.xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                p = M.ExtractTranslation()
                px, py, pz = float(p[0]), float(p[1]), float(p[2])
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
            prev = self.state.prev_pos_xy_m.get(k, None)
            if prev is None or dt <= 1e-9:
                vx_ego, vy_ego = 0.0, 0.0
            else:
                vx_w = (px - prev[0]) / dt
                vy_w = (py - prev[1]) / dt
                vx_ego, vy_ego = _world_to_ego_xy(vx_w, vy_w, yaw)

            self.state.prev_pos_xy_m[k] = (px, py)

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
            mask[i] = True
        return obs, mask, keys
