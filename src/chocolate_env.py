# chocolate_env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

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
    newly_success: np.ndarray # (N,) bool
    road_contact_done: np.ndarray  # (N,) bool
    vehicle_contact_done: np.ndarray  # (N,) bool
    collided: np.ndarray      # (N,) bool
    road_collided: np.ndarray # (N,) bool
    vehicle_collided: np.ndarray  # (N,) bool
    below_min_z: np.ndarray   # (N,) bool
    off_road: np.ndarray      # (N,) bool
    lane_hit: np.ndarray      # (N,) bool
    lane_error_m: np.ndarray  # (N,) float32
    heading_alignment: np.ndarray  # (N,) float32
    route_progress_m: np.ndarray  # (N,) float32
    active: np.ndarray        # (N,) bool
    pending: np.ndarray       # (N,) bool
    timeout: bool
    t_env: int


def _yaw_from_xform(M: Gf.Matrix4d) -> float:
    """Yaw (rad) from world transform, using +X as forward (matches your obs builder)."""
    fwd = M.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
    fx, fy = float(fwd[0]), float(fwd[1])
    return math.atan2(fy, fx)


def _wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def _ego_to_world_xy(x_ego: float, y_ego: float, yaw_world: float) -> Tuple[float, float]:
    c = math.cos(float(yaw_world))
    s = math.sin(float(yaw_world))
    return (c * float(x_ego) - s * float(y_ego), s * float(x_ego) + c * float(y_ego))


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
        hard_remove_done_agents: bool = False,
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
        geom_lane_reward_enable: bool = False,
        geom_lane_reward_per_step: float = 0.0,
        geom_lane_tolerance_m: float = 1.75,
        geom_lane_heading_weight: float = 0.5,
        geom_lane_min_alignment: float = 0.5,
        geom_route_progress_weight: float = 0.0,
        geom_offroad_metrics_enable: bool = False,
        geom_offroad_lateral_threshold_m: float = 3.0,
        geom_offroad_distance_threshold_m: float = 6.0,
        geom_lane_types: Optional[List[int]] = None,
        geom_road_edge_types: Optional[List[int]] = None,
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
        road_points_include_dirs: bool = False,
        vehicle_obs_enable: bool = False,
        vehicle_obs_k: int = 63,
        ttc_penalty_enable: bool = False,
        ttc_penalty_alpha: float = 1.0,
        ttc_penalty_max: float = 1.0,
        ttc_penalty_min_ttc: float = 0.2,
        road_edge_ttc_penalty_enable: bool = False,
        road_edge_ttc_penalty_alpha: float = 0.0,
        road_edge_ttc_penalty_max: float = 0.5,
        road_edge_ttc_penalty_min_ttc: float = 0.5,
        road_edge_ttc_hard_min_ttc: float = 0.5,
        road_edge_ttc_radius_m: Optional[float] = None,
        ttc_delta_penalty_enable: bool = False,
        ttc_delta_penalty_alpha: float = 0.0,
        ttc_delta_penalty_max: float = 0.5,
        ttc_delta_penalty_normalize_by_dt: bool = False,
        ttc_use_vehicle_size: bool = True,
        ttc_vehicle_radius_scale: float = 0.75,
        ttc_vehicle_radius_margin_m: float = 0.20,
        ttc_backend: str = "numpy",
        obs_viz_enable: bool = False,
        obs_viz_world_idx: int = 0,
        obs_viz_agent_rank: int = 0,
        render: bool = False,
        root_container: str = "/World/MiniWorlds",
        world_prefix: str = "world_",
        warmup_on_reset_steps: int = 1,
        respawn_on_reset: bool = False,
        respawn_mode: str = "rebuild",
        respawn_params: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ):
        self.sim = sim
        self.stage = stage
        self.ctrl = ctrl
        self.obs_builder = obs_builder
        self.bounds_size_m = float(bounds_size_m)
        self.physics_dt = float(physics_dt)
        self.action_repeat = max(1, int(action_repeat))
        self.max_steps = int(max_steps)

        self.clear_on_done = bool(clear_on_done)
        self.hard_remove_done_agents = bool(hard_remove_done_agents)
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
        self.geom_lane_reward_enable = bool(geom_lane_reward_enable)
        self.geom_lane_reward_per_step = float(geom_lane_reward_per_step)
        self.geom_lane_tolerance_m = max(1e-3, float(geom_lane_tolerance_m))
        self.geom_lane_heading_weight = float(np.clip(geom_lane_heading_weight, 0.0, 1.0))
        self.geom_lane_min_alignment = float(np.clip(geom_lane_min_alignment, -1.0, 1.0))
        self.geom_route_progress_weight = float(geom_route_progress_weight)
        self.geom_offroad_metrics_enable = bool(geom_offroad_metrics_enable)
        self.geom_offroad_lateral_threshold_m = max(1e-3, float(geom_offroad_lateral_threshold_m))
        self.geom_offroad_distance_threshold_m = max(1e-3, float(geom_offroad_distance_threshold_m))
        self.geom_lane_types = set(int(x) for x in (geom_lane_types or [1, 2]))
        self.geom_road_edge_types = set(int(x) for x in (geom_road_edge_types or [15, 16]))
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
        self.road_points_include_dirs = bool(road_points_include_dirs)
        self.road_point_feat_dim = 5 if self.road_points_include_dirs else 3
        self.vehicle_obs_enable = bool(vehicle_obs_enable)
        self.vehicle_obs_k = int(vehicle_obs_k)
        self.ttc_penalty_enable = bool(ttc_penalty_enable)
        self.ttc_penalty_alpha = float(ttc_penalty_alpha)
        self.ttc_penalty_max = float(ttc_penalty_max)
        self.ttc_penalty_min_ttc = float(ttc_penalty_min_ttc)
        self.road_edge_ttc_penalty_enable = bool(road_edge_ttc_penalty_enable)
        self.road_edge_ttc_penalty_alpha = max(0.0, float(road_edge_ttc_penalty_alpha))
        self.road_edge_ttc_penalty_max = max(0.0, float(road_edge_ttc_penalty_max))
        self.road_edge_ttc_penalty_min_ttc = max(1e-3, float(road_edge_ttc_penalty_min_ttc))
        self.road_edge_ttc_hard_min_ttc = max(0.0, float(road_edge_ttc_hard_min_ttc))
        self.road_edge_ttc_radius_m = (
            None if road_edge_ttc_radius_m is None else max(0.0, float(road_edge_ttc_radius_m))
        )
        self.ttc_delta_penalty_enable = bool(ttc_delta_penalty_enable)
        self.ttc_delta_penalty_alpha = max(0.0, float(ttc_delta_penalty_alpha))
        self.ttc_delta_penalty_max = max(0.0, float(ttc_delta_penalty_max))
        self.ttc_delta_penalty_normalize_by_dt = bool(ttc_delta_penalty_normalize_by_dt)
        self.ttc_use_vehicle_size = bool(ttc_use_vehicle_size)
        self.ttc_vehicle_radius_scale = max(0.0, float(ttc_vehicle_radius_scale))
        self.ttc_vehicle_radius_margin_m = max(0.0, float(ttc_vehicle_radius_margin_m))
        backend = str(ttc_backend).strip().lower()
        if backend not in {"numpy", "torch_cuda"}:
            print(
                f"[warn][fallback] Unknown ttc_backend='{ttc_backend}', falling back to 'numpy'."
            )
            backend = "numpy"
        self.ttc_backend = backend
        self.obs_viz_enable = bool(obs_viz_enable)
        self.obs_viz_world_idx = int(obs_viz_world_idx)
        self.obs_viz_agent_rank = int(obs_viz_agent_rank)

        self.render = bool(render)
        self.root_container = str(root_container)
        self.world_prefix = str(world_prefix)
        self.warmup_on_reset_steps = max(0, int(warmup_on_reset_steps))
        self.respawn_on_reset = bool(respawn_on_reset)
        mode = str(respawn_mode).strip().lower()
        if mode not in {"hybrid", "rebuild"}:
            print(
                f"[warn][fallback] Unknown respawn_mode='{respawn_mode}', falling back to 'rebuild'."
            )
            mode = "rebuild"
        self.respawn_mode = mode
        self.respawn_params = respawn_params or {}
        self.respawn_hold_radius_m = float(self.respawn_params.get("respawn_hold_radius_m", 3.0))
        self.respawn_clear_ignore_non_controllable = bool(
            self.respawn_params.get("respawn_clear_ignore_non_controllable", True)
        )
        self.respawn_release_max_wait_steps = max(
            0,
            int(self.respawn_params.get("respawn_release_max_wait_steps", 2)),
        )
        self.respawn_spawn_z_m = float(self.respawn_params.get("spawn_z_m", 1.0))
        self.startup_below_min_z_preflight_steps = max(
            0,
            int(self.respawn_params.get("startup_below_min_z_preflight_steps", 0)),
        )
        self.verbose = bool(verbose)
        self._fallback_warn_counts: Dict[str, int] = {}
        self._fallback_warn_limit = 3
        self._respawn_friction_debug_logged = False
        self._respawn_rebuild_count = 0
        self.respawn_friction_debug_every = max(
            0, int(self.respawn_params.get("respawn_friction_debug_every", 0))
        )
        self.respawn_rebuild_flush_steps_before_create = max(
            0, int(self.respawn_params.get("respawn_rebuild_flush_steps_before_create", 1))
        )
        self.respawn_rebuild_flush_steps_after_create = max(
            0, int(self.respawn_params.get("respawn_rebuild_flush_steps_after_create", 1))
        )
        self._respawn_shared_snapshot_debug_logged = False

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
        self._quarantined_tokens: Set[Tuple[int, int]] = set()
        self._prev_world_xy_m: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self._prev_min_ttc_s: Dict[Tuple[int, int], float] = {}
        self._world_road_geometry: Dict[int, Dict[str, np.ndarray]] = {}
        self._vehicle_radius_u_cache: Dict[Tuple[int, int], float] = {}
        self._vehicle_shape_u_cache: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
        self._bbox_cache = None
        self._mpu = float(getattr(self.sim, "meters_per_unit", 1.0))  # IsaacSim usually has this
        self._collision_tracker = None
        if self.collision_penalty_types or self.collision_debug:
            self._collision_tracker = _RoadCollisionTracker(self.stage, self.ctrl, self.collision_penalty_types)
        self._collision_debug_printed = False


    # -------------------------
    # Internal helpers
    # -------------------------

    def _warn_fallback(self, key: str, message: str) -> None:
        count = int(self._fallback_warn_counts.get(key, 0))
        if count < self._fallback_warn_limit:
            suffix = " [further repeats suppressed]" if (count + 1) == self._fallback_warn_limit else ""
            print(f"[warn][fallback] {message}{suffix}")
        self._fallback_warn_counts[key] = count + 1

    def _get_ttc_torch_cuda(self):
        if self.ttc_backend != "torch_cuda":
            return None, None
        try:
            import torch  # type: ignore
        except Exception as exc:
            self._warn_fallback(
                "ttc_backend_torch_import",
                f"TTC backend 'torch_cuda' requested but torch import failed ({exc}); using numpy TTC.",
            )
            return None, None
        try:
            if not torch.cuda.is_available():
                self._warn_fallback(
                    "ttc_backend_torch_cuda_unavailable",
                    "TTC backend 'torch_cuda' requested but CUDA is unavailable; using numpy TTC.",
                )
                return None, None
            return torch, torch.device("cuda")
        except Exception as exc:
            self._warn_fallback(
                "ttc_backend_torch_cuda_error",
                f"TTC backend 'torch_cuda' init failed ({exc}); using numpy TTC.",
            )
            return None, None

    def _world_root_path(self, world_idx: int) -> str:
        return f"{self.root_container}/{self.world_prefix}{int(world_idx):03d}"

    def _world_ground_material_path(self, world_idx: int) -> str:
        return f"{self._world_root_path(world_idx)}/Materials/GroundSurface"

    def _world_ground_friction_value(self, world_idx: int) -> Optional[float]:
        world_prim = self.stage.GetPrimAtPath(self._world_root_path(world_idx))
        if not world_prim.IsValid():
            return None
        try:
            cd = world_prim.GetCustomData()
        except Exception:
            cd = {}
        if not isinstance(cd, dict):
            return None
        mu = cd.get("ground_friction", None)
        if mu is None:
            return None
        try:
            return float(mu)
        except Exception:
            return None

    def _collect_shared_friction_table_snapshot(
        self,
        *,
        shared_root_path: str = "/World/VehicleShared",
    ) -> Dict[str, Dict[str, Any]]:
        try:
            from pxr import PhysxSchema
        except Exception as exc:
            self._warn_fallback(
                "respawn_friction_debug_import",
                f"Failed importing PhysxSchema for friction debug ({exc}).",
            )
            return {}

        snapshot: Dict[str, Dict[str, Any]] = {}
        for name in ("SummerTireFrictionTable", "AllSeasonFrictionTable", "SlickTireFrictionTable"):
            table_path = f"{shared_root_path}/{name}"
            prim = self.stage.GetPrimAtPath(table_path)
            if not prim.IsValid():
                continue
            try:
                table = PhysxSchema.PhysxVehicleTireFrictionTable.Get(self.stage, table_path)
            except Exception:
                table = None
            if not table:
                continue

            try:
                rel = table.GetGroundMaterialsRel()
                targets = [p.pathString for p in list(rel.GetTargets() or [])]
            except Exception:
                targets = []
            try:
                values = [float(v) for v in list(table.GetFrictionValuesAttr().Get() or [])]
            except Exception:
                values = []

            snapshot[name] = {
                "targets": targets,
                "values": values,
            }
        return snapshot

    def _log_shared_friction_table_snapshot(
        self,
        *,
        tag: str,
        material_path: Optional[str] = None,
        shared_root_path: str = "/World/VehicleShared",
    ) -> None:
        snapshot = self._collect_shared_friction_table_snapshot(shared_root_path=shared_root_path)
        if not snapshot:
            print(f"[respawn][friction-debug] {tag} no tire friction tables found")
            return

        print(f"[respawn][friction-debug] {tag} tables={len(snapshot)} material={material_path}")
        for name in sorted(snapshot.keys()):
            info = snapshot[name]
            targets = list(info.get("targets", []) or [])
            values = list(info.get("values", []) or [])
            if material_path and material_path in targets:
                idx = int(targets.index(material_path))
                val = values[idx] if idx < len(values) else None
                print(
                    f"[respawn][friction-debug] {tag} table={name} "
                    f"material_idx={idx} material_value={val} targets={len(targets)} values={len(values)}"
                )
            else:
                print(
                    f"[respawn][friction-debug] {tag} table={name} "
                    f"material_absent targets={len(targets)} values={len(values)}"
                )

    def _reapply_world_friction_patch(self, world_idx: int) -> bool:
        try:
            from gpudrive_chocolate.env.choco_world import patch_vehicle_shared_friction_tables
        except Exception as exc:
            self._warn_fallback(
                "respawn_friction_patch_import",
                f"Failed importing tire friction patch helper ({exc}).",
            )
            return False

        material_path = self._world_ground_material_path(world_idx)
        if not self.stage.GetPrimAtPath(material_path).IsValid():
            self._warn_fallback(
                "respawn_friction_patch_material_missing",
                f"Ground material missing for friction re-patch world={int(world_idx)} path={material_path}.",
            )
            return False

        mu_eff = self._world_ground_friction_value(world_idx)
        if mu_eff is None or not np.isfinite(mu_eff):
            self._warn_fallback(
                "respawn_friction_patch_mu_missing",
                f"Ground friction value missing/invalid for world={int(world_idx)}.",
            )
            return False

        try:
            patch_vehicle_shared_friction_tables(
                self.stage,
                shared_root_path="/World/VehicleShared",
                material_path=material_path,
                friction_value=float(mu_eff),
            )
            return True
        except Exception as exc:
            self._warn_fallback(
                "respawn_friction_patch_apply",
                f"Failed applying friction re-patch world={int(world_idx)} mu={float(mu_eff):.4f} ({exc}).",
            )
            return False

    def _collect_prim_attr_snapshot(self, root_path: str) -> Dict[str, str]:
        root = self.stage.GetPrimAtPath(str(root_path))
        if not root.IsValid():
            return {}
        snap: Dict[str, str] = {}
        for prim in Usd.PrimRange(root):
            p = prim.GetPath().pathString
            for attr in prim.GetAttributes():
                if not attr.IsValid():
                    continue
                try:
                    val = attr.Get(Usd.TimeCode.Default())
                except Exception:
                    continue
                if val is None:
                    continue
                try:
                    key = f"{p}.{attr.GetName()}"
                except Exception:
                    continue
                try:
                    snap[key] = repr(val)
                except Exception:
                    snap[key] = str(val)
        return snap

    def _log_snapshot_diff(self, *, tag: str, before: Dict[str, str], after: Dict[str, str]) -> None:
        before_keys = set(before.keys())
        after_keys = set(after.keys())
        added = sorted(after_keys - before_keys)
        removed = sorted(before_keys - after_keys)
        changed = sorted(k for k in (before_keys & after_keys) if before.get(k) != after.get(k))
        print(
            f"[respawn][shared-debug] {tag} "
            f"added={len(added)} removed={len(removed)} changed={len(changed)}"
        )
        preview_n = 8
        if added:
            print(f"[respawn][shared-debug] {tag} added_preview={added[:preview_n]}")
        if removed:
            print(f"[respawn][shared-debug] {tag} removed_preview={removed[:preview_n]}")
        if changed:
            print(f"[respawn][shared-debug] {tag} changed_preview={changed[:preview_n]}")




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

    def _get_agent_world_pose(self, h) -> Optional[Tuple[float, float, float, float]]:
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
            return (
                float(p[0]) * self._mpu,
                float(p[1]) * self._mpu,
                float(p[2]) * self._mpu,
                float(_yaw_from_xform(M)),
            )
        except Exception:
            return None

    def _get_world_road_geometry(self, world_idx: int) -> Optional[Dict[str, np.ndarray]]:
        wi = int(world_idx)
        cached = self._world_road_geometry.get(wi, None)
        if cached is not None:
            return cached

        world_root = f"{self.root_container}/{self.world_prefix}{wi:03d}"
        prim = self.stage.GetPrimAtPath(world_root)
        if not prim.IsValid():
            return None
        try:
            cd = prim.GetCustomData()
        except Exception:
            cd = {}
        if not isinstance(cd, dict):
            return None

        pts = cd.get("road_points_m", None)
        types = cd.get("road_point_types", None)
        dirs = cd.get("road_point_dirs", None)
        if pts is None or types is None:
            return None

        try:
            pts_np = np.asarray(pts, dtype=np.float32)
            types_np = np.asarray(types, dtype=np.int32)
            dirs_np = np.asarray(dirs, dtype=np.float32) if dirs is not None else None
        except Exception:
            return None

        if pts_np.ndim != 2 or pts_np.shape[1] < 2 or types_np.ndim != 1:
            return None

        if dirs_np is None or dirs_np.ndim != 2 or dirs_np.shape[0] != pts_np.shape[0] or dirs_np.shape[1] < 2:
            dirs_np = np.zeros((pts_np.shape[0], 3), dtype=np.float32)

        try:
            M = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = M.ExtractTranslation()
            pts_np = pts_np.copy()
            pts_np[:, 0] += float(t[0])
            pts_np[:, 1] += float(t[1])
            if pts_np.shape[1] >= 3:
                pts_np[:, 2] += float(t[2])
        except Exception:
            pts_np = pts_np.copy()

        pts_xy_m = pts_np[:, :2] * float(self._mpu)
        dirs_xy = dirs_np[:, :2].astype(np.float32, copy=True)
        norms = np.linalg.norm(dirs_xy, axis=1, keepdims=True)
        dirs_xy = np.divide(
            dirs_xy,
            np.maximum(norms, 1e-6),
            out=np.zeros_like(dirs_xy),
            where=norms > 1e-6,
        )

        geom = {
            "points_xy_m": pts_xy_m.astype(np.float32, copy=False),
            "dirs_xy": dirs_xy.astype(np.float32, copy=False),
            "types": types_np.astype(np.int32, copy=False),
        }
        self._world_road_geometry[wi] = geom
        return geom

    def _compute_geometric_lane_features(
        self,
        keys: List[object],
        obs: np.ndarray,
        active: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(keys)
        lane_hit = np.zeros((n,), dtype=bool)
        off_road = np.zeros((n,), dtype=bool)
        lane_error_m = np.zeros((n,), dtype=np.float32)
        heading_alignment = np.zeros((n,), dtype=np.float32)
        route_progress_m = np.zeros((n,), dtype=np.float32)

        if not (
            self.geom_lane_reward_enable
            or self.geom_offroad_metrics_enable
            or self.geom_route_progress_weight != 0.0
        ):
            return lane_hit, off_road, lane_error_m, heading_alignment, route_progress_m

        for i, k in enumerate(keys):
            if not active[i]:
                continue
            h = self.ctrl.get(k.world_idx, k.agent_id)
            pose = self._get_agent_world_pose(h)
            if pose is None:
                continue
            px_m, py_m, _pz_m, yaw = pose
            geom = self._get_world_road_geometry(k.world_idx)
            if geom is None:
                continue

            types = geom["types"]
            lane_mask = np.isin(types, list(self.geom_lane_types))
            if not np.any(lane_mask):
                continue
            lane_points = geom["points_xy_m"][lane_mask]
            lane_dirs = geom["dirs_xy"][lane_mask]
            if lane_points.shape[0] == 0:
                continue

            pos = np.asarray([px_m, py_m], dtype=np.float32)
            deltas = lane_points - pos[None, :]
            dist2 = np.einsum("ij,ij->i", deltas, deltas)
            idx = int(np.argmin(dist2))

            nearest_point = lane_points[idx]
            tangent = lane_dirs[idx].astype(np.float32, copy=True)
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm <= 1e-6:
                continue
            tangent /= tangent_norm

            goal_x_ego = float(obs[i, 0]) * float(self.bounds_size_m)
            goal_y_ego = float(obs[i, 1]) * float(self.bounds_size_m)
            goal_dx_w, goal_dy_w = _ego_to_world_xy(goal_x_ego, goal_y_ego, yaw)
            if goal_dx_w * tangent[0] + goal_dy_w * tangent[1] < 0.0:
                tangent *= -1.0

            normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
            rel = pos - nearest_point
            lateral = abs(float(np.dot(rel, normal)))
            nearest_dist = math.sqrt(max(0.0, float(dist2[idx])))
            tangent_yaw = math.atan2(float(tangent[1]), float(tangent[0]))
            align = max(0.0, math.cos(_wrap_pi(yaw - tangent_yaw)))
            quality = math.exp(-((lateral / self.geom_lane_tolerance_m) ** 2))
            quality *= (1.0 - self.geom_lane_heading_weight) + self.geom_lane_heading_weight * align

            lane_hit[i] = lateral <= self.geom_lane_tolerance_m and align >= self.geom_lane_min_alignment
            off_road[i] = (
                lateral > self.geom_offroad_lateral_threshold_m
                or nearest_dist > self.geom_offroad_distance_threshold_m
            )
            lane_error_m[i] = float(lateral)
            heading_alignment[i] = float(align)

            token = self._agent_token_from_key(k)
            prev_xy = self._prev_world_xy_m.get(token, None)
            if prev_xy is not None:
                step_delta = pos - np.asarray(prev_xy, dtype=np.float32)
                route_progress_m[i] = float(np.dot(step_delta, tangent))
            self._prev_world_xy_m[token] = (float(px_m), float(py_m))

            if self.geom_lane_reward_enable:
                lane_error_m[i] = float(lateral)

        return lane_hit, off_road, lane_error_m, heading_alignment, route_progress_m

    def _pick_vehicle_size_prim(self, vehicle_prim: Usd.Prim) -> Optional[Usd.Prim]:
        if vehicle_prim is None or not vehicle_prim.IsValid():
            return None
        try:
            child = self.stage.GetPrimAtPath(f"{vehicle_prim.GetPath()}/Vehicle")
            if child.IsValid():
                return child
        except Exception:
            pass
        return vehicle_prim

    def _vehicle_shape_u_for_state(
        self,
        *,
        world_idx: int,
        agent_id: int,
        vehicle_prim: Usd.Prim,
        custom_data: Dict[str, Any],
    ) -> Tuple[float, float, float]:
        if not self.ttc_use_vehicle_size:
            return (0.0, 0.0, 0.0)

        token = self._agent_token(world_idx, agent_id)
        cached = self._vehicle_shape_u_cache.get(token, None)
        if cached is not None:
            return (float(cached[0]), float(cached[1]), float(cached[2]))

        mpu = max(float(self._mpu), 1e-6)
        length_u = 0.0
        width_u = 0.0

        # Optional direct metadata path if available in future builders.
        try:
            metadata_dicts: List[Dict[str, Any]] = []
            if isinstance(custom_data, dict):
                metadata_dicts.append(custom_data)
            try:
                child = self.stage.GetPrimAtPath(f"{vehicle_prim.GetPath()}/Vehicle")
                if child.IsValid():
                    cd_child = child.GetCustomData()
                    if isinstance(cd_child, dict):
                        metadata_dicts.append(cd_child)
            except Exception:
                pass
            try:
                parent = vehicle_prim.GetParent()
                if parent is not None and parent.IsValid():
                    cd_parent = parent.GetCustomData()
                    if isinstance(cd_parent, dict):
                        metadata_dicts.append(cd_parent)
            except Exception:
                pass

            for md in metadata_dicts:
                if length_u > 0.0 and width_u > 0.0:
                    break
                if "vehicle_size_m" in md:
                    sz = md.get("vehicle_size_m", None)
                    if isinstance(sz, (list, tuple)) and len(sz) >= 2:
                        length_u = float(sz[0]) / mpu
                        width_u = float(sz[1]) / mpu
                if length_u <= 0.0 and width_u <= 0.0:
                    lm = md.get("vehicle_length_m", md.get("length_m", None))
                    wm = md.get("vehicle_width_m", md.get("width_m", None))
                    if lm is not None and wm is not None:
                        length_u = float(lm) / mpu
                        width_u = float(wm) / mpu
        except Exception as exc:
            self._warn_fallback(
                "ttc_vehicle_size_metadata_error",
                f"Vehicle size metadata parsing failed ({exc}); falling back to bbox/nominal size.",
            )
            length_u = 0.0
            width_u = 0.0

        # Fallback to local bbox size.
        if length_u <= 0.0 or width_u <= 0.0:
            self._warn_fallback(
                "ttc_vehicle_size_bbox_fallback",
                "Missing explicit vehicle size metadata for TTC; falling back to local bbox size.",
            )
            size_prim = self._pick_vehicle_size_prim(vehicle_prim)
            if size_prim is not None and size_prim.IsValid():
                try:
                    if self._bbox_cache is None:
                        self._bbox_cache = UsdGeom.BBoxCache(
                            Usd.TimeCode.Default(),
                            [UsdGeom.Tokens.default_],
                            useExtentsHint=True,
                        )
                    box = self._bbox_cache.ComputeLocalBound(size_prim)
                    rng = box.GetRange()
                    size = rng.GetSize()
                    length_u = abs(float(size[0]))
                    width_u = abs(float(size[1]))
                except Exception as exc:
                    self._warn_fallback(
                        "ttc_vehicle_size_bbox_error",
                        f"Local bbox size query failed ({exc}); falling back to nominal vehicle dimensions.",
                    )
                    length_u = 0.0
                    width_u = 0.0

        # Final fallback to nominal sedan-like dimensions.
        if length_u <= 0.0:
            self._warn_fallback(
                "ttc_vehicle_size_nominal_length",
                "TTC size fallback: using nominal vehicle length 4.0m.",
            )
            length_u = 4.0 / mpu
        if width_u <= 0.0:
            self._warn_fallback(
                "ttc_vehicle_size_nominal_width",
                "TTC size fallback: using nominal vehicle width 2.0m.",
            )
            width_u = 2.0 / mpu

        # Three-circle TTC model: each circle radius follows the vehicle width.
        radius_u = max(0.0, 0.5 * width_u)
        self._vehicle_shape_u_cache[token] = (float(length_u), float(width_u), float(radius_u))
        self._vehicle_radius_u_cache[token] = float(radius_u)
        return (float(length_u), float(width_u), float(radius_u))

    def _vehicle_radius_u_for_state(
        self,
        *,
        world_idx: int,
        agent_id: int,
        vehicle_prim: Usd.Prim,
        custom_data: Dict[str, Any],
    ) -> float:
        _length_u, _width_u, radius_u = self._vehicle_shape_u_for_state(
            world_idx=world_idx,
            agent_id=agent_id,
            vehicle_prim=vehicle_prim,
            custom_data=custom_data,
        )
        return float(radius_u)

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
                token = self._agent_token(wi, int(agent_id))

                # Use a consistent moving frame for TTC state.
                # For controllable vehicles, prefer rigid-body prim pose/velocity from controller handle.
                px = py = None
                yaw = None
                vx, vy = 0.0, 0.0
                if controllable:
                    h = self.ctrl.get(wi, int(agent_id))
                    if h is not None and getattr(h, "pose_prim", None) is not None:
                        rb_prim = self._find_rb_prim(h.pose_prim)
                        pose_prim = rb_prim if rb_prim is not None else h.pose_prim
                        try:
                            M = UsdGeom.Xformable(pose_prim).ComputeLocalToWorldTransform(tc)
                            p = M.ExtractTranslation()
                            px, py = float(p[0]), float(p[1])
                            yaw = float(_yaw_from_xform(M))
                        except Exception:
                            px = py = None
                            yaw = None
                        if rb_prim is not None:
                            vx, vy, _ = self._get_rb_linear_velocity_world(rb_prim)
                        else:
                            self._warn_fallback(
                                "ttc_velocity_missing_rb",
                                "TTC velocity fallback: rigid-body prim not found; using zero velocity.",
                            )

                # Fallback for non-controllable (or missing handle): use vehicle prim transform.
                if px is None or py is None:
                    try:
                        M = UsdGeom.Xformable(vehicle_prim).ComputeLocalToWorldTransform(tc)
                        p = M.ExtractTranslation()
                        px, py = float(p[0]), float(p[1])
                        yaw = float(_yaw_from_xform(M))
                    except Exception:
                        continue
                if yaw is None or not np.isfinite(float(yaw)):
                    yaw = 0.0
                fwd_x = float(math.cos(float(yaw)))
                fwd_y = float(math.sin(float(yaw)))
                fwd_norm = math.hypot(fwd_x, fwd_y)
                if fwd_norm <= 1e-6:
                    fwd_x, fwd_y = 1.0, 0.0
                else:
                    fwd_x /= fwd_norm
                    fwd_y /= fwd_norm

                length_u, width_u, radius_u = self._vehicle_shape_u_for_state(
                    world_idx=wi,
                    agent_id=int(agent_id),
                    vehicle_prim=vehicle_prim,
                    custom_data=cd,
                )
                if self.ttc_use_vehicle_size and radius_u > 0.0 and length_u > 0.0:
                    spine_offset_u = max(0.0, 0.5 * float(length_u) - float(radius_u))
                    offsets_u = np.asarray(
                        [-spine_offset_u, 0.0, spine_offset_u], dtype=np.float32
                    )
                    cx = float(px) + offsets_u * float(fwd_x)
                    cy = float(py) + offsets_u * float(fwd_y)
                    ttc_centers_u = np.stack([cx, cy], axis=-1).astype(np.float32)
                    ttc_radii_u = np.full((3,), float(radius_u), dtype=np.float32)
                else:
                    ttc_centers_u = np.asarray([[float(px), float(py)]], dtype=np.float32)
                    ttc_radii_u = np.asarray([float(max(0.0, radius_u))], dtype=np.float32)

                states.append(
                    {
                        "agent_id": int(agent_id),
                        "token": token,
                        "pos": (px, py),
                        "vel": (vx, vy),
                        "yaw": float(yaw),
                        "fwd": (float(fwd_x), float(fwd_y)),
                        "length_u": float(length_u),
                        "width_u": float(width_u),
                        "radius_u": radius_u,
                        "ttc_centers_u": ttc_centers_u,
                        "ttc_radii_u": ttc_radii_u,
                        "controllable": controllable,
                    }
                )
            if states:
                out[wi] = states
        return out

    def _compute_ttc_penalty_torch_cuda(
        self,
        keys: List[object],
        active: np.ndarray,
        *,
        states_by_world: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> Optional[np.ndarray]:
        torch, torch_dev = self._get_ttc_torch_cuda()
        if torch is None or torch_dev is None:
            return None
        if not self.ttc_use_vehicle_size:
            self._warn_fallback(
                "ttc_point_agent_mode",
                "TTC fallback mode active: ttc_use_vehicle_size=false, so vehicle radii are ignored.",
            )

        if states_by_world is None:
            states_by_world = self._collect_all_vehicle_states()
        penalties = np.zeros((len(keys),), dtype=np.float32)
        env_step_dt = max(1e-6, float(self.physics_dt) * float(self.action_repeat))
        excluded_tokens = set(self._pending_respawns.keys()) | set(self._quarantined_tokens)

        try:
            for i, k in enumerate(keys):
                if not active[i]:
                    continue
                token = self._agent_token_from_key(k)
                wi = int(getattr(k, "world_idx", -1))
                world_states = states_by_world.get(wi, None)
                if not world_states:
                    self._prev_min_ttc_s.pop(token, None)
                    continue
                ego_state = None
                for s in world_states:
                    if s["agent_id"] == int(k.agent_id):
                        ego_state = s
                        break
                if ego_state is None:
                    self._prev_min_ttc_s.pop(token, None)
                    continue

                ego_centers_np = np.asarray(
                    ego_state.get(
                        "ttc_centers_u",
                        np.asarray([ego_state["pos"]], dtype=np.float32),
                    ),
                    dtype=np.float32,
                )
                ego_radii_np = np.asarray(
                    ego_state.get(
                        "ttc_radii_u",
                        np.asarray([ego_state.get("radius_u", 0.0)], dtype=np.float32),
                    ),
                    dtype=np.float32,
                )
                if ego_centers_np.ndim != 2 or ego_centers_np.shape[0] <= 0:
                    ego_centers_np = np.asarray([ego_state["pos"]], dtype=np.float32)
                if ego_radii_np.ndim != 1 or ego_radii_np.shape[0] != ego_centers_np.shape[0]:
                    ego_radii_np = np.full(
                        (ego_centers_np.shape[0],),
                        float(max(0.0, ego_state.get("radius_u", 0.0))),
                        dtype=np.float32,
                    )

                ego_vel_np = np.asarray(ego_state["vel"], dtype=np.float32)[:2]
                ego_fwd_np = np.asarray(ego_state.get("fwd", (1.0, 0.0)), dtype=np.float32)[:2]
                fwd_norm = float(np.linalg.norm(ego_fwd_np))
                if fwd_norm <= 1e-6:
                    ego_fwd_np = np.asarray([1.0, 0.0], dtype=np.float32)
                else:
                    ego_fwd_np = ego_fwd_np / fwd_norm

                other_centers_list: List[np.ndarray] = []
                other_radii_list: List[np.ndarray] = []
                other_vel_list: List[np.ndarray] = []
                for s in world_states:
                    if s["agent_id"] == int(k.agent_id):
                        continue
                    if s.get("token", None) in excluded_tokens:
                        continue
                    centers = np.asarray(
                        s.get("ttc_centers_u", np.asarray([s["pos"]], dtype=np.float32)),
                        dtype=np.float32,
                    )
                    if centers.ndim != 2 or centers.shape[0] <= 0:
                        centers = np.asarray([s["pos"]], dtype=np.float32)
                    radii = np.asarray(
                        s.get("ttc_radii_u", np.asarray([s.get("radius_u", 0.0)], dtype=np.float32)),
                        dtype=np.float32,
                    )
                    if radii.ndim != 1 or radii.shape[0] != centers.shape[0]:
                        radii = np.full(
                            (centers.shape[0],),
                            float(max(0.0, s.get("radius_u", 0.0))),
                            dtype=np.float32,
                        )
                    vel = np.asarray(s["vel"], dtype=np.float32)[:2]
                    vel_rep = np.repeat(vel[None, :], centers.shape[0], axis=0)
                    other_centers_list.append(centers)
                    other_radii_list.append(radii)
                    other_vel_list.append(vel_rep)

                if not other_centers_list:
                    self._prev_min_ttc_s.pop(token, None)
                    continue

                ego_centers = torch.as_tensor(ego_centers_np, dtype=torch.float32, device=torch_dev)
                ego_radii = torch.as_tensor(ego_radii_np, dtype=torch.float32, device=torch_dev)
                ego_vel = torch.as_tensor(ego_vel_np, dtype=torch.float32, device=torch_dev)
                ego_fwd = torch.as_tensor(ego_fwd_np, dtype=torch.float32, device=torch_dev)
                other_centers = torch.as_tensor(
                    np.concatenate(other_centers_list, axis=0),
                    dtype=torch.float32,
                    device=torch_dev,
                )
                other_radii = torch.as_tensor(
                    np.concatenate(other_radii_list, axis=0),
                    dtype=torch.float32,
                    device=torch_dev,
                )
                other_vel = torch.as_tensor(
                    np.concatenate(other_vel_list, axis=0),
                    dtype=torch.float32,
                    device=torch_dev,
                )

                rel = other_centers.unsqueeze(0) - ego_centers.unsqueeze(1)
                rv = other_vel.unsqueeze(0) - ego_vel.view(1, 1, 2)
                rx = rel[:, :, 0]
                ry = rel[:, :, 1]
                rvx = rv[:, :, 0]
                rvy = rv[:, :, 1]
                r2 = rx * rx + ry * ry
                v2 = rvx * rvx + rvy * rvy
                rdotv = rx * rvx + ry * rvy
                combined_r = ego_radii.view(-1, 1) + other_radii.view(1, -1)
                combined_r2 = combined_r * combined_r

                forward_dot = rx * ego_fwd[0] + ry * ego_fwd[1]
                forward_mask = forward_dot > 0.0
                inf = torch.full_like(r2, float("inf"))
                ttc = inf.clone()
                overlap_mask = forward_mask & (r2 <= combined_r2)
                ttc = torch.where(overlap_mask, torch.zeros_like(ttc), ttc)

                moving_mask = forward_mask & (~overlap_mask) & (v2 > 1e-6)
                if bool(torch.any(moving_mask).item()):
                    a = torch.where(moving_mask, v2, torch.ones_like(v2))
                    b = torch.where(moving_mask, 2.0 * rdotv, torch.zeros_like(rdotv))
                    c = torch.where(moving_mask, r2 - combined_r2, torch.zeros_like(r2))
                    disc = b * b - 4.0 * a * c
                    valid_quad = moving_mask & (disc >= 0.0)
                    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
                    denom = 2.0 * torch.clamp(a, min=1e-6)
                    t_enter = (-b - sqrt_disc) / denom
                    t_exit = (-b + sqrt_disc) / denom
                    valid_enter = valid_quad & (t_exit >= 0.0)
                    ttc_quad = torch.where(valid_enter, torch.clamp(t_enter, min=0.0), inf)
                    ttc = torch.minimum(ttc, ttc_quad)

                    unresolved = moving_mask & (~torch.isfinite(ttc))
                    if bool(torch.any(unresolved).item()):
                        dist = torch.sqrt(torch.clamp(r2, min=1e-9))
                        closing_speed = -rdotv / torch.clamp(dist, min=1e-6)
                        clearance = torch.clamp(dist - combined_r, min=0.0)
                        valid_fb = unresolved & (rdotv < 0.0) & (closing_speed > 1e-6)
                        ttc_fb = torch.where(
                            valid_fb,
                            clearance / torch.clamp(closing_speed, min=1e-6),
                            inf,
                        )
                        ttc = torch.minimum(ttc, ttc_fb)

                finite_mask = torch.isfinite(ttc)
                if not bool(torch.any(finite_mask).item()):
                    self._prev_min_ttc_s.pop(token, None)
                    continue
                min_ttc = float(torch.min(ttc[finite_mask]).item())

                abs_penalty = 0.0
                if self.ttc_penalty_enable:
                    if float(min_ttc) < 0.5:
                        abs_penalty = float(self.ttc_penalty_max)
                    else:
                        denom = max(float(min_ttc), float(self.ttc_penalty_min_ttc))
                        abs_penalty = min(
                            float(self.ttc_penalty_max),
                            float(self.ttc_penalty_alpha) / denom,
                        )

                delta_penalty = 0.0
                if self.ttc_delta_penalty_enable and self.ttc_delta_penalty_alpha > 0.0:
                    prev_min_ttc = self._prev_min_ttc_s.get(token, None)
                    if prev_min_ttc is not None and np.isfinite(prev_min_ttc):
                        delta_ttc = float(prev_min_ttc) - float(min_ttc)
                        if self.ttc_delta_penalty_normalize_by_dt:
                            delta_ttc /= env_step_dt
                        if delta_ttc > 0.0:
                            delta_penalty = min(
                                float(self.ttc_delta_penalty_max),
                                float(self.ttc_delta_penalty_alpha) * float(delta_ttc),
                            )

                penalties[i] = -float(abs_penalty + delta_penalty)
                self._prev_min_ttc_s[token] = float(min_ttc)
        except Exception as exc:
            self._warn_fallback(
                "ttc_backend_torch_vehicle_runtime",
                f"torch_cuda vehicle TTC failed ({exc}); falling back to numpy TTC.",
            )
            return None

        return penalties

    def _compute_ttc_penalty(
        self,
        keys: List[object],
        active: np.ndarray,
        states_by_world: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> np.ndarray:
        if not (self.ttc_penalty_enable or self.ttc_delta_penalty_enable):
            return np.zeros((len(keys),), dtype=np.float32)
        if self.ttc_backend == "torch_cuda":
            penalties_torch = self._compute_ttc_penalty_torch_cuda(
                keys,
                active,
                states_by_world=states_by_world,
            )
            if penalties_torch is not None:
                return penalties_torch
        if not self.ttc_use_vehicle_size:
            self._warn_fallback(
                "ttc_point_agent_mode",
                "TTC fallback mode active: ttc_use_vehicle_size=false, so vehicle radii are ignored.",
            )

        if states_by_world is None:
            states_by_world = self._collect_all_vehicle_states()
        penalties = np.zeros((len(keys),), dtype=np.float32)
        env_step_dt = max(1e-6, float(self.physics_dt) * float(self.action_repeat))
        excluded_tokens = set(self._pending_respawns.keys()) | set(self._quarantined_tokens)

        for i, k in enumerate(keys):
            if not active[i]:
                continue
            token = self._agent_token_from_key(k)
            wi = int(getattr(k, "world_idx", -1))
            world_states = states_by_world.get(wi, None)
            if not world_states:
                self._prev_min_ttc_s.pop(token, None)
                continue
            ego_state = None
            for s in world_states:
                if s["agent_id"] == int(k.agent_id):
                    ego_state = s
                    break
            if ego_state is None:
                self._prev_min_ttc_s.pop(token, None)
                continue

            ego_centers = np.asarray(
                ego_state.get("ttc_centers_u", np.asarray([ego_state["pos"]], dtype=np.float32)),
                dtype=np.float32,
            )
            ego_radii = np.asarray(
                ego_state.get("ttc_radii_u", np.asarray([ego_state.get("radius_u", 0.0)], dtype=np.float32)),
                dtype=np.float32,
            )
            if ego_centers.ndim != 2 or ego_centers.shape[0] <= 0:
                ego_centers = np.asarray([ego_state["pos"]], dtype=np.float32)
            if ego_radii.ndim != 1 or ego_radii.shape[0] != ego_centers.shape[0]:
                ego_radii = np.full(
                    (ego_centers.shape[0],),
                    float(max(0.0, ego_state.get("radius_u", 0.0))),
                    dtype=np.float32,
                )

            ego_vel = np.asarray(ego_state["vel"], dtype=np.float32)[:2]
            ego_fwd = np.asarray(ego_state.get("fwd", (1.0, 0.0)), dtype=np.float32)[:2]
            fwd_norm = float(np.linalg.norm(ego_fwd))
            if fwd_norm <= 1e-6:
                ego_fwd = np.asarray([1.0, 0.0], dtype=np.float32)
            else:
                ego_fwd = ego_fwd / fwd_norm

            other_centers_list: List[np.ndarray] = []
            other_radii_list: List[np.ndarray] = []
            other_vel_list: List[np.ndarray] = []
            for s in world_states:
                if s["agent_id"] == int(k.agent_id):
                    continue
                if s.get("token", None) in excluded_tokens:
                    continue
                centers = np.asarray(
                    s.get("ttc_centers_u", np.asarray([s["pos"]], dtype=np.float32)),
                    dtype=np.float32,
                )
                if centers.ndim != 2 or centers.shape[0] <= 0:
                    centers = np.asarray([s["pos"]], dtype=np.float32)
                radii = np.asarray(
                    s.get("ttc_radii_u", np.asarray([s.get("radius_u", 0.0)], dtype=np.float32)),
                    dtype=np.float32,
                )
                if radii.ndim != 1 or radii.shape[0] != centers.shape[0]:
                    radii = np.full(
                        (centers.shape[0],),
                        float(max(0.0, s.get("radius_u", 0.0))),
                        dtype=np.float32,
                    )
                vel = np.asarray(s["vel"], dtype=np.float32)[:2]
                vel_rep = np.repeat(vel[None, :], centers.shape[0], axis=0)
                other_centers_list.append(centers)
                other_radii_list.append(radii)
                other_vel_list.append(vel_rep)

            if not other_centers_list:
                self._prev_min_ttc_s.pop(token, None)
                continue
            other_centers = np.concatenate(other_centers_list, axis=0)
            other_radii = np.concatenate(other_radii_list, axis=0)
            other_vel = np.concatenate(other_vel_list, axis=0)

            # Pairwise ego-circle vs target-circle geometry.
            # rel/rv shape: (E, O, 2), where E=ego circle count, O=all other circles.
            rel = other_centers[None, :, :] - ego_centers[:, None, :]
            rv = other_vel[None, :, :] - ego_vel[None, None, :]
            rx = rel[:, :, 0]
            ry = rel[:, :, 1]
            rvx = rv[:, :, 0]
            rvy = rv[:, :, 1]
            r2 = rx * rx + ry * ry
            v2 = rvx * rvx + rvy * rvy
            rdotv = rx * rvx + ry * rvy
            combined_r = ego_radii[:, None] + other_radii[None, :]
            combined_r2 = combined_r * combined_r

            # Forward-direction TTC: only consider objects ahead of ego heading.
            forward_dot = rx * float(ego_fwd[0]) + ry * float(ego_fwd[1])
            forward_mask = forward_dot > 0.0

            ttc = np.full(r2.shape, np.inf, dtype=np.float32)
            overlap_mask = forward_mask & (r2 <= combined_r2)
            ttc[overlap_mask] = 0.0

            moving_mask = forward_mask & (~overlap_mask) & (v2 > 1e-6)
            if np.any(moving_mask):
                a = np.where(moving_mask, v2, 1.0)
                b = np.where(moving_mask, 2.0 * rdotv, 0.0)
                c = np.where(moving_mask, r2 - combined_r2, 0.0)
                disc = b * b - 4.0 * a * c
                valid_quad = moving_mask & (disc >= 0.0)
                sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
                denom = 2.0 * np.maximum(a, 1e-6)
                t_enter = (-b - sqrt_disc) / denom
                t_exit = (-b + sqrt_disc) / denom
                valid_enter = valid_quad & (t_exit >= 0.0)
                ttc_quad = np.where(valid_enter, np.maximum(0.0, t_enter), np.inf).astype(np.float32)
                ttc = np.minimum(ttc, ttc_quad)

                unresolved = moving_mask & (~np.isfinite(ttc))
                if np.any(unresolved):
                    # Continuous fallback for closing pairs without usable quadratic root.
                    dist = np.sqrt(np.maximum(r2, 1e-9))
                    closing_speed = -rdotv / np.maximum(dist, 1e-6)
                    clearance = np.maximum(0.0, dist - combined_r)
                    valid_fb = unresolved & (rdotv < 0.0) & (closing_speed > 1e-6)
                    ttc_fb = np.where(
                        valid_fb,
                        clearance / np.maximum(closing_speed, 1e-6),
                        np.inf,
                    ).astype(np.float32)
                    ttc = np.minimum(ttc, ttc_fb)

            finite_ttc = ttc[np.isfinite(ttc)]
            if finite_ttc.size == 0:
                self._prev_min_ttc_s.pop(token, None)
                continue
            min_ttc = float(np.min(finite_ttc))

            abs_penalty = 0.0
            if self.ttc_penalty_enable:
                # Hard safety clamp: imminent risk (<0.5 s TTC) gets maximum penalty.
                if float(min_ttc) < 0.5:
                    abs_penalty = float(self.ttc_penalty_max)
                else:
                    denom = max(float(min_ttc), float(self.ttc_penalty_min_ttc))
                    abs_penalty = min(float(self.ttc_penalty_max), float(self.ttc_penalty_alpha) / denom)

            delta_penalty = 0.0
            if self.ttc_delta_penalty_enable and self.ttc_delta_penalty_alpha > 0.0:
                prev_min_ttc = self._prev_min_ttc_s.get(token, None)
                if prev_min_ttc is not None and np.isfinite(prev_min_ttc):
                    delta_ttc = float(prev_min_ttc) - float(min_ttc)
                    if self.ttc_delta_penalty_normalize_by_dt:
                        delta_ttc /= env_step_dt
                    if delta_ttc > 0.0:
                        delta_penalty = min(
                            float(self.ttc_delta_penalty_max),
                            float(self.ttc_delta_penalty_alpha) * float(delta_ttc),
                        )

            penalties[i] = -float(abs_penalty + delta_penalty)
            self._prev_min_ttc_s[token] = float(min_ttc)
        return penalties

    def _compute_road_edge_ttc_penalty_torch_cuda(
        self,
        keys: List[object],
        active: np.ndarray,
        *,
        states_by_world: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> Optional[np.ndarray]:
        torch, torch_dev = self._get_ttc_torch_cuda()
        if torch is None or torch_dev is None:
            return None

        forbidden_types = (
            set(int(x) for x in self.road_contact_done_types)
            if self.road_contact_done_types
            else set(int(x) for x in self.geom_road_edge_types)
        )
        if not forbidden_types:
            self._warn_fallback(
                "road_edge_ttc_missing_forbidden_types",
                "Road-edge TTC requested, but no forbidden road types are configured.",
            )
            return np.zeros((len(keys),), dtype=np.float32)

        radius_m = (
            float(self.road_edge_ttc_radius_m)
            if self.road_edge_ttc_radius_m is not None
            else float(self.road_points_radius_m)
        )
        if radius_m <= 0.0:
            self._warn_fallback(
                "road_edge_ttc_radius_nonpositive",
                "Road-edge TTC radius <= 0; skipping road-edge TTC penalty.",
            )
            return np.zeros((len(keys),), dtype=np.float32)

        if states_by_world is None:
            states_by_world = self._collect_all_vehicle_states()

        penalties = np.zeros((len(keys),), dtype=np.float32)
        radius2_m = radius_m * radius_m
        mpu = float(self._mpu)
        forbidden_type_list = list(forbidden_types)

        try:
            for i, k in enumerate(keys):
                if not active[i]:
                    continue

                wi = int(getattr(k, "world_idx", -1))
                world_states = states_by_world.get(wi, None)
                if not world_states:
                    continue
                ego_state = None
                for s in world_states:
                    if s.get("agent_id", None) == int(k.agent_id):
                        ego_state = s
                        break
                if ego_state is None:
                    continue

                geom = self._get_world_road_geometry(wi)
                if geom is None:
                    continue
                types = geom["types"]
                points_xy_m = geom["points_xy_m"]
                forbidden_mask = np.isin(types, forbidden_type_list)
                if not np.any(forbidden_mask):
                    self._warn_fallback(
                        "road_edge_ttc_no_points_forbidden_types",
                        "Road-edge TTC found no geometry points for configured forbidden types.",
                    )
                    continue
                forbidden_points = points_xy_m[forbidden_mask]
                if forbidden_points.shape[0] == 0:
                    continue

                ego_centers_u = np.asarray(
                    ego_state.get("ttc_centers_u", np.asarray([ego_state["pos"]], dtype=np.float32)),
                    dtype=np.float32,
                )
                ego_radii_u = np.asarray(
                    ego_state.get("ttc_radii_u", np.asarray([ego_state.get("radius_u", 0.0)], dtype=np.float32)),
                    dtype=np.float32,
                )
                if ego_centers_u.ndim != 2 or ego_centers_u.shape[0] <= 0:
                    ego_centers_u = np.asarray([ego_state["pos"]], dtype=np.float32)
                if ego_radii_u.ndim != 1 or ego_radii_u.shape[0] != ego_centers_u.shape[0]:
                    ego_radii_u = np.full(
                        (ego_centers_u.shape[0],),
                        float(max(0.0, ego_state.get("radius_u", 0.0))),
                        dtype=np.float32,
                    )

                ego_centers_m = ego_centers_u * float(mpu)
                ego_radii_m = ego_radii_u * float(mpu)
                evx_mps = float(ego_state["vel"][0]) * mpu
                evy_mps = float(ego_state["vel"][1]) * mpu
                ego_fwd_np = np.asarray(ego_state.get("fwd", (1.0, 0.0)), dtype=np.float32)[:2]
                fwd_norm = float(np.linalg.norm(ego_fwd_np))
                if fwd_norm <= 1e-6:
                    ego_fwd_np = np.asarray([1.0, 0.0], dtype=np.float32)
                else:
                    ego_fwd_np = ego_fwd_np / fwd_norm

                ego_centers_m_t = torch.as_tensor(ego_centers_m, dtype=torch.float32, device=torch_dev)
                ego_radii_m_t = torch.as_tensor(ego_radii_m, dtype=torch.float32, device=torch_dev)
                forbidden_points_t = torch.as_tensor(
                    forbidden_points,
                    dtype=torch.float32,
                    device=torch_dev,
                )
                ego_fwd_t = torch.as_tensor(ego_fwd_np, dtype=torch.float32, device=torch_dev)

                rel = forbidden_points_t.unsqueeze(0) - ego_centers_m_t.unsqueeze(1)
                rx = rel[:, :, 0]
                ry = rel[:, :, 1]
                dist2 = rx * rx + ry * ry
                near_mask = dist2 <= float(radius2_m)
                if not bool(torch.any(near_mask).item()):
                    continue

                forward_dot = rx * ego_fwd_t[0] + ry * ego_fwd_t[1]
                candidate_mask = near_mask & (forward_dot > 0.0)
                if not bool(torch.any(candidate_mask).item()):
                    continue

                dist = torch.sqrt(torch.clamp(dist2, min=1e-12))
                inf = torch.full_like(dist, float("inf"))
                ttc = inf.clone()
                overlap_mask = candidate_mask & (dist <= ego_radii_m_t.view(-1, 1))
                ttc = torch.where(overlap_mask, torch.zeros_like(ttc), ttc)

                dirs = rel / torch.clamp(dist.unsqueeze(-1), min=1e-6)
                closing_speed = dirs[:, :, 0] * float(evx_mps) + dirs[:, :, 1] * float(evy_mps)
                valid = candidate_mask & (closing_speed > 1e-6)
                if bool(torch.any(valid).item()):
                    clearance = torch.clamp(dist - ego_radii_m_t.view(-1, 1), min=0.0)
                    ttc_vals = torch.where(
                        valid,
                        clearance / torch.clamp(closing_speed, min=1e-6),
                        inf,
                    )
                    ttc = torch.minimum(ttc, ttc_vals)

                finite_mask = torch.isfinite(ttc)
                if not bool(torch.any(finite_mask).item()):
                    continue
                min_ttc = float(torch.min(ttc[finite_mask]).item())

                if float(min_ttc) < float(self.road_edge_ttc_hard_min_ttc):
                    abs_penalty = float(self.road_edge_ttc_penalty_max)
                else:
                    denom = max(float(min_ttc), float(self.road_edge_ttc_penalty_min_ttc))
                    abs_penalty = min(
                        float(self.road_edge_ttc_penalty_max),
                        float(self.road_edge_ttc_penalty_alpha) / denom,
                    )
                penalties[i] = -float(abs_penalty)
        except Exception as exc:
            self._warn_fallback(
                "ttc_backend_torch_road_runtime",
                f"torch_cuda road-edge TTC failed ({exc}); falling back to numpy TTC.",
            )
            return None

        return penalties

    def _compute_road_edge_ttc_penalty(
        self,
        keys: List[object],
        active: np.ndarray,
        states_by_world: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> np.ndarray:
        if not self.road_edge_ttc_penalty_enable:
            return np.zeros((len(keys),), dtype=np.float32)
        if self.road_edge_ttc_penalty_alpha <= 0.0 or self.road_edge_ttc_penalty_max <= 0.0:
            return np.zeros((len(keys),), dtype=np.float32)
        if self.ttc_backend == "torch_cuda":
            penalties_torch = self._compute_road_edge_ttc_penalty_torch_cuda(
                keys,
                active,
                states_by_world=states_by_world,
            )
            if penalties_torch is not None:
                return penalties_torch

        forbidden_types = (
            set(int(x) for x in self.road_contact_done_types)
            if self.road_contact_done_types
            else set(int(x) for x in self.geom_road_edge_types)
        )
        if not forbidden_types:
            self._warn_fallback(
                "road_edge_ttc_missing_forbidden_types",
                "Road-edge TTC requested, but no forbidden road types are configured.",
            )
            return np.zeros((len(keys),), dtype=np.float32)

        radius_m = (
            float(self.road_edge_ttc_radius_m)
            if self.road_edge_ttc_radius_m is not None
            else float(self.road_points_radius_m)
        )
        if radius_m <= 0.0:
            self._warn_fallback(
                "road_edge_ttc_radius_nonpositive",
                "Road-edge TTC radius <= 0; skipping road-edge TTC penalty.",
            )
            return np.zeros((len(keys),), dtype=np.float32)

        if states_by_world is None:
            states_by_world = self._collect_all_vehicle_states()

        penalties = np.zeros((len(keys),), dtype=np.float32)
        radius2_m = radius_m * radius_m
        mpu = float(self._mpu)
        forbidden_type_list = list(forbidden_types)

        for i, k in enumerate(keys):
            if not active[i]:
                continue

            wi = int(getattr(k, "world_idx", -1))
            world_states = states_by_world.get(wi, None)
            if not world_states:
                continue
            ego_state = None
            for s in world_states:
                if s.get("agent_id", None) == int(k.agent_id):
                    ego_state = s
                    break
            if ego_state is None:
                continue

            geom = self._get_world_road_geometry(wi)
            if geom is None:
                continue
            types = geom["types"]
            points_xy_m = geom["points_xy_m"]
            forbidden_mask = np.isin(types, forbidden_type_list)
            if not np.any(forbidden_mask):
                self._warn_fallback(
                    "road_edge_ttc_no_points_forbidden_types",
                    "Road-edge TTC found no geometry points for configured forbidden types.",
                )
                continue
            forbidden_points = points_xy_m[forbidden_mask]
            if forbidden_points.shape[0] == 0:
                continue

            ego_centers_u = np.asarray(
                ego_state.get("ttc_centers_u", np.asarray([ego_state["pos"]], dtype=np.float32)),
                dtype=np.float32,
            )
            ego_radii_u = np.asarray(
                ego_state.get("ttc_radii_u", np.asarray([ego_state.get("radius_u", 0.0)], dtype=np.float32)),
                dtype=np.float32,
            )
            if ego_centers_u.ndim != 2 or ego_centers_u.shape[0] <= 0:
                ego_centers_u = np.asarray([ego_state["pos"]], dtype=np.float32)
            if ego_radii_u.ndim != 1 or ego_radii_u.shape[0] != ego_centers_u.shape[0]:
                ego_radii_u = np.full(
                    (ego_centers_u.shape[0],),
                    float(max(0.0, ego_state.get("radius_u", 0.0))),
                    dtype=np.float32,
                )

            ego_centers_m = ego_centers_u * float(mpu)
            ego_radii_m = ego_radii_u * float(mpu)
            evx_mps = float(ego_state["vel"][0]) * mpu
            evy_mps = float(ego_state["vel"][1]) * mpu
            ego_fwd = np.asarray(ego_state.get("fwd", (1.0, 0.0)), dtype=np.float32)[:2]
            fwd_norm = float(np.linalg.norm(ego_fwd))
            if fwd_norm <= 1e-6:
                ego_fwd = np.asarray([1.0, 0.0], dtype=np.float32)
            else:
                ego_fwd = ego_fwd / fwd_norm

            # Pairwise ego circles vs forbidden road points.
            # rel shape: (E, P, 2), E=ego circle count, P=forbidden point count.
            rel = forbidden_points[None, :, :] - ego_centers_m[:, None, :]
            rx = rel[:, :, 0]
            ry = rel[:, :, 1]
            dist2 = rx * rx + ry * ry
            near_mask = dist2 <= radius2_m
            if not np.any(near_mask):
                continue

            forward_dot = rx * float(ego_fwd[0]) + ry * float(ego_fwd[1])
            forward_mask = forward_dot > 0.0
            candidate_mask = near_mask & forward_mask
            if not np.any(candidate_mask):
                continue

            dist = np.sqrt(np.maximum(dist2, 1e-12))
            ttc = np.full(dist.shape, np.inf, dtype=np.float32)
            overlap_mask = candidate_mask & (dist <= ego_radii_m[:, None])
            ttc[overlap_mask] = 0.0

            dirs = rel / np.maximum(dist[:, :, None], 1e-6)
            closing_speed = dirs[:, :, 0] * evx_mps + dirs[:, :, 1] * evy_mps
            valid = candidate_mask & (closing_speed > 1e-6)
            if np.any(valid):
                clearance = np.maximum(0.0, dist - ego_radii_m[:, None])
                ttc_vals = np.where(
                    valid,
                    clearance / np.maximum(closing_speed, 1e-6),
                    np.inf,
                ).astype(np.float32)
                ttc = np.minimum(ttc, ttc_vals)

            finite_ttc = ttc[np.isfinite(ttc)]
            if finite_ttc.size == 0:
                continue
            min_ttc = float(np.min(finite_ttc))

            if float(min_ttc) < float(self.road_edge_ttc_hard_min_ttc):
                abs_penalty = float(self.road_edge_ttc_penalty_max)
            else:
                denom = max(float(min_ttc), float(self.road_edge_ttc_penalty_min_ttc))
                abs_penalty = min(
                    float(self.road_edge_ttc_penalty_max),
                    float(self.road_edge_ttc_penalty_alpha) / denom,
                )
            penalties[i] = -float(abs_penalty)

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
        # PhysX sim interface name differs slightly across Isaac Sim builds,
        # so we try common variants.
        sim_iface = None
        if hasattr(omni.physx, "get_physx_simulation_interface"):
            sim_iface = omni.physx.get_physx_simulation_interface()
        elif hasattr(omni.physx, "get_physx_interface"):
            sim_iface = omni.physx.get_physx_interface()
            self._warn_fallback(
                "physx_legacy_interface",
                "Using legacy omni.physx.get_physx_interface() fallback for teleport.",
            )
        if sim_iface is None:
            raise RuntimeError("No omni.physx simulation interface found.")
        px, py, pz = pos_units
        w, x, y, z = quat_wxyz

        # build Gf types (float versions)
        p = Gf.Vec3f(float(px), float(py), float(pz))
        q = Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))
        # --- pose setters (try a few common names) ---
        pose_setters = [
            ("set_rigid_body_pose", False),
            ("setRigidBodyPose", True),
            ("set_rigid_body_global_pose", True),
            ("setRigidBodyGlobalPose", True),
        ]
        ok = False
        for fn, is_fallback in pose_setters:
            if hasattr(sim_iface, fn):
                getattr(sim_iface, fn)(rb_path, p, q)
                if is_fallback:
                    self._warn_fallback(
                        f"physx_pose_setter::{fn}",
                        f"Using fallback PhysX pose setter '{fn}'.",
                    )
                ok = True
                break
        if not ok:
            cand = [m for m in dir(sim_iface) if ("rigid" in m.lower() and "pose" in m.lower())]
            raise RuntimeError(f"Couldn't find pose setter on physx iface. Candidates: {cand[:30]}")
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
                self._warn_fallback(
                    f"physx_velocity_setter::{b}",
                    f"Using fallback PhysX velocity setter '{b}'.",
                )


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
            road_points_include_dirs=self.road_points_include_dirs,
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
            feat = int(self.road_point_feat_dim)
            for j in range(k):
                off = base + feat * j
                rx = float(obs_vec[off + 0]) * radius
                ry = float(obs_vec[off + 1]) * radius
                t = float(obs_vec[off + 2])
                dir_x = float(obs_vec[off + 3]) if feat >= 5 else 0.0
                dir_y = float(obs_vec[off + 4]) if feat >= 5 else 0.0
                if rx == 0.0 and ry == 0.0 and t == 0.0:
                    continue
                s = UsdGeom.Sphere.Define(stage, f"{agent_root}/RoadPoints/P{j:03d}")
                s.GetRadiusAttr().Set(float(0.25 / mpu))
                sx = UsdGeom.XformCommonAPI(s)
                sx.SetTranslate(Gf.Vec3d(rx / mpu, ry / mpu, 0.2 / mpu))
                color = Gf.Vec3f(0.2, 0.4 + 0.6 * max(0.0, min(1.0, t)), 1.0)
                UsdGeom.Gprim(s.GetPrim()).CreateDisplayColorAttr().Set([color])
                if abs(dir_x) > 1e-4 or abs(dir_y) > 1e-4:
                    a = UsdGeom.Cube.Define(stage, f"{agent_root}/RoadPoints/P{j:03d}_Dir")
                    a.GetSizeAttr().Set(1.0)
                    ax = UsdGeom.XformCommonAPI(a)
                    yaw_deg = math.degrees(math.atan2(dir_y, dir_x))
                    ax.SetTranslate(Gf.Vec3d(rx / mpu, ry / mpu, 0.35 / mpu))
                    ax.SetRotate(
                        Gf.Vec3f(0.0, 0.0, float(yaw_deg)),
                        UsdGeom.XformCommonAPI.RotationOrderXYZ,
                    )
                    ax.SetScale(Gf.Vec3f(1.2 / mpu, 0.08 / mpu, 0.08 / mpu))
                    UsdGeom.Gprim(a.GetPrim()).CreateDisplayColorAttr().Set([color])

        # Vehicle observations visualization
        if self.vehicle_obs_enable:
            veh_root = UsdGeom.Xform.Define(stage, f"{agent_root}/Vehicles")
            base = 7 + weather_context_dim() + (
                int(self.road_points_k) * int(self.road_point_feat_dim)
                if self.road_points_enable
                else 0
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

    def is_agent_quarantined(self, key: object) -> bool:
        return self._agent_token_from_key(key) in self._quarantined_tokens

    def quarantine_agents(self, tokens: List[Tuple[int, int]]) -> None:
        if not tokens:
            return
        token_set = {self._agent_token(world_idx, agent_id) for world_idx, agent_id in tokens}
        for token in token_set:
            if token in self._quarantined_tokens:
                continue
            self._quarantined_tokens.add(token)
            self._pending_respawns.pop(token, None)
            self._prev_min_ttc_s.pop(token, None)
            world_idx, agent_id = token
            h = self._get_agent_handle(world_idx, agent_id)
            if h is not None:
                self._reset_agent_contact_state(h)
                self._set_agent_collision_enabled(h, False)
                self._set_agent_visible(h, False)
            for key in self._keys:
                if self._agent_token_from_key(key) != token:
                    continue
                self._move_agent_to_respawn_hold(key)
                break

        if self._done.shape[0] == len(self._keys):
            for i, key in enumerate(self._keys):
                if self._agent_token_from_key(key) in token_set:
                    self._done[i] = False
                    self._success_latched[i] = False

    def collect_startup_below_min_z_offenders(self, steps: int) -> List[Tuple[int, int]]:
        if steps <= 0 or self.min_vehicle_z_m is None or not self._keys:
            return []

        offenders: Set[Tuple[int, int]] = set()

        for _ in range(int(steps)):
            n_keys = int(len(self._keys))
            if n_keys <= 0:
                break
            zero_actions = np.zeros((n_keys, 2), dtype=np.float32)
            _, _, done, info = self.step(zero_actions)
            for key, flag in zip(info.keys, info.below_min_z):
                if bool(flag):
                    offenders.add(self._agent_token_from_key(key))

            if getattr(info, "timeout", False):
                self.reset_timeout()
            elif np.any(done):
                self.reset_done(done)

        return sorted(offenders)

    def _agent_token(self, world_idx: int, agent_id: int) -> Tuple[int, int]:
        return (int(world_idx), int(agent_id))

    def _agent_token_from_key(self, key: object) -> Tuple[int, int]:
        return self._agent_token(getattr(key, "world_idx"), getattr(key, "agent_id"))

    def _clear_ttc_history_for_tokens(self, tokens: List[Tuple[int, int]]) -> None:
        if not tokens or not self._prev_min_ttc_s:
            return
        for token in tokens:
            self._prev_min_ttc_s.pop(self._agent_token(token[0], token[1]), None)

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
                try:
                    M = UsdGeom.Xformable(rb_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    p = M.ExtractTranslation()
                    q = M.ExtractRotationQuat()
                    q_im = q.GetImaginary()
                    self._physx_teleport_rb(
                        rb_prim,
                        (float(p[0]), float(p[1]), float(p[2])),
                        (
                            float(q.GetReal()),
                            float(q_im[0]),
                            float(q_im[1]),
                            float(q_im[2]),
                        ),
                    )
                except Exception as exc:
                    self._warn_fallback(
                        "respawn_physx_teleport_failed",
                        f"PhysX teleport failed during respawn pose set ({exc}); using USD pose + velocity-zero fallback.",
                    )
                    _zero_rb_vel(rb_prim)
        return True

    def _get_agent_world_xy_m(self, h) -> Optional[Tuple[float, float]]:
        pose = self._get_agent_world_pose(h)
        if pose is None:
            return None
        return (float(pose[0]), float(pose[1]))

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
            if self.respawn_clear_ignore_non_controllable and (not bool(state.get("controllable", False))):
                continue
            if (
                other_token == token
                or other_token in self._pending_respawns
                or other_token in self._quarantined_tokens
            ):
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
            "queued_step": int(self.t),
        }
        return True

    def _release_pending_respawns(self) -> None:
        if not self._pending_respawns:
            return

        released: List[Tuple[int, int]] = []
        for token, pending in list(self._pending_respawns.items()):
            if token in self._quarantined_tokens:
                continue
            radius_m = float(pending.get("radius_m", self.respawn_hold_radius_m))
            is_clear = self._spawn_area_is_clear(token, radius_m)
            queued_step = int(pending.get("queued_step", self.t))
            waited_steps = max(0, int(self.t - queued_step))
            force_release = (
                self.respawn_release_max_wait_steps > 0
                and waited_steps >= self.respawn_release_max_wait_steps
            )
            if not is_clear and not force_release:
                continue
            if force_release and not is_clear:
                self._warn_fallback(
                    "respawn_force_release",
                    "Respawn release timeout reached; forcing release despite blocked spawn area. "
                    f"world={token[0]} agent={token[1]} waited_steps={waited_steps}",
                )
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
        if not self._pending_respawns and not self._quarantined_tokens:
            return np.zeros((len(keys),), dtype=bool)
        return np.asarray(
            [
                (self._agent_token_from_key(k) in self._pending_respawns)
                or (self._agent_token_from_key(k) in self._quarantined_tokens)
                for k in keys
            ],
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

    def _remove_agents(self, keys: List[object], which: np.ndarray) -> bool:
        idx = np.where(which)[0]
        if idx.size == 0:
            return False

        removed_any = False
        removed_tokens: List[Tuple[int, int]] = []
        for i in idx:
            k = keys[i]
            token = self._agent_token_from_key(k)

            world_root = f"{self.root_container}/{self.world_prefix}{int(k.world_idx):03d}"
            goals_root = self.stage.GetPrimAtPath(f"{world_root}/Goals")
            if goals_root.IsValid():
                goal_suffix = f"_id{int(k.agent_id)}"
                for goal_prim in list(goals_root.GetAllChildren()):
                    goal_name = goal_prim.GetName()
                    goal_path = goal_prim.GetPath().pathString
                    if goal_name.endswith(goal_suffix) or goal_suffix in goal_name or goal_suffix in goal_path:
                        try:
                            self.stage.RemovePrim(goal_prim.GetPath())
                            removed_any = True
                        except Exception:
                            pass

            agent_path = self._find_agent_prim_path(int(k.world_idx), int(k.agent_id))
            if agent_path is not None:
                try:
                    self.stage.RemovePrim(agent_path)
                    removed_any = True
                    removed_tokens.append(token)
                except Exception:
                    pass

        for token in removed_tokens:
            self._pending_respawns.pop(token, None)
            self._quarantined_tokens.discard(token)
            self._prev_world_xy_m.pop(token, None)
            self._prev_min_ttc_s.pop(token, None)
            self._vehicle_radius_u_cache.pop(token, None)
            self._vehicle_shape_u_cache.pop(token, None)

        if removed_any:
            try:
                self.ctrl.refresh()
            except Exception:
                pass
            # One settle step helps PhysX drop removed actors before next obs build.
            try:
                self.sim.step(render=False)
            except Exception:
                pass
        return bool(removed_any)

    def _sync_keyed_state_from_stage(self) -> None:
        """
        Refresh internal keyed arrays/maps after hard-removing agents from stage.
        Keeps remaining-agent bookkeeping aligned for the next env.step call.
        """
        old_keys = list(self._keys)
        old_done = (
            self._done.copy()
            if isinstance(self._done, np.ndarray)
            else np.zeros((len(old_keys),), dtype=bool)
        )
        old_success = (
            self._success_latched.copy()
            if isinstance(self._success_latched, np.ndarray)
            else np.zeros((len(old_keys),), dtype=bool)
        )
        old_prev_dist = (
            self._prev_dist_m.copy()
            if isinstance(self._prev_dist_m, np.ndarray)
            else np.zeros((len(old_keys),), dtype=np.float32)
        )
        old_idx_by_token = {
            self._agent_token_from_key(k): i
            for i, k in enumerate(old_keys)
        }

        obs2, mask2, keys2 = self._build_obs()
        dist_n2 = obs2[:, 4].astype(np.float32) if obs2.size else np.zeros((0,), dtype=np.float32)
        dist_m2 = dist_n2 * (self.bounds_size_m * math.sqrt(2.0))

        self._keys = keys2
        self._mask = mask2.copy()

        N2 = len(keys2)
        new_done = np.zeros((N2,), dtype=bool)
        new_success = np.zeros((N2,), dtype=bool)
        new_prev_dist = dist_m2.copy()

        for j, k in enumerate(keys2):
            old_idx = old_idx_by_token.get(self._agent_token_from_key(k), None)
            if old_idx is None:
                continue
            if 0 <= old_idx < old_done.shape[0]:
                new_done[j] = bool(old_done[old_idx])
            if 0 <= old_idx < old_success.shape[0]:
                new_success[j] = bool(old_success[old_idx])
            if 0 <= old_idx < old_prev_dist.shape[0]:
                new_prev_dist[j] = float(old_prev_dist[old_idx])

        self._done = new_done
        self._success_latched = new_success
        self._prev_dist_m = new_prev_dist

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
        material_path = self._world_ground_material_path(world_idx)
        self._respawn_rebuild_count += 1
        should_log_snapshot = (not self._respawn_friction_debug_logged) or (
            self.respawn_friction_debug_every > 0
            and (self._respawn_rebuild_count % self.respawn_friction_debug_every) == 0
        )
        if should_log_snapshot:
            self._log_shared_friction_table_snapshot(
                tag=f"before_rebuild_respawn_{self._respawn_rebuild_count}",
                material_path=material_path,
            )
        bounds = LocalBounds(
            width_m=float(self.bounds_size_m),
            length_m=float(self.bounds_size_m),
            origin_xy=(0.0, 0.0),
        )
        shared_before: Dict[str, str] = {}
        if not self._respawn_shared_snapshot_debug_logged:
            shared_before = self._collect_prim_attr_snapshot("/World/VehicleShared")

        builder = WaymoJsonMiniWorldBuilder(
            stage=self.stage,
            world_root=world_root,
            bounds=bounds,
            origin_mode="center",
        )

        goal_path = f"{world_root}/Goals/Goal_{kept_idx:04d}_id{int(agent_id)}"
        self.stage.RemovePrim(goal_path)
        self.stage.RemovePrim(agent_path)
        for _ in range(self.respawn_rebuild_flush_steps_before_create):
            self.sim.step(render=False)

        builder.respawn_agent_with_goal(
            kept_idx=int(kept_idx),
            agent_id=int(agent_id),
            start_local_m=(float(start_local[0]), float(start_local[1]), float(start_local[2])),
            start_yaw_deg=float(start_yaw_deg),
            goal_local_m=(float(goal_local[0]), float(goal_local[1]), float(goal_local[2])),
            start_in_goal=bool(start_in_goal),
            spawn_z_m=float(self.respawn_params.get("spawn_z_m", self.respawn_spawn_z_m)),
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
        for _ in range(self.respawn_rebuild_flush_steps_after_create):
            self.sim.step(render=False)

        if not self._reapply_world_friction_patch(int(world_idx)):
            self._warn_fallback(
                "respawn_friction_patch_missing",
                f"Rebuild respawn finished without confirmed friction re-patch world={int(world_idx)} agent={int(agent_id)}.",
            )

        if not self._respawn_shared_snapshot_debug_logged:
            shared_after = self._collect_prim_attr_snapshot("/World/VehicleShared")
            self._log_snapshot_diff(
                tag="vehicle_shared_first_rebuild_respawn",
                before=shared_before,
                after=shared_after,
            )
            self._respawn_shared_snapshot_debug_logged = True

        if should_log_snapshot:
            self._log_shared_friction_table_snapshot(
                tag=f"after_rebuild_respawn_{self._respawn_rebuild_count}",
                material_path=material_path,
            )
            if not self._respawn_friction_debug_logged:
                self._respawn_friction_debug_logged = True

        return True

    def reset_done(self, done_mask: np.ndarray) -> None:
        if done_mask is None:
            return

        done_mask = np.asarray(done_mask, dtype=bool).reshape(-1)
        if done_mask.size == 0 or not np.any(done_mask):
            return

        n_keys = int(len(self._keys))
        if n_keys <= 0:
            return
        if done_mask.shape[0] != n_keys:
            common = min(int(done_mask.shape[0]), n_keys)
            self._warn_fallback(
                "reset_done_mask_size_mismatch",
                f"reset_done got done_mask size={int(done_mask.shape[0])} but keys={n_keys}; "
                f"using common prefix size={common}.",
            )
            if common <= 0:
                return
            done_mask = done_mask[:common]
            if not np.any(done_mask):
                return

        idx = np.where(done_mask)[0]
        if idx.size == 0:
            return
        done_tokens = [self._agent_token_from_key(self._keys[i]) for i in idx]
        self._clear_ttc_history_for_tokens(done_tokens)
        key_by_token = {self._agent_token_from_key(self._keys[i]): self._keys[i] for i in idx}

        if self.respawn_on_reset:
            used_builder_respawn = False
            builder_respawn_keys: List[object] = []
            for token in done_tokens:
                k = key_by_token.get(token)
                if k is None:
                    continue
                world_idx, agent_id = token
                if self.respawn_mode == "rebuild":
                    if self._respawn_agent_from_metadata(world_idx, agent_id):
                        builder_respawn_keys.append(k)
                        used_builder_respawn = True
                        self._pending_respawns.pop(token, None)
                    else:
                        if self._queue_respawn(k):
                            continue
                        if self._respawn_agent_from_metadata(world_idx, agent_id):
                            builder_respawn_keys.append(k)
                            used_builder_respawn = True
                else:
                    if not self._queue_respawn(k):
                        if self._respawn_agent_from_metadata(world_idx, agent_id):
                            builder_respawn_keys.append(k)
                            used_builder_respawn = True
            if used_builder_respawn:
                self.ctrl.refresh()
                for k in builder_respawn_keys:
                    self._start_local_translate.pop(k, None)
                    self._start_local_yaw_deg.pop(k, None)
                    self._start_world_xy_m.pop(k, None)
                    h = self.ctrl.get(k.world_idx, k.agent_id)
                    if h is not None:
                        self._cache_spawn_pose(k, h)
            self._release_pending_respawns()
            self.sim.step(render=False)
            obs, mask, keys2 = self._build_obs()
            self._keys = keys2
            dist_n = obs[:, 4].astype(np.float32)
            dist_m = dist_n * (self.bounds_size_m * math.sqrt(2.0))
            self._mask = mask.copy()
            if self._done.shape[0] != len(self._keys):
                self._done = np.zeros((len(self._keys),), dtype=bool)
                self._success_latched = np.zeros((len(self._keys),), dtype=bool)
                self._prev_dist_m = dist_m.copy()
            token_to_new_idx = {
                self._agent_token_from_key(k): i for i, k in enumerate(self._keys)
            }
            reset_idx = [token_to_new_idx[tok] for tok in done_tokens if tok in token_to_new_idx]
            if reset_idx:
                reset_idx_arr = np.asarray(reset_idx, dtype=np.int64)
                self._done[reset_idx_arr] = False
                self._success_latched[reset_idx_arr] = False
                self._prev_dist_m[reset_idx_arr] = dist_m[reset_idx_arr]
            for tok in done_tokens:
                j = token_to_new_idx.get(tok, None)
                if j is None:
                    continue
                key = self._keys[j]
                h = self.ctrl.get(key.world_idx, key.agent_id)
                xy_m = self._get_agent_world_xy_m(h)
                if xy_m is not None:
                    self._prev_world_xy_m[tok] = xy_m
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
        for i in idx:
            token = self._agent_token_from_key(self._keys[i])
            h = self.ctrl.get(self._keys[i].world_idx, self._keys[i].agent_id)
            xy_m = self._get_agent_world_xy_m(h)
            if xy_m is not None:
                self._prev_world_xy_m[token] = xy_m

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
        self._prev_world_xy_m.clear()
        self._prev_min_ttc_s.clear()

        # Refresh controller registry
        self.ctrl.refresh()
        keys = self.ctrl.keys()
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
        # init per-agent episode state
        self._done = np.zeros((N,), dtype=bool)
        self._success_latched = np.zeros((N,), dtype=bool)

        dist_n = obs[:, 4].astype(np.float32)
        dist_m = dist_n * (self.bounds_size_m * math.sqrt(2.0))
        self._prev_dist_m = dist_m.copy()
        for key in keys:
            h = self.ctrl.get(key.world_idx, key.agent_id)
            xy_m = self._get_agent_world_xy_m(h)
            if xy_m is not None:
                self._prev_world_xy_m[self._agent_token_from_key(key)] = xy_m


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
            self._prev_dist_m = (
                obs[:, 4].astype(np.float32) * (self.bounds_size_m * math.sqrt(2.0))
            )
            self._prev_world_xy_m.clear()
            self._prev_min_ttc_s.clear()
            for key in keys2:
                h = self.ctrl.get(key.world_idx, key.agent_id)
                xy_m = self._get_agent_world_xy_m(h)
                if xy_m is not None:
                    self._prev_world_xy_m[self._agent_token_from_key(key)] = xy_m
            keys = keys2

        dist_n = obs[:, 4].astype(np.float32)  # normalized
        dist_m = dist_n * (self.bounds_size_m * math.sqrt(2.0))

        pending_mask = self._pending_mask_for_keys(keys)
        active = mask & (~self._done) & (~pending_mask)
        removed_agents_this_step = False

        # SUCCESS: distance threshold in meters
        success_now = (dist_m <= self.goal_success_dist_m) & active
        newly_success = success_now & (~self._success_latched)

        if newly_success.any():
            self._success_latched[newly_success] = True
            self._done[newly_success] = True
            self._freeze_agents(keys, newly_success)
            if self.clear_on_done:
                if self.respawn_on_reset or (not self.hard_remove_done_agents):
                    self._hide_agents(keys, newly_success)
                else:
                    removed_agents_this_step = (
                        self._remove_agents(keys, newly_success) or removed_agents_this_step
                    )
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

        # Dense TTC penalties (vehicle-to-vehicle and optional vehicle-to-forbidden-road-edge).
        states_by_world = None
        if (
            self.ttc_penalty_enable
            or self.ttc_delta_penalty_enable
            or self.road_edge_ttc_penalty_enable
        ):
            states_by_world = self._collect_all_vehicle_states()
        if self.ttc_penalty_enable or self.ttc_delta_penalty_enable:
            ttc_pen = self._compute_ttc_penalty(keys, active, states_by_world=states_by_world)
            reward[active] += ttc_pen[active]
        if self.road_edge_ttc_penalty_enable:
            edge_ttc_pen = self._compute_road_edge_ttc_penalty(
                keys, active, states_by_world=states_by_world
            )
            reward[active] += edge_ttc_pen[active]
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

        road_contact_done = np.zeros((N,), dtype=bool)
        # Road-contact termination based on trigger contact list
        if self.road_contact_done_types:
            for i, k in enumerate(keys):
                if not active[i]:
                    continue
                h = self.ctrl.get(k.world_idx, k.agent_id)
                if h is None:
                    continue
                contact_types = self._get_contact_types(h)
                if any(t in self.road_contact_done_types for t in contact_types):
                    road_contact_done[i] = True
            if road_contact_done.any():
                reward[road_contact_done] += float(self.road_contact_done_penalty)
                self._done[road_contact_done] = True

        # Lane-center per-step reward (contact-type proxy, legacy path)
        lane_hit_contact = np.zeros((N,), dtype=bool)
        if self.lane_center_reward_enable:
            for i, k in enumerate(keys):
                if not active[i]:
                    continue
                h = self.ctrl.get(k.world_idx, k.agent_id)
                if h is None:
                    continue
                contact_types = self._get_contact_types(h)
                if any(t in self.lane_center_reward_types for t in contact_types):
                    lane_hit_contact[i] = True
            if lane_hit_contact.any():
                reward[lane_hit_contact] += float(self.lane_center_reward_per_step)

        geom_lane_hit, geom_off_road, lane_error_m, heading_alignment, route_progress_m = (
            self._compute_geometric_lane_features(keys, obs, active)
        )
        if self.geom_lane_reward_enable:
            lane_quality = np.exp(
                -np.square(lane_error_m / max(self.geom_lane_tolerance_m, 1e-3))
            ).astype(np.float32)
            lane_quality *= (
                (1.0 - self.geom_lane_heading_weight)
                + self.geom_lane_heading_weight * heading_alignment
            ).astype(np.float32)
            reward[active] += float(self.geom_lane_reward_per_step) * lane_quality[active]
        if self.geom_route_progress_weight != 0.0:
            reward[active] += float(self.geom_route_progress_weight) * np.clip(
                route_progress_m[active],
                -2.0,
                2.0,
            )

        vehicle_contact_done = np.zeros((N,), dtype=bool)
        # Vehicle-trigger termination based on vehicle contact list
        if self.vehicle_contact_done:
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
                vehicle_contact_done[i] = True
                if self.vehicle_contact_done_mark_both:
                    for other_id in contact_ids:
                        j = agent_id_to_idx.get(int(other_id))
                        if j is not None:
                            vehicle_contact_done[j] = True
            if vehicle_contact_done.any():
                reward[vehicle_contact_done] += float(self.vehicle_contact_done_penalty)
                self._done[vehicle_contact_done] = True

        # Apply terminal handling for non-success failures as soon as they become done.
        # This keeps behavior consistent with success terminals when clear_on_done is enabled.
        terminal_failure_now = below_min_z | road_contact_done | vehicle_contact_done
        if terminal_failure_now.any():
            self._freeze_agents(keys, terminal_failure_now)
            if self.clear_on_done:
                if self.respawn_on_reset or (not self.hard_remove_done_agents):
                    self._hide_agents(keys, terminal_failure_now)
                else:
                    removed_agents_this_step = (
                        self._remove_agents(keys, terminal_failure_now)
                        or removed_agents_this_step
                    )

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
        if self.verbose and self.t % 10 == 0:
            print(f"[env] t={self.t} timeout={timeout} done_any={done.any()} active={active.sum()}")

        # update prev dist for next step
        self._prev_dist_m = dist_m.copy()
        self._mask = mask.copy()

        lane_hit = lane_hit_contact.copy()
        off_road = np.zeros((N,), dtype=bool)
        if self.geom_lane_reward_enable or self.geom_offroad_metrics_enable or self.geom_route_progress_weight != 0.0:
            lane_hit = geom_lane_hit.copy()
            if self.geom_offroad_metrics_enable:
                off_road = geom_off_road.copy()
        elif self.lane_center_reward_enable:
            lane_hit = lane_hit_contact.copy()
            off_road = active & (~lane_hit_contact)

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
            newly_success=newly_success.copy(),
            road_contact_done=road_contact_done.copy(),
            vehicle_contact_done=vehicle_contact_done.copy(),
            collided=collided_flags.copy(),
            road_collided=road_collided.copy(),
            vehicle_collided=vehicle_collided.copy(),
            below_min_z=below_min_z.copy(),
            off_road=off_road,
            lane_hit=lane_hit.copy(),
            lane_error_m=lane_error_m.copy(),
            heading_alignment=heading_alignment.copy(),
            route_progress_m=route_progress_m.copy(),
            active=active.copy(),
            pending=pending_mask.copy(),
            timeout=bool(timeout),
            t_env=int(self.t),
        )
        if removed_agents_this_step and self.hard_remove_done_agents and (not self.respawn_on_reset):
            self._sync_keyed_state_from_stage()
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
            elif hasattr(omni.physx, "get_physx_interface"):
                sim_iface = omni.physx.get_physx_interface()
                print(
                    "[warn][fallback] RoadCollisionTracker using legacy "
                    "omni.physx.get_physx_interface() fallback."
                )
            else:
                sim_iface = None

            if sim_iface is not None and hasattr(sim_iface, "subscribe_contact_report_events"):
                self._sub = sim_iface.subscribe_contact_report_events(self._on_contact)
            if sim_iface is not None and hasattr(sim_iface, "subscribe_trigger_report_events"):
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
