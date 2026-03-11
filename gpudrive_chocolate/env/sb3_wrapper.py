from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn

from gpudrive_chocolate.env.choco_world import ChocoWorldBuilder, load_config
from gpudrive_chocolate.utils.obs_vis import save_obs_png


def _safe_mean(values) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    try:
        return float(np.nanmean(arr))
    except Exception:
        return 0.0


class ChocolateSB3MultiAgentEnv(VecEnv):
    """SB3 VecEnv wrapper for ChocolateEnv.

    This mirrors gpudrive's SB3MultiAgentEnv design:
    - Flatten controlled agents across worlds
    - Track dead agents and return NaN rewards/obs after done
    """

    def __init__(
        self,
        *,
        choco_config_path: str,
        exp_config,
        device: str = "cuda",
        reward_type: str = "weighted_combination",
        collision_weight: float = -0.5,
        goal_achieved_weight: float = 1.0,
        off_road_weight: float = -0.5,
        log_distance_weight: float = 0.01,
    ):
        self.device = device
        self.exp_config = exp_config

        self.choco_cfg = load_config(choco_config_path)
        self.builder = ChocoWorldBuilder(self.choco_cfg)
        ctx = self.builder.start()

        wcfg = self.choco_cfg["world"]
        cfg_env = self.choco_cfg.get("env", {})

        # Import after SimulationApp is started to avoid pxr import issues.
        from src.chocolate_env import ChocolateEnv

        agents_cfg = self.choco_cfg.get("agents", {})
        min_vehicle_z_cfg = cfg_env.get("min_vehicle_z_m", None)
        self.choco_env = ChocolateEnv(
            sim=ctx["sim"],
            stage=ctx["stage"],
            ctrl=ctx["ctrl"],
            obs_builder=ctx["obs_builder"],
            bounds_size_m=float(wcfg["bounds_size_m"]),
            physics_dt=float(ctx["physics_dt"]),
            action_repeat=int(ctx["action_repeat"]),
            max_steps=int(cfg_env.get("max_steps", 300)),
            clear_on_done=bool(cfg_env.get("clear_on_done", False)),
            hard_remove_done_agents=bool(cfg_env.get("hard_remove_done_agents", False)),
            goal_success_dist_m=float(cfg_env.get("goal_success_dist_m", 2.0)),
            reward_scale=float(cfg_env.get("reward_scale", 1.0)),
            success_bonus=float(cfg_env.get("success_bonus", 10.0)),
            action_l2_penalty=float(cfg_env.get("action_l2_penalty", 1e-3)),
            collision_penalty=float(cfg_env.get("collision_penalty", 0.0)),
            min_vehicle_z_m=None if min_vehicle_z_cfg is None else float(min_vehicle_z_cfg),
            collision_penalty_types=list(cfg_env.get("collision_penalty_types", [])),
            collision_debug=bool(cfg_env.get("collision_debug", False)),
            road_contact_done_types=list(cfg_env.get("road_contact_done_types", [])),
            road_contact_done_penalty=float(cfg_env.get("road_contact_done_penalty", -1.0)),
            lane_center_reward_enable=bool(cfg_env.get("lane_center_reward_enable", False)),
            lane_center_reward_type=cfg_env.get("lane_center_reward_type", 2),
            lane_center_reward_per_step=float(cfg_env.get("lane_center_reward_per_step", 0.05)),
            geom_lane_reward_enable=bool(cfg_env.get("geom_lane_reward_enable", False)),
            geom_lane_reward_per_step=float(cfg_env.get("geom_lane_reward_per_step", 0.0)),
            geom_lane_tolerance_m=float(cfg_env.get("geom_lane_tolerance_m", 1.75)),
            geom_lane_heading_weight=float(cfg_env.get("geom_lane_heading_weight", 0.5)),
            geom_lane_min_alignment=float(cfg_env.get("geom_lane_min_alignment", 0.5)),
            geom_route_progress_weight=float(cfg_env.get("geom_route_progress_weight", 0.0)),
            geom_offroad_metrics_enable=bool(cfg_env.get("geom_offroad_metrics_enable", False)),
            geom_offroad_lateral_threshold_m=float(
                cfg_env.get("geom_offroad_lateral_threshold_m", 3.0)
            ),
            geom_offroad_distance_threshold_m=float(
                cfg_env.get("geom_offroad_distance_threshold_m", 6.0)
            ),
            geom_lane_types=list(cfg_env.get("geom_lane_types", [1, 2])),
            geom_road_edge_types=list(cfg_env.get("geom_road_edge_types", [15, 16])),
            survival_reward_per_step=float(cfg_env.get("survival_reward_per_step", 0.0)),
            idle_penalty_enable=bool(cfg_env.get("idle_penalty_enable", False)),
            idle_penalty_per_step=float(cfg_env.get("idle_penalty_per_step", 0.05)),
            idle_speed_threshold_mps=float(cfg_env.get("idle_speed_threshold_mps", 0.5)),
            vehicle_contact_done=bool(cfg_env.get("vehicle_contact_done", False)),
            vehicle_contact_done_penalty=float(cfg_env.get("vehicle_contact_done_penalty", -5.0)),
            vehicle_contact_done_mark_both=bool(cfg_env.get("vehicle_contact_done_mark_both", True)),
            road_contact_debug=bool(cfg_env.get("road_contact_debug", False)),
            road_contact_debug_every=int(cfg_env.get("road_contact_debug_every", 100)),
            road_points_enable=bool(cfg_env["road_points_enable"]),
            road_points_k=int(cfg_env["road_points_k"]),
            road_points_radius_m=float(cfg_env["road_points_radius_m"]),
            road_points_type_norm=float(cfg_env["road_points_type_norm"]),
            road_points_mode=str(cfg_env.get("road_points_mode", "knn")),
            road_points_include_dirs=bool(cfg_env.get("road_points_include_dirs", False)),
            vehicle_obs_enable=bool(cfg_env["vehicle_obs_enable"]),
            vehicle_obs_k=int(cfg_env["vehicle_obs_k"]),
            ttc_penalty_enable=bool(cfg_env.get("ttc_penalty_enable", False)),
            ttc_penalty_alpha=float(cfg_env.get("ttc_penalty_alpha", 1.0)),
            ttc_penalty_max=float(cfg_env.get("ttc_penalty_max", 1.0)),
            ttc_penalty_min_ttc=float(cfg_env.get("ttc_penalty_min_ttc", 0.2)),
            ttc_penalty_function=str(cfg_env.get("ttc_penalty_function", "inverse")),
            ttc_proximity_zuo_a=float(cfg_env.get("ttc_proximity_zuo_a", 0.5)),
            ttc_proximity_zuo_b=float(cfg_env.get("ttc_proximity_zuo_b", 5.0)),
            road_edge_ttc_penalty_enable=bool(
                cfg_env.get("road_edge_ttc_penalty_enable", False)
            ),
            road_edge_ttc_penalty_alpha=float(
                cfg_env.get("road_edge_ttc_penalty_alpha", 0.0)
            ),
            road_edge_ttc_penalty_max=float(cfg_env.get("road_edge_ttc_penalty_max", 0.5)),
            road_edge_ttc_penalty_min_ttc=float(
                cfg_env.get("road_edge_ttc_penalty_min_ttc", 0.5)
            ),
            road_edge_ttc_hard_min_ttc=float(
                cfg_env.get("road_edge_ttc_hard_min_ttc", 0.5)
            ),
            road_edge_ttc_radius_m=(
                None
                if cfg_env.get("road_edge_ttc_radius_m", None) is None
                else float(cfg_env.get("road_edge_ttc_radius_m"))
            ),
            road_edge_ttc_penalty_function=cfg_env.get("road_edge_ttc_penalty_function", None),
            road_edge_ttc_proximity_zuo_a=cfg_env.get("road_edge_ttc_proximity_zuo_a", None),
            road_edge_ttc_proximity_zuo_b=cfg_env.get("road_edge_ttc_proximity_zuo_b", None),
            ttc_delta_penalty_enable=bool(cfg_env.get("ttc_delta_penalty_enable", False)),
            ttc_delta_penalty_alpha=float(cfg_env.get("ttc_delta_penalty_alpha", 0.0)),
            ttc_delta_penalty_max=float(cfg_env.get("ttc_delta_penalty_max", 0.5)),
            ttc_delta_penalty_normalize_by_dt=bool(
                cfg_env.get("ttc_delta_penalty_normalize_by_dt", False)
            ),
            ttc_use_vehicle_size=bool(cfg_env.get("ttc_use_vehicle_size", True)),
            ttc_vehicle_radius_scale=float(cfg_env.get("ttc_vehicle_radius_scale", 0.75)),
            ttc_vehicle_radius_margin_m=float(cfg_env.get("ttc_vehicle_radius_margin_m", 0.20)),
            ttc_backend=str(cfg_env.get("ttc_backend", "numpy")),
            obs_viz_enable=bool(cfg_env.get("obs_viz_enable", False)),
            obs_viz_world_idx=int(cfg_env.get("obs_viz_world_idx", 0)),
            obs_viz_agent_rank=int(cfg_env.get("obs_viz_agent_rank", 0)),
            render=bool(cfg_env.get("render", False)),
            respawn_on_reset=bool(cfg_env.get("respawn_on_reset", False)),
            respawn_mode=str(cfg_env.get("respawn_mode", "rebuild")),
            respawn_params={
                "spawn_z_m": float(agents_cfg.get("spawn_z_m", 1.0)),
                "parked_ground_z_m": float(agents_cfg.get("parked_ground_z_m", 0.0)),
                "parked_chassis_size_m": tuple(
                    map(float, agents_cfg.get("parked_chassis_size_m", [4.0, 2.0, 1.0]))
                ),
                "parked_wheel_radius_m": float(agents_cfg.get("parked_wheel_radius_m", 0.35)),
                "parked_wheel_thickness_m": float(agents_cfg.get("parked_wheel_thickness_m", 0.15)),
                "parked_wheel_inset_x_m": float(agents_cfg.get("parked_wheel_inset_x_m", 0.6)),
                "parked_wheel_inset_y_m": float(agents_cfg.get("parked_wheel_inset_y_m", 0.05)),
                "parked_ground_clearance_m": float(agents_cfg.get("parked_ground_clearance_m", 0.25)),
                "goal_radius_m": float(agents_cfg.get("goal_radius_m", 3.0)),
                "goal_ring_z_m": float(agents_cfg.get("goal_ring_z_m", 0.0)),
                "goal_ring_tube_radius_m": float(agents_cfg.get("goal_ring_tube_radius_m", 0.12)),
                "goal_trigger_height_m": float(agents_cfg.get("goal_trigger_height_m", 0.6)),
                "vehicle_trigger_enable": bool(agents_cfg.get("vehicle_trigger_enable", False)),
                "vehicle_trigger_offset_m": tuple(
                    map(float, agents_cfg.get("vehicle_trigger_offset_m", [0.0, 0.0, 0.0]))
                ),
                "vehicle_trigger_size_m": tuple(
                    map(float, agents_cfg.get("vehicle_trigger_size_m", [1.0, 1.0, 1.0]))
                ),
                "vehicle_trigger_script_enable": bool(
                    agents_cfg.get("vehicle_trigger_script_enable", True)
                ),
                "respawn_hold_radius_m": float(cfg_env.get("respawn_hold_radius_m", 3.0)),
                "respawn_clear_ignore_non_controllable": bool(
                    cfg_env.get("respawn_clear_ignore_non_controllable", True)
                ),
                "respawn_release_max_wait_steps": int(
                    cfg_env.get("respawn_release_max_wait_steps", 2)
                ),
                "respawn_friction_debug_every": int(
                    cfg_env.get("respawn_friction_debug_every", 0)
                ),
                "respawn_rebuild_flush_steps_before_create": int(
                    cfg_env.get("respawn_rebuild_flush_steps_before_create", 1)
                ),
                "respawn_rebuild_flush_steps_after_create": int(
                    cfg_env.get("respawn_rebuild_flush_steps_after_create", 1)
                ),
                "startup_below_min_z_preflight_steps": int(
                    cfg_env.get("startup_below_min_z_preflight_steps", 0)
                ),
            },
            verbose=bool(cfg_env.get("verbose", False)),
        )

        self.num_worlds = int(wcfg["world_count"])
        self.max_agent_count = int(wcfg["max_agents_per_world"])

        self.reward_type = str(reward_type)
        self.collision_weight = float(collision_weight)
        self.goal_achieved_weight = float(goal_achieved_weight)
        self.off_road_weight = float(off_road_weight)
        self.log_distance_weight = float(log_distance_weight)
        self.auto_reset_done = bool(cfg_env.get("auto_reset_done", True))
        self.auto_reset_timeout = bool(cfg_env.get("auto_reset_timeout", True))
        self.hard_remove_done_agents = bool(cfg_env.get("hard_remove_done_agents", False))
        self._dynamic_key_mapping = bool(self.hard_remove_done_agents)
        self._spawned_episode_total = 0
        self._pending_initial_spawn_count = 0
        self.verbose = bool(cfg_env.get("verbose", False))
        self.obs_vis_enable = bool(cfg_env.get("obs_vis_enable", False))
        self.obs_vis_every_steps = max(1, int(cfg_env.get("obs_vis_every_steps", 2000)))
        self.obs_vis_out_dir = str(cfg_env.get("obs_vis_out_dir", "runs/obs_vis"))
        self.obs_vis_agent_index = max(0, int(cfg_env.get("obs_vis_agent_index", 0)))
        self._step_count = 0

        self._keys: List[object] = []
        self._key_index: Dict[object, int] = {}

        self.controlled_agent_mask = torch.zeros(
            (self.num_worlds, self.max_agent_count), dtype=torch.bool
        ).to(self.device)
        self.dead_agent_mask = torch.zeros_like(self.controlled_agent_mask)

        self.slot_keys: List[List[Optional[object]]] = [
            [None for _ in range(self.max_agent_count)]
            for _ in range(self.num_worlds)
        ]
        self.flat_slot_keys: List[object] = []
        self.flat_key_indices: List[int] = []

        obs, mask, keys = self.choco_env.reset()
        startup_preflight_steps = int(cfg_env.get("startup_below_min_z_preflight_steps", 0))
        if startup_preflight_steps > 0:
            offenders = self.choco_env.collect_startup_below_min_z_offenders(startup_preflight_steps)
            if offenders:
                self.choco_env.quarantine_agents(offenders)
                print(f"[sb3] quarantined startup below_min_z agents: {offenders}")
            obs, mask, keys = self.choco_env.reset()
        # TODO: swap in a richer observation builder and update obs_dim accordingly.
        self._rebuild_slot_mapping(keys)

        self.obs_dim = int(obs.shape[-1])
        print(f"[sb3] initial obs shape={tuple(obs.shape)} obs_dim={self.obs_dim}")
        self.action_dim = 2

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.single_observation_space = self.observation_space

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        self.single_action_space = self.action_space

        self.num_envs = int(self.controlled_agent_mask.sum().item())
        self.info_dim = 1

        self.buf_rews = torch.full(
            (self.num_worlds, self.max_agent_count), fill_value=float("nan")
        ).to(self.device)
        self.buf_dones = torch.full(
            (self.num_worlds, self.max_agent_count), fill_value=float("nan")
        ).to(self.device)
        self.obs_alive = torch.full(
            (self.num_envs, self.obs_dim), fill_value=float("nan")
        ).to(self.device)

        self.info_dict = {}
        self.last_step_info_raw = None
        self._actions = None

    def _reset_seeds(self) -> None:
        self._seeds = None

    def step_async(self, actions) -> None:
        self._actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        if self._actions is None:
            raise RuntimeError("step_async must be called before step_wait")
        return self.step(self._actions)

    def get_attr(self, attr_name, indices=None):
        if indices is None:
            indices = list(range(self.num_envs))
        return [getattr(self, attr_name) for _ in indices]

    def set_attr(self, attr_name, value, indices=None) -> None:
        if indices is None:
            indices = list(range(self.num_envs))
        for _ in indices:
            setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if indices is None:
            indices = list(range(self.num_envs))
        results = []
        for _ in indices:
            method = getattr(self, method_name)
            results.append(method(*method_args, **method_kwargs))
        return results

    def env_is_wrapped(self, wrapper_class, indices=None):
        if indices is None:
            indices = list(range(self.num_envs))
        return [False for _ in indices]

    def _rebuild_slot_mapping(self, keys: Sequence[object]) -> None:
        self._keys = list(keys)
        self._key_index = {k: i for i, k in enumerate(self._keys)}

        self.controlled_agent_mask.zero_()

        self.slot_keys = [
            [None for _ in range(self.max_agent_count)]
            for _ in range(self.num_worlds)
        ]
        next_slot = [0 for _ in range(self.num_worlds)]

        for k in self._keys:
            if hasattr(self.choco_env, "is_agent_quarantined") and self.choco_env.is_agent_quarantined(k):
                continue
            wi = int(k.world_idx)
            if wi < 0 or wi >= self.num_worlds:
                continue
            slot = next_slot[wi]
            if slot >= self.max_agent_count:
                continue
            self.slot_keys[wi][slot] = k
            self.controlled_agent_mask[wi, slot] = True
            next_slot[wi] += 1

        self.dead_agent_mask = ~self.controlled_agent_mask.clone()

        self.flat_slot_keys = []
        self.flat_key_indices = []
        for wi in range(self.num_worlds):
            for si in range(self.max_agent_count):
                if not bool(self.controlled_agent_mask[wi, si].item()):
                    continue
                k = self.slot_keys[wi][si]
                if k is None:
                    continue
                self.flat_slot_keys.append(k)
                self.flat_key_indices.append(self._key_index[k])
        self.num_envs = int(len(self.flat_key_indices))

    @staticmethod
    def _keys_signature(keys: Sequence[object]) -> tuple:
        return tuple((int(k.world_idx), int(k.agent_id)) for k in keys)

    def _sync_slot_mapping_from_env(self) -> bool:
        env_keys = list(getattr(self.choco_env, "_keys", []) or [])
        if self._keys_signature(env_keys) == self._keys_signature(self._keys):
            return False
        self._rebuild_slot_mapping(env_keys)
        if hasattr(self, "obs_dim"):
            self.obs_alive = torch.full(
                (self.num_envs, self.obs_dim), fill_value=float("nan"), device=self.device
            )
        if self.verbose:
            print(
                "[sb3] remapped keys from env "
                f"num_envs={self.num_envs} keys={len(self._keys)}"
            )
        return True

    def _build_info_dict(
        self,
        *,
        info,
        done: np.ndarray,
        reward: np.ndarray,
        base_reward: np.ndarray,
        spawned_count_step: int = 0,
    ) -> dict:
        mask = np.asarray(getattr(info, "mask", np.zeros((0,), dtype=bool)), dtype=bool)
        success = np.asarray(getattr(info, "success", np.zeros_like(mask)), dtype=bool)
        newly_success = np.asarray(
            getattr(info, "newly_success", np.zeros_like(mask)),
            dtype=bool,
        )
        road_contact_done = np.asarray(
            getattr(info, "road_contact_done", np.zeros_like(mask)),
            dtype=bool,
        )
        vehicle_contact_done = np.asarray(
            getattr(info, "vehicle_contact_done", np.zeros_like(mask)),
            dtype=bool,
        )
        collided = np.asarray(getattr(info, "collided", np.zeros_like(mask)), dtype=bool)
        road_collided = np.asarray(
            getattr(info, "road_collided", np.zeros_like(mask)),
            dtype=bool,
        )
        vehicle_collided = np.asarray(
            getattr(info, "vehicle_collided", np.zeros_like(mask)),
            dtype=bool,
        )
        below_min_z = np.asarray(
            getattr(info, "below_min_z", np.zeros_like(mask)),
            dtype=bool,
        )
        off_road = np.asarray(getattr(info, "off_road", np.zeros_like(mask)), dtype=bool)
        lane_hit = np.asarray(getattr(info, "lane_hit", np.zeros_like(mask)), dtype=bool)
        active = np.asarray(getattr(info, "active", np.zeros_like(mask)), dtype=bool)
        pending = np.asarray(getattr(info, "pending", np.zeros_like(mask)), dtype=bool)
        dist_m = np.asarray(
            getattr(info, "dist_m", np.zeros(mask.shape, dtype=np.float32)),
            dtype=np.float32,
        )
        lane_error_m = np.asarray(
            getattr(info, "lane_error_m", np.zeros(mask.shape, dtype=np.float32)),
            dtype=np.float32,
        )
        heading_alignment = np.asarray(
            getattr(info, "heading_alignment", np.zeros(mask.shape, dtype=np.float32)),
            dtype=np.float32,
        )
        route_progress_m = np.asarray(
            getattr(info, "route_progress_m", np.zeros(mask.shape, dtype=np.float32)),
            dtype=np.float32,
        )
        done = np.asarray(done, dtype=bool)
        reward = np.asarray(reward, dtype=np.float32)
        base_reward = np.asarray(base_reward, dtype=np.float32)

        if mask.size and self.flat_key_indices:
            sel = np.asarray(self.flat_key_indices, dtype=np.int64)
            mask = mask[sel]
            success = success[sel]
            newly_success = newly_success[sel]
            road_contact_done = road_contact_done[sel]
            vehicle_contact_done = vehicle_contact_done[sel]
            collided = collided[sel]
            road_collided = road_collided[sel]
            vehicle_collided = vehicle_collided[sel]
            below_min_z = below_min_z[sel]
            off_road = off_road[sel]
            lane_hit = lane_hit[sel]
            active = active[sel]
            pending = pending[sel]
            dist_m = dist_m[sel]
            lane_error_m = lane_error_m[sel]
            heading_alignment = heading_alignment[sel]
            route_progress_m = route_progress_m[sel]
            done = done[sel]
            reward = reward[sel]
            base_reward = base_reward[sel]

        controlled_agents = int(len(self.flat_key_indices))
        valid_agents = int(mask.sum())
        active_agents = int(active.sum())
        pending_agents = int(pending.sum())
        done_count = int(done.sum())
        new_success_count = int(newly_success.sum())
        success_latched_count = int(success.sum())
        road_contact_done_count = int(road_contact_done.sum())
        vehicle_contact_done_count = int(vehicle_contact_done.sum())
        off_road_count = int(off_road.sum())
        lane_hit_count = int(lane_hit.sum())
        collided_count = int(collided.sum())
        road_collided_count = int(road_collided.sum())
        vehicle_collided_count = int(vehicle_collided.sum())
        done_vehicle_collided_count = int(np.logical_and(done, vehicle_collided).sum())
        below_min_z_count = int(below_min_z.sum())

        active_dist = dist_m[active] if dist_m.size and active.any() else np.asarray([], dtype=np.float32)
        valid_dist = dist_m[mask] if dist_m.size and mask.any() else np.asarray([], dtype=np.float32)
        active_lane_error = (
            lane_error_m[active] if lane_error_m.size and active.any() else np.asarray([], dtype=np.float32)
        )
        active_heading_alignment = (
            heading_alignment[active]
            if heading_alignment.size and active.any()
            else np.asarray([], dtype=np.float32)
        )
        active_route_progress = (
            route_progress_m[active]
            if route_progress_m.size and active.any()
            else np.asarray([], dtype=np.float32)
        )

        denom_controlled = max(1, controlled_agents)
        denom_active = max(1, active_agents)
        denom_done = max(1, done_count)

        return {
            "num_controlled_agents": controlled_agents,
            "num_valid_agents": valid_agents,
            "num_active_agents": active_agents,
            "pending_respawn_count": pending_agents,
            "spawned_count_step": int(spawned_count_step),
            "spawned_episode_total": int(self._spawned_episode_total),
            "done_count": done_count,
            "goal_achieved": float(new_success_count),
            "new_success_count": new_success_count,
            "success_latched_count": success_latched_count,
            "done_success_count": int(np.logical_and(done, success).sum()),
            "road_contact_done_count": road_contact_done_count,
            "vehicle_contact_done_count": vehicle_contact_done_count,
            "truncated": float(bool(getattr(info, "timeout", False))),
            "off_road": float(off_road_count),
            "off_road_count": off_road_count,
            "lane_hit_count": lane_hit_count,
            "collided": float(collided_count),
            "collided_count": collided_count,
            "road_collided_count": road_collided_count,
            "vehicle_collided_count": vehicle_collided_count,
            # GPUDRIVE-style numerator: done agents that collided at least once.
            "done_vehicle_collided_count": done_vehicle_collided_count,
            "below_min_z_count": below_min_z_count,
            "goal_rate_step": float(new_success_count) / float(denom_controlled),
            "success_latched_rate_step": float(success_latched_count) / float(denom_controlled),
            "road_contact_done_rate_step": float(road_contact_done_count) / float(denom_controlled),
            "vehicle_contact_done_rate_step": float(vehicle_contact_done_count) / float(denom_controlled),
            "off_road_rate_step": float(off_road_count) / float(denom_controlled),
            "lane_hit_rate_step": float(lane_hit_count) / float(denom_controlled),
            "collision_rate_step": float(collided_count) / float(denom_controlled),
            "road_collision_rate_step": float(road_collided_count) / float(denom_controlled),
            # Done-conditioned: among agents set done this step, how many are vehicle-contact done.
            "vehicle_collision_rate_step": float(vehicle_contact_done_count) / float(denom_done),
            # Legacy controlled-conditioned vehicle collision incidence (kept for debugging).
            "vehicle_collision_rate_step_per_controlled": float(vehicle_collided_count)
            / float(denom_controlled),
            # GPUDRIVE-style per-step parity metric.
            "perc_veh_collisions_step": float(done_vehicle_collided_count) / float(denom_done),
            "done_rate_step": float(done_count) / float(denom_controlled),
            "success_given_done_rate_step": float(new_success_count) / float(denom_done),
            "mean_dist_to_goal_m": _safe_mean(valid_dist),
            "mean_active_dist_to_goal_m": _safe_mean(active_dist),
            "min_active_dist_to_goal_m": float(np.min(active_dist)) if active_dist.size else 0.0,
            "mean_lane_error_m": _safe_mean(active_lane_error),
            "mean_heading_alignment": _safe_mean(active_heading_alignment),
            "mean_route_progress_m": _safe_mean(active_route_progress),
            "mean_reward_step": _safe_mean(reward),
            "mean_base_reward_step": _safe_mean(base_reward),
            "active_fraction": float(active_agents) / float(denom_controlled),
            "pending_fraction": float(pending_agents) / float(denom_controlled),
            "road_contact_done_given_active_rate_step": float(road_contact_done_count) / float(denom_active),
            "vehicle_contact_done_given_active_rate_step": float(vehicle_contact_done_count) / float(denom_active),
            "lane_hit_given_active_rate_step": float(lane_hit_count) / float(denom_active),
            "off_road_given_active_rate_step": float(off_road_count) / float(denom_active),
            "collision_given_active_rate_step": float(collided_count) / float(denom_active),
            "t_env": int(getattr(info, "t_env", 0)),
        }

    def reset(self, world_idx=None, seed=None):
        if world_idx is not None:
            # ChocolateEnv resets all agents; per-world reset is not implemented.
            return None

        obs, mask, keys = self.choco_env.reset()
        self._rebuild_slot_mapping(keys)
        initial_spawn_count = int(len(self.flat_key_indices))
        self._spawned_episode_total += initial_spawn_count
        self._pending_initial_spawn_count += initial_spawn_count

        obs_flat = obs[self.flat_key_indices]
        if self.verbose:
            print(f"[sb3] reset obs shape={tuple(obs_flat.shape)}")
        obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=self.device)

        self.dead_agent_mask = ~self.controlled_agent_mask.clone()
        self.buf_rews = torch.full_like(self.buf_rews, fill_value=float("nan"))
        self.buf_dones = torch.full_like(self.buf_dones, fill_value=float("nan"))
        self.obs_alive = obs_t.clone()

        return obs_flat.astype(np.float32)

    def _compute_rewards(self, obs, reward, done, info):
        reward = np.asarray(reward, dtype=np.float32)
        if self.reward_type == "sparse_on_goal_achieved":
            return (info.success.astype(np.float32)).copy()

        if self.reward_type == "weighted_combination":
            collided = getattr(info, "collided", np.zeros_like(reward)).astype(np.float32)
            off_road = getattr(info, "off_road", np.zeros_like(reward)).astype(np.float32)
            goal_achieved = info.success.astype(np.float32)
            return reward + (
                self.collision_weight * collided
                + self.goal_achieved_weight * goal_achieved
                + self.off_road_weight * off_road
            )

        if self.reward_type == "distance_to_logs":
            # TODO: implement distance-to-logs reward using recorded trajectories.
            return reward.copy()

        return reward.copy()

    def step(self, actions) -> VecEnvStepReturn:
        self._step_count += 1
        spawned_count_step = int(self._pending_initial_spawn_count)
        self._pending_initial_spawn_count = 0
        if self._dynamic_key_mapping:
            self._sync_slot_mapping_from_env()
        if torch.is_tensor(actions):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = np.asarray(actions, dtype=np.float32)

        if actions_np.ndim == 1:
            actions_np = actions_np.reshape(-1, self.action_dim)

        if self._dynamic_key_mapping:
            expected_envs = int(len(self.flat_slot_keys))
            input_rows = int(actions_np.shape[0])
            if actions_np.shape[0] != expected_envs:
                if actions_np.shape[0] > expected_envs:
                    actions_np = actions_np[:expected_envs, :]
                else:
                    pad = np.zeros((expected_envs - actions_np.shape[0], self.action_dim), dtype=np.float32)
                    actions_np = np.concatenate([actions_np, pad], axis=0)
                if self.verbose:
                    print(
                        "[sb3] adjusted action batch to current env size "
                        f"got={input_rows} "
                        f"expected={expected_envs}"
                    )

        U = np.zeros((len(self._keys), self.action_dim), dtype=np.float32)

        for env_idx, key in enumerate(self.flat_slot_keys):
            key_idx = self._key_index.get(key)
            if key_idx is None:
                continue
            U[key_idx, :] = actions_np[env_idx, :]

        obs, base_reward, done, info = self.choco_env.step(U)
        self.last_step_info_raw = info
        reward = self._compute_rewards(obs, base_reward, done, info)

        done_mask = np.zeros((self.num_worlds, self.max_agent_count), dtype=bool)
        success_mask = np.zeros((self.num_worlds, self.max_agent_count), dtype=bool)
        reward_mask = np.full(
            (self.num_worlds, self.max_agent_count), fill_value=np.nan, dtype=np.float32
        )
        for wi in range(self.num_worlds):
            for si in range(self.max_agent_count):
                if not bool(self.controlled_agent_mask[wi, si].item()):
                    continue
                k = self.slot_keys[wi][si]
                if k is None:
                    continue
                key_idx = self._key_index[k]
                done_mask[wi, si] = bool(done[key_idx])
                if hasattr(info, "success"):
                    success_mask[wi, si] = bool(info.success[key_idx])
                reward_mask[wi, si] = float(reward[key_idx])

        done_mask_t = torch.tensor(done_mask, device=self.device)

        reward_mask_t = torch.tensor(reward_mask, device=self.device)
        self.buf_rews = reward_mask_t.clone()
        self.buf_dones = done_mask_t.to(torch.float32)

        if hasattr(info, "timeout") and info.timeout:
            if self.auto_reset_timeout:
                timeout_respawn_count = int(len(self.flat_key_indices))
                self._spawned_episode_total += timeout_respawn_count
                spawned_count_step += timeout_respawn_count
                self.choco_env.reset_timeout()
                obs, _, _ = self.choco_env._build_obs()
        elif np.any(done):
            if self.auto_reset_done:
                if self.flat_key_indices:
                    sel = np.asarray(self.flat_key_indices, dtype=np.int64)
                    done_respawn_count = int(np.asarray(done, dtype=bool)[sel].sum())
                else:
                    done_respawn_count = int(np.asarray(done, dtype=bool).sum())
                self._spawned_episode_total += done_respawn_count
                spawned_count_step += done_respawn_count
                self.choco_env.reset_done(done)
                obs, _, _ = self.choco_env._build_obs()

        obs_flat = obs[self.flat_key_indices]
        if self.verbose:
            print(f"[sb3] step obs shape={tuple(obs_flat.shape)}")
        obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=self.device)
        self.obs_alive = obs_t.clone()

        info_flat = [{} for _ in range(self.num_envs)]

        # Lightweight info summary for optional logging.
        self.info_dict = self._build_info_dict(
            info=info,
            done=done,
            reward=reward,
            base_reward=base_reward,
            spawned_count_step=spawned_count_step,
        )

        mask_cpu = self.controlled_agent_mask.cpu()
        rewards_flat = (
            self.buf_rews.cpu()[mask_cpu]
            .reshape(self.num_envs)
            .numpy()
        )
        dones_flat = (
            self.buf_dones.cpu()[mask_cpu]
            .reshape(self.num_envs)
            .numpy()
            .astype(bool)
        )

        if self.obs_vis_enable and (self._step_count % self.obs_vis_every_steps == 0):
            out_dir = Path(self.obs_vis_out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            idx = min(self.obs_vis_agent_index, max(0, obs_flat.shape[0] - 1))
            obs_vec = obs_flat[idx]
            out_path = out_dir / f"obs_step_{self._step_count:07d}.png"
            save_obs_png(obs_vec, out_path, title=f"obs step {self._step_count} agent {idx}")

        return (
            obs_flat.astype(np.float32),
            rewards_flat.astype(np.float32),
            dones_flat,
            info_flat,
        )

    def close(self) -> None:
        self.builder.close()

    def capture_frame(self, filepath: str) -> bool:
        return self.builder.capture_frame(filepath)

    def set_render(self, enabled: bool) -> None:
        self.choco_env.render = bool(enabled)
