from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn

from gpudrive_chocolate.env.choco_world import ChocoWorldBuilder, load_config
from gpudrive_chocolate.utils.obs_vis import save_obs_png


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
            vehicle_obs_enable=bool(cfg_env["vehicle_obs_enable"]),
            vehicle_obs_k=int(cfg_env["vehicle_obs_k"]),
            ttc_penalty_enable=bool(cfg_env.get("ttc_penalty_enable", False)),
            ttc_penalty_alpha=float(cfg_env.get("ttc_penalty_alpha", 1.0)),
            ttc_penalty_max=float(cfg_env.get("ttc_penalty_max", 1.0)),
            ttc_penalty_min_ttc=float(cfg_env.get("ttc_penalty_min_ttc", 0.2)),
            obs_viz_enable=bool(cfg_env.get("obs_viz_enable", False)),
            obs_viz_world_idx=int(cfg_env.get("obs_viz_world_idx", 0)),
            obs_viz_agent_rank=int(cfg_env.get("obs_viz_agent_rank", 0)),
            render=bool(cfg_env.get("render", False)),
            respawn_on_reset=bool(cfg_env.get("respawn_on_reset", False)),
            respawn_params={
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

    def reset(self, world_idx=None, seed=None):
        if world_idx is not None:
            # ChocolateEnv resets all agents; per-world reset is not implemented.
            return None

        obs, mask, keys = self.choco_env.reset()
        self._rebuild_slot_mapping(keys)

        obs_flat = obs[self.flat_key_indices]
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
        if torch.is_tensor(actions):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = np.asarray(actions, dtype=np.float32)

        if actions_np.ndim == 1:
            actions_np = actions_np.reshape(-1, self.action_dim)

        U = np.zeros((len(self._keys), self.action_dim), dtype=np.float32)

        for env_idx, key in enumerate(self.flat_slot_keys):
            key_idx = self._key_index.get(key)
            if key_idx is None:
                continue
            U[key_idx, :] = actions_np[env_idx, :]

        obs, reward, done, info = self.choco_env.step(U)
        reward = self._compute_rewards(obs, reward, done, info)

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
            self.choco_env.reset_timeout()
            obs, _, _ = self.choco_env._build_obs()
        elif np.any(done):
            self.choco_env.reset_done(done)
            obs, _, _ = self.choco_env._build_obs()

        obs_flat = obs[self.flat_key_indices]
        print(f"[sb3] step obs shape={tuple(obs_flat.shape)}")
        obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=self.device)
        self.obs_alive = obs_t.clone()

        info_flat = [{} for _ in range(self.num_envs)]

        # Lightweight info summary for optional logging.
        self.info_dict = {
            "num_controlled_agents": int(self.controlled_agent_mask.sum().item()),
            "goal_achieved": float(info.success.sum()) if hasattr(info, "success") else 0.0,
            "done_count": float(done_mask.sum()),
            "done_success_count": float(np.logical_and(done_mask, success_mask).sum()),
            "truncated": float(info.timeout) if hasattr(info, "timeout") else 0.0,
            "off_road": float(info.off_road.sum()) if hasattr(info, "off_road") else 0.0,
            "collided": float(info.collided.sum()) if hasattr(info, "collided") else 0.0,
        }

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
