from __future__ import annotations

import warnings
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn

from src.isaaclab_bootstrap import ensure_isaaclab_source_paths

ensure_isaaclab_source_paths()

from isaaclab.envs import DirectMARLEnv


warnings.filterwarnings("ignore", message="You are trying to run PPO on the GPU")


class Sb3SharedMultiAgentVecEnvWrapper(VecEnv):
    """Expose each agent in a DirectMARLEnv as an SB3 rollout slot with shared policy parameters."""

    def __init__(self, env: DirectMARLEnv):
        if not isinstance(env.unwrapped, DirectMARLEnv):
            raise ValueError(f"Expected DirectMARLEnv, received: {type(env)}")

        self.env = env
        self.agent_ids = list(self.env.unwrapped.possible_agents)
        self.num_worlds = int(self.env.unwrapped.num_envs)
        self.num_agents_per_env = len(self.agent_ids)
        self.num_envs = self.num_worlds * self.num_agents_per_env
        self.sim_device = self.env.unwrapped.device
        self.render_mode = self.env.unwrapped.render_mode

        observation_space = self.env.unwrapped.observation_spaces[self.agent_ids[0]]
        action_space = self.env.unwrapped.action_spaces[self.agent_ids[0]]
        if isinstance(action_space, gym.spaces.Box) and not action_space.is_bounded("both"):
            action_space = gym.spaces.Box(low=-100.0, high=100.0, shape=action_space.shape)

        VecEnv.__init__(self, self.num_envs, observation_space, action_space)
        self._ep_rew_buf = np.zeros(self.num_envs, dtype=np.float32)
        self._ep_len_buf = np.zeros(self.num_envs, dtype=np.int32)

    @property
    def unwrapped(self) -> DirectMARLEnv:
        return self.env.unwrapped

    def seed(self, seed: int | None = None) -> list[int | None]:
        return [self.unwrapped.seed(seed)] * self.num_envs

    def reset(self) -> VecEnvObs:
        obs_dict, _ = self.env.reset()
        self._ep_rew_buf[:] = 0.0
        self._ep_len_buf[:] = 0
        return self._flatten_tensor_dict(obs_dict)

    def step_async(self, actions):
        if not isinstance(actions, torch.Tensor):
            actions = np.asarray(actions)
            actions = torch.from_numpy(actions).to(device=self.sim_device, dtype=torch.float32)
        else:
            actions = actions.to(device=self.sim_device, dtype=torch.float32)
        self._async_actions = self._unflatten_actions(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs_dict, reward_dict, terminated_dict, truncated_dict, extras = self.env.step(self._async_actions)

        obs = self._flatten_tensor_dict(obs_dict)
        rewards = self._flatten_tensor_dict(reward_dict).reshape(self.num_envs)
        terminated = self._flatten_tensor_dict(terminated_dict).reshape(self.num_envs).astype(bool)
        truncated = self._flatten_tensor_dict(truncated_dict).reshape(self.num_envs).astype(bool)
        dones = terminated | truncated

        self._ep_rew_buf += rewards
        self._ep_len_buf += 1
        reset_ids = np.nonzero(dones)[0]
        infos = [{} for _ in range(self.num_envs)]

        for idx in reset_ids:
            world_idx = idx // self.num_agents_per_env
            agent_idx = idx % self.num_agents_per_env
            agent_id = self.agent_ids[agent_idx]
            infos[idx]["episode"] = {
                "r": float(self._ep_rew_buf[idx]),
                "l": float(self._ep_len_buf[idx]),
            }
            infos[idx]["TimeLimit.truncated"] = bool(truncated[idx] and not terminated[idx])
            infos[idx]["terminal_observation"] = obs[idx]
            agent_extras = extras.get(agent_id, {})
            if "log" in agent_extras:
                for key, value in agent_extras["log"].items():
                    infos[idx]["episode"][key] = value
            infos[idx]["world_index"] = int(world_idx)
            infos[idx]["agent_index"] = int(agent_idx)
            infos[idx]["agent_id"] = agent_id

        self._ep_rew_buf[reset_ids] = 0.0
        self._ep_len_buf[reset_ids] = 0
        return obs, rewards, dones, infos

    def close(self):
        self.env.close()

    def get_attr(self, attr_name, indices=None):
        if indices is None:
            return [getattr(self.env, attr_name)] * self.num_envs
        return [getattr(self.env, attr_name)] * len(indices)

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("Setting attributes is not supported.")

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        if method_name == "render":
            return self.env.render()
        env_method = getattr(self.env, method_name)
        return env_method(*method_args, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False]

    def get_images(self):
        raise NotImplementedError("Getting images is not supported.")

    def _flatten_tensor_dict(self, tensor_dict: dict[str, torch.Tensor]) -> np.ndarray:
        stacked = torch.stack([tensor_dict[agent_id] for agent_id in self.agent_ids], dim=1)
        flat = stacked.reshape(self.num_envs, *stacked.shape[2:])
        return flat.detach().cpu().numpy()

    def _unflatten_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        reshaped = actions.reshape(self.num_worlds, self.num_agents_per_env, *actions.shape[1:])
        return {
            agent_id: reshaped[:, agent_idx].to(device=self.sim_device, dtype=torch.float32)
            for agent_idx, agent_id in enumerate(self.agent_ids)
        }
