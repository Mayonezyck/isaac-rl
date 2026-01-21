from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class Dot2DConfig:
    action_type: str = "continuous"  # "continuous" or "discrete"
    dynamics_mode: str = "position"  # "position" or "velocity"
    observation_mode: str = "relative"  # "absolute", "relative", "relative_only"
    vel_enabled: bool = False
    step_size: float = 0.1
    accel_scale: float = 0.1
    max_speed: float = 0.5
    dt: float = 1.0
    goal_radius: float = 0.1
    max_steps: int = 200
    spawn_range: float = 1.0
    goal_range: float = 1.0
    fixed_spawn: Optional[Tuple[float, float]] = None
    fixed_goal: Optional[Tuple[float, float]] = None
    reward_mode: str = "dense_neg_dist"  # "dense_neg_dist", "progress", "sparse"
    alive_penalty: float = 0.0
    success_bonus: float = 1.0
    render_mode: Optional[str] = None  # "human", "rgb_array", or None


class Dot2DEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: Dict[str, Any] | Dot2DConfig):
        if isinstance(config, dict):
            self.cfg = Dot2DConfig(**config)
        else:
            self.cfg = config

        self._step_count = 0
        self._pos = np.zeros(2, dtype=np.float32)
        self._vel = np.zeros(2, dtype=np.float32)
        self._goal = np.zeros(2, dtype=np.float32)
        self._prev_dist = 0.0

        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()
        self.render_mode = self.cfg.render_mode

        self._fig = None
        self._ax = None
        self._render_limit = max(self.cfg.spawn_range, self.cfg.goal_range) + 1.0

    def _build_action_space(self) -> spaces.Space:
        if self.cfg.action_type == "discrete":
            return spaces.Discrete(4)
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def _build_observation_space(self) -> spaces.Space:
        size = 0
        if self.cfg.observation_mode == "absolute":
            size = 4
        elif self.cfg.observation_mode == "relative":
            size = 4
        elif self.cfg.observation_mode == "relative_only":
            size = 2
        else:
            raise ValueError(f"Unknown observation_mode: {self.cfg.observation_mode}")
        if self.cfg.vel_enabled:
            size += 2
        return spaces.Box(low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._step_count = 0
        self._vel = np.zeros(2, dtype=np.float32)

        if self.cfg.fixed_spawn is not None:
            self._pos = np.array(self.cfg.fixed_spawn, dtype=np.float32)
        else:
            self._pos = self.np_random.uniform(
                low=-self.cfg.spawn_range,
                high=self.cfg.spawn_range,
                size=(2,),
            ).astype(np.float32)

        if self.cfg.fixed_goal is not None:
            self._goal = np.array(self.cfg.fixed_goal, dtype=np.float32)
        else:
            self._goal = self.np_random.uniform(
                low=-self.cfg.goal_range,
                high=self.cfg.goal_range,
                size=(2,),
            ).astype(np.float32)

        self._prev_dist = self._distance_to_goal()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._step_count += 1
        action_vec = self._action_to_vector(action)

        if self.cfg.dynamics_mode == "position":
            self._pos = self._pos + self.cfg.step_size * action_vec
        elif self.cfg.dynamics_mode == "velocity":
            self._vel = self._vel + self.cfg.accel_scale * action_vec
            self._vel = np.clip(self._vel, -self.cfg.max_speed, self.cfg.max_speed)
            self._pos = self._pos + self._vel * self.cfg.dt
        else:
            raise ValueError(f"Unknown dynamics_mode: {self.cfg.dynamics_mode}")

        dist = self._distance_to_goal()
        terminated = dist < self.cfg.goal_radius
        truncated = self._step_count >= self.cfg.max_steps
        reward = self._compute_reward(dist, terminated)

        info = {"distance": dist}
        obs = self._get_obs()
        self._prev_dist = dist
        return obs, reward, terminated, truncated, info

    def _action_to_vector(self, action: np.ndarray) -> np.ndarray:
        if self.cfg.action_type == "discrete":
            mapping = {
                0: np.array([1.0, 0.0], dtype=np.float32),
                1: np.array([-1.0, 0.0], dtype=np.float32),
                2: np.array([0.0, 1.0], dtype=np.float32),
                3: np.array([0.0, -1.0], dtype=np.float32),
            }
            return mapping[int(action)]
        return np.array(action, dtype=np.float32)

    def _compute_reward(self, dist: float, success: bool) -> float:
        if self.cfg.reward_mode == "dense_neg_dist":
            reward = -dist
        elif self.cfg.reward_mode == "progress":
            reward = self._prev_dist - dist
        elif self.cfg.reward_mode == "sparse":
            reward = 0.0
        else:
            raise ValueError(f"Unknown reward_mode: {self.cfg.reward_mode}")

        reward += self.cfg.alive_penalty
        if success:
            reward += self.cfg.success_bonus
        return float(reward)

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self._goal - self._pos))

    def _get_obs(self) -> np.ndarray:
        if self.cfg.observation_mode == "absolute":
            obs = [self._pos[0], self._pos[1], self._goal[0], self._goal[1]]
        elif self.cfg.observation_mode == "relative":
            obs = [
                self._pos[0],
                self._pos[1],
                self._goal[0] - self._pos[0],
                self._goal[1] - self._pos[1],
            ]
        elif self.cfg.observation_mode == "relative_only":
            obs = [self._goal[0] - self._pos[0], self._goal[1] - self._pos[1]]
        else:
            raise ValueError(f"Unknown observation_mode: {self.cfg.observation_mode}")

        if self.cfg.vel_enabled:
            obs.extend([self._vel[0], self._vel[1]])
        return np.array(obs, dtype=np.float32)

    def render(self):
        if self.cfg.render_mode is None:
            return None

        import matplotlib.pyplot as plt

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(4, 4))
            self._ax.set_xlim(-self._render_limit, self._render_limit)
            self._ax.set_ylim(-self._render_limit, self._render_limit)
            self._ax.set_aspect("equal")
            self._ax.set_title("Dot2D")

        self._ax.clear()
        self._ax.set_xlim(-self._render_limit, self._render_limit)
        self._ax.set_ylim(-self._render_limit, self._render_limit)
        self._ax.scatter(self._goal[0], self._goal[1], c="green", s=80, marker="*")
        self._ax.scatter(self._pos[0], self._pos[1], c="blue", s=40)
        self._ax.plot([self._pos[0], self._goal[0]], [self._pos[1], self._goal[1]], c="gray", lw=1)

        if self.cfg.render_mode == "human":
            self._fig.canvas.draw()
            plt.pause(1.0 / self.metadata["render_fps"])
            return None

        if self.cfg.render_mode == "rgb_array":
            self._fig.canvas.draw()
            width, height = self._fig.canvas.get_width_height()
            image = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            return image.reshape((height, width, 3))

        raise ValueError(f"Unknown render_mode: {self.cfg.render_mode}")

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None
