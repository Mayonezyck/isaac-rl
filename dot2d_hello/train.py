from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Callable

import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from env_dot2d import Dot2DEnv


CONFIG: Dict[str, Any] = {
    "seed": 42,
    "total_timesteps": 200_000,
    "num_envs": 1,
    "vec_env_type": "dummy",  # "dummy" or "subproc"
    "algo": None,  # None for auto, or "PPO", "A2C", "DQN"
    "tensorboard_log": "runs",
    "log_interval": 10,
    "save_dir": "runs",
    "model_name": "dot2d",
    "action_noise_std": 0.1,  # None or float for continuous actions
    "render_during_training": True,
    "render_freq": 1,
    "use_vecnormalize": True,
    "norm_obs": True,
    "norm_reward": False,
    "clip_obs": 10.0,
    "gamma": 0.99,
    "algo_kwargs": {
        "PPO": {
            "learning_rate": 3e-4,
            "n_steps": 1024,
            "batch_size": 64,
            "gae_lambda": 0.95,
            "ent_coef": 0.0,
            "clip_range": 0.2,
        },
        "A2C": {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "ent_coef": 0.0,
        },
        "DQN": {
            "learning_rate": 1e-3,
            "buffer_size": 50_000,
            "learning_starts": 1_000,
            "batch_size": 64,
            "tau": 1.0,
            "train_freq": 4,
            "target_update_interval": 1_000,
        },
    },
    "env_config": {
        "action_type": "continuous",  # "continuous" or "discrete"
        "dynamics_mode": "velocity",  # "position" or "velocity"
        "observation_mode": "relative",  # "absolute", "relative", "relative_only"
        "vel_enabled": False,
        "step_size": 0.1,
        "accel_scale": 0.1,
        "max_speed": 0.5,
        "dt": 1.0,
        "goal_radius": 0.1,
        "max_steps": 200,
        "spawn_range": 1.0,
        "goal_range": 1.0,
        "fixed_spawn": None,
        "fixed_goal": None,
        "reward_mode": "dense_neg_dist",  # "dense_neg_dist", "progress", "sparse"
        "alive_penalty": 0.0,
        "success_bonus": 1.0,
        "render_mode": None,  # Set to "human" to see live rendering
    },
}


class ActionNoiseWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, std: float, seed: int):
        super().__init__(env)
        self._rng = np.random.default_rng(seed)
        self._std = float(std)

    def action(self, action: np.ndarray) -> np.ndarray:
        noise = self._rng.normal(0.0, self._std, size=action.shape)
        return np.clip(action + noise, -1.0, 1.0)


class RenderCallback(BaseCallback):
    def __init__(self, render_freq: int):
        super().__init__()
        self._render_freq = max(1, int(render_freq))

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            self.training_env.envs[0].render()
        return True


def make_env(config: Dict[str, Any], seed: int, rank: int) -> Callable[[], Dot2DEnv]:
    def _init() -> Dot2DEnv:
        env = Dot2DEnv(config)
        env.reset(seed=seed + rank)
        if config["action_type"] == "continuous" and CONFIG["action_noise_std"]:
            env = ActionNoiseWrapper(env, CONFIG["action_noise_std"], seed + rank)
        return env

    return _init


def build_vec_env(config: Dict[str, Any]) -> DummyVecEnv | SubprocVecEnv:
    seed = int(config["seed"])
    env_fns = [make_env(config["env_config"], seed, i) for i in range(config["num_envs"])]
    if config["vec_env_type"] == "subproc":
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def select_algo(config: Dict[str, Any]) -> str:
    env_action_type = config["env_config"]["action_type"]
    if config["algo"]:
        return str(config["algo"]).upper()
    return "DQN" if env_action_type == "discrete" else "PPO"


def main() -> None:
    cfg = CONFIG
    os.makedirs(cfg["save_dir"], exist_ok=True)

    if cfg["render_during_training"]:
        if cfg["vec_env_type"] != "dummy" or cfg["num_envs"] != 1:
            raise ValueError("Live rendering requires vec_env_type='dummy' and num_envs=1.")
        cfg["env_config"]["render_mode"] = "human"

    vec_env = build_vec_env(cfg)

    if cfg["use_vecnormalize"]:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=cfg["norm_obs"],
            norm_reward=cfg["norm_reward"],
            clip_obs=cfg["clip_obs"],
            gamma=cfg["gamma"],
        )

    algo = select_algo(cfg)
    algo_kwargs = cfg["algo_kwargs"].get(algo, {}).copy()
    tensorboard_log = cfg["tensorboard_log"]
    if algo == "DQN" and cfg["env_config"]["action_type"] != "discrete":
        raise ValueError("DQN only supports discrete action spaces. Set action_type to \"discrete\".")

    if algo == "PPO":
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            seed=cfg["seed"],
            **algo_kwargs,
        )
    elif algo == "A2C":
        model = A2C(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            seed=cfg["seed"],
            **algo_kwargs,
        )
    elif algo == "DQN":
        model = DQN(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            seed=cfg["seed"],
            **algo_kwargs,
        )
    else:
        raise ValueError(f"Unsupported algo: {algo}")

    callback = RenderCallback(cfg["render_freq"]) if cfg["render_during_training"] else None
    model.learn(
        total_timesteps=cfg["total_timesteps"],
        log_interval=cfg["log_interval"],
        callback=callback,
    )

    run_dir = Path(cfg["save_dir"]) / cfg["model_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.zip"
    model.save(model_path)

    if cfg["use_vecnormalize"] and isinstance(vec_env, VecNormalize):
        vec_env.save(str(run_dir / "vecnormalize.pkl"))

    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    vec_env.close()


if __name__ == "__main__":
    main()
