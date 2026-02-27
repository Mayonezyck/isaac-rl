import os
import sys
import yaml
from box import Box
from datetime import datetime
from typing import Callable

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gpudrive_chocolate.env.sb3_wrapper import ChocolateSB3MultiAgentEnv
from gpudrive_chocolate.networks.late_fusion_policy import LateFusionPolicy
from gpudrive_chocolate.baselines.ppo.callbacks import RolloutCaptureCallback


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return Box(yaml.safe_load(f))


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule that decays to zero."""

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def build_resume_custom_objects(env: ChocolateSB3MultiAgentEnv) -> dict[str, object]:
    # SB3 checkpoints may embed NumPy-internal module paths in pickled metadata.
    # Reuse the current env spaces and force a reset instead of deserializing them.
    return {
        "_last_obs": None,
        "_last_episode_starts": None,
        "observation_space": env.observation_space,
        "action_space": env.action_space,
    }


def train(exp_config: Box):
    if isinstance(exp_config.device, str) and exp_config.device.startswith("cuda") and ":" not in exp_config.device:
        exp_config.device = "cuda:0"
    if isinstance(exp_config.device, str) and exp_config.device.startswith("cuda"):
        torch.cuda.set_device(exp_config.device)

    env = ChocolateSB3MultiAgentEnv(
        choco_config_path=exp_config.choco_config_path,
        exp_config=exp_config,
        device=exp_config.device,
        reward_type=exp_config.reward_type,
        collision_weight=exp_config.collision_weight,
        goal_achieved_weight=exp_config.goal_achieved_weight,
        off_road_weight=exp_config.off_road_weight,
        log_distance_weight=exp_config.log_distance_weight,
    )

    exp_config.num_envs = env.num_envs
    exp_config.batch_size = (
        exp_config.num_envs * exp_config.n_steps
    ) // exp_config.num_minibatches

    run_id = datetime.now().strftime("%m_%d_%H_%S")

    policy_type = str(getattr(exp_config, "policy_type", "mlp"))
    policy_kwargs = {}
    if policy_type == "mlp":
        if hasattr(exp_config, "policy_net_arch") and exp_config.policy_net_arch:
            policy_kwargs["net_arch"] = exp_config.policy_net_arch
        policy = "MlpPolicy"
    elif policy_type == "late_fusion":
        lf_cfg = getattr(exp_config, "late_fusion", {})
        policy = LateFusionPolicy
        policy_kwargs.update(
            {
                "ego_dim": int(lf_cfg.get("ego_dim", 7)),
                "point_dim": int(lf_cfg.get("point_dim", 3)),
                "point_k": lf_cfg.get("point_k", None),
                "ego_layers": lf_cfg.get("ego_layers", [64, 64]),
                "point_layers": lf_cfg.get("point_layers", [64, 64]),
                "shared_layers": lf_cfg.get("shared_layers", [64]),
                "last_layer_dim_pi": int(lf_cfg.get("last_layer_dim_pi", 64)),
                "last_layer_dim_vf": int(lf_cfg.get("last_layer_dim_vf", 64)),
                "act": str(lf_cfg.get("act", "relu")),
                "dropout": float(lf_cfg.get("dropout", 0.0)),
                "pool": str(lf_cfg.get("pool", "max")),
            }
        )
    else:
        raise ValueError(f"Unknown policy_type: {policy_type}")

    if getattr(exp_config, "resume_from", None):
        model = PPO.load(
            exp_config.resume_from,
            env=env,
            device=exp_config.device,
            custom_objects=build_resume_custom_objects(env),
            tensorboard_log=f"runs/{run_id}",
        )
    else:
        model = PPO(
            policy=policy,
            env=env,
            n_steps=exp_config.n_steps,
            batch_size=exp_config.batch_size,
            n_epochs=exp_config.n_epochs,
            gamma=exp_config.gamma,
            gae_lambda=exp_config.gae_lambda,
            clip_range=exp_config.clip_range,
            ent_coef=exp_config.ent_coef,
            vf_coef=exp_config.vf_coef,
            learning_rate=linear_schedule(float(exp_config.lr)),
            verbose=exp_config.verbose,
            seed=exp_config.seed,
            device=exp_config.device,
            tensorboard_log=f"runs/{run_id}",
            policy_kwargs=policy_kwargs,
        )

    capture_callback = RolloutCaptureCallback(
        render_every_updates=exp_config.render_every_updates,
        render_rollout_steps=exp_config.render_rollout_steps,
        render_dir=exp_config.render_dir,
        always_render=bool(getattr(exp_config, "render_during_training", False)),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=int(exp_config.save_freq),
        save_path=str(exp_config.save_dir),
        name_prefix=str(exp_config.save_prefix),
    )

    model.learn(
        total_timesteps=exp_config.total_timesteps,
        callback=[capture_callback, checkpoint_cb],
    )
    env.close()


if __name__ == "__main__":
    exp_config = load_config("gpudrive_chocolate/config/ppo_choco_stage2.yaml")
    train(exp_config)
