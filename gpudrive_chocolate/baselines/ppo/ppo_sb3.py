import os
import sys
import yaml
from box import Box
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_schedule_fn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gpudrive_chocolate.env.sb3_wrapper import ChocolateSB3MultiAgentEnv
from gpudrive_chocolate.baselines.ppo.callbacks import RolloutCaptureCallback


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return Box(yaml.safe_load(f))


def train(exp_config: Box):
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

    policy_kwargs = {}
    if hasattr(exp_config, "policy_net_arch") and exp_config.policy_net_arch:
        policy_kwargs["net_arch"] = exp_config.policy_net_arch

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=exp_config.n_steps,
        batch_size=exp_config.batch_size,
        n_epochs=exp_config.n_epochs,
        gamma=exp_config.gamma,
        gae_lambda=exp_config.gae_lambda,
        clip_range=exp_config.clip_range,
        ent_coef=exp_config.ent_coef,
        vf_coef=exp_config.vf_coef,
        learning_rate=get_schedule_fn(exp_config.lr),
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
    )

    model.learn(total_timesteps=exp_config.total_timesteps, callback=capture_callback)
    env.close()


if __name__ == "__main__":
    exp_config = load_config("gpudrive_chocolate/config/ppo_choco_sb3.yaml")
    train(exp_config)
