from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
from datetime import datetime

from src.isaaclab_bootstrap import ensure_isaaclab_source_paths

ensure_isaaclab_source_paths()

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train a PPO policy for the student vehicle to reach random goals.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel vehicle environments.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--total_timesteps", type=int, default=3_000_000, help="Total PPO environment timesteps.")
parser.add_argument("--log_dir", type=str, default="logs/sb3/student_vehicle_goal", help="Training log root.")
parser.add_argument("--student_usd", type=str, default="", help="Path to the student vehicle USD.")
parser.add_argument(
    "--tunable_config_json",
    type=str,
    default="",
    help="Path to a tuned student config JSON. Empty uses the environment default.",
)
parser.add_argument("--spawn_height_m", type=float, default=1.6, help="Vehicle spawn height above each env origin.")
parser.add_argument(
    "--ground_mode",
    choices=("plane", "cuboid"),
    default="plane",
    help="Ground implementation for the training scene.",
)
parser.add_argument("--env_spacing", type=float, default=12.0, help="Spacing between vectorized environments.")
parser.add_argument("--episode_length_s", type=float, default=15.0, help="Episode length in seconds.")
parser.add_argument("--goal_radius_min_m", type=float, default=4.0, help="Minimum goal radius from env origin.")
parser.add_argument("--goal_radius_max_m", type=float, default=7.0, help="Maximum goal radius from env origin.")
parser.add_argument("--n_steps", type=int, default=64, help="SB3 PPO rollout horizon per environment.")
parser.add_argument("--batch_size", type=int, default=2048, help="SB3 PPO minibatch size.")
parser.add_argument("--learning_rate", type=float, default=3.0e-4, help="SB3 PPO learning rate.")
parser.add_argument("--gamma", type=float, default=0.99, help="SB3 PPO gamma.")
parser.add_argument("--gae_lambda", type=float, default=0.95, help="SB3 PPO GAE lambda.")
parser.add_argument("--ent_coef", type=float, default=0.01, help="SB3 PPO entropy coefficient.")
parser.add_argument("--vf_coef", type=float, default=0.5, help="SB3 PPO value loss coefficient.")
parser.add_argument("--clip_range", type=float, default=0.2, help="SB3 PPO clip range.")
parser.add_argument("--n_epochs", type=int, default=5, help="SB3 PPO epochs per update.")
parser.add_argument(
    "--checkpoint_every",
    type=int,
    default=250_000,
    help="Save a checkpoint every N environment timesteps.",
)
parser.add_argument(
    "--normalize_obs",
    action="store_true",
    default=False,
    help="Enable VecNormalize observation normalization.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from src.student_vehicle_goal_env import (
    DEFAULT_STUDENT_VEHICLE_USD,
    StudentVehicleGoalEnv,
    StudentVehicleGoalEnvCfg,
    build_student_vehicle_articulation_cfg,
)


def _resolve_seed(seed: int) -> int:
    if int(seed) >= 0:
        return int(seed)
    return random.randint(0, 10_000)


def _build_env_cfg() -> StudentVehicleGoalEnvCfg:
    cfg = StudentVehicleGoalEnvCfg()
    cfg.seed = _resolve_seed(args_cli.seed)
    cfg.scene.num_envs = int(args_cli.num_envs)
    cfg.scene.env_spacing = float(args_cli.env_spacing)
    if args_cli.device is not None:
        cfg.sim.device = str(args_cli.device)
    else:
        cfg.sim.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.scene.clone_in_fabric = bool(getattr(args_cli, "headless", False)) and (
        not str(cfg.sim.device).lower().startswith("cpu")
    )
    grid_extent = max(1.0, math.ceil(math.sqrt(max(1, cfg.scene.num_envs))) * float(cfg.scene.env_spacing))
    cfg.viewer.eye = (max(18.0, grid_extent), max(18.0, grid_extent), max(14.0, 0.7 * grid_extent))
    cfg.viewer.lookat = (0.0, 0.0, 0.0)
    cfg.spawn_height_m = float(args_cli.spawn_height_m)
    cfg.ground_mode = str(args_cli.ground_mode)
    cfg.episode_length_s = float(args_cli.episode_length_s)
    cfg.goal_radius_min_m = float(args_cli.goal_radius_min_m)
    cfg.goal_radius_max_m = float(args_cli.goal_radius_max_m)
    cfg.vehicle = build_student_vehicle_articulation_cfg(
        args_cli.student_usd or DEFAULT_STUDENT_VEHICLE_USD,
        spawn_height_m=float(args_cli.spawn_height_m),
    )
    if str(args_cli.tunable_config_json):
        cfg.tunable_config_json = str(Path(args_cli.tunable_config_json).expanduser().resolve())
    return cfg


def _make_run_dir(log_root: Path) -> Path:
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = log_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_run_metadata(run_dir: Path, env_cfg: StudentVehicleGoalEnvCfg, env: StudentVehicleGoalEnv) -> None:
    (run_dir / "params").mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(" ".join(sys.orig_argv) + "\n", encoding="utf-8")
    params = {
        "cli": vars(args_cli),
        "env_cfg": {
            "num_envs": env_cfg.scene.num_envs,
            "sim_device": env_cfg.sim.device,
            "student_usd": env_cfg.vehicle.spawn.usd_path,
            "tunable_config_json": env_cfg.tunable_config_json,
            "spawn_height_m": env_cfg.spawn_height_m,
            "ground_mode": env_cfg.ground_mode,
            "env_spacing": env_cfg.scene.env_spacing,
            "clone_in_fabric": env_cfg.scene.clone_in_fabric,
            "episode_length_s": env_cfg.episode_length_s,
            "goal_radius_min_m": env_cfg.goal_radius_min_m,
            "goal_radius_max_m": env_cfg.goal_radius_max_m,
        },
        "student_tunable_config": env.tunable_config_dict(),
    }
    (run_dir / "params" / "run.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")


def main():
    env_cfg = _build_env_cfg()
    log_root = Path(args_cli.log_dir).expanduser().resolve()
    run_dir = _make_run_dir(log_root)
    print(f"[INFO] Logging PPO run in: {run_dir}")

    start_time = time.time()
    env = StudentVehicleGoalEnv(env_cfg, render_mode=None)
    _write_run_metadata(run_dir, env_cfg, env)

    vec_env = Sb3VecEnvWrapper(env, fast_variant=False)
    if args_cli.normalize_obs:
        vec_env = VecNormalize(
            vec_env,
            training=True,
            norm_obs=True,
            norm_reward=False,
            clip_obs=20.0,
            gamma=float(args_cli.gamma),
        )

    policy_kwargs = {
        "activation_fn": torch.nn.Tanh,
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
    }
    agent = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=float(args_cli.learning_rate),
        n_steps=int(args_cli.n_steps),
        batch_size=int(args_cli.batch_size),
        n_epochs=int(args_cli.n_epochs),
        gamma=float(args_cli.gamma),
        gae_lambda=float(args_cli.gae_lambda),
        ent_coef=float(args_cli.ent_coef),
        vf_coef=float(args_cli.vf_coef),
        clip_range=float(args_cli.clip_range),
        seed=int(env_cfg.seed),
        tensorboard_log=str(run_dir),
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="auto",
    )

    checkpoint_freq = max(1, int(args_cli.checkpoint_every) // max(1, int(env_cfg.scene.num_envs)))
    callbacks = [
        CheckpointCallback(save_freq=checkpoint_freq, save_path=str(run_dir), name_prefix="model", verbose=2),
        LogEveryNTimesteps(n_steps=checkpoint_freq),
    ]

    try:
        agent.learn(total_timesteps=int(args_cli.total_timesteps), callback=callbacks, progress_bar=True)
    finally:
        agent.save(str(run_dir / "model"))
        if isinstance(vec_env, VecNormalize):
            vec_env.save(str(run_dir / "model_vecnormalize.pkl"))
        vec_env.close()
        print(f"[INFO] Training finished in {time.time() - start_time:.2f}s")
        print(f"[INFO] Final model: {run_dir / 'model.zip'}")


if __name__ == "__main__":
    main()
    simulation_app.close()
