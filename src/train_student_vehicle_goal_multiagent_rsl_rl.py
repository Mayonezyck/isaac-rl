from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import types
from datetime import datetime
from typing import Any

import yaml

from src.isaaclab_bootstrap import ensure_isaaclab_source_paths

ensure_isaaclab_source_paths()

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp_cache")

from isaaclab.app import AppLauncher


DEFAULT_CONFIG_PATH = "configs/scene_factory/goal_reaching_train.yaml"


def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Training config root must be a mapping, got {type(payload).__name__}")
    return payload


def _cfg_value(cfg: dict[str, Any], section: str, key: str, default: Any) -> Any:
    section_payload = cfg.get(section, {}) or {}
    if not isinstance(section_payload, dict):
        return default
    return section_payload.get(key, default)


pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
pre_args, _ = pre_parser.parse_known_args()
config_path = str(Path(pre_args.config).expanduser().resolve())
file_cfg = _load_yaml_config(config_path)

parser = argparse.ArgumentParser(
    parents=[pre_parser],
    description="Train a shared-policy PPO controller for the multi-agent student-vehicle goal task using Isaac Lab RSL-RL."
)
parser.add_argument("--num_envs", type=int, default=int(_cfg_value(file_cfg, "env", "num_envs", 16)), help="Number of parallel world instances.")
parser.add_argument("--num_agents_per_env", type=int, default=int(_cfg_value(file_cfg, "env", "num_agents_per_env", 2)), help="Number of vehicles inside each world.")
parser.add_argument("--seed", type=int, default=int(_cfg_value(file_cfg, "runner", "seed", 42)), help="Random seed.")
parser.add_argument("--max_iterations", type=int, default=int(_cfg_value(file_cfg, "runner", "max_iterations", 10)), help="Number of RSL-RL PPO learning iterations.")
parser.add_argument("--log_dir", type=str, default=str(_cfg_value(file_cfg, "runner", "log_dir", "logs/rsl_rl")), help="Training log root.")
parser.add_argument("--student_usd", type=str, default=str(_cfg_value(file_cfg, "assets", "student_usd", "")), help="Path to the student vehicle USD.")
parser.add_argument(
    "--observation_mode",
    choices=("full", "goal_reaching"),
    default=str(_cfg_value(file_cfg, "env", "observation_mode", "goal_reaching")),
    help="Policy observation preset.",
)
parser.add_argument(
    "--tunable_config_json",
    type=str,
    default=str(_cfg_value(file_cfg, "assets", "tunable_config_json", "")),
    help="Path to the tuned student config JSON. Empty uses the environment default.",
)
parser.add_argument("--spawn_height_m", type=float, default=float(_cfg_value(file_cfg, "env", "spawn_height_m", 1.6)), help="Vehicle spawn height above each env origin.")
parser.add_argument(
    "--ground_mode",
    choices=("plane", "cuboid"),
    default=str(_cfg_value(file_cfg, "env", "ground_mode", "plane")),
    help="Ground implementation for the training scene.",
)
parser.add_argument(
    "--use_scene_factory_roads",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "env", "use_scene_factory_roads", False)),
    help="Build SceneFactory road geometry inside env_0 and clone it across vectorized worlds.",
)
parser.add_argument(
    "--scene_factory_config",
    type=str,
    default=str(_cfg_value(file_cfg, "scene_factory", "config_path", "configs/scene_factory/multiworld_scene.yaml")),
    help="SceneFactory YAML config used to source road geometry and scenario start/goal pairs.",
)
parser.add_argument(
    "--scene_factory_world_index",
    type=int,
    default=int(_cfg_value(file_cfg, "scene_factory", "world_index", 0)),
    help="Which resolved SceneFactory world to use as the cloned source world.",
)
parser.add_argument("--env_spacing", type=float, default=float(_cfg_value(file_cfg, "env", "env_spacing", 18.0)), help="Spacing between vectorized environments.")
parser.add_argument("--start_radius_m", type=float, default=float(_cfg_value(file_cfg, "env", "start_radius_m", 0.5)), help="Shared per-world spawn offset radius.")
parser.add_argument(
    "--agent_spawn_circle_radius_m",
    type=float,
    default=float(_cfg_value(file_cfg, "env", "agent_spawn_circle_radius_m", 3.5)),
    help="Radius of the within-world vehicle spawn ring.",
)
parser.add_argument(
    "--agent_spawn_jitter_m",
    type=float,
    default=float(_cfg_value(file_cfg, "env", "agent_spawn_jitter_m", 0.12)),
    help="Random XY jitter added to each vehicle spawn.",
)
parser.add_argument("--episode_length_s", type=float, default=float(_cfg_value(file_cfg, "env", "episode_length_s", 15.0)), help="Episode length in seconds.")
parser.add_argument("--goal_radius_min_m", type=float, default=float(_cfg_value(file_cfg, "env", "goal_radius_min_m", 5.0)), help="Minimum goal radius from env origin.")
parser.add_argument("--goal_radius_max_m", type=float, default=float(_cfg_value(file_cfg, "env", "goal_radius_max_m", 8.0)), help="Maximum goal radius from env origin.")
parser.add_argument(
    "--max_distance_from_origin_m",
    type=float,
    default=float(_cfg_value(file_cfg, "env", "max_distance_from_origin_m", 14.0)),
    help="Logical world radius used for out-of-bounds termination.",
)
parser.add_argument(
    "--agent_neighbor_obs_scale_m",
    type=float,
    default=float(_cfg_value(file_cfg, "env", "agent_neighbor_obs_scale_m", 12.0)),
    help="Normalization scale for nearest-neighbor observation features.",
)
parser.add_argument(
    "--replicate_physics",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "env", "replicate_physics", True)),
)
parser.add_argument(
    "--clone_in_fabric",
    choices=("auto", "true", "false"),
    default=str(_cfg_value(file_cfg, "env", "clone_in_fabric", "auto")),
)
parser.add_argument(
    "--rl_device",
    type=str,
    default=str(_cfg_value(file_cfg, "app", "rl_device", "")),
    help="Device for the PPO runner. Empty defaults to sim device.",
)
parser.add_argument("--num_steps_per_env", type=int, default=int(_cfg_value(file_cfg, "runner", "num_steps_per_env", 32)), help="Rollout steps per env for each PPO update.")
parser.add_argument("--save_interval", type=int, default=int(_cfg_value(file_cfg, "runner", "save_interval", 10)), help="Checkpoint save interval in iterations.")
parser.add_argument("--learning_rate", type=float, default=float(_cfg_value(file_cfg, "runner", "learning_rate", 3.0e-4)), help="PPO learning rate.")
parser.add_argument("--num_learning_epochs", type=int, default=int(_cfg_value(file_cfg, "runner", "num_learning_epochs", 5)), help="PPO epochs per update.")
parser.add_argument("--num_mini_batches", type=int, default=int(_cfg_value(file_cfg, "runner", "num_mini_batches", 4)), help="Number of PPO mini-batches per update.")
parser.add_argument("--entropy_coef", type=float, default=float(_cfg_value(file_cfg, "runner", "entropy_coef", 0.01)), help="PPO entropy coefficient.")
parser.add_argument("--clip_param", type=float, default=float(_cfg_value(file_cfg, "runner", "clip_param", 0.2)), help="PPO clip parameter.")
parser.add_argument("--desired_kl", type=float, default=float(_cfg_value(file_cfg, "runner", "desired_kl", 0.01)), help="Target KL for adaptive LR schedule.")
parser.add_argument("--experiment_name", type=str, default=str(_cfg_value(file_cfg, "runner", "experiment_name", "student_vehicle_goal_multiagent")), help="Experiment name.")
parser.add_argument("--run_name", type=str, default=str(_cfg_value(file_cfg, "runner", "run_name", "smoke")), help="Optional run-name suffix.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device=_cfg_value(file_cfg, "app", "device", None))
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)

from src.student_vehicle_goal_env import DEFAULT_STUDENT_VEHICLE_USD
from src.student_vehicle_multiagent_goal_env import (
    StudentVehicleMultiAgentGoalEnv,
    StudentVehicleMultiAgentGoalEnvCfg,
    configure_multi_agent_spaces,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _resolve_seed(seed: int) -> int:
    if int(seed) >= 0:
        return int(seed)
    return random.randint(0, 10_000)


def _build_env_cfg() -> StudentVehicleMultiAgentGoalEnvCfg:
    cfg = StudentVehicleMultiAgentGoalEnvCfg()
    cfg.seed = _resolve_seed(args_cli.seed)
    cfg.scene.num_envs = int(args_cli.num_envs)
    cfg.scene.env_spacing = float(args_cli.env_spacing)
    cfg.scene.replicate_physics = bool(args_cli.replicate_physics)
    if args_cli.device is not None:
        cfg.sim.device = str(args_cli.device)
    else:
        cfg.sim.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_gpu_device = not str(cfg.sim.device).lower().startswith("cpu")
    if args_cli.clone_in_fabric == "auto":
        cfg.scene.clone_in_fabric = (
            bool(getattr(args_cli, "headless", False)) and cfg.scene.replicate_physics and use_gpu_device
        )
    else:
        cfg.scene.clone_in_fabric = (
            args_cli.clone_in_fabric == "true" and cfg.scene.replicate_physics and use_gpu_device
        )
    grid_extent = max(1.0, math.ceil(math.sqrt(max(1, cfg.scene.num_envs))) * float(cfg.scene.env_spacing))
    world_extent = max(
        float(args_cli.max_distance_from_origin_m),
        float(args_cli.goal_radius_max_m),
        float(args_cli.agent_spawn_circle_radius_m) + 10.0,
    )
    viewer_extent = max(grid_extent, 1.25 * world_extent)
    cfg.viewer.eye = (max(20.0, viewer_extent), max(20.0, viewer_extent), max(16.0, 0.8 * viewer_extent))
    cfg.viewer.lookat = (0.0, 0.0, 0.0)
    cfg.spawn_height_m = float(args_cli.spawn_height_m)
    cfg.ground_mode = str(args_cli.ground_mode)
    cfg.use_scene_factory_roads = bool(args_cli.use_scene_factory_roads)
    cfg.scene_factory_config_path = str(Path(args_cli.scene_factory_config).expanduser().resolve())
    cfg.scene_factory_world_index = int(args_cli.scene_factory_world_index)
    cfg.start_radius_m = float(args_cli.start_radius_m)
    cfg.agent_spawn_circle_radius_m = float(args_cli.agent_spawn_circle_radius_m)
    cfg.agent_spawn_jitter_m = float(args_cli.agent_spawn_jitter_m)
    cfg.episode_length_s = float(args_cli.episode_length_s)
    cfg.goal_radius_min_m = float(args_cli.goal_radius_min_m)
    cfg.goal_radius_max_m = float(args_cli.goal_radius_max_m)
    cfg.max_distance_from_origin_m = float(args_cli.max_distance_from_origin_m)
    cfg.agent_neighbor_obs_scale_m = float(args_cli.agent_neighbor_obs_scale_m)
    cfg.observation_mode = str(args_cli.observation_mode)
    cfg.student_usd_path = str(Path(args_cli.student_usd or DEFAULT_STUDENT_VEHICLE_USD).expanduser().resolve())
    if str(args_cli.tunable_config_json):
        cfg.tunable_config_json = str(Path(args_cli.tunable_config_json).expanduser().resolve())
    configure_multi_agent_spaces(cfg, int(args_cli.num_agents_per_env))
    return cfg


def _build_runner_cfg(sim_device: str) -> RslRlOnPolicyRunnerCfg:
    rl_device = str(args_cli.rl_device or sim_device)
    return RslRlOnPolicyRunnerCfg(
        seed=int(_resolve_seed(args_cli.seed)),
        device=rl_device,
        num_steps_per_env=int(args_cli.num_steps_per_env),
        max_iterations=int(args_cli.max_iterations),
        save_interval=max(1, int(args_cli.save_interval)),
        experiment_name=str(args_cli.experiment_name),
        run_name=str(args_cli.run_name),
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        clip_actions=1.0,
        logger="tensorboard",
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=[256, 256],
            critic_hidden_dims=[256, 256],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=float(args_cli.clip_param),
            entropy_coef=float(args_cli.entropy_coef),
            num_learning_epochs=int(args_cli.num_learning_epochs),
            num_mini_batches=int(args_cli.num_mini_batches),
            learning_rate=float(args_cli.learning_rate),
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=float(args_cli.desired_kl),
            max_grad_norm=1.0,
        ),
    )


def _make_run_dir(log_root: Path, runner_cfg: RslRlOnPolicyRunnerCfg) -> Path:
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if runner_cfg.run_name:
        run_name += f"_{runner_cfg.run_name}"
    run_dir = log_root / runner_cfg.experiment_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_resolved_config(
    env_cfg: StudentVehicleMultiAgentGoalEnvCfg, runner_cfg: RslRlOnPolicyRunnerCfg
) -> dict[str, Any]:
    return {
        "env": {
            "num_envs": int(env_cfg.scene.num_envs),
            "num_agents_per_env": int(env_cfg.num_agents_per_env),
            "observation_mode": str(env_cfg.observation_mode),
            "env_spacing": float(env_cfg.scene.env_spacing),
            "spawn_height_m": float(env_cfg.spawn_height_m),
            "ground_mode": str(env_cfg.ground_mode),
            "use_scene_factory_roads": bool(env_cfg.use_scene_factory_roads),
            "start_radius_m": float(env_cfg.start_radius_m),
            "agent_spawn_circle_radius_m": float(env_cfg.agent_spawn_circle_radius_m),
            "agent_spawn_jitter_m": float(env_cfg.agent_spawn_jitter_m),
            "episode_length_s": float(env_cfg.episode_length_s),
            "goal_radius_min_m": float(env_cfg.goal_radius_min_m),
            "goal_radius_max_m": float(env_cfg.goal_radius_max_m),
            "max_distance_from_origin_m": float(env_cfg.max_distance_from_origin_m),
            "agent_neighbor_obs_scale_m": float(env_cfg.agent_neighbor_obs_scale_m),
            "replicate_physics": bool(env_cfg.scene.replicate_physics),
            "clone_in_fabric": bool(env_cfg.scene.clone_in_fabric),
        },
        "scene_factory": {
            "config_path": str(env_cfg.scene_factory_config_path),
            "world_index": int(env_cfg.scene_factory_world_index),
        },
        "assets": {
            "student_usd": str(env_cfg.student_usd_path),
            "tunable_config_json": str(env_cfg.tunable_config_json),
        },
        "runner": {
            "log_dir": str(Path(args_cli.log_dir).expanduser().resolve()),
            "seed": int(env_cfg.seed),
            "experiment_name": str(runner_cfg.experiment_name),
            "run_name": str(runner_cfg.run_name),
            "max_iterations": int(runner_cfg.max_iterations),
            "num_steps_per_env": int(runner_cfg.num_steps_per_env),
            "save_interval": int(runner_cfg.save_interval),
            "learning_rate": float(runner_cfg.algorithm.learning_rate),
            "num_learning_epochs": int(runner_cfg.algorithm.num_learning_epochs),
            "num_mini_batches": int(runner_cfg.algorithm.num_mini_batches),
            "entropy_coef": float(runner_cfg.algorithm.entropy_coef),
            "clip_param": float(runner_cfg.algorithm.clip_param),
            "desired_kl": float(runner_cfg.algorithm.desired_kl),
        },
        "app": {
            "device": str(env_cfg.sim.device),
            "rl_device": str(runner_cfg.device),
            "headless": bool(getattr(args_cli, "headless", False)),
        },
    }


def _write_run_metadata(run_dir: Path, env_cfg: StudentVehicleMultiAgentGoalEnvCfg, runner_cfg: RslRlOnPolicyRunnerCfg):
    (run_dir / "params").mkdir(parents=True, exist_ok=True)
    resolved_cfg = _build_resolved_config(env_cfg, runner_cfg)
    payload = {
        "config_path": str(Path(args_cli.config).expanduser().resolve()),
        "command": sys.orig_argv,
        "env_cfg": {
            "num_envs": env_cfg.scene.num_envs,
            "num_agents_per_env": env_cfg.num_agents_per_env,
            "sim_device": env_cfg.sim.device,
            "student_usd_path": env_cfg.student_usd_path,
            "tunable_config_json": env_cfg.tunable_config_json,
            "spawn_height_m": env_cfg.spawn_height_m,
            "ground_mode": env_cfg.ground_mode,
            "use_scene_factory_roads": env_cfg.use_scene_factory_roads,
            "scene_factory_config_path": env_cfg.scene_factory_config_path,
            "scene_factory_world_index": env_cfg.scene_factory_world_index,
            "start_radius_m": env_cfg.start_radius_m,
            "agent_spawn_circle_radius_m": env_cfg.agent_spawn_circle_radius_m,
            "agent_spawn_jitter_m": env_cfg.agent_spawn_jitter_m,
            "env_spacing": env_cfg.scene.env_spacing,
            "replicate_physics": env_cfg.scene.replicate_physics,
            "clone_in_fabric": env_cfg.scene.clone_in_fabric,
            "episode_length_s": env_cfg.episode_length_s,
            "goal_radius_min_m": env_cfg.goal_radius_min_m,
            "goal_radius_max_m": env_cfg.goal_radius_max_m,
            "max_distance_from_origin_m": env_cfg.max_distance_from_origin_m,
            "agent_neighbor_obs_scale_m": env_cfg.agent_neighbor_obs_scale_m,
            "observation_mode": env_cfg.observation_mode,
        },
        "runner_cfg": runner_cfg.to_dict(),
    }
    (run_dir / "params" / "run.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    dump_yaml(str(run_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(run_dir / "params" / "agent.yaml"), runner_cfg)
    dump_yaml(str(run_dir / "params" / "resolved_config.yaml"), resolved_cfg)


def _patch_single_agent_marl_observation_bridge(env) -> None:
    """Patch Isaac Lab's MARL->single-agent adapter with _get_observations for RSL-RL.

    Isaac Lab's multi_agent_to_single_agent helper provides reset/step but does not implement
    _get_observations(), while RslRlVecEnvWrapper calls it during runner initialization.
    """

    if not hasattr(env, "env") or not isinstance(env.env, DirectMARLEnv):
        return

    env_cls = type(env)

    def _get_observations(self):
        if getattr(self, "_state_as_observation", False):
            return {"policy": self.env.state()}
        obs = self.env._get_observations()
        return {
            "policy": torch.cat(
                [obs[agent].reshape(self.num_envs, -1) for agent in self.env.possible_agents],
                dim=-1,
            )
        }

    def __getattr__(self, key: str):
        return getattr(self.env, key)

    env._get_observations = types.MethodType(_get_observations, env)
    env_cls.__getattr__ = __getattr__
    env_cls.episode_length_buf = property(
        lambda self: self.env.episode_length_buf,
        lambda self, value: setattr(self.env, "episode_length_buf", value),
    )


def main():
    env_cfg = _build_env_cfg()
    runner_cfg = _build_runner_cfg(env_cfg.sim.device)
    log_root = Path(args_cli.log_dir).expanduser().resolve()
    run_dir = _make_run_dir(log_root, runner_cfg)
    print(f"[INFO] Logging RSL-RL run in: {run_dir}")

    start_time = time.time()
    env = StudentVehicleMultiAgentGoalEnv(env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
        _patch_single_agent_marl_observation_bridge(env)
    env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)

    _write_run_metadata(run_dir, env_cfg, runner_cfg)

    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=str(run_dir), device=str(runner_cfg.device))
    runner.add_git_repo_to_log(__file__)
    try:
        runner.learn(num_learning_iterations=int(runner_cfg.max_iterations), init_at_random_ep_len=True)
    finally:
        env.close()
        print(f"[INFO] Training finished in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
    simulation_app.close()
