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


def _cfg_int_tuple(cfg: dict[str, Any], section: str, key: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = _cfg_value(cfg, section, key, default)
    if value is None:
        return tuple(int(item) for item in default)
    if isinstance(value, str):
        tokens = [token.strip() for token in value.replace(";", ",").split(",") if token.strip()]
        return tuple(int(token) for token in tokens) if tokens else tuple(int(item) for item in default)
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return (int(value),)


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
    choices=("full", "goal_reaching", "choco_reference"),
    default=str(_cfg_value(file_cfg, "env", "observation_mode", "goal_reaching")),
    help="Policy observation preset.",
)
parser.add_argument(
    "--obs_weather_context_enable",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "observation", "weather_context_enable", True)),
    help="Include SceneFactory weather/friction context in the choco_reference observation mode.",
)
parser.add_argument(
    "--obs_road_points_enable",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "observation", "road_points_enable", True)),
)
parser.add_argument(
    "--obs_road_points_k",
    type=int,
    default=int(_cfg_value(file_cfg, "observation", "road_points_k", 64)),
)
parser.add_argument(
    "--obs_road_points_radius_m",
    type=float,
    default=float(_cfg_value(file_cfg, "observation", "road_points_radius_m", 35.0)),
)
parser.add_argument(
    "--obs_road_points_type_norm",
    type=float,
    default=float(_cfg_value(file_cfg, "observation", "road_points_type_norm", 1.0)),
)
parser.add_argument(
    "--obs_road_points_mode",
    choices=("knn", "road_running", "road-running"),
    default=str(_cfg_value(file_cfg, "observation", "road_points_mode", "knn")),
)
parser.add_argument(
    "--obs_road_points_include_dirs",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "observation", "road_points_include_dirs", True)),
)
parser.add_argument(
    "--obs_neighbor_enable",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "observation", "neighbor_enable", True)),
)
parser.add_argument(
    "--obs_neighbor_k",
    type=int,
    default=int(_cfg_value(file_cfg, "observation", "neighbor_k", 8)),
)
parser.add_argument(
    "--obs_neighbor_include_ttc",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "observation", "neighbor_include_ttc", True)),
)
parser.add_argument(
    "--obs_neighbor_include_index",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "observation", "neighbor_include_index", True)),
)
parser.add_argument(
    "--obs_neighbor_ttc_max_s",
    type=float,
    default=float(_cfg_value(file_cfg, "observation", "neighbor_ttc_max_s", 10.0)),
)
parser.add_argument(
    "--obs_timing_print_enable",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "observation", "timing_print_enable", False)),
)
parser.add_argument(
    "--obs_timing_print_every_n",
    type=int,
    default=int(_cfg_value(file_cfg, "observation", "timing_print_every_n", 32)),
)
parser.add_argument(
    "--reward_lane_center_enable",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "reward", "lane_center_enable", True)),
)
parser.add_argument(
    "--reward_lane_center_per_step",
    type=float,
    default=float(_cfg_value(file_cfg, "reward", "lane_center_per_step", 0.05)),
)
parser.add_argument(
    "--reward_lane_forbidden_enable",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "reward", "lane_forbidden_enable", True)),
)
parser.add_argument(
    "--reward_lane_forbidden_penalty",
    type=float,
    default=float(_cfg_value(file_cfg, "reward", "lane_forbidden_penalty", -30.0)),
)
parser.add_argument(
    "--reward_goal_bonus",
    type=float,
    default=float(_cfg_value(file_cfg, "reward", "goal_bonus", 20.0)),
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
parser.add_argument(
    "--scene_factory_world_selection_mode",
    choices=("fixed", "random_envs", "random-envs"),
    default=str(_cfg_value(file_cfg, "scene_factory", "world_selection_mode", "fixed")),
    help="How SceneFactory worlds are assigned across vectorized environments.",
)
parser.add_argument(
    "--scene_factory_random_world_seed",
    type=int,
    default=int(_cfg_value(file_cfg, "scene_factory", "random_world_seed", 42)),
    help="Seed used when SceneFactory world_selection_mode=random_envs.",
)
parser.add_argument(
    "--test_mode",
    choices=("none", "collision_test", "scene_factory_collision_test", "scene_factory_multiworld_random_steer_test"),
    default=str(_cfg_value(file_cfg, "test", "mode", "none")),
    help=(
        "Optional deterministic debug rollout mode. "
        "'collision_test' runs a flat-world head-on crash. "
        "'scene_factory_collision_test' runs the same head-on crash with SceneFactory roads enabled. "
        "'scene_factory_multiworld_random_steer_test' runs multiple SceneFactory worlds with full throttle and "
        "random steering."
    ),
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
    "--goal_reached_threshold_m",
    type=float,
    default=float(_cfg_value(file_cfg, "env", "goal_reached_threshold_m", 0.85)),
    help="Distance threshold for goal completion.",
)
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
    "--agent_collision_warmup_steps",
    type=int,
    default=int(_cfg_value(file_cfg, "env", "agent_collision_warmup_steps", 24)),
    help="Ignore inter-vehicle collision detection for this many environment steps after reset.",
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
parser.add_argument(
    "--silence_runtime_warnings",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "app", "silence_runtime_warnings", False)),
    help="Suppress Isaac Kit warning-level runtime logs in the terminal for this run.",
)
parser.add_argument(
    "--use_fabric",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "sim", "use_fabric", True)),
    help="Enable Isaac Lab Fabric. Disable this for SceneFactory road stage-dump/debug runs to isolate PointInstancer issues.",
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
parser.add_argument(
    "--shared_policy_mode",
    choices=("agent_slots", "joint_world"),
    default=str(_cfg_value(file_cfg, "runner", "shared_policy_mode", "agent_slots")),
    help=(
        "How to flatten the multi-agent env for PPO. "
        "'agent_slots' exposes one shared-policy slot per vehicle. "
        "'joint_world' keeps the older concatenated multi-agent world policy."
    ),
)
parser.add_argument(
    "--video",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "video", "enabled", False)),
    help="Record rollout videos during training using a fixed camera sensor.",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=int(_cfg_value(file_cfg, "video", "interval", 2500)),
    help="Global environment-step interval between recorded training videos.",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=int(_cfg_value(file_cfg, "video", "length", 300)),
    help="Number of environment steps captured in each training video.",
)
parser.add_argument(
    "--video_name_prefix",
    type=str,
    default=str(_cfg_value(file_cfg, "video", "name_prefix", "train")),
    help="Filename prefix for recorded training videos.",
)
parser.add_argument(
    "--video_width",
    type=int,
    default=int(_cfg_value(file_cfg, "video", "width", 1280)),
    help="Width in pixels for the fixed training capture camera.",
)
parser.add_argument(
    "--video_height",
    type=int,
    default=int(_cfg_value(file_cfg, "video", "height", 720)),
    help="Height in pixels for the fixed training capture camera.",
)
parser.add_argument(
    "--video_fps",
    type=int,
    default=int(_cfg_value(file_cfg, "video", "fps", 20)),
    help="Output fps for saved training clips.",
)
parser.add_argument(
    "--video_view_mode",
    choices=("whole_grid", "single_env"),
    default=str(_cfg_value(file_cfg, "video", "view_mode", "whole_grid")),
    help="Capture either the whole training grid or a single environment.",
)
parser.add_argument(
    "--video_env_index",
    type=int,
    default=int(_cfg_value(file_cfg, "video", "env_index", 0)),
    help="Environment index to focus when video_view_mode=single_env.",
)
parser.add_argument(
    "--save_stage_usd",
    type=str,
    default=str(_cfg_value(file_cfg, "debug", "save_stage_usd", "")),
    help="Optional path to export the initialized training stage before learning starts.",
)
parser.add_argument(
    "--exit_after_stage_save",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "debug", "exit_after_stage_save", False)),
    help="Exit immediately after saving the initialized stage debug dump.",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(
    device=_cfg_value(file_cfg, "app", "device", None),
    enable_cameras=bool(_cfg_value(file_cfg, "video", "enabled", False)),
)
args_cli = parser.parse_args()


def _configure_headless_camera_environment(args: argparse.Namespace) -> None:
    """Force a true offscreen path for headless camera/video runs.

    On workstation setups with an active X display, Isaac Sim can still attempt a
    GLX-backed initialization even when `--headless` is set. That breaks video
    capture with errors such as `GLXBadFBConfig`. For headless runs that need
    cameras, scrub GUI display variables before AppLauncher starts the app.
    """

    needs_offscreen_cameras = bool(getattr(args, "headless", False)) and bool(
        getattr(args, "video", False) or getattr(args, "enable_cameras", False)
    )
    if not needs_offscreen_cameras:
        return

    if os.environ.get("DISPLAY"):
        print(
            f"[INFO][SceneFactory]: Unsetting DISPLAY={os.environ['DISPLAY']} for headless camera/video rendering."
        )
        os.environ.pop("DISPLAY", None)
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("ENABLE_CAMERAS", "1")


_configure_headless_camera_environment(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _configure_runtime_warning_filter() -> None:
    if not bool(getattr(args_cli, "silence_runtime_warnings", False)):
        return
    try:
        import carb.settings

        carb_settings = carb.settings.get_settings()
        carb_settings.set_string("/log/outputStreamLevel", "Error")
        print("[INFO][SceneFactory] Suppressing Kit warning-level runtime logs for this run.")
    except Exception as exc:
        print(f"[WARN][SceneFactory] Failed to configure runtime warning filter: {exc}")


_configure_runtime_warning_filter()


import torch
import gymnasium as gym
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)

from rsl_rl.runners import on_policy_runner as rsl_on_policy_runner_module

from src.scene_factory_late_fusion_actor_critic import SceneFactoryLateFusionActorCritic
from src.student_vehicle_goal_env import DEFAULT_STUDENT_VEHICLE_USD
from src.student_vehicle_multiagent_goal_env import (
    StudentVehicleMultiAgentGoalEnv,
    StudentVehicleMultiAgentGoalEnvCfg,
    _reference_road_point_feat_dim,
    _reference_vehicle_feat_dim,
    configure_multi_agent_spaces,
)
from src.trfc import weather_context_dim


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _register_scene_factory_custom_policy_classes() -> None:
    rsl_on_policy_runner_module.SceneFactoryLateFusionActorCritic = SceneFactoryLateFusionActorCritic


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
    cfg.sim.use_fabric = bool(args_cli.use_fabric)
    use_gpu_device = not str(cfg.sim.device).lower().startswith("cpu")
    requires_camera_rendering = bool(args_cli.video) or bool(getattr(args_cli, "enable_cameras", False))
    if args_cli.clone_in_fabric == "auto":
        cfg.scene.clone_in_fabric = (
            bool(getattr(args_cli, "headless", False))
            and cfg.scene.replicate_physics
            and use_gpu_device
            and not requires_camera_rendering
        )
    else:
        cfg.scene.clone_in_fabric = (
            args_cli.clone_in_fabric == "true" and cfg.scene.replicate_physics and use_gpu_device
        )
    if cfg.scene.clone_in_fabric and requires_camera_rendering:
        print("[INFO][SceneFactory]: Disabling Fabric cloning for camera/video capture runs.")
        cfg.scene.clone_in_fabric = False
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
    cfg.scene_factory_world_selection_mode = str(args_cli.scene_factory_world_selection_mode)
    cfg.scene_factory_random_world_seed = int(args_cli.scene_factory_random_world_seed)
    if bool(cfg.use_scene_factory_roads):
        road_cfg = dict(_load_yaml_config(cfg.scene_factory_config_path).get("road", {}) or {})
        if str(road_cfg.get("render_mode", "point_instancer")).strip().lower() == "point_instancer":
            print(
                "[INFO][SceneFactory]: road_render_mode=point_instancer "
                f"use_fabric={cfg.sim.use_fabric} clone_in_fabric={cfg.scene.clone_in_fabric}"
            )
    cfg.start_radius_m = float(args_cli.start_radius_m)
    cfg.agent_spawn_circle_radius_m = float(args_cli.agent_spawn_circle_radius_m)
    cfg.agent_spawn_jitter_m = float(args_cli.agent_spawn_jitter_m)
    cfg.episode_length_s = float(args_cli.episode_length_s)
    cfg.goal_radius_min_m = float(args_cli.goal_radius_min_m)
    cfg.goal_radius_max_m = float(args_cli.goal_radius_max_m)
    cfg.goal_reached_threshold_m = float(args_cli.goal_reached_threshold_m)
    cfg.max_distance_from_origin_m = float(args_cli.max_distance_from_origin_m)
    cfg.agent_neighbor_obs_scale_m = float(args_cli.agent_neighbor_obs_scale_m)
    cfg.agent_collision_warmup_steps = int(args_cli.agent_collision_warmup_steps)
    cfg.observation_mode = str(args_cli.observation_mode)
    cfg.obs_weather_context_enable = bool(args_cli.obs_weather_context_enable)
    cfg.obs_road_points_enable = bool(args_cli.obs_road_points_enable)
    cfg.obs_road_points_k = int(args_cli.obs_road_points_k)
    cfg.obs_road_points_radius_m = float(args_cli.obs_road_points_radius_m)
    cfg.obs_road_points_type_norm = float(args_cli.obs_road_points_type_norm)
    cfg.obs_road_points_mode = str(args_cli.obs_road_points_mode)
    cfg.obs_road_points_include_dirs = bool(args_cli.obs_road_points_include_dirs)
    cfg.obs_neighbor_enable = bool(args_cli.obs_neighbor_enable)
    cfg.obs_neighbor_k = int(args_cli.obs_neighbor_k)
    cfg.obs_neighbor_include_ttc = bool(args_cli.obs_neighbor_include_ttc)
    cfg.obs_neighbor_include_index = bool(args_cli.obs_neighbor_include_index)
    cfg.obs_neighbor_ttc_max_s = float(args_cli.obs_neighbor_ttc_max_s)
    cfg.obs_timing_print_enable = bool(args_cli.obs_timing_print_enable)
    cfg.obs_timing_print_every_n = int(args_cli.obs_timing_print_every_n)
    cfg.reward_lane_center_enable = bool(args_cli.reward_lane_center_enable)
    cfg.reward_lane_center_types = _cfg_int_tuple(file_cfg, "reward", "lane_center_types", (1, 2))
    cfg.reward_lane_center_per_step = float(args_cli.reward_lane_center_per_step)
    cfg.reward_lane_forbidden_enable = bool(args_cli.reward_lane_forbidden_enable)
    cfg.reward_lane_forbidden_types = _cfg_int_tuple(file_cfg, "reward", "lane_forbidden_types", (15, 16))
    cfg.reward_lane_forbidden_penalty = float(args_cli.reward_lane_forbidden_penalty)
    cfg.reward_goal_bonus = float(args_cli.reward_goal_bonus)
    cfg.test_mode = str(args_cli.test_mode).strip().lower()
    cfg.collision_test_post_collision_steps = int(_cfg_value(file_cfg, "test", "post_collision_steps", 120))
    cfg.collision_test_post_collision_throttle = float(
        _cfg_value(file_cfg, "test", "post_collision_throttle", 0.0)
    )
    cfg.collision_test_post_collision_steering = float(
        _cfg_value(file_cfg, "test", "post_collision_steering", 0.0)
    )
    cfg.collision_test_post_collision_brake = float(_cfg_value(file_cfg, "test", "post_collision_brake", 1.0))
    cfg.random_steer_test_settle_steps = int(_cfg_value(file_cfg, "test", "settle_steps", 24))
    cfg.random_steer_test_drive_steps = int(_cfg_value(file_cfg, "test", "drive_steps", 600))
    cfg.random_steer_test_throttle = float(_cfg_value(file_cfg, "test", "throttle", 1.0))
    cfg.random_steer_test_brake = float(_cfg_value(file_cfg, "test", "brake", 0.0))
    cfg.random_steer_test_steering_min = float(_cfg_value(file_cfg, "test", "steering_min", -1.0))
    cfg.random_steer_test_steering_max = float(_cfg_value(file_cfg, "test", "steering_max", 1.0))
    cfg.random_steer_test_steering_hold_steps = int(_cfg_value(file_cfg, "test", "steering_hold_steps", 12))
    cfg.random_steer_test_seed = int(_cfg_value(file_cfg, "test", "seed", 123))
    cfg.capture_camera_enabled = bool(args_cli.video)
    cfg.capture_camera_width = int(args_cli.video_width)
    cfg.capture_camera_height = int(args_cli.video_height)
    cfg.capture_camera_view_mode = str(args_cli.video_view_mode)
    cfg.capture_camera_env_index = int(args_cli.video_env_index)
    cfg.student_usd_path = str(Path(args_cli.student_usd or DEFAULT_STUDENT_VEHICLE_USD).expanduser().resolve())
    if str(args_cli.tunable_config_json):
        cfg.tunable_config_json = str(Path(args_cli.tunable_config_json).expanduser().resolve())
    if cfg.test_mode == "collision_test":
        if cfg.scene.num_envs != 1:
            print("[INFO][SceneFactory] collision_test forces num_envs=1.")
        cfg.scene.num_envs = 1
        if int(args_cli.num_agents_per_env) != 2:
            print("[INFO][SceneFactory] collision_test forces num_agents_per_env=2.")
        args_cli.num_agents_per_env = 2
        if cfg.use_scene_factory_roads:
            print(
                "[INFO][SceneFactory] collision_test disables SceneFactory roads and uses a flat plane "
                "to isolate vehicle-vehicle contact and rendering."
            )
        cfg.use_scene_factory_roads = False
        if bool(args_cli.video) and not bool(args_cli.use_fabric):
            print("[INFO][SceneFactory] collision_test enables Fabric for headless vehicle video capture.")
        cfg.sim.use_fabric = True if bool(args_cli.video) else bool(args_cli.use_fabric)
    elif cfg.test_mode == "scene_factory_collision_test":
        if cfg.scene.num_envs != 1:
            print("[INFO][SceneFactory] scene_factory_collision_test forces num_envs=1.")
        cfg.scene.num_envs = 1
        if int(args_cli.num_agents_per_env) != 2:
            print("[INFO][SceneFactory] scene_factory_collision_test forces num_agents_per_env=2.")
        args_cli.num_agents_per_env = 2
        if not cfg.use_scene_factory_roads:
            print("[INFO][SceneFactory] scene_factory_collision_test enables SceneFactory roads.")
        cfg.use_scene_factory_roads = True
        if bool(args_cli.video) and not bool(args_cli.use_fabric):
            print("[INFO][SceneFactory] scene_factory_collision_test enables Fabric for headless vehicle video capture.")
        cfg.sim.use_fabric = True if bool(args_cli.video) else bool(args_cli.use_fabric)
    elif cfg.test_mode == "scene_factory_multiworld_random_steer_test":
        if not cfg.use_scene_factory_roads:
            print("[INFO][SceneFactory] scene_factory_multiworld_random_steer_test enables SceneFactory roads.")
        cfg.use_scene_factory_roads = True
        if bool(args_cli.video) and not bool(args_cli.use_fabric):
            print(
                "[INFO][SceneFactory] scene_factory_multiworld_random_steer_test enables Fabric "
                "for headless vehicle video capture."
            )
        cfg.sim.use_fabric = True if bool(args_cli.video) else bool(args_cli.use_fabric)
    configure_multi_agent_spaces(cfg, int(args_cli.num_agents_per_env))
    return cfg


def _build_runner_cfg(sim_device: str) -> RslRlOnPolicyRunnerCfg:
    rl_device = str(args_cli.rl_device or sim_device)
    policy_type = str(_cfg_value(file_cfg, "policy", "type", "mlp")).strip().lower().replace("-", "_")
    policy_class_name = "SceneFactoryLateFusionActorCritic" if policy_type == "late_fusion" else "ActorCritic"
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
            class_name=policy_class_name,
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


def _build_late_fusion_policy_kwargs(env_cfg: StudentVehicleMultiAgentGoalEnvCfg) -> dict[str, Any]:
    ego_dim = 7 + (int(weather_context_dim()) if bool(env_cfg.obs_weather_context_enable) else 0)
    road_point_dim = (
        int(_reference_road_point_feat_dim(env_cfg.obs_road_points_include_dirs)) if bool(env_cfg.obs_road_points_enable) else 0
    )
    road_point_k = int(env_cfg.obs_road_points_k) if bool(env_cfg.obs_road_points_enable) else 0
    vehicle_dim = (
        int(_reference_vehicle_feat_dim(env_cfg.obs_neighbor_include_ttc, env_cfg.obs_neighbor_include_index))
        if bool(env_cfg.obs_neighbor_enable)
        else 0
    )
    vehicle_k = int(env_cfg.obs_neighbor_k) if bool(env_cfg.obs_neighbor_enable) else 0
    return {
        "ego_dim": int(_cfg_value(file_cfg, "policy", "ego_dim", ego_dim)),
        "road_point_dim": int(_cfg_value(file_cfg, "policy", "road_point_dim", road_point_dim)),
        "road_point_k": int(_cfg_value(file_cfg, "policy", "road_point_k", road_point_k)),
        "vehicle_dim": int(_cfg_value(file_cfg, "policy", "vehicle_dim", vehicle_dim)),
        "vehicle_k": int(_cfg_value(file_cfg, "policy", "vehicle_k", vehicle_k)),
        "ego_layers": list(_cfg_value(file_cfg, "policy", "ego_layers", [64, 64])),
        "road_layers": list(_cfg_value(file_cfg, "policy", "road_layers", [96, 96])),
        "vehicle_layers": list(_cfg_value(file_cfg, "policy", "vehicle_layers", [96, 96])),
        "shared_layers": list(_cfg_value(file_cfg, "policy", "shared_layers", [128, 64])),
        "last_layer_dim_pi": int(_cfg_value(file_cfg, "policy", "last_layer_dim_pi", 64)),
        "last_layer_dim_vf": int(_cfg_value(file_cfg, "policy", "last_layer_dim_vf", 64)),
        "activation": str(_cfg_value(file_cfg, "policy", "activation", "relu")),
        "dropout": float(_cfg_value(file_cfg, "policy", "dropout", 0.0)),
        "pool": str(_cfg_value(file_cfg, "policy", "pool", "max")),
    }


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
    policy_type = str(_cfg_value(file_cfg, "policy", "type", "mlp")).strip().lower().replace("-", "_")
    policy_cfg: dict[str, Any] = {
        "type": policy_type,
        "class_name": str(runner_cfg.policy.class_name),
    }
    if policy_type == "late_fusion":
        policy_cfg.update(_build_late_fusion_policy_kwargs(env_cfg))
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
            "goal_reached_threshold_m": float(env_cfg.goal_reached_threshold_m),
            "max_distance_from_origin_m": float(env_cfg.max_distance_from_origin_m),
            "agent_neighbor_obs_scale_m": float(env_cfg.agent_neighbor_obs_scale_m),
            "agent_collision_warmup_steps": int(env_cfg.agent_collision_warmup_steps),
            "replicate_physics": bool(env_cfg.scene.replicate_physics),
            "clone_in_fabric": bool(env_cfg.scene.clone_in_fabric),
        },
        "scene_factory": {
            "config_path": str(env_cfg.scene_factory_config_path),
            "world_index": int(env_cfg.scene_factory_world_index),
            "world_selection_mode": str(env_cfg.scene_factory_world_selection_mode),
            "random_world_seed": int(env_cfg.scene_factory_random_world_seed),
        },
        "observation": {
            "weather_context_enable": bool(env_cfg.obs_weather_context_enable),
            "road_points_enable": bool(env_cfg.obs_road_points_enable),
            "road_points_k": int(env_cfg.obs_road_points_k),
            "road_points_radius_m": float(env_cfg.obs_road_points_radius_m),
            "road_points_type_norm": float(env_cfg.obs_road_points_type_norm),
            "road_points_mode": str(env_cfg.obs_road_points_mode),
            "road_points_include_dirs": bool(env_cfg.obs_road_points_include_dirs),
            "neighbor_enable": bool(env_cfg.obs_neighbor_enable),
            "neighbor_k": int(env_cfg.obs_neighbor_k),
            "neighbor_include_ttc": bool(env_cfg.obs_neighbor_include_ttc),
            "neighbor_include_index": bool(env_cfg.obs_neighbor_include_index),
            "neighbor_ttc_max_s": float(env_cfg.obs_neighbor_ttc_max_s),
            "timing_print_enable": bool(env_cfg.obs_timing_print_enable),
            "timing_print_every_n": int(env_cfg.obs_timing_print_every_n),
        },
        "reward": {
            "lane_center_enable": bool(env_cfg.reward_lane_center_enable),
            "lane_center_types": [int(v) for v in env_cfg.reward_lane_center_types],
            "lane_center_per_step": float(env_cfg.reward_lane_center_per_step),
            "lane_forbidden_enable": bool(env_cfg.reward_lane_forbidden_enable),
            "lane_forbidden_types": [int(v) for v in env_cfg.reward_lane_forbidden_types],
            "lane_forbidden_penalty": float(env_cfg.reward_lane_forbidden_penalty),
            "goal_bonus": float(env_cfg.reward_goal_bonus),
        },
        "policy": policy_cfg,
        "test": {
            "mode": str(env_cfg.test_mode),
            "collision_force_threshold_n": float(env_cfg.agent_collision_force_threshold_n),
            "post_collision_steps": int(env_cfg.collision_test_post_collision_steps),
            "post_collision_throttle": float(env_cfg.collision_test_post_collision_throttle),
            "post_collision_steering": float(env_cfg.collision_test_post_collision_steering),
            "post_collision_brake": float(env_cfg.collision_test_post_collision_brake),
            "settle_steps": int(env_cfg.random_steer_test_settle_steps),
            "drive_steps": int(env_cfg.random_steer_test_drive_steps),
            "throttle": float(env_cfg.random_steer_test_throttle),
            "brake": float(env_cfg.random_steer_test_brake),
            "steering_min": float(env_cfg.random_steer_test_steering_min),
            "steering_max": float(env_cfg.random_steer_test_steering_max),
            "steering_hold_steps": int(env_cfg.random_steer_test_steering_hold_steps),
            "seed": int(env_cfg.random_steer_test_seed),
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
            "shared_policy_mode": str(args_cli.shared_policy_mode),
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
        "video": {
            "enabled": bool(args_cli.video),
            "interval": int(args_cli.video_interval),
            "length": int(args_cli.video_length),
            "name_prefix": str(args_cli.video_name_prefix),
            "width": int(args_cli.video_width),
            "height": int(args_cli.video_height),
            "fps": int(args_cli.video_fps),
            "view_mode": str(args_cli.video_view_mode),
            "env_index": int(args_cli.video_env_index),
        },
        "app": {
            "device": str(env_cfg.sim.device),
            "rl_device": str(runner_cfg.device),
            "headless": bool(getattr(args_cli, "headless", False)),
            "enable_cameras": bool(getattr(args_cli, "enable_cameras", False)),
            "silence_runtime_warnings": bool(getattr(args_cli, "silence_runtime_warnings", False)),
        },
    }


def _write_run_metadata(run_dir: Path, env_cfg: StudentVehicleMultiAgentGoalEnvCfg, runner_cfg: RslRlOnPolicyRunnerCfg):
    (run_dir / "params").mkdir(parents=True, exist_ok=True)
    resolved_cfg = _build_resolved_config(env_cfg, runner_cfg)
    policy_type = str(_cfg_value(file_cfg, "policy", "type", "mlp")).strip().lower().replace("-", "_")
    policy_payload: dict[str, Any] = {
        "type": policy_type,
        "class_name": str(runner_cfg.policy.class_name),
    }
    if policy_type == "late_fusion":
        policy_payload.update(_build_late_fusion_policy_kwargs(env_cfg))
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
            "scene_factory_world_selection_mode": env_cfg.scene_factory_world_selection_mode,
            "scene_factory_random_world_seed": env_cfg.scene_factory_random_world_seed,
            "test_mode": env_cfg.test_mode,
            "collision_test_post_collision_steps": env_cfg.collision_test_post_collision_steps,
            "collision_test_post_collision_throttle": env_cfg.collision_test_post_collision_throttle,
            "collision_test_post_collision_steering": env_cfg.collision_test_post_collision_steering,
            "collision_test_post_collision_brake": env_cfg.collision_test_post_collision_brake,
            "random_steer_test_settle_steps": env_cfg.random_steer_test_settle_steps,
            "random_steer_test_drive_steps": env_cfg.random_steer_test_drive_steps,
            "random_steer_test_throttle": env_cfg.random_steer_test_throttle,
            "random_steer_test_brake": env_cfg.random_steer_test_brake,
            "random_steer_test_steering_min": env_cfg.random_steer_test_steering_min,
            "random_steer_test_steering_max": env_cfg.random_steer_test_steering_max,
            "random_steer_test_steering_hold_steps": env_cfg.random_steer_test_steering_hold_steps,
            "random_steer_test_seed": env_cfg.random_steer_test_seed,
            "start_radius_m": env_cfg.start_radius_m,
            "agent_spawn_circle_radius_m": env_cfg.agent_spawn_circle_radius_m,
            "agent_spawn_jitter_m": env_cfg.agent_spawn_jitter_m,
            "env_spacing": env_cfg.scene.env_spacing,
            "replicate_physics": env_cfg.scene.replicate_physics,
            "clone_in_fabric": env_cfg.scene.clone_in_fabric,
            "agent_collision_force_threshold_n": env_cfg.agent_collision_force_threshold_n,
            "agent_collision_warmup_steps": env_cfg.agent_collision_warmup_steps,
            "episode_length_s": env_cfg.episode_length_s,
            "goal_radius_min_m": env_cfg.goal_radius_min_m,
            "goal_radius_max_m": env_cfg.goal_radius_max_m,
            "goal_reached_threshold_m": env_cfg.goal_reached_threshold_m,
            "max_distance_from_origin_m": env_cfg.max_distance_from_origin_m,
            "agent_neighbor_obs_scale_m": env_cfg.agent_neighbor_obs_scale_m,
            "observation_mode": env_cfg.observation_mode,
            "obs_weather_context_enable": env_cfg.obs_weather_context_enable,
            "obs_road_points_enable": env_cfg.obs_road_points_enable,
            "obs_road_points_k": env_cfg.obs_road_points_k,
            "obs_road_points_radius_m": env_cfg.obs_road_points_radius_m,
            "obs_road_points_type_norm": env_cfg.obs_road_points_type_norm,
            "obs_road_points_mode": env_cfg.obs_road_points_mode,
            "obs_road_points_include_dirs": env_cfg.obs_road_points_include_dirs,
            "obs_neighbor_enable": env_cfg.obs_neighbor_enable,
            "obs_neighbor_k": env_cfg.obs_neighbor_k,
            "obs_neighbor_include_ttc": env_cfg.obs_neighbor_include_ttc,
            "obs_neighbor_include_index": env_cfg.obs_neighbor_include_index,
            "obs_neighbor_ttc_max_s": env_cfg.obs_neighbor_ttc_max_s,
            "obs_timing_print_enable": env_cfg.obs_timing_print_enable,
            "obs_timing_print_every_n": env_cfg.obs_timing_print_every_n,
            "reward_lane_center_enable": env_cfg.reward_lane_center_enable,
            "reward_lane_center_types": list(env_cfg.reward_lane_center_types),
            "reward_lane_center_per_step": env_cfg.reward_lane_center_per_step,
            "reward_lane_forbidden_enable": env_cfg.reward_lane_forbidden_enable,
            "reward_lane_forbidden_types": list(env_cfg.reward_lane_forbidden_types),
            "reward_lane_forbidden_penalty": env_cfg.reward_lane_forbidden_penalty,
            "reward_goal_bonus": env_cfg.reward_goal_bonus,
        },
        "runner_cfg": {
            **runner_cfg.to_dict(),
            "shared_policy_mode": str(args_cli.shared_policy_mode),
        },
        "policy_cfg": policy_payload,
        "video_cfg": {
            "enabled": bool(args_cli.video),
            "interval": int(args_cli.video_interval),
            "length": int(args_cli.video_length),
            "name_prefix": str(args_cli.video_name_prefix),
            "width": int(args_cli.video_width),
            "height": int(args_cli.video_height),
            "fps": int(args_cli.video_fps),
            "view_mode": str(args_cli.video_view_mode),
            "env_index": int(args_cli.video_env_index),
        },
        "app_cfg": {
            "silence_runtime_warnings": bool(getattr(args_cli, "silence_runtime_warnings", False)),
        },
    }
    (run_dir / "params" / "run.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    dump_yaml(str(run_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(run_dir / "params" / "agent.yaml"), runner_cfg)
    dump_yaml(str(run_dir / "params" / "resolved_config.yaml"), resolved_cfg)


def _maybe_save_stage_usd(save_stage_usd: str) -> None:
    if not str(save_stage_usd).strip():
        return
    import omni.usd

    save_path = Path(save_stage_usd).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    usd_context = omni.usd.get_context()
    stage = usd_context.get_stage()
    if stage is None:
        raise RuntimeError("Unable to access current USD stage for stage export.")
    print(f"[INFO][SceneFactory] Exporting initialized training stage to {save_path} ...", flush=True)
    start_time = time.time()
    stage.Export(str(save_path))
    print(
        f"[INFO][SceneFactory] Saved initialized training stage to {save_path} "
        f"in {time.time() - start_time:.2f}s",
        flush=True,
    )


def _aggregate_agent_log_dict(
    extras: dict,
    *,
    agent_ids: list[str],
    device: torch.device | str,
) -> dict:
    if not isinstance(extras, dict):
        return extras
    if "log" in extras or "episode" in extras:
        return extras

    aggregated_values: dict[str, list[torch.Tensor]] = {}
    for agent_id in agent_ids:
        agent_extra = extras.get(agent_id)
        if not isinstance(agent_extra, dict):
            continue
        agent_log = agent_extra.get("log") or agent_extra.get("episode")
        if not isinstance(agent_log, dict):
            continue
        for key, value in agent_log.items():
            if isinstance(value, torch.Tensor):
                tensor_value = value.detach().to(device=device, dtype=torch.float32).reshape(-1)
            else:
                try:
                    tensor_value = torch.tensor([float(value)], device=device, dtype=torch.float32)
                except (TypeError, ValueError):
                    continue
            aggregated_values.setdefault(key, []).append(tensor_value)

    if not aggregated_values:
        return extras

    aggregated_log = {
        key: torch.mean(torch.cat(values, dim=0))
        for key, values in aggregated_values.items()
        if values
    }
    merged_extras = dict(extras)
    merged_extras["log"] = aggregated_log
    return merged_extras


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

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, extras = self.env.reset(seed, options)
        if getattr(self, "_state_as_observation", False):
            obs = {"policy": self.env.state()}
        else:
            obs = {
                "policy": torch.cat(
                    [obs[agent].reshape(self.num_envs, -1) for agent in self.env.possible_agents],
                    dim=-1,
                )
            }
        return obs, _aggregate_agent_log_dict(
            extras,
            agent_ids=list(getattr(self.env, "possible_agents", [])),
            device=self.env.device,
        )

    def step(self, action: torch.Tensor):
        index = 0
        _actions = {}
        for agent in self.env.possible_agents:
            delta = gym.spaces.flatdim(self.env.action_spaces[agent])
            _actions[agent] = action[:, index : index + delta]
            index += delta

        obs, rewards, terminated, time_outs, extras = self.env.step(_actions)
        if getattr(self, "_state_as_observation", False):
            obs = {"policy": self.env.state()}
        else:
            obs = {
                "policy": torch.cat(
                    [obs[agent].reshape(self.num_envs, -1) for agent in self.env.possible_agents],
                    dim=-1,
                )
            }

        rewards = sum(rewards.values())
        terminated = math.prod(terminated.values()).to(dtype=torch.bool)
        time_outs = math.prod(time_outs.values()).to(dtype=torch.bool)
        return obs, rewards, terminated, time_outs, _aggregate_agent_log_dict(
            extras,
            agent_ids=list(getattr(self.env, "possible_agents", [])),
            device=self.env.device,
        )

    def __getattr__(self, key: str):
        return getattr(self.env, key)

    env._get_observations = types.MethodType(_get_observations, env)
    env.reset = types.MethodType(reset, env)
    env.step = types.MethodType(step, env)
    env_cls.__getattr__ = __getattr__
    env_cls.episode_length_buf = property(
        lambda self: self.env.episode_length_buf,
        lambda self, value: setattr(self.env, "episode_length_buf", value),
    )


class AgentSlotSharedPolicyVecEnv(VecEnv):
    """Expose one PPO slot per agent while keeping the underlying world env shared.

    This mirrors the old gpudrive_choco shared-policy setup more closely than the
    joint-world concatenation bridge. Each slot corresponds to a fixed (world, agent)
    pair, while the underlying Isaac Lab env still performs resets at the world level.
    """

    def __init__(self, env: StudentVehicleMultiAgentGoalEnv, clip_actions: float | None = None):
        self.env = env
        self.clip_actions = clip_actions
        self.cfg = env.cfg
        self.device = env.device
        self.num_worlds = int(env.num_envs)
        self.agent_ids = list(env.possible_agents)
        self.num_agents_per_world = len(self.agent_ids)
        self.num_envs = self.num_worlds * self.num_agents_per_world
        self.max_episode_length = env.max_episode_length
        self.num_actions = gym.spaces.flatdim(env.action_spaces[self.agent_ids[0]])
        self._slot_world_indices = torch.arange(self.num_worlds, device=self.device).repeat_interleave(
            self.num_agents_per_world
        )
        self._slot_episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._slot_dead_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=float,
        )
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.single_observation_space = None
        self.observation_space = None
        obs_dict, _ = self.env.reset()
        flat_obs = self._flatten_obs_dict(obs_dict)
        obs_dim = int(flat_obs.shape[-1])
        self.single_observation_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(obs_dim,),
            dtype=float,
        )
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self._slot_dead_mask = self._flatten_agent_done_mask()
        print(
            "[INFO][SceneFactory] Using per-agent shared-policy wrapper: "
            f"worlds={self.num_worlds} agents_per_world={self.num_agents_per_world} slots={self.num_envs}",
            flush=True,
        )

    @property
    def unwrapped(self) -> StudentVehicleMultiAgentGoalEnv:
        return self.env

    @property
    def render_mode(self) -> str | None:
        return getattr(self.env, "render_mode", None)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf.repeat_interleave(self.num_agents_per_world)

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        world_lengths = value.view(self.num_worlds, self.num_agents_per_world)[:, 0]
        self.env.episode_length_buf = world_lengths.to(
            device=self.env.episode_length_buf.device,
            dtype=self.env.episode_length_buf.dtype,
        )

    def seed(self, seed: int = -1) -> int:
        return self.env.seed(seed)

    def _flatten_obs_dict(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack([obs_dict[agent_id].reshape(self.num_worlds, -1) for agent_id in self.agent_ids], dim=1)
        return stacked.reshape(self.num_envs, -1)

    def _flatten_reward_or_done_dict(self, payload: dict[str, torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack([payload[agent_id].reshape(self.num_worlds) for agent_id in self.agent_ids], dim=1)
        return stacked.reshape(self.num_envs)

    def _flatten_agent_done_mask(self) -> torch.Tensor:
        return self.env._agent_done_mask.transpose(0, 1).reshape(self.num_envs)

    def _reshape_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        action_view = actions.view(self.num_worlds, self.num_agents_per_world, -1)
        return {
            agent_id: action_view[:, agent_idx, :]
            for agent_idx, agent_id in enumerate(self.agent_ids)
        }

    def reset(self) -> tuple[TensorDict, dict]:
        obs_dict, extras = self.env.reset()
        self._slot_dead_mask.zero_()
        self._slot_episode_length_buf.zero_()
        flat_obs = self._flatten_obs_dict(obs_dict)
        return TensorDict({"policy": flat_obs}, batch_size=[self.num_envs]), _aggregate_agent_log_dict(
            extras,
            agent_ids=self.agent_ids,
            device=self.device,
        )

    def get_observations(self) -> TensorDict:
        obs_dict = self.env._get_observations()
        return TensorDict({"policy": self._flatten_obs_dict(obs_dict)}, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        prev_dead_mask = self._flatten_agent_done_mask()
        prev_world_episode_length = self.env.episode_length_buf.clone()

        obs_dict, reward_dict, terminated_dict, truncated_dict, extras = self.env.step(self._reshape_actions(actions))

        flat_obs = self._flatten_obs_dict(obs_dict)
        flat_rewards = self._flatten_reward_or_done_dict(reward_dict)
        flat_terminated = self._flatten_reward_or_done_dict(terminated_dict).to(dtype=torch.bool)
        flat_truncated = self._flatten_reward_or_done_dict(truncated_dict).to(dtype=torch.bool)

        world_reset_mask = self.env.episode_length_buf < (prev_world_episode_length + 1)
        flat_world_reset_mask = world_reset_mask.repeat_interleave(self.num_agents_per_world)
        newly_done = (flat_terminated | flat_truncated) & (~prev_dead_mask)
        dones = newly_done | flat_world_reset_mask

        flat_rewards = torch.where(prev_dead_mask, torch.zeros_like(flat_rewards), flat_rewards)
        self._slot_episode_length_buf += 1
        self._slot_episode_length_buf[dones] = 0
        self._slot_dead_mask = self._flatten_agent_done_mask()

        merged_extras = _aggregate_agent_log_dict(
            extras,
            agent_ids=self.agent_ids,
            device=self.device,
        )
        if not bool(getattr(self.env.cfg, "is_finite_horizon", True)):
            merged_extras["time_outs"] = flat_truncated

        return (
            TensorDict({"policy": flat_obs}, batch_size=[self.num_envs]),
            flat_rewards,
            dones.to(dtype=torch.long),
            merged_extras,
        )

    def close(self):
        return self.env.close()


class SensorVideoRecorderWrapper(gym.Wrapper):
    """Record periodic training clips from a fixed Isaac Lab camera sensor."""

    def __init__(self, env, capture_env: StudentVehicleMultiAgentGoalEnv, run_dir: Path):
        super().__init__(env)
        self._capture_env = capture_env
        self._video_dir = run_dir / "videos" / "train"
        self._video_dir.mkdir(parents=True, exist_ok=True)
        self._interval = max(1, int(args_cli.video_interval))
        self._length = max(1, int(args_cli.video_length))
        self._fps = max(1, int(args_cli.video_fps))
        self._name_prefix = str(args_cli.video_name_prefix)
        self._global_step = 0
        self._clip_index = 0
        self._recording = False
        self._frames: list = []

    def _start_clip(self) -> None:
        if self._recording:
            return
        self._recording = True
        self._frames = []

    def _capture_frame(self) -> None:
        if not self._recording:
            return
        frame = self._capture_env.capture_fixed_camera_frame()
        if frame is not None:
            self._frames.append(frame)
        if len(self._frames) >= self._length:
            self._finish_clip()

    def _finish_clip(self) -> None:
        if not self._recording:
            return
        self._recording = False
        if len(self._frames) == 0:
            self._frames = []
            return
        import imageio.v2 as imageio

        output_path = self._video_dir / f"{self._name_prefix}_{self._clip_index:04d}.mp4"
        with imageio.get_writer(str(output_path), fps=self._fps) as writer:
            for frame in self._frames:
                writer.append_data(frame)
        self._clip_index += 1
        self._frames = []

    def reset(self, **kwargs):
        output = self.env.reset(**kwargs)
        if self._global_step % self._interval == 0:
            self._start_clip()
        self._capture_frame()
        return output

    def step(self, action):
        output = self.env.step(action)
        self._global_step += 1
        if (not self._recording) and self._global_step % self._interval == 0:
            self._start_clip()
        self._capture_frame()
        return output

    def close(self):
        self._finish_clip()
        return super().close()


def _maybe_wrap_video(env, capture_env: StudentVehicleMultiAgentGoalEnv, run_dir: Path):
    if not bool(args_cli.video):
        return env
    return SensorVideoRecorderWrapper(env, capture_env=capture_env, run_dir=run_dir)


def _collision_test_action_dict(
    env: StudentVehicleMultiAgentGoalEnv,
    throttle: float,
    steering: float,
    brake: float,
) -> dict[str, torch.Tensor]:
    action = torch.tensor(
        [float(throttle), float(steering), float(brake)],
        dtype=torch.float32,
        device=env.device,
    ).unsqueeze(0).repeat(env.num_envs, 1)
    return {agent_id: action.clone() for agent_id in env.cfg.possible_agents}


def _action_dict_from_components(
    env: StudentVehicleMultiAgentGoalEnv,
    throttle_by_agent: torch.Tensor,
    steering_by_agent: torch.Tensor,
    brake_by_agent: torch.Tensor,
) -> dict[str, torch.Tensor]:
    action_dict: dict[str, torch.Tensor] = {}
    for agent_idx, agent_id in enumerate(env.cfg.possible_agents):
        action_dict[agent_id] = torch.stack(
            [
                throttle_by_agent[agent_idx],
                steering_by_agent[agent_idx],
                brake_by_agent[agent_idx],
            ],
            dim=-1,
        ).to(device=env.device, dtype=torch.float32)
    return action_dict


def _run_collision_test(env: StudentVehicleMultiAgentGoalEnv, run_dir: Path) -> None:
    import imageio.v2 as imageio

    test_mode_name = str(env.cfg.test_mode).strip().lower() or "collision_test"
    metrics_path = run_dir / f"{test_mode_name}_metrics.jsonl"
    summary_path = run_dir / f"{test_mode_name}_summary.json"
    video_path = run_dir / "videos" / f"{test_mode_name}.mp4"
    video_writer = None
    if bool(args_cli.video):
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(str(video_path), fps=max(1, int(args_cli.video_fps)))

    def _write_frame() -> None:
        if video_writer is None:
            return
        frame = env.capture_fixed_camera_frame()
        if frame is not None:
            video_writer.append_data(frame)

    print(f"[INFO][SceneFactory] Running deterministic {test_mode_name} rollout.", flush=True)
    obs, extras = env.reset()
    _write_frame()
    collision_step: int | None = None
    max_collision_force_n = 0.0
    raw_max_collision_force_n = 0.0
    collision_detection_armed_step = int(env.cfg.collision_test_settle_steps)
    total_steps = int(
        env.cfg.collision_test_settle_steps
        + env.cfg.collision_test_drive_steps
        + env.cfg.collision_test_post_collision_steps
    )
    frames_written = 1
    post_collision_steps_remaining = int(env.cfg.collision_test_post_collision_steps)

    with metrics_path.open("w", encoding="utf-8") as handle:
        for step in range(total_steps):
            if step < int(env.cfg.collision_test_settle_steps):
                action_dict = _collision_test_action_dict(env, throttle=0.0, steering=0.0, brake=0.0)
            elif collision_step is not None:
                action_dict = _collision_test_action_dict(
                    env,
                    throttle=float(env.cfg.collision_test_post_collision_throttle),
                    steering=float(env.cfg.collision_test_post_collision_steering),
                    brake=float(env.cfg.collision_test_post_collision_brake),
                )
            else:
                action_dict = _collision_test_action_dict(
                    env,
                    throttle=float(env.cfg.collision_test_throttle),
                    steering=float(env.cfg.collision_test_steering),
                    brake=float(env.cfg.collision_test_brake),
                )

            obs, rewards, terminated, time_outs, extras = env.step(action_dict)
            _write_frame()
            frames_written += 1

            collision_force_by_agent = env.collision_force_by_agent_n()
            collision_world_force = env.collision_world_force_n()
            lane_touch_types_by_agent = env.lane_touch_types_by_agent()
            raw_collision_detected = bool(
                torch.any(collision_world_force >= float(env.cfg.agent_collision_force_threshold_n))
            )
            raw_max_collision_force_n = max(raw_max_collision_force_n, float(collision_world_force.max().item()))
            collision_detected = raw_collision_detected and step >= collision_detection_armed_step
            if collision_detected:
                max_collision_force_n = max(max_collision_force_n, float(collision_world_force.max().item()))
            if collision_detected and collision_step is None:
                collision_step = step
                print(
                    f"[INFO][SceneFactory] {test_mode_name} detected contact at "
                    f"step={step} force_n={float(collision_world_force.max().item()):.2f}",
                    flush=True,
                )
            elif collision_step is not None:
                post_collision_steps_remaining -= 1

            step_record: dict[str, Any] = {
                "step": int(step),
                "collision_detection_armed": bool(step >= collision_detection_armed_step),
                "raw_collision_detected": raw_collision_detected,
                "collision_detected": collision_detected,
                "collision_step": collision_step,
                "post_collision_steps_remaining": int(max(post_collision_steps_remaining, 0)),
                "collision_world_force_n": [float(x) for x in collision_world_force.detach().cpu().tolist()],
                "terminated": {agent_id: bool(terminated[agent_id][0].item()) for agent_id in env.cfg.possible_agents},
                "time_outs": {agent_id: bool(time_outs[agent_id][0].item()) for agent_id in env.cfg.possible_agents},
                "rewards": {agent_id: float(rewards[agent_id][0].item()) for agent_id in env.cfg.possible_agents},
                "agents": {},
            }
            for agent_idx, agent_id in enumerate(env.cfg.possible_agents):
                vehicle = env._vehicles[agent_idx]
                root_pos_w = vehicle.data.root_pos_w[0]
                root_lin_vel_w = vehicle.data.root_lin_vel_w[0]
                step_record["agents"][agent_id] = {
                    "root_pos_w": [float(x) for x in root_pos_w.detach().cpu().tolist()],
                    "root_lin_vel_w": [float(x) for x in root_lin_vel_w.detach().cpu().tolist()],
                    "planar_speed_mps": float(torch.linalg.norm(root_lin_vel_w[:2]).item()),
                    "goal_distance_m": float(env._current_goal_distance[agent_idx, 0].item()),
                    "collision_force_n": float(collision_force_by_agent[agent_id][0].item()),
                    "lane_touch_types": list(lane_touch_types_by_agent.get(agent_id, [[]])[0]),
            }
            handle.write(json.dumps(step_record) + "\n")

            if collision_step is not None and post_collision_steps_remaining <= 0:
                break

    if video_writer is not None:
        video_writer.close()

    summary = {
        "test_mode": test_mode_name,
        "collision_detection_armed_step": int(collision_detection_armed_step),
        "collision_detected": collision_step is not None,
        "collision_step": collision_step,
        "max_collision_force_n": max_collision_force_n,
        "raw_max_collision_force_n": raw_max_collision_force_n,
        "frames_written": int(frames_written),
        "video_fps": int(args_cli.video_fps) if bool(args_cli.video) else 0,
        "video_duration_s": float(frames_written / max(1, int(args_cli.video_fps))) if bool(args_cli.video) else 0.0,
        "metrics_path": str(metrics_path),
        "video_path": str(video_path) if bool(args_cli.video) else "",
        "config_path": str(Path(args_cli.config).expanduser().resolve()),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[INFO][SceneFactory] {test_mode_name} finished. "
        f"collision_detected={summary['collision_detected']} max_force_n={max_collision_force_n:.2f}",
        flush=True,
    )


def _run_scene_factory_multiworld_random_steer_test(env: StudentVehicleMultiAgentGoalEnv, run_dir: Path) -> None:
    import imageio.v2 as imageio

    test_mode_name = str(env.cfg.test_mode).strip().lower() or "scene_factory_multiworld_random_steer_test"
    metrics_path = run_dir / f"{test_mode_name}_metrics.jsonl"
    summary_path = run_dir / f"{test_mode_name}_summary.json"
    video_path = run_dir / "videos" / f"{test_mode_name}.mp4"
    video_writer = None
    if bool(args_cli.video):
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(str(video_path), fps=max(1, int(args_cli.video_fps)))

    def _write_frame() -> None:
        if video_writer is None:
            return
        frame = env.capture_fixed_camera_frame()
        if frame is not None:
            video_writer.append_data(frame)

    print(f"[INFO][SceneFactory] Running deterministic {test_mode_name} rollout.", flush=True)
    obs, extras = env.reset()
    _write_frame()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(env.cfg.random_steer_test_seed))
    total_steps = int(env.cfg.random_steer_test_settle_steps + env.cfg.random_steer_test_drive_steps)
    hold_steps = max(1, int(env.cfg.random_steer_test_steering_hold_steps))
    steering_min = float(env.cfg.random_steer_test_steering_min)
    steering_max = float(env.cfg.random_steer_test_steering_max)
    frames_written = 1
    max_collision_force_n = 0.0
    collision_step_count = 0
    worlds_with_collision: set[int] = set()
    lane_types_touched_global: set[int] = set()
    current_steering = torch.zeros((env._num_agents, env.num_envs), dtype=torch.float32, device=env.device)

    with metrics_path.open("w", encoding="utf-8") as handle:
        for step in range(total_steps):
            if step < int(env.cfg.random_steer_test_settle_steps):
                throttle_by_agent = torch.zeros((env._num_agents, env.num_envs), dtype=torch.float32, device=env.device)
                steering_by_agent = torch.zeros((env._num_agents, env.num_envs), dtype=torch.float32, device=env.device)
                brake_by_agent = torch.zeros((env._num_agents, env.num_envs), dtype=torch.float32, device=env.device)
            else:
                if (step - int(env.cfg.random_steer_test_settle_steps)) % hold_steps == 0:
                    current_steering = (
                        steering_min
                        + (steering_max - steering_min)
                        * torch.rand((env._num_agents, env.num_envs), generator=generator, dtype=torch.float32)
                    ).to(env.device)
                throttle_by_agent = torch.full(
                    (env._num_agents, env.num_envs),
                    float(env.cfg.random_steer_test_throttle),
                    dtype=torch.float32,
                    device=env.device,
                )
                steering_by_agent = current_steering
                brake_by_agent = torch.full(
                    (env._num_agents, env.num_envs),
                    float(env.cfg.random_steer_test_brake),
                    dtype=torch.float32,
                    device=env.device,
                )

            action_dict = _action_dict_from_components(env, throttle_by_agent, steering_by_agent, brake_by_agent)
            obs, rewards, terminated, time_outs, extras = env.step(action_dict)
            _write_frame()
            frames_written += 1

            collision_force_by_agent = env.collision_force_by_agent_n()
            collision_world_force = env.collision_world_force_n()
            lane_touch_types_by_agent = env.lane_touch_types_by_agent()
            collision_world_mask = collision_world_force >= float(env.cfg.agent_collision_force_threshold_n)
            if bool(torch.any(collision_world_mask).item()):
                collision_step_count += 1
                hit_env_ids = torch.nonzero(collision_world_mask, as_tuple=False).view(-1).tolist()
                for env_id in hit_env_ids:
                    worlds_with_collision.add(int(env_id))
            max_collision_force_n = max(max_collision_force_n, float(collision_world_force.max().item()))

            step_record: dict[str, Any] = {
                "step": int(step),
                "collision_world_force_n": [float(x) for x in collision_world_force.detach().cpu().tolist()],
                "collision_world_mask": [bool(x) for x in collision_world_mask.detach().cpu().tolist()],
                "envs": [],
            }
            for env_idx in range(env.num_envs):
                env_record: dict[str, Any] = {
                    "env_index": int(env_idx),
                    "collision_world_force_n": float(collision_world_force[env_idx].item()),
                    "collision_world": bool(collision_world_mask[env_idx].item()),
                    "agents": {},
                }
                for agent_idx, agent_id in enumerate(env.cfg.possible_agents):
                    vehicle = env._vehicles[agent_idx]
                    root_pos_w = vehicle.data.root_pos_w[env_idx]
                    root_lin_vel_w = vehicle.data.root_lin_vel_w[env_idx]
                    lane_types = list(lane_touch_types_by_agent.get(agent_id, [[]])[env_idx])
                    lane_types_touched_global.update(int(t) for t in lane_types)
                    env_record["agents"][agent_id] = {
                        "root_pos_w": [float(x) for x in root_pos_w.detach().cpu().tolist()],
                        "root_lin_vel_w": [float(x) for x in root_lin_vel_w.detach().cpu().tolist()],
                        "planar_speed_mps": float(torch.linalg.norm(root_lin_vel_w[:2]).item()),
                        "goal_distance_m": float(env._current_goal_distance[agent_idx, env_idx].item()),
                        "collision_force_n": float(collision_force_by_agent[agent_id][env_idx].item()),
                        "lane_touch_types": lane_types,
                        "steering_cmd": float(steering_by_agent[agent_idx, env_idx].item()),
                    }
                step_record["envs"].append(env_record)
            handle.write(json.dumps(step_record) + "\n")

    if video_writer is not None:
        video_writer.close()

    summary = {
        "test_mode": test_mode_name,
        "num_envs": int(env.num_envs),
        "num_agents_per_env": int(env._num_agents),
        "total_steps": int(total_steps),
        "collision_step_count": int(collision_step_count),
        "worlds_with_collision": sorted(int(x) for x in worlds_with_collision),
        "world_collision_count": int(len(worlds_with_collision)),
        "max_collision_force_n": float(max_collision_force_n),
        "lane_types_touched_global": sorted(int(x) for x in lane_types_touched_global),
        "frames_written": int(frames_written),
        "video_fps": int(args_cli.video_fps) if bool(args_cli.video) else 0,
        "video_duration_s": float(frames_written / max(1, int(args_cli.video_fps))) if bool(args_cli.video) else 0.0,
        "metrics_path": str(metrics_path),
        "video_path": str(video_path) if bool(args_cli.video) else "",
        "config_path": str(Path(args_cli.config).expanduser().resolve()),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[INFO][SceneFactory] {test_mode_name} finished. "
        f"world_collision_count={summary['world_collision_count']} "
        f"max_force_n={summary['max_collision_force_n']:.2f} "
        f"lane_types={summary['lane_types_touched_global']}",
        flush=True,
    )


def main():
    env_cfg = _build_env_cfg()
    runner_cfg = _build_runner_cfg(env_cfg.sim.device)
    log_root = Path(args_cli.log_dir).expanduser().resolve()
    run_dir = _make_run_dir(log_root, runner_cfg)
    print(f"[INFO] Logging RSL-RL run in: {run_dir}")

    start_time = time.time()
    base_env = StudentVehicleMultiAgentGoalEnv(env_cfg, render_mode=None)
    _write_run_metadata(run_dir, env_cfg, runner_cfg)
    _maybe_save_stage_usd(args_cli.save_stage_usd)
    if bool(args_cli.exit_after_stage_save):
        base_env.close()
        print("[INFO][SceneFactory] Exiting after stage save as requested.")
        return
    if str(args_cli.test_mode).strip().lower() in {"collision_test", "scene_factory_collision_test"}:
        try:
            _run_collision_test(base_env, run_dir)
        finally:
            base_env.close()
            print(f"[INFO] Collision test finished in {time.time() - start_time:.2f}s")
        return
    if str(args_cli.test_mode).strip().lower() == "scene_factory_multiworld_random_steer_test":
        try:
            _run_scene_factory_multiworld_random_steer_test(base_env, run_dir)
        finally:
            base_env.close()
            print(f"[INFO] Random steer test finished in {time.time() - start_time:.2f}s")
        return

    if str(args_cli.shared_policy_mode).strip().lower() == "agent_slots":
        env = AgentSlotSharedPolicyVecEnv(base_env, clip_actions=runner_cfg.clip_actions)
    else:
        env = base_env
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
            _patch_single_agent_marl_observation_bridge(env)
        env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    env = _maybe_wrap_video(env, capture_env=base_env, run_dir=run_dir)

    train_cfg = runner_cfg.to_dict()
    policy_type = str(_cfg_value(file_cfg, "policy", "type", "mlp")).strip().lower().replace("-", "_")
    if policy_type == "late_fusion":
        _register_scene_factory_custom_policy_classes()
        train_cfg["policy"]["class_name"] = "SceneFactoryLateFusionActorCritic"
        train_cfg["policy"].update(_build_late_fusion_policy_kwargs(env_cfg))
        print(
            "[INFO][SceneFactory] Using late-fusion actor-critic policy "
            f"with road_k={train_cfg['policy']['road_point_k']} vehicle_k={train_cfg['policy']['vehicle_k']}.",
            flush=True,
        )

    runner = OnPolicyRunner(env, train_cfg, log_dir=str(run_dir), device=str(runner_cfg.device))
    runner.git_status_repos = [__file__]
    try:
        runner.learn(num_learning_iterations=int(runner_cfg.max_iterations), init_at_random_ep_len=True)
    finally:
        env.close()
        print(f"[INFO] Training finished in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
    simulation_app.close()
