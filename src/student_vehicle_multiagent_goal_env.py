from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import random
from time import perf_counter
from typing import Sequence

import gymnasium as gym
import numpy as np
import torch

from src.isaaclab_bootstrap import ensure_isaaclab_source_paths
from src.student_vehicle_goal_env import (
    DEFAULT_STUDENT_VEHICLE_USD,
    _default_tunable_config_json,
    _dry_ground_material_cfg,
    build_goal_beacon_marker,
    goal_beacon_visualization,
    _hide_ground_visuals,
    _spawn_ground,
    _source_env_vehicle_root_path,
    build_student_vehicle_articulation_cfg,
)
from src.scene_factory_multiworld_scene import (
    _build_single_world_roads_only,
    _load_yaml,
    extract_vehicle_spawns_from_json,
)
from src.student_vehicle_sysid import (
    StudentTunableConfig,
    _apply_runtime_student_dynamics,
    load_tunable_config,
    normalize_tunable_config,
)
from src.trfc import encode_weather_context, prepare_stage_world_specs, weather_context_dim

ensure_isaaclab_source_paths()

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, sample_uniform, subtract_frame_transforms


OBSERVATION_MODE_DIMS = {
    "full": 22,
    "goal_reaching": 6,
}
_COLLIDABLE_VEHICLE_BODIES = (
    "base_link",
    "front_left_wheel_link",
    "front_right_wheel_link",
    "rear_left_wheel_link",
    "rear_right_wheel_link",
)


def _load_scene_factory_lane_touch_metadata(stage, *, world_root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(str(world_root))
    if not prim.IsValid():
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    try:
        custom_data = prim.GetCustomData()
    except Exception:
        custom_data = {}
    if not isinstance(custom_data, dict):
        custom_data = {}

    points = custom_data.get("road_points_m", None)
    dirs = custom_data.get("road_point_dirs", None)
    types = custom_data.get("road_point_types", None)
    half_lengths = custom_data.get("road_point_half_lengths_m", None)
    half_widths = custom_data.get("road_point_half_widths_m", None)
    if points is None or dirs is None or types is None:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    try:
        points_np = np.asarray(points, dtype=np.float32)
        dirs_np = np.asarray(dirs, dtype=np.float32)
        types_np = np.asarray(types, dtype=np.int64)
    except Exception:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    if points_np.ndim != 2 or points_np.shape[0] == 0 or points_np.shape[1] < 2:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    if dirs_np.ndim != 2 or dirs_np.shape[0] != points_np.shape[0] or dirs_np.shape[1] < 2:
        dirs_np = np.zeros((points_np.shape[0], 2), dtype=np.float32)
    else:
        dirs_np = dirs_np[:, :2]
    norms = np.linalg.norm(dirs_np, axis=1, keepdims=True)
    dirs_np = np.divide(dirs_np, np.maximum(norms, 1.0e-6), out=np.zeros_like(dirs_np), where=norms > 1.0e-6)

    mpu = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    if not math.isfinite(mpu) or mpu <= 0.0:
        mpu = 1.0
    points_xy = points_np[:, :2] * mpu

    if half_lengths is not None:
        try:
            half_lengths_np = np.asarray(half_lengths, dtype=np.float32).reshape(-1) * mpu
        except Exception:
            half_lengths_np = np.zeros((points_np.shape[0],), dtype=np.float32)
    else:
        half_lengths_np = np.zeros((points_np.shape[0],), dtype=np.float32)

    if half_widths is not None:
        try:
            half_widths_np = np.asarray(half_widths, dtype=np.float32).reshape(-1) * mpu
        except Exception:
            half_widths_np = np.zeros((points_np.shape[0],), dtype=np.float32)
    else:
        half_widths_np = np.zeros((points_np.shape[0],), dtype=np.float32)

    if half_lengths_np.shape[0] != points_np.shape[0]:
        half_lengths_np = np.zeros((points_np.shape[0],), dtype=np.float32)
    if half_widths_np.shape[0] != points_np.shape[0]:
        half_widths_np = np.zeros((points_np.shape[0],), dtype=np.float32)

    return points_xy, dirs_np, half_lengths_np, half_widths_np, types_np.reshape(-1)


def _load_student_vehicle_dimensions_m(usd_path: str | Path) -> tuple[float, float, float]:
    default_chassis_length_m = 4.0
    default_chassis_width_m = 2.0
    default_wheelbase_m = 2.6

    usd_path = Path(usd_path).expanduser().resolve()
    meta_path = usd_path.with_name("student_vehicle_import_meta.json")
    chassis_length_m = default_chassis_length_m
    chassis_width_m = default_chassis_width_m
    wheelbase_m = default_wheelbase_m
    if meta_path.is_file():
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            spec = payload.get("spec", {}) or {}
            chassis_length_m = float(spec.get("chassis_length_m", chassis_length_m))
            chassis_width_m = float(spec.get("chassis_width_m", chassis_width_m))
            wheelbase_m = float(spec.get("wheelbase_m", wheelbase_m))
        except Exception:
            pass

    return float(chassis_length_m), float(chassis_width_m), float(wheelbase_m)


def _build_vehicle_lane_touch_circle_proxy(usd_path: str | Path) -> tuple[torch.Tensor, float]:
    chassis_length_m, chassis_width_m, wheelbase_m = _load_student_vehicle_dimensions_m(usd_path)

    half_length_m = max(0.5, 0.5 * float(chassis_length_m))
    radius_m = max(0.45, 0.55 * float(chassis_width_m))
    offset_mag_m = min(0.5 * float(wheelbase_m), max(0.0, half_length_m - 0.8 * radius_m))
    centers_b = torch.tensor(
        [
            [float(offset_mag_m), 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-float(offset_mag_m), 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return centers_b, float(radius_m)


def _scene_factory_bounds_size_from_cfg(cfg: "StudentVehicleMultiAgentGoalEnvCfg") -> float:
    scene_factory_cfg = _load_yaml(cfg.scene_factory_config_path)
    world_cfg = dict(scene_factory_cfg.get("world", {}) or {})
    return float(world_cfg.get("bounds_size_m", 200.0))


def _scene_factory_weather_context_from_spec(cfg: "StudentVehicleMultiAgentGoalEnvCfg", world_spec) -> np.ndarray:
    if world_spec is None or getattr(world_spec, "friction_estimate", None) is None:
        return np.zeros((weather_context_dim(),), dtype=np.float32)
    estimate = world_spec.friction_estimate
    return np.asarray(
        encode_weather_context(
            water_film_mm=getattr(estimate, "water_film_mm", 0.0),
            road_type=getattr(estimate, "road_type", None),
        ),
        dtype=np.float32,
    )


def _reference_road_point_feat_dim(include_dirs: bool) -> int:
    return 5 if bool(include_dirs) else 3


def _reference_vehicle_feat_dim(include_ttc: bool, include_index: bool) -> int:
    dim = 6
    if bool(include_ttc):
        dim += 1
    if bool(include_index):
        dim += 1
    return dim


def _reference_observation_dim(cfg: "StudentVehicleMultiAgentGoalEnvCfg") -> int:
    dim = 7
    if bool(cfg.obs_weather_context_enable):
        dim += int(weather_context_dim())
    if bool(cfg.obs_road_points_enable):
        dim += int(cfg.obs_road_points_k) * int(_reference_road_point_feat_dim(cfg.obs_road_points_include_dirs))
    if bool(cfg.obs_neighbor_enable):
        dim += int(cfg.obs_neighbor_k) * int(
            _reference_vehicle_feat_dim(cfg.obs_neighbor_include_ttc, cfg.obs_neighbor_include_index)
        )
    return int(dim)


def _wrap_pi_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _world_to_ego_xy_torch(dx: torch.Tensor, dy: torch.Tensor, yaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    x_ego = cy * dx + sy * dy
    y_ego = -sy * dx + cy * dy
    return x_ego, y_ego


def multi_agent_obs_dim(observation_mode: str, cfg: "StudentVehicleMultiAgentGoalEnvCfg" | None = None) -> int:
    mode = str(observation_mode).strip().lower()
    if mode == "choco_reference":
        if cfg is None:
            raise ValueError("cfg is required to size the 'choco_reference' observation mode")
        return _reference_observation_dim(cfg)
    if mode not in OBSERVATION_MODE_DIMS:
        raise ValueError(f"Unsupported observation mode: {observation_mode!r}")
    return int(OBSERVATION_MODE_DIMS[mode])


def configure_multi_agent_spaces(cfg: "StudentVehicleMultiAgentGoalEnvCfg", num_agents_per_env: int):
    agent_ids = [f"vehicle_{idx}" for idx in range(int(num_agents_per_env))]
    obs_dim = multi_agent_obs_dim(getattr(cfg, "observation_mode", "full"), cfg=cfg)
    cfg.num_agents_per_env = int(num_agents_per_env)
    cfg.possible_agents = agent_ids
    cfg.action_spaces = {
        agent_id: gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32) for agent_id in agent_ids
    }
    cfg.observation_spaces = {agent_id: obs_dim for agent_id in agent_ids}
    cfg.state_space = 0
    return cfg


def resolve_scene_factory_world_and_spawns(cfg: "StudentVehicleMultiAgentGoalEnvCfg"):
    scene_factory_cfg = _load_yaml(cfg.scene_factory_config_path)
    world_specs = prepare_stage_world_specs(scene_factory_cfg)
    if not world_specs:
        raise RuntimeError(f"No SceneFactory worlds resolved from {cfg.scene_factory_config_path}")

    vehicles_cfg = dict(scene_factory_cfg.get("vehicles", {}) or {})
    bounds_size_m = float((scene_factory_cfg.get("world", {}) or {}).get("bounds_size_m", 200.0))
    origin_mode = str((scene_factory_cfg.get("world", {}) or {}).get("origin_mode", "center"))
    origin_center_mode = str((scene_factory_cfg.get("world", {}) or {}).get("origin_center_mode", "mean"))
    requested_agents = max(int(cfg.num_agents_per_env), 1)

    requested_world_index = int(cfg.scene_factory_world_index)
    world_spec = world_specs[requested_world_index % len(world_specs)]
    spawns = extract_vehicle_spawns_from_json(
        world_spec.scene_json_path,
        bounds_size_m=bounds_size_m,
        origin_mode=origin_mode,
        origin_center_mode=origin_center_mode,
        max_controllable=requested_agents,
        require_goal_in_bounds=bool(vehicles_cfg.get("require_goal_in_bounds", True)),
        skip_if_start_in_goal=bool(vehicles_cfg.get("skip_if_start_in_goal", True)),
        goal_radius_m=float(vehicles_cfg.get("goal_radius_m", cfg.goal_reached_threshold_m)),
        start_goal_thresh_m=vehicles_cfg.get("start_goal_thresh_m"),
    )
    print(
        "[INFO] SceneFactory world selection: "
        f"world_index={world_spec.world_index} scene={world_spec.scene_json_name} spawns={len(spawns)} "
        f"crop={bounds_size_m:.1f}x{bounds_size_m:.1f}m origin_mode={origin_mode} center_mode={origin_center_mode}"
    )
    return world_spec, list(spawns)


def resolve_scene_factory_spawn_subset(cfg: "StudentVehicleMultiAgentGoalEnvCfg") -> list:
    _, spawns = resolve_scene_factory_world_and_spawns(cfg)
    return spawns


def resolve_scene_factory_env_assignments(cfg: "StudentVehicleMultiAgentGoalEnvCfg"):
    scene_factory_cfg = _load_yaml(cfg.scene_factory_config_path)
    world_specs = prepare_stage_world_specs(scene_factory_cfg)
    if not world_specs:
        raise RuntimeError(f"No SceneFactory worlds resolved from {cfg.scene_factory_config_path}")

    vehicles_cfg = dict(scene_factory_cfg.get("vehicles", {}) or {})
    bounds_size_m = float((scene_factory_cfg.get("world", {}) or {}).get("bounds_size_m", 200.0))
    origin_mode = str((scene_factory_cfg.get("world", {}) or {}).get("origin_mode", "center"))
    origin_center_mode = str((scene_factory_cfg.get("world", {}) or {}).get("origin_center_mode", "mean"))
    requested_agents = max(int(cfg.num_agents_per_env), 1)
    num_envs = max(int(cfg.scene.num_envs), 1)
    selection_mode = str(getattr(cfg, "scene_factory_world_selection_mode", "fixed")).strip().lower().replace("_", "-")
    world_seed = int(getattr(cfg, "scene_factory_random_world_seed", cfg.seed if getattr(cfg, "seed", None) is not None else 0))

    if selection_mode == "fixed":
        requested_world_index = int(cfg.scene_factory_world_index)
        selected_specs = [world_specs[requested_world_index % len(world_specs)] for _ in range(num_envs)]
    elif selection_mode == "random-envs":
        rng = random.Random(world_seed)
        order = list(range(len(world_specs)))
        selected_specs = []
        while len(selected_specs) < num_envs:
            rng.shuffle(order)
            for spec_idx in order:
                selected_specs.append(world_specs[spec_idx])
                if len(selected_specs) >= num_envs:
                    break
    else:
        raise ValueError(f"Unsupported scene_factory_world_selection_mode: {selection_mode!r}")

    per_env_spawns: list[list] = []
    per_env_specs: list = []
    min_available = requested_agents
    for env_index, world_spec in enumerate(selected_specs):
        spawns = extract_vehicle_spawns_from_json(
            world_spec.scene_json_path,
            bounds_size_m=bounds_size_m,
            origin_mode=origin_mode,
            origin_center_mode=origin_center_mode,
            max_controllable=requested_agents,
            require_goal_in_bounds=bool(vehicles_cfg.get("require_goal_in_bounds", True)),
            skip_if_start_in_goal=bool(vehicles_cfg.get("skip_if_start_in_goal", True)),
            goal_radius_m=float(vehicles_cfg.get("goal_radius_m", cfg.goal_reached_threshold_m)),
            start_goal_thresh_m=vehicles_cfg.get("start_goal_thresh_m"),
        )
        if len(spawns) <= 0:
            raise RuntimeError(
                "SceneFactory source world does not provide any controllable spawns "
                f"for env_{env_index}: {world_spec.scene_json_name}"
            )
        min_available = min(min_available, len(spawns))
        per_env_specs.append(world_spec)
        per_env_spawns.append(list(spawns))

    if min_available < requested_agents:
        print(
            "[WARN] SceneFactory selected worlds provide fewer controllable spawns than requested: "
            f"requested={requested_agents}, available_min={min_available}. Training with {min_available} agents."
        )

    num_agents = min(requested_agents, min_available)
    trimmed_spawns = [spawns[:num_agents] for spawns in per_env_spawns]
    scenes = ", ".join(f"env_{env_idx}:{spec.scene_json_name}" for env_idx, spec in enumerate(per_env_specs))
    print(
        "[INFO] SceneFactory env assignments: "
        f"selection_mode={selection_mode} num_envs={num_envs} agents={num_agents} scenes=[{scenes}]"
    )
    return per_env_specs, trimmed_spawns


def _scene_factory_road_render_mode(cfg: "StudentVehicleMultiAgentGoalEnvCfg") -> str:
    scene_factory_cfg = _load_yaml(cfg.scene_factory_config_path)
    road_cfg = dict(scene_factory_cfg.get("road", {}) or {})
    return str(road_cfg.get("render_mode", "point_instancer")).strip().lower()


@configclass
class StudentVehicleMultiAgentGoalEnvCfg(DirectMARLEnvCfg):
    episode_length_s = 15.0
    decimation = 4
    debug_vis = True
    ui_window_class_type = None

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="min",
            restitution_combine_mode="min",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=12.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    num_agents_per_env: int = 2
    possible_agents = ["vehicle_0", "vehicle_1"]
    action_spaces = {
        "vehicle_0": gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        "vehicle_1": gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
    }
    observation_mode: str = "full"
    observation_spaces = {
        "vehicle_0": OBSERVATION_MODE_DIMS["full"],
        "vehicle_1": OBSERVATION_MODE_DIMS["full"],
    }
    state_space = 0

    student_usd_path: str = DEFAULT_STUDENT_VEHICLE_USD
    tunable_config_json: str = _default_tunable_config_json()

    spawn_height_m: float = 1.6
    ground_mode: str = "plane"
    use_scene_factory_roads: bool = False
    scene_factory_config_path: str = "configs/scene_factory/multiworld_scene.yaml"
    scene_factory_world_index: int = 0
    scene_factory_world_selection_mode: str = "fixed"
    scene_factory_random_world_seed: int = 42
    start_radius_m: float = 0.5
    agent_spawn_circle_radius_m: float = 3.5
    agent_spawn_jitter_m: float = 0.12
    randomize_spawn_phase: bool = True
    spawn_yaw_noise_rad: float = 0.5
    goal_heading_noise_rad: float = 0.75
    apply_runtime_external_wrench: bool = True
    goal_radius_min_m: float = 5.0
    goal_radius_max_m: float = 8.0
    goal_height_m: float = 0.05
    goal_reached_threshold_m: float = 0.85
    fall_height_threshold_m: float = 0.18
    bad_tilt_gravity_threshold: float = -0.15
    max_distance_from_origin_m: float = 14.0
    agent_neighbor_obs_scale_m: float = 12.0
    agent_safe_distance_m: float = 2.0
    agent_collision_distance_m: float = 1.1
    agent_collision_force_threshold_n: float = 25.0
    agent_collision_warmup_steps: int = 24
    lane_touch_enabled: bool = True
    lane_touch_margin_m: float = 0.40
    reward_lane_center_enable: bool = True
    reward_lane_center_types: tuple[int, ...] = (1, 2)
    reward_lane_center_per_step: float = 0.05
    reward_lane_forbidden_enable: bool = True
    reward_lane_forbidden_types: tuple[int, ...] = (15, 16)
    reward_lane_forbidden_penalty: float = -30.0
    obs_weather_context_enable: bool = True
    obs_road_points_enable: bool = True
    obs_road_points_k: int = 64
    obs_road_points_radius_m: float = 35.0
    obs_road_points_type_norm: float = 1.0
    obs_road_points_mode: str = "knn"
    obs_road_points_include_dirs: bool = True
    obs_neighbor_enable: bool = True
    obs_neighbor_k: int = 8
    obs_neighbor_include_ttc: bool = True
    obs_neighbor_include_index: bool = True
    obs_neighbor_ttc_max_s: float = 10.0
    obs_timing_print_enable: bool = False
    obs_timing_print_every_n: int = 32

    test_mode: str = "none"
    collision_test_half_distance_m: float = 12.0
    collision_test_goal_distance_m: float = 40.0
    collision_test_settle_steps: int = 24
    collision_test_drive_steps: int = 360
    collision_test_post_collision_steps: int = 120
    collision_test_throttle: float = 0.85
    collision_test_steering: float = 0.0
    collision_test_brake: float = 0.0
    collision_test_post_collision_throttle: float = 0.0
    collision_test_post_collision_steering: float = 0.0
    collision_test_post_collision_brake: float = 1.0
    collision_test_debug_markers: bool = False
    random_steer_test_settle_steps: int = 24
    random_steer_test_drive_steps: int = 600
    random_steer_test_throttle: float = 1.0
    random_steer_test_brake: float = 0.0
    random_steer_test_steering_min: float = -1.0
    random_steer_test_steering_max: float = 1.0
    random_steer_test_steering_hold_steps: int = 12
    random_steer_test_seed: int = 123

    reward_scale_alive: float = 0.05
    reward_scale_progress: float = 10.0
    reward_scale_goal_shaping: float = 1.5
    reward_scale_heading: float = 0.35
    reward_scale_speed_to_goal: float = 0.20
    reward_scale_lateral_velocity: float = -0.08
    reward_scale_yaw_rate: float = -0.03
    reward_scale_action_rate: float = -0.02
    reward_scale_action_magnitude: float = -0.002
    reward_scale_throttle_brake_conflict: float = -0.10
    reward_scale_neighbor_proximity: float = -0.20
    reward_goal_bonus: float = 20.0
    reward_collision_penalty: float = -15.0
    reward_crash_penalty: float = -10.0

    capture_camera_enabled: bool = False
    capture_camera_width: int = 1280
    capture_camera_height: int = 720
    capture_camera_focal_length: float = 24.0
    capture_camera_horizontal_aperture: float = 20.955
    capture_camera_padding_scale: float = 1.35
    capture_camera_height_scale: float = 1.6
    capture_camera_view_mode: str = "whole_grid"
    capture_camera_env_index: int = 0


class StudentVehicleMultiAgentGoalEnv(DirectMARLEnv):
    cfg: StudentVehicleMultiAgentGoalEnvCfg

    def __init__(self, cfg: StudentVehicleMultiAgentGoalEnvCfg, render_mode: str | None = None, **kwargs):
        self._capture_camera: Camera | None = None
        self._scenario_spawns: list | None = None
        self._scenario_spawns_by_env: list[list] | None = None
        self._scene_factory_scene_json_path: str | None = None
        self._scene_factory_scene_json_paths_by_env: list[str] | None = None
        self._scene_factory_specs_by_env: list | None = None
        self._scene_factory_bounds_size_m = float(_scene_factory_bounds_size_from_cfg(cfg))
        self._weather_context_np = np.zeros((weather_context_dim(),), dtype=np.float32)
        self._weather_context = torch.zeros((0, weather_context_dim()), dtype=torch.float32)
        self._obs_timing_call_count = 0
        self._obs_timing_last_ms = 0.0
        self._obs_timing_ema_ms = 0.0
        self._collision_sensor_names_by_agent: list[list[str]] = []
        self._lane_touch_points_xy_m = torch.zeros((0, 0, 2), dtype=torch.float32)
        self._lane_touch_dirs_xy = torch.zeros((0, 0, 2), dtype=torch.float32)
        self._lane_touch_half_lengths_m = torch.zeros((0, 0), dtype=torch.float32)
        self._lane_touch_half_widths_m = torch.zeros((0, 0), dtype=torch.float32)
        self._lane_touch_types = torch.zeros((0, 0), dtype=torch.long)
        self._lane_touch_valid = torch.zeros((0, 0), dtype=torch.bool)
        self._lane_touch_type_dim = 1
        self._lane_touch_circle_centers_b = torch.zeros((3, 3), dtype=torch.float32)
        self._lane_touch_circle_radius_m = 1.0
        self._lane_touch_mask = torch.zeros((0, 0, 1), dtype=torch.bool)
        if bool(cfg.use_scene_factory_roads) and str(cfg.test_mode).strip().lower() != "collision_test":
            resolved_specs, resolved_spawns_by_env = resolve_scene_factory_env_assignments(cfg)
            cfg.scene_factory_world_index = int(resolved_specs[0].world_index)
            cfg.num_agents_per_env = min(int(cfg.num_agents_per_env), len(resolved_spawns_by_env[0]))
            self._scene_factory_specs_by_env = list(resolved_specs)
            self._scenario_spawns_by_env = [list(spawns[: int(cfg.num_agents_per_env)]) for spawns in resolved_spawns_by_env]
            self._scenario_spawns = list(self._scenario_spawns_by_env[0])
            self._scene_factory_scene_json_paths_by_env = [
                str(Path(spec.scene_json_path).expanduser().resolve()) for spec in resolved_specs
            ]
            self._scene_factory_scene_json_path = str(self._scene_factory_scene_json_paths_by_env[0])
            self._weather_context_np = np.stack(
                [_scene_factory_weather_context_from_spec(cfg, spec) for spec in resolved_specs],
                axis=0,
            ).astype(np.float32)
        configure_multi_agent_spaces(cfg, cfg.num_agents_per_env)
        self._tunable_config = normalize_tunable_config(
            load_tunable_config(cfg.tunable_config_json) if str(cfg.tunable_config_json) else StudentTunableConfig()
        )
        vehicle_length_m, vehicle_width_m, _wheelbase_m = _load_student_vehicle_dimensions_m(cfg.student_usd_path)
        self._vehicle_length_m = float(vehicle_length_m)
        self._vehicle_width_m = float(vehicle_width_m)
        circle_centers_b, circle_radius_m = _build_vehicle_lane_touch_circle_proxy(cfg.student_usd_path)
        self._lane_touch_circle_centers_b = circle_centers_b
        self._lane_touch_circle_radius_m = float(circle_radius_m)
        self._agent_ids = list(cfg.possible_agents)
        self._collision_sensor_names_by_agent = [[] for _ in self._agent_ids]
        super().__init__(cfg, render_mode, **kwargs)
        self._configure_capture_camera_pose()
        self._lane_touch_circle_centers_b = self._lane_touch_circle_centers_b.to(self.device)
        weather_context = torch.as_tensor(self._weather_context_np, dtype=torch.float32, device=self.device)
        if weather_context.ndim == 1:
            weather_context = weather_context.unsqueeze(0).repeat(self.num_envs, 1)
        self._weather_context = weather_context

        self._num_agents = len(self._agent_ids)
        self._vehicles = [self.scene.articulations[agent_id] for agent_id in self._agent_ids]

        self._raw_actions = torch.zeros(self._num_agents, self.num_envs, 3, device=self.device)
        self._semantic_actions = torch.zeros_like(self._raw_actions)
        self._previous_raw_actions = torch.zeros_like(self._raw_actions)
        self._goal_pos_w = torch.zeros(self._num_agents, self.num_envs, 3, device=self.device)
        self._previous_goal_distance = torch.zeros(self._num_agents, self.num_envs, device=self.device)
        self._current_goal_distance = torch.zeros(self._num_agents, self.num_envs, device=self.device)
        self._terminal_goal_distance = torch.zeros(self._num_agents, self.num_envs, device=self.device)
        self._agent_done_mask = torch.zeros(self._num_agents, self.num_envs, dtype=torch.bool, device=self.device)
        self._goal_done_mask = torch.zeros(self._num_agents, self.num_envs, dtype=torch.bool, device=self.device)
        self._collision_done_mask = torch.zeros(self._num_agents, self.num_envs, dtype=torch.bool, device=self.device)
        self._crash_done_mask = torch.zeros(self._num_agents, self.num_envs, dtype=torch.bool, device=self.device)
        self._lane_forbidden_done_mask = torch.zeros(
            self._num_agents, self.num_envs, dtype=torch.bool, device=self.device
        )
        self._pending_goal_done_mask = torch.zeros(self._num_agents, self.num_envs, dtype=torch.bool, device=self.device)
        self._pending_collision_done_mask = torch.zeros(self._num_agents, self.num_envs, dtype=torch.bool, device=self.device)
        self._pending_crash_done_mask = torch.zeros(self._num_agents, self.num_envs, dtype=torch.bool, device=self.device)
        self._pending_lane_forbidden_done_mask = torch.zeros(
            self._num_agents, self.num_envs, dtype=torch.bool, device=self.device
        )

        self._steer_joint_ids: list[list[int]] = []
        self._drive_joint_ids: list[list[int]] = []
        self._brake_joint_ids: list[list[int]] = []
        self._wheel_joint_ids: list[list[int]] = []
        self._suspension_joint_ids: list[list[int]] = []
        self._base_body_id: list[list[int]] = []
        self._base_body_ids: list[torch.Tensor] = []
        self._joint_effort_targets: list[torch.Tensor] = []
        self._external_forces: list[torch.Tensor] = []
        self._external_torques: list[torch.Tensor] = []
        self._brake_sign_memory: list[torch.Tensor] = []

        for vehicle in self._vehicles:
            steer_joint_ids, _ = vehicle.find_joints(
                ["front_left_steer_joint", "front_right_steer_joint"], preserve_order=True
            )
            drive_joint_ids, _ = vehicle.find_joints(
                ["front_left_wheel_joint", "front_right_wheel_joint"], preserve_order=True
            )
            brake_joint_ids, _ = vehicle.find_joints(
                [
                    "front_left_wheel_joint",
                    "front_right_wheel_joint",
                    "rear_left_wheel_joint",
                    "rear_right_wheel_joint",
                ],
                preserve_order=True,
            )
            suspension_joint_ids, _ = vehicle.find_joints(
                [
                    "front_left_suspension_joint",
                    "front_right_suspension_joint",
                    "rear_left_suspension_joint",
                    "rear_right_suspension_joint",
                ],
                preserve_order=True,
            )
            base_body_id, _ = vehicle.find_bodies("base_link")
            self._steer_joint_ids.append(list(steer_joint_ids))
            self._drive_joint_ids.append(list(drive_joint_ids))
            self._brake_joint_ids.append(list(brake_joint_ids))
            self._wheel_joint_ids.append(list(brake_joint_ids))
            self._suspension_joint_ids.append(list(suspension_joint_ids))
            self._base_body_id.append(list(base_body_id))
            self._base_body_ids.append(torch.tensor(base_body_id, dtype=torch.int32, device=self.device))
            self._joint_effort_targets.append(torch.zeros(self.num_envs, vehicle.num_joints, device=self.device))
            self._external_forces.append(torch.zeros(self.num_envs, len(base_body_id), 3, device=self.device))
            self._external_torques.append(torch.zeros(self.num_envs, len(base_body_id), 3, device=self.device))
            self._brake_sign_memory.append(torch.ones(self.num_envs, len(brake_joint_ids), device=self.device))

            vehicle.write_joint_viscous_friction_coefficient_to_sim(
                joint_viscous_friction_coeff=torch.full(
                    (self.num_envs, len(steer_joint_ids)),
                    float(self._tunable_config.steering_viscous_friction),
                    device=self.device,
                ),
                joint_ids=steer_joint_ids,
            )
            vehicle.write_joint_viscous_friction_coefficient_to_sim(
                joint_viscous_friction_coeff=torch.full(
                    (self.num_envs, len(brake_joint_ids)),
                    float(self._tunable_config.wheel_viscous_friction),
                    device=self.device,
                ),
                joint_ids=brake_joint_ids,
            )
            vehicle.write_joint_viscous_friction_coefficient_to_sim(
                joint_viscous_friction_coeff=torch.full(
                    (self.num_envs, len(suspension_joint_ids)),
                    float(self._tunable_config.suspension_viscous_friction),
                    device=self.device,
                ),
                joint_ids=suspension_joint_ids,
            )

        self._steer_limit = float(self._tunable_config.steering_limit_rad)
        self._dry_longitudinal_scale = float(self._tunable_config.surface_longitudinal_scale.get("dry_asphalt", 1.0))
        self._dry_lateral_scale = float(self._tunable_config.surface_lateral_scale.get("dry_asphalt", 1.0))

        reward_keys = (
            "alive",
            "progress",
            "goal_shaping",
            "heading",
            "speed_to_goal",
            "lateral_velocity",
            "yaw_rate",
            "action_rate",
            "action_magnitude",
            "throttle_brake_conflict",
            "neighbor_proximity",
            "goal_bonus",
            "collision_penalty",
            "crash_penalty",
            "lane_center_bonus",
            "lane_forbidden_penalty",
        )
        self._episode_sums = {
            key: torch.zeros(self._num_agents, self.num_envs, dtype=torch.float32, device=self.device)
            for key in reward_keys
        }
        self._lane_touch_mask = torch.zeros(
            (self._num_agents, self.num_envs, int(self._lane_touch_type_dim)),
            dtype=torch.bool,
            device=self.device,
        )

        self.set_debug_vis(bool(self.cfg.debug_vis))

    def _setup_scene(self):
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        spawned_vehicles: dict[str, Articulation] = {}
        for agent_idx, agent_id in enumerate(self._agent_ids):
            prim_path = f"/World/envs/env_.*/Vehicle_{agent_idx}"
            vehicle_cfg = build_student_vehicle_articulation_cfg(
                self.cfg.student_usd_path,
                spawn_height_m=float(self.cfg.spawn_height_m),
                prim_path=prim_path,
            )
            vehicle = Articulation(vehicle_cfg)
            spawned_vehicles[agent_id] = vehicle
            _apply_runtime_student_dynamics(
                stage=stage,
                student_root_path=_source_env_vehicle_root_path(prim_path),
                config=self._tunable_config,
            )

        _spawn_ground("/World/ground", _dry_ground_material_cfg(self._tunable_config), mode=self.cfg.ground_mode)
        if self.cfg.use_scene_factory_roads and str(self.cfg.ground_mode).strip().lower() == "plane":
            _hide_ground_visuals("/World/ground")

        self.scene.clone_environments(copy_from_source=False)
        if self.cfg.use_scene_factory_roads:
            print("[INFO] Building SceneFactory roads independently inside each env after cloning.")
            self._build_scene_factory_worlds(stage)
            self._initialize_lane_touch_metadata(stage)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        for agent_id, vehicle in spawned_vehicles.items():
            # Register scene entities after cloning to match Isaac Lab's direct MARL task setup.
            self.scene.articulations[agent_id] = vehicle
        self._register_vehicle_contact_sensors()

        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self._spawn_capture_camera()

    def _vehicle_collision_filter_paths(self, sensor_agent_idx: int) -> list[str]:
        filter_paths: list[str] = []
        for other_agent_idx in range(len(self._agent_ids)):
            if other_agent_idx == sensor_agent_idx:
                continue
            for body_name in _COLLIDABLE_VEHICLE_BODIES:
                filter_paths.append(f"/World/envs/env_.*/Vehicle_{other_agent_idx}/{body_name}")
        return filter_paths

    def _register_vehicle_contact_sensors(self) -> None:
        if len(self._agent_ids) <= 1:
            return
        for agent_idx, agent_id in enumerate(self._agent_ids):
            filter_paths = self._vehicle_collision_filter_paths(agent_idx)
            for body_name in _COLLIDABLE_VEHICLE_BODIES:
                sensor_name = f"{agent_id}_contact_{body_name}"
                sensor_cfg = ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Vehicle_{agent_idx}/{body_name}",
                    update_period=0.0,
                    debug_vis=False,
                    filter_prim_paths_expr=filter_paths,
                )
                self.scene.sensors[sensor_name] = ContactSensor(sensor_cfg)
                self._collision_sensor_names_by_agent[agent_idx].append(sensor_name)

    def _spawn_capture_camera(self) -> None:
        if not bool(self.cfg.capture_camera_enabled):
            self._capture_camera = None
            return

        camera_cfg = CameraCfg(
            prim_path="/World/SceneFactoryCaptureCamera",
            update_period=0.0,
            height=int(self.cfg.capture_camera_height),
            width=int(self.cfg.capture_camera_width),
            data_types=["rgb"],
            colorize_instance_id_segmentation=False,
            colorize_instance_segmentation=False,
            colorize_semantic_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=float(self.cfg.capture_camera_focal_length),
                focus_distance=400.0,
                horizontal_aperture=float(self.cfg.capture_camera_horizontal_aperture),
                clipping_range=(0.1, 1.0e6),
            ),
        )
        self._capture_camera = Camera(camera_cfg)

    def _configure_capture_camera_pose(self) -> None:
        if self._capture_camera is None:
            return

        env_origins = self.scene.env_origins.detach()
        if str(self.cfg.test_mode).strip().lower() in {"collision_test", "scene_factory_collision_test"}:
            env_index = int(np.clip(int(self.cfg.capture_camera_env_index), 0, max(self.num_envs - 1, 0)))
            scene_center = env_origins[env_index]
            eye = torch.tensor(
                [[float(scene_center[0]), float(scene_center[1] - 34.0), 14.0]],
                dtype=torch.float32,
                device=self.device,
            )
            target = torch.tensor(
                [[float(scene_center[0]), float(scene_center[1]), 1.2]],
                dtype=torch.float32,
                device=self.device,
            )
            self._capture_camera.set_world_poses_from_view(eyes=eye, targets=target)
            return

        view_mode = str(self.cfg.capture_camera_view_mode).strip().lower()
        if view_mode == "single_env":
            env_index = int(np.clip(int(self.cfg.capture_camera_env_index), 0, max(self.num_envs - 1, 0)))
            scene_center = env_origins[env_index]
            xy_extent = 0.0
            coverage_radius = float(self.cfg.max_distance_from_origin_m) * float(self.cfg.capture_camera_padding_scale)
        else:
            scene_center = env_origins.mean(dim=0)
            if self.num_envs > 1:
                xy_extent = torch.linalg.norm(env_origins[:, :2] - scene_center[:2], dim=1).max().item()
            else:
                xy_extent = 0.0
            coverage_radius = (
                float(xy_extent)
                + float(self.cfg.max_distance_from_origin_m)
                + float(self.cfg.scene.env_spacing) * 0.25
            ) * float(self.cfg.capture_camera_padding_scale)
        capture_height = max(
            40.0,
            float(self.cfg.capture_camera_height_scale) * float(max(25.0, coverage_radius)),
        )
        eye = torch.tensor(
            [[float(scene_center[0]), float(scene_center[1]), capture_height]],
            dtype=torch.float32,
            device=self.device,
        )
        target = torch.tensor(
            [[float(scene_center[0]), float(scene_center[1]), 0.0]],
            dtype=torch.float32,
            device=self.device,
        )
        self._capture_camera.set_world_poses_from_view(eyes=eye, targets=target)

    def capture_fixed_camera_frame(self) -> np.ndarray | None:
        if self._capture_camera is None:
            return None
        self._capture_camera.update(self.step_dt)
        rgb = self._capture_camera.data.output.get("rgb")
        if rgb is None or rgb.numel() == 0:
            return None
        return rgb[0].detach().cpu().numpy().copy()

    def _build_scene_factory_world(self, stage, *, world_root: str, env_index: int = 0) -> None:
        scene_factory_cfg = _load_yaml(self.cfg.scene_factory_config_path)
        if self._scene_factory_scene_json_paths_by_env is not None:
            scene_json_path = self._scene_factory_scene_json_paths_by_env[int(env_index)]
        elif self._scene_factory_scene_json_path:
            scene_json_path = self._scene_factory_scene_json_path
        else:
            world_spec, _ = resolve_scene_factory_world_and_spawns(self.cfg)
            scene_json_path = str(Path(world_spec.scene_json_path).expanduser().resolve())
            self._scene_factory_scene_json_path = scene_json_path
        if self._scenario_spawns is None:
            self._scenario_spawns = resolve_scene_factory_spawn_subset(self.cfg)[: int(self.cfg.num_agents_per_env)]
        build_cfg = dict(scene_factory_cfg)
        build_cfg["world"] = dict(scene_factory_cfg.get("world", {}) or {})
        _build_single_world_roads_only(
            stage=stage,
            cfg=build_cfg,
            json_path=scene_json_path,
            world_root=world_root,
        )
        self._build_scene_factory_visual_floor(stage, world_root=world_root)

    def _build_scene_factory_worlds(self, stage) -> None:
        for env_index in range(int(self.cfg.scene.num_envs)):
            world_root = f"/World/envs/env_{env_index}/SceneFactoryWorlds/world_000"
            self._build_scene_factory_world(stage, world_root=world_root, env_index=env_index)

    def _initialize_lane_touch_metadata(self, stage) -> None:
        agent_count = len(self._agent_ids)
        if not bool(self.cfg.lane_touch_enabled) or not bool(self.cfg.use_scene_factory_roads):
            self._lane_touch_points_xy_m = torch.zeros((self.num_envs, 0, 2), dtype=torch.float32, device=self.device)
            self._lane_touch_dirs_xy = torch.zeros((self.num_envs, 0, 2), dtype=torch.float32, device=self.device)
            self._lane_touch_half_lengths_m = torch.zeros((self.num_envs, 0), dtype=torch.float32, device=self.device)
            self._lane_touch_half_widths_m = torch.zeros((self.num_envs, 0), dtype=torch.float32, device=self.device)
            self._lane_touch_types = torch.zeros((self.num_envs, 0), dtype=torch.long, device=self.device)
            self._lane_touch_valid = torch.zeros((self.num_envs, 0), dtype=torch.bool, device=self.device)
            self._lane_touch_type_dim = 1
            self._lane_touch_mask = torch.zeros((agent_count, self.num_envs, 1), dtype=torch.bool, device=self.device)
            return

        points_by_env: list[np.ndarray] = []
        dirs_by_env: list[np.ndarray] = []
        half_lengths_by_env: list[np.ndarray] = []
        half_widths_by_env: list[np.ndarray] = []
        types_by_env: list[np.ndarray] = []
        max_segments = 0
        max_type = 0
        for env_index in range(int(self.cfg.scene.num_envs)):
            world_root = f"/World/envs/env_{env_index}/SceneFactoryWorlds/world_000"
            points_xy, dirs_xy, half_lengths_m, half_widths_m, types = _load_scene_factory_lane_touch_metadata(
                stage, world_root=world_root
            )
            points_by_env.append(points_xy)
            dirs_by_env.append(dirs_xy)
            half_lengths_by_env.append(half_lengths_m)
            half_widths_by_env.append(half_widths_m)
            types_by_env.append(types)
            max_segments = max(max_segments, int(points_xy.shape[0]))
            if types.size > 0:
                max_type = max(max_type, int(np.max(types)))

        self._lane_touch_type_dim = max(1, max_type + 1)
        self._lane_touch_points_xy_m = torch.zeros(
            (self.num_envs, max_segments, 2), dtype=torch.float32, device=self.device
        )
        self._lane_touch_dirs_xy = torch.zeros(
            (self.num_envs, max_segments, 2), dtype=torch.float32, device=self.device
        )
        self._lane_touch_half_lengths_m = torch.zeros(
            (self.num_envs, max_segments), dtype=torch.float32, device=self.device
        )
        self._lane_touch_half_widths_m = torch.zeros(
            (self.num_envs, max_segments), dtype=torch.float32, device=self.device
        )
        self._lane_touch_types = torch.zeros((self.num_envs, max_segments), dtype=torch.long, device=self.device)
        self._lane_touch_valid = torch.zeros((self.num_envs, max_segments), dtype=torch.bool, device=self.device)
        for env_index in range(int(self.cfg.scene.num_envs)):
            segment_count = int(points_by_env[env_index].shape[0])
            if segment_count <= 0:
                continue
            env_slice = slice(0, segment_count)
            self._lane_touch_points_xy_m[env_index, env_slice] = torch.as_tensor(
                points_by_env[env_index], dtype=torch.float32, device=self.device
            )
            self._lane_touch_dirs_xy[env_index, env_slice] = torch.as_tensor(
                dirs_by_env[env_index], dtype=torch.float32, device=self.device
            )
            self._lane_touch_half_lengths_m[env_index, env_slice] = torch.as_tensor(
                half_lengths_by_env[env_index], dtype=torch.float32, device=self.device
            )
            self._lane_touch_half_widths_m[env_index, env_slice] = torch.as_tensor(
                half_widths_by_env[env_index], dtype=torch.float32, device=self.device
            )
            self._lane_touch_types[env_index, env_slice] = torch.as_tensor(
                types_by_env[env_index], dtype=torch.long, device=self.device
            )
            self._lane_touch_valid[env_index, env_slice] = True
        self._lane_touch_mask = torch.zeros(
            (agent_count, self.num_envs, int(self._lane_touch_type_dim)),
            dtype=torch.bool,
            device=self.device,
        )

    def _update_lane_touch_mask(self) -> None:
        if (
            not bool(self.cfg.lane_touch_enabled)
            or not bool(self.cfg.use_scene_factory_roads)
            or self._lane_touch_valid.numel() == 0
            or self._lane_touch_valid.shape[1] == 0
        ):
            self._lane_touch_mask.zero_()
            return

        env_origins_xy = self.scene.env_origins[:, :2]
        dirs_xy = self._lane_touch_dirs_xy
        perp_xy = torch.stack((-dirs_xy[..., 1], dirs_xy[..., 0]), dim=-1)
        half_lengths = self._lane_touch_half_lengths_m
        half_widths = self._lane_touch_half_widths_m
        valid = self._lane_touch_valid
        circle_radius = float(self._lane_touch_circle_radius_m) + float(self.cfg.lane_touch_margin_m)
        circle_centers_b = self._lane_touch_circle_centers_b

        self._lane_touch_mask.zero_()
        for agent_idx, vehicle in enumerate(self._vehicles):
            root_pos_xy = vehicle.data.root_pos_w[:, :2] - env_origins_xy
            circle_centers_w = root_pos_xy.unsqueeze(1) + quat_apply(
                vehicle.data.root_quat_w.unsqueeze(1).expand(-1, circle_centers_b.shape[0], -1),
                circle_centers_b.unsqueeze(0).expand(self.num_envs, -1, -1),
            )[:, :, :2]
            delta = circle_centers_w.unsqueeze(2) - self._lane_touch_points_xy_m.unsqueeze(1)
            along = torch.sum(delta * dirs_xy.unsqueeze(1), dim=-1).abs() - half_lengths.unsqueeze(1)
            perp = torch.sum(delta * perp_xy.unsqueeze(1), dim=-1).abs() - half_widths.unsqueeze(1)
            outside_dx = torch.clamp(along, min=0.0)
            outside_dy = torch.clamp(perp, min=0.0)
            distance_sq = outside_dx.square() + outside_dy.square()
            touch = (
                valid.unsqueeze(1)
                & (distance_sq <= (circle_radius * circle_radius))
            )
            for road_type in range(int(self._lane_touch_type_dim)):
                type_mask = valid & (self._lane_touch_types == int(road_type))
                if not bool(torch.any(type_mask).item()):
                    continue
                self._lane_touch_mask[agent_idx, :, road_type] = torch.any(
                    touch & type_mask.unsqueeze(1), dim=(1, 2)
                )

    def lane_touch_type_mask_by_agent(self) -> dict[str, torch.Tensor]:
        return {agent_id: self._lane_touch_mask[idx].clone() for idx, agent_id in enumerate(self._agent_ids)}

    def lane_touch_types_by_agent(self) -> dict[str, list[list[int]]]:
        result: dict[str, list[list[int]]] = {}
        for agent_idx, agent_id in enumerate(self._agent_ids):
            mask = self._lane_touch_mask[agent_idx].detach().cpu()
            result[agent_id] = [torch.nonzero(mask[env_idx], as_tuple=False).view(-1).tolist() for env_idx in range(mask.shape[0])]
        return result

    def _lane_touch_any_type_mask(self, agent_idx: int, lane_types: Sequence[int]) -> torch.Tensor:
        if (
            not bool(self.cfg.lane_touch_enabled)
            or self._lane_touch_mask.numel() == 0
            or self._lane_touch_mask.shape[-1] <= 0
        ):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        valid_types = []
        type_dim = int(self._lane_touch_type_dim)
        for road_type in lane_types:
            road_type_int = int(road_type)
            if 0 <= road_type_int < type_dim:
                valid_types.append(road_type_int)
        if not valid_types:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return torch.any(self._lane_touch_mask[agent_idx, :, valid_types], dim=1)

    def _done_vehicle_root_pose(self, agent_idx: int, env_ids: torch.Tensor) -> torch.Tensor:
        vehicle = self._vehicles[agent_idx]
        root_pose = vehicle.data.default_root_state[env_ids, :7].clone()
        env_origins = self.scene.env_origins[env_ids]
        root_pose[:, 0] = env_origins[:, 0] + 1000.0 + 40.0 * float(agent_idx)
        root_pose[:, 1] = env_origins[:, 1] + 1000.0 + 25.0 * float(agent_idx)
        root_pose[:, 2] = env_origins[:, 2] + 10.0
        return root_pose

    def _build_scene_factory_visual_floor(self, stage, *, world_root: str) -> None:
        from pxr import Gf, UsdGeom

        scene_factory_cfg = _load_yaml(self.cfg.scene_factory_config_path)
        world_cfg = dict(scene_factory_cfg.get("world", {}) or {})
        bounds_size_m = float(world_cfg.get("bounds_size_m", 200.0))
        # Keep a thin visual-only floor slightly above the hidden physics plane so it reliably
        # wins the render without intersecting the road bars.
        floor_thickness_m = 0.01
        floor_color = Gf.Vec3f(0.18, 0.18, 0.20)
        floor_path = f"{world_root}/VisualFloor"
        cube = UsdGeom.Cube.Define(stage, floor_path)
        cube.GetSizeAttr().Set(1.0)
        api = UsdGeom.XformCommonAPI(cube)
        api.SetTranslate(Gf.Vec3d(0.0, 0.0, 0.0))
        api.SetScale(Gf.Vec3f(bounds_size_m, bounds_size_m, floor_thickness_m))
        try:
            UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr([floor_color])
        except Exception:
            pass

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]):
        for agent_idx, agent_id in enumerate(self._agent_ids):
            self._raw_actions[agent_idx] = actions[agent_id].clone().clamp_(-1.0, 1.0)
            # Rectify throttle and brake so a zero policy output maps to a true neutral command.
            self._semantic_actions[agent_idx, :, 0] = torch.clamp(
                self._raw_actions[agent_idx, :, 0], min=0.0, max=1.0
            )
            self._semantic_actions[agent_idx, :, 1] = self._raw_actions[agent_idx, :, 1]
            self._semantic_actions[agent_idx, :, 2] = torch.clamp(
                self._raw_actions[agent_idx, :, 2], min=0.0, max=1.0
            )
            done_mask = self._agent_done_mask[agent_idx]
            if bool(torch.any(done_mask).item()):
                self._raw_actions[agent_idx, done_mask] = 0.0
                self._semantic_actions[agent_idx, done_mask, 0] = 0.0
                self._semantic_actions[agent_idx, done_mask, 1] = 0.0
                self._semantic_actions[agent_idx, done_mask, 2] = 0.0

    def _apply_action(self):
        for agent_idx, vehicle in enumerate(self._vehicles):
            joint_pos = vehicle.data.joint_pos
            joint_vel = vehicle.data.joint_vel
            done_env_ids = torch.nonzero(self._agent_done_mask[agent_idx], as_tuple=False).squeeze(-1)

            self._joint_effort_targets[agent_idx].zero_()

            steer_target = self._semantic_actions[agent_idx, :, 1:2] * self._steer_limit
            steer_pos_error = steer_target - joint_pos[:, self._steer_joint_ids[agent_idx]]
            steer_vel_error = -joint_vel[:, self._steer_joint_ids[agent_idx]]
            steer_effort = (
                float(self._tunable_config.steering_kp_nm_per_rad) * steer_pos_error
                + float(self._tunable_config.steering_kd_nm_s_per_rad) * steer_vel_error
            )
            steer_effort.clamp_(
                -float(self._tunable_config.steering_effort_limit_nm),
                float(self._tunable_config.steering_effort_limit_nm),
            )
            self._joint_effort_targets[agent_idx][:, self._steer_joint_ids[agent_idx]] = steer_effort

            drive_effort = (
                self._semantic_actions[agent_idx, :, 0:1]
                * float(self._tunable_config.drive_torque_nm)
                * float(self._dry_longitudinal_scale)
            )
            self._joint_effort_targets[agent_idx][:, self._drive_joint_ids[agent_idx]] += drive_effort

            brake_joint_vel = joint_vel[:, self._brake_joint_ids[agent_idx]]
            moving_mask = torch.abs(brake_joint_vel) > 1.0e-4
            current_sign = torch.sign(brake_joint_vel)
            current_sign = torch.where(current_sign == 0.0, self._brake_sign_memory[agent_idx], current_sign)
            self._brake_sign_memory[agent_idx] = torch.where(
                moving_mask, current_sign, self._brake_sign_memory[agent_idx]
            )
            brake_sign = torch.where(moving_mask, current_sign, self._brake_sign_memory[agent_idx])

            brake = self._semantic_actions[agent_idx, :, 2:3]
            front_brake_effort = (
                brake * float(self._tunable_config.brake_front_torque_nm) * float(self._dry_longitudinal_scale)
            )
            rear_brake_effort = (
                brake * float(self._tunable_config.brake_rear_torque_nm) * float(self._dry_longitudinal_scale)
            )
            self._joint_effort_targets[agent_idx][:, self._brake_joint_ids[agent_idx][0:2]] -= (
                front_brake_effort * brake_sign[:, 0:2]
            )
            self._joint_effort_targets[agent_idx][:, self._brake_joint_ids[agent_idx][2:4]] -= (
                rear_brake_effort * brake_sign[:, 2:4]
            )
            if len(done_env_ids) > 0:
                self._joint_effort_targets[agent_idx][done_env_ids] = 0.0

            vehicle.set_joint_effort_target(self._joint_effort_targets[agent_idx])

            self._external_forces[agent_idx].zero_()
            self._external_torques[agent_idx].zero_()
            if self.cfg.apply_runtime_external_wrench:
                self._external_forces[agent_idx][:, 0, 1] = (
                    -float(self._tunable_config.lateral_velocity_damping_n_per_mps)
                    * float(self._dry_lateral_scale)
                    * vehicle.data.root_lin_vel_b[:, 1]
                )
                self._external_torques[agent_idx][:, 0, 2] = (
                    -float(self._tunable_config.yaw_stability_damping_nm_per_rad_s)
                    * float(self._dry_lateral_scale)
                    * vehicle.data.root_ang_vel_b[:, 2]
                )
            if len(done_env_ids) > 0:
                self._external_forces[agent_idx][done_env_ids] = 0.0
                self._external_torques[agent_idx][done_env_ids] = 0.0
            vehicle.permanent_wrench_composer.set_forces_and_torques(
                forces=self._external_forces[agent_idx],
                torques=self._external_torques[agent_idx],
                body_ids=self._base_body_ids[agent_idx],
                is_global=False,
            )
            if len(done_env_ids) > 0:
                parked_root_pose = self._done_vehicle_root_pose(agent_idx, done_env_ids)
                parked_root_velocity = torch.zeros((len(done_env_ids), 6), dtype=torch.float32, device=self.device)
                parked_joint_pos = vehicle.data.default_joint_pos[done_env_ids].clone()
                parked_joint_vel = torch.zeros_like(vehicle.data.default_joint_vel[done_env_ids])
                vehicle.write_root_pose_to_sim(parked_root_pose, env_ids=done_env_ids)
                vehicle.write_root_velocity_to_sim(parked_root_velocity, env_ids=done_env_ids)
                vehicle.write_joint_state_to_sim(parked_joint_pos, parked_joint_vel, None, done_env_ids)

    def _compute_goal_position_body(self, agent_idx: int) -> torch.Tensor:
        goal_pos_b, _ = subtract_frame_transforms(
            self._vehicles[agent_idx].data.root_pos_w,
            self._vehicles[agent_idx].data.root_quat_w,
            self._goal_pos_w[agent_idx],
        )
        return goal_pos_b

    def _compute_goal_distance(self, agent_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        goal_pos_b = self._compute_goal_position_body(agent_idx)
        distance = torch.linalg.norm(goal_pos_b[:, :2], dim=1)
        return goal_pos_b, distance

    def _pairwise_distances_xy(self) -> torch.Tensor:
        positions = torch.stack([vehicle.data.root_pos_w[:, :2] for vehicle in self._vehicles], dim=1)
        deltas = positions.unsqueeze(2) - positions.unsqueeze(1)
        distances = torch.linalg.norm(deltas, dim=-1)
        eye = torch.eye(self._num_agents, device=self.device, dtype=torch.bool).unsqueeze(0)
        return torch.where(eye, torch.full_like(distances, float("inf")), distances)

    def _collision_force_by_agent_n_tensor(self) -> torch.Tensor:
        if self._num_agents <= 1:
            return torch.zeros(1, self.num_envs, dtype=torch.float32, device=self.device)

        agent_forces: list[torch.Tensor] = []
        for sensor_names in self._collision_sensor_names_by_agent:
            sensor_forces: list[torch.Tensor] = []
            for sensor_name in sensor_names:
                contact_sensor = self.scene.sensors[sensor_name]
                force_matrix_w = contact_sensor.data.force_matrix_w
                if force_matrix_w is not None and force_matrix_w.numel() > 0:
                    sensor_force = torch.linalg.norm(force_matrix_w, dim=-1).amax(dim=(1, 2))
                else:
                    sensor_force = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
                sensor_forces.append(sensor_force)
            if sensor_forces:
                agent_forces.append(torch.stack(sensor_forces, dim=0).amax(dim=0))
            else:
                agent_forces.append(torch.zeros(self.num_envs, dtype=torch.float32, device=self.device))
        forces = torch.stack(agent_forces, dim=0)
        warmup_steps = max(0, int(self.cfg.agent_collision_warmup_steps))
        if warmup_steps > 0:
            warmup_mask = self.episode_length_buf < warmup_steps
            if bool(torch.any(warmup_mask).item()):
                forces[:, warmup_mask] = 0.0
        return forces

    def _collision_world_mask(self) -> torch.Tensor:
        if self._num_agents <= 1:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        max_force_per_world = self._collision_force_by_agent_n_tensor().amax(dim=0)
        return max_force_per_world >= float(self.cfg.agent_collision_force_threshold_n)

    def _collision_by_agent_mask_tensor(self) -> torch.Tensor:
        return self._collision_force_by_agent_n_tensor() >= float(self.cfg.agent_collision_force_threshold_n)

    def collision_force_by_agent_n(self) -> dict[str, torch.Tensor]:
        forces = self._collision_force_by_agent_n_tensor()
        return {agent_id: forces[idx].clone() for idx, agent_id in enumerate(self._agent_ids)}

    def collision_world_force_n(self) -> torch.Tensor:
        return self._collision_force_by_agent_n_tensor().amax(dim=0).clone()

    def _compute_yaw_by_agent(self, root_quat_w: torch.Tensor) -> torch.Tensor:
        forward_b = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        forward_w = quat_apply(
            root_quat_w.reshape(-1, 4),
            forward_b.unsqueeze(0).expand(root_quat_w.shape[0] * root_quat_w.shape[1], -1),
        ).reshape(root_quat_w.shape[0], root_quat_w.shape[1], 3)
        return torch.atan2(forward_w[..., 1], forward_w[..., 0])

    def _compute_pairwise_vehicle_ttc_s(
        self,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
        root_lin_vel_w: torch.Tensor,
    ) -> torch.Tensor:
        if self._num_agents <= 1:
            return torch.full((1, self.num_envs, 1), float("inf"), device=self.device)

        env_origins_xy = self.scene.env_origins[:, :2]
        circle_centers_b = self._lane_touch_circle_centers_b
        circle_radius = float(self._lane_touch_circle_radius_m)
        num_circles = int(circle_centers_b.shape[0])
        root_pos_xy = root_pos_w[..., :2] - env_origins_xy.unsqueeze(0)
        flat_centers_w = quat_apply(
            root_quat_w.reshape(-1, 4).unsqueeze(1).expand(-1, num_circles, -1).reshape(-1, 4),
            circle_centers_b.unsqueeze(0)
            .expand(root_quat_w.shape[0] * root_quat_w.shape[1], -1, -1)
            .reshape(-1, 3),
        ).reshape(self._num_agents, self.num_envs, num_circles, 3)[..., :2]
        centers_w = root_pos_xy.unsqueeze(2) + flat_centers_w
        forward_w = quat_apply(
            root_quat_w.reshape(-1, 4),
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .expand(root_quat_w.shape[0] * root_quat_w.shape[1], -1),
        ).reshape(self._num_agents, self.num_envs, 3)[..., :2]
        forward_w = forward_w / torch.linalg.norm(forward_w, dim=-1, keepdim=True).clamp_min(1.0e-6)
        velocities_xy = root_lin_vel_w[..., :2]

        ttc = torch.full((self._num_agents, self.num_envs, self._num_agents), float("inf"), device=self.device)
        combined_radius = 2.0 * circle_radius
        combined_radius_sq = float(combined_radius * combined_radius)

        for ego_idx in range(self._num_agents):
            ego_centers = centers_w[ego_idx]
            ego_forward = forward_w[ego_idx]
            ego_vel = velocities_xy[ego_idx]
            for other_idx in range(self._num_agents):
                if other_idx == ego_idx:
                    continue
                other_centers = centers_w[other_idx]
                other_vel = velocities_xy[other_idx]
                rel = other_centers.unsqueeze(1) - ego_centers.unsqueeze(2)
                rel_vel = other_vel - ego_vel
                rx = rel[..., 0]
                ry = rel[..., 1]
                rvx = rel_vel[:, 0].unsqueeze(-1).unsqueeze(-1)
                rvy = rel_vel[:, 1].unsqueeze(-1).unsqueeze(-1)
                dist_sq = rx.square() + ry.square()
                rel_speed_sq = rvx.square() + rvy.square()
                rdotv = rx * rvx + ry * rvy
                forward_dot = rx * ego_forward[:, 0].unsqueeze(-1).unsqueeze(-1) + ry * ego_forward[:, 1].unsqueeze(
                    -1
                ).unsqueeze(-1)
                forward_mask = forward_dot > 0.0
                overlap_mask = forward_mask & (dist_sq <= combined_radius_sq)
                t_pair = torch.full_like(dist_sq, float("inf"))
                t_pair = torch.where(overlap_mask, torch.zeros_like(t_pair), t_pair)

                moving_mask = forward_mask & (~overlap_mask) & (rel_speed_sq > 1.0e-6)
                if bool(torch.any(moving_mask).item()):
                    a = torch.where(moving_mask, rel_speed_sq, torch.ones_like(rel_speed_sq))
                    b = torch.where(moving_mask, 2.0 * rdotv, torch.zeros_like(rdotv))
                    c = torch.where(moving_mask, dist_sq - combined_radius_sq, torch.zeros_like(dist_sq))
                    disc = b.square() - 4.0 * a * c
                    valid_quad = moving_mask & (disc >= 0.0)
                    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
                    denom = 2.0 * torch.clamp(a, min=1.0e-6)
                    t_enter = (-b - sqrt_disc) / denom
                    t_exit = (-b + sqrt_disc) / denom
                    valid_enter = valid_quad & (t_exit >= 0.0)
                    t_pair = torch.minimum(t_pair, torch.where(valid_enter, torch.clamp(t_enter, min=0.0), t_pair))

                    unresolved = moving_mask & (~torch.isfinite(t_pair))
                    if bool(torch.any(unresolved).item()):
                        dist = torch.sqrt(torch.clamp(dist_sq, min=1.0e-9))
                        closing_speed = -rdotv / torch.clamp(dist, min=1.0e-6)
                        clearance = torch.clamp(dist - combined_radius, min=0.0)
                        valid_fb = unresolved & (rdotv < 0.0) & (closing_speed > 1.0e-6)
                        fallback_t = torch.where(valid_fb, clearance / torch.clamp(closing_speed, min=1.0e-6), t_pair)
                        t_pair = torch.minimum(t_pair, fallback_t)

                ttc[ego_idx, :, other_idx] = torch.amin(t_pair, dim=(1, 2))
        return ttc

    def _build_reference_neighbor_context(
        self,
        agent_idx: int,
        root_pos_w: torch.Tensor,
        yaw_by_agent: torch.Tensor,
        speed_by_agent: torch.Tensor,
        pairwise_ttc_s: torch.Tensor | None,
    ) -> torch.Tensor:
        k = max(0, int(self.cfg.obs_neighbor_k))
        feat_dim = _reference_vehicle_feat_dim(
            self.cfg.obs_neighbor_include_ttc,
            self.cfg.obs_neighbor_include_index,
        )
        if (not bool(self.cfg.obs_neighbor_enable)) or k <= 0 or self._num_agents <= 1:
            return torch.zeros((self.num_envs, k * feat_dim), dtype=torch.float32, device=self.device)

        env_origins_xy = self.scene.env_origins[:, :2]
        root_pos_xy = root_pos_w[..., :2] - env_origins_xy.unsqueeze(0)
        dx = root_pos_xy[:, :, 0] - root_pos_xy[agent_idx, :, 0].unsqueeze(0)
        dy = root_pos_xy[:, :, 1] - root_pos_xy[agent_idx, :, 1].unsqueeze(0)
        relx_b, rely_b = _world_to_ego_xy_torch(dx, dy, yaw_by_agent[agent_idx].unsqueeze(0))
        distances_sq = dx.square() + dy.square()
        distances_sq = torch.where(
            self._agent_done_mask,
            torch.full_like(distances_sq, float("inf")),
            distances_sq,
        )
        distances_sq[agent_idx] = float("inf")
        agent_tie_break = (
            torch.arange(self._num_agents, dtype=torch.float32, device=self.device).unsqueeze(1) * 1.0e-4
        )
        sort_keys = distances_sq + agent_tie_break
        sorted_indices = torch.argsort(sort_keys.transpose(0, 1), dim=1)
        bounds_scale = float(max(1.0e-6, self._scene_factory_bounds_size_m))
        speed_scale = 10.0
        yaw_scale = math.pi
        ttc_max_s = float(max(0.1, self.cfg.obs_neighbor_ttc_max_s))
        obs = torch.zeros((self.num_envs, k, feat_dim), dtype=torch.float32, device=self.device)
        available_slots = min(k, max(0, self._num_agents - 1), int(sorted_indices.shape[1]))

        for slot in range(available_slots):
            neighbor_idx = sorted_indices[:, slot]
            env_ids = torch.arange(self.num_envs, device=self.device)
            obs[:, slot, 0] = relx_b.transpose(0, 1)[env_ids, neighbor_idx] / bounds_scale
            obs[:, slot, 1] = rely_b.transpose(0, 1)[env_ids, neighbor_idx] / bounds_scale
            obs[:, slot, 2] = float(self._vehicle_length_m) / bounds_scale
            obs[:, slot, 3] = float(self._vehicle_width_m) / bounds_scale
            rel_yaw = _wrap_pi_torch(yaw_by_agent[neighbor_idx, env_ids] - yaw_by_agent[agent_idx, env_ids])
            obs[:, slot, 4] = rel_yaw / yaw_scale
            obs[:, slot, 5] = speed_by_agent[neighbor_idx, env_ids] / speed_scale
            write_idx = 6
            if bool(self.cfg.obs_neighbor_include_ttc):
                ttc = pairwise_ttc_s[agent_idx, env_ids, neighbor_idx]
                ttc_n = torch.ones_like(ttc)
                finite_mask = torch.isfinite(ttc)
                ttc_n = torch.where(
                    finite_mask,
                    torch.clamp(ttc, min=0.0, max=ttc_max_s) / ttc_max_s,
                    ttc_n,
                )
                obs[:, slot, write_idx] = ttc_n
                write_idx += 1
            if bool(self.cfg.obs_neighbor_include_index):
                denom = float(max(1, self._num_agents - 1))
                obs[:, slot, write_idx] = neighbor_idx.to(torch.float32) / denom
        return obs.reshape(self.num_envs, -1)

    def _build_reference_road_context(
        self,
        agent_idx: int,
        root_pos_w: torch.Tensor,
        yaw_by_agent: torch.Tensor,
    ) -> torch.Tensor:
        k = max(0, int(self.cfg.obs_road_points_k))
        feat_dim = _reference_road_point_feat_dim(self.cfg.obs_road_points_include_dirs)
        if (
            (not bool(self.cfg.obs_road_points_enable))
            or k <= 0
            or self._lane_touch_valid.numel() == 0
            or self._lane_touch_valid.shape[1] == 0
        ):
            return torch.zeros((self.num_envs, k * feat_dim), dtype=torch.float32, device=self.device)

        env_origins_xy = self.scene.env_origins[:, :2]
        root_pos_xy = root_pos_w[agent_idx, :, :2] - env_origins_xy
        dx_all = self._lane_touch_points_xy_m[..., 0] - root_pos_xy[:, 0].unsqueeze(1)
        dy_all = self._lane_touch_points_xy_m[..., 1] - root_pos_xy[:, 1].unsqueeze(1)
        dist_sq = dx_all.square() + dy_all.square()
        valid = self._lane_touch_valid
        radius_m = float(max(0.0, self.cfg.obs_road_points_radius_m))
        if radius_m > 0.0:
            valid = valid & (dist_sq <= float(radius_m * radius_m))
        mode = str(self.cfg.obs_road_points_mode).strip().lower().replace("_", "-")
        if mode == "knn":
            sort_keys = torch.where(valid, dist_sq, torch.full_like(dist_sq, float("inf")))
            sorted_indices = torch.argsort(sort_keys, dim=1)[:, :k]
        elif mode == "road-running":
            insertion_order = (
                torch.arange(self._lane_touch_valid.shape[1], dtype=torch.float32, device=self.device)
                .unsqueeze(0)
                .expand(self.num_envs, -1)
            )
            sort_keys = torch.where(valid, insertion_order, torch.full_like(insertion_order, float("inf")))
            sorted_indices = torch.argsort(sort_keys, dim=1)[:, :k]
        else:
            sort_keys = torch.where(valid, dist_sq, torch.full_like(dist_sq, float("inf")))
            sorted_indices = torch.argsort(sort_keys, dim=1)[:, :k]

        norm = float(radius_m if radius_m > 0.0 else max(1.0e-6, self._scene_factory_bounds_size_m))
        type_norm = float(self.cfg.obs_road_points_type_norm)
        obs = torch.zeros((self.num_envs, k, feat_dim), dtype=torch.float32, device=self.device)
        if sorted_indices.shape[1] == 0:
            return obs.reshape(self.num_envs, -1)
        env_ids = torch.arange(self.num_envs, device=self.device)
        selected_dx = dx_all[env_ids.unsqueeze(1), sorted_indices]
        selected_dy = dy_all[env_ids.unsqueeze(1), sorted_indices]
        x_ego, y_ego = _world_to_ego_xy_torch(selected_dx, selected_dy, yaw_by_agent[agent_idx].unsqueeze(1))
        slot_count = int(sorted_indices.shape[1])
        obs[:, :slot_count, 0] = x_ego / norm
        obs[:, :slot_count, 1] = y_ego / norm
        types = self._lane_touch_types[env_ids.unsqueeze(1), sorted_indices].to(torch.float32)
        obs[:, :slot_count, 2] = types / type_norm if type_norm > 0.0 else types
        invalid = ~valid[env_ids.unsqueeze(1), sorted_indices]
        obs_view = obs[:, :slot_count]
        obs_view[invalid] = 0.0
        if feat_dim >= 5:
            dirs = self._lane_touch_dirs_xy[env_ids.unsqueeze(1), sorted_indices]
            dir_x_ego, dir_y_ego = _world_to_ego_xy_torch(dirs[..., 0], dirs[..., 1], yaw_by_agent[agent_idx].unsqueeze(1))
            obs_view[:, :, 3] = dir_x_ego
            obs_view[:, :, 4] = dir_y_ego
            obs_view[invalid] = 0.0
        return obs.reshape(self.num_envs, -1)

    def _nearest_neighbor_features(self, agent_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._num_agents <= 1:
            zeros = torch.zeros(self.num_envs, 3, device=self.device)
            return zeros, zeros, torch.zeros(self.num_envs, device=self.device)

        vehicle = self._vehicles[agent_idx]
        root_pos_w = vehicle.data.root_pos_w
        root_quat_w = vehicle.data.root_quat_w
        root_lin_vel_w = vehicle.data.root_lin_vel_w

        nearest_distance = torch.full((self.num_envs,), float("inf"), device=self.device)
        nearest_rel_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        nearest_rel_vel_b = torch.zeros(self.num_envs, 3, device=self.device)

        for other_idx, other_vehicle in enumerate(self._vehicles):
            if other_idx == agent_idx:
                continue
            rel_pos_b, _ = subtract_frame_transforms(
                root_pos_w,
                root_quat_w,
                other_vehicle.data.root_pos_w,
            )
            rel_vel_w = other_vehicle.data.root_lin_vel_w - root_lin_vel_w
            rel_vel_b = quat_apply_inverse(root_quat_w, rel_vel_w)
            distance = torch.linalg.norm(rel_pos_b[:, :2], dim=1)
            distance = torch.where(
                self._agent_done_mask[other_idx],
                torch.full_like(distance, float("inf")),
                distance,
            )
            update_mask = distance < nearest_distance
            nearest_distance = torch.where(update_mask, distance, nearest_distance)
            nearest_rel_pos_b = torch.where(update_mask.unsqueeze(-1), rel_pos_b, nearest_rel_pos_b)
            nearest_rel_vel_b = torch.where(update_mask.unsqueeze(-1), rel_vel_b, nearest_rel_vel_b)

        nearest_distance = torch.where(torch.isfinite(nearest_distance), nearest_distance, 0.0)
        return nearest_rel_pos_b, nearest_rel_vel_b, nearest_distance

    def _agent_crash_mask(self, agent_idx: int) -> torch.Tensor:
        vehicle = self._vehicles[agent_idx]
        root_pos_rel = vehicle.data.root_pos_w - self.scene.env_origins
        too_low = root_pos_rel[:, 2] < float(self.cfg.fall_height_threshold_m)
        too_far = torch.linalg.norm(root_pos_rel[:, :2], dim=1) > float(self.cfg.max_distance_from_origin_m)
        bad_tilt = vehicle.data.projected_gravity_b[:, 2] > float(self.cfg.bad_tilt_gravity_threshold)
        return too_low | too_far | bad_tilt

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs_start = perf_counter()
        self._update_lane_touch_mask()
        observations = {}
        observation_mode = str(self.cfg.observation_mode).strip().lower()
        neighbor_obs_scale = float(max(1.0e-6, self.cfg.agent_neighbor_obs_scale_m))
        goal_obs_scale = float(max(1.0e-6, self.cfg.goal_radius_max_m))
        if observation_mode == "choco_reference":
            root_pos_w = torch.stack([vehicle.data.root_pos_w for vehicle in self._vehicles], dim=0)
            root_quat_w = torch.stack([vehicle.data.root_quat_w for vehicle in self._vehicles], dim=0)
            root_lin_vel_w = torch.stack([vehicle.data.root_lin_vel_w for vehicle in self._vehicles], dim=0)
            yaw_by_agent = self._compute_yaw_by_agent(root_quat_w)
            speed_by_agent = torch.linalg.norm(root_lin_vel_w[..., :2], dim=-1)
            pairwise_ttc_s = (
                self._compute_pairwise_vehicle_ttc_s(root_pos_w, root_quat_w, root_lin_vel_w)
                if bool(self.cfg.obs_neighbor_enable and self.cfg.obs_neighbor_include_ttc)
                else None
            )
        for agent_idx, agent_id in enumerate(self._agent_ids):
            goal_pos_b, goal_distance = self._compute_goal_distance(agent_idx)
            self._current_goal_distance[agent_idx] = goal_distance
            if observation_mode == "goal_reaching":
                obs = torch.cat(
                    [
                        goal_pos_b[:, :2] / goal_obs_scale,
                        goal_distance.unsqueeze(-1) / goal_obs_scale,
                        self._vehicles[agent_idx].data.root_lin_vel_b[:, :2] / 10.0,
                        self._vehicles[agent_idx].data.root_ang_vel_b[:, 2:3] / 10.0,
                    ],
                    dim=-1,
                )
            elif observation_mode == "full":
                nearest_rel_pos_b, nearest_rel_vel_b, nearest_distance = self._nearest_neighbor_features(agent_idx)
                obs = torch.cat(
                    [
                        goal_pos_b[:, :2] / goal_obs_scale,
                        goal_distance.unsqueeze(-1) / goal_obs_scale,
                        self._vehicles[agent_idx].data.root_lin_vel_b[:, :2] / 10.0,
                        self._vehicles[agent_idx].data.root_ang_vel_b[:, 2:3] / 10.0,
                        self._vehicles[agent_idx].data.projected_gravity_b[:, :2],
                        self._vehicles[agent_idx].data.joint_pos[:, self._steer_joint_ids[agent_idx]]
                        / float(max(1.0e-6, self._steer_limit)),
                        self._vehicles[agent_idx].data.joint_vel[:, self._wheel_joint_ids[agent_idx]] / 50.0,
                        self._raw_actions[agent_idx],
                        nearest_rel_pos_b[:, :2] / neighbor_obs_scale,
                        nearest_distance.unsqueeze(-1) / neighbor_obs_scale,
                        nearest_rel_vel_b[:, :2] / 10.0,
                    ],
                    dim=-1,
                )
            elif observation_mode == "choco_reference":
                heading_error = torch.atan2(goal_pos_b[:, 1], goal_pos_b[:, 0])
                bounds_scale = float(max(1.0e-6, self._scene_factory_bounds_size_m))
                distance_scale = float(max(1.0e-6, self._scene_factory_bounds_size_m * math.sqrt(2.0)))
                obs_parts = [
                    goal_pos_b[:, 0:1] / bounds_scale,
                    goal_pos_b[:, 1:2] / bounds_scale,
                    torch.sin(heading_error).unsqueeze(-1),
                    torch.cos(heading_error).unsqueeze(-1),
                    goal_distance.unsqueeze(-1) / distance_scale,
                    self._vehicles[agent_idx].data.root_lin_vel_b[:, :2] / 10.0,
                ]
                if bool(self.cfg.obs_weather_context_enable):
                    obs_parts.append(self._weather_context)
                if bool(self.cfg.obs_road_points_enable):
                    obs_parts.append(self._build_reference_road_context(agent_idx, root_pos_w, yaw_by_agent))
                if bool(self.cfg.obs_neighbor_enable):
                    obs_parts.append(
                        self._build_reference_neighbor_context(
                            agent_idx,
                            root_pos_w,
                            yaw_by_agent,
                            speed_by_agent,
                            pairwise_ttc_s,
                        )
                    )
                obs = torch.cat(obs_parts, dim=-1)
            else:
                raise ValueError(f"Unsupported observation mode: {self.cfg.observation_mode!r}")
            done_mask = self._agent_done_mask[agent_idx]
            if bool(torch.any(done_mask).item()):
                obs = obs.clone()
                obs[done_mask] = 0.0
            observations[agent_id] = obs
        elapsed_ms = (perf_counter() - obs_start) * 1000.0
        self._obs_timing_call_count += 1
        self._obs_timing_last_ms = float(elapsed_ms)
        if self._obs_timing_call_count == 1:
            self._obs_timing_ema_ms = float(elapsed_ms)
        else:
            self._obs_timing_ema_ms = 0.9 * float(self._obs_timing_ema_ms) + 0.1 * float(elapsed_ms)
        if bool(self.cfg.obs_timing_print_enable):
            every_n = max(1, int(self.cfg.obs_timing_print_every_n))
            if self._obs_timing_call_count % every_n == 0:
                print(
                    "[INFO][SceneFactory][ObsTiming] "
                    f"call={self._obs_timing_call_count} mode={observation_mode} "
                    f"last_ms={self._obs_timing_last_ms:.2f} ema_ms={self._obs_timing_ema_ms:.2f} "
                    f"envs={self.num_envs} agents={self._num_agents}",
                    flush=True,
                )
        return observations

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        self._update_lane_touch_mask()
        rewards = {}
        pairwise_distances = self._pairwise_distances_xy()
        nearest_distances = pairwise_distances.min(dim=2).values if self._num_agents > 1 else None

        for agent_idx, agent_id in enumerate(self._agent_ids):
            goal_pos_b, goal_distance = self._compute_goal_distance(agent_idx)
            self._current_goal_distance[agent_idx] = goal_distance

            active_mask = ~self._agent_done_mask[agent_idx]
            active_float = active_mask.float()
            goal_dir_b = goal_pos_b[:, :2] / goal_distance.unsqueeze(-1).clamp_min(1.0e-6)
            progress = self._previous_goal_distance[agent_idx] - goal_distance
            heading_alignment = goal_dir_b[:, 0]
            speed_to_goal = torch.sum(self._vehicles[agent_idx].data.root_lin_vel_b[:, :2] * goal_dir_b, dim=1).clamp_min(
                0.0
            )
            lateral_velocity = torch.abs(self._vehicles[agent_idx].data.root_lin_vel_b[:, 1])
            yaw_rate = torch.abs(self._vehicles[agent_idx].data.root_ang_vel_b[:, 2])
            action_rate = torch.sum(torch.square(self._raw_actions[agent_idx] - self._previous_raw_actions[agent_idx]), dim=1)
            action_magnitude = torch.sum(torch.square(self._semantic_actions[agent_idx]), dim=1)
            throttle_brake_conflict = self._semantic_actions[agent_idx, :, 0] * self._semantic_actions[agent_idx, :, 2]
            goal_shaping = 1.0 - torch.tanh(goal_distance / float(max(1.0, self.cfg.goal_radius_max_m * 0.5)))
            goal_bonus = self._pending_goal_done_mask[agent_idx].float() * float(self.cfg.reward_goal_bonus)
            crash_penalty = self._pending_crash_done_mask[agent_idx].float() * float(self.cfg.reward_crash_penalty)
            collision_penalty = (
                self._pending_collision_done_mask[agent_idx].float() * float(self.cfg.reward_collision_penalty)
            )
            lane_center_touch = (
                self._lane_touch_any_type_mask(agent_idx, self.cfg.reward_lane_center_types)
                if bool(self.cfg.reward_lane_center_enable)
                else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            )
            lane_forbidden_new = self._pending_lane_forbidden_done_mask[agent_idx]

            if nearest_distances is not None:
                proximity_violation = torch.relu(
                    float(self.cfg.agent_safe_distance_m) - nearest_distances[:, agent_idx]
                ) / float(max(1.0e-6, self.cfg.agent_safe_distance_m))
            else:
                proximity_violation = torch.zeros(self.num_envs, device=self.device)
            neighbor_proximity = proximity_violation * float(self.cfg.reward_scale_neighbor_proximity)

            reward_terms = {
                "alive": torch.full_like(goal_distance, float(self.cfg.reward_scale_alive)) * active_float,
                "progress": progress * float(self.cfg.reward_scale_progress) * active_float,
                "goal_shaping": goal_shaping * float(self.cfg.reward_scale_goal_shaping) * active_float,
                "heading": heading_alignment * float(self.cfg.reward_scale_heading) * active_float,
                "speed_to_goal": speed_to_goal * float(self.cfg.reward_scale_speed_to_goal) * active_float,
                "lateral_velocity": lateral_velocity * float(self.cfg.reward_scale_lateral_velocity) * active_float,
                "yaw_rate": yaw_rate * float(self.cfg.reward_scale_yaw_rate) * active_float,
                "action_rate": action_rate * float(self.cfg.reward_scale_action_rate) * active_float,
                "action_magnitude": action_magnitude * float(self.cfg.reward_scale_action_magnitude) * active_float,
                "throttle_brake_conflict": throttle_brake_conflict
                * float(self.cfg.reward_scale_throttle_brake_conflict)
                * active_float,
                "neighbor_proximity": neighbor_proximity * active_float,
                "goal_bonus": goal_bonus,
                "collision_penalty": collision_penalty,
                "crash_penalty": crash_penalty,
                "lane_center_bonus": lane_center_touch.float() * float(self.cfg.reward_lane_center_per_step) * active_float,
                "lane_forbidden_penalty": lane_forbidden_new.float() * float(self.cfg.reward_lane_forbidden_penalty),
            }
            rewards[agent_id] = torch.sum(torch.stack(list(reward_terms.values())), dim=0)

            for key, value in reward_terms.items():
                self._episode_sums[key][agent_idx] += value

            self._previous_goal_distance[agent_idx] = goal_distance
            self._previous_raw_actions[agent_idx] = self._raw_actions[agent_idx]
            new_done = (
                self._pending_goal_done_mask[agent_idx]
                | self._pending_collision_done_mask[agent_idx]
                | self._pending_crash_done_mask[agent_idx]
                | self._pending_lane_forbidden_done_mask[agent_idx]
            )
            self._terminal_goal_distance[agent_idx] = torch.where(
                new_done,
                goal_distance,
                self._terminal_goal_distance[agent_idx],
            )
            self._agent_done_mask[agent_idx] |= new_done
            self._goal_done_mask[agent_idx] |= self._pending_goal_done_mask[agent_idx]
            self._collision_done_mask[agent_idx] |= self._pending_collision_done_mask[agent_idx]
            self._crash_done_mask[agent_idx] |= self._pending_crash_done_mask[agent_idx]
            self._lane_forbidden_done_mask[agent_idx] |= self._pending_lane_forbidden_done_mask[agent_idx]

        self._pending_goal_done_mask.zero_()
        self._pending_collision_done_mask.zero_()
        self._pending_crash_done_mask.zero_()
        self._pending_lane_forbidden_done_mask.zero_()
        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if str(self.cfg.test_mode).strip().lower() in {
            "collision_test",
            "scene_factory_collision_test",
            "scene_factory_multiworld_random_steer_test",
        }:
            false_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            terminated = {agent_id: false_buf.clone() for agent_id in self._agent_ids}
            time_outs = {agent_id: false_buf.clone() for agent_id in self._agent_ids}
            return terminated, time_outs

        self._update_lane_touch_mask()
        collision_masks = self._collision_by_agent_mask_tensor()
        self._pending_goal_done_mask.zero_()
        self._pending_collision_done_mask.zero_()
        self._pending_crash_done_mask.zero_()
        self._pending_lane_forbidden_done_mask.zero_()
        terminated = {}
        for agent_idx in range(self._num_agents):
            _, goal_distance = self._compute_goal_distance(agent_idx)
            self._current_goal_distance[agent_idx] = goal_distance
            active_mask = ~self._agent_done_mask[agent_idx]
            goal_reached = goal_distance <= float(self.cfg.goal_reached_threshold_m)
            crash_mask = self._agent_crash_mask(agent_idx)
            collision_mask = collision_masks[agent_idx]
            lane_forbidden_touch = (
                self._lane_touch_any_type_mask(agent_idx, self.cfg.reward_lane_forbidden_types)
                if bool(self.cfg.reward_lane_forbidden_enable)
                else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            )
            self._pending_goal_done_mask[agent_idx] = active_mask & goal_reached
            self._pending_crash_done_mask[agent_idx] = active_mask & crash_mask
            self._pending_collision_done_mask[agent_idx] = active_mask & collision_mask
            self._pending_lane_forbidden_done_mask[agent_idx] = active_mask & lane_forbidden_touch
            terminated[self._agent_ids[agent_idx]] = (
                self._agent_done_mask[agent_idx]
                | self._pending_goal_done_mask[agent_idx]
                | self._pending_crash_done_mask[agent_idx]
                | self._pending_collision_done_mask[agent_idx]
                | self._pending_lane_forbidden_done_mask[agent_idx]
            )

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        time_outs = {agent_id: time_out.clone() for agent_id in self._agent_ids}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._vehicles[0]._ALL_INDICES

        collision_force_by_agent = self._collision_force_by_agent_n_tensor()
        self._update_lane_touch_mask()
        lane_center_touch_by_agent = torch.stack(
            [
                self._lane_touch_any_type_mask(agent_idx, self.cfg.reward_lane_center_types)
                if bool(self.cfg.reward_lane_center_enable)
                else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                for agent_idx in range(self._num_agents)
            ],
            dim=0,
        )

        for agent_idx, agent_id in enumerate(self._agent_ids):
            self.extras[agent_id] = {"log": {}}
            for key, value in self._episode_sums.items():
                self.extras[agent_id]["log"][f"Episode_Reward/{key}"] = torch.mean(value[agent_idx, env_ids]).item() / max(
                    1.0, float(self.max_episode_length_s)
                )
                value[agent_idx, env_ids] = 0.0
            final_distance = torch.where(
                self._agent_done_mask[agent_idx, env_ids],
                self._terminal_goal_distance[agent_idx, env_ids],
                self._current_goal_distance[agent_idx, env_ids],
            )
            self.extras[agent_id]["log"]["Metrics/final_distance_to_goal"] = torch.mean(final_distance).item()
            self.extras[agent_id]["log"]["Metrics/success_rate"] = torch.mean(
                self._goal_done_mask[agent_idx, env_ids].float()
            ).item()
            self.extras[agent_id]["log"]["Metrics/all_goals_reached_rate"] = torch.mean(
                torch.all(self._goal_done_mask[:, env_ids], dim=0).float()
            ).item()
            self.extras[agent_id]["log"]["Metrics/crash_rate"] = torch.mean(
                self._crash_done_mask[agent_idx, env_ids].float()
            ).item()
            self.extras[agent_id]["log"]["Metrics/collision_rate"] = torch.mean(
                self._collision_done_mask[agent_idx, env_ids].float()
            ).item()
            self.extras[agent_id]["log"]["Metrics/collision_count"] = float(
                torch.count_nonzero(self._collision_done_mask[agent_idx, env_ids]).item()
            )
            self.extras[agent_id]["log"]["Metrics/max_collision_force_n"] = torch.mean(
                collision_force_by_agent[agent_idx, env_ids]
            ).item()
            self.extras[agent_id]["log"]["Metrics/lane_center_touch_rate"] = torch.mean(
                lane_center_touch_by_agent[agent_idx, env_ids].float()
            ).item()
            self.extras[agent_id]["log"]["Metrics/lane_forbidden_done_count"] = float(
                torch.count_nonzero(self._lane_forbidden_done_mask[agent_idx, env_ids]).item()
            )
            self.extras[agent_id]["log"]["Metrics/lane_forbidden_done_rate"] = torch.mean(
                self._lane_forbidden_done_mask[agent_idx, env_ids].float()
            ).item()

        for vehicle in self._vehicles:
            vehicle.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        num_resets = len(env_ids)
        env_origins = self.scene.env_origins[env_ids]
        test_mode = str(self.cfg.test_mode).strip().lower()
        use_collision_test = test_mode in {"collision_test", "scene_factory_collision_test"}
        use_scene_factory_roads = bool(
            self.cfg.use_scene_factory_roads and self._scenario_spawns_by_env is not None and not use_collision_test
        )
        if self.cfg.randomize_spawn_phase and not use_scene_factory_roads:
            phase = sample_uniform(-math.pi, math.pi, (num_resets,), self.device)
        else:
            phase = torch.zeros(num_resets, device=self.device)
        if use_scene_factory_roads:
            shared_world_offset = torch.zeros(num_resets, 2, device=self.device)
            goal_radius = torch.zeros(num_resets, device=self.device)
        else:
            shared_world_offset = sample_uniform(
                -float(self.cfg.start_radius_m),
                float(self.cfg.start_radius_m),
                (num_resets, 2),
                self.device,
            )
            goal_radius = sample_uniform(
                float(self.cfg.goal_radius_min_m),
                float(self.cfg.goal_radius_max_m),
                (num_resets,),
                self.device,
            )

        for agent_idx, vehicle in enumerate(self._vehicles):
            self._raw_actions[agent_idx, env_ids] = 0.0
            self._semantic_actions[agent_idx, env_ids] = 0.0
            self._previous_raw_actions[agent_idx, env_ids] = 0.0
            self._terminal_goal_distance[agent_idx, env_ids] = 0.0
            self._agent_done_mask[agent_idx, env_ids] = False
            self._goal_done_mask[agent_idx, env_ids] = False
            self._collision_done_mask[agent_idx, env_ids] = False
            self._crash_done_mask[agent_idx, env_ids] = False
            self._lane_forbidden_done_mask[agent_idx, env_ids] = False
            self._pending_goal_done_mask[agent_idx, env_ids] = False
            self._pending_collision_done_mask[agent_idx, env_ids] = False
            self._pending_crash_done_mask[agent_idx, env_ids] = False
            self._pending_lane_forbidden_done_mask[agent_idx, env_ids] = False
            self._joint_effort_targets[agent_idx][env_ids] = 0.0
            self._external_forces[agent_idx][env_ids] = 0.0
            self._external_torques[agent_idx][env_ids] = 0.0
            self._brake_sign_memory[agent_idx][env_ids] = 1.0

            root_state = vehicle.data.default_root_state[env_ids].clone()
            if use_collision_test:
                half_distance = float(self.cfg.collision_test_half_distance_m)
                goal_distance = float(self.cfg.collision_test_goal_distance_m)
                start_x = -half_distance if agent_idx == 0 else half_distance
                goal_x = goal_distance if agent_idx == 0 else -goal_distance
                yaw_value = 0.0 if agent_idx == 0 else math.pi
                root_state[:, 0] = env_origins[:, 0] + start_x
                root_state[:, 1] = env_origins[:, 1]
                root_state[:, 2] = env_origins[:, 2] + float(self.cfg.spawn_height_m)
                yaw = torch.full((num_resets,), yaw_value, device=self.device)
                goal_local = torch.tensor(
                    [goal_x, 0.0, float(self.cfg.goal_height_m)],
                    device=self.device,
                ).unsqueeze(0).repeat(num_resets, 1)
                self._goal_pos_w[agent_idx, env_ids] = env_origins + goal_local
            elif use_scene_factory_roads:
                spawn_jitter = sample_uniform(
                    -float(self.cfg.agent_spawn_jitter_m),
                    float(self.cfg.agent_spawn_jitter_m),
                    (num_resets, 2),
                    self.device,
                )
                env_id_list = env_ids.detach().cpu().tolist()
                scenario_spawns = [self._scenario_spawns_by_env[int(env_id)][agent_idx] for env_id in env_id_list]
                start_local_xyz = torch.tensor(
                    [spawn.start_local_xyz for spawn in scenario_spawns],
                    dtype=torch.float32,
                    device=self.device,
                )
                start_yaw = torch.tensor(
                    [float(spawn.start_yaw_rad) for spawn in scenario_spawns],
                    dtype=torch.float32,
                    device=self.device,
                )
                goal_local_xyz = torch.tensor(
                    [spawn.goal_local_xyz for spawn in scenario_spawns],
                    dtype=torch.float32,
                    device=self.device,
                )
                root_state[:, 0] = env_origins[:, 0] + start_local_xyz[:, 0] + spawn_jitter[:, 0]
                root_state[:, 1] = env_origins[:, 1] + start_local_xyz[:, 1] + spawn_jitter[:, 1]
                root_state[:, 2] = (
                    env_origins[:, 2]
                    + float(self.cfg.spawn_height_m)
                    + start_local_xyz[:, 2]
                )
                yaw = start_yaw + sample_uniform(
                    -float(self.cfg.spawn_yaw_noise_rad),
                    float(self.cfg.spawn_yaw_noise_rad),
                    (num_resets,),
                    self.device,
                )
                goal_local = goal_local_xyz.clone()
                goal_local[:, 2] = torch.maximum(
                    goal_local[:, 2],
                    torch.full((num_resets,), float(self.cfg.goal_height_m), dtype=torch.float32, device=self.device),
                )
                self._goal_pos_w[agent_idx, env_ids] = env_origins + goal_local
            else:
                formation_angle = phase + (2.0 * math.pi * agent_idx / max(1, self._num_agents))
                spawn_jitter = sample_uniform(
                    -float(self.cfg.agent_spawn_jitter_m),
                    float(self.cfg.agent_spawn_jitter_m),
                    (num_resets, 2),
                    self.device,
                )
                spawn_offset = torch.stack(
                    [
                        float(self.cfg.agent_spawn_circle_radius_m) * torch.cos(formation_angle),
                        float(self.cfg.agent_spawn_circle_radius_m) * torch.sin(formation_angle),
                    ],
                    dim=-1,
                )
                root_state[:, 0:2] = env_origins[:, 0:2] + shared_world_offset + spawn_offset + spawn_jitter
                root_state[:, 2] = env_origins[:, 2] + float(self.cfg.spawn_height_m)
                yaw = formation_angle + math.pi + sample_uniform(
                    -float(self.cfg.spawn_yaw_noise_rad),
                    float(self.cfg.spawn_yaw_noise_rad),
                    (num_resets,),
                    self.device,
                )
                goal_heading = formation_angle + math.pi + sample_uniform(
                    -float(self.cfg.goal_heading_noise_rad),
                    float(self.cfg.goal_heading_noise_rad),
                    (num_resets,),
                    self.device,
                )
                env_goal_offset = torch.stack(
                    [
                        goal_radius * torch.cos(goal_heading),
                        goal_radius * torch.sin(goal_heading),
                        torch.full_like(goal_radius, float(self.cfg.goal_height_m)),
                    ],
                    dim=-1,
                )
                self._goal_pos_w[agent_idx, env_ids] = env_origins + env_goal_offset

            zeros = torch.zeros_like(yaw)
            root_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
            root_state[:, 7:] = 0.0
            self._previous_goal_distance[agent_idx, env_ids] = torch.linalg.norm(
                self._goal_pos_w[agent_idx, env_ids, :2] - root_state[:, :2],
                dim=1,
            )
            self._current_goal_distance[agent_idx, env_ids] = self._previous_goal_distance[agent_idx, env_ids]

            joint_pos = vehicle.data.default_joint_pos[env_ids].clone()
            joint_vel = vehicle.data.default_joint_vel[env_ids].clone()

            vehicle.write_root_pose_to_sim(root_state[:, :7], env_ids)
            vehicle.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
            vehicle.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
            vehicle.set_joint_effort_target(self._joint_effort_targets[agent_idx][env_ids], env_ids=env_ids)
            vehicle.permanent_wrench_composer.set_forces_and_torques(
                forces=self._external_forces[agent_idx][env_ids],
                torques=self._external_torques[agent_idx][env_ids],
                body_ids=self._base_body_ids[agent_idx],
                env_ids=env_ids,
                is_global=False,
            )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_goal_marker"):
                self._goal_marker = build_goal_beacon_marker("/Visuals/MultiGoalMarker")
            self._goal_marker.set_visibility(True)
            if bool(self.cfg.collision_test_debug_markers) and str(self.cfg.test_mode).strip().lower() in {
                "collision_test",
                "scene_factory_collision_test",
            }:
                if not hasattr(self, "_collision_test_vehicle_marker"):
                    self._collision_test_vehicle_marker = build_goal_beacon_marker(
                        "/Visuals/CollisionTestVehicles"
                    )
                self._collision_test_vehicle_marker.set_visibility(True)
        else:
            if hasattr(self, "_goal_marker"):
                self._goal_marker.set_visibility(False)
            if hasattr(self, "_collision_test_vehicle_marker"):
                self._collision_test_vehicle_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "scene") or self.scene is None or not hasattr(self, "_vehicles"):
            return
        if hasattr(self, "_goal_marker"):
            goal_positions = self._goal_pos_w.permute(1, 0, 2).reshape(-1, 3)
            marker_positions, marker_indices = goal_beacon_visualization(goal_positions)
            self._goal_marker.visualize(marker_positions, marker_indices=marker_indices)
        if hasattr(self, "_collision_test_vehicle_marker"):
            positions = []
            indices = []
            for agent_idx, vehicle in enumerate(self._vehicles):
                pos_w = vehicle.data.root_pos_w.clone()
                pos_w[:, 2] += 2.15
                positions.append(pos_w)
                indices.append(
                    torch.full((self.num_envs,), min(agent_idx, 1), dtype=torch.int32, device=self.device)
                )
            self._collision_test_vehicle_marker.visualize(
                torch.cat(positions, dim=0),
                marker_indices=torch.cat(indices, dim=0),
            )

    @property
    def tunable_config(self) -> StudentTunableConfig:
        return self._tunable_config

    def tunable_config_dict(self) -> dict:
        return asdict(self._tunable_config)

    def close(self):
        if self._capture_camera is not None:
            del self._capture_camera
            self._capture_camera = None
        super().close()
