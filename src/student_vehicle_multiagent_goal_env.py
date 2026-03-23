from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
import torch

from src.isaaclab_bootstrap import ensure_isaaclab_source_paths
from src.student_vehicle_goal_env import (
    DEFAULT_STUDENT_VEHICLE_USD,
    _default_tunable_config_json,
    _dry_ground_material_cfg,
    _spawn_ground,
    _source_env_vehicle_root_path,
    build_student_vehicle_articulation_cfg,
)
from src.scene_factory_multiworld_scene import extract_vehicle_spawns_from_json, _build_roads_only, _load_yaml
from src.student_vehicle_sysid import (
    StudentTunableConfig,
    _apply_runtime_student_dynamics,
    load_tunable_config,
    normalize_tunable_config,
)
from src.trfc import prepare_stage_world_specs

ensure_isaaclab_source_paths()

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, sample_uniform, subtract_frame_transforms


OBSERVATION_MODE_DIMS = {
    "full": 22,
    "goal_reaching": 6,
}


def multi_agent_obs_dim(observation_mode: str) -> int:
    mode = str(observation_mode).strip().lower()
    if mode not in OBSERVATION_MODE_DIMS:
        raise ValueError(f"Unsupported observation mode: {observation_mode!r}")
    return int(OBSERVATION_MODE_DIMS[mode])


def configure_multi_agent_spaces(cfg: "StudentVehicleMultiAgentGoalEnvCfg", num_agents_per_env: int):
    agent_ids = [f"vehicle_{idx}" for idx in range(int(num_agents_per_env))]
    obs_dim = multi_agent_obs_dim(getattr(cfg, "observation_mode", "full"))
    cfg.num_agents_per_env = int(num_agents_per_env)
    cfg.possible_agents = agent_ids
    cfg.action_spaces = {
        agent_id: gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32) for agent_id in agent_ids
    }
    cfg.observation_spaces = {agent_id: obs_dim for agent_id in agent_ids}
    cfg.state_space = 0
    return cfg


def _scene_bbox_extent_xy(scene_json_path: str | Path) -> tuple[float, float]:
    import json

    with Path(scene_json_path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        scene_cfg = json.load(handle)
    points: list[tuple[float, float]] = []
    for polyline in ((scene_cfg.get("road") or {}).get("polylines") or []):
        for point in (polyline.get("xyz") or []):
            if len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
    if not points:
        return (0.0, 0.0)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (float(max(xs) - min(xs)), float(max(ys) - min(ys)))


def resolve_scene_factory_world_and_spawns(cfg: "StudentVehicleMultiAgentGoalEnvCfg"):
    scene_factory_cfg = _load_yaml(cfg.scene_factory_config_path)
    world_specs = prepare_stage_world_specs(scene_factory_cfg)
    if not world_specs:
        raise RuntimeError(f"No SceneFactory worlds resolved from {cfg.scene_factory_config_path}")

    vehicles_cfg = dict(scene_factory_cfg.get("vehicles", {}) or {})
    bounds_size_m = float((scene_factory_cfg.get("world", {}) or {}).get("bounds_size_m", 200.0))
    origin_mode = str((scene_factory_cfg.get("world", {}) or {}).get("origin_mode", "center"))
    requested_agents = max(int(cfg.num_agents_per_env), 1)

    def _extract(spec):
        spawns = extract_vehicle_spawns_from_json(
            spec.scene_json_path,
            bounds_size_m=bounds_size_m,
            origin_mode=origin_mode,
            max_controllable=requested_agents,
            require_goal_in_bounds=bool(vehicles_cfg.get("require_goal_in_bounds", True)),
            skip_if_start_in_goal=bool(vehicles_cfg.get("skip_if_start_in_goal", True)),
            goal_radius_m=float(vehicles_cfg.get("goal_radius_m", cfg.goal_reached_threshold_m)),
            start_goal_thresh_m=vehicles_cfg.get("start_goal_thresh_m"),
        )
        width_m, height_m = _scene_bbox_extent_xy(spec.scene_json_path)
        overflow_m = max(0.0, width_m - bounds_size_m) + max(0.0, height_m - bounds_size_m)
        return list(spawns), width_m, height_m, overflow_m

    requested_world_index = int(cfg.scene_factory_world_index)
    if requested_world_index >= 0:
        world_spec = world_specs[requested_world_index % len(world_specs)]
        spawns, width_m, height_m, overflow_m = _extract(world_spec)
        print(
            "[INFO] SceneFactory world selection: "
            f"world_index={world_spec.world_index} scene={world_spec.scene_json_name} "
            f"bbox={width_m:.1f}x{height_m:.1f}m overflow={overflow_m:.1f}m spawns={len(spawns)}"
        )
        return world_spec, spawns

    best_spec = None
    best_spawns = None
    best_key = None
    best_dims = None
    for world_spec in world_specs:
        spawns, width_m, height_m, overflow_m = _extract(world_spec)
        enough_agents = len(spawns) >= requested_agents
        key = (
            0 if enough_agents else 1,
            overflow_m,
            -min(len(spawns), requested_agents),
            width_m * height_m,
            world_spec.world_index,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_spec = world_spec
            best_spawns = spawns
            best_dims = (width_m, height_m, overflow_m)

    assert best_spec is not None and best_spawns is not None and best_dims is not None
    print(
        "[INFO] Auto-selected SceneFactory world: "
        f"world_index={best_spec.world_index} scene={best_spec.scene_json_name} "
        f"bbox={best_dims[0]:.1f}x{best_dims[1]:.1f}m overflow={best_dims[2]:.1f}m spawns={len(best_spawns)}"
    )
    return best_spec, best_spawns


def resolve_scene_factory_spawn_subset(cfg: "StudentVehicleMultiAgentGoalEnvCfg") -> list:
    _, spawns = resolve_scene_factory_world_and_spawns(cfg)
    return spawns


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


class StudentVehicleMultiAgentGoalEnv(DirectMARLEnv):
    cfg: StudentVehicleMultiAgentGoalEnvCfg

    def __init__(self, cfg: StudentVehicleMultiAgentGoalEnvCfg, render_mode: str | None = None, **kwargs):
        self._scenario_spawns: list | None = None
        self._scene_factory_scene_json_path: str | None = None
        if bool(cfg.use_scene_factory_roads):
            resolved_spec, resolved_spawns = resolve_scene_factory_world_and_spawns(cfg)
            cfg.scene_factory_world_index = int(resolved_spec.world_index)
            self._scene_factory_scene_json_path = str(Path(resolved_spec.scene_json_path).expanduser().resolve())
            available = len(resolved_spawns)
            requested = int(cfg.num_agents_per_env)
            if available <= 0:
                raise RuntimeError(
                    "SceneFactory source world does not provide any controllable spawns "
                    f"for {cfg.scene_factory_config_path}"
                )
            if available < requested:
                print(
                    "[WARN] SceneFactory source world provides fewer controllable spawns than requested: "
                    f"requested={requested}, available={available}. Training with {available} agents."
                )
            cfg.num_agents_per_env = min(requested, available)
            self._scenario_spawns = resolved_spawns[: int(cfg.num_agents_per_env)]
        configure_multi_agent_spaces(cfg, cfg.num_agents_per_env)
        self._tunable_config = normalize_tunable_config(
            load_tunable_config(cfg.tunable_config_json) if str(cfg.tunable_config_json) else StudentTunableConfig()
        )
        self._agent_ids = list(cfg.possible_agents)
        super().__init__(cfg, render_mode, **kwargs)

        self._num_agents = len(self._agent_ids)
        self._vehicles = [self.scene.articulations[agent_id] for agent_id in self._agent_ids]

        self._raw_actions = torch.zeros(self._num_agents, self.num_envs, 3, device=self.device)
        self._semantic_actions = torch.zeros_like(self._raw_actions)
        self._previous_raw_actions = torch.zeros_like(self._raw_actions)
        self._goal_pos_w = torch.zeros(self._num_agents, self.num_envs, 3, device=self.device)
        self._previous_goal_distance = torch.zeros(self._num_agents, self.num_envs, device=self.device)
        self._current_goal_distance = torch.zeros(self._num_agents, self.num_envs, device=self.device)

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
        )
        self._episode_sums = {
            key: torch.zeros(self._num_agents, self.num_envs, dtype=torch.float32, device=self.device)
            for key in reward_keys
        }

        self.set_debug_vis(bool(self.cfg.debug_vis))

    def _setup_scene(self):
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if self.cfg.use_scene_factory_roads:
            self._setup_scene_factory_source_world(stage)

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

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        for agent_id, vehicle in spawned_vehicles.items():
            # Register scene entities after cloning to match Isaac Lab's direct MARL task setup.
            self.scene.articulations[agent_id] = vehicle

        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _setup_scene_factory_source_world(self, stage) -> None:
        scene_factory_cfg = _load_yaml(self.cfg.scene_factory_config_path)
        if not self._scene_factory_scene_json_path:
            world_spec, _ = resolve_scene_factory_world_and_spawns(self.cfg)
            self._scene_factory_scene_json_path = str(Path(world_spec.scene_json_path).expanduser().resolve())

        build_cfg = dict(scene_factory_cfg)
        build_cfg["world"] = dict(scene_factory_cfg.get("world", {}) or {})
        build_cfg["world"]["root_container"] = "/World/envs/env_0/SceneFactoryWorlds"
        build_cfg["world"]["world_count"] = 1
        build_cfg["world"]["grid_cols"] = 1
        build_cfg["world"]["padding_m"] = 0.0
        _build_roads_only(stage=stage, cfg=build_cfg, json_paths=[self._scene_factory_scene_json_path])
        if self._scenario_spawns is None:
            self._scenario_spawns = resolve_scene_factory_spawn_subset(self.cfg)[: int(self.cfg.num_agents_per_env)]

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

    def _apply_action(self):
        for agent_idx, vehicle in enumerate(self._vehicles):
            joint_pos = vehicle.data.joint_pos
            joint_vel = vehicle.data.joint_vel

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
            vehicle.set_external_force_and_torque(
                forces=self._external_forces[agent_idx],
                torques=self._external_torques[agent_idx],
                body_ids=self._base_body_ids[agent_idx],
                is_global=False,
            )

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
        observations = {}
        observation_mode = str(self.cfg.observation_mode).strip().lower()
        neighbor_obs_scale = float(max(1.0e-6, self.cfg.agent_neighbor_obs_scale_m))
        goal_obs_scale = float(max(1.0e-6, self.cfg.goal_radius_max_m))
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
            else:
                raise ValueError(f"Unsupported observation mode: {self.cfg.observation_mode!r}")
            observations[agent_id] = obs
        return observations

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        rewards = {}
        pairwise_distances = self._pairwise_distances_xy()
        nearest_distances = pairwise_distances.min(dim=2).values if self._num_agents > 1 else None
        collision_world = (
            torch.any(pairwise_distances < float(self.cfg.agent_collision_distance_m), dim=(1, 2))
            if self._num_agents > 1
            else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )

        for agent_idx, agent_id in enumerate(self._agent_ids):
            goal_pos_b, goal_distance = self._compute_goal_distance(agent_idx)
            self._current_goal_distance[agent_idx] = goal_distance

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
            goal_bonus = (
                goal_distance <= float(self.cfg.goal_reached_threshold_m)
            ).float() * float(self.cfg.reward_goal_bonus)

            crash_mask = self._agent_crash_mask(agent_idx)
            crash_penalty = crash_mask.float() * float(self.cfg.reward_crash_penalty)
            collision_penalty = collision_world.float() * float(self.cfg.reward_collision_penalty)

            if nearest_distances is not None:
                proximity_violation = torch.relu(
                    float(self.cfg.agent_safe_distance_m) - nearest_distances[:, agent_idx]
                ) / float(max(1.0e-6, self.cfg.agent_safe_distance_m))
            else:
                proximity_violation = torch.zeros(self.num_envs, device=self.device)
            neighbor_proximity = proximity_violation * float(self.cfg.reward_scale_neighbor_proximity)

            reward_terms = {
                "alive": torch.full_like(goal_distance, float(self.cfg.reward_scale_alive)),
                "progress": progress * float(self.cfg.reward_scale_progress),
                "goal_shaping": goal_shaping * float(self.cfg.reward_scale_goal_shaping),
                "heading": heading_alignment * float(self.cfg.reward_scale_heading),
                "speed_to_goal": speed_to_goal * float(self.cfg.reward_scale_speed_to_goal),
                "lateral_velocity": lateral_velocity * float(self.cfg.reward_scale_lateral_velocity),
                "yaw_rate": yaw_rate * float(self.cfg.reward_scale_yaw_rate),
                "action_rate": action_rate * float(self.cfg.reward_scale_action_rate),
                "action_magnitude": action_magnitude * float(self.cfg.reward_scale_action_magnitude),
                "throttle_brake_conflict": throttle_brake_conflict
                * float(self.cfg.reward_scale_throttle_brake_conflict),
                "neighbor_proximity": neighbor_proximity,
                "goal_bonus": goal_bonus,
                "collision_penalty": collision_penalty,
                "crash_penalty": crash_penalty,
            }
            rewards[agent_id] = torch.sum(torch.stack(list(reward_terms.values())), dim=0)

            for key, value in reward_terms.items():
                self._episode_sums[key][agent_idx] += value

            self._previous_goal_distance[agent_idx] = goal_distance
            self._previous_raw_actions[agent_idx] = self._raw_actions[agent_idx]

        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        goal_reached = []
        crash_masks = []
        for agent_idx in range(self._num_agents):
            _, goal_distance = self._compute_goal_distance(agent_idx)
            self._current_goal_distance[agent_idx] = goal_distance
            goal_reached.append(goal_distance <= float(self.cfg.goal_reached_threshold_m))
            crash_masks.append(self._agent_crash_mask(agent_idx))

        all_goals_reached = torch.stack(goal_reached, dim=0).all(dim=0)
        any_crash = torch.stack(crash_masks, dim=0).any(dim=0)
        pairwise_distances = self._pairwise_distances_xy()
        collision_world = (
            torch.any(pairwise_distances < float(self.cfg.agent_collision_distance_m), dim=(1, 2))
            if self._num_agents > 1
            else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated_world = all_goals_reached | any_crash | collision_world

        terminated = {agent_id: terminated_world.clone() for agent_id in self._agent_ids}
        time_outs = {agent_id: time_out.clone() for agent_id in self._agent_ids}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._vehicles[0]._ALL_INDICES

        pairwise_distances = self._pairwise_distances_xy()
        collision_world = (
            torch.any(pairwise_distances < float(self.cfg.agent_collision_distance_m), dim=(1, 2))
            if self._num_agents > 1
            else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )

        for agent_idx, agent_id in enumerate(self._agent_ids):
            self.extras[agent_id] = {"log": {}}
            for key, value in self._episode_sums.items():
                self.extras[agent_id]["log"][f"Episode_Reward/{key}"] = torch.mean(value[agent_idx, env_ids]).item() / max(
                    1.0, float(self.max_episode_length_s)
                )
                value[agent_idx, env_ids] = 0.0
            self.extras[agent_id]["log"]["Metrics/final_distance_to_goal"] = torch.mean(
                self._current_goal_distance[agent_idx, env_ids]
            ).item()
            self.extras[agent_id]["log"]["Metrics/collision_count"] = float(
                torch.count_nonzero(collision_world[env_ids]).item()
            )

        for vehicle in self._vehicles:
            vehicle.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        num_resets = len(env_ids)
        env_origins = self.scene.env_origins[env_ids]
        use_scene_factory_roads = bool(self.cfg.use_scene_factory_roads and self._scenario_spawns is not None)
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
            self._joint_effort_targets[agent_idx][env_ids] = 0.0
            self._external_forces[agent_idx][env_ids] = 0.0
            self._external_torques[agent_idx][env_ids] = 0.0
            self._brake_sign_memory[agent_idx][env_ids] = 1.0

            root_state = vehicle.data.default_root_state[env_ids].clone()
            if use_scene_factory_roads:
                scenario_spawn = self._scenario_spawns[agent_idx]
                spawn_jitter = sample_uniform(
                    -float(self.cfg.agent_spawn_jitter_m),
                    float(self.cfg.agent_spawn_jitter_m),
                    (num_resets, 2),
                    self.device,
                )
                root_state[:, 0] = env_origins[:, 0] + float(scenario_spawn.start_local_xyz[0]) + spawn_jitter[:, 0]
                root_state[:, 1] = env_origins[:, 1] + float(scenario_spawn.start_local_xyz[1]) + spawn_jitter[:, 1]
                root_state[:, 2] = (
                    env_origins[:, 2]
                    + float(self.cfg.spawn_height_m)
                    + float(scenario_spawn.start_local_xyz[2])
                )
                yaw = torch.full(
                    (num_resets,),
                    float(scenario_spawn.start_yaw_rad),
                    device=self.device,
                ) + sample_uniform(
                    -float(self.cfg.spawn_yaw_noise_rad),
                    float(self.cfg.spawn_yaw_noise_rad),
                    (num_resets,),
                    self.device,
                )
                goal_local = torch.tensor(
                    [
                        float(scenario_spawn.goal_local_xyz[0]),
                        float(scenario_spawn.goal_local_xyz[1]),
                        max(float(self.cfg.goal_height_m), float(scenario_spawn.goal_local_xyz[2])),
                    ],
                    device=self.device,
                ).unsqueeze(0).repeat(num_resets, 1)
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
            vehicle.set_external_force_and_torque(
                forces=self._external_forces[agent_idx][env_ids],
                torques=self._external_torques[agent_idx][env_ids],
                body_ids=self._base_body_ids[agent_idx],
                env_ids=env_ids,
                is_global=False,
            )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_goal_marker"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.28, 0.28, 0.08)
                marker_cfg.prim_path = "/Visuals/MultiGoalMarker"
                self._goal_marker = VisualizationMarkers(marker_cfg)
            self._goal_marker.set_visibility(True)
        else:
            if hasattr(self, "_goal_marker"):
                self._goal_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        if hasattr(self, "_goal_marker"):
            self._goal_marker.visualize(self._goal_pos_w.permute(1, 0, 2).reshape(-1, 3))

    @property
    def tunable_config(self) -> StudentTunableConfig:
        return self._tunable_config

    def tunable_config_dict(self) -> dict:
        return asdict(self._tunable_config)
