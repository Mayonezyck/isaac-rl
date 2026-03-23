from __future__ import annotations

import argparse
import math
from pathlib import Path

from src.isaaclab_bootstrap import ensure_isaaclab_source_paths

ensure_isaaclab_source_paths()

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Spawn the multi-agent student-vehicle scene and step it with zero actions to validate stability."
)
parser.add_argument("--num_envs", type=int, default=2, help="Number of parallel environments.")
parser.add_argument("--num_agents_per_env", type=int, default=2, help="Number of vehicles inside each environment.")
parser.add_argument("--num_steps", type=int, default=600, help="Number of environment steps to simulate.")
parser.add_argument("--report_every", type=int, default=60, help="Print metrics every N environment steps.")
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument("--student_usd", type=str, default="", help="Path to the student vehicle USD.")
parser.add_argument(
    "--tunable_config_json",
    type=str,
    default="",
    help="Path to the tuned student config JSON. Empty uses the environment default.",
)
parser.add_argument("--spawn_height_m", type=float, default=1.6, help="Vehicle spawn height above each env origin.")
parser.add_argument(
    "--ground_mode",
    choices=("cuboid", "plane"),
    default="plane",
    help="Ground implementation to spawn for the debug scene.",
)
parser.add_argument("--env_spacing", type=float, default=14.0, help="Spacing between vectorized environments.")
parser.add_argument("--start_radius_m", type=float, default=None, help="Optional override for per-env shared spawn offset.")
parser.add_argument(
    "--agent_spawn_circle_radius_m",
    type=float,
    default=None,
    help="Optional override for the circle radius used to place vehicles within one world.",
)
parser.add_argument(
    "--agent_spawn_jitter_m",
    type=float,
    default=None,
    help="Optional override for per-agent XY spawn jitter.",
)
parser.add_argument(
    "--randomize_spawn_phase",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Enable or disable random rotation of the whole within-world formation.",
)
parser.add_argument(
    "--spawn_yaw_noise_rad",
    type=float,
    default=None,
    help="Optional override for additional random yaw noise at reset.",
)
parser.add_argument(
    "--goal_heading_noise_rad",
    type=float,
    default=None,
    help="Optional override for goal heading randomness.",
)
parser.add_argument(
    "--apply_runtime_external_wrench",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Enable or disable the runtime lateral/yaw stabilization wrench.",
)
parser.add_argument(
    "--disable_base_collision",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Disable base-link collision for debugging whether the chassis is bottoming out before the wheels support it.",
)
parser.add_argument(
    "--disable_wheel_collision",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Disable wheel-link collision for debugging whether wheel contact shapes are the source of the issue.",
)
parser.add_argument(
    "--replicate_physics",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Whether to use Isaac Lab physics replication when cloning homogeneous environments.",
)
parser.add_argument(
    "--clone_in_fabric",
    choices=("auto", "true", "false"),
    default="auto",
    help="Whether to enable fabric cloning. 'auto' uses it only for headless runs.",
)
parser.add_argument(
    "--debug_vis",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Whether to show debug markers. Default enables them only for GUI runs.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

from src.student_vehicle_goal_env import DEFAULT_STUDENT_VEHICLE_USD
from src.student_vehicle_multiagent_goal_env import (
    StudentVehicleMultiAgentGoalEnv,
    StudentVehicleMultiAgentGoalEnvCfg,
    configure_multi_agent_spaces,
)


def _build_env_cfg() -> StudentVehicleMultiAgentGoalEnvCfg:
    cfg = StudentVehicleMultiAgentGoalEnvCfg()
    cfg.seed = int(args_cli.seed)
    cfg.scene.num_envs = int(args_cli.num_envs)
    cfg.scene.env_spacing = float(args_cli.env_spacing)
    cfg.scene.replicate_physics = bool(args_cli.replicate_physics)
    cfg.debug_vis = bool(args_cli.debug_vis) if args_cli.debug_vis is not None else (not bool(args_cli.headless))
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
    cfg.viewer.eye = (max(20.0, grid_extent), max(20.0, grid_extent), max(16.0, 0.8 * grid_extent))
    cfg.viewer.lookat = (0.0, 0.0, 0.0)
    cfg.spawn_height_m = float(args_cli.spawn_height_m)
    cfg.ground_mode = str(args_cli.ground_mode)
    if args_cli.start_radius_m is not None:
        cfg.start_radius_m = float(args_cli.start_radius_m)
    if args_cli.agent_spawn_circle_radius_m is not None:
        cfg.agent_spawn_circle_radius_m = float(args_cli.agent_spawn_circle_radius_m)
    if args_cli.agent_spawn_jitter_m is not None:
        cfg.agent_spawn_jitter_m = float(args_cli.agent_spawn_jitter_m)
    if args_cli.randomize_spawn_phase is not None:
        cfg.randomize_spawn_phase = bool(args_cli.randomize_spawn_phase)
    if args_cli.spawn_yaw_noise_rad is not None:
        cfg.spawn_yaw_noise_rad = float(args_cli.spawn_yaw_noise_rad)
    if args_cli.goal_heading_noise_rad is not None:
        cfg.goal_heading_noise_rad = float(args_cli.goal_heading_noise_rad)
    if args_cli.apply_runtime_external_wrench is not None:
        cfg.apply_runtime_external_wrench = bool(args_cli.apply_runtime_external_wrench)
    cfg.student_usd_path = str(Path(args_cli.student_usd or DEFAULT_STUDENT_VEHICLE_USD).expanduser().resolve())
    if str(args_cli.tunable_config_json):
        cfg.tunable_config_json = str(Path(args_cli.tunable_config_json).expanduser().resolve())
    configure_multi_agent_spaces(cfg, int(args_cli.num_agents_per_env))
    return cfg


def _scene_metrics(env: StudentVehicleMultiAgentGoalEnv) -> tuple[float, float, float, float]:
    heights = torch.stack(
        [vehicle.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] for vehicle in env._vehicles],
        dim=1,
    )
    speeds = torch.stack(
        [torch.linalg.norm(vehicle.data.root_lin_vel_w[:, :2], dim=1) for vehicle in env._vehicles],
        dim=1,
    )
    pairwise_distances = env._pairwise_distances_xy()
    finite_pairwise = pairwise_distances[torch.isfinite(pairwise_distances)]
    min_pairwise = float(finite_pairwise.min().item()) if finite_pairwise.numel() > 0 else float("nan")
    min_height = float(heights.min().item())
    max_height = float(heights.max().item())
    max_speed = float(speeds.max().item())
    return min_height, max_height, min_pairwise, max_speed


def _set_collision_enabled(stage, prim_path: str, enabled: bool) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    stack = [prim]
    while stack:
        current = stack.pop()
        attr = current.GetAttribute("physics:collisionEnabled")
        if attr.IsValid():
            attr.Set(bool(enabled))
        for child in current.GetChildren():
            stack.append(child)


def _apply_collision_debug_overrides(env: StudentVehicleMultiAgentGoalEnv) -> None:
    if not args_cli.disable_base_collision and not args_cli.disable_wheel_collision:
        return

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    for env_idx in range(env.num_envs):
        for agent_idx in range(env.cfg.num_agents_per_env):
            root_path = f"/World/envs/env_{env_idx}/Vehicle_{agent_idx}"
            if args_cli.disable_base_collision:
                _set_collision_enabled(stage, f"{root_path}/base_link", False)
            if args_cli.disable_wheel_collision:
                for wheel_name in (
                    "front_left_wheel_link",
                    "front_right_wheel_link",
                    "rear_left_wheel_link",
                    "rear_right_wheel_link",
                ):
                    _set_collision_enabled(stage, f"{root_path}/{wheel_name}", False)


def _vehicle_debug_lines(env: StudentVehicleMultiAgentGoalEnv, env_index: int = 0) -> list[str]:
    debug_lines: list[str] = []
    for agent_idx, vehicle in enumerate(env._vehicles):
        wheel_body_ids, _ = vehicle.find_bodies(
            [
                "front_left_wheel_link",
                "front_right_wheel_link",
                "rear_left_wheel_link",
                "rear_right_wheel_link",
            ],
            preserve_order=True,
        )
        base_body_id, _ = vehicle.find_bodies("base_link")
        suspension_joint_ids, _ = vehicle.find_joints(
            [
                "front_left_suspension_joint",
                "front_right_suspension_joint",
                "rear_left_suspension_joint",
                "rear_right_suspension_joint",
            ],
            preserve_order=True,
        )
        root_pos = vehicle.data.root_pos_w[env_index]
        base_pos = vehicle.data.body_pos_w[env_index, base_body_id[0]]
        wheel_pos = vehicle.data.body_pos_w[env_index, wheel_body_ids]
        suspension_pos = vehicle.data.joint_pos[env_index, suspension_joint_ids]
        env_origin_z = env.scene.env_origins[env_index, 2]
        debug_lines.append(
            "agent="
            f"{env.possible_agents[agent_idx]} "
            f"root_z={float(root_pos[2] - env_origin_z):.3f} "
            f"base_z={float(base_pos[2] - env_origin_z):.3f} "
            f"min_wheel_z={float(wheel_pos[:, 2].min() - env_origin_z):.3f} "
            f"max_wheel_z={float(wheel_pos[:, 2].max() - env_origin_z):.3f} "
            f"susp=[{', '.join(f'{float(v):.3f}' for v in suspension_pos.tolist())}]"
        )
    return debug_lines


def main():
    env_cfg = _build_env_cfg()
    print(
        "[INFO] Multi-agent spawn check:",
        f"num_envs={env_cfg.scene.num_envs}",
        f"num_agents_per_env={env_cfg.num_agents_per_env}",
        f"replicate_physics={env_cfg.scene.replicate_physics}",
        f"clone_in_fabric={env_cfg.scene.clone_in_fabric}",
        f"device={env_cfg.sim.device}",
    )
    env = StudentVehicleMultiAgentGoalEnv(env_cfg, render_mode=None)
    _apply_collision_debug_overrides(env)
    zero_actions = {
        agent_id: torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32) for agent_id in env.possible_agents
    }

    try:
        env.reset()
        print("[RESET]")
        for line in _vehicle_debug_lines(env):
            print("  " + line)
        for step_idx in range(int(args_cli.num_steps)):
            env.step(zero_actions)
            should_report = step_idx == 0 or (step_idx + 1) % max(1, int(args_cli.report_every)) == 0
            if should_report:
                min_height, max_height, min_pairwise, max_speed = _scene_metrics(env)
                print(
                    f"[STEP {step_idx + 1:05d}] "
                    f"min_height={min_height:.3f} "
                    f"max_height={max_height:.3f} "
                    f"min_pairwise_xy={min_pairwise:.3f} "
                    f"max_planar_speed={max_speed:.3f}"
                )
                for line in _vehicle_debug_lines(env):
                    print("  " + line)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
