from __future__ import annotations

from pathlib import Path

from src.student_vehicle_sysid import (
    ReplayLossWeights,
    SearchStage,
    StudentTunableConfig,
    TeacherDataset,
    TeacherRollout,
    auto_search_space_for_rollout,
    build_staged_search_plan,
    compute_rollout_loss,
    default_search_space,
    load_teacher_rollout,
    normalize_tunable_config,
    sample_tunable_config,
    touched_surface_names,
)


def _frame(
    *,
    x: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
    speed: float = 0.0,
    yaw_rate: float = 0.0,
    wheel_speed: float = 0.0,
    steer: float = 0.0,
    suspension: float = 0.0,
):
    return {
        "vehicle": {
            "position_m": [x, y, 0.0],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "yaw_rad": yaw,
            "linear_velocity_mps": [speed, 0.0, 0.0],
            "angular_velocity_rad_s": [0.0, 0.0, yaw_rate],
        },
        "wheels": [
            {
                "label": "front_left",
                "rotation_speed_rad_s": wheel_speed,
                "steer_angle_rad": steer,
                "suspension_jounce": suspension,
            },
            {
                "label": "front_right",
                "rotation_speed_rad_s": wheel_speed,
                "steer_angle_rad": steer,
                "suspension_jounce": suspension,
            },
            {
                "label": "rear_left",
                "rotation_speed_rad_s": wheel_speed,
                "steer_angle_rad": 0.0,
                "suspension_jounce": suspension,
            },
            {
                "label": "rear_right",
                "rotation_speed_rad_s": wheel_speed,
                "steer_angle_rad": 0.0,
                "suspension_jounce": suspension,
            },
        ],
    }


def test_compute_rollout_loss_is_zero_for_identical_rollouts() -> None:
    teacher = [_frame(x=1.0, y=2.0, yaw=0.2, speed=3.0, yaw_rate=0.1, wheel_speed=4.0, steer=0.05)]
    student = [_frame(x=1.0, y=2.0, yaw=0.2, speed=3.0, yaw_rate=0.1, wheel_speed=4.0, steer=0.05)]

    loss = compute_rollout_loss(teacher, student, ReplayLossWeights())

    assert loss["total_loss"] == 0.0
    assert loss["position_xy_mse"] == 0.0
    assert loss["wheel_speed_mse"] == 0.0


def test_compute_rollout_loss_wraps_yaw_error() -> None:
    teacher = [_frame(yaw=3.13)]
    student = [_frame(yaw=-3.13)]

    loss = compute_rollout_loss(teacher, student, ReplayLossWeights())

    assert loss["yaw_mse"] < 0.01


def test_sample_tunable_config_respects_bounds() -> None:
    search_space = default_search_space()
    config = sample_tunable_config(StudentTunableConfig(), search_space, rng=__import__("random").Random(7))

    assert search_space["drive_torque_nm"][0] <= config.drive_torque_nm <= search_space["drive_torque_nm"][1]
    assert (
        search_space["surface_longitudinal_scale.wet_asphalt"][0]
        <= config.surface_longitudinal_scale["wet_asphalt"]
        <= search_space["surface_longitudinal_scale.wet_asphalt"][1]
    )
    assert search_space["wheel_mass_kg"][0] <= config.wheel_mass_kg <= search_space["wheel_mass_kg"][1]


def test_load_teacher_rollout_reads_sample_rollout() -> None:
    rollout_dir = Path("artifacts/physx_teacher_datasets/smoke_recorded/rollouts/straight_accel_brake_smoke")
    rollout = load_teacher_rollout(rollout_dir)

    assert rollout.metadata["dt_s"] > 0.0
    assert len(rollout.frames) >= 1
    assert [patch.name for patch in rollout.patches] == ["dry_asphalt", "wet_asphalt", "gravel"]


def test_compute_rollout_loss_uses_minimum_frame_count() -> None:
    teacher = [_frame(x=0.0), _frame(x=10.0)]
    student = [_frame(x=1.0)]

    loss = compute_rollout_loss(teacher, student, ReplayLossWeights())

    assert loss["num_frames"] == 1.0
    assert loss["position_xy_mse"] == 1.0


def test_compute_rollout_loss_includes_terminal_penalty() -> None:
    teacher = [_frame(x=0.0, speed=0.0), _frame(x=0.0, speed=0.0)]
    student = [_frame(x=0.0, speed=0.0), _frame(x=2.0, speed=1.0)]

    loss = compute_rollout_loss(teacher, student, ReplayLossWeights())

    assert loss["terminal_position_xy_se"] == 4.0
    assert loss["terminal_speed_se"] == 1.0
    assert loss["total_loss"] > 6.0


def test_normalize_tunable_config_enforces_front_brake_bias() -> None:
    config = normalize_tunable_config(
        StudentTunableConfig(
            brake_front_torque_nm=600.0,
            brake_rear_torque_nm=1200.0,
            surface_longitudinal_scale={"dry_asphalt": 1.2},
        )
    )

    assert config.brake_front_torque_nm == 1200.0
    assert config.brake_rear_torque_nm == 600.0
    assert config.surface_lateral_scale["dry_asphalt"] == 1.0
    assert config.surface_longitudinal_scale["dry_asphalt"] == 1.2


def test_auto_search_space_for_straight_rollout_is_longitudinal_only() -> None:
    rollout = TeacherRollout(
        name="straight_accel_brake",
        rollout_dir=Path("/tmp/straight_accel_brake"),
        metadata={},
        frames=[
            {
                "vehicle": _frame()["vehicle"],
                "wheels": [
                    {"label": "front_left", "surface_name": "dry_asphalt"},
                    {"label": "front_right", "surface_name": "wet_asphalt"},
                ],
            }
        ],
        patches=[],
    )

    search_space = auto_search_space_for_rollout(rollout)

    assert "drive_torque_nm" in search_space
    assert "wheel_viscous_friction" in search_space
    assert "wheel_mass_kg" in search_space
    assert "steering_limit_rad" not in search_space
    assert "surface_longitudinal_scale.dry_asphalt" in search_space
    assert "surface_lateral_scale.dry_asphalt" not in search_space
    assert "surface_longitudinal_scale.gravel" not in search_space


def test_touched_surface_names_ignores_none() -> None:
    rollout = TeacherRollout(
        name="surface_transition_s",
        rollout_dir=Path("/tmp/surface_transition_s"),
        metadata={},
        frames=[
            {
                "vehicle": _frame()["vehicle"],
                "wheels": [
                    {"label": "front_left", "surface_name": "dry_asphalt"},
                    {"label": "front_right", "surface_name": None},
                    {"label": "rear_left", "surface_name": "gravel"},
                ],
            }
        ],
        patches=[],
    )

    assert touched_surface_names(rollout) == ["dry_asphalt", "gravel"]


def test_build_staged_search_plan_uses_expected_rollout_groups() -> None:
    def teacher(name: str, surfaces: list[str]) -> TeacherRollout:
        return TeacherRollout(
            name=name,
            rollout_dir=Path(f"/tmp/{name}"),
            metadata={},
            frames=[
                {
                    "vehicle": _frame()["vehicle"],
                    "wheels": [{"label": f"wheel_{idx}", "surface_name": surface} for idx, surface in enumerate(surfaces)],
                }
            ],
            patches=[],
        )

    dataset = TeacherDataset(
        manifest_path=Path("/tmp/manifest.json"),
        manifest={},
        rollouts=[
            teacher("straight_accel_brake", ["dry_asphalt", "wet_asphalt"]),
            teacher("step_steer_left", ["dry_asphalt"]),
            teacher("step_steer_right", ["dry_asphalt"]),
            teacher("sine_steer", ["dry_asphalt"]),
            teacher("surface_transition_s", ["dry_asphalt", "wet_asphalt", "gravel"]),
        ],
    )

    stages = build_staged_search_plan(dataset, total_random_trials=32)

    assert [stage.name for stage in stages] == ["longitudinal", "steering", "surface", "refinement", "brake_preservation"]
    assert [teacher.name for teacher in stages[0].teachers] == ["straight_accel_brake"]
    assert "drive_torque_nm" in stages[0].search_space
    assert "steering_limit_rad" not in stages[0].search_space
    assert "surface_longitudinal_scale.wet_asphalt" in stages[0].search_space
    assert [teacher.name for teacher in stages[1].teachers] == ["step_steer_left", "step_steer_right", "sine_steer"]
    assert "steering_limit_rad" in stages[1].search_space
    assert "surface_longitudinal_scale.gravel" in stages[2].search_space
    assert "surface_lateral_scale.gravel" in stages[2].search_space
    assert stages[3].search_window_fraction == 0.18
    assert stages[3].search_min_fraction == 0.05
    assert stages[4].seed_from_stage == "longitudinal"
    assert stages[4].search_window_fraction == 0.10
    assert [teacher.name for teacher in stages[4].teachers] == ["straight_accel_brake"]
    assert sum(stage.random_trials for stage in stages) == 32
