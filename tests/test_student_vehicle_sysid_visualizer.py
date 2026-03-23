from __future__ import annotations

import json
from pathlib import Path

from src.student_vehicle_sysid_visualizer import load_sysid_dir, write_sysid_report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _frame(step: int, x: float, speed: float) -> dict:
    return {
        "step": step,
        "sim_time_s": 0.1 * (step + 1),
        "command": {"accelerator": 0.2, "steering": 0.0, "brake": 0.0},
        "vehicle": {
            "position_m": [x, 0.0, 0.0],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "yaw_rad": 0.0,
            "linear_velocity_mps": [speed, 0.0, 0.0],
            "angular_velocity_rad_s": [0.0, 0.0, 0.0],
        },
        "wheels": [],
    }


def test_write_sysid_report_generates_html(tmp_path: Path) -> None:
    teacher_rollout_dir = tmp_path / "teacher_rollout"
    sysid_dir = tmp_path / "sysid"
    trial_dir = sysid_dir / "trials" / "sample_000"

    teacher_meta = {
        "dt_s": 0.1,
        "command_program_source": "demo.json",
        "surface_patches": [
            {
                "name": "dry_asphalt",
                "x_center_m": 0.0,
                "y_center_m": 0.0,
                "length_m": 10.0,
                "width_m": 4.0,
                "static_friction": 1.0,
                "dynamic_friction": 0.9,
                "tire_friction": 1.0,
                "color_srgb": [0.2, 0.2, 0.2],
            }
        ],
    }
    teacher_frames = [_frame(0, 0.0, 0.1), _frame(1, 0.1, 0.2)]
    student_frames = [_frame(0, 0.01, 0.09), _frame(1, 0.11, 0.19)]
    student_meta = {
        "teacher_rollout_dir": str(teacher_rollout_dir),
        "student_config": {"drive_torque_nm": 900.0},
        "loss_weights": {"position_xy": 1.0},
    }
    loss = {
        "total_loss": 0.02,
        "position_xy_mse": 0.01,
        "yaw_mse": 0.0,
        "speed_mse": 0.0,
        "yaw_rate_mse": 0.0,
        "wheel_speed_mse": 0.0,
        "steer_angle_mse": 0.0,
        "suspension_mse": 0.0,
    }
    best_result = {
        "teacher_rollout_dir": str(teacher_rollout_dir),
        "best_trial_name": "sample_000",
        "best_trial_output_dir": str(trial_dir),
        "best_loss": loss,
        "best_config": {"drive_torque_nm": 900.0},
        "num_trials": 1,
    }
    history = [
        {
            "trial_name": "sample_000",
            "output_dir": str(trial_dir),
            "num_frames": 2,
            "loss": loss,
            "config": {"drive_torque_nm": 900.0},
        }
    ]

    _write_json(teacher_rollout_dir / "rollout_meta.json", teacher_meta)
    _write_jsonl(teacher_rollout_dir / "rollout_frames.jsonl", teacher_frames)
    _write_json(sysid_dir / "best_result.json", best_result)
    _write_json(sysid_dir / "best_config.json", {"drive_torque_nm": 900.0})
    _write_jsonl(sysid_dir / "search_history.jsonl", history)
    _write_json(trial_dir / "student_sysid_meta.json", student_meta)
    _write_jsonl(trial_dir / "student_rollout_frames.jsonl", student_frames)

    payload = load_sysid_dir(sysid_dir)
    assert payload["best_result"]["best_trial_name"] == "sample_000"
    assert len(payload["teacher_frames"]) == 2
    assert len(payload["student_frames"]) == 2

    report_path = write_sysid_report(sysid_dir)
    html_text = report_path.read_text(encoding="utf-8")
    assert "Student Vehicle SysId Report" in html_text
    assert "Search Progress" in html_text
    assert "Trajectory Match" in html_text


def test_write_sysid_report_generates_staged_root_html(tmp_path: Path) -> None:
    teacher_rollout_dir = tmp_path / "teacher_rollout"
    sysid_dir = tmp_path / "sysid"
    stage_dir = sysid_dir / "stages" / "01_longitudinal"
    trial_dir = stage_dir / "trials" / "sample_003"
    final_report_dir = sysid_dir / "final_report"
    final_bundle_dir = sysid_dir / "final_bundle_report"

    teacher_meta = {
        "dt_s": 0.1,
        "command_program_source": "straight_accel_brake.json",
        "surface_patches": [
            {
                "name": "dry_asphalt",
                "x_center_m": 0.0,
                "y_center_m": 0.0,
                "length_m": 10.0,
                "width_m": 4.0,
                "static_friction": 1.0,
                "dynamic_friction": 0.9,
                "tire_friction": 1.0,
                "color_srgb": [0.2, 0.2, 0.2],
            }
        ],
    }
    teacher_frames = [_frame(0, 0.0, 0.1), _frame(1, 0.1, 0.2)]
    student_frames = [_frame(0, 0.01, 0.09), _frame(1, 0.11, 0.19)]
    student_meta = {
        "teacher_rollout_dir": str(teacher_rollout_dir),
        "student_config": {"drive_torque_nm": 900.0},
        "loss_weights": {"position_xy": 1.0},
    }
    loss = {
        "total_loss": 0.5,
        "position_xy_mse": 0.1,
        "yaw_mse": 0.0,
        "speed_mse": 0.0,
        "yaw_rate_mse": 0.0,
        "wheel_speed_mse": 0.0,
        "steer_angle_mse": 0.0,
        "suspension_mse": 0.0,
    }
    representative_best_result = {
        "teacher_rollout_dir": str(teacher_rollout_dir),
        "best_trial_name": "single",
        "best_trial_output_dir": str(final_report_dir),
        "best_loss": loss,
        "best_config": {"drive_torque_nm": 900.0},
        "num_trials": 1,
    }
    stage_history = [
        {
            "trial_name": "baseline",
            "stage_name": "longitudinal",
            "output_dir": str(stage_dir / "trials" / "baseline"),
            "num_frames": 2,
            "loss": {"total_loss": 5.0},
            "config": {"drive_torque_nm": 700.0},
        },
        {
            "trial_name": "sample_003",
            "stage_name": "longitudinal",
            "output_dir": str(trial_dir),
            "num_frames": 2,
            "loss": loss,
            "config": {"drive_torque_nm": 900.0},
        },
    ]
    staged_summary = {
        "teacher_dataset_manifest": str(tmp_path / "manifest.json"),
        "student_usd_path": str(tmp_path / "student.usd"),
        "search_mode": "staged",
        "optimizer": "cem",
        "best_config": {"drive_torque_nm": 900.0},
        "stages": [
            {
                "name": "longitudinal",
                "best_trial_name": "sample_003",
                "best_trial_output_dir": str(trial_dir),
                "best_loss": loss,
            }
        ],
        "representative_rollout_dir": str(teacher_rollout_dir),
        "representative_report_html": str(final_report_dir / "sysid_report.html"),
        "final_bundle_report_dir": str(final_bundle_dir),
        "final_bundle_rollout_names": ["straight_accel_brake", "step_steer_left", "sine_steer"],
    }

    _write_json(teacher_rollout_dir / "rollout_meta.json", teacher_meta)
    _write_jsonl(teacher_rollout_dir / "rollout_frames.jsonl", teacher_frames)
    _write_json(sysid_dir / "best_result.json", staged_summary)
    _write_json(sysid_dir / "best_config.json", {"drive_torque_nm": 900.0})
    _write_jsonl(stage_dir / "search_history.jsonl", stage_history)
    _write_json(final_report_dir / "best_result.json", representative_best_result)
    _write_json(final_report_dir / "student_sysid_meta.json", student_meta)
    _write_jsonl(final_report_dir / "student_rollout_frames.jsonl", student_frames)

    bundle_rollout_names = ["straight_accel_brake", "step_steer_left", "sine_steer"]
    teacher_rollout_dirs = {}
    per_rollout = {}
    for index, rollout_name in enumerate(bundle_rollout_names):
        rollout_teacher_dir = tmp_path / rollout_name
        rollout_student_dir = final_bundle_dir / rollout_name
        teacher_rollout_dirs[rollout_name] = str(rollout_teacher_dir)
        _write_json(
            rollout_teacher_dir / "rollout_meta.json",
            {
                **teacher_meta,
                "command_program_source": f"{rollout_name}.json",
            },
        )
        _write_jsonl(
            rollout_teacher_dir / "rollout_frames.jsonl",
            [_frame(0, 0.1 * index, 0.1), _frame(1, 0.1 * index + 0.1, 0.2)],
        )
        _write_json(
            rollout_student_dir / "student_sysid_meta.json",
            {
                "teacher_rollout_dir": str(rollout_teacher_dir),
                "student_config": {"drive_torque_nm": 900.0},
                "loss_weights": {"position_xy": 1.0},
            },
        )
        _write_jsonl(
            rollout_student_dir / "student_rollout_frames.jsonl",
            [_frame(0, 0.1 * index + 0.01, 0.09), _frame(1, 0.1 * index + 0.11, 0.19)],
        )
        per_rollout[rollout_name] = {
            "teacher_rollout_dir": str(rollout_teacher_dir),
            "output_dir": str(rollout_student_dir),
            "loss": {"total_loss": 0.5 + 0.1 * index},
            "num_frames": 2,
        }
    _write_json(
        final_bundle_dir / "trial_bundle_summary.json",
        {
            "teacher_rollout_dirs": teacher_rollout_dirs,
            "student_usd_path": str(tmp_path / "student.usd"),
            "student_config": {"drive_torque_nm": 900.0},
            "loss_weights": {"position_xy": 1.0},
            "per_rollout": per_rollout,
            "aggregate_loss": {"total_loss": 0.75},
            "num_rollouts": 3,
            "num_frames": 6,
        },
    )

    payload = load_sysid_dir(sysid_dir)
    assert payload["is_staged_summary"] is True
    assert len(payload["history"]) == 2
    assert payload["best_result"]["best_trial_name"] == "sample_003"
    assert len(payload["report_rollouts"]) == 3

    report_path = write_sysid_report(sysid_dir)
    html_text = report_path.read_text(encoding="utf-8")
    assert "longitudinal" in html_text
    assert "Search Progress" in html_text
    assert "straight_accel_brake" in html_text
    assert "step_steer_left" in html_text
    assert "sine_steer" in html_text
