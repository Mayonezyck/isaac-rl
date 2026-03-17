import json
import math
from pathlib import Path

from src.physx_teacher_command_program_generator import (
    build_program_from_preset,
    build_constant_steer_program,
    build_sine_steer_program,
    build_step_steer_program,
    build_straight_accel_brake_program,
    build_surface_transition_s_program,
)
from src.physx_teacher_patch_track import load_command_program


def test_build_straight_accel_brake_program_has_expected_sequence() -> None:
    program = build_straight_accel_brake_program(idle_s=1.0, accel_s=2.0, coast_s=0.5, brake_s=1.0, throttle=0.5, brake=0.7)
    payload = program.to_dict_list()
    assert [segment["label"] for segment in payload] == ["idle", "straight_accel", "coast", "service_brake"]
    assert payload[1]["command"]["accelerator"] == 0.5
    assert payload[3]["command"]["brake"] == 0.7


def test_build_step_steer_program_emits_step_then_recenter() -> None:
    program = build_step_steer_program(entry_s=1.5, step_hold_s=2.5, recenter_s=0.75, throttle=0.4, steer=-0.3)
    payload = program.to_dict_list()
    assert payload[1]["label"] == "entry"
    assert payload[2]["label"] == "step_steer"
    assert payload[2]["command"]["steering"] == -0.3
    assert payload[3]["label"] == "recenter"
    assert payload[3]["command"]["steering"] == 0.0


def test_build_constant_steer_program_holds_one_command() -> None:
    program = build_constant_steer_program(hold_s=3.5, throttle=0.33, steer=0.12)
    payload = program.to_dict_list()
    assert payload[1]["label"] == "constant_steer"
    assert payload[1]["duration_s"] == 3.5
    assert payload[1]["command"]["accelerator"] == 0.33
    assert payload[1]["command"]["steering"] == 0.12


def test_build_sine_steer_program_discretizes_run_interval() -> None:
    program = build_sine_steer_program(run_s=0.5, sample_dt_s=0.2, amplitude=0.25, frequency_hz=1.0)
    payload = program.to_dict_list()
    sine_segments = [segment for segment in payload if segment["label"].startswith("sine_steer_")]
    assert len(sine_segments) == 3
    assert math.isclose(sum(segment["duration_s"] for segment in sine_segments), 0.5, rel_tol=0.0, abs_tol=1e-9)
    assert sine_segments[0]["command"]["steering"] == 0.0


def test_generated_program_round_trips_through_loader(tmp_path: Path) -> None:
    path = tmp_path / "generated_step.json"
    payload = {
        "segments": build_step_steer_program(throttle=0.4, steer=0.2).to_dict_list(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_command_program(path)
    assert loaded.command_at(0.0).to_dict()["accelerator"] == 0.0
    assert loaded.command_at(1.5).to_dict()["accelerator"] == 0.4


def test_build_surface_transition_s_program_has_patch_labels() -> None:
    program = build_surface_transition_s_program()
    labels = [segment["label"] for segment in program.to_dict_list()]
    assert labels == ["launch", "cross_dry", "wet_patch_left", "gravel_patch_right", "brake"]


def test_build_program_from_preset_supports_surface_transition() -> None:
    program = build_program_from_preset("surface-transition-s", cruise_throttle=0.5)
    payload = program.to_dict_list()
    assert payload[2]["command"]["accelerator"] == 0.5
