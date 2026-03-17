import json
from pathlib import Path
import tempfile

from src.physx_teacher_patch_track import (
    BRAKE_MAX,
    BRAKE_MIN,
    STEER_MAX,
    STEER_MIN,
    CommandProgram,
    CommandSegment,
    SurfacePatch,
    VehicleCommand,
    build_default_surface_patches,
    command_program_from_payload,
    load_command_program,
    select_surface_patch,
)


def test_vehicle_command_clamps_to_contract() -> None:
    command = VehicleCommand(accelerator=2.0, steering=-4.0, brake=-1.0).clamped()
    assert command.accelerator == 1.0
    assert command.steering == STEER_MIN
    assert command.brake == BRAKE_MIN

    command = VehicleCommand(accelerator=-0.1, steering=3.0, brake=5.0).clamped()
    assert command.accelerator == 0.0
    assert command.steering == STEER_MAX
    assert command.brake == BRAKE_MAX


def test_command_program_holds_last_segment() -> None:
    program = CommandProgram(
        [
            CommandSegment(1.0, VehicleCommand(0.1, 0.0, 0.0), "a"),
            CommandSegment(2.0, VehicleCommand(0.4, 0.2, 0.0), "b"),
        ]
    )

    assert program.command_at(0.0).to_dict() == VehicleCommand(0.1, 0.0, 0.0).to_dict()
    assert program.command_at(0.9).to_dict() == VehicleCommand(0.1, 0.0, 0.0).to_dict()
    assert program.command_at(1.0).to_dict() == VehicleCommand(0.4, 0.2, 0.0).to_dict()
    assert program.command_at(10.0).to_dict() == VehicleCommand(0.4, 0.2, 0.0).to_dict()


def test_surface_patch_lookup_by_xy() -> None:
    patches = build_default_surface_patches(patch_length_m=12.0, track_width_m=10.0)
    assert select_surface_patch(patches, -12.0, 0.0).name == "dry_asphalt"
    assert select_surface_patch(patches, 0.0, 0.0).name == "wet_asphalt"
    assert select_surface_patch(patches, 12.0, 0.0).name == "gravel"
    assert select_surface_patch(patches, 100.0, 0.0) is None


def test_surface_patch_bounds_are_closed() -> None:
    patch = SurfacePatch(
        name="demo",
        x_center_m=1.0,
        y_center_m=2.0,
        length_m=4.0,
        width_m=6.0,
        static_friction=0.8,
        dynamic_friction=0.7,
        tire_friction=0.75,
        color_srgb=(0.1, 0.2, 0.3),
    )

    assert patch.contains_xy(-1.0, -1.0) is True
    assert patch.contains_xy(3.0, 5.0) is True
    assert patch.contains_xy(3.1, 5.0) is False


def test_command_program_loads_from_payload_and_path() -> None:
    payload = {
        "segments": [
            {
                "label": "a",
                "duration_s": 1.0,
                "command": {"accelerator": 0.1, "steering": 0.0, "brake": 0.0},
            },
            {
                "label": "b",
                "duration_s": 2.0,
                "command": {"accelerator": 0.3, "steering": 0.2, "brake": 0.1},
            },
        ]
    }
    program = command_program_from_payload(payload)
    assert program.command_at(0.0).to_dict() == VehicleCommand(0.1, 0.0, 0.0).to_dict()
    assert program.command_at(1.5).to_dict() == VehicleCommand(0.3, 0.2, 0.1).to_dict()

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "program.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = load_command_program(path)
        assert loaded.command_at(1.5).to_dict() == VehicleCommand(0.3, 0.2, 0.1).to_dict()
