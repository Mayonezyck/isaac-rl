from pathlib import Path

from src.physx_teacher_dataset_builder import (
    build_dataset_manifest,
    build_dataset_suite,
    build_teacher_record_command,
    generate_programs,
)


def test_build_dataset_suite_sysid_basic_contains_expected_maneuvers() -> None:
    suite = build_dataset_suite("sysid-basic-fwd")
    names = [spec.name for spec in suite]
    assert names == [
        "straight_accel_brake",
        "step_steer_left",
        "step_steer_right",
        "constant_steer_left",
        "constant_steer_right",
        "sine_steer",
        "surface_transition_s",
    ]


def test_generate_programs_writes_json_files(tmp_path: Path) -> None:
    suite = build_dataset_suite("smoke")
    program_entries = generate_programs(tmp_path, suite)
    assert len(program_entries) == 1
    assert Path(program_entries[0]["path"]).exists()


def test_build_teacher_record_command_contains_required_args(tmp_path: Path) -> None:
    program_path = tmp_path / "program.json"
    rollout_dir = tmp_path / "rollout"
    cmd = build_teacher_record_command(
        record_python="/env/python",
        program_path=program_path,
        rollout_dir=rollout_dir,
        headless=True,
        dt=1.0 / 60.0,
        track_width_m=10.0,
        patch_length_m=12.0,
        spawn_height_m=1.2,
        warmup_steps=20,
        settle_steps=60,
        max_steps=25,
    )
    assert cmd[:3] == ["/env/python", "-m", "src.physx_teacher_patch_track"]
    assert "--headless" in cmd
    assert "--command-program" in cmd
    assert "--max-steps" in cmd
    assert "25" in cmd


def test_build_dataset_manifest_tracks_programs_and_rollouts(tmp_path: Path) -> None:
    program_entries = [
        {
            "name": "straight_accel_brake",
            "preset": "straight-accel-brake",
            "description": "demo",
            "params": {"throttle": 0.6},
            "path": str(tmp_path / "programs" / "straight_accel_brake.json"),
        }
    ]
    manifest = build_dataset_manifest(
        dataset_dir=tmp_path,
        dataset_name="demo_dataset",
        suite_name="smoke",
        program_entries=program_entries,
        headless=True,
        dt_s=1.0 / 60.0,
        track_width_m=10.0,
        patch_length_m=12.0,
        spawn_height_m=1.2,
        warmup_steps=20,
        settle_steps=60,
        max_steps=0,
        record_python="/env/python",
    )
    assert manifest["dataset_name"] == "demo_dataset"
    assert manifest["suite_name"] == "smoke"
    assert len(manifest["rollouts"]) == 1
    assert manifest["rollouts"][0]["status"] == "pending"
