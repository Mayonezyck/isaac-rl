from __future__ import annotations

from pathlib import Path

from src.procedural_student_vehicle import build_default_student_vehicle_spec
from src.procedural_student_vehicle_import import (
    _default_vehicle_usd_path,
    _vehicle_asset_root_path,
    _vehicle_collision_material_bind_targets,
    _vehicle_physics_material_paths,
)


def test_vehicle_asset_root_path_drops_world_prefix():
    assert _vehicle_asset_root_path("/World/student_fwd_vehicle") == "/student_fwd_vehicle"


def test_default_vehicle_usd_path_uses_vehicle_name(tmp_path: Path):
    spec = build_default_student_vehicle_spec()
    expected = tmp_path / f"{spec.name}.usd"
    assert _default_vehicle_usd_path(tmp_path, spec) == expected.resolve()


def test_vehicle_physics_material_paths_are_local_to_asset_root():
    paths = _vehicle_physics_material_paths("/student_fwd_vehicle")
    assert paths == {
        "wheel": "/student_fwd_vehicle/PhysicsMaterials/wheel_contact_material",
        "chassis": "/student_fwd_vehicle/PhysicsMaterials/chassis_contact_material",
    }


def test_vehicle_collision_material_bind_targets_cover_wheels_and_base_link():
    targets = _vehicle_collision_material_bind_targets("/student_fwd_vehicle")
    assert targets == {
        "wheel": [
            "/student_fwd_vehicle/front_left_wheel_link",
            "/student_fwd_vehicle/front_right_wheel_link",
            "/student_fwd_vehicle/rear_left_wheel_link",
            "/student_fwd_vehicle/rear_right_wheel_link",
        ],
        "chassis": [
            "/student_fwd_vehicle/base_link",
        ],
    }
