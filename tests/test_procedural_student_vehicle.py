from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

from src.procedural_student_vehicle import (
    PHYSX_WIZARD_DEFAULT_CHASSIS_HEIGHT_M,
    PHYSX_WIZARD_DEFAULT_CHASSIS_LENGTH_M,
    PHYSX_WIZARD_DEFAULT_CHASSIS_WIDTH_M,
    PHYSX_WIZARD_DEFAULT_GROUND_CLEARANCE_M,
    PHYSX_WIZARD_DEFAULT_TRACK_WIDTH_M,
    PHYSX_WIZARD_DEFAULT_WHEELBASE_M,
    PHYSX_WIZARD_DEFAULT_WHEEL_RADIUS_M,
    PHYSX_WIZARD_DEFAULT_WHEEL_WIDTH_M,
    build_default_student_vehicle_spec,
    build_student_vehicle_urdf,
    load_student_vehicle_spec,
    nominal_root_height_m,
    suspension_rest_length_m,
    write_student_vehicle_spec,
)


def _joint_map(urdf_text: str):
    root = ET.fromstring(urdf_text)
    return {joint.attrib["name"]: joint for joint in root.findall("joint")}


def _link_names(urdf_text: str):
    root = ET.fromstring(urdf_text)
    return {link.attrib["name"] for link in root.findall("link")}


def _link_map(urdf_text: str):
    root = ET.fromstring(urdf_text)
    return {link.attrib["name"]: link for link in root.findall("link")}


def _joint_origin_xyz(joint) -> tuple[float, float, float]:
    origin = joint.find("origin")
    assert origin is not None
    return tuple(float(value) for value in origin.attrib["xyz"].split())


def test_default_student_vehicle_topology_counts():
    urdf_text = build_student_vehicle_urdf(build_default_student_vehicle_spec())
    joints = _joint_map(urdf_text)
    links = _link_names(urdf_text)

    assert len(joints) == 10
    assert len(links) == 11
    assert "base_link" in links
    assert "front_left_steer_link" in links
    assert "rear_left_steer_link" not in links


def test_front_and_rear_joint_layout():
    urdf_text = build_student_vehicle_urdf(build_default_student_vehicle_spec())
    joints = _joint_map(urdf_text)

    assert joints["front_left_suspension_joint"].attrib["type"] == "prismatic"
    assert joints["front_left_steer_joint"].attrib["type"] == "revolute"
    assert joints["front_left_wheel_joint"].attrib["type"] == "continuous"
    assert joints["rear_left_wheel_joint"].attrib["type"] == "continuous"
    assert "rear_left_steer_joint" not in joints

    assert joints["front_left_wheel_joint"].find("parent").attrib["link"] == "front_left_steer_link"
    assert joints["rear_left_wheel_joint"].find("parent").attrib["link"] == "rear_left_suspension_link"


def test_rest_geometry_is_positive():
    spec = build_default_student_vehicle_spec()
    assert suspension_rest_length_m(spec) > 0.0
    assert nominal_root_height_m(spec) > spec.wheel_radius_m


def test_default_student_vehicle_matches_physx_wizard_geometry():
    spec = build_default_student_vehicle_spec()
    assert spec.chassis_length_m == PHYSX_WIZARD_DEFAULT_CHASSIS_LENGTH_M
    assert spec.chassis_width_m == PHYSX_WIZARD_DEFAULT_CHASSIS_WIDTH_M
    assert spec.chassis_height_m == PHYSX_WIZARD_DEFAULT_CHASSIS_HEIGHT_M
    assert spec.wheelbase_m == PHYSX_WIZARD_DEFAULT_WHEELBASE_M
    assert spec.track_width_m == PHYSX_WIZARD_DEFAULT_TRACK_WIDTH_M
    assert spec.wheel_radius_m == PHYSX_WIZARD_DEFAULT_WHEEL_RADIUS_M
    assert spec.wheel_width_m == PHYSX_WIZARD_DEFAULT_WHEEL_WIDTH_M
    assert spec.ground_clearance_m == PHYSX_WIZARD_DEFAULT_GROUND_CLEARANCE_M


def test_default_student_vehicle_uses_wizard_like_visible_links_only():
    urdf_text = build_student_vehicle_urdf(build_default_student_vehicle_spec())
    links = _link_map(urdf_text)

    base_visual_material = links["base_link"].find("./visual/material")
    assert base_visual_material is not None
    assert base_visual_material.attrib["name"] == "wizard_chassis_blue"

    assert links["front_left_suspension_link"].find("./visual/material").attrib["name"] == "hidden_helper"
    assert links["front_left_suspension_link"].find("collision") is None
    assert links["front_left_steer_link"].find("./visual/material").attrib["name"] == "hidden_helper"
    assert links["front_left_steer_link"].find("collision") is None
    assert links["front_left_wheel_link"].find("./visual/material").attrib["name"] == "wheel_black"


def test_front_and_rear_wheel_centers_share_nominal_height():
    urdf_text = build_student_vehicle_urdf(build_default_student_vehicle_spec())
    joints = _joint_map(urdf_text)

    front_height = (
        _joint_origin_xyz(joints["front_left_suspension_joint"])[2]
        + _joint_origin_xyz(joints["front_left_steer_joint"])[2]
        + _joint_origin_xyz(joints["front_left_wheel_joint"])[2]
    )
    rear_height = (
        _joint_origin_xyz(joints["rear_left_suspension_joint"])[2]
        + _joint_origin_xyz(joints["rear_left_wheel_joint"])[2]
    )

    assert front_height == rear_height


def test_spec_round_trip_with_override(tmp_path: Path):
    override_path = tmp_path / "student_vehicle_override.json"
    override_path.write_text(json.dumps({"wheelbase_m": 1.42, "name": "custom_student"}) + "\n", encoding="utf-8")

    spec = load_student_vehicle_spec(override_path)
    assert spec.wheelbase_m == 1.42
    assert spec.name == "custom_student"

    saved_path = write_student_vehicle_spec(tmp_path / "resolved.json", spec)
    payload = json.loads(saved_path.read_text(encoding="utf-8"))
    assert payload["wheelbase_m"] == 1.42
    assert payload["name"] == "custom_student"
