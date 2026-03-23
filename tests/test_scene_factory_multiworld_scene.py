from __future__ import annotations

from src.scene_factory_multiworld_scene import extract_vehicle_spawns_from_scene_cfg


def test_extract_vehicle_spawns_from_scene_cfg_recenters_and_filters():
    scene_cfg = {
        "road": {
            "polylines": [
                {
                    "type": 1,
                    "xyz": [
                        [100.0, 100.0, 0.0],
                        [110.0, 100.0, 0.0],
                    ],
                }
            ]
        },
        "agents": {
            "items": [
                {
                    "agent_id": 10,
                    "start": {"x": 100.0, "y": 100.0, "z": 0.0, "yaw": 0.25},
                    "end": {"x": 108.0, "y": 100.0, "z": 0.0},
                },
                {
                    "agent_id": 11,
                    "start": {"x": 400.0, "y": 400.0, "z": 0.0, "yaw": 0.0},
                    "end": {"x": 410.0, "y": 400.0, "z": 0.0},
                },
            ]
        },
    }

    spawns = extract_vehicle_spawns_from_scene_cfg(
        scene_cfg,
        bounds_size_m=200.0,
        origin_mode="center",
        max_controllable=8,
        require_goal_in_bounds=True,
        skip_if_start_in_goal=True,
        goal_radius_m=3.0,
        start_goal_thresh_m=3.0,
    )

    assert len(spawns) == 1
    assert spawns[0].agent_id == 10
    assert abs(spawns[0].start_local_xyz[0] + 5.0) < 1.0e-6
    assert abs(spawns[0].start_local_xyz[1] - 0.0) < 1.0e-6
    assert abs(spawns[0].goal_local_xyz[0] - 3.0) < 1.0e-6


def test_extract_vehicle_spawns_from_scene_cfg_skips_start_in_goal():
    scene_cfg = {
        "road": {"polylines": [{"type": 1, "xyz": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]}]},
        "agents": {
            "items": [
                {
                    "agent_id": 5,
                    "start": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                    "end": {"x": 1.0, "y": 0.0, "z": 0.0},
                }
            ]
        },
    }

    spawns = extract_vehicle_spawns_from_scene_cfg(
        scene_cfg,
        bounds_size_m=200.0,
        origin_mode="zero",
        max_controllable=8,
        require_goal_in_bounds=True,
        skip_if_start_in_goal=True,
        goal_radius_m=3.0,
        start_goal_thresh_m=3.0,
    )

    assert spawns == []
