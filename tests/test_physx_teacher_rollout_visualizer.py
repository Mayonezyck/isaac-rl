import json
from pathlib import Path

from src.physx_teacher_rollout_visualizer import (
    build_replay_html,
    compute_world_bounds,
    estimate_vehicle_footprint,
    load_rollout_dir,
)


def _write_sample_rollout(tmp_path: Path) -> Path:
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    meta = {
        "dt_s": 0.1,
        "surface_patches": [
            {
                "name": "dry_asphalt",
                "x_center_m": 0.0,
                "y_center_m": 0.0,
                "length_m": 10.0,
                "width_m": 4.0,
                "color_srgb": [0.3, 0.3, 0.3],
            }
        ],
    }
    frames = [
        {
            "step": 0,
            "sim_time_s": 0.1,
            "command": {"accelerator": 0.2, "steering": 0.0, "brake": 0.0},
            "vehicle": {
                "position_m": [0.0, 0.0, 0.9],
                "yaw_rad": 0.0,
                "linear_velocity_mps": [1.0, 0.0, 0.0],
            },
            "drive_state": {"accelerator": 0.2, "brake0": 0.0, "steer": 0.0, "target_gear": 0},
            "wheels": [
                {
                    "label": "front_left",
                    "surface_name": "dry_asphalt",
                    "ground_hit_position_m": [1.0, 0.8, 0.0],
                    "tire_longitudinal_slip": 0.01,
                    "tire_lateral_slip": 0.02,
                },
                {
                    "label": "front_right",
                    "surface_name": "dry_asphalt",
                    "ground_hit_position_m": [1.0, -0.8, 0.0],
                    "tire_longitudinal_slip": 0.01,
                    "tire_lateral_slip": 0.02,
                },
                {
                    "label": "rear_left",
                    "surface_name": "dry_asphalt",
                    "ground_hit_position_m": [-1.0, 0.8, 0.0],
                    "tire_longitudinal_slip": 0.01,
                    "tire_lateral_slip": 0.02,
                },
                {
                    "label": "rear_right",
                    "surface_name": "dry_asphalt",
                    "ground_hit_position_m": [-1.0, -0.8, 0.0],
                    "tire_longitudinal_slip": 0.01,
                    "tire_lateral_slip": 0.02,
                },
            ],
        }
    ]
    (rollout_dir / "rollout_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (rollout_dir / "rollout_frames.jsonl").write_text(
        "\n".join(json.dumps(frame) for frame in frames) + "\n",
        encoding="utf-8",
    )
    return rollout_dir


def test_load_rollout_dir_reads_meta_and_frames(tmp_path: Path) -> None:
    rollout_dir = _write_sample_rollout(tmp_path)
    meta, frames = load_rollout_dir(rollout_dir)
    assert meta["dt_s"] == 0.1
    assert len(frames) == 1
    assert frames[0]["vehicle"]["position_m"][0] == 0.0


def test_compute_world_bounds_covers_patches_and_hits(tmp_path: Path) -> None:
    rollout_dir = _write_sample_rollout(tmp_path)
    meta, frames = load_rollout_dir(rollout_dir)
    bounds = compute_world_bounds(meta, frames, margin_m=0.5)
    assert bounds["x_min"] <= -5.5
    assert bounds["x_max"] >= 5.5
    assert bounds["y_min"] <= -2.5
    assert bounds["y_max"] >= 2.5


def test_estimate_vehicle_footprint_uses_wheel_hit_span(tmp_path: Path) -> None:
    rollout_dir = _write_sample_rollout(tmp_path)
    _, frames = load_rollout_dir(rollout_dir)
    footprint = estimate_vehicle_footprint(frames)
    assert footprint["length_m"] >= 2.6
    assert footprint["width_m"] >= 2.0


def test_build_replay_html_embeds_rollout_payload(tmp_path: Path) -> None:
    rollout_dir = _write_sample_rollout(tmp_path)
    meta, frames = load_rollout_dir(rollout_dir)
    html_text = build_replay_html(
        meta,
        frames,
        title="Replay Demo",
        source_rollout_dir=str(rollout_dir),
        frame_stride=1,
    )
    assert "Replay Demo" in html_text
    assert "rollout_frames.jsonl" not in html_text
    assert "front_left" in html_text
    assert "dry_asphalt" in html_text
