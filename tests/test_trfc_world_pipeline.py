from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from src.trfc.world_pipeline import prepare_stage_world_specs


def _touch_scene_files(root: Path, count: int) -> None:
    for idx in range(count):
        (root / f"scene_{idx:06d}.json").write_text("{}", encoding="utf-8")


def test_prepare_stage_world_specs_from_assignments_estimates_per_world() -> None:
    with TemporaryDirectory() as tmpdir:
        scene_dir = Path(tmpdir)
        _touch_scene_files(scene_dir, 4)

        cfg = {
            "io": {"scene_json_dir": str(scene_dir)},
            "world": {
                "world_count": 4,
                "assignments": [
                    {
                        "scene_json": "scene_000000.json",
                        "friction": {
                            "road_type": "AC",
                            "precip_type": "rain",
                            "precip_intensity_mmph": 1.0,
                            "water_film_mm": 0.05,
                        },
                    },
                    {
                        "scene_json": "scene_000001.json",
                        "friction": {
                            "road_type": "AC",
                            "precip_type": "rain",
                            "precip_intensity_mmph": 6.0,
                            "water_film_mm": 0.30,
                        },
                    },
                    {
                        "scene_json": "scene_000002.json",
                        "friction": {
                            "road_type": "SMA",
                            "precip_type": "rain",
                            "precip_intensity_mmph": 4.0,
                            "water_film_mm": 0.20,
                        },
                    },
                    {
                        "scene_json": "scene_000003.json",
                        "friction": {
                            "road_type": "OGFC",
                            "precip_type": "rain",
                            "precip_intensity_mmph": 4.0,
                            "water_film_mm": 0.20,
                        },
                    },
                ],
            },
            "ground": {
                "friction_pipeline": {
                    "enable": True,
                    "defaults": {
                        "v_ref_mps": 13.89,
                        "s_ref_static": 0.15,
                        "s_ref_dynamic": 0.8,
                    },
                }
            },
        }

        specs = prepare_stage_world_specs(cfg)

        assert [spec.scene_json_name for spec in specs] == [
            "scene_000000.json",
            "scene_000001.json",
            "scene_000002.json",
            "scene_000003.json",
        ]
        assert all(spec.friction_estimate is not None for spec in specs)
        assert specs[1].friction_estimate.mu_static < specs[0].friction_estimate.mu_static
        assert specs[3].friction_estimate.mu_static > specs[2].friction_estimate.mu_static


def test_prepare_stage_world_specs_uses_explicit_scene_json_list() -> None:
    with TemporaryDirectory() as tmpdir:
        scene_dir = Path(tmpdir)
        _touch_scene_files(scene_dir, 3)

        cfg = {
            "io": {
                "scene_json_dir": str(scene_dir),
                "scene_jsons": [
                    "scene_000002.json",
                    "scene_000000.json",
                ],
            },
            "world": {
                "world_count": 2,
            },
            "ground": {},
        }

        specs = prepare_stage_world_specs(cfg)

        assert [spec.scene_json_name for spec in specs] == [
            "scene_000002.json",
            "scene_000000.json",
        ]
        assert all(spec.friction_estimate is None for spec in specs)


def test_prepare_stage_world_specs_requires_explicit_water_film_for_friction() -> None:
    with TemporaryDirectory() as tmpdir:
        scene_dir = Path(tmpdir)
        _touch_scene_files(scene_dir, 1)

        cfg = {
            "io": {"scene_json_dir": str(scene_dir)},
            "world": {
                "world_count": 1,
                "assignments": [
                    {
                        "scene_json": "scene_000000.json",
                        "friction": {
                            "road_type": "AC",
                            "precip_type": "rain",
                            "precip_intensity_mmph": 4.0,
                        },
                    }
                ],
            },
            "ground": {
                "friction_pipeline": {
                    "enable": True,
                }
            },
        }

        try:
            prepare_stage_world_specs(cfg)
        except ValueError as exc:
            assert "water_film_mm must be provided" in str(exc)
        else:
            raise AssertionError("prepare_stage_world_specs should require water_film_mm")
