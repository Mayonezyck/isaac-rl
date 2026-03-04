from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from gpudrive_chocolate.baselines.ppo.eval_weather_pair import (
    ActionTraceSample,
    build_pair_curriculum_config,
    build_weather_sweep_curriculum_config,
    default_weather_variants,
    choose_tracked_agent_ids,
    collect_world_agent_indices,
    ensure_checkpoint_exists,
    parse_agent_ids_arg,
    parse_args,
    merge_friction,
    parse_mapping_arg,
    render_action_trace_svg,
)


def test_parse_mapping_arg_inline_yaml() -> None:
    parsed = parse_mapping_arg("{water_film_mm: 0.05, precip_type: rain}")
    assert parsed["water_film_mm"] == 0.05
    assert parsed["precip_type"] == "rain"


def test_merge_friction_override() -> None:
    merged = merge_friction(
        {"road_type": "AC", "water_film_mm": 0.025},
        {"water_film_mm": 0.10, "precip_intensity_mmph": 2.0},
    )
    assert merged == {
        "road_type": "AC",
        "water_film_mm": 0.10,
        "precip_intensity_mmph": 2.0,
    }


def test_build_weather_sweep_curriculum_config_replaces_world_assignments() -> None:
    cfg = {
        "app": {"headless": False},
        "world": {
            "world_count": 16,
            "grid_cols": 4,
            "rows": 4,
            "assignments": [{"scene_json": "old.json"}],
        },
        "env": {"render": True},
        "physics": {"report_gpu_dynamics_once": True},
        "light": {"enable": False, "intensity": 3000.0},
    }
    out = build_weather_sweep_curriculum_config(
        cfg,
        scene_json="scene_000002.json",
        weather_variants=default_weather_variants({"road_type": "AC"})[:3],
        headless=False,
        render=True,
    )
    assert out["world"]["world_count"] == 3
    assert out["world"]["grid_cols"] == 3
    assert out["world"]["rows"] == 1
    assert out["world"]["assignments"][0]["scene_json"] == "scene_000002.json"
    assert out["world"]["assignments"][1]["friction"]["water_film_mm"] == 0.025
    assert out["app"]["headless"] is False
    assert out["env"]["render"] is True
    assert out["env"]["auto_reset_done"] is False
    assert out["env"]["auto_reset_timeout"] is False
    assert out["physics"]["report_gpu_dynamics_once"] is False
    assert out["light"]["enable"] is False


def test_build_pair_curriculum_config_keeps_pair_compatibility() -> None:
    out = build_pair_curriculum_config(
        {
            "app": {"headless": True},
            "world": {"world_count": 16, "grid_cols": 4, "rows": 4, "assignments": []},
            "env": {"render": False},
            "physics": {},
        },
        scene_json="scene_000002.json",
        friction_a={"road_type": "AC", "water_film_mm": 0.025},
        friction_b={"road_type": "AC", "water_film_mm": 0.10},
    )
    assert out["world"]["world_count"] == 2
    assert out["world"]["assignments"][1]["friction"]["water_film_mm"] == 0.10


def test_collect_world_agent_indices_and_choose_tracked_agents() -> None:
    keys = [
        SimpleNamespace(world_idx=0, agent_id=11),
        SimpleNamespace(world_idx=0, agent_id=20),
        SimpleNamespace(world_idx=1, agent_id=11),
        SimpleNamespace(world_idx=1, agent_id=30),
    ]
    mapping = collect_world_agent_indices(keys)
    assert mapping == {0: {11: 0, 20: 1}, 1: {11: 2, 30: 3}}
    assert choose_tracked_agent_ids(mapping, None) == [11]
    assert choose_tracked_agent_ids(mapping, 11) == [11]


def test_parse_agent_ids_arg_deduplicates_and_preserves_order() -> None:
    assert parse_agent_ids_arg("11, 20,11, 30") == [11, 20, 30]


def test_default_weather_variants_include_extreme_case() -> None:
    variants = default_weather_variants({"road_type": "AC"})
    assert [variant.label for variant in variants] == [
        "dry-clear",
        "light-rain",
        "moderate-rain",
        "heavy-rain",
        "extreme-rain",
    ]
    assert variants[-1].friction["water_film_mm"] == 0.50
    assert variants[-1].friction["precip_intensity_mmph"] == 12.0


def test_render_action_trace_svg_contains_labels_and_markers() -> None:
    traces = {
        7: {
            0: [
                ActionTraceSample(0, 0, "dry", 7, 0.1, 0.2, 0.0, False),
                ActionTraceSample(1, 0, "dry", 7, 0.2, 0.1, 0.0, True),
            ],
            1: [
                ActionTraceSample(0, 1, "wet", 7, -0.1, -0.2, 0.0, False),
                ActionTraceSample(1, 1, "wet", 7, -0.2, -0.1, 0.0, False),
            ],
        },
        9: {
            0: [ActionTraceSample(0, 0, "dry", 9, 0.0, 0.4, 0.0, False)],
            1: [ActionTraceSample(0, 1, "wet", 9, 0.1, 0.3, 0.0, False)],
        },
    }
    svg = render_action_trace_svg(
        traces,
        world_labels={0: "dry", 1: "wet"},
        title="example",
    )
    assert "Deterministic" not in svg
    assert "example" in svg
    assert "dry" in svg
    assert "wet" in svg
    assert "agent_id=7" in svg
    assert "agent_id=9" in svg
    assert "<circle" in svg


def test_ensure_checkpoint_exists_resolves_latest_matching_zip(tmp_path: Path) -> None:
    save_dir = tmp_path / "ckpts"
    save_dir.mkdir()
    older = save_dir / "ppo_stage1_navigation_foundation_200000_steps.zip"
    newer = save_dir / "ppo_stage1_navigation_foundation_3080000_steps.zip"
    older.write_text("old", encoding="utf-8")
    newer.write_text("new", encoding="utf-8")

    exp_config = SimpleNamespace(
        save_dir=str(save_dir),
        save_prefix="ppo_stage1_navigation_foundation",
    )
    resolved = ensure_checkpoint_exists(
        str(save_dir / "ppo_stage1_navigation_foundation_999999_steps.zip"),
        exp_config,
    )
    assert resolved == newer.resolve()


def test_parse_args_stop_on_tracked_done_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_weather_pair.py",
            "--config",
            "cfg.yaml",
            "--checkpoint",
            "model.zip",
        ],
    )
    args = parse_args()
    assert args.stop_on_tracked_done is False
    assert args.agent_ids is None
