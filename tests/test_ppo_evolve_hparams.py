from __future__ import annotations

from pathlib import Path

from gpudrive_chocolate.baselines.ppo.evolve_hparams import (
    CandidatePlan,
    CandidateResult,
    ParamSpec,
    ResourceSlot,
    _apply_genome_to_configs,
    _build_param_specs,
    _extract_base_genome,
    _make_generation_zero,
    _make_next_generation,
    _score_candidate,
    _set_nested,
    _write_study_report,
    _write_candidate_configs,
)


def test_set_nested_updates_list_index() -> None:
    payload = {"control": {"thr_clip": [0.0, 0.4]}}
    _set_nested(payload, "control.thr_clip.1", 0.6)
    assert payload["control"]["thr_clip"][1] == 0.6


def test_generation_zero_includes_base_genome() -> None:
    specs = _build_param_specs(
        [
            {
                "name": "lr",
                "target": "ppo",
                "path": "lr",
                "type": "choice",
                "values": [0.0001, 0.0003],
            }
        ]
    )
    base = {"lr": 0.0003}
    plans = _make_generation_zero(base, specs, population_size=2, rng=__import__("random").Random(1))
    assert plans[0].genome == base
    assert plans[0].origin == "base"
    assert len(plans) == 2


def test_apply_genome_to_configs_splits_targets() -> None:
    specs = _build_param_specs(
        [
            {
                "name": "lr",
                "target": "ppo",
                "path": "lr",
                "type": "choice",
                "values": [0.0001, 0.0003],
            },
            {
                "name": "thr_clip_max",
                "target": "curriculum",
                "path": "control.thr_clip.1",
                "type": "choice",
                "values": [0.4, 0.6],
            },
        ]
    )
    ppo_cfg = {"lr": 0.0003}
    curriculum_cfg = {"control": {"thr_clip": [0.0, 0.4]}}
    genome = {"lr": 0.0001, "thr_clip_max": 0.6}
    new_ppo, new_curriculum = _apply_genome_to_configs(genome, specs, ppo_cfg, curriculum_cfg)
    assert new_ppo["lr"] == 0.0001
    assert new_curriculum["control"]["thr_clip"][1] == 0.6


def test_score_candidate_uses_weighted_metrics() -> None:
    metrics = {
        "rollout/success_rate": 0.10,
        "rollout/mean_dist_to_goal_m": 20.0,
    }
    scoring_cfg = {
        "weights": {
            "rollout/success_rate": 100.0,
            "rollout/mean_dist_to_goal_m": -0.5,
        }
    }
    assert _score_candidate(metrics, scoring_cfg) == 0.0


def test_write_candidate_configs_overrides_device_and_timesteps(tmp_path: Path) -> None:
    specs = _build_param_specs(
        [
            {
                "name": "lr",
                "target": "ppo",
                "path": "lr",
                "type": "choice",
                "values": [0.0001],
            }
        ]
    )
    base_ppo = {
        "choco_config_path": "dummy.yaml",
        "device": "cuda:1",
        "save_freq": 100,
        "save_dir": "runs/checkpoints",
        "save_prefix": "baseline",
        "lr": 0.0003,
    }
    base_curriculum = {
        "app": {"active_gpu": 1, "physics_gpu": 1, "headless": False},
        "env": {"render": True},
    }
    slot = ResourceSlot(label="slot0", device="cpu", active_gpu=None, physics_gpu=None)
    ppo_path, curriculum_path, _, candidate_id = _write_candidate_configs(
        study_root=tmp_path,
        generation=0,
        index=0,
        plan=CandidatePlan(genome={"lr": 0.0001}, origin="random", parent_ids=[]),
        specs=specs,
        base_ppo=base_ppo,
        base_curriculum=base_curriculum,
        slot=slot,
        trial_timesteps=1234,
    )
    assert candidate_id == "g000_c000"
    assert ppo_path.exists()
    assert curriculum_path.exists()
    ppo_data = __import__("yaml").safe_load(ppo_path.read_text(encoding="utf-8"))
    curriculum_data = __import__("yaml").safe_load(curriculum_path.read_text(encoding="utf-8"))
    assert ppo_data["device"] == "cpu"
    assert ppo_data["total_timesteps"] == 1234
    assert ppo_data["save_freq"] == 1235
    assert curriculum_data["app"]["headless"] is True
    assert curriculum_data["env"]["render"] is False
    meta_data = __import__("json").loads((tmp_path / "generation_000" / "candidate_000" / "candidate.json").read_text(encoding="utf-8"))
    assert meta_data["origin"] == "random"
    assert meta_data["parent_ids"] == []


def test_make_next_generation_keeps_elites() -> None:
    specs = [ParamSpec(name="lr", target="ppo", path="lr", kind="choice", values=[0.1, 0.2, 0.3])]
    results = [
        __import__("types").SimpleNamespace(
            candidate_id="g000_c000",
            genome={"lr": 0.1},
            status="completed",
            score=3.0,
        ),
        __import__("types").SimpleNamespace(
            candidate_id="g000_c001",
            genome={"lr": 0.2},
            status="completed",
            score=2.0,
        ),
    ]
    plans = _make_next_generation(
        prior_results=results,  # type: ignore[arg-type]
        specs=specs,
        population_size=2,
        elite_count=1,
        mutation_rate=0.2,
        crossover_rate=0.7,
        random_candidates_per_generation=0,
        rng=__import__("random").Random(2),
    )
    assert plans[0].genome == {"lr": 0.1}
    assert plans[0].origin == "elite"
    assert plans[0].parent_ids == ["g000_c000"]


def test_extract_base_genome_reads_both_files() -> None:
    specs = _build_param_specs(
        [
            {
                "name": "goal_weight",
                "target": "ppo",
                "path": "goal_achieved_weight",
                "type": "choice",
                "values": [1.0],
            },
            {
                "name": "max_steps",
                "target": "curriculum",
                "path": "env.max_steps",
                "type": "choice",
                "values": [400],
            },
        ]
    )
    genome = _extract_base_genome(
        specs,
        {"goal_achieved_weight": 1.0},
        {"env": {"max_steps": 400}},
    )
    assert genome == {"goal_weight": 1.0, "max_steps": 400}


def test_write_study_report_emits_family_tree(tmp_path: Path) -> None:
    results = [
        CandidateResult(
            candidate_id="g000_c000",
            generation=0,
            index=0,
            status="completed",
            score=10.0,
            genome={"lr": 0.1},
            origin="base",
            parent_ids=[],
            metrics={
                "rollout/success_rate": 0.5,
                "rollout/goal_rate": 0.01,
                "rollout/mean_dist_to_goal_m": 20.0,
                "rollout/road_contact_done_rate": 0.1,
                "rollout/vehicle_contact_done_rate": 0.0,
            },
            device="cpu",
            active_gpu=None,
            physics_gpu=None,
            run_id="g000_c000",
            returncode=0,
            log_dir="log0",
            log_file="trial0.log",
            ppo_config_path="ppo0.yaml",
            curriculum_config_path="curr0.yaml",
            started_at=0.0,
            finished_at=1.0,
        ),
        CandidateResult(
            candidate_id="g001_c000",
            generation=1,
            index=0,
            status="completed",
            score=12.0,
            genome={"lr": 0.2},
            origin="elite",
            parent_ids=["g000_c000"],
            metrics={
                "rollout/success_rate": 0.6,
                "rollout/goal_rate": 0.02,
                "rollout/mean_dist_to_goal_m": 18.0,
                "rollout/road_contact_done_rate": 0.05,
                "rollout/vehicle_contact_done_rate": 0.0,
            },
            device="cpu",
            active_gpu=None,
            physics_gpu=None,
            run_id="g001_c000",
            returncode=0,
            log_dir="log1",
            log_file="trial1.log",
            ppo_config_path="ppo1.yaml",
            curriculum_config_path="curr1.yaml",
            started_at=1.0,
            finished_at=2.0,
        ),
    ]
    _write_study_report(tmp_path, "study", results)
    assert (tmp_path / "family_tree.svg").exists()
    assert (tmp_path / "family_tree.json").exists()
