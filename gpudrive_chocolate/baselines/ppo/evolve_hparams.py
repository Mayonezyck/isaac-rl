from __future__ import annotations

import argparse
import copy
import html
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded


def _dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _require_tensorboard() -> None:
    try:
        from tensorboard.backend.event_processing import event_accumulator  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "tensorboard is required to score search candidates. "
            "Run the search from an environment that has tensorboard installed. "
            f"python={sys.executable} error={exc}"
        ) from exc


def _path_tokens(path: str) -> list[str | int]:
    tokens: list[str | int] = []
    for token in path.split("."):
        if token.isdigit():
            tokens.append(int(token))
        else:
            tokens.append(token)
    return tokens


def _get_nested(data: Any, path: str) -> Any:
    cur = data
    for token in _path_tokens(path):
        cur = cur[token]
    return cur


def _set_nested(data: Any, path: str, value: Any) -> None:
    tokens = _path_tokens(path)
    cur = data
    for token in tokens[:-1]:
        cur = cur[token]
    cur[tokens[-1]] = value


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return value


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _slugify(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "study"


def _compact_json(data: dict[str, Any], max_len: int = 220) -> str:
    text = json.dumps(data, sort_keys=True)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _fmt_metric(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


@dataclass(frozen=True)
class ParamSpec:
    name: str
    target: str
    path: str
    kind: str
    values: list[Any] | None = None
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    scale: str = "linear"

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ParamSpec":
        kind = str(data["type"])
        return ParamSpec(
            name=str(data["name"]),
            target=str(data["target"]),
            path=str(data["path"]),
            kind=kind,
            values=data.get("values"),
            min_value=data.get("min"),
            max_value=data.get("max"),
            step=data.get("step"),
            scale=str(data.get("scale", "linear")),
        )


@dataclass(frozen=True)
class ResourceSlot:
    label: str
    device: str
    active_gpu: int | None
    physics_gpu: int | None


@dataclass
class CandidatePlan:
    genome: dict[str, Any]
    origin: str
    parent_ids: list[str]


def _expand_slots(slots: list[ResourceSlot], target_count: int) -> list[ResourceSlot]:
    if not slots:
        return []
    if target_count <= len(slots):
        return list(slots[:target_count])
    expanded: list[ResourceSlot] = []
    for idx in range(target_count):
        base = slots[idx % len(slots)]
        replica_idx = idx // len(slots)
        label = base.label if replica_idx == 0 else f"{base.label}_x{replica_idx + 1}"
        expanded.append(
            ResourceSlot(
                label=label,
                device=base.device,
                active_gpu=base.active_gpu,
                physics_gpu=base.physics_gpu,
            )
        )
    return expanded


@dataclass
class CandidateResult:
    candidate_id: str
    generation: int
    index: int
    status: str
    score: float
    genome: dict[str, Any]
    origin: str
    parent_ids: list[str]
    metrics: dict[str, float]
    device: str
    active_gpu: int | None
    physics_gpu: int | None
    run_id: str
    returncode: int | None
    log_dir: str
    log_file: str
    ppo_config_path: str
    curriculum_config_path: str
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def duration_sec(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return max(0.0, self.finished_at - self.started_at)


def _build_param_specs(space_cfg: list[dict[str, Any]]) -> list[ParamSpec]:
    return [ParamSpec.from_dict(item) for item in space_cfg]


def _sample_numeric(spec: ParamSpec, rng: random.Random) -> float:
    if spec.min_value is None or spec.max_value is None:
        raise ValueError(f"Numeric spec {spec.name} is missing min/max")
    lo = float(spec.min_value)
    hi = float(spec.max_value)
    if spec.scale == "log":
        if lo <= 0 or hi <= 0:
            raise ValueError(f"Log-scale spec {spec.name} requires positive min/max")
        raw = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    else:
        raw = rng.uniform(lo, hi)
    if spec.step:
        step = float(spec.step)
        raw = round(raw / step) * step
    return min(hi, max(lo, raw))


def _cast_spec_value(spec: ParamSpec, value: Any) -> Any:
    if spec.kind == "int":
        return int(round(float(value)))
    if spec.kind == "float":
        return float(value)
    if spec.kind == "bool":
        return bool(value)
    return _coerce_scalar(value)


def _sample_value(spec: ParamSpec, rng: random.Random) -> Any:
    if spec.kind == "choice":
        if not spec.values:
            raise ValueError(f"Choice spec {spec.name} has no values")
        return copy.deepcopy(rng.choice(spec.values))
    if spec.kind in {"float", "int"}:
        return _cast_spec_value(spec, _sample_numeric(spec, rng))
    if spec.kind == "bool":
        return bool(rng.choice([False, True]))
    raise ValueError(f"Unsupported spec kind: {spec.kind}")


def _mutate_value(spec: ParamSpec, value: Any, rng: random.Random) -> Any:
    if spec.kind == "choice":
        if not spec.values:
            return value
        choices = [item for item in spec.values if item != value]
        if not choices:
            return copy.deepcopy(value)
        return copy.deepcopy(rng.choice(choices))
    if spec.kind == "bool":
        return not bool(value)
    if spec.kind in {"float", "int"}:
        if spec.min_value is None or spec.max_value is None:
            return value
        lo = float(spec.min_value)
        hi = float(spec.max_value)
        base = float(value)
        span = hi - lo
        if spec.scale == "log":
            mutated = math.exp(
                min(
                    math.log(hi),
                    max(math.log(lo), math.log(base) + rng.uniform(-0.5, 0.5)),
                )
            )
        else:
            mutated = base + rng.uniform(-0.25 * span, 0.25 * span)
        if spec.step:
            step = float(spec.step)
            mutated = round(mutated / step) * step
        mutated = min(hi, max(lo, mutated))
        return _cast_spec_value(spec, mutated)
    return copy.deepcopy(value)


def _crossover_value(spec: ParamSpec, left: Any, right: Any, rng: random.Random) -> Any:
    if spec.kind in {"choice", "bool"}:
        return copy.deepcopy(left if rng.random() < 0.5 else right)
    if spec.kind in {"float", "int"}:
        mixed = (float(left) + float(right)) / 2.0
        if spec.kind == "float":
            jitter = abs(float(left) - float(right)) * rng.uniform(-0.25, 0.25)
            mixed += jitter
        return _mutate_value(spec, mixed, rng) if rng.random() < 0.15 else _cast_spec_value(spec, mixed)
    return copy.deepcopy(left)


def _extract_base_genome(
    specs: list[ParamSpec],
    ppo_cfg: dict[str, Any],
    curriculum_cfg: dict[str, Any],
) -> dict[str, Any]:
    genome: dict[str, Any] = {}
    for spec in specs:
        root = ppo_cfg if spec.target == "ppo" else curriculum_cfg
        genome[spec.name] = copy.deepcopy(_get_nested(root, spec.path))
    return genome


def _random_genome(specs: list[ParamSpec], rng: random.Random) -> dict[str, Any]:
    return {spec.name: _sample_value(spec, rng) for spec in specs}


def _mutate_genome(
    genome: dict[str, Any],
    specs: list[ParamSpec],
    mutation_rate: float,
    rng: random.Random,
    force_mutation: bool = False,
) -> dict[str, Any]:
    mutated = copy.deepcopy(genome)
    changed = False
    for spec in specs:
        if force_mutation or rng.random() < mutation_rate:
            mutated[spec.name] = _mutate_value(spec, mutated[spec.name], rng)
            changed = True
    if force_mutation and not changed and specs:
        spec = rng.choice(specs)
        mutated[spec.name] = _mutate_value(spec, mutated[spec.name], rng)
    return mutated


def _crossover_genomes(
    left: dict[str, Any],
    right: dict[str, Any],
    specs: list[ParamSpec],
    rng: random.Random,
) -> dict[str, Any]:
    child: dict[str, Any] = {}
    for spec in specs:
        child[spec.name] = _crossover_value(spec, left[spec.name], right[spec.name], rng)
    return child


def _genome_key(genome: dict[str, Any]) -> str:
    return json.dumps(genome, sort_keys=True)


def _make_generation_zero(
    base_genome: dict[str, Any],
    specs: list[ParamSpec],
    population_size: int,
    rng: random.Random,
) -> list[CandidatePlan]:
    plans = [CandidatePlan(genome=copy.deepcopy(base_genome), origin="base", parent_ids=[])]
    seen = {_genome_key(plans[0].genome)}
    while len(plans) < population_size:
        candidate = _random_genome(specs, rng)
        key = _genome_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        plans.append(CandidatePlan(genome=candidate, origin="random", parent_ids=[]))
    return plans


def _make_next_generation(
    prior_results: list[CandidateResult],
    specs: list[ParamSpec],
    population_size: int,
    elite_count: int,
    mutation_rate: float,
    crossover_rate: float,
    random_candidates_per_generation: int,
    rng: random.Random,
) -> list[CandidatePlan]:
    ranked = [item for item in prior_results if item.status == "completed"]
    ranked.sort(key=lambda item: item.score, reverse=True)
    if not ranked:
        raise ValueError("Cannot build next generation from empty ranked results")
    plans: list[CandidatePlan] = []
    seen: set[str] = set()
    for item in ranked:
        if len(plans) >= max(1, elite_count):
            break
        key = _genome_key(item.genome)
        if key in seen:
            continue
        seen.add(key)
        plans.append(
            CandidatePlan(
                genome=copy.deepcopy(item.genome),
                origin="elite",
                parent_ids=[item.candidate_id],
            )
        )

    parent_pool = ranked[: max(2, min(len(ranked), elite_count * 2))]
    while len(plans) < population_size:
        if len(plans) >= population_size - random_candidates_per_generation:
            plan = CandidatePlan(genome=_random_genome(specs, rng), origin="random", parent_ids=[])
        else:
            left_parent = rng.choice(parent_pool)
            parent_ids = [left_parent.candidate_id]
            child = copy.deepcopy(left_parent.genome)
            origin = "mutation"
            if rng.random() < crossover_rate and len(parent_pool) >= 2:
                right_parent = rng.choice(parent_pool)
                if len(parent_pool) > 1:
                    while right_parent.candidate_id == left_parent.candidate_id:
                        right_parent = rng.choice(parent_pool)
                child = _crossover_genomes(
                    copy.deepcopy(left_parent.genome),
                    copy.deepcopy(right_parent.genome),
                    specs,
                    rng,
                )
                parent_ids = [left_parent.candidate_id, right_parent.candidate_id]
                origin = "crossover_mutation"
            child = _mutate_genome(child, specs, mutation_rate, rng, force_mutation=True)
            plan = CandidatePlan(genome=child, origin=origin, parent_ids=parent_ids)
        key = _genome_key(plan.genome)
        if key in seen:
            continue
        seen.add(key)
        plans.append(plan)
    return plans[:population_size]


def _resolve_resource_slots(study_cfg: dict[str, Any]) -> list[ResourceSlot]:
    resources = study_cfg.get("resources", {})
    slots_cfg = resources.get("slots")
    slots: list[ResourceSlot] = []
    if isinstance(slots_cfg, list) and slots_cfg:
        for idx, slot_cfg in enumerate(slots_cfg):
            if not isinstance(slot_cfg, dict):
                raise ValueError("Each resource slot must be a mapping")
            label = str(slot_cfg.get("label", f"slot_{idx:02d}"))
            slots.append(
                ResourceSlot(
                    label=label,
                    device=str(slot_cfg.get("device", "cpu")),
                    active_gpu=slot_cfg.get("active_gpu"),
                    physics_gpu=slot_cfg.get("physics_gpu"),
                )
            )
        return slots

    devices = list(resources.get("devices", []))
    physics = list(resources.get("physics_gpus", []))
    active = list(resources.get("active_gpus", physics))
    if not devices:
        return [ResourceSlot(label="slot_00", device="cpu", active_gpu=None, physics_gpu=None)]
    for idx, device in enumerate(devices):
        slots.append(
            ResourceSlot(
                label=f"slot_{idx:02d}",
                device=str(device),
                active_gpu=active[idx] if idx < len(active) else None,
                physics_gpu=physics[idx] if idx < len(physics) else None,
            )
        )
    return slots


def _load_scalar_summary(log_dir: Path, tags: list[str], window: int) -> dict[str, float]:
    from tensorboard.backend.event_processing import event_accumulator

    candidate_event_dirs = sorted(
        [
            path
            for path in log_dir.iterdir()
            if path.is_dir() and path.name.startswith("PPO_")
        ],
        key=lambda path: path.name,
    )
    if not candidate_event_dirs:
        if log_dir.exists():
            candidate_event_dirs = [log_dir]
        else:
            raise FileNotFoundError(f"TensorBoard directory not found: {log_dir}")

    event_dir = None
    for candidate_dir in sorted(candidate_event_dirs, key=lambda path: path.name, reverse=True):
        event_files = sorted(candidate_dir.glob("events.out.tfevents.*"))
        if event_files:
            event_dir = candidate_dir
            break
    if event_dir is None:
        raise FileNotFoundError(f"No TensorBoard event files found under: {log_dir}")

    accumulator = event_accumulator.EventAccumulator(
        str(event_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    summary: dict[str, float] = {}
    for tag in tags:
        if tag not in available:
            continue
        events = accumulator.Scalars(tag)
        if not events:
            continue
        values = [float(item.value) for item in events]
        summary[tag] = _mean(values[-window:]) if window > 0 else float(values[-1])
        summary[f"{tag}__last"] = float(values[-1])
        summary[f"{tag}__count"] = float(len(values))
    return summary


def _score_candidate(metrics: dict[str, float], scoring_cfg: dict[str, Any]) -> float:
    score = 0.0
    default_missing = float(scoring_cfg.get("missing_value", 0.0))
    for tag, weight in scoring_cfg.get("weights", {}).items():
        metric_value = float(metrics.get(tag, default_missing))
        score += float(weight) * metric_value
    return float(score)


def _apply_genome_to_configs(
    genome: dict[str, Any],
    specs: list[ParamSpec],
    base_ppo: dict[str, Any],
    base_curriculum: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ppo_cfg = copy.deepcopy(base_ppo)
    curriculum_cfg = copy.deepcopy(base_curriculum)
    for spec in specs:
        root = ppo_cfg if spec.target == "ppo" else curriculum_cfg
        _set_nested(root, spec.path, copy.deepcopy(genome[spec.name]))
    return ppo_cfg, curriculum_cfg


def _candidate_paths(study_root: Path, generation: int, index: int) -> dict[str, Path]:
    gen_dir = study_root / f"generation_{generation:03d}"
    cand_dir = gen_dir / f"candidate_{index:03d}"
    return {
        "generation_dir": gen_dir,
        "candidate_dir": cand_dir,
        "ppo_config": cand_dir / "ppo_config.yaml",
        "curriculum_config": cand_dir / "curriculum.yaml",
        "metadata": cand_dir / "candidate.json",
        "log_file": cand_dir / "trial.log",
    }


def _write_candidate_configs(
    *,
    study_root: Path,
    generation: int,
    index: int,
    plan: CandidatePlan,
    specs: list[ParamSpec],
    base_ppo: dict[str, Any],
    base_curriculum: dict[str, Any],
    slot: ResourceSlot,
    trial_timesteps: int,
    preserve_resume_from: bool,
) -> tuple[Path, Path, Path, str]:
    paths = _candidate_paths(study_root, generation, index)
    candidate_id = f"g{generation:03d}_c{index:03d}"
    ppo_cfg, curriculum_cfg = _apply_genome_to_configs(plan.genome, specs, base_ppo, base_curriculum)
    if slot.active_gpu is not None:
        curriculum_cfg.setdefault("app", {})
        curriculum_cfg["app"]["active_gpu"] = int(slot.active_gpu)
    if slot.physics_gpu is not None:
        curriculum_cfg.setdefault("app", {})
        curriculum_cfg["app"]["physics_gpu"] = int(slot.physics_gpu)

    curriculum_cfg.setdefault("app", {})
    curriculum_cfg["app"]["headless"] = True
    curriculum_cfg.setdefault("env", {})
    curriculum_cfg["env"]["render"] = False

    curriculum_path = paths["curriculum_config"]
    _dump_yaml(curriculum_path, curriculum_cfg)

    ppo_cfg["choco_config_path"] = str(curriculum_path)
    ppo_cfg["device"] = slot.device
    if not preserve_resume_from:
        ppo_cfg["resume_from"] = None
    ppo_cfg["total_timesteps"] = int(trial_timesteps)
    ppo_cfg["render_during_training"] = False
    ppo_cfg["record_training_video"] = False
    ppo_cfg["render_rollout_steps"] = 0
    ppo_cfg["save_freq"] = int(max(int(ppo_cfg.get("save_freq", 0)), trial_timesteps + 1))
    ppo_cfg["save_dir"] = str(paths["candidate_dir"] / "checkpoints")
    ppo_cfg["save_prefix"] = candidate_id
    ppo_cfg["runs_root"] = str(study_root / "tensorboard")
    ppo_cfg["run_id"] = candidate_id

    ppo_path = paths["ppo_config"]
    _dump_yaml(ppo_path, ppo_cfg)

    candidate_metadata = {
        "candidate_id": candidate_id,
        "generation": generation,
        "index": index,
        "origin": plan.origin,
        "parent_ids": list(plan.parent_ids),
        "slot": asdict(slot),
        "genome": plan.genome,
        "ppo_config": str(ppo_path),
        "curriculum_config": str(curriculum_path),
        "trial_timesteps": int(trial_timesteps),
    }
    _dump_json(paths["metadata"], candidate_metadata)
    return ppo_path, curriculum_path, paths["log_file"], candidate_id


def _train_command(study_cfg: dict[str, Any], ppo_config_path: Path, candidate_id: str, study_root: Path) -> list[str]:
    runner_cfg = study_cfg.get("runner", {})
    command_prefix = list(runner_cfg.get("command_prefix", []))
    if not command_prefix:
        command_prefix = [sys.executable, "-u"]
    train_entry = str(runner_cfg.get("train_entry", "gpudrive_chocolate/baselines/ppo/ppo_sb3.py"))
    return command_prefix + [
        train_entry,
        "--config",
        str(ppo_config_path),
        "--fresh",
        "--run-id",
        candidate_id,
        "--runs-root",
        str(study_root / "tensorboard"),
    ]


def _svg_escape(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _score_fill_color(status: str, score: float, min_score: float, max_score: float) -> str:
    if status != "completed" or not math.isfinite(score):
        return "#f3d6d6" if status in {"failed", "metric_error", "missing_metrics"} else "#dddddd"
    if max_score <= min_score:
        norm = 1.0
    else:
        norm = max(0.0, min(1.0, (score - min_score) / (max_score - min_score)))
    start = (236, 240, 241)
    end = (117, 184, 102)
    rgb = tuple(int(start[i] + (end[i] - start[i]) * norm) for i in range(3))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def _origin_stroke_color(origin: str) -> str:
    if origin == "elite":
        return "#c89f2d"
    if origin == "crossover_mutation":
        return "#4e79a7"
    if origin == "mutation":
        return "#59a14f"
    if origin == "base":
        return "#555555"
    return "#999999"


def _write_lineage_artifacts(study_root: Path, study_name: str, all_results: list[CandidateResult]) -> None:
    if not all_results:
        return

    generations = sorted({item.generation for item in all_results})
    results_by_generation: dict[int, list[CandidateResult]] = {}
    for generation in generations:
        group = [item for item in all_results if item.generation == generation]
        group.sort(key=lambda item: item.score, reverse=True)
        results_by_generation[generation] = group

    node_width = 220
    node_height = 78
    col_gap = 28
    row_gap = 88
    margin = 28
    max_count = max(len(items) for items in results_by_generation.values())
    width = margin * 2 + max_count * node_width + max(0, max_count - 1) * col_gap
    height = margin * 2 + len(generations) * node_height + max(0, len(generations) - 1) * row_gap

    score_values = [item.score for item in all_results if item.status == "completed" and math.isfinite(item.score)]
    min_score = min(score_values) if score_values else 0.0
    max_score = max(score_values) if score_values else 1.0

    positions: dict[str, tuple[float, float]] = {}
    nodes_payload: list[dict[str, Any]] = []
    for row_idx, generation in enumerate(generations):
        items = results_by_generation[generation]
        total_width = len(items) * node_width + max(0, len(items) - 1) * col_gap
        x_start = margin + max(0.0, (width - 2 * margin - total_width) / 2.0)
        y = margin + row_idx * (node_height + row_gap)
        for col_idx, item in enumerate(items):
            x = x_start + col_idx * (node_width + col_gap)
            positions[item.candidate_id] = (x, y)
            nodes_payload.append(
                {
                    "candidate_id": item.candidate_id,
                    "generation": item.generation,
                    "origin": item.origin,
                    "parent_ids": list(item.parent_ids),
                    "score": item.score,
                    "status": item.status,
                    "metrics": item.metrics,
                    "position": {"x": x, "y": y},
                }
            )

    edges: list[dict[str, str]] = []
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" viewBox="0 0 {int(width)} {int(height)}">',
        '<style>',
        'text { font-family: "DejaVu Sans Mono", monospace; fill: #111; }',
        '.gen-label { font-size: 14px; font-weight: 700; }',
        '.node-title { font-size: 13px; font-weight: 700; }',
        '.node-meta { font-size: 11px; }',
        '.edge { stroke: #9aa0a6; stroke-width: 1.5; fill: none; opacity: 0.9; }',
        '</style>',
        f'<rect x="0" y="0" width="{int(width)}" height="{int(height)}" fill="#fafafa" />',
        f'<text x="{margin}" y="20" class="gen-label">{_svg_escape(study_name)} Family Tree</text>',
    ]

    result_by_id = {item.candidate_id: item for item in all_results}
    for item in all_results:
        child_pos = positions.get(item.candidate_id)
        if child_pos is None:
            continue
        child_x = child_pos[0] + node_width / 2.0
        child_y = child_pos[1]
        for parent_id in item.parent_ids:
            parent_pos = positions.get(parent_id)
            if parent_pos is None:
                continue
            parent_x = parent_pos[0] + node_width / 2.0
            parent_y = parent_pos[1] + node_height
            mid_y = (parent_y + child_y) / 2.0
            svg_lines.append(
                f'<path class="edge" d="M {parent_x:.1f} {parent_y:.1f} C {parent_x:.1f} {mid_y:.1f}, {child_x:.1f} {mid_y:.1f}, {child_x:.1f} {child_y:.1f}" />'
            )
            edges.append({"parent_id": parent_id, "child_id": item.candidate_id})

    for generation in generations:
        items = results_by_generation[generation]
        if not items:
            continue
        y = positions[items[0].candidate_id][1] - 10.0
        svg_lines.append(f'<text x="{margin}" y="{y:.1f}" class="gen-label">Generation {generation}</text>')

    for item in all_results:
        x, y = positions[item.candidate_id]
        fill = _score_fill_color(item.status, item.score, min_score, max_score)
        stroke = _origin_stroke_color(item.origin)
        parents_label = ",".join(item.parent_ids) if item.parent_ids else "-"
        success_rate = item.metrics.get("rollout/success_rate", 0.0)
        road_done = item.metrics.get("rollout/road_contact_done_rate", 0.0)
        veh_done = item.metrics.get("rollout/vehicle_contact_done_rate", 0.0)
        svg_lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{node_width}" height="{node_height}" rx="10" ry="10" fill="{fill}" stroke="{stroke}" stroke-width="3" />'
        )
        svg_lines.append(
            f'<text x="{x + 10:.1f}" y="{y + 18:.1f}" class="node-title">{_svg_escape(item.candidate_id)} [{_svg_escape(item.origin)}]</text>'
        )
        svg_lines.append(
            f'<text x="{x + 10:.1f}" y="{y + 36:.1f}" class="node-meta">score={_svg_escape(_fmt_metric(item.score))} success={_svg_escape(_fmt_metric(success_rate))}</text>'
        )
        svg_lines.append(
            f'<text x="{x + 10:.1f}" y="{y + 52:.1f}" class="node-meta">road={_svg_escape(_fmt_metric(road_done))} veh={_svg_escape(_fmt_metric(veh_done))}</text>'
        )
        svg_lines.append(
            f'<text x="{x + 10:.1f}" y="{y + 68:.1f}" class="node-meta">parents={_svg_escape(parents_label)}</text>'
        )
    svg_lines.append("</svg>")
    (study_root / "family_tree.svg").write_text("\n".join(svg_lines) + "\n", encoding="utf-8")

    lineage_payload = {
        "study_name": study_name,
        "nodes": nodes_payload,
        "edges": edges,
    }
    _dump_json(study_root / "family_tree.json", lineage_payload)


def _write_generation_report(
    *,
    generation: int,
    generation_dir: Path,
    results: list[CandidateResult],
    scoring_cfg: dict[str, Any],
) -> None:
    generation_dir.mkdir(parents=True, exist_ok=True)
    payload = [asdict(item) | {"duration_sec": item.duration_sec} for item in results]
    _dump_json(generation_dir / "results.json", payload)

    lines = [
        f"# Generation {generation:03d}",
        "",
        f"- Candidates: {len(results)}",
        f"- Score weights: `{json.dumps(scoring_cfg.get('weights', {}), sort_keys=True)}`",
        "",
        "| rank | candidate | origin | parents | status | score | success_rate | goal_rate | mean_dist | road_done | below_min_z | device |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    ranked = sorted(results, key=lambda item: item.score, reverse=True)
    for rank, item in enumerate(ranked, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    item.candidate_id,
                    item.origin,
                    ",".join(item.parent_ids) if item.parent_ids else "-",
                    item.status,
                    f"{item.score:.4f}",
                    f"{item.metrics.get('rollout/success_rate', 0.0):.4f}",
                    f"{item.metrics.get('rollout/goal_rate', 0.0):.4f}",
                    f"{item.metrics.get('rollout/mean_dist_to_goal_m', 0.0):.3f}",
                    f"{item.metrics.get('rollout/road_contact_done_rate', 0.0):.4f}",
                    f"{item.metrics.get('rollout/below_min_z_rate', 0.0):.4f}",
                    item.device,
                ]
            )
            + " |"
        )
    (generation_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_study_report(study_root: Path, study_name: str, all_results: list[CandidateResult]) -> None:
    ranked = sorted(all_results, key=lambda item: item.score, reverse=True)
    payload = [asdict(item) | {"duration_sec": item.duration_sec} for item in ranked]
    _dump_json(study_root / "leaderboard.json", payload)
    lines = [
        f"# Study Leaderboard: {study_name}",
        "",
        f"- Completed candidates: {len(all_results)}",
        "",
        "| rank | generation | candidate | origin | parents | score | success_rate | goal_rate | mean_dist | road_done | duration_sec |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, item in enumerate(ranked, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(item.generation),
                    item.candidate_id,
                    item.origin,
                    ",".join(item.parent_ids) if item.parent_ids else "-",
                    f"{item.score:.4f}",
                    f"{item.metrics.get('rollout/success_rate', 0.0):.4f}",
                    f"{item.metrics.get('rollout/goal_rate', 0.0):.4f}",
                    f"{item.metrics.get('rollout/mean_dist_to_goal_m', 0.0):.3f}",
                    f"{item.metrics.get('rollout/road_contact_done_rate', 0.0):.4f}",
                    f"{item.duration_sec or 0.0:.1f}",
                ]
            )
            + " |"
        )
    (study_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_lineage_artifacts(study_root, study_name, all_results)


def _launch_generation(
    *,
    study_cfg: dict[str, Any],
    study_root: Path,
    generation: int,
    plans: list[CandidatePlan],
    specs: list[ParamSpec],
    base_ppo: dict[str, Any],
    base_curriculum: dict[str, Any],
    slots: list[ResourceSlot],
    trial_timesteps: int,
    dry_run: bool,
) -> list[CandidateResult]:
    resources_cfg = study_cfg.get("resources", {})
    scoring_cfg = study_cfg.get("scoring", {})
    search_cfg = study_cfg.get("search", {})
    metric_window = int(study_cfg.get("search", {}).get("metric_window", 5))
    heartbeat_sec = float(search_cfg.get("heartbeat_sec", 15.0))
    tags = list(scoring_cfg.get("weights", {}).keys())
    if scoring_cfg.get("primary_metric"):
        tags.append(str(scoring_cfg["primary_metric"]))
    tags = sorted(set(tags))
    primary_metric = str(scoring_cfg.get("primary_metric", "")) or None
    preserve_resume_from = bool(search_cfg.get("preserve_resume_from", False))

    max_parallel = int(resources_cfg.get("max_parallel", max(1, len(slots))))
    auto_expand_slots = bool(resources_cfg.get("auto_expand_slots", True))
    resolved_slots = _expand_slots(slots, max_parallel) if auto_expand_slots else list(slots[: max_parallel])
    slot_queue = list(resolved_slots)
    pending = list(enumerate(plans))
    running: list[dict[str, Any]] = []
    finished: list[CandidateResult] = []
    last_heartbeat = 0.0

    while pending or running:
        while pending and slot_queue and len(running) < max_parallel:
            index, plan = pending.pop(0)
            slot = slot_queue.pop(0)
            ppo_path, curriculum_path, log_path, candidate_id = _write_candidate_configs(
                study_root=study_root,
                generation=generation,
                index=index,
                plan=plan,
                specs=specs,
                base_ppo=base_ppo,
                base_curriculum=base_curriculum,
                slot=slot,
                trial_timesteps=trial_timesteps,
                preserve_resume_from=preserve_resume_from,
            )
            result = CandidateResult(
                candidate_id=candidate_id,
                generation=generation,
                index=index,
                status="pending",
                score=float("-inf"),
                genome=copy.deepcopy(plan.genome),
                origin=plan.origin,
                parent_ids=list(plan.parent_ids),
                metrics={},
                device=slot.device,
                active_gpu=slot.active_gpu,
                physics_gpu=slot.physics_gpu,
                run_id=candidate_id,
                returncode=None,
                log_dir=str(study_root / "tensorboard" / candidate_id),
                log_file=str(log_path),
                ppo_config_path=str(ppo_path),
                curriculum_config_path=str(curriculum_path),
                started_at=time.time(),
            )
            if dry_run:
                result.status = "dry_run"
                result.score = 0.0
                result.finished_at = time.time()
                finished.append(result)
                print(
                    f"[search] dry-run candidate={candidate_id} slot={slot.label} "
                    f"device={slot.device} origin={plan.origin} parents={plan.parent_ids} "
                    f"genome={_compact_json(plan.genome)}",
                    flush=True,
                )
                slot_queue.append(slot)
                continue

            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = _train_command(study_cfg, ppo_path, candidate_id, study_root)
            log_handle = log_path.open("w", encoding="utf-8")
            log_handle.write("COMMAND: " + " ".join(command) + "\n\n")
            log_handle.flush()
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            result.status = "running"
            print(
                f"[search] launch candidate={candidate_id} generation={generation} slot={slot.label} "
                f"device={slot.device} pid={process.pid} origin={plan.origin} "
                f"parents={plan.parent_ids} log={log_path} genome={_compact_json(plan.genome)}",
                flush=True,
            )
            running.append(
                {
                    "process": process,
                    "log_handle": log_handle,
                    "slot": slot,
                    "result": result,
                }
            )

        if not running:
            continue

        time.sleep(2.0)
        now = time.time()
        if heartbeat_sec > 0 and (now - last_heartbeat) >= heartbeat_sec:
            running_ids = [
                f"{item['result'].candidate_id}(pid={item['process'].pid},slot={item['slot'].label})"
                for item in running
            ]
            print(
                f"[search] heartbeat generation={generation} pending={len(pending)} "
                f"running={len(running)} finished={len(finished)} "
                f"running_ids={running_ids}",
                flush=True,
            )
            last_heartbeat = now
        still_running: list[dict[str, Any]] = []
        for item in running:
            process = item["process"]
            result: CandidateResult = item["result"]
            returncode = process.poll()
            if returncode is None:
                still_running.append(item)
                continue

            item["log_handle"].close()
            result.returncode = int(returncode)
            result.finished_at = time.time()
            if returncode == 0:
                try:
                    metrics = _load_scalar_summary(Path(result.log_dir), tags, metric_window)
                    result.metrics = metrics
                    result.score = _score_candidate(metrics, scoring_cfg)
                    if primary_metric and primary_metric not in metrics:
                        result.status = "missing_metrics"
                    else:
                        result.status = "completed"
                except Exception as exc:
                    result.status = "metric_error"
                    result.metrics = {"error": str(exc)}
                    result.score = float("-inf")
            else:
                result.status = "failed"
                result.score = float("-inf")
            primary_value = result.metrics.get(primary_metric, None) if primary_metric else None
            print(
                f"[search] done candidate={result.candidate_id} status={result.status} "
                f"returncode={result.returncode} duration_sec={_fmt_metric(result.duration_sec, 1)} "
                f"score={_fmt_metric(result.score)} primary={primary_metric or 'n/a'}="
                f"{_fmt_metric(primary_value)} log={result.log_file}",
                flush=True,
            )
            finished.append(result)
            slot_queue.append(item["slot"])
        running = still_running

    generation_dir = study_root / f"generation_{generation:03d}"
    _write_generation_report(
        generation=generation,
        generation_dir=generation_dir,
        results=finished,
        scoring_cfg=scoring_cfg,
    )
    completed_count = sum(1 for item in finished if item.status == "completed")
    failed_count = sum(1 for item in finished if item.status not in {"completed", "dry_run"})
    best_score = max((item.score for item in finished), default=float("-inf"))
    print(
        f"[search] generation_complete generation={generation} completed={completed_count} "
        f"failed={failed_count} best_score={_fmt_metric(best_score)} report={generation_dir / 'results.md'}",
        flush=True,
    )
    return finished


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolutionary hyperparameter search for GPUDrive chocolate PPO.")
    parser.add_argument(
        "--study",
        required=True,
        help="Path to the study YAML file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate candidate configs without launching training.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Optional override for the number of generations.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help="Optional override for population size.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Optional override for timesteps per trial.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Optional override for max parallel trial count.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    study_path = Path(args.study).resolve()
    study_cfg = _load_yaml(study_path)
    if not args.dry_run:
        _require_tensorboard()

    study_section = study_cfg.get("study", {})
    search_cfg = study_cfg.get("search", {})
    resources_cfg = study_cfg.get("resources", {})

    if args.generations is not None:
        search_cfg["generations"] = int(args.generations)
    if args.population_size is not None:
        search_cfg["population_size"] = int(args.population_size)
    if args.timesteps is not None:
        search_cfg["timesteps_per_trial"] = int(args.timesteps)
    if args.max_parallel is not None:
        resources_cfg["max_parallel"] = int(args.max_parallel)
    study_cfg["search"] = search_cfg
    study_cfg["resources"] = resources_cfg

    base_ppo_path = (REPO_ROOT / str(study_cfg["base"]["ppo_config"])).resolve()
    base_ppo = _load_yaml(base_ppo_path)
    base_curriculum_path = Path(str(base_ppo["choco_config_path"]))
    if not base_curriculum_path.is_absolute():
        base_curriculum_path = (REPO_ROOT / base_curriculum_path).resolve()
    base_curriculum = _load_yaml(base_curriculum_path)

    study_name = _slugify(str(study_section.get("name", study_path.stem)))
    output_root = study_section.get("output_root", f"runs/hparam_search/{study_name}")
    study_root = (REPO_ROOT / str(output_root)).resolve()
    study_root.mkdir(parents=True, exist_ok=True)
    _dump_json(study_root / "study_resolved.json", study_cfg)

    specs = _build_param_specs(list(study_cfg.get("space", [])))
    if not specs:
        raise ValueError("Study space is empty")

    rng = random.Random(int(search_cfg.get("random_seed", 42)))
    population_size = int(search_cfg.get("population_size", 4))
    generations = int(search_cfg.get("generations", 2))
    elite_count = int(search_cfg.get("elite_count", 2))
    mutation_rate = float(search_cfg.get("mutation_rate", 0.35))
    crossover_rate = float(search_cfg.get("crossover_rate", 0.7))
    random_candidates = int(search_cfg.get("random_candidates_per_generation", 1))
    trial_timesteps = int(search_cfg.get("timesteps_per_trial", 32768))
    slots = _resolve_resource_slots(study_cfg)
    if not slots:
        raise ValueError("No resource slots resolved")
    configured_max_parallel = int(resources_cfg.get("max_parallel", max(1, len(slots))))
    auto_expand_slots = bool(resources_cfg.get("auto_expand_slots", True))
    resolved_slots = _expand_slots(slots, configured_max_parallel) if auto_expand_slots else list(slots)

    print(
        f"[search] study={study_name} output_root={study_root} base_ppo={base_ppo_path} "
        f"base_curriculum={base_curriculum_path} python={sys.executable}",
        flush=True,
    )
    print(
        f"[search] generations={generations} population_size={population_size} "
        f"timesteps_per_trial={trial_timesteps} max_parallel={configured_max_parallel} "
        f"slots={[asdict(slot) for slot in resolved_slots]}",
        flush=True,
    )

    base_genome = _extract_base_genome(specs, base_ppo, base_curriculum)
    all_results: list[CandidateResult] = []

    generation_plans = _make_generation_zero(base_genome, specs, population_size, rng)
    for generation in range(generations):
        print(
            f"[search] generation={generation} population={len(generation_plans)} "
            f"timesteps={trial_timesteps}",
            flush=True,
        )
        generation_results = _launch_generation(
            study_cfg=study_cfg,
            study_root=study_root,
            generation=generation,
            plans=generation_plans,
            specs=specs,
            base_ppo=base_ppo,
            base_curriculum=base_curriculum,
            slots=slots,
            trial_timesteps=trial_timesteps,
            dry_run=args.dry_run,
        )
        all_results.extend(generation_results)
        _write_study_report(study_root, study_name, all_results)
        if args.dry_run:
            continue
        completed = [item for item in generation_results if item.status == "completed"]
        if not completed:
            print(f"[search] stopping after generation {generation}: no completed candidates", flush=True)
            break
        if generation == generations - 1:
            continue
        generation_plans = _make_next_generation(
            prior_results=completed,
            specs=specs,
            population_size=population_size,
            elite_count=elite_count,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            random_candidates_per_generation=random_candidates,
            rng=rng,
        )

    ranked = sorted(all_results, key=lambda item: item.score, reverse=True)
    if ranked:
        best = ranked[0]
        print(
            "[search] best",
            json.dumps(
                {
                    "candidate_id": best.candidate_id,
                    "generation": best.generation,
                    "score": best.score,
                    "metrics": best.metrics,
                    "genome": best.genome,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
