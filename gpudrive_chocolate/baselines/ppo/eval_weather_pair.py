from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Sequence

import numpy as np
import yaml
from box import Box


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@dataclass
class ActionTraceSample:
    step: int
    world_idx: int
    label: str
    agent_id: int
    longitudinal_cmd: float
    steer_cmd: float
    reward: float
    done: bool


@dataclass(frozen=True)
class WeatherVariant:
    label: str
    friction: Dict[str, Any]


ActionTraceMap = Dict[int, Dict[int, List[ActionTraceSample]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load an SB3 PPO checkpoint, duplicate one scene into two worlds with different "
            "weather/friction settings, run deterministic inference, and plot action traces."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the PPO experiment YAML used for training.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the SB3 PPO checkpoint (.zip).",
    )
    parser.add_argument(
        "--scene-json",
        default=None,
        help="Optional scene_json override. Defaults to the selected assignment in the stage YAML.",
    )
    parser.add_argument(
        "--assignment-index",
        type=int,
        default=0,
        help="World assignment index from the base stage YAML to duplicate when --scene-json is omitted.",
    )
    parser.add_argument(
        "--weather-a",
        default="{}",
        help=(
            "Inline YAML/JSON mapping or file path for weather/friction override A. "
            "Example: '{water_film_mm: 0.025, precip_type: rain, precip_intensity_mmph: 0.5}'"
        ),
    )
    parser.add_argument(
        "--weather-b",
        default="{}",
        help=(
            "Inline YAML/JSON mapping or file path for weather/friction override B. "
            "Example: '{water_film_mm: 0.10, precip_type: rain, precip_intensity_mmph: 2.0}'"
        ),
    )
    parser.add_argument("--label-a", default=None, help="Optional legend label for world A.")
    parser.add_argument("--label-b", default=None, help="Optional legend label for world B.")
    parser.add_argument(
        "--weather",
        action="append",
        default=None,
        help=(
            "Repeatable inline YAML/JSON mapping or file path for a weather/friction override. "
            "May include an optional label field. When omitted, the evaluator builds a default "
            "multi-weather sweep including an extreme wet case."
        ),
    )
    parser.add_argument(
        "--agent-id",
        type=int,
        default=None,
        help="Optional single agent_id to track. Cannot be combined with --agent-ids.",
    )
    parser.add_argument(
        "--agent-ids",
        default=None,
        help="Optional comma-separated list of agent_ids to track. Defaults to all common controllable agent_ids.",
    )
    parser.add_argument(
        "--max-agents",
        type=int,
        default=None,
        help="Optional cap on how many common agent_ids to track when --agent-id/--agent-ids are omitted.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of deterministic env steps to simulate.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, for example cpu, cuda:0, or cuda:1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic policy evaluation.",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/eval_weather_pair",
        help="Directory where the derived curriculum, CSV, JSON, and SVG outputs will be written.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch Isaac Sim with a visible GUI window and render each eval step.",
    )
    parser.add_argument(
        "--gui-step-delay-sec",
        type=float,
        default=0.0,
        help="Optional sleep between GUI eval steps so the rollout is easier to watch.",
    )
    parser.add_argument(
        "--stop-on-tracked-done",
        action="store_true",
        help="Stop early once the tracked agent is done in both worlds. Defaults to false.",
    )
    return parser.parse_args()


def load_mapping_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return dict(data)


def parse_mapping_arg(text: str) -> Dict[str, Any]:
    if text is None:
        return {}
    stripped = str(text).strip()
    if not stripped:
        return {}
    candidate = Path(stripped)
    if candidate.exists():
        return load_mapping_file(candidate)
    data = yaml.safe_load(stripped)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected inline weather/friction mapping, got {type(data).__name__}: {stripped}"
        )
    return dict(data)


def merge_friction(base: Dict[str, Any] | None, override: Dict[str, Any] | None) -> Dict[str, Any]:
    merged = dict(base or {})
    for key, value in dict(override or {}).items():
        merged[key] = value
    return merged


def infer_weather_label(label: str | None, friction: Dict[str, Any], fallback_name: str) -> str:
    if label:
        return str(label)
    road = str(friction.get("road_type", "unknown"))
    precip = str(friction.get("precip_type", "clear"))
    intensity = friction.get("precip_intensity_mmph", None)
    water = friction.get("water_film_mm", None)
    parts = [fallback_name, road, precip]
    if intensity is not None:
        parts.append(f"{float(intensity):.2f}mmph")
    if water is not None:
        parts.append(f"{float(water):.3f}mm")
    return " | ".join(parts)


def build_weather_variant(
    *,
    index: int,
    base_friction: Dict[str, Any],
    override: Dict[str, Any],
    fallback_name: str | None = None,
) -> WeatherVariant:
    payload = dict(override)
    label = payload.pop("label", None)
    friction = merge_friction(base_friction, payload)
    return WeatherVariant(
        label=infer_weather_label(label, friction, fallback_name or f"world-{index}"),
        friction=friction,
    )


def default_weather_variants(base_friction: Dict[str, Any]) -> List[WeatherVariant]:
    sweep = [
        {
            "label": "dry-clear",
            "precip_type": "clear",
            "precip_intensity_mmph": 0.0,
            "water_film_mm": 0.0,
        },
        {
            "label": "light-rain",
            "precip_type": "rain",
            "precip_intensity_mmph": 0.5,
            "water_film_mm": 0.025,
        },
        {
            "label": "moderate-rain",
            "precip_type": "rain",
            "precip_intensity_mmph": 2.0,
            "water_film_mm": 0.10,
        },
        {
            "label": "heavy-rain",
            "precip_type": "rain",
            "precip_intensity_mmph": 6.0,
            "water_film_mm": 0.30,
        },
        {
            "label": "extreme-rain",
            "precip_type": "rain",
            "precip_intensity_mmph": 12.0,
            "water_film_mm": 0.50,
        },
    ]
    return [
        build_weather_variant(index=i, base_friction=base_friction, override=override)
        for i, override in enumerate(sweep)
    ]


def resolve_weather_variants(
    args: argparse.Namespace,
    *,
    base_friction: Dict[str, Any],
) -> List[WeatherVariant]:
    if args.weather:
        return [
            build_weather_variant(
                index=i,
                base_friction=base_friction,
                override=parse_mapping_arg(text),
            )
            for i, text in enumerate(args.weather)
        ]

    if (
        str(args.weather_a).strip() != "{}"
        or str(args.weather_b).strip() != "{}"
        or args.label_a is not None
        or args.label_b is not None
    ):
        friction_a = merge_friction(base_friction, parse_mapping_arg(args.weather_a))
        friction_b = merge_friction(base_friction, parse_mapping_arg(args.weather_b))
        return [
            WeatherVariant(
                label=infer_weather_label(args.label_a, friction_a, "world-0"),
                friction=friction_a,
            ),
            WeatherVariant(
                label=infer_weather_label(args.label_b, friction_b, "world-1"),
                friction=friction_b,
            ),
        ]

    return default_weather_variants(base_friction)


def build_weather_sweep_curriculum_config(
    base_cfg: Dict[str, Any],
    *,
    scene_json: str,
    weather_variants: Sequence[WeatherVariant],
    headless: bool = True,
    render: bool = False,
) -> Dict[str, Any]:
    cfg = deepcopy(base_cfg)
    world_cfg = dict(cfg.get("world", {}))
    world_count = max(1, len(weather_variants))
    grid_cols = min(4, world_count)
    rows = int((world_count + grid_cols - 1) / grid_cols)
    world_cfg["world_count"] = world_count
    world_cfg["grid_cols"] = grid_cols
    world_cfg["rows"] = rows
    world_cfg["assignments"] = [
        {"scene_json": str(scene_json), "friction": dict(variant.friction)}
        for variant in weather_variants
    ]
    cfg["world"] = world_cfg

    app_cfg = dict(cfg.get("app", {}))
    app_cfg["headless"] = bool(headless)
    cfg["app"] = app_cfg

    env_cfg = dict(cfg.get("env", {}))
    env_cfg["render"] = bool(render)
    env_cfg["auto_reset_done"] = False
    env_cfg["auto_reset_timeout"] = False
    cfg["env"] = env_cfg

    physics_cfg = dict(cfg.get("physics", {}))
    physics_cfg["report_gpu_dynamics_once"] = False
    cfg["physics"] = physics_cfg

    return cfg


def build_pair_curriculum_config(
    base_cfg: Dict[str, Any],
    *,
    scene_json: str,
    friction_a: Dict[str, Any],
    friction_b: Dict[str, Any],
    headless: bool = True,
    render: bool = False,
) -> Dict[str, Any]:
    return build_weather_sweep_curriculum_config(
        base_cfg,
        scene_json=scene_json,
        weather_variants=[
            WeatherVariant(label=infer_weather_label(None, friction_a, "world-0"), friction=dict(friction_a)),
            WeatherVariant(label=infer_weather_label(None, friction_b, "world-1"), friction=dict(friction_b)),
        ],
        headless=headless,
        render=render,
    )


def collect_world_agent_indices(flat_slot_keys: Sequence[object]) -> Dict[int, Dict[int, int]]:
    mapping: Dict[int, Dict[int, int]] = {}
    for env_idx, key in enumerate(flat_slot_keys):
        world_idx = int(getattr(key, "world_idx"))
        agent_id = int(getattr(key, "agent_id"))
        mapping.setdefault(world_idx, {})[agent_id] = env_idx
    return mapping


def parse_agent_ids_arg(text: str | None) -> List[int]:
    if text is None:
        return []
    values: List[int] = []
    for part in str(text).split(","):
        stripped = part.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    seen: set[int] = set()
    out: List[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def choose_tracked_agent_ids(
    world_agent_indices: Dict[int, Dict[int, int]],
    requested_agent_id: int | None,
    requested_agent_ids: Sequence[int] | None = None,
    max_agents: int | None = None,
) -> List[int]:
    world_maps = [set(agent_map.keys()) for _, agent_map in sorted(world_agent_indices.items())]
    if not world_maps:
        raise RuntimeError("No controllable agent mappings found across worlds.")
    common_ids = sorted(set.intersection(*world_maps))
    if not common_ids:
        raise RuntimeError("No common controllable agent_id found across all requested worlds.")

    explicit_ids = [int(x) for x in requested_agent_ids or []]
    if requested_agent_id is not None and explicit_ids:
        raise ValueError("Use either --agent-id or --agent-ids, not both.")

    if requested_agent_id is not None:
        if int(requested_agent_id) not in common_ids:
            raise ValueError(
                f"Requested agent_id={requested_agent_id} is not controllable in both worlds. "
                f"Common ids: {common_ids}"
            )
        return [int(requested_agent_id)]

    if explicit_ids:
        missing = [agent_id for agent_id in explicit_ids if agent_id not in common_ids]
        if missing:
            raise ValueError(
                f"Requested agent_ids are not controllable in both worlds: {missing}. "
                f"Common ids: {common_ids}"
            )
        return explicit_ids

    if max_agents is not None and int(max_agents) > 0:
        return common_ids[: int(max_agents)]
    return common_ids


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def write_action_csv(path: Path, traces: ActionTraceMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "world_idx",
                "label",
                "agent_id",
                "longitudinal_cmd",
                "steer_cmd",
                "reward",
                "done",
            ],
        )
        writer.writeheader()
        for agent_id in sorted(traces):
            for world_idx in sorted(traces[agent_id]):
                for sample in traces[agent_id][world_idx]:
                    writer.writerow(asdict(sample))


def _svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _line_points(
    values: Sequence[float],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    y_min: float,
    y_max: float,
) -> str:
    if not values:
        return ""
    span_x = max(1, len(values) - 1)
    span_y = max(1e-6, float(y_max - y_min))
    points: List[str] = []
    for idx, value in enumerate(values):
        x = left + (float(idx) / float(span_x)) * width
        norm_y = (float(value) - float(y_min)) / span_y
        y = top + (1.0 - norm_y) * height
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _done_markers(
    samples: Sequence[ActionTraceSample],
    values: Sequence[float],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    y_min: float,
    y_max: float,
    color: str,
) -> List[str]:
    if not samples or not values:
        return []
    span_x = max(1, len(values) - 1)
    span_y = max(1e-6, float(y_max - y_min))
    out: List[str] = []
    for idx, sample in enumerate(samples):
        if not sample.done:
            continue
        x = left + (float(idx) / float(span_x)) * width
        norm_y = (float(values[idx]) - float(y_min)) / span_y
        y = top + (1.0 - norm_y) * height
        out.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}" stroke="#ffffff" stroke-width="1" />'
        )
    return out


def render_action_trace_svg(
    traces: ActionTraceMap,
    *,
    world_labels: Dict[int, str],
    title: str,
) -> str:
    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#17becf",
        "#e377c2",
    ]
    world_indices = sorted(world_labels)
    colors = {world_idx: palette[i % len(palette)] for i, world_idx in enumerate(world_indices)}
    agent_ids = sorted(traces)
    width = 1180
    legend_cols = 2
    legend_rows = max(1, (len(world_indices) + legend_cols - 1) // legend_cols)
    legend_row_height = 20
    header_height = 64 + legend_rows * legend_row_height
    panel_height = 170
    panel_gap = 40
    block_gap = 70
    block_height = (2 * panel_height) + panel_gap + 28
    height = header_height + max(1, len(agent_ids)) * block_height + max(0, len(agent_ids) - 1) * block_gap + 40
    margin_left = 72
    margin_right = 28
    margin_top = header_height
    plot_width = width - margin_left - margin_right
    legend_y = 38

    lines: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfc" />',
        f'<text x="{margin_left}" y="28" font-size="22" font-family="monospace" fill="#111827">{_svg_escape(title)}</text>',
        f'<text x="{margin_left}" y="{legend_y}" font-size="12" font-family="monospace" fill="#374151">done markers are circles on each line</text>',
    ]

    legend_start_y = legend_y + 18
    legend_col_width = 500
    for offset, world_idx in enumerate(world_indices):
        col = offset % legend_cols
        row = offset // legend_cols
        x = margin_left + col * legend_col_width
        y = legend_start_y + row * legend_row_height
        color = colors[world_idx]
        lines.append(f'<line x1="{x}" y1="{y - 6}" x2="{x + 30}" y2="{y - 6}" stroke="{color}" stroke-width="3" />')
        lines.append(
            f'<text x="{x + 38}" y="{y - 2}" font-size="13" font-family="monospace" fill="#111827">{_svg_escape(world_labels[world_idx])}</text>'
        )

    for agent_offset, agent_id in enumerate(agent_ids):
        block_top = margin_top + agent_offset * (block_height + block_gap)
        lines.append(
            f'<text x="{margin_left}" y="{block_top - 14}" font-size="16" font-family="monospace" fill="#111827">agent_id={agent_id}</text>'
        )
        panels = [
            ("Longitudinal Cmd", "longitudinal_cmd", block_top),
            ("Steer Cmd", "steer_cmd", block_top + panel_height + panel_gap),
        ]
        max_steps = max(
            (
                len(samples)
                for world_samples in traces.get(agent_id, {}).values()
                for samples in [world_samples]
            ),
            default=0,
        )

        for panel_title, attr_name, top in panels:
            bottom = top + panel_height
            lines.append(
                f'<rect x="{margin_left}" y="{top}" width="{plot_width}" height="{panel_height}" fill="#ffffff" stroke="#d1d5db" />'
            )
            lines.append(
                f'<text x="{margin_left}" y="{top - 10}" font-size="14" font-family="monospace" fill="#111827">{_svg_escape(panel_title)}</text>'
            )
            for tick_value in (-1.0, -0.5, 0.0, 0.5, 1.0):
                y = top + (1.0 - ((tick_value + 1.0) / 2.0)) * panel_height
                lines.append(
                    f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1" />'
                )
                lines.append(
                    f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="monospace" fill="#4b5563">{tick_value:.1f}</text>'
                )

            if max_steps > 1:
                for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                    x = margin_left + frac * plot_width
                    step = int(round(frac * (max_steps - 1)))
                    lines.append(
                        f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="#f3f4f6" stroke-width="1" />'
                    )
                    lines.append(
                        f'<text x="{x:.2f}" y="{bottom + 18}" text-anchor="middle" font-size="12" font-family="monospace" fill="#4b5563">{step}</text>'
                    )
                lines.append(
                    f'<text x="{margin_left + plot_width / 2:.2f}" y="{bottom + 38}" text-anchor="middle" font-size="13" font-family="monospace" fill="#111827">step</text>'
                )

            for world_idx in world_indices:
                samples = traces.get(agent_id, {}).get(world_idx, [])
                values = [float(getattr(sample, attr_name)) for sample in samples]
                points = _line_points(
                    values,
                    left=margin_left,
                    top=top,
                    width=plot_width,
                    height=panel_height,
                    y_min=-1.0,
                    y_max=1.0,
                )
                if points:
                    lines.append(
                        f'<polyline fill="none" stroke="{colors[world_idx]}" stroke-width="2.5" points="{points}" />'
                    )
                    lines.extend(
                        _done_markers(
                            samples,
                            values,
                            left=margin_left,
                            top=top,
                            width=plot_width,
                            height=panel_height,
                            y_min=-1.0,
                            y_max=1.0,
                            color=colors[world_idx],
                        )
                    )

    lines.append("</svg>")
    return "\n".join(lines)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_device(device: str | None) -> str:
    if not device:
        return "cpu"
    text = str(device)
    if text.startswith("cuda") and ":" not in text:
        return "cuda:0"
    return text


def load_ppo_experiment_config(path: str) -> Box:
    with open(path, "r", encoding="utf-8") as f:
        return Box(yaml.safe_load(f))


def load_base_curriculum(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping in curriculum YAML {path}")
    return cfg


def _checkpoint_steps(path: Path) -> int:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    if match:
        return int(match.group(1))
    return -1


def _checkpoint_prefix(name: str) -> str:
    stem = Path(name).stem
    match = re.match(r"(.+)_\d+_steps$", stem)
    if match:
        return str(match.group(1))
    return stem


def _latest_checkpoint_in_dir(search_dir: Path, prefix: str | None = None) -> Path | None:
    if not search_dir.exists() or not search_dir.is_dir():
        return None
    candidates = [p for p in search_dir.glob("*.zip") if p.is_file()]
    if prefix:
        candidates = [p for p in candidates if p.name.startswith(prefix)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (_checkpoint_steps(p), p.name))
    return candidates[-1]


def ensure_checkpoint_exists(path: str, exp_config: Box) -> Path:
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.exists() and checkpoint_path.is_file():
        return checkpoint_path.resolve()

    requested_prefix = _checkpoint_prefix(checkpoint_path.name)
    search_specs: List[tuple[Path, str | None]] = []
    if checkpoint_path.exists() and checkpoint_path.is_dir():
        search_specs.append((checkpoint_path, requested_prefix or None))
        search_specs.append((checkpoint_path, None))
    else:
        search_specs.append((checkpoint_path.parent, requested_prefix or None))
        search_specs.append((checkpoint_path.parent, None))

    save_dir = Path(str(getattr(exp_config, "save_dir", ""))).expanduser()
    save_prefix = str(getattr(exp_config, "save_prefix", "")).strip() or None
    if save_dir:
        search_specs.append((save_dir, save_prefix))
        if requested_prefix and requested_prefix != save_prefix:
            search_specs.append((save_dir, requested_prefix))
        search_specs.append((save_dir, None))

    seen: set[tuple[str, str | None]] = set()
    for search_dir, prefix in search_specs:
        key = (str(search_dir), prefix)
        if key in seen:
            continue
        seen.add(key)
        resolved = _latest_checkpoint_in_dir(search_dir, prefix)
        if resolved is not None:
            print(
                f"[eval] resolved checkpoint {checkpoint_path} -> {resolved}"
            )
            return resolved.resolve()

    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def run(args: argparse.Namespace) -> None:
    import torch
    from stable_baselines3 import PPO

    from gpudrive_chocolate.baselines.ppo.ppo_sb3 import build_resume_custom_objects
    from gpudrive_chocolate.env.sb3_wrapper import ChocolateSB3MultiAgentEnv
    from gpudrive_chocolate.networks.late_fusion_policy import LateFusionPolicy  # noqa: F401

    del LateFusionPolicy

    exp_config = load_ppo_experiment_config(args.config)
    checkpoint_path = ensure_checkpoint_exists(args.checkpoint, exp_config)
    device = normalize_device(args.device or getattr(exp_config, "device", "cpu"))
    exp_config.device = device
    if isinstance(device, str) and device.startswith("cuda"):
        torch.cuda.set_device(device)

    base_stage_cfg = load_base_curriculum(exp_config.choco_config_path)
    assignments = list(base_stage_cfg.get("world", {}).get("assignments", []))
    if not assignments:
        raise ValueError(f"No world assignments found in {exp_config.choco_config_path}")

    assignment_idx = int(args.assignment_index)
    if assignment_idx < 0 or assignment_idx >= len(assignments):
        raise IndexError(
            f"assignment-index={assignment_idx} out of range for {len(assignments)} assignments"
        )
    base_assignment = dict(assignments[assignment_idx])
    scene_json = str(args.scene_json or base_assignment.get("scene_json"))
    if not scene_json:
        raise ValueError("Could not resolve scene_json from assignment or --scene-json")

    friction_base = dict(base_assignment.get("friction", {}))
    weather_variants = resolve_weather_variants(args, base_friction=friction_base)
    world_labels = {i: variant.label for i, variant in enumerate(weather_variants)}

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    derived_cfg = build_weather_sweep_curriculum_config(
        base_stage_cfg,
        scene_json=scene_json,
        weather_variants=weather_variants,
        headless=not bool(args.gui),
        render=bool(args.gui),
    )
    derived_cfg_path = out_dir / "paired_eval_curriculum.yaml"
    write_yaml(derived_cfg_path, derived_cfg)

    eval_exp_config = deepcopy(exp_config)
    eval_exp_config.choco_config_path = str(derived_cfg_path)
    eval_exp_config.device = device

    seed_everything(int(args.seed))

    env = None
    env = ChocolateSB3MultiAgentEnv(
        choco_config_path=eval_exp_config.choco_config_path,
        exp_config=eval_exp_config,
        device=eval_exp_config.device,
        reward_type=eval_exp_config.reward_type,
        collision_weight=eval_exp_config.collision_weight,
        goal_achieved_weight=eval_exp_config.goal_achieved_weight,
        off_road_weight=eval_exp_config.off_road_weight,
        log_distance_weight=eval_exp_config.log_distance_weight,
    )

    try:
        model = PPO.load(
            str(checkpoint_path),
            env=env,
            device=eval_exp_config.device,
            custom_objects=build_resume_custom_objects(env),
        )
        model.policy.set_training_mode(False)

        obs = env.reset()
        world_agent_indices = collect_world_agent_indices(env.flat_slot_keys)
        tracked_agent_ids = choose_tracked_agent_ids(
            world_agent_indices,
            requested_agent_id=args.agent_id,
            requested_agent_ids=parse_agent_ids_arg(args.agent_ids),
            max_agents=args.max_agents,
        )
        traces: ActionTraceMap = {
            int(agent_id): {world_idx: [] for world_idx in sorted(world_labels)}
            for agent_id in tracked_agent_ids
        }
        world_finished = {
            (int(agent_id), world_idx): False
            for agent_id in tracked_agent_ids
            for world_idx in sorted(world_labels)
        }
        stop_reason = "max_steps"

        for step_idx in range(int(args.steps)):
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, _ = env.step(actions)
            if args.gui and float(args.gui_step_delay_sec) > 0.0:
                sleep(float(args.gui_step_delay_sec))
            for agent_id in tracked_agent_ids:
                for world_idx, label in world_labels.items():
                    if world_finished[(int(agent_id), world_idx)]:
                        continue
                    env_idx = world_agent_indices[world_idx][int(agent_id)]
                    done_flag = bool(dones[env_idx])
                    traces[int(agent_id)][world_idx].append(
                        ActionTraceSample(
                            step=step_idx,
                            world_idx=world_idx,
                            label=label,
                            agent_id=int(agent_id),
                            longitudinal_cmd=float(actions[env_idx, 0]),
                            steer_cmd=float(actions[env_idx, 1]),
                            reward=float(rewards[env_idx]),
                            done=done_flag,
                        )
                    )
                    if done_flag:
                        world_finished[(int(agent_id), world_idx)] = True

            if bool(getattr(env, "info_dict", {}).get("truncated", False)):
                stop_reason = "timeout"
                break
            if bool(args.stop_on_tracked_done) and all(world_finished.values()):
                stop_reason = "tracked_agents_done"
                break

        csv_path = out_dir / "tracked_agent_actions.csv"
        write_action_csv(csv_path, traces)

        svg_text = render_action_trace_svg(
            traces,
            world_labels=world_labels,
            title=f"Deterministic Action Trace | scene={scene_json} | agents={len(tracked_agent_ids)}",
        )
        svg_path = out_dir / "tracked_agent_actions.svg"
        svg_path.write_text(svg_text, encoding="utf-8")

        common_agent_ids = sorted(
            set.intersection(*(set(agent_map.keys()) for _, agent_map in sorted(world_agent_indices.items())))
        )
        metadata = {
            "checkpoint": str(checkpoint_path.resolve()),
            "ppo_config": str(Path(args.config).expanduser().resolve()),
            "base_curriculum": str(Path(exp_config.choco_config_path).expanduser().resolve()),
            "derived_curriculum": str(derived_cfg_path),
            "scene_json": scene_json,
            "steps": int(args.steps),
            "steps_executed": int(
                max(
                    (
                        len(world_samples)
                        for agent_samples in traces.values()
                        for world_samples in agent_samples.values()
                    ),
                    default=0,
                )
            ),
            "device": device,
            "seed": int(args.seed),
            "gui": bool(args.gui),
            "stop_on_tracked_done": bool(args.stop_on_tracked_done),
            "tracked_agent_ids": [int(agent_id) for agent_id in tracked_agent_ids],
            "num_tracked_agents": int(len(tracked_agent_ids)),
            "num_weather_worlds": int(len(weather_variants)),
            "stop_reason": stop_reason,
            "world_labels": {str(k): v for k, v in world_labels.items()},
            "weather_variants": [
                {"world_idx": i, "label": variant.label, "friction": variant.friction}
                for i, variant in enumerate(weather_variants)
            ],
            "common_agent_ids": common_agent_ids,
            "quarantined_tokens": sorted(getattr(env.choco_env, "_quarantined_tokens", [])),
            "num_envs": int(env.num_envs),
            "csv": str(csv_path),
            "svg": str(svg_path),
        }
        write_json(out_dir / "metadata.json", metadata)

        print(f"[eval] wrote derived curriculum to {derived_cfg_path}")
        print(f"[eval] tracked agent_ids={tracked_agent_ids}")
        print(f"[eval] wrote action CSV to {csv_path}")
        print(f"[eval] wrote action plot to {svg_path}")
    finally:
        if env is not None:
            env.close()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
