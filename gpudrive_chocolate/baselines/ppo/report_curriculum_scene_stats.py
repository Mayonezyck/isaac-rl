#!/usr/bin/env python3
"""Summarize curriculum scene statistics and generate distribution figures.

Outputs:
- summary JSON with aggregate stats
- per-scene CSV
- per-agent OD CSV
- histogram figures (agents per scene, OD distance)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.trfc import prepare_stage_world_specs


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping YAML at {path}, got {type(payload).__name__}")
    return payload


def _resolve_curriculum_from_ppo(ppo_path: Path) -> Path:
    ppo_cfg = _load_yaml(ppo_path)
    raw = ppo_cfg.get("choco_config_path", None)
    if raw is None:
        raise ValueError(f"Missing choco_config_path in PPO config: {ppo_path}")
    candidate = Path(str(raw)).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (ppo_path.parent / candidate).resolve()
        if not resolved.exists():
            resolved = (Path.cwd() / candidate).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Could not resolve curriculum config path: {raw}")
    return resolved


def _safe_float(raw: Any) -> Optional[float]:
    try:
        value = float(raw)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _iter_agents(scene_cfg: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    agents = scene_cfg.get("agents", {}) or {}
    items = list((agents.get("items", []) or []))
    for item in items:
        if isinstance(item, Mapping):
            yield item


def _od_distance_xy(agent: Mapping[str, Any]) -> Optional[float]:
    start = agent.get("start", {}) or {}
    goal = agent.get("end", {}) or {}
    sx = _safe_float(start.get("x", None))
    sy = _safe_float(start.get("y", None))
    gx = _safe_float(goal.get("x", None))
    gy = _safe_float(goal.get("y", None))
    if sx is None or sy is None or gx is None or gy is None:
        return None
    dx = float(gx - sx)
    dy = float(gy - sy)
    return float(math.sqrt(dx * dx + dy * dy))


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float32), q))


def _summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "min": 0.0,
            "p10": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    vals = [float(v) for v in values]
    return {
        "count": float(len(vals)),
        "min": float(min(vals)),
        "p10": _percentile(vals, 0.10),
        "mean": float(sum(vals) / len(vals)),
        "median": _percentile(vals, 0.50),
        "p90": _percentile(vals, 0.90),
        "max": float(max(vals)),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _render_figures(
    *,
    out_dir: Path,
    scene_rows: Sequence[Mapping[str, Any]],
    od_distances: Sequence[float],
    bins: int,
) -> Dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"[stats][warn] matplotlib unavailable; skipping figures ({exc})")
        return {}

    out: Dict[str, str] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    agents_per_scene = [float(row.get("num_agents_total", 0.0)) for row in scene_rows]
    if agents_per_scene:
        fig = plt.figure(figsize=(9, 5), dpi=150)
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(agents_per_scene, bins=max(5, int(bins)), color="#2E86DE", alpha=0.9)
        ax.set_title("Agents Per Scene")
        ax.set_xlabel("Agents")
        ax.set_ylabel("Scene Count")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = out_dir / "hist_agents_per_scene.png"
        fig.savefig(path)
        plt.close(fig)
        out["hist_agents_per_scene_png"] = str(path.resolve())

    if od_distances:
        fig = plt.figure(figsize=(9, 5), dpi=150)
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(od_distances, bins=max(10, int(bins)), color="#27AE60", alpha=0.9)
        ax.set_title("OD Distance Distribution (XY)")
        ax.set_xlabel("OD Distance (m)")
        ax.set_ylabel("Agent Count")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = out_dir / "hist_od_distance_xy_m.png"
        fig.savefig(path)
        plt.close(fig)
        out["hist_od_distance_xy_m_png"] = str(path.resolve())

    if scene_rows:
        xs = [float(row.get("num_agents_total", 0.0)) for row in scene_rows]
        ys = [float(row.get("od_mean_xy_m", 0.0)) for row in scene_rows]
        fig = plt.figure(figsize=(9, 5), dpi=150)
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(xs, ys, s=18, alpha=0.8, color="#8E44AD")
        ax.set_title("Scene Difficulty Scatter")
        ax.set_xlabel("Agents Per Scene")
        ax.set_ylabel("Mean OD Distance (m)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = out_dir / "scatter_agents_vs_mean_od.png"
        fig.savefig(path)
        plt.close(fig)
        out["scatter_agents_vs_mean_od_png"] = str(path.resolve())

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate curriculum-level scene stats: OD distance distribution, "
            "agents per scene, and summary report."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--curriculum", type=str, help="Curriculum YAML path.")
    group.add_argument("--ppo-config", type=str, help="PPO YAML path (uses choco_config_path).")
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="If >0, only process first N world specs from curriculum.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Histogram bins for figures.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/curriculum_scene_stats",
        help="Directory for JSON/CSV report and figures.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.curriculum is not None:
        curriculum_path = Path(args.curriculum).expanduser().resolve()
    else:
        ppo_path = Path(args.ppo_config).expanduser().resolve()
        curriculum_path = _resolve_curriculum_from_ppo(ppo_path)

    if not curriculum_path.exists():
        raise FileNotFoundError(f"Curriculum config not found: {curriculum_path}")
    cfg = _load_yaml(curriculum_path)

    specs = prepare_stage_world_specs(cfg)
    if int(args.max_scenes) > 0:
        specs = specs[: int(args.max_scenes)]
    if not specs:
        raise ValueError(f"No world specs resolved from curriculum: {curriculum_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_rows: List[Dict[str, Any]] = []
    agent_rows: List[Dict[str, Any]] = []
    all_od_distances: List[float] = []

    for wi, spec in enumerate(specs):
        scene_path = Path(spec.scene_json_path).resolve()
        scene_name = scene_path.name
        scene_cfg = json.loads(scene_path.read_text(encoding="utf-8"))

        agents = list(_iter_agents(scene_cfg))
        road_polylines = list(((scene_cfg.get("road", {}) or {}).get("polylines", []) or []))
        road_types = [int((pl or {}).get("type", -1)) for pl in road_polylines if isinstance(pl, Mapping)]

        ods: List[float] = []
        invalid_od = 0
        for ai, agent in enumerate(agents):
            od = _od_distance_xy(agent)
            if od is None:
                invalid_od += 1
                continue
            ods.append(float(od))
            all_od_distances.append(float(od))
            agent_rows.append(
                {
                    "world_idx": int(wi),
                    "scene_json": str(scene_name),
                    "agent_idx": int(ai),
                    "agent_id": int(agent.get("agent_id", ai)),
                    "track_idx": int(agent.get("track_idx", ai)),
                    "agent_type": int(agent.get("agent_type", -1)),
                    "od_distance_xy_m": float(od),
                }
            )

        scene_rows.append(
            {
                "world_idx": int(wi),
                "scene_json": str(scene_name),
                "num_agents_total": int(len(agents)),
                "num_agents_valid_od": int(len(ods)),
                "num_agents_invalid_od": int(invalid_od),
                "od_min_xy_m": float(min(ods)) if ods else 0.0,
                "od_p10_xy_m": _percentile(ods, 0.10) if ods else 0.0,
                "od_mean_xy_m": float(sum(ods) / len(ods)) if ods else 0.0,
                "od_median_xy_m": _percentile(ods, 0.50) if ods else 0.0,
                "od_p90_xy_m": _percentile(ods, 0.90) if ods else 0.0,
                "od_max_xy_m": float(max(ods)) if ods else 0.0,
                "road_polyline_count": int(len(road_polylines)),
                "road_lane_12_count": int(sum(1 for t in road_types if t in {1, 2})),
                "road_edge_1516_count": int(sum(1 for t in road_types if t in {15, 16})),
            }
        )

    summary = {
        "curriculum_path": str(curriculum_path),
        "num_world_specs": int(len(specs)),
        "num_unique_scene_files": int(len({str(r["scene_json"]) for r in scene_rows})),
        "agents_per_scene": _summarize([float(r["num_agents_total"]) for r in scene_rows]),
        "od_distance_xy_m": _summarize(all_od_distances),
        "invalid_od_total": int(sum(int(r["num_agents_invalid_od"]) for r in scene_rows)),
    }

    figures = _render_figures(
        out_dir=out_dir,
        scene_rows=scene_rows,
        od_distances=all_od_distances,
        bins=max(5, int(args.bins)),
    )
    summary["figures"] = figures

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _write_csv(
        out_dir / "scene_summary.csv",
        scene_rows,
        fieldnames=[
            "world_idx",
            "scene_json",
            "num_agents_total",
            "num_agents_valid_od",
            "num_agents_invalid_od",
            "od_min_xy_m",
            "od_p10_xy_m",
            "od_mean_xy_m",
            "od_median_xy_m",
            "od_p90_xy_m",
            "od_max_xy_m",
            "road_polyline_count",
            "road_lane_12_count",
            "road_edge_1516_count",
        ],
    )

    _write_csv(
        out_dir / "agent_od_distances.csv",
        agent_rows,
        fieldnames=[
            "world_idx",
            "scene_json",
            "agent_idx",
            "agent_id",
            "track_idx",
            "agent_type",
            "od_distance_xy_m",
        ],
    )

    print(f"[stats] curriculum={curriculum_path}")
    print(f"[stats] worlds={len(specs)} unique_scenes={summary['num_unique_scene_files']}")
    print(
        "[stats] agents_per_scene "
        f"mean={summary['agents_per_scene']['mean']:.2f} "
        f"p10={summary['agents_per_scene']['p10']:.2f} "
        f"p90={summary['agents_per_scene']['p90']:.2f}"
    )
    print(
        "[stats] od_distance_xy_m "
        f"mean={summary['od_distance_xy_m']['mean']:.2f} "
        f"p10={summary['od_distance_xy_m']['p10']:.2f} "
        f"p90={summary['od_distance_xy_m']['p90']:.2f}"
    )
    print(f"[stats] wrote summary_json={summary_path}")
    print(f"[stats] wrote scene_csv={out_dir / 'scene_summary.csv'}")
    print(f"[stats] wrote agent_csv={out_dir / 'agent_od_distances.csv'}")
    for key, value in figures.items():
        print(f"[stats] wrote figure {key}={value}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
