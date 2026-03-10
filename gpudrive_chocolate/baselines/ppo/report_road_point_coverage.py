from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from src.trfc import prepare_stage_world_specs
from src.trfc.lane_center_sampler import compute_scene_center_from_road


@dataclass(frozen=True)
class QueryPoint:
    point_id: int
    source: str
    x_local_m: float
    y_local_m: float


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping YAML at {path}, got {type(payload).__name__}")
    return payload


def _resolve_curriculum_path(ppo_config_path: Path, ppo_cfg: Mapping[str, Any]) -> Path:
    raw = ppo_cfg.get("choco_config_path", None)
    if raw is None:
        raise ValueError(f"Missing choco_config_path in PPO config: {ppo_config_path}")
    candidate = Path(str(raw)).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (ppo_config_path.parent / candidate).resolve()
        if not resolved.exists():
            resolved = (Path.cwd() / candidate).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Could not resolve curriculum config path: {raw}")
    return resolved


def _parse_radii(radii_arg: str) -> List[float]:
    parts = [p.strip() for p in str(radii_arg).split(",")]
    vals: List[float] = []
    for part in parts:
        if not part:
            continue
        vals.append(float(part))
    vals = sorted({v for v in vals if v > 0.0})
    if not vals:
        raise ValueError(f"Expected at least one positive radius, got: {radii_arg}")
    return vals


def _in_bounds_xy(x_local_m: float, y_local_m: float, bounds_size_m: float) -> bool:
    half = 0.5 * float(bounds_size_m)
    return (-half <= float(x_local_m) <= half) and (-half <= float(y_local_m) <= half)


def _coerce_xyz(raw_xyz: Any) -> np.ndarray | None:
    arr = np.asarray(raw_xyz, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    if arr.shape[1] == 2:
        arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=np.float32)], axis=1)
    return arr[:, :3].astype(np.float32, copy=False)


def _build_world_road_points(
    scene_cfg: Mapping[str, Any],
    *,
    bounds_size_m: float,
    origin_mode: str,
    jump_break_m: float,
    flatten_road_z: bool,
    road_z_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    road_cfg = (scene_cfg.get("road", {}) or {})
    polylines = list(road_cfg.get("polylines", []) or [])
    scene_center = (
        compute_scene_center_from_road(dict(scene_cfg))
        if str(origin_mode).strip().lower() == "center"
        else np.zeros((3,), dtype=np.float32)
    )
    by_type: Dict[int, List[np.ndarray]] = {}
    for poly in polylines:
        if not isinstance(poly, Mapping):
            continue
        road_type = int(poly.get("type", -1))
        pts = _coerce_xyz(poly.get("xyz", None))
        if pts is None:
            continue
        pts_local = pts - scene_center[None, :]
        if flatten_road_z:
            pts_local[:, 2] = float(road_z_m)
        in_bounds = np.asarray(
            [_in_bounds_xy(float(p[0]), float(p[1]), float(bounds_size_m)) for p in pts_local],
            dtype=bool,
        )
        pts_local = pts_local[in_bounds]
        if pts_local.shape[0] < 2:
            continue
        by_type.setdefault(int(road_type), []).append(pts_local)

    points_xy: List[Tuple[float, float]] = []
    point_types: List[int] = []
    jump_thr = float(jump_break_m)
    for road_type in sorted(by_type.keys()):
        for pts_local in by_type[road_type]:
            for i in range(pts_local.shape[0] - 1):
                p0 = pts_local[i]
                p1 = pts_local[i + 1]
                if not (
                    _in_bounds_xy(float(p0[0]), float(p0[1]), bounds_size_m)
                    and _in_bounds_xy(float(p1[0]), float(p1[1]), bounds_size_m)
                ):
                    continue
                dx = float(p1[0] - p0[0])
                dy = float(p1[1] - p0[1])
                seg_len = float(math.sqrt(dx * dx + dy * dy))
                if seg_len <= 1e-6:
                    continue
                if seg_len > jump_thr:
                    continue
                mx = float((p0[0] + p1[0]) * 0.5)
                my = float((p0[1] + p1[1]) * 0.5)
                points_xy.append((mx, my))
                point_types.append(int(road_type))

    if not points_xy:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return (
        np.asarray(points_xy, dtype=np.float32),
        np.asarray(point_types, dtype=np.int32),
    )


def _iter_scene_agents(scene_cfg: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    agents = scene_cfg.get("agents", {})
    if isinstance(agents, Mapping):
        items = agents.get("items", []) or []
    elif isinstance(agents, Sequence):
        items = agents
    else:
        items = []
    for item in items:
        if isinstance(item, Mapping):
            yield item


def _collect_query_points(
    scene_cfg: Mapping[str, Any],
    *,
    bounds_size_m: float,
    origin_mode: str,
    source: str,
) -> List[QueryPoint]:
    mode = str(source).strip().lower()
    scene_center = (
        compute_scene_center_from_road(dict(scene_cfg))
        if str(origin_mode).strip().lower() == "center"
        else np.zeros((3,), dtype=np.float32)
    )
    out: List[QueryPoint] = []
    next_id = 0
    for agent in _iter_scene_agents(scene_cfg):
        start = agent.get("start", {}) or {}
        end = agent.get("end", {}) or {}
        if mode in {"agent_start", "agent_start_goal"}:
            if "x" in start and "y" in start:
                sx = float(start["x"]) - float(scene_center[0])
                sy = float(start["y"]) - float(scene_center[1])
                if _in_bounds_xy(sx, sy, bounds_size_m):
                    out.append(QueryPoint(point_id=next_id, source="start", x_local_m=sx, y_local_m=sy))
                    next_id += 1
        if mode == "agent_start_goal":
            if "x" in end and "y" in end:
                ex = float(end["x"]) - float(scene_center[0])
                ey = float(end["y"]) - float(scene_center[1])
                if _in_bounds_xy(ex, ey, bounds_size_m):
                    out.append(QueryPoint(point_id=next_id, source="goal", x_local_m=ex, y_local_m=ey))
                    next_id += 1
    return out


def _subsample_query_points(
    points: Sequence[QueryPoint],
    *,
    max_points: int,
    rng: np.random.Generator,
) -> List[QueryPoint]:
    if max_points <= 0:
        return list(points)
    if len(points) <= max_points:
        return list(points)
    picked = rng.choice(len(points), size=int(max_points), replace=False)
    picked = sorted(int(i) for i in picked.tolist())
    return [points[i] for i in picked]


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float32), q))


def _summarize_numeric(values: Sequence[float]) -> Dict[str, float]:
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


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _generate_figures(
    *,
    out_dir: Path,
    global_radius_rows: Sequence[Mapping[str, Any]],
    per_map_radius_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"[coverage][warn] matplotlib unavailable; skipping figures ({exc})")
        return {}

    out_paths: Dict[str, str] = {}
    if not global_radius_rows:
        return out_paths

    radii = [_safe_float(r.get("radius_m", 0.0)) for r in global_radius_rows]
    any_edge_rate = [_safe_float(r.get("selected_has_any_edge_rate", 0.0)) for r in global_radius_rows]
    selected_edge_mean = [_safe_float(r.get("selected_edge_mean", 0.0)) for r in global_radius_rows]
    in_radius_edge_mean = [_safe_float(r.get("in_radius_edge_mean", 0.0)) for r in global_radius_rows]

    fig_global = out_dir / "coverage_global_trends.png"
    fig, ax_left = plt.subplots(figsize=(8.5, 5.0), dpi=150)
    ax_right = ax_left.twinx()
    line_l = ax_left.plot(
        radii,
        any_edge_rate,
        marker="o",
        linewidth=2.0,
        color="#0b84a5",
        label="selected_has_any_edge_rate",
    )[0]
    line_r1 = ax_right.plot(
        radii,
        selected_edge_mean,
        marker="s",
        linewidth=2.0,
        color="#f6c85f",
        label="selected_edge_mean",
    )[0]
    line_r2 = ax_right.plot(
        radii,
        in_radius_edge_mean,
        marker="^",
        linewidth=2.0,
        color="#6f4e7c",
        label="in_radius_edge_mean",
    )[0]
    ax_left.set_xlabel("Radius (m)")
    ax_left.set_ylabel("Selected has edge rate")
    ax_right.set_ylabel("Mean edge points")
    ax_left.set_ylim(0.0, 1.0)
    ax_left.grid(True, alpha=0.25)
    ax_left.set_title("Road-Edge Visibility vs Radius")
    lines = [line_l, line_r1, line_r2]
    ax_left.legend(lines, [l.get_label() for l in lines], loc="best")
    fig.tight_layout()
    fig.savefig(fig_global)
    plt.close(fig)
    out_paths["global_trends_png"] = str(fig_global)

    per_map_by_scene: Dict[str, Dict[float, Mapping[str, Any]]] = {}
    unique_radii = sorted({_safe_float(r.get("radius_m", 0.0)) for r in per_map_radius_rows})
    for row in per_map_radius_rows:
        scene_name = str(row.get("scene_json", "unknown_scene"))
        radius_m = _safe_float(row.get("radius_m", 0.0))
        per_map_by_scene.setdefault(scene_name, {})[radius_m] = row

    scene_items: List[Tuple[str, float]] = []
    for scene_name, radius_map in per_map_by_scene.items():
        vals = [
            _safe_float(radius_map.get(radius, {}).get("selected_has_any_edge_rate", 0.0))
            for radius in unique_radii
        ]
        mean_rate = float(sum(vals) / max(1, len(vals)))
        scene_items.append((scene_name, mean_rate))
    scene_items.sort(key=lambda x: x[1])
    ordered_scene_names = [x[0] for x in scene_items]

    if ordered_scene_names and unique_radii:
        heat = np.zeros((len(ordered_scene_names), len(unique_radii)), dtype=np.float32)
        for i, scene_name in enumerate(ordered_scene_names):
            radius_map = per_map_by_scene.get(scene_name, {})
            for j, radius in enumerate(unique_radii):
                row = radius_map.get(radius, {})
                heat[i, j] = _safe_float(row.get("selected_has_any_edge_rate", 0.0))

        fig_heatmap = out_dir / "coverage_per_map_edge_visibility_heatmap.png"
        fig, ax = plt.subplots(figsize=(10.0, max(4.0, 0.35 * len(ordered_scene_names))), dpi=150)
        im = ax.imshow(heat, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title("Selected Edge Visibility Rate by Map and Radius")
        ax.set_xlabel("Radius (m)")
        ax.set_ylabel("Scene JSON")
        ax.set_xticks(list(range(len(unique_radii))))
        ax.set_xticklabels([f"{r:.0f}" for r in unique_radii])
        ax.set_yticks(list(range(len(ordered_scene_names))))
        ax.set_yticklabels(ordered_scene_names, fontsize=8)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("selected_has_any_edge_rate")
        fig.tight_layout()
        fig.savefig(fig_heatmap)
        plt.close(fig)
        out_paths["per_map_edge_visibility_heatmap_png"] = str(fig_heatmap)

    if unique_radii:
        by_radius: Dict[float, List[float]] = {r: [] for r in unique_radii}
        for row in per_map_radius_rows:
            radius_m = _safe_float(row.get("radius_m", 0.0))
            if radius_m in by_radius:
                by_radius[radius_m].append(_safe_float(row.get("selected_has_any_edge_rate", 0.0)))
        box_vals = [by_radius[r] for r in unique_radii]
        if any(len(v) > 0 for v in box_vals):
            fig_box = out_dir / "coverage_per_map_rate_distribution_by_radius.png"
            fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)
            tick_labels = [f"{r:.0f}" for r in unique_radii]
            try:
                ax.boxplot(box_vals, tick_labels=tick_labels, showmeans=True)
            except TypeError:
                ax.boxplot(box_vals, labels=tick_labels, showmeans=True)
            ax.set_xlabel("Radius (m)")
            ax.set_ylabel("selected_has_any_edge_rate")
            ax.set_title("Per-Map Edge Visibility Distribution by Radius")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(fig_box)
            plt.close(fig)
            out_paths["per_map_rate_distribution_by_radius_png"] = str(fig_box)

    return out_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze road-point visibility coverage for a PPO training config. "
            "For each scene and sampled query point, sweep radius values and count road points."
        )
    )
    parser.add_argument(
        "--ppo-config",
        required=True,
        help="Path to PPO YAML config containing choco_config_path.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to runs/road_point_coverage/<ppo_config_stem>.",
    )
    parser.add_argument(
        "--radii-m",
        default="5,10,20,30,40,50",
        help="Comma-separated radius values in meters.",
    )
    parser.add_argument(
        "--sample-source",
        choices=["agent_start", "agent_start_goal"],
        default="agent_start",
        help="How to sample query points from scene JSON.",
    )
    parser.add_argument(
        "--max-points-per-map",
        type=int,
        default=256,
        help="Cap sampled query points per map. <=0 uses all points.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for per-map query-point subsampling.",
    )
    parser.add_argument(
        "--dedupe-scenes",
        action="store_true",
        help="Analyze each unique scene_json once, even if repeated in assignments.",
    )
    parser.add_argument(
        "--edge-types",
        default=None,
        help="Optional comma-separated edge types override (default from curriculum env settings).",
    )
    parser.add_argument(
        "--road-points-k-override",
        type=int,
        default=None,
        help="Optional override for env.road_points_k used in analysis.",
    )
    parser.add_argument(
        "--road-points-mode-override",
        default=None,
        help="Optional override for env.road_points_mode used in analysis (knn or road-running).",
    )
    parser.add_argument(
        "--road-points-k-grid",
        default=None,
        help="Optional comma-separated K sweep values for recommendation search.",
    )
    parser.add_argument(
        "--road-points-mode-grid",
        default=None,
        help="Optional comma-separated mode sweep values for recommendation search (knn, road-running).",
    )
    parser.add_argument(
        "--recommend-top-n",
        type=int,
        default=10,
        help="How many top-ranked (mode,k,radius) candidates to include in reports.",
    )
    parser.add_argument(
        "--recommend-fill-ratio-min",
        type=float,
        default=0.8,
        help="Minimum selected_total_mean/k ratio target before underfill penalty is applied.",
    )
    parser.add_argument(
        "--recommend-weight-edge-rate",
        type=float,
        default=1.0,
        help="Score weight for selected_has_any_edge_rate.",
    )
    parser.add_argument(
        "--recommend-weight-edge-mean",
        type=float,
        default=0.02,
        help="Score weight for selected_edge_mean.",
    )
    parser.add_argument(
        "--recommend-weight-underfill",
        type=float,
        default=1.0,
        help="Score penalty weight for underfill max(0, fill_ratio_min - selected_total_mean/k).",
    )
    return parser.parse_args()


def _parse_int_list(raw: str | None) -> List[int]:
    if raw is None:
        return []
    vals: List[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return vals


def _parse_mode_list(raw: str | None) -> List[str]:
    if raw is None:
        return []
    vals: List[str] = []
    for part in str(raw).split(","):
        mode = part.strip().lower().replace("_", "-")
        if not mode:
            continue
        vals.append(mode)
    return vals


def run(args: argparse.Namespace) -> None:
    ppo_path = Path(args.ppo_config).expanduser().resolve()
    ppo_cfg = _load_yaml(ppo_path)
    curriculum_path = _resolve_curriculum_path(ppo_path, ppo_cfg)
    curriculum_cfg = _load_yaml(curriculum_path)
    specs = prepare_stage_world_specs(curriculum_cfg)

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (Path("runs/road_point_coverage") / ppo_path.stem).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    world_cfg = (curriculum_cfg.get("world", {}) or {})
    road_cfg = (curriculum_cfg.get("road", {}) or {})
    env_cfg = (curriculum_cfg.get("env", {}) or {})

    bounds_size_m = float(world_cfg.get("bounds_size_m", 200.0))
    origin_mode = str(world_cfg.get("origin_mode", "center"))
    jump_break_m = float(road_cfg.get("jump_break_m", 3.0))
    flatten_road_z = bool(road_cfg.get("flatten_road_z", True))
    road_z_m = float(road_cfg.get("road_z_m", 0.0))

    road_points_enable = bool(env_cfg.get("road_points_enable", False))
    road_points_k = int(env_cfg.get("road_points_k", 0))
    road_points_mode = str(env_cfg.get("road_points_mode", "knn")).strip().lower().replace("_", "-")
    if args.road_points_k_override is not None:
        road_points_k = int(args.road_points_k_override)
    if args.road_points_mode_override is not None:
        road_points_mode = str(args.road_points_mode_override).strip().lower().replace("_", "-")
    if road_points_mode not in {"knn", "road-running"}:
        print(
            f"[coverage][warn] unknown road_points_mode='{road_points_mode}', "
            "falling back to knn"
        )
        road_points_mode = "knn"
    primary_mode = str(road_points_mode)
    primary_k = int(road_points_k)

    k_from_grid = _parse_int_list(args.road_points_k_grid)
    if k_from_grid:
        k_candidates = sorted({int(v) for v in k_from_grid if int(v) > 0})
    else:
        k_candidates = [int(primary_k)] if int(primary_k) > 0 else [1]

    mode_from_grid = _parse_mode_list(args.road_points_mode_grid)
    if mode_from_grid:
        mode_candidates = list(mode_from_grid)
    else:
        mode_candidates = [primary_mode]
    mode_candidates = [m for m in mode_candidates if m in {"knn", "road-running"}]
    if not mode_candidates:
        mode_candidates = [primary_mode]
    mode_candidates = list(dict.fromkeys(mode_candidates))
    combo_list: List[Tuple[str, int]] = []
    for mode in mode_candidates:
        for kval in k_candidates:
            combo_list.append((str(mode), int(kval)))
    report_mode = primary_mode
    report_k = int(primary_k)
    if (report_mode, report_k) not in combo_list and combo_list:
        report_mode, report_k = combo_list[0]
        print(
            "[coverage][warn] base config combo not in sweep grid; "
            f"reporting primary summaries for first sweep combo mode={report_mode} k={report_k}"
        )

    print(
        "[coverage] combo_sweep "
        f"modes={mode_candidates} "
        f"k_values={k_candidates} "
        f"total={len(combo_list)}"
    )

    if args.edge_types is not None:
        edge_types = set(_parse_int_list(args.edge_types))
    else:
        edge_types = set(int(v) for v in (env_cfg.get("road_contact_done_types", []) or []))
        if not edge_types:
            edge_types = set(int(v) for v in (env_cfg.get("geom_road_edge_types", [15, 16]) or []))
    lane_types = set(int(v) for v in (env_cfg.get("geom_lane_types", [1, 2]) or []))

    radii = _parse_radii(args.radii_m)
    per_point_rows: List[Dict[str, Any]] = []
    seen_scene_paths: set[str] = set()

    analyzed_maps = 0
    skipped_maps = 0
    total_query_points = 0
    rng_root = np.random.default_rng(int(args.seed))

    for spec in specs:
        scene_path = Path(spec.scene_json_path).resolve()
        scene_key = str(scene_path)
        if args.dedupe_scenes and scene_key in seen_scene_paths:
            skipped_maps += 1
            continue
        seen_scene_paths.add(scene_key)

        scene_cfg = json.loads(scene_path.read_text(encoding="utf-8"))
        road_points_xy, road_point_types = _build_world_road_points(
            scene_cfg,
            bounds_size_m=bounds_size_m,
            origin_mode=origin_mode,
            jump_break_m=jump_break_m,
            flatten_road_z=flatten_road_z,
            road_z_m=road_z_m,
        )
        query_points = _collect_query_points(
            scene_cfg,
            bounds_size_m=bounds_size_m,
            origin_mode=origin_mode,
            source=str(args.sample_source),
        )
        map_seed = int(rng_root.integers(0, 2**31 - 1))
        map_rng = np.random.default_rng(map_seed)
        sampled_points = _subsample_query_points(
            query_points,
            max_points=int(args.max_points_per_map),
            rng=map_rng,
        )
        total_query_points += len(sampled_points)
        analyzed_maps += 1

        print(
            "[coverage] map "
            f"world={int(spec.world_index):03d} "
            f"scene={scene_path.name} "
            f"road_points={int(road_points_xy.shape[0])} "
            f"query_points={len(sampled_points)}"
        )

        if road_points_xy.shape[0] == 0 or len(sampled_points) == 0:
            continue

        for qp in sampled_points:
            dx = road_points_xy[:, 0] - float(qp.x_local_m)
            dy = road_points_xy[:, 1] - float(qp.y_local_m)
            dist2 = dx * dx + dy * dy
            for radius_m in radii:
                r2 = float(radius_m * radius_m)
                in_radius = np.where(dist2 <= r2)[0]
                in_total = int(in_radius.shape[0])
                if in_total > 0:
                    in_types = road_point_types[in_radius]
                    in_edge = int(np.isin(in_types, list(edge_types)).sum())
                    in_lane = int(np.isin(in_types, list(lane_types)).sum())
                else:
                    in_edge = 0
                    in_lane = 0

                knn_ordered = np.zeros((0,), dtype=np.int64)
                if in_total > 0:
                    knn_order = np.argsort(dist2[in_radius])
                    knn_ordered = in_radius[knn_order]
                for combo_mode, combo_k in combo_list:
                    if road_points_enable and int(combo_k) > 0 and in_total > 0:
                        if combo_mode == "knn":
                            selected = knn_ordered[: int(combo_k)]
                        else:
                            selected = in_radius[: int(combo_k)]
                    else:
                        selected = np.zeros((0,), dtype=np.int64)
                    sel_total = int(selected.shape[0])
                    if sel_total > 0:
                        sel_types = road_point_types[selected]
                        sel_edge = int(np.isin(sel_types, list(edge_types)).sum())
                        sel_lane = int(np.isin(sel_types, list(lane_types)).sum())
                    else:
                        sel_edge = 0
                        sel_lane = 0

                    per_point_rows.append(
                        {
                            "world_index": int(spec.world_index),
                            "scene_json": scene_path.name,
                            "point_id": int(qp.point_id),
                            "point_source": str(qp.source),
                            "point_x_local_m": float(qp.x_local_m),
                            "point_y_local_m": float(qp.y_local_m),
                            "radius_m": float(radius_m),
                            "road_points_mode": str(combo_mode),
                            "road_points_k": int(combo_k),
                            "in_radius_total_points": int(in_total),
                            "in_radius_edge_points": int(in_edge),
                            "in_radius_lane_points": int(in_lane),
                            "in_radius_edge_fraction": float(in_edge / in_total) if in_total > 0 else 0.0,
                            "selected_total_points": int(sel_total),
                            "selected_edge_points": int(sel_edge),
                            "selected_lane_points": int(sel_lane),
                            "selected_edge_fraction": float(sel_edge / sel_total) if sel_total > 0 else 0.0,
                            "selected_has_any_edge": int(sel_edge > 0),
                        }
                    )

    if not per_point_rows:
        raise RuntimeError("No coverage rows generated. Check config scenes and sample source.")

    map_radius_groups: Dict[Tuple[str, int, str, float], List[Dict[str, Any]]] = {}
    global_radius_groups: Dict[Tuple[str, int, float], List[Dict[str, Any]]] = {}
    for row in per_point_rows:
        key = (
            str(row["road_points_mode"]),
            int(row["road_points_k"]),
            str(row["scene_json"]),
            float(row["radius_m"]),
        )
        map_radius_groups.setdefault(key, []).append(row)
        gkey = (
            str(row["road_points_mode"]),
            int(row["road_points_k"]),
            float(row["radius_m"]),
        )
        global_radius_groups.setdefault(gkey, []).append(row)

    all_per_map_radius_rows: List[Dict[str, Any]] = []
    for (mode_name, k_val, scene_name, radius_m), rows in sorted(
        map_radius_groups.items(),
        key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3]),
    ):
        in_total_vals = [float(r["in_radius_total_points"]) for r in rows]
        in_edge_vals = [float(r["in_radius_edge_points"]) for r in rows]
        sel_total_vals = [float(r["selected_total_points"]) for r in rows]
        sel_edge_vals = [float(r["selected_edge_points"]) for r in rows]
        sel_any_edge_vals = [float(r["selected_has_any_edge"]) for r in rows]
        all_per_map_radius_rows.append(
            {
                "road_points_mode": str(mode_name),
                "road_points_k": int(k_val),
                "scene_json": scene_name,
                "radius_m": float(radius_m),
                "query_point_count": int(len(rows)),
                "in_radius_total_mean": _summarize_numeric(in_total_vals)["mean"],
                "in_radius_total_p10": _summarize_numeric(in_total_vals)["p10"],
                "in_radius_total_p50": _summarize_numeric(in_total_vals)["median"],
                "in_radius_total_p90": _summarize_numeric(in_total_vals)["p90"],
                "in_radius_edge_mean": _summarize_numeric(in_edge_vals)["mean"],
                "selected_total_mean": _summarize_numeric(sel_total_vals)["mean"],
                "selected_edge_mean": _summarize_numeric(sel_edge_vals)["mean"],
                "selected_has_any_edge_rate": float(sum(sel_any_edge_vals) / max(1, len(sel_any_edge_vals))),
            }
        )

    all_global_radius_rows: List[Dict[str, Any]] = []
    for (mode_name, k_val, radius_m), rows in sorted(
        global_radius_groups.items(),
        key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]),
    ):
        in_total_vals = [float(r["in_radius_total_points"]) for r in rows]
        in_edge_vals = [float(r["in_radius_edge_points"]) for r in rows]
        sel_total_vals = [float(r["selected_total_points"]) for r in rows]
        sel_edge_vals = [float(r["selected_edge_points"]) for r in rows]
        sel_any_edge_vals = [float(r["selected_has_any_edge"]) for r in rows]
        all_global_radius_rows.append(
            {
                "road_points_mode": str(mode_name),
                "road_points_k": int(k_val),
                "radius_m": float(radius_m),
                "query_point_count": int(len(rows)),
                "in_radius_total_mean": _summarize_numeric(in_total_vals)["mean"],
                "in_radius_total_p10": _summarize_numeric(in_total_vals)["p10"],
                "in_radius_total_p50": _summarize_numeric(in_total_vals)["median"],
                "in_radius_total_p90": _summarize_numeric(in_total_vals)["p90"],
                "in_radius_edge_mean": _summarize_numeric(in_edge_vals)["mean"],
                "selected_total_mean": _summarize_numeric(sel_total_vals)["mean"],
                "selected_edge_mean": _summarize_numeric(sel_edge_vals)["mean"],
                "selected_has_any_edge_rate": float(sum(sel_any_edge_vals) / max(1, len(sel_any_edge_vals))),
            }
        )

    primary_per_map_radius_rows = [
        row
        for row in all_per_map_radius_rows
        if str(row["road_points_mode"]) == report_mode and int(row["road_points_k"]) == int(report_k)
    ]
    primary_global_radius_rows = [
        row
        for row in all_global_radius_rows
        if str(row["road_points_mode"]) == report_mode and int(row["road_points_k"]) == int(report_k)
    ]

    ranking_rows: List[Dict[str, Any]] = []
    fill_ratio_min = float(args.recommend_fill_ratio_min)
    w_edge_rate = float(args.recommend_weight_edge_rate)
    w_edge_mean = float(args.recommend_weight_edge_mean)
    w_underfill = float(args.recommend_weight_underfill)
    for row in all_global_radius_rows:
        k_val = max(1, int(row["road_points_k"]))
        selected_total_mean = float(row["selected_total_mean"])
        fill_ratio = selected_total_mean / float(k_val)
        underfill_penalty = max(0.0, fill_ratio_min - fill_ratio)
        score = (
            w_edge_rate * float(row["selected_has_any_edge_rate"])
            + w_edge_mean * float(row["selected_edge_mean"])
            - w_underfill * float(underfill_penalty)
        )
        ranking_rows.append(
            {
                "road_points_mode": str(row["road_points_mode"]),
                "road_points_k": int(k_val),
                "radius_m": float(row["radius_m"]),
                "score": float(score),
                "fill_ratio": float(fill_ratio),
                "underfill_penalty": float(underfill_penalty),
                "selected_has_any_edge_rate": float(row["selected_has_any_edge_rate"]),
                "selected_edge_mean": float(row["selected_edge_mean"]),
                "selected_total_mean": float(selected_total_mean),
                "query_point_count": int(row["query_point_count"]),
            }
        )
    ranking_rows.sort(key=lambda r: float(r["score"]), reverse=True)
    top_n = max(1, int(args.recommend_top_n))
    top_recommendations = ranking_rows[:top_n]

    point_csv = out_dir / "coverage_per_point_radius.csv"
    map_csv = out_dir / "coverage_per_map_radius_summary.csv"
    global_csv = out_dir / "coverage_global_radius_summary.csv"
    map_all_csv = out_dir / "coverage_per_map_radius_summary_all_combos.csv"
    global_all_csv = out_dir / "coverage_global_radius_summary_all_combos.csv"
    ranking_csv = out_dir / "coverage_combo_ranking.csv"
    report_json = out_dir / "coverage_report.json"
    report_md = out_dir / "coverage_report.md"

    _write_csv(
        point_csv,
        per_point_rows,
        fieldnames=[
            "world_index",
            "scene_json",
            "point_id",
            "point_source",
            "point_x_local_m",
            "point_y_local_m",
            "radius_m",
            "road_points_mode",
            "road_points_k",
            "in_radius_total_points",
            "in_radius_edge_points",
            "in_radius_lane_points",
            "in_radius_edge_fraction",
            "selected_total_points",
            "selected_edge_points",
            "selected_lane_points",
            "selected_edge_fraction",
            "selected_has_any_edge",
        ],
    )
    _write_csv(
        map_csv,
        primary_per_map_radius_rows,
        fieldnames=[
            "road_points_mode",
            "road_points_k",
            "scene_json",
            "radius_m",
            "query_point_count",
            "in_radius_total_mean",
            "in_radius_total_p10",
            "in_radius_total_p50",
            "in_radius_total_p90",
            "in_radius_edge_mean",
            "selected_total_mean",
            "selected_edge_mean",
            "selected_has_any_edge_rate",
        ],
    )
    _write_csv(
        global_csv,
        primary_global_radius_rows,
        fieldnames=[
            "road_points_mode",
            "road_points_k",
            "radius_m",
            "query_point_count",
            "in_radius_total_mean",
            "in_radius_total_p10",
            "in_radius_total_p50",
            "in_radius_total_p90",
            "in_radius_edge_mean",
            "selected_total_mean",
            "selected_edge_mean",
            "selected_has_any_edge_rate",
        ],
    )
    _write_csv(
        map_all_csv,
        all_per_map_radius_rows,
        fieldnames=[
            "road_points_mode",
            "road_points_k",
            "scene_json",
            "radius_m",
            "query_point_count",
            "in_radius_total_mean",
            "in_radius_total_p10",
            "in_radius_total_p50",
            "in_radius_total_p90",
            "in_radius_edge_mean",
            "selected_total_mean",
            "selected_edge_mean",
            "selected_has_any_edge_rate",
        ],
    )
    _write_csv(
        global_all_csv,
        all_global_radius_rows,
        fieldnames=[
            "road_points_mode",
            "road_points_k",
            "radius_m",
            "query_point_count",
            "in_radius_total_mean",
            "in_radius_total_p10",
            "in_radius_total_p50",
            "in_radius_total_p90",
            "in_radius_edge_mean",
            "selected_total_mean",
            "selected_edge_mean",
            "selected_has_any_edge_rate",
        ],
    )
    _write_csv(
        ranking_csv,
        ranking_rows,
        fieldnames=[
            "road_points_mode",
            "road_points_k",
            "radius_m",
            "score",
            "fill_ratio",
            "underfill_penalty",
            "selected_has_any_edge_rate",
            "selected_edge_mean",
            "selected_total_mean",
            "query_point_count",
        ],
    )
    figure_paths = _generate_figures(
        out_dir=out_dir,
        global_radius_rows=primary_global_radius_rows,
        per_map_radius_rows=primary_per_map_radius_rows,
    )

    payload = {
        "input": {
            "ppo_config": str(ppo_path),
            "curriculum_config": str(curriculum_path),
            "out_dir": str(out_dir),
            "radii_m": [float(v) for v in radii],
            "sample_source": str(args.sample_source),
            "max_points_per_map": int(args.max_points_per_map),
            "seed": int(args.seed),
            "dedupe_scenes": bool(args.dedupe_scenes),
            "bounds_size_m": float(bounds_size_m),
            "origin_mode": str(origin_mode),
            "jump_break_m": float(jump_break_m),
            "road_points_enable": bool(road_points_enable),
            "road_points_k": int(report_k),
            "road_points_mode": str(report_mode),
            "base_config_road_points_k": int(primary_k),
            "base_config_road_points_mode": str(primary_mode),
            "road_points_k_override": (
                int(args.road_points_k_override) if args.road_points_k_override is not None else None
            ),
            "road_points_mode_override": (
                str(args.road_points_mode_override)
                if args.road_points_mode_override is not None
                else None
            ),
            "road_points_k_grid": [int(v) for v in k_candidates],
            "road_points_mode_grid": [str(v) for v in mode_candidates],
            "edge_types": sorted(int(v) for v in edge_types),
            "lane_types": sorted(int(v) for v in lane_types),
        },
        "stats": {
            "maps_analyzed": int(analyzed_maps),
            "maps_skipped_due_to_dedupe": int(skipped_maps),
            "query_points_total": int(total_query_points),
            "rows_total": int(len(per_point_rows)),
        },
        "global_radius_summary": primary_global_radius_rows,
        "per_map_radius_summary": primary_per_map_radius_rows,
        "global_radius_summary_all_combos": all_global_radius_rows,
        "top_recommendations": top_recommendations,
        "outputs": {
            "coverage_per_point_radius_csv": str(point_csv),
            "coverage_per_map_radius_summary_csv": str(map_csv),
            "coverage_global_radius_summary_csv": str(global_csv),
            "coverage_per_map_radius_summary_all_combos_csv": str(map_all_csv),
            "coverage_global_radius_summary_all_combos_csv": str(global_all_csv),
            "coverage_combo_ranking_csv": str(ranking_csv),
            "figures": dict(figure_paths),
        },
    }
    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Road Point Coverage Report")
    lines.append("")
    lines.append(f"- PPO config: `{ppo_path}`")
    lines.append(f"- Curriculum config: `{curriculum_path}`")
    lines.append(f"- Maps analyzed: **{analyzed_maps}**")
    lines.append(f"- Query points analyzed: **{total_query_points}**")
    lines.append(
        "- Observation selection mode: "
        f"`{report_mode}` (enabled={road_points_enable}, k={report_k})"
    )
    if report_mode != primary_mode or int(report_k) != int(primary_k):
        lines.append(
            "- Base config mode/k: "
            f"`{primary_mode}` / `{primary_k}`"
        )
    lines.append(
        "- Combo sweep: "
        f"modes=`{mode_candidates}` k_values=`{k_candidates}` total=`{len(combo_list)}`"
    )
    lines.append(
        f"- Edge types: `{sorted(int(v) for v in edge_types)}` | "
        f"Lane types: `{sorted(int(v) for v in lane_types)}`"
    )
    lines.append("")
    lines.append("## Global Radius Summary")
    lines.append("")
    lines.append("| radius_m | query_points | in_radius_total_mean | in_radius_edge_mean | selected_total_mean | selected_edge_mean | selected_has_any_edge_rate |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in primary_global_radius_rows:
        lines.append(
            "| "
            f"{row['radius_m']:.1f} | {row['query_point_count']} | "
            f"{row['in_radius_total_mean']:.2f} | {row['in_radius_edge_mean']:.2f} | "
            f"{row['selected_total_mean']:.2f} | {row['selected_edge_mean']:.2f} | "
            f"{row['selected_has_any_edge_rate']:.4f} |"
        )
    lines.append("")
    lines.append("## Top Recommendations")
    lines.append("")
    lines.append("| rank | mode | k | radius_m | score | edge_rate | edge_mean | fill_ratio |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for i, row in enumerate(top_recommendations, start=1):
        lines.append(
            "| "
            f"{i} | {row['road_points_mode']} | {int(row['road_points_k'])} | "
            f"{row['radius_m']:.1f} | {row['score']:.4f} | "
            f"{row['selected_has_any_edge_rate']:.4f} | {row['selected_edge_mean']:.2f} | "
            f"{row['fill_ratio']:.3f} |"
        )
    lines.append("")
    if figure_paths:
        lines.append("## Figures")
        lines.append("")
        for _, fig_path in sorted(figure_paths.items(), key=lambda kv: kv[0]):
            lines.append(f"- `{fig_path}`")
        lines.append("")
    else:
        lines.append("## Figures")
        lines.append("")
        lines.append("- Not generated (matplotlib unavailable).")
        lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- `{point_csv}`")
    lines.append(f"- `{map_csv}`")
    lines.append(f"- `{global_csv}`")
    lines.append(f"- `{map_all_csv}`")
    lines.append(f"- `{global_all_csv}`")
    lines.append(f"- `{ranking_csv}`")
    lines.append(f"- `{report_json}`")
    lines.append(f"- `{report_md}`")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[coverage] wrote {point_csv}")
    print(f"[coverage] wrote {map_csv}")
    print(f"[coverage] wrote {global_csv}")
    print(f"[coverage] wrote {map_all_csv}")
    print(f"[coverage] wrote {global_all_csv}")
    print(f"[coverage] wrote {ranking_csv}")
    print(f"[coverage] wrote {report_json}")
    print(f"[coverage] wrote {report_md}")
    for name, fig_path in sorted(figure_paths.items(), key=lambda kv: kv[0]):
        print(f"[coverage] wrote figure {name}={fig_path}")
    if top_recommendations:
        best = top_recommendations[0]
        print(
            "[coverage][recommend] "
            f"mode={best['road_points_mode']} "
            f"k={int(best['road_points_k'])} "
            f"radius_m={float(best['radius_m']):.1f} "
            f"score={float(best['score']):.4f}"
        )
        print("[coverage][recommend] yaml_patch:")
        print("env:")
        print(f"  road_points_mode: \"{best['road_points_mode']}\"")
        print(f"  road_points_k: {int(best['road_points_k'])}")
        print(f"  road_points_radius_m: {float(best['radius_m']):.1f}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
