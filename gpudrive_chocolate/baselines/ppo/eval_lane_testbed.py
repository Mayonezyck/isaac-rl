from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from copy import deepcopy
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Sequence

import numpy as np
import yaml
from box import Box


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic policy testbed. By default it samples start/goal pairs from lane "
            "centerlines and builds a derived world. Use --real_start_end to evaluate with "
            "original scene starts/goals (training-style spawn)."
        )
    )
    parser.add_argument(
        "--ppo-config",
        default=None,
        help=(
            "Optional PPO experiment YAML (for reward-wrapper settings). "
            "If omitted, default wrapper weights are used."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to PPO checkpoint (.zip).",
    )
    parser.add_argument(
        "--world-config",
        required=True,
        help="Path to curriculum YAML used as world/base environment template.",
    )
    parser.add_argument(
        "--assignment-index",
        type=int,
        default=0,
        help="Assignment index inside world-config when world.assignments exists.",
    )
    parser.add_argument(
        "--world-count",
        type=int,
        default=1,
        help="Number of worlds to evaluate in parallel; summary rates are aggregated over all worlds.",
    )
    parser.add_argument(
        "--scene-json",
        default=None,
        help="Optional scene_json override (name or absolute path).",
    )
    parser.add_argument(
        "--num-vehicles",
        type=int,
        default=24,
        help=(
            "Number of controllable vehicles to spawn in sampled mode (default). "
            "Ignored with --real_start_end."
        ),
    )
    parser.add_argument(
        "--real_start_end",
        action="store_true",
        help=(
            "Use original scene starts/goals and assignment-style spawn (as in training) "
            "instead of lane-center sampling."
        ),
    )
    parser.add_argument(
        "--invincible",
        action="store_true",
        help=(
            "Eval-only mode: disable vehicle-collision and road-edge done conditions, "
            "while still counting collision/road-edge contact in summary metrics."
        ),
    )
    parser.add_argument(
        "--lane-types",
        default="1,2",
        help="Comma-separated road polyline types treated as lane centerlines (default: 1,2).",
    )
    parser.add_argument(
        "--min-travel-distance-m",
        type=float,
        default=20.0,
        help="Minimum lane-center trace-back distance between sampled goal and start.",
    )
    parser.add_argument(
        "--max-travel-distance-m",
        type=float,
        default=60.0,
        help="Maximum lane-center trace-back distance between sampled goal and start.",
    )
    parser.add_argument(
        "--min-start-gap-m",
        type=float,
        default=8.0,
        help="Minimum XY separation between sampled starts.",
    )
    parser.add_argument(
        "--min-goal-gap-m",
        type=float,
        default=6.0,
        help="Minimum XY separation between sampled goals.",
    )
    parser.add_argument(
        "--endpoint-margin-m",
        type=float,
        default=2.0,
        help="Margin from polyline endpoints when sampling start/goal arc-length.",
    )
    parser.add_argument(
        "--min-polyline-length-m",
        type=float,
        default=40.0,
        help="Minimum candidate lane polyline length for sampling.",
    )
    parser.add_argument(
        "--max-segment-gap-m",
        type=float,
        default=8.0,
        help=(
            "Maximum allowed gap between consecutive points on a lane polyline. "
            "Polylines with larger jumps are rejected as broken."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optional max deterministic rollout steps override. Defaults to env.max_steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic evaluation and lane sampling.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, for example cpu, cuda:0, cuda:1.",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/eval_lane_testbed",
        help="Output directory for derived scene/curriculum and metrics report.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run with Isaac GUI enabled.",
    )
    parser.add_argument(
        "--gui-step-delay-sec",
        type=float,
        default=0.0,
        help="Optional sleep per step when --gui is enabled.",
    )
    parser.add_argument(
        "--ttc-overlay",
        action="store_true",
        help="Enable per-vehicle TTC labels next to vehicles (GUI mode).",
    )
    parser.add_argument(
        "--ttc-overlay-char-height-m",
        type=float,
        default=1.5,
        help="Height of TTC overlay glyphs in meters.",
    )
    parser.add_argument(
        "--ttc-overlay-z-offset-m",
        type=float,
        default=2.2,
        help="Vertical offset above vehicle center for TTC labels.",
    )
    parser.add_argument(
        "--ttc-overlay-y-offset-m",
        type=float,
        default=1.2,
        help="Lateral offset in world Y for TTC labels.",
    )
    parser.add_argument(
        "--ttc-overlay-max-display-s",
        type=float,
        default=9.9,
        help="Maximum TTC displayed on overlay (values above clamp to this number).",
    )
    parser.add_argument(
        "--ttc-overlay-source",
        type=str,
        default="vehicle",
        choices=["vehicle", "forbidden_road", "min"],
        help=(
            "TTC source for overlay: "
            "'vehicle' (vehicle-vehicle), "
            "'forbidden_road' (vehicle-forbidden-road), "
            "'min' (minimum of both)."
        ),
    )
    parser.add_argument(
        "--ttc-radius-overlay",
        action="store_true",
        help="Enable TTC collision-radius disk overlay per vehicle (GUI mode).",
    )
    parser.add_argument(
        "--ttc-radius-height-m",
        type=float,
        default=0.08,
        help="Height/thickness in meters for TTC radius disk overlay.",
    )
    parser.add_argument(
        "--ttc-radius-z-offset-m",
        type=float,
        default=0.03,
        help="Vertical offset above vehicle base for TTC radius disk overlay.",
    )
    parser.add_argument(
        "--ttc-radius-opacity",
        type=float,
        default=0.20,
        help="Display opacity [0,1] for TTC radius disk overlay.",
    )
    parser.add_argument(
        "--distant-light",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a distant light at startup (useful for GUI visibility).",
    )
    parser.add_argument(
        "--distant-light-intensity",
        type=float,
        default=7000.0,
        help="Intensity for the startup distant light.",
    )
    return parser.parse_args()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return int(default)


def parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(_safe_int(part))
    if not out:
        raise ValueError(f"Expected non-empty comma-separated integer list, got: {text!r}")
    # preserve order, remove duplicates
    seen = set()
    unique: List[int] = []
    for value in out:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


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


def load_yaml_mapping(path: str | Path) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {p}")
    return data


def load_box(path: str | Path) -> Box:
    return Box(load_yaml_mapping(path))


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _checkpoint_steps(path: Path) -> int:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    if not match:
        return -1
    return int(match.group(1))


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
            print(f"[testbed] resolved checkpoint {checkpoint_path} -> {resolved}")
            return resolved.resolve()

    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def resolve_scene_assignment(
    world_cfg: Dict[str, Any],
    *,
    scene_json_override: str | None,
    assignment_index: int,
) -> tuple[Path, Dict[str, Any], Dict[str, Any]]:
    from src.trfc import resolve_scene_json_path

    io_cfg = world_cfg.get("io", {}) or {}
    scene_json_dir = io_cfg.get("scene_json_dir", None)
    if scene_json_dir is None:
        raise ValueError("world-config is missing io.scene_json_dir")

    assignments = list((world_cfg.get("world", {}) or {}).get("assignments", []) or [])
    if assignments:
        idx = int(assignment_index)
        if idx < 0 or idx >= len(assignments):
            raise IndexError(
                f"assignment-index={idx} out of range for {len(assignments)} assignments"
            )
        base_assignment = dict(assignments[idx])
        raw_scene_json = scene_json_override or base_assignment.get("scene_json", None)
        if raw_scene_json is None:
            raise ValueError("assignment does not contain scene_json")
        scene_path = resolve_scene_json_path(scene_json_dir, raw_scene_json)
        friction = dict(base_assignment.get("friction", {}) or {})
        return scene_path, friction, base_assignment

    scene_jsons = list((io_cfg.get("scene_jsons", []) or []))
    if scene_json_override is not None:
        scene_path = resolve_scene_json_path(scene_json_dir, scene_json_override)
        return scene_path, {}, {"scene_json": str(scene_path)}

    if scene_jsons:
        idx = int(assignment_index)
        if idx < 0 or idx >= len(scene_jsons):
            raise IndexError(f"assignment-index={idx} out of range for io.scene_jsons[{len(scene_jsons)}]")
        scene_path = resolve_scene_json_path(scene_json_dir, scene_jsons[idx])
        return scene_path, {}, {"scene_json": str(scene_path)}

    raise ValueError(
        "Could not resolve scene_json from world-config. "
        "Provide world.assignments, io.scene_jsons, or --scene-json."
    )


def build_eval_curriculum(
    base_world_cfg: Dict[str, Any],
    *,
    assignment_entries: List[Dict[str, Any]],
    max_agents_override: int | None,
    gui: bool,
    steps_override: int | None,
    invincible: bool,
) -> Dict[str, Any]:
    cfg = deepcopy(base_world_cfg)

    world = dict(cfg.get("world", {}) or {})
    world_count = max(1, int(len(assignment_entries)))
    world["world_count"] = int(world_count)
    grid_cols = max(1, int(math.ceil(math.sqrt(float(world_count)))))
    rows = max(1, int(math.ceil(float(world_count) / float(grid_cols))))
    world["grid_cols"] = int(grid_cols)
    world["rows"] = int(rows)
    if max_agents_override is not None:
        world["max_agents_per_world"] = max(
            int(world.get("max_agents_per_world", 0)),
            int(max_agents_override),
        )
    world["assignments"] = [dict(a) for a in assignment_entries]
    cfg["world"] = world

    app = dict(cfg.get("app", {}) or {})
    app["headless"] = not bool(gui)
    cfg["app"] = app

    env = dict(cfg.get("env", {}) or {})
    env["render"] = bool(gui)
    env["auto_reset_done"] = False
    env["auto_reset_timeout"] = False
    env["respawn_on_reset"] = False
    env["clear_on_done"] = True
    env["hard_remove_done_agents"] = True
    env["verbose"] = False
    if bool(invincible):
        env["vehicle_contact_done"] = False
        env["road_contact_done_types"] = []
    if steps_override is not None:
        env["max_steps"] = int(steps_override)
    cfg["env"] = env

    return cfg


def write_rollout_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "new_success_count",
                    "vehicle_contact_done_count",
                    "road_contact_done_count",
                    "below_min_z_count",
                    "goal_rate_step",
                    "vehicle_contact_done_rate_step",
                    "road_contact_done_rate_step",
                    "collision_rate_step",
                    "vehicle_collision_rate_step_per_controlled",
                ]
            )
        return

    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


_SEVEN_SEGMENT_MAP = {
    "0": "abcedf",
    "1": "bc",
    "2": "abdeg",
    "3": "abcdg",
    "4": "bcfg",
    "5": "acdfg",
    "6": "acdefg",
    "7": "abc",
    "8": "abcdefg",
    "9": "abcdfg",
    ".": ".",
    "-": "g",
}


def _display_ttc_value(ttc_s: float | None, max_display_s: float) -> float:
    max_display = max(0.1, float(max_display_s))
    if ttc_s is None or (not np.isfinite(float(ttc_s))):
        return float(max_display)
    return float(max(0.0, min(float(ttc_s), max_display)))


def _format_ttc_text(ttc_s: float | None, max_display_s: float) -> str:
    return f"{_display_ttc_value(ttc_s, max_display_s):.1f}"


def _ttc_color_rgb(ttc_s: float | None, max_display_s: float) -> tuple[float, float, float]:
    value = _display_ttc_value(ttc_s, max_display_s)
    denom = max(0.1, float(max_display_s))
    ratio = max(0.0, min(1.0, float(value) / denom))
    # small TTC -> red, large TTC -> blue-ish
    r = float(1.0 - ratio)
    g = float(0.05 + 0.35 * ratio)
    b = float(0.05 + 0.95 * ratio)
    return (r, g, b)


def _compute_min_ttc_by_token(
    choco_env,
    *,
    states_by_world: Dict[int, List[Dict[str, Any]]] | None = None,
) -> tuple[Dict[tuple[int, int], float | None], Dict[tuple[int, int], float]]:
    if states_by_world is None:
        states_by_world = choco_env._collect_all_vehicle_states()
    use_vehicle_size = bool(getattr(choco_env, "ttc_use_vehicle_size", True))
    excluded_tokens = set(getattr(choco_env, "_pending_respawns", {}).keys()) | set(
        getattr(choco_env, "_quarantined_tokens", set())
    )

    out_ttc: Dict[tuple[int, int], float | None] = {}
    out_radius: Dict[tuple[int, int], float] = {}
    for world_idx, states in states_by_world.items():
        del world_idx
        for ego in states:
            token = ego.get("token", None)
            if token is None:
                continue
            token = (int(token[0]), int(token[1]))
            ego_r = float(ego.get("radius_u", 0.0)) if use_vehicle_size else 0.0
            out_radius[token] = float(max(0.0, ego_r))
            if token in excluded_tokens:
                out_ttc[token] = None
                continue

            ex, ey = ego["pos"]
            evx, evy = ego["vel"]
            min_ttc = None

            for other in states:
                other_token = other.get("token", None)
                if other_token is not None:
                    other_token = (int(other_token[0]), int(other_token[1]))
                if other_token == token:
                    continue
                if other_token in excluded_tokens:
                    continue

                ox, oy = other["pos"]
                ovx, ovy = other["vel"]
                rx = ox - ex
                ry = oy - ey
                rvx = ovx - evx
                rvy = ovy - evy
                v2 = rvx * rvx + rvy * rvy
                if v2 < 1e-6:
                    continue
                rdotv = rx * rvx + ry * rvy
                r2 = rx * rx + ry * ry
                other_r = float(other.get("radius_u", 0.0)) if use_vehicle_size else 0.0
                combined_r = max(0.0, ego_r + other_r)

                ttc = None
                if r2 <= combined_r * combined_r:
                    ttc = 0.0
                else:
                    a = v2
                    b = 2.0 * rdotv
                    c = r2 - combined_r * combined_r
                    disc = b * b - 4.0 * a * c
                    if disc >= 0.0:
                        sqrt_disc = float(np.sqrt(disc))
                        t_enter = (-b - sqrt_disc) / (2.0 * a)
                        t_exit = (-b + sqrt_disc) / (2.0 * a)
                        if t_exit >= 0.0:
                            ttc = max(0.0, float(t_enter))

                if ttc is None:
                    if rdotv >= 0.0:
                        continue
                    dist = float(np.sqrt(max(r2, 1e-9)))
                    closing_speed = -rdotv / max(dist, 1e-6)
                    if closing_speed <= 1e-6:
                        continue
                    clearance = max(0.0, dist - combined_r)
                    ttc = clearance / closing_speed

                if min_ttc is None or float(ttc) < float(min_ttc):
                    min_ttc = float(ttc)

            out_ttc[token] = min_ttc
    return out_ttc, out_radius


def _forbidden_road_types(choco_env) -> List[int]:
    road_done = list(getattr(choco_env, "road_contact_done_types", set()) or [])
    if road_done:
        return sorted(int(v) for v in road_done)
    geom_edge = list(getattr(choco_env, "geom_road_edge_types", set()) or [])
    return sorted(int(v) for v in geom_edge)


def _compute_min_road_edge_ttc_by_token(
    choco_env,
    *,
    states_by_world: Dict[int, List[Dict[str, Any]]] | None = None,
) -> tuple[Dict[tuple[int, int], float | None], Dict[tuple[int, int], float]]:
    if states_by_world is None:
        states_by_world = choco_env._collect_all_vehicle_states()

    forbidden_types = _forbidden_road_types(choco_env)
    use_vehicle_size = bool(getattr(choco_env, "ttc_use_vehicle_size", True))
    mpu = max(float(getattr(choco_env, "_mpu", 1.0)), 1e-6)
    radius_m = getattr(choco_env, "road_edge_ttc_radius_m", None)
    if radius_m is None:
        radius_m = float(getattr(choco_env, "road_points_radius_m", 50.0))
    radius_m = max(0.0, float(radius_m))
    radius2_m = float(radius_m * radius_m)

    out_ttc: Dict[tuple[int, int], float | None] = {}
    out_radius: Dict[tuple[int, int], float] = {}
    if not forbidden_types or radius_m <= 0.0:
        for states in states_by_world.values():
            for ego in states:
                token = ego.get("token", None)
                if token is None:
                    continue
                token = (int(token[0]), int(token[1]))
                ego_r_u = float(ego.get("radius_u", 0.0)) if use_vehicle_size else 0.0
                out_radius[token] = float(max(0.0, ego_r_u))
                out_ttc[token] = None
        return out_ttc, out_radius

    forbidden_types_np = np.asarray(forbidden_types, dtype=np.int32)
    for world_idx, states in states_by_world.items():
        geom = choco_env._get_world_road_geometry(int(world_idx))
        if geom is None:
            for ego in states:
                token = ego.get("token", None)
                if token is None:
                    continue
                token = (int(token[0]), int(token[1]))
                ego_r_u = float(ego.get("radius_u", 0.0)) if use_vehicle_size else 0.0
                out_radius[token] = float(max(0.0, ego_r_u))
                out_ttc[token] = None
            continue

        points_xy_m = np.asarray(geom.get("points_xy_m"), dtype=np.float32)
        types = np.asarray(geom.get("types"), dtype=np.int32)
        if points_xy_m.ndim != 2 or points_xy_m.shape[1] < 2 or types.ndim != 1:
            for ego in states:
                token = ego.get("token", None)
                if token is None:
                    continue
                token = (int(token[0]), int(token[1]))
                ego_r_u = float(ego.get("radius_u", 0.0)) if use_vehicle_size else 0.0
                out_radius[token] = float(max(0.0, ego_r_u))
                out_ttc[token] = None
            continue

        forbidden_mask = np.isin(types, forbidden_types_np)
        forbidden_points = points_xy_m[forbidden_mask]
        if forbidden_points.shape[0] == 0:
            for ego in states:
                token = ego.get("token", None)
                if token is None:
                    continue
                token = (int(token[0]), int(token[1]))
                ego_r_u = float(ego.get("radius_u", 0.0)) if use_vehicle_size else 0.0
                out_radius[token] = float(max(0.0, ego_r_u))
                out_ttc[token] = None
            continue

        for ego in states:
            token = ego.get("token", None)
            if token is None:
                continue
            token = (int(token[0]), int(token[1]))
            ego_r_u = float(ego.get("radius_u", 0.0)) if use_vehicle_size else 0.0
            out_radius[token] = float(max(0.0, ego_r_u))

            ex_m = float(ego["pos"][0]) * mpu
            ey_m = float(ego["pos"][1]) * mpu
            evx_mps = float(ego["vel"][0]) * mpu
            evy_mps = float(ego["vel"][1]) * mpu
            ego_r_m = float(max(0.0, ego_r_u) * mpu)

            rel = forbidden_points - np.asarray([[ex_m, ey_m]], dtype=np.float32)
            dist2 = np.einsum("ij,ij->i", rel, rel)
            near_mask = dist2 <= radius2_m
            if not np.any(near_mask):
                out_ttc[token] = None
                continue

            rel_near = rel[near_mask]
            dist_near = np.sqrt(np.maximum(dist2[near_mask], 1e-12))
            min_ttc = None

            if ego_r_m > 0.0 and np.any(dist_near <= ego_r_m):
                min_ttc = 0.0

            dirs = rel_near / np.maximum(dist_near[:, None], 1e-6)
            closing_speed = dirs[:, 0] * evx_mps + dirs[:, 1] * evy_mps
            valid = closing_speed > 1e-6
            if np.any(valid):
                clearance = np.maximum(0.0, dist_near[valid] - ego_r_m)
                ttc_vals = clearance / np.maximum(closing_speed[valid], 1e-6)
                if ttc_vals.size > 0:
                    cand = float(np.min(ttc_vals))
                    if min_ttc is None or cand < min_ttc:
                        min_ttc = cand

            out_ttc[token] = min_ttc

    return out_ttc, out_radius


def _set_display_color(prim, rgb: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    try:
        UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set([Gf.Vec3f(*rgb)])
    except Exception:
        pass


def _set_display_opacity(prim, opacity: float) -> None:
    from pxr import UsdGeom

    try:
        alpha = float(np.clip(float(opacity), 0.0, 1.0))
        UsdGeom.Gprim(prim).CreateDisplayOpacityAttr().Set([alpha])
    except Exception:
        pass


def _spawn_ttc_radius_disk(
    stage,
    *,
    prim_path: str,
    center_u: tuple[float, float, float],
    radius_u: float,
    height_u: float,
    color_rgb: tuple[float, float, float],
    opacity: float,
) -> None:
    from pxr import Gf, UsdGeom

    r = max(1e-4, float(radius_u))
    h = max(1e-4, float(height_u))
    disk = UsdGeom.Cylinder.Define(stage, prim_path)
    disk.GetRadiusAttr().Set(r)
    disk.GetHeightAttr().Set(h)
    try:
        disk.CreateAxisAttr().Set(UsdGeom.Tokens.z)
    except Exception:
        pass
    xapi = UsdGeom.XformCommonAPI(disk)
    xapi.SetTranslate(Gf.Vec3d(float(center_u[0]), float(center_u[1]), float(center_u[2])))
    _set_display_color(disk.GetPrim(), color_rgb)
    _set_display_opacity(disk.GetPrim(), opacity)


def _spawn_seven_segment_text(
    stage,
    *,
    root_path: str,
    text: str,
    mpu: float,
    char_height_m: float,
    thickness_m: float = 0.06,
    color_rgb: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> None:
    from pxr import Gf, UsdGeom

    text = str(text)
    if not text:
        text = "-"

    char_h_u = max(0.05, float(char_height_m)) / max(float(mpu), 1e-6)
    char_w_u = 0.62 * char_h_u
    seg_t_u = 0.16 * char_h_u
    h_len_u = max(char_w_u - 1.5 * seg_t_u, 0.5 * seg_t_u)
    v_len_u = max(0.5 * char_h_u - 1.5 * seg_t_u, 0.5 * seg_t_u)
    spacing_u = 0.25 * char_h_u
    z_th_u = max(0.02, float(thickness_m)) / max(float(mpu), 1e-6)

    total_w_u = max(0.0, len(text) * char_w_u + max(0, len(text) - 1) * spacing_u)
    board_pad_u = 0.20 * char_h_u
    board_w_u = total_w_u + 2.0 * board_pad_u
    board_h_u = char_h_u + 2.0 * board_pad_u

    root = UsdGeom.Xform.Define(stage, root_path)
    board = UsdGeom.Cube.Define(stage, f"{root_path}/Board")
    board.GetSizeAttr().Set(1.0)
    board_x = UsdGeom.XformCommonAPI(board)
    board_x.SetTranslate(Gf.Vec3d(0.0, 0.0, -0.5 * z_th_u))
    board_x.SetScale(Gf.Vec3f(board_w_u, board_h_u, 0.5 * z_th_u))
    _set_display_color(board.GetPrim(), (0.03, 0.03, 0.03))

    x_cursor_u = -0.5 * total_w_u + 0.5 * char_w_u
    y_top_u = +0.5 * char_h_u - 0.5 * seg_t_u
    y_mid_u = 0.0
    y_bot_u = -0.5 * char_h_u + 0.5 * seg_t_u
    y_up_u = 0.25 * char_h_u
    y_lo_u = -0.25 * char_h_u
    x_left_u = -0.5 * char_w_u + 0.5 * seg_t_u
    x_right_u = +0.5 * char_w_u - 0.5 * seg_t_u

    for idx, ch in enumerate(text):
        segments = _SEVEN_SEGMENT_MAP.get(ch, "")
        char_path = f"{root_path}/C{idx:02d}_{ord(ch):03d}"
        char_xf = UsdGeom.Xform.Define(stage, char_path)
        char_api = UsdGeom.XformCommonAPI(char_xf)
        char_api.SetTranslate(Gf.Vec3d(x_cursor_u, 0.0, 0.0))

        for seg_name in segments:
            if seg_name == ".":
                seg_x, seg_y, sx, sy, token = x_right_u, y_bot_u, seg_t_u, seg_t_u, "dot"
            elif seg_name == "a":
                seg_x, seg_y, sx, sy, token = 0.0, y_top_u, h_len_u, seg_t_u, "a"
            elif seg_name == "b":
                seg_x, seg_y, sx, sy, token = x_right_u, y_up_u, seg_t_u, v_len_u, "b"
            elif seg_name == "c":
                seg_x, seg_y, sx, sy, token = x_right_u, y_lo_u, seg_t_u, v_len_u, "c"
            elif seg_name == "d":
                seg_x, seg_y, sx, sy, token = 0.0, y_bot_u, h_len_u, seg_t_u, "d"
            elif seg_name == "e":
                seg_x, seg_y, sx, sy, token = x_left_u, y_lo_u, seg_t_u, v_len_u, "e"
            elif seg_name == "f":
                seg_x, seg_y, sx, sy, token = x_left_u, y_up_u, seg_t_u, v_len_u, "f"
            else:
                seg_x, seg_y, sx, sy, token = 0.0, y_mid_u, h_len_u, seg_t_u, "g"

            seg = UsdGeom.Cube.Define(stage, f"{char_path}/S_{token}")
            seg.GetSizeAttr().Set(1.0)
            seg_api = UsdGeom.XformCommonAPI(seg)
            seg_api.SetTranslate(Gf.Vec3d(seg_x, seg_y, 0.0))
            seg_api.SetScale(Gf.Vec3f(sx, sy, z_th_u))
            _set_display_color(seg.GetPrim(), color_rgb)

        x_cursor_u += char_w_u + spacing_u


def _ensure_distant_light(stage, *, path: str = "/World/__EvalDistantLight", intensity: float = 7000.0) -> None:
    from pxr import Gf, UsdGeom, UsdLux

    light = UsdLux.DistantLight.Define(stage, path)
    try:
        light.CreateIntensityAttr().Set(float(intensity))
    except Exception:
        pass
    try:
        light.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    except Exception:
        pass
    try:
        light.CreateAngleAttr().Set(0.7)
    except Exception:
        pass
    try:
        xapi = UsdGeom.XformCommonAPI(light)
        xapi.SetRotate(Gf.Vec3f(-52.0, 28.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    except Exception:
        pass


def _update_ttc_overlay(
    choco_env,
    flat_slot_keys: Sequence[object],
    *,
    root_path: str,
    overlay_source: str,
    show_text: bool,
    z_offset_m: float,
    y_offset_m: float,
    char_height_m: float,
    max_display_s: float,
    radius_overlay: bool = False,
    radius_height_m: float = 0.08,
    radius_z_offset_m: float = 0.03,
    radius_opacity: float = 0.20,
) -> None:
    from pxr import Gf, UsdGeom

    stage = choco_env.stage
    mpu = max(float(getattr(choco_env, "_mpu", 0.01)), 1e-6)

    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    for child in list(root.GetChildren()):
        stage.RemovePrim(child.GetPath())

    source = str(overlay_source).strip().lower()
    states_by_world = choco_env._collect_all_vehicle_states()
    veh_ttc_by_token, veh_radius_by_token = _compute_min_ttc_by_token(
        choco_env, states_by_world=states_by_world
    )
    if source == "vehicle":
        ttc_by_token = veh_ttc_by_token
        radius_by_token = veh_radius_by_token
    elif source == "forbidden_road":
        ttc_by_token, radius_by_token = _compute_min_road_edge_ttc_by_token(
            choco_env, states_by_world=states_by_world
        )
    elif source == "min":
        road_ttc_by_token, road_radius_by_token = _compute_min_road_edge_ttc_by_token(
            choco_env, states_by_world=states_by_world
        )
        all_tokens = set(veh_ttc_by_token.keys()) | set(road_ttc_by_token.keys())
        ttc_by_token = {}
        radius_by_token = {}
        for token in all_tokens:
            v = veh_ttc_by_token.get(token, None)
            r = road_ttc_by_token.get(token, None)
            if v is None:
                ttc_by_token[token] = r
            elif r is None:
                ttc_by_token[token] = v
            else:
                ttc_by_token[token] = min(float(v), float(r))
            radius_by_token[token] = max(
                float(veh_radius_by_token.get(token, 0.0)),
                float(road_radius_by_token.get(token, 0.0)),
            )
    else:
        raise ValueError(f"Unknown TTC overlay source: {overlay_source!r}")

    for key in flat_slot_keys:
        world_idx = int(getattr(key, "world_idx"))
        agent_id = int(getattr(key, "agent_id"))
        token = (world_idx, agent_id)

        h = choco_env.ctrl.get(world_idx, agent_id)
        pose = choco_env._get_agent_world_pose(h) if h is not None else None
        if pose is None:
            continue
        x_m, y_m, z_m, _ = pose
        raw_ttc = ttc_by_token.get(token, None)
        label_text = _format_ttc_text(raw_ttc, float(max_display_s))
        label_color = _ttc_color_rgb(raw_ttc, float(max_display_s))

        label_path = f"{root_path}/W{world_idx:03d}_A{agent_id:06d}"
        UsdGeom.Xform.Define(stage, label_path)
        if bool(show_text):
            text_anchor_path = f"{label_path}/TextAnchor"
            text_anchor = UsdGeom.Xform.Define(stage, text_anchor_path)
            text_anchor_api = UsdGeom.XformCommonAPI(text_anchor)
            text_anchor_api.SetTranslate(
                Gf.Vec3d(
                    float(x_m) / mpu,
                    float(y_m + float(y_offset_m)) / mpu,
                    float(z_m + float(z_offset_m)) / mpu,
                )
            )

            _spawn_seven_segment_text(
                stage,
                root_path=f"{text_anchor_path}/Text",
                text=label_text,
                mpu=mpu,
                char_height_m=float(char_height_m),
                thickness_m=0.06,
                color_rgb=label_color,
            )

        if bool(radius_overlay):
            radius_u = float(max(0.0, radius_by_token.get(token, 0.0)))
            if radius_u > 0.0:
                _spawn_ttc_radius_disk(
                    stage,
                    prim_path=f"{label_path}/RadiusDisk",
                    center_u=(
                        float(x_m) / mpu,
                        float(y_m) / mpu,
                        float(z_m + float(radius_z_offset_m)) / mpu,
                    ),
                    radius_u=radius_u,
                    height_u=float(radius_height_m) / mpu,
                    color_rgb=label_color,
                    opacity=float(radius_opacity),
                )


def run(args: argparse.Namespace) -> None:
    import torch
    from stable_baselines3 import PPO

    from gpudrive_chocolate.baselines.ppo.ppo_sb3 import build_resume_custom_objects
    from gpudrive_chocolate.env.sb3_wrapper import ChocolateSB3MultiAgentEnv
    from gpudrive_chocolate.networks.late_fusion_policy import LateFusionPolicy  # noqa: F401
    from src.trfc import resolve_scene_json_path
    from src.trfc.lane_center_sampler import (
        build_scene_with_sampled_agents,
        sample_lane_center_start_goal_pairs,
    )

    del LateFusionPolicy

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ppo_config:
        exp_cfg = load_box(args.ppo_config)
        ppo_config_path_for_report = str(Path(args.ppo_config).expanduser().resolve())
    else:
        exp_cfg = Box(
            {
                "choco_config_path": str(Path(args.world_config).expanduser().resolve()),
                "device": "cpu",
                "reward_type": "weighted_combination",
                "collision_weight": 0.0,
                "goal_achieved_weight": 0.0,
                "off_road_weight": 0.0,
                "log_distance_weight": 0.0,
            }
        )
        ppo_config_path_for_report = None
    world_cfg = load_yaml_mapping(args.world_config)
    checkpoint_path = ensure_checkpoint_exists(args.checkpoint, exp_cfg)

    device = normalize_device(args.device or getattr(exp_cfg, "device", "cpu"))
    exp_cfg.device = device
    if isinstance(device, str) and device.startswith("cuda"):
        torch.cuda.set_device(device)

    world_count = max(1, int(args.world_count))
    lane_types = parse_int_list(args.lane_types)
    bounds_size_m = float((world_cfg.get("world", {}) or {}).get("bounds_size_m", 200.0))
    origin_mode = str((world_cfg.get("world", {}) or {}).get("origin_mode", "center"))
    assignments_cfg = list((world_cfg.get("world", {}) or {}).get("assignments", []) or [])
    io_scene_jsons = list(((world_cfg.get("io", {}) or {}).get("scene_jsons", []) or []))

    source_world_entries: List[Dict[str, Any]] = []
    for wi in range(world_count):
        if args.scene_json is not None:
            world_assignment_index = int(args.assignment_index)
            scene_override = args.scene_json
        else:
            scene_override = None
            if assignments_cfg:
                world_assignment_index = (int(args.assignment_index) + int(wi)) % len(assignments_cfg)
            elif io_scene_jsons:
                world_assignment_index = (int(args.assignment_index) + int(wi)) % len(io_scene_jsons)
            else:
                world_assignment_index = int(args.assignment_index)
        world_scene_path, world_friction_cfg, world_base_assignment = resolve_scene_assignment(
            world_cfg,
            scene_json_override=scene_override,
            assignment_index=world_assignment_index,
        )
        source_world_entries.append(
            {
                "scene_path": world_scene_path,
                "friction_cfg": dict(world_friction_cfg),
                "base_assignment": dict(world_base_assignment),
                "assignment_index": int(world_assignment_index),
            }
        )

    scene_path = Path(source_world_entries[0]["scene_path"]).resolve()
    friction_cfg = dict(source_world_entries[0]["friction_cfg"])
    base_assignment = dict(source_world_entries[0]["base_assignment"])

    samples_by_world: List[list] = []
    sampled_scene_paths: List[Path] = []
    assignment_entries: List[Dict[str, Any]] = []
    max_agents_override: int | None = None
    if not bool(args.real_start_end):
        scene_cfg_cache: Dict[str, Dict[str, Any]] = {}
        for wi in range(world_count):
            source_entry = source_world_entries[wi]
            world_scene_path = Path(source_entry["scene_path"]).resolve()
            world_friction_cfg = dict(source_entry.get("friction_cfg", {}) or {})
            scene_key = str(world_scene_path)
            scene_cfg = scene_cfg_cache.get(scene_key, None)
            if scene_cfg is None:
                scene_cfg = json.loads(world_scene_path.read_text(encoding="utf-8"))
                scene_cfg_cache[scene_key] = scene_cfg
            world_seed = int(args.seed) + int(wi)
            world_samples = sample_lane_center_start_goal_pairs(
                scene_cfg,
                num_agents=int(args.num_vehicles),
                bounds_size_m=float(bounds_size_m),
                origin_mode=origin_mode,
                lane_types=lane_types,
                min_travel_distance_m=float(args.min_travel_distance_m),
                max_travel_distance_m=float(args.max_travel_distance_m),
                min_start_gap_m=float(args.min_start_gap_m),
                min_goal_gap_m=float(args.min_goal_gap_m),
                endpoint_margin_m=float(args.endpoint_margin_m),
                min_polyline_length_m=float(args.min_polyline_length_m),
                max_segment_gap_m=float(args.max_segment_gap_m),
                seed=int(world_seed),
            )
            sampled_scene_cfg = build_scene_with_sampled_agents(
                scene_cfg,
                world_samples,
                agent_id_start=10000,
                agent_type=1,
            )
            sampled_scene_path = out_dir / f"derived_lane_sampled_scene_w{wi:03d}.json"
            sampled_scene_path.write_text(json.dumps(sampled_scene_cfg, indent=2), encoding="utf-8")
            sampled_scene_paths.append(sampled_scene_path)
            samples_by_world.append(world_samples)

            assignment_entry: Dict[str, Any] = {
                "scene_json": str(sampled_scene_path),
                "source_scene_json": str(world_scene_path),
            }
            if world_friction_cfg:
                assignment_entry["friction"] = dict(world_friction_cfg)
            assignment_entries.append(assignment_entry)
        max_agents_override = int(args.num_vehicles)
    else:
        print(
            "[testbed] real_start_end mode enabled: using original scene starts/goals "
            "and training-style assignment spawn."
        )
        if int(args.num_vehicles) > 0:
            print(
                "[testbed] note: --num-vehicles is ignored in --real_start_end mode."
            )
        assignments_cfg = list((world_cfg.get("world", {}) or {}).get("assignments", []) or [])
        if args.scene_json is not None or not assignments_cfg:
            for _ in range(world_count):
                assignment_entry: Dict[str, Any] = dict(base_assignment or {})
                assignment_entry["scene_json"] = str(scene_path)
                assignment_entry["source_scene_json"] = str(scene_path)
                if friction_cfg:
                    assignment_entry["friction"] = dict(friction_cfg)
                assignment_entries.append(assignment_entry)
        else:
            io_cfg = world_cfg.get("io", {}) or {}
            scene_json_dir = io_cfg.get("scene_json_dir", None)
            if scene_json_dir is None:
                raise ValueError("world-config is missing io.scene_json_dir")
            for wi in range(world_count):
                src = dict(assignments_cfg[(int(args.assignment_index) + wi) % len(assignments_cfg)])
                raw_scene = src.get("scene_json", None)
                if raw_scene is None:
                    raise ValueError(f"assignment {wi} is missing scene_json")
                src["source_scene_json"] = str(raw_scene)
                src["scene_json"] = str(resolve_scene_json_path(scene_json_dir, raw_scene))
                assignment_entries.append(src)

    eval_curriculum = build_eval_curriculum(
        world_cfg,
        assignment_entries=assignment_entries,
        max_agents_override=max_agents_override,
        gui=bool(args.gui),
        steps_override=args.steps,
        invincible=bool(args.invincible),
    )
    derived_cfg_path = out_dir / "derived_lane_testbed_curriculum.yaml"
    write_yaml(derived_cfg_path, eval_curriculum)

    eval_exp_cfg = deepcopy(exp_cfg)
    eval_exp_cfg.choco_config_path = str(derived_cfg_path)
    eval_exp_cfg.device = device

    seed_everything(int(args.seed))
    source_scene_names = [
        Path(str(a.get("source_scene_json", a.get("scene_json", "")))).name
        for a in assignment_entries
    ]
    unique_source_scene_names = sorted({name for name in source_scene_names if name})
    if len(unique_source_scene_names) <= 4:
        source_scene_desc = ",".join(unique_source_scene_names) if unique_source_scene_names else "n/a"
    else:
        source_scene_desc = ",".join(unique_source_scene_names[:4]) + ",..."

    print(
        "[testbed] starting deterministic evaluation "
        f"source_maps={len(unique_source_scene_names)} source={source_scene_desc} "
        f"mode={'real_start_end' if bool(args.real_start_end) else 'sampled'} "
        f"invincible={bool(args.invincible)} "
        f"world_count={world_count} sampled_agents_total={sum(len(s) for s in samples_by_world)} lane_types={lane_types} "
        f"checkpoint={checkpoint_path}"
    )

    env = ChocolateSB3MultiAgentEnv(
        choco_config_path=eval_exp_cfg.choco_config_path,
        exp_config=eval_exp_cfg,
        device=eval_exp_cfg.device,
        reward_type=eval_exp_cfg.reward_type,
        collision_weight=eval_exp_cfg.collision_weight,
        goal_achieved_weight=eval_exp_cfg.goal_achieved_weight,
        off_road_weight=eval_exp_cfg.off_road_weight,
        log_distance_weight=eval_exp_cfg.log_distance_weight,
    )

    model = None
    try:
        if bool(args.distant_light):
            _ensure_distant_light(
                env.choco_env.stage,
                path="/World/__EvalDistantLight",
                intensity=float(args.distant_light_intensity),
            )

        model = PPO.load(
            str(checkpoint_path),
            env=env,
            device=eval_exp_cfg.device,
            custom_objects=build_resume_custom_objects(env),
        )
        model.policy.set_training_mode(False)

        obs = env.reset()
        assignment_names = [
            Path(
                str(
                    a.get(
                        "source_scene_json",
                        a.get("scene_json", f"world_{wi:03d}.json"),
                    )
                )
            ).name
            for wi, a in enumerate((eval_curriculum.get("world", {}) or {}).get("assignments", []) or [])
        ]
        spawned_by_world: Dict[int, int] = {}
        for key in list(getattr(env, "flat_slot_keys", []) or []):
            wi = int(getattr(key, "world_idx"))
            spawned_by_world[wi] = int(spawned_by_world.get(wi, 0)) + 1
        success_by_world: Dict[int, int] = {wi: 0 for wi in range(int(world_count))}

        ttc_overlay_enabled = bool(args.gui) and (
            bool(args.ttc_overlay) or bool(args.ttc_radius_overlay)
        )
        if ttc_overlay_enabled:
            _update_ttc_overlay(
                env.choco_env,
                env.flat_slot_keys,
                root_path="/World/TTCOverlay",
                overlay_source=str(args.ttc_overlay_source),
                show_text=bool(args.ttc_overlay),
                z_offset_m=float(args.ttc_overlay_z_offset_m),
                y_offset_m=float(args.ttc_overlay_y_offset_m),
                char_height_m=float(args.ttc_overlay_char_height_m),
                max_display_s=float(args.ttc_overlay_max_display_s),
                radius_overlay=bool(args.ttc_radius_overlay),
                radius_height_m=float(args.ttc_radius_height_m),
                radius_z_offset_m=float(args.ttc_radius_z_offset_m),
                radius_opacity=float(args.ttc_radius_opacity),
            )
        controlled_n = int(env.num_envs)
        expected_controlled = int(sum(len(s) for s in samples_by_world))
        if (not bool(args.real_start_end)) and expected_controlled > 0 and controlled_n != expected_controlled:
            print(
                "[testbed][warn] controlled agent count differs from requested. "
                f"requested={expected_controlled} controlled={controlled_n}"
            )

        max_steps = int(args.steps or (eval_curriculum.get("env", {}) or {}).get("max_steps", 300))
        ever_done = np.zeros((controlled_n,), dtype=bool)
        steps_executed = 0
        stop_reason = "max_steps"

        success_count = 0
        vehicle_contact_done_count = 0
        road_contact_done_count = 0
        below_min_z_count = 0
        rollout_rows: List[Dict[str, Any]] = []
        vehicle_collided_any_tokens: set[tuple[int, int]] = set()
        road_edge_contact_any_tokens: set[tuple[int, int]] = set()
        forbidden_road_types = set(
            int(v) for v in (
                getattr(env.choco_env, "road_contact_done_types", None)
                or getattr(env.choco_env, "geom_road_edge_types", [])
                or []
            )
        )

        for step_idx in range(max_steps):
            actions, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env.step(actions)
            raw_info = getattr(env, "last_step_info_raw", None)
            if raw_info is not None:
                raw_keys = list(getattr(raw_info, "keys", []) or [])
                raw_newly_success = np.asarray(
                    getattr(raw_info, "newly_success", np.zeros((len(raw_keys),), dtype=bool)),
                    dtype=bool,
                )
                n = min(len(raw_keys), int(raw_newly_success.shape[0]))
                for i in range(n):
                    if not bool(raw_newly_success[i]):
                        continue
                    wi = int(getattr(raw_keys[i], "world_idx", -1))
                    if wi < 0:
                        continue
                    success_by_world[wi] = int(success_by_world.get(wi, 0)) + 1
            if ttc_overlay_enabled:
                _update_ttc_overlay(
                    env.choco_env,
                    env.flat_slot_keys,
                    root_path="/World/TTCOverlay",
                    overlay_source=str(args.ttc_overlay_source),
                    show_text=bool(args.ttc_overlay),
                    z_offset_m=float(args.ttc_overlay_z_offset_m),
                    y_offset_m=float(args.ttc_overlay_y_offset_m),
                    char_height_m=float(args.ttc_overlay_char_height_m),
                    max_display_s=float(args.ttc_overlay_max_display_s),
                    radius_overlay=bool(args.ttc_radius_overlay),
                    radius_height_m=float(args.ttc_radius_height_m),
                    radius_z_offset_m=float(args.ttc_radius_z_offset_m),
                    radius_opacity=float(args.ttc_radius_opacity),
                )
            if args.gui and float(args.gui_step_delay_sec) > 0.0:
                sleep(float(args.gui_step_delay_sec))

            dones_np = np.asarray(dones, dtype=bool).reshape(-1)
            if dones_np.size == ever_done.size:
                ever_done |= dones_np

            info = dict(getattr(env, "info_dict", {}) or {})
            success_count += int(info.get("new_success_count", 0))
            vehicle_contact_done_count += int(info.get("vehicle_contact_done_count", 0))
            road_contact_done_count += int(info.get("road_contact_done_count", 0))
            below_min_z_count += int(info.get("below_min_z_count", 0))

            # Track "ever collided/contacted road-edge" per spawned token.
            for key in list(getattr(env, "flat_slot_keys", []) or []):
                wi = int(getattr(key, "world_idx"))
                aid = int(getattr(key, "agent_id"))
                token = (wi, aid)
                h = env.choco_env.ctrl.get(wi, aid)
                if h is None:
                    continue
                if bool(env.choco_env._get_vehicle_collided(h)):
                    vehicle_collided_any_tokens.add(token)
                if forbidden_road_types:
                    contact_types = env.choco_env._get_contact_types(h)
                    if any(int(t) in forbidden_road_types for t in contact_types):
                        road_edge_contact_any_tokens.add(token)

            rollout_rows.append(
                {
                    "step": int(step_idx),
                    "new_success_count": int(info.get("new_success_count", 0)),
                    "vehicle_contact_done_count": int(info.get("vehicle_contact_done_count", 0)),
                    "road_contact_done_count": int(info.get("road_contact_done_count", 0)),
                    "below_min_z_count": int(info.get("below_min_z_count", 0)),
                    "goal_rate_step": float(info.get("goal_rate_step", 0.0)),
                    "vehicle_contact_done_rate_step": float(
                        info.get("vehicle_contact_done_rate_step", 0.0)
                    ),
                    "road_contact_done_rate_step": float(info.get("road_contact_done_rate_step", 0.0)),
                    "collision_rate_step": float(info.get("collision_rate_step", 0.0)),
                    "vehicle_collision_rate_step_per_controlled": float(
                        info.get("vehicle_collision_rate_step_per_controlled", 0.0)
                    ),
                }
            )

            steps_executed = int(step_idx + 1)
            if bool(info.get("truncated", False)):
                stop_reason = "timeout"
                break
            if ever_done.size > 0 and bool(np.all(ever_done)):
                stop_reason = "all_done"
                break

        spawned_total = int(getattr(env, "_spawned_episode_total", 0))
        if spawned_total <= 0:
            spawned_total = int(controlled_n)
        done_total = int(ever_done.sum())

        if bool(args.invincible):
            vehicle_collision_count_metric = int(len(vehicle_collided_any_tokens))
            road_done_count_metric = int(len(road_edge_contact_any_tokens))
        else:
            vehicle_collision_count_metric = int(vehicle_contact_done_count)
            road_done_count_metric = int(road_contact_done_count)

        metrics = {
            "spawned_total": int(spawned_total),
            "controlled_agents": int(controlled_n),
            "steps_executed": int(steps_executed),
            "max_steps": int(max_steps),
            "stop_reason": str(stop_reason),
            "counts": {
                "success": int(success_count),
                "vehicle_contact_done": int(vehicle_contact_done_count),
                "road_contact_done": int(road_contact_done_count),
                "vehicle_collision_count_metric": int(vehicle_collision_count_metric),
                "road_done_count_metric": int(road_done_count_metric),
                "vehicle_collision_any_count": int(len(vehicle_collided_any_tokens)),
                "road_edge_contact_any_count": int(len(road_edge_contact_any_tokens)),
                "below_min_z": int(below_min_z_count),
                "done_total": int(done_total),
            },
            "rates": {
                "success_rate": float(success_count) / float(max(1, spawned_total)),
                "vehicle_collision_rate": float(vehicle_collision_count_metric) / float(max(1, spawned_total)),
                "road_done_rate": float(road_done_count_metric) / float(max(1, spawned_total)),
                "below_min_z_rate": float(below_min_z_count) / float(max(1, spawned_total)),
                "done_rate": float(done_total) / float(max(1, spawned_total)),
            },
        }
        per_map_summary = []
        for wi in range(int(world_count)):
            spawned_w = int(spawned_by_world.get(wi, 0))
            success_w = int(success_by_world.get(wi, 0))
            map_name = assignment_names[wi] if wi < len(assignment_names) else f"world_{wi:03d}"
            per_map_summary.append(
                {
                    "world_idx": int(wi),
                    "map_name": str(map_name),
                    "spawned_total": int(spawned_w),
                    "success_count": int(success_w),
                    "success_rate": float(success_w) / float(max(1, spawned_w)),
                }
            )

        agent_dump_by_world = []
        for wi, samples in enumerate(samples_by_world):
            world_dump = [
                {
                    "sample_idx": int(sample.sample_idx),
                    "polyline_idx": int(sample.polyline_idx),
                    "road_type": int(sample.road_type),
                    "travel_distance_m": float(sample.travel_distance_m),
                    "start_xyz": [float(v) for v in sample.start_xyz],
                    "start_yaw_rad": float(sample.start_yaw_rad),
                    "goal_xyz": [float(v) for v in sample.goal_xyz],
                    "goal_yaw_rad": float(sample.goal_yaw_rad),
                }
                for sample in samples
            ]
            agent_dump_by_world.append({"world_idx": int(wi), "agents": world_dump})

        agent_dump = [
            entry
            for world_entry in agent_dump_by_world
            for entry in world_entry["agents"]
        ]

        metadata = {
            "checkpoint": str(checkpoint_path),
            "ppo_config": ppo_config_path_for_report,
            "world_config": str(Path(args.world_config).expanduser().resolve()),
            "derived_curriculum": str(derived_cfg_path),
            "source_scene_json": str(scene_path),
            "source_scene_jsons": [
                str(a.get("source_scene_json", a.get("scene_json", "")))
                for a in assignment_entries
            ],
            "derived_scene_json": str(sampled_scene_paths[0]) if sampled_scene_paths else None,
            "derived_scene_jsons": [str(p) for p in sampled_scene_paths],
            "assignment_index": int(args.assignment_index),
            "base_assignment": base_assignment,
            "friction_cfg": friction_cfg,
            "world_count": int(world_count),
            "lane_types": lane_types,
            "sampling": {
                "mode": "real_start_end" if bool(args.real_start_end) else "sampled",
                "num_vehicles_requested": int(args.num_vehicles),
                "num_vehicles_sampled": int(len(agent_dump)),
                "num_vehicles_sampled_per_world": [int(len(s)) for s in samples_by_world],
                "min_travel_distance_m": float(args.min_travel_distance_m),
                "max_travel_distance_m": float(args.max_travel_distance_m),
                "min_start_gap_m": float(args.min_start_gap_m),
                "min_goal_gap_m": float(args.min_goal_gap_m),
                "endpoint_margin_m": float(args.endpoint_margin_m),
                "min_polyline_length_m": float(args.min_polyline_length_m),
                "max_segment_gap_m": float(args.max_segment_gap_m),
                "bounds_size_m": float(bounds_size_m),
                "origin_mode": str(origin_mode),
            },
            "runtime": {
                "seed": int(args.seed),
                "device": str(device),
                "gui": bool(args.gui),
                "invincible": bool(args.invincible),
                "gui_step_delay_sec": float(args.gui_step_delay_sec),
                "ttc_overlay": bool(ttc_overlay_enabled),
                "ttc_text_overlay": bool(args.ttc_overlay),
                "ttc_overlay_source": str(args.ttc_overlay_source),
                "ttc_overlay_char_height_m": float(args.ttc_overlay_char_height_m),
                "ttc_overlay_z_offset_m": float(args.ttc_overlay_z_offset_m),
                "ttc_overlay_y_offset_m": float(args.ttc_overlay_y_offset_m),
                "ttc_overlay_max_display_s": float(args.ttc_overlay_max_display_s),
                "ttc_radius_overlay": bool(args.ttc_radius_overlay),
                "ttc_radius_height_m": float(args.ttc_radius_height_m),
                "ttc_radius_z_offset_m": float(args.ttc_radius_z_offset_m),
                "ttc_radius_opacity": float(args.ttc_radius_opacity),
                "distant_light": bool(args.distant_light),
                "distant_light_intensity": float(args.distant_light_intensity),
            },
            "metrics": metrics,
            "per_map_summary": per_map_summary,
            "sampled_agents": agent_dump,
            "sampled_agents_by_world": agent_dump_by_world,
        }

        metrics_path = out_dir / "metrics.json"
        write_json(metrics_path, metadata)
        rollout_csv_path = out_dir / "rollout_step_metrics.csv"
        write_rollout_csv(rollout_csv_path, rollout_rows)

        print(f"[testbed] wrote derived curriculum: {derived_cfg_path}")
        if sampled_scene_paths:
            print(
                f"[testbed] wrote sampled scenes: n={len(sampled_scene_paths)} "
                f"first={sampled_scene_paths[0]}"
            )
        print(f"[testbed] wrote rollout metrics CSV: {rollout_csv_path}")
        print(f"[testbed] wrote summary metrics JSON: {metrics_path}")
        print(
            "[testbed] summary "
            f"success_rate={metrics['rates']['success_rate']:.4f} "
            f"vehicle_collision_rate={metrics['rates']['vehicle_collision_rate']:.4f} "
            f"road_done_rate={metrics['rates']['road_done_rate']:.4f} "
            f"done_rate={metrics['rates']['done_rate']:.4f}"
        )
        print(
            "[testbed] summary_counts "
            f"success={metrics['counts']['success']}/{metrics['spawned_total']} "
            f"({metrics['rates']['success_rate']:.4f}) "
            f"vehicle_collision={metrics['counts']['vehicle_collision_count_metric']}/{metrics['spawned_total']} "
            f"({metrics['rates']['vehicle_collision_rate']:.4f}) "
            f"road_done={metrics['counts']['road_done_count_metric']}/{metrics['spawned_total']} "
            f"({metrics['rates']['road_done_rate']:.4f})"
        )
        for row in per_map_summary:
            print(
                "[testbed] summary_map "
                f"map={row['map_name']} world={int(row['world_idx']):03d} "
                f"success={int(row['success_count'])}/{int(row['spawned_total'])} "
                f"({float(row['success_rate']):.4f})"
            )
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
