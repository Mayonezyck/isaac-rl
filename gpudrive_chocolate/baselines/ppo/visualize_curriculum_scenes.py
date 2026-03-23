#!/usr/bin/env python3
"""Serial matplotlib visualizer for all scenes in a curriculum."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.trfc import prepare_stage_world_specs


def _parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def _parse_float_pair(text: str) -> Tuple[float, float]:
    parts = [x.strip() for x in str(text).split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected two comma-separated floats, got {text!r}")
    return float(parts[0]), float(parts[1])


def _safe_xy(block: Any) -> np.ndarray | None:
    if not isinstance(block, Mapping):
        return None
    try:
        x = float(block.get("x", None))
        y = float(block.get("y", None))
    except Exception:
        return None
    return np.asarray([x, y], dtype=np.float32)


def _load_scene(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Scene JSON is not an object: {path}")
    return data


def _iter_polylines(scene_cfg: Mapping[str, Any]) -> Iterable[Tuple[int, np.ndarray]]:
    road = scene_cfg.get("road", {}) or {}
    polylines = list((road.get("polylines", []) or []))
    for pl in polylines:
        try:
            road_type = int(pl.get("type", -1))
        except Exception:
            road_type = -1
        pts = np.asarray(pl.get("xyz", []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
            continue
        yield road_type, pts[:, :2]


def _iter_agent_od_pairs(scene_cfg: Mapping[str, Any]) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    agents = scene_cfg.get("agents", {}) or {}
    items = list((agents.get("items", []) or []))
    for item in items:
        start_xy = _safe_xy(item.get("start", None))
        goal_xy = _safe_xy(item.get("end", None))
        if start_xy is None or goal_xy is None:
            continue
        yield start_xy, goal_xy


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Visualize all scenes in a curriculum serially using matplotlib for inspection."
        )
    )
    p.add_argument("--curriculum", required=True, help="Curriculum YAML path.")
    p.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="If >0, only visualize first N worlds/scenes.",
    )
    p.add_argument(
        "--lane-types",
        default="1,2",
        help="Polyline types rendered as lanes (blue).",
    )
    p.add_argument(
        "--edge-types",
        default="15,16",
        help="Polyline types rendered as forbidden road edge (red).",
    )
    p.add_argument(
        "--show-agents",
        action="store_true",
        help="Render start/goal markers for agents.",
    )
    p.add_argument(
        "--show-od-lines",
        action="store_true",
        help="Render start->goal connector lines for agents.",
    )
    p.add_argument(
        "--agent-limit",
        type=int,
        default=0,
        help="If >0, draw only first N agents per scene.",
    )
    p.add_argument(
        "--figsize",
        default="11,11",
        help="Figure size in inches as 'W,H'.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="Figure DPI for saves.",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="If set, save one PNG per scene into this directory.",
    )
    p.add_argument(
        "--save-only",
        action="store_true",
        help="Do not open a window; just save PNGs (requires --out-dir).",
    )
    p.add_argument(
        "--auto-advance-sec",
        type=float,
        default=0.0,
        help="If >0 and interactive mode, auto-advance after this many seconds per scene.",
    )
    return p


def _plot_scene(
    *,
    plt: Any,
    scene_cfg: Mapping[str, Any],
    scene_name: str,
    world_idx: int,
    lane_types: Sequence[int],
    edge_types: Sequence[int],
    show_agents: bool,
    show_od_lines: bool,
    agent_limit: int,
    figsize: Tuple[float, float],
) -> Tuple[Any, int]:
    fig, ax = plt.subplots(figsize=figsize)
    lane_set = {int(x) for x in lane_types}
    edge_set = {int(x) for x in edge_types}

    n_lane = 0
    n_edge = 0
    n_other = 0

    for road_type, pts in _iter_polylines(scene_cfg):
        if road_type in lane_set:
            color = "#2E86DE"
            lw = 0.9
            alpha = 0.80
            n_lane += 1
        elif road_type in edge_set:
            color = "#E74C3C"
            lw = 1.0
            alpha = 0.88
            n_edge += 1
        else:
            color = "#B0B0B0"
            lw = 0.6
            alpha = 0.38
            n_other += 1
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=lw, alpha=alpha)

    drawn_agents = 0
    start_arr: np.ndarray | None = None
    goal_arr: np.ndarray | None = None
    if show_agents or show_od_lines:
        starts: List[np.ndarray] = []
        goals: List[np.ndarray] = []
        for idx, (start_xy, goal_xy) in enumerate(_iter_agent_od_pairs(scene_cfg)):
            if agent_limit > 0 and idx >= int(agent_limit):
                break
            starts.append(start_xy)
            goals.append(goal_xy)
            if show_od_lines:
                ax.plot(
                    [float(start_xy[0]), float(goal_xy[0])],
                    [float(start_xy[1]), float(goal_xy[1])],
                    color="#C218D4",
                    linewidth=2.0,
                    alpha=0.74,
                    solid_capstyle="round",
                    zorder=4,
                )
        if starts:
            start_arr = np.stack(starts, axis=0)
            goal_arr = np.stack(goals, axis=0)
            if show_agents:
                ax.scatter(
                    start_arr[:, 0],
                    start_arr[:, 1],
                    s=52.0,
                    c="#2ECC71",
                    alpha=0.95,
                    edgecolors="#0b0b0b",
                    linewidths=0.4,
                    label="Start",
                    zorder=6,
                )
                ax.scatter(
                    goal_arr[:, 0],
                    goal_arr[:, 1],
                    s=96.0,
                    c="#FFD34E",
                    marker="*",
                    alpha=0.95,
                    edgecolors="#0b0b0b",
                    linewidths=0.45,
                    label="Goal",
                    zorder=7,
                )
            drawn_agents = int(start_arr.shape[0])
            if show_od_lines:
                # Arrow field to emphasize OD direction.
                vec = goal_arr - start_arr
                ax.quiver(
                    start_arr[:, 0],
                    start_arr[:, 1],
                    vec[:, 0],
                    vec[:, 1],
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    width=0.0020,
                    color="#C218D4",
                    alpha=0.42,
                    zorder=5,
                )

    agents = scene_cfg.get("agents", {}) or {}
    item_count = len(list((agents.get("items", []) or [])))
    count_valid = agents.get("count_valid", None)
    count_valid_text = f"{count_valid}" if count_valid is not None else "n/a"

    ax.set_title(
        f"world={world_idx:03d} scene={scene_name}\n"
        f"agents={item_count} count_valid={count_valid_text} "
        f"drawn_agents={drawn_agents} lane={n_lane} edge={n_edge} other={n_other}"
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(False)
    ax.set_aspect("equal", adjustable="box")
    return fig, drawn_agents


def _wait_for_scene_advance(*, fig: Any, plt: Any, timeout_sec: float) -> str:
    state: Dict[str, str | None] = {"cmd": None}

    def _on_key(event: Any) -> None:
        key = str(getattr(event, "key", "") or "").lower()
        if key in ("enter", "return", "space", " ", "right"):
            state["cmd"] = "next"
            return
        if key in ("q", "escape"):
            state["cmd"] = "quit"

    def _on_close(_event: Any) -> None:
        if state["cmd"] is None:
            state["cmd"] = "quit"

    key_cid = fig.canvas.mpl_connect("key_press_event", _on_key)
    close_cid = fig.canvas.mpl_connect("close_event", _on_close)
    t0 = time.monotonic()
    try:
        while state["cmd"] is None:
            plt.pause(0.05)
            if timeout_sec > 0.0 and (time.monotonic() - t0) >= float(timeout_sec):
                state["cmd"] = "next"
                break
    finally:
        try:
            fig.canvas.mpl_disconnect(key_cid)
            fig.canvas.mpl_disconnect(close_cid)
        except Exception:
            pass
    return str(state["cmd"] or "quit")


def main() -> None:
    args = _build_parser().parse_args()
    curriculum_path = Path(args.curriculum).expanduser().resolve()
    if not curriculum_path.exists():
        raise FileNotFoundError(f"curriculum not found: {curriculum_path}")

    with curriculum_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, Mapping):
        raise ValueError(f"invalid curriculum yaml: {curriculum_path}")

    specs = prepare_stage_world_specs(cfg)
    if int(args.max_scenes) > 0:
        specs = specs[: int(args.max_scenes)]
    if not specs:
        raise ValueError("No scenes resolved from curriculum.")

    out_dir = Path(args.out_dir).expanduser().resolve() if str(args.out_dir).strip() else None
    if args.save_only and out_dir is None:
        raise ValueError("--save-only requires --out-dir")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Delay matplotlib import so we can select non-interactive backend for save-only mode.
    import matplotlib

    if bool(args.save_only):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    lane_types = _parse_int_list(args.lane_types)
    edge_types = _parse_int_list(args.edge_types)
    figsize = _parse_float_pair(args.figsize)

    print(
        f"[scene-viz] curriculum={curriculum_path} scenes={len(specs)} "
        f"mode={'save-only' if args.save_only else 'interactive'}"
    )
    if not bool(args.save_only):
        print(
            "[scene-viz] controls: focus figure window, press Enter/Space/Right for next; q or Esc to quit."
        )

    viewed = 0
    for spec in specs:
        scene_path = Path(spec.scene_json_path).resolve()
        scene_name = scene_path.name
        scene_cfg = _load_scene(scene_path)

        fig, drawn_agents = _plot_scene(
            plt=plt,
            scene_cfg=scene_cfg,
            scene_name=scene_name,
            world_idx=int(spec.world_index),
            lane_types=lane_types,
            edge_types=edge_types,
            show_agents=bool(args.show_agents),
            show_od_lines=bool(args.show_od_lines),
            agent_limit=int(args.agent_limit),
            figsize=figsize,
        )

        png_path: Path | None = None
        if out_dir is not None:
            png_path = out_dir / f"{int(spec.world_index):03d}_{scene_path.stem}.png"
            fig.savefig(png_path, dpi=int(args.dpi), bbox_inches="tight")

        print(
            f"[scene-viz] world={int(spec.world_index):03d} scene={scene_name} "
            f"drawn_agents={drawn_agents} saved={png_path if png_path else 'no'}"
        )

        viewed += 1
        if bool(args.save_only):
            plt.close(fig)
            continue

        plt.show(block=False)
        user_cmd = _wait_for_scene_advance(
            fig=fig,
            plt=plt,
            timeout_sec=float(args.auto_advance_sec),
        )
        plt.close(fig)
        if user_cmd == "quit":
            break

    print(f"[scene-viz] done viewed={viewed}")


if __name__ == "__main__":
    main()
