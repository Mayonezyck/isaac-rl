#!/usr/bin/env python3
"""Interactive curriculum scene curator.

Workflow:
1) Review existing scenes in a curriculum one-by-one.
   - Press `y` to keep scene.
   - Press `n` to remove scene.
2) If scenes were removed, review new candidate scenes from scene_json_dir.
   - Press `y` to add candidate.
   - Press `n` to skip candidate.
3) Stop when output scene count matches input count.

The script enforces output scene count == input scene count.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


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


def _coerce_scene_json_name(scene_json: Any) -> str:
    if scene_json is None:
        raise ValueError("assignment.scene_json is missing")
    text = str(scene_json).strip()
    if not text:
        raise ValueError("assignment.scene_json is empty")
    if not text.endswith(".json"):
        text = f"{text}.json"
    return text


def _resolve_scene_path(scene_dir: Path, scene_json: Any) -> Path:
    scene_name = _coerce_scene_json_name(scene_json)
    p = Path(scene_name).expanduser()
    if not p.is_absolute():
        p = (scene_dir / p).resolve()
    else:
        p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"scene_json not found: {p}")
    return p


def _scene_json_value_for_assignment(scene_path: Path, scene_dir: Path) -> str:
    try:
        rel = scene_path.resolve().relative_to(scene_dir.resolve())
        return rel.as_posix()
    except Exception:
        return str(scene_path.resolve())


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
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid scene json object: {path}")
    return payload


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


@dataclass
class SceneEntry:
    scene_path: Path
    assignment: Dict[str, Any]
    world_index: int
    source: str  # "existing" | "candidate"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Interactively curate curriculum scenes with matplotlib figure key presses."
        )
    )
    p.add_argument("--curriculum", required=True, help="Input curriculum YAML path.")
    p.add_argument(
        "--out-curriculum",
        default="",
        help=(
            "Output curriculum YAML path. "
            "Default: <input_stem>_curated.yaml next to input curriculum."
        ),
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input curriculum in place.",
    )
    p.add_argument(
        "--scene-glob",
        default="scene_*.json",
        help="Glob pattern used to enumerate candidate scenes in io.scene_json_dir.",
    )
    p.add_argument(
        "--lane-types",
        default="1,2",
        help="Road types drawn as lane center (blue).",
    )
    p.add_argument(
        "--edge-types",
        default="15,16",
        help="Road types drawn as forbidden edge (red).",
    )
    p.add_argument(
        "--show-agents",
        action="store_true",
        help="Render start/goal markers.",
    )
    p.add_argument(
        "--show-od-lines",
        action="store_true",
        help="Render OD connectors/arrows.",
    )
    p.add_argument(
        "--agent-limit",
        type=int,
        default=0,
        help="If >0, only draw first N OD pairs per scene.",
    )
    p.add_argument("--figsize", default="11,11", help="Figure size as 'W,H'.")
    p.add_argument("--dpi", type=int, default=140, help="PNG DPI.")
    p.add_argument(
        "--save-frames-dir",
        default="",
        help="Optional: if set, save each reviewed frame PNG for audit.",
    )
    return p


def _plot_scene(
    *,
    plt: Any,
    scene_cfg: Mapping[str, Any],
    scene_name: str,
    phase_label: str,
    progress_label: str,
    lane_types: Sequence[int],
    edge_types: Sequence[int],
    show_agents: bool,
    show_od_lines: bool,
    agent_limit: int,
    figsize: Tuple[float, float],
) -> Any:
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
            alpha = 0.90
            n_edge += 1
        else:
            color = "#B0B0B0"
            lw = 0.6
            alpha = 0.36
            n_other += 1
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=lw, alpha=alpha, zorder=1)

    drawn_agents = 0
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
            if show_od_lines:
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
            if show_agents:
                ax.scatter(
                    start_arr[:, 0],
                    start_arr[:, 1],
                    s=52.0,
                    c="#2ECC71",
                    alpha=0.95,
                    edgecolors="#0b0b0b",
                    linewidths=0.4,
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
                    zorder=7,
                )
            drawn_agents = int(start_arr.shape[0])

    agents = scene_cfg.get("agents", {}) or {}
    n_items = len(list((agents.get("items", []) or [])))
    count_valid = agents.get("count_valid", None)
    count_valid_text = f"{count_valid}" if count_valid is not None else "n/a"

    ax.set_title(
        f"{phase_label} | {progress_label}\n"
        f"{scene_name} | agents={n_items} count_valid={count_valid_text} drawn={drawn_agents}\n"
        "Keys: y=keep/add, n=remove/skip, q=abort"
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    return fig


def _wait_for_decision(fig: Any, plt: Any) -> str:
    state: Dict[str, str | None] = {"decision": None}

    def _on_key(event: Any) -> None:
        key = str(getattr(event, "key", "") or "").lower()
        if key == "y":
            state["decision"] = "y"
            return
        if key == "n":
            state["decision"] = "n"
            return
        if key in ("q", "escape"):
            state["decision"] = "q"

    def _on_close(_event: Any) -> None:
        if state["decision"] is None:
            state["decision"] = "q"

    key_cid = fig.canvas.mpl_connect("key_press_event", _on_key)
    close_cid = fig.canvas.mpl_connect("close_event", _on_close)
    try:
        while state["decision"] is None:
            plt.pause(0.05)
    finally:
        try:
            fig.canvas.mpl_disconnect(key_cid)
            fig.canvas.mpl_disconnect(close_cid)
        except Exception:
            pass
    return str(state["decision"] or "q")


def _load_curriculum(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid curriculum YAML object: {path}")
    return cfg


def _write_curriculum(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(payload), f, sort_keys=False)


def main() -> None:
    args = _build_parser().parse_args()
    curriculum_path = Path(args.curriculum).expanduser().resolve()
    if not curriculum_path.exists():
        raise FileNotFoundError(f"curriculum not found: {curriculum_path}")

    cfg = _load_curriculum(curriculum_path)
    io_cfg = cfg.get("io", {}) or {}
    world_cfg = cfg.get("world", {}) or {}
    assignments = list((world_cfg.get("assignments", []) or []))
    if not assignments:
        raise ValueError("world.assignments is empty or missing.")

    scene_dir = Path(str(io_cfg.get("scene_json_dir", ""))).expanduser().resolve()
    if not scene_dir.exists():
        raise FileNotFoundError(f"io.scene_json_dir does not exist: {scene_dir}")

    target_count = int(len(assignments))
    lane_types = _parse_int_list(args.lane_types)
    edge_types = _parse_int_list(args.edge_types)
    if not lane_types:
        raise ValueError("--lane-types resolved to empty list")
    if not edge_types:
        raise ValueError("--edge-types resolved to empty list")
    figsize = _parse_float_pair(args.figsize)

    save_frames_dir = (
        Path(args.save_frames_dir).expanduser().resolve()
        if str(args.save_frames_dir).strip()
        else None
    )
    if save_frames_dir is not None:
        save_frames_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt  # type: ignore

    # Build existing scene entries.
    existing_entries: List[SceneEntry] = []
    original_scene_paths_set: set[Path] = set()
    for wi, a in enumerate(assignments):
        if not isinstance(a, Mapping):
            raise ValueError(f"world.assignments[{wi}] must be a mapping")
        scene_path = _resolve_scene_path(scene_dir, a.get("scene_json"))
        original_scene_paths_set.add(scene_path.resolve())
        existing_entries.append(
            SceneEntry(
                scene_path=scene_path.resolve(),
                assignment=dict(a),
                world_index=int(wi),
                source="existing",
            )
        )

    print(
        f"[curate] curriculum={curriculum_path} target_scene_count={target_count} "
        f"scene_dir={scene_dir}"
    )
    print("[curate] controls: focus figure and press y / n / q")

    kept_assignments: List[Dict[str, Any]] = []
    selected_scene_paths: set[Path] = set()
    removed_count = 0
    frame_idx = 0

    # Phase 1: review existing scenes.
    for i, entry in enumerate(existing_entries):
        scene_cfg = _load_scene(entry.scene_path)
        fig = _plot_scene(
            plt=plt,
            scene_cfg=scene_cfg,
            scene_name=entry.scene_path.name,
            phase_label="Review Existing",
            progress_label=f"{i + 1}/{len(existing_entries)}",
            lane_types=lane_types,
            edge_types=edge_types,
            show_agents=bool(args.show_agents),
            show_od_lines=bool(args.show_od_lines),
            agent_limit=int(args.agent_limit),
            figsize=figsize,
        )
        plt.show(block=False)
        if save_frames_dir is not None:
            fig.savefig(
                save_frames_dir / f"{frame_idx:05d}_existing_{entry.scene_path.stem}.png",
                dpi=int(args.dpi),
                bbox_inches="tight",
            )
        decision = _wait_for_decision(fig, plt)
        plt.close(fig)
        frame_idx += 1

        if decision == "q":
            print("[curate] aborted by user; no output written.")
            return
        if decision == "y":
            kept_assignments.append(dict(entry.assignment))
            selected_scene_paths.add(entry.scene_path)
        else:
            removed_count += 1

        print(
            f"[curate] existing scene={entry.scene_path.name} decision={decision} "
            f"kept={len(kept_assignments)} removed={removed_count}"
        )

    need = target_count - len(kept_assignments)
    print(f"[curate] phase1 complete kept={len(kept_assignments)} need_replacements={max(0, need)}")

    # Phase 2: review new candidates not present in original curriculum.
    added_count = 0
    if need > 0:
        candidate_paths = sorted(scene_dir.glob(str(args.scene_glob)))
        candidate_paths = [p.resolve() for p in candidate_paths]
        if not candidate_paths:
            raise RuntimeError(f"No candidate scenes found in {scene_dir} for glob {args.scene_glob!r}")

        # Use first existing assignment as template for non-scene fields (e.g., friction).
        template_assignment = dict(assignments[0])
        candidate_cursor = 0
        while len(kept_assignments) < target_count and candidate_cursor < len(candidate_paths):
            scene_path = candidate_paths[candidate_cursor]
            candidate_cursor += 1

            # Must be non-duplicate with original configured scenes and currently selected scenes.
            if scene_path in original_scene_paths_set:
                continue
            if scene_path in selected_scene_paths:
                continue

            scene_cfg = _load_scene(scene_path)
            progress = f"{added_count + 1} reviewed / need {target_count - len(kept_assignments)}"
            fig = _plot_scene(
                plt=plt,
                scene_cfg=scene_cfg,
                scene_name=scene_path.name,
                phase_label="Add Replacement",
                progress_label=progress,
                lane_types=lane_types,
                edge_types=edge_types,
                show_agents=bool(args.show_agents),
                show_od_lines=bool(args.show_od_lines),
                agent_limit=int(args.agent_limit),
                figsize=figsize,
            )
            plt.show(block=False)
            if save_frames_dir is not None:
                fig.savefig(
                    save_frames_dir / f"{frame_idx:05d}_candidate_{scene_path.stem}.png",
                    dpi=int(args.dpi),
                    bbox_inches="tight",
                )
            decision = _wait_for_decision(fig, plt)
            plt.close(fig)
            frame_idx += 1

            if decision == "q":
                print("[curate] aborted by user; no output written.")
                return
            if decision == "y":
                new_assignment = copy.deepcopy(template_assignment)
                new_assignment["scene_json"] = _scene_json_value_for_assignment(scene_path, scene_dir)
                kept_assignments.append(new_assignment)
                selected_scene_paths.add(scene_path)
                added_count += 1
                print(
                    f"[curate] candidate scene={scene_path.name} decision=y "
                    f"added={added_count} total={len(kept_assignments)}/{target_count}"
                )
            else:
                print(f"[curate] candidate scene={scene_path.name} decision=n")

    if len(kept_assignments) != target_count:
        raise RuntimeError(
            "Could not reach target scene count with accepted replacements. "
            f"target={target_count} got={len(kept_assignments)}. "
            "No output written."
        )

    # Rebuild output config with exactly same scene count.
    out_cfg = copy.deepcopy(cfg)
    out_world_cfg = dict((out_cfg.get("world", {}) or {}))
    out_world_cfg["assignments"] = kept_assignments
    out_world_cfg["world_count"] = int(target_count)
    if "grid_cols" in out_world_cfg and "rows" in out_world_cfg:
        cols = max(1, int(out_world_cfg.get("grid_cols", 1)))
        out_world_cfg["rows"] = int(math.ceil(float(target_count) / float(cols)))
    out_cfg["world"] = out_world_cfg

    if bool(args.in_place):
        out_path = curriculum_path
    elif str(args.out_curriculum).strip():
        out_path = Path(args.out_curriculum).expanduser().resolve()
    else:
        out_path = curriculum_path.with_name(f"{curriculum_path.stem}_curated.yaml")

    _write_curriculum(out_path, out_cfg)
    print(
        f"[curate] wrote={out_path} kept_existing={target_count - added_count} "
        f"removed_existing={removed_count} added_new={added_count} target={target_count}"
    )


if __name__ == "__main__":
    main()
