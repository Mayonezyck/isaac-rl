#!/usr/bin/env python3
"""Convert GPUDrive scene JSONs (objects/roads schema) to choco scene JSONs.

Input (GPUDrive) schema (per file):
  {
    "name": "...",
    "scenario_id": "...",
    "objects": [...],
    "roads": [...],
    "tl_states": {...},
    "metadata": {...}
  }

Output (choco) schema (per file):
  {
    "meta": {...},
    "road": {"polylines": [...]},
    "agents": {"items": [...], "count_valid": N}
  }
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ERR_VAL = -1e4

# Waymo/Waymax map IDs used by GPUDrive conversion.
# See: gpudrive/data_utils/datatypes.py
VALID_ROAD_IDS = {
    0, 1, 2, 3,  # lanes
    5, 6, 7, 8, 9, 10, 11, 12, 13,  # road lines
    14, 15, 16,  # road edges
    17, 18, 19, 20,  # stop_sign/crosswalk/speed_bump/driveway
}

AGENT_TYPE_MAP = {
    "vehicle": 1,
    "pedestrian": 2,
    "cyclist": 3,
}


@dataclass
class ConvertStats:
    files_seen: int = 0
    files_converted: int = 0
    files_skipped: int = 0
    roads_in: int = 0
    roads_out: int = 0
    objects_in: int = 0
    agents_out: int = 0
    experts_skipped: int = 0
    nonvehicle_skipped: int = 0
    invalid_objects_skipped: int = 0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _is_err_xyz(xyz: Dict[str, Any]) -> bool:
    x = _safe_float(xyz.get("x", ERR_VAL), ERR_VAL)
    y = _safe_float(xyz.get("y", ERR_VAL), ERR_VAL)
    z = _safe_float(xyz.get("z", ERR_VAL), ERR_VAL)
    return bool(x <= ERR_VAL * 0.9 or y <= ERR_VAL * 0.9 or z <= ERR_VAL * 0.9)


def _xy_norm(x: float, y: float) -> float:
    return float(math.sqrt(float(x) * float(x) + float(y) * float(y)))


def _yaw_from_velocity(vxy: Dict[str, Any]) -> Optional[float]:
    vx = _safe_float(vxy.get("x", 0.0), 0.0)
    vy = _safe_float(vxy.get("y", 0.0), 0.0)
    if _xy_norm(vx, vy) <= 1e-6:
        return None
    return float(math.atan2(vy, vx))


def _yaw_from_positions(p0: Dict[str, Any], p1: Dict[str, Any]) -> Optional[float]:
    dx = _safe_float(p1.get("x", 0.0), 0.0) - _safe_float(p0.get("x", 0.0), 0.0)
    dy = _safe_float(p1.get("y", 0.0), 0.0) - _safe_float(p0.get("y", 0.0), 0.0)
    if _xy_norm(dx, dy) <= 1e-6:
        return None
    return float(math.atan2(dy, dx))


def _pick_first_last_valid_indices(obj: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    valid_seq = list(obj.get("valid", []) or [])
    positions = list(obj.get("position", []) or [])
    n = min(len(valid_seq), len(positions)) if valid_seq else len(positions)
    if n <= 0:
        return None, None

    first_idx: Optional[int] = None
    last_idx: Optional[int] = None
    if valid_seq:
        for i in range(n):
            if bool(valid_seq[i]) and not _is_err_xyz(positions[i]):
                first_idx = i
                break
        for i in range(n - 1, -1, -1):
            if bool(valid_seq[i]) and not _is_err_xyz(positions[i]):
                last_idx = i
                break
    else:
        for i in range(n):
            if not _is_err_xyz(positions[i]):
                first_idx = i
                break
        for i in range(n - 1, -1, -1):
            if not _is_err_xyz(positions[i]):
                last_idx = i
                break
    return first_idx, last_idx


def _pick_yaw(
    obj: Dict[str, Any],
    idx: int,
    *,
    fallback_other_idx: Optional[int],
) -> float:
    headings = list(obj.get("heading", []) or [])
    if 0 <= idx < len(headings):
        h = _safe_float(headings[idx], ERR_VAL)
        if h > ERR_VAL * 0.9 and math.isfinite(h):
            return float(h)

    velocities = list(obj.get("velocity", []) or [])
    if 0 <= idx < len(velocities):
        yaw = _yaw_from_velocity(velocities[idx] or {})
        if yaw is not None:
            return yaw

    positions = list(obj.get("position", []) or [])
    if (
        fallback_other_idx is not None
        and 0 <= idx < len(positions)
        and 0 <= fallback_other_idx < len(positions)
    ):
        yaw = _yaw_from_positions(positions[idx] or {}, positions[fallback_other_idx] or {})
        if yaw is not None:
            return yaw

    return 0.0


def _convert_roads(roads: Sequence[Dict[str, Any]], *, keep_all_road_ids: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r_idx, road in enumerate(roads):
        map_element_id = _safe_int(road.get("map_element_id", -1), -1)
        if (not keep_all_road_ids) and (map_element_id not in VALID_ROAD_IDS):
            continue

        geom = list(road.get("geometry", []) or [])
        xyz: List[List[float]] = []
        for p in geom:
            px = _safe_float((p or {}).get("x", 0.0), 0.0)
            py = _safe_float((p or {}).get("y", 0.0), 0.0)
            pz = _safe_float((p or {}).get("z", 0.0), 0.0)
            if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(pz)):
                continue
            xyz.append([float(px), float(py), float(pz)])
        if len(xyz) < 1:
            continue

        out.append(
            {
                "id": _safe_int(road.get("id", r_idx), r_idx),
                "type": int(map_element_id),
                "xyz": xyz,
                "src_type": str(road.get("type", "")),
            }
        )
    return out


def _convert_object_to_agent_item(
    obj: Dict[str, Any],
    *,
    track_idx: int,
    use_goal_position_for_end: bool,
) -> Optional[Dict[str, Any]]:
    first_idx, last_idx = _pick_first_last_valid_indices(obj)
    if first_idx is None or last_idx is None:
        return None

    positions = list(obj.get("position", []) or [])
    p_start = positions[first_idx] or {}
    p_last = positions[last_idx] or {}
    p_goal = obj.get("goalPosition", {}) or {}

    if use_goal_position_for_end and (not _is_err_xyz(p_goal)):
        p_end = p_goal
    else:
        p_end = p_last

    start_yaw = _pick_yaw(obj, first_idx, fallback_other_idx=last_idx)
    end_yaw = _pick_yaw(obj, last_idx, fallback_other_idx=first_idx)

    obj_type = str(obj.get("type", "vehicle")).strip().lower()
    agent_type = int(AGENT_TYPE_MAP.get(obj_type, 1))
    agent_id = _safe_int(obj.get("id", track_idx), track_idx)

    return {
        "track_idx": int(track_idx),
        "is_sdc": False,  # filled by caller from metadata.sdc_track_index
        "agent_type": int(agent_type),
        "agent_id": int(agent_id),
        "start": {
            "x": _safe_float(p_start.get("x", 0.0), 0.0),
            "y": _safe_float(p_start.get("y", 0.0), 0.0),
            "z": _safe_float(p_start.get("z", 0.0), 0.0),
            "yaw": float(start_yaw),
        },
        "end": {
            "x": _safe_float(p_end.get("x", 0.0), 0.0),
            "y": _safe_float(p_end.get("y", 0.0), 0.0),
            "z": _safe_float(p_end.get("z", 0.0), 0.0),
            "yaw": float(end_yaw),
        },
        "src_object_type": obj_type,
    }


def convert_scene_dict(
    src: Dict[str, Any],
    *,
    keep_all_road_ids: bool,
    include_non_vehicle: bool,
    include_marked_expert: bool,
    use_goal_position_for_end: bool,
    stats: ConvertStats,
) -> Dict[str, Any]:
    roads = list(src.get("roads", []) or [])
    objects = list(src.get("objects", []) or [])
    metadata = dict(src.get("metadata", {}) or {})

    stats.roads_in += len(roads)
    stats.objects_in += len(objects)

    polylines = _convert_roads(roads, keep_all_road_ids=keep_all_road_ids)
    stats.roads_out += len(polylines)

    sdc_track_index = _safe_int(metadata.get("sdc_track_index", -1), -1)
    items: List[Dict[str, Any]] = []
    for track_idx, obj in enumerate(objects):
        obj_type = str((obj or {}).get("type", "")).strip().lower()
        if (not include_non_vehicle) and (obj_type != "vehicle"):
            stats.nonvehicle_skipped += 1
            continue
        if (not include_marked_expert) and bool((obj or {}).get("mark_as_expert", False)):
            stats.experts_skipped += 1
            continue

        item = _convert_object_to_agent_item(
            obj or {},
            track_idx=int(track_idx),
            use_goal_position_for_end=bool(use_goal_position_for_end),
        )
        if item is None:
            stats.invalid_objects_skipped += 1
            continue
        item["is_sdc"] = bool(int(track_idx) == int(sdc_track_index))
        items.append(item)

    stats.agents_out += len(items)
    out = {
        "meta": {
            "source_name": str(src.get("name", "")),
            "scenario_id": str(src.get("scenario_id", "")),
            "source_schema": "gpudrive",
            "source_metadata": metadata,
        },
        "road": {"polylines": polylines},
        "agents": {"items": items, "count_valid": int(len(items))},
    }
    return out


def _iter_input_jsons(root: Path, *, recursive: bool) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    it = root.rglob("*.json") if recursive else root.glob("*.json")
    for p in sorted(it):
        if p.is_file():
            yield p


def _looks_like_gpudrive_scene(scene: Dict[str, Any]) -> bool:
    return bool(isinstance(scene, dict) and ("objects" in scene) and ("roads" in scene))


def _output_path_for(input_file: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        if output_root.suffix.lower() == ".json":
            return output_root
        return output_root / input_file.name
    rel = input_file.relative_to(input_root)
    return output_root / rel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert GPUDrive scene JSON files (objects/roads) into choco-compatible "
            "scene JSON files (meta/road.polylines/agents.items)."
        )
    )
    p.add_argument(
        "--input",
        required=True,
        help="Input GPUDrive JSON file or directory (can include training/testing/validation subdirs).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output JSON file or directory for converted scenes.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input directory for JSON files.",
    )
    p.add_argument(
        "--include-non-vehicle",
        action="store_true",
        help="Include pedestrian/cyclist objects as controllable agents. Default: vehicle only.",
    )
    p.add_argument(
        "--include-marked-expert",
        action="store_true",
        help="Include objects with mark_as_expert=true. Default: skip these objects.",
    )
    p.add_argument(
        "--keep-all-road-ids",
        action="store_true",
        help=(
            "Keep all road features regardless of map_element_id. "
            "Default keeps known Waymo/Waymax IDs only."
        ),
    )
    p.add_argument(
        "--end-from-last-valid",
        action="store_true",
        help="Use last valid trajectory state as agent end instead of goalPosition.",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Stop after converting this many files (0 means all).",
    )
    p.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation for output files.",
    )
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    input_root = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_root}")

    stats = ConvertStats()
    max_files = max(0, int(args.max_files))
    converted_paths: List[Path] = []

    for src_path in _iter_input_jsons(input_root, recursive=bool(args.recursive)):
        if max_files > 0 and stats.files_converted >= max_files:
            break
        stats.files_seen += 1

        try:
            src = json.loads(src_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[convert][skip] {src_path} read/json error: {exc}")
            stats.files_skipped += 1
            continue
        if not _looks_like_gpudrive_scene(src):
            print(f"[convert][skip] {src_path} not a GPUDrive scene JSON (missing objects/roads).")
            stats.files_skipped += 1
            continue

        out = convert_scene_dict(
            src,
            keep_all_road_ids=bool(args.keep_all_road_ids),
            include_non_vehicle=bool(args.include_non_vehicle),
            include_marked_expert=bool(args.include_marked_expert),
            use_goal_position_for_end=(not bool(args.end_from_last_valid)),
            stats=stats,
        )

        dst_path = _output_path_for(src_path, input_root, output_root)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_text(json.dumps(out, indent=int(args.indent)), encoding="utf-8")
        converted_paths.append(dst_path)
        stats.files_converted += 1
        if stats.files_converted <= 5:
            print(
                "[convert] "
                f"{src_path.name} -> {dst_path} "
                f"roads={len((out.get('road', {}) or {}).get('polylines', []) or [])} "
                f"agents={len(((out.get('agents', {}) or {}).get('items', []) or []))}"
            )

    print(
        "[convert][summary] "
        f"seen={stats.files_seen} converted={stats.files_converted} skipped_files={stats.files_skipped} "
        f"roads_in={stats.roads_in} roads_out={stats.roads_out} "
        f"objects_in={stats.objects_in} agents_out={stats.agents_out}"
    )
    print(
        "[convert][summary] "
        f"experts_skipped={stats.experts_skipped} nonvehicle_skipped={stats.nonvehicle_skipped} "
        f"invalid_objects_skipped={stats.invalid_objects_skipped}"
    )
    if converted_paths:
        print(f"[convert] first_output={converted_paths[0]}")
        print(f"[convert] last_output={converted_paths[-1]}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

