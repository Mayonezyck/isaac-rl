from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.isaaclab_bootstrap import ensure_isaaclab_source_paths

ensure_isaaclab_source_paths()

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp_cache")

from isaaclab.app import AppLauncher

from src.scene_factory_multiworld_scene import _build_single_world_roads_only, _load_yaml


DEFAULT_SCENE_FACTORY_CONFIG = "configs/scene_factory/generated/scene_factory_64scene_curated_0326_train.yaml"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Quick smoke test for SceneFactory road authoring equivalence. "
            "Build the same scene once with explicit prims and once with point instancer, "
            "then compare the stored road metadata."
        )
    )
    parser.add_argument(
        "--scene_factory_config",
        type=str,
        default=DEFAULT_SCENE_FACTORY_CONFIG,
        help="SceneFactory world/scene YAML used to source the scene JSON and road settings.",
    )
    parser.add_argument(
        "--assignment_index",
        type=int,
        default=0,
        help="Assignment index inside world.assignments to compare.",
    )
    parser.add_argument(
        "--scene_json",
        type=str,
        default="",
        help="Optional direct scene JSON path. Overrides assignment_index lookup.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-6,
        help="Absolute tolerance for float metadata comparison.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1.0e-6,
        help="Relative tolerance for float metadata comparison.",
    )
    parser.add_argument(
        "--side_by_side_offset_m",
        type=float,
        default=260.0,
        help="Horizontal spacing between explicit and instancer worlds in the optional exported stage.",
    )
    parser.add_argument(
        "--save_stage_usd",
        type=str,
        default="",
        help="Optional USD path for exporting the side-by-side comparison stage.",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default="",
        help="Optional JSON file for writing the comparison summary.",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser


def _resolve_scene_json_path(scene_factory_cfg: dict[str, Any], assignment_index: int, scene_json_override: str) -> Path:
    if scene_json_override:
        return Path(scene_json_override).expanduser().resolve()

    io_cfg = dict(scene_factory_cfg.get("io", {}) or {})
    world_cfg = dict(scene_factory_cfg.get("world", {}) or {})
    assignments = list(world_cfg.get("assignments", []) or [])
    if not assignments:
        raise RuntimeError("SceneFactory config has no world.assignments.")
    if not (0 <= int(assignment_index) < len(assignments)):
        raise IndexError(
            f"assignment_index={assignment_index} is out of range for {len(assignments)} assignments."
        )
    scene_json_dir = Path(str(io_cfg.get("scene_json_dir", ""))).expanduser().resolve()
    if not scene_json_dir.is_dir():
        raise FileNotFoundError(f"scene_json_dir does not exist: {scene_json_dir}")
    scene_name = str(dict(assignments[int(assignment_index)]).get("scene_json", "")).strip()
    if not scene_name:
        raise RuntimeError(f"Assignment {assignment_index} does not define scene_json.")
    return (scene_json_dir / scene_name).resolve()


def _road_metadata_from_world_root(stage: Any, world_root: str) -> dict[str, np.ndarray]:
    prim = stage.GetPrimAtPath(str(world_root))
    if not prim.IsValid():
        raise RuntimeError(f"Invalid world root: {world_root}")
    try:
        custom_data = prim.GetCustomData()
    except Exception:
        custom_data = {}
    if not isinstance(custom_data, dict):
        custom_data = {}

    return {
        "road_points_m": np.asarray(custom_data.get("road_points_m", []), dtype=np.float32),
        "road_point_dirs": np.asarray(custom_data.get("road_point_dirs", []), dtype=np.float32),
        "road_point_types": np.asarray(custom_data.get("road_point_types", []), dtype=np.int64),
        "road_point_half_lengths_m": np.asarray(custom_data.get("road_point_half_lengths_m", []), dtype=np.float32),
        "road_point_half_widths_m": np.asarray(custom_data.get("road_point_half_widths_m", []), dtype=np.float32),
    }


def _compare_array(
    name: str,
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "name": name,
        "shape_equal": list(lhs.shape) == list(rhs.shape),
        "lhs_shape": list(lhs.shape),
        "rhs_shape": list(rhs.shape),
        "match": False,
    }
    if list(lhs.shape) != list(rhs.shape):
        return summary

    if np.issubdtype(lhs.dtype, np.integer) or np.issubdtype(rhs.dtype, np.integer):
        equal = bool(np.array_equal(lhs, rhs))
        summary["match"] = equal
        if not equal:
            summary["mismatch_count"] = int(np.count_nonzero(lhs != rhs))
        return summary

    if lhs.size == 0 and rhs.size == 0:
        summary["match"] = True
        summary["max_abs_diff"] = 0.0
        return summary

    diff = np.abs(lhs - rhs)
    max_abs_diff = float(np.max(diff)) if diff.size > 0 else 0.0
    summary["max_abs_diff"] = max_abs_diff
    summary["match"] = bool(np.allclose(lhs, rhs, atol=atol, rtol=rtol))
    return summary


def _comparison_summary(
    *,
    explicit_meta: dict[str, np.ndarray],
    instancer_meta: dict[str, np.ndarray],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    comparisons = {
        key: _compare_array(key, explicit_meta[key], instancer_meta[key], atol=atol, rtol=rtol)
        for key in explicit_meta.keys()
    }
    metadata_match = all(item["match"] for item in comparisons.values())
    point_count = int(explicit_meta["road_points_m"].shape[0]) if explicit_meta["road_points_m"].ndim >= 1 else 0
    unique_types = (
        np.unique(explicit_meta["road_point_types"]).astype(np.int64).tolist()
        if explicit_meta["road_point_types"].size > 0
        else []
    )
    return {
        "metadata_match": bool(metadata_match),
        "point_count": point_count,
        "unique_road_types": unique_types,
        "comparisons": comparisons,
    }


def _offset_parent_xform(stage: Any, parent_path: str, translate_xyz: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    parent_prim = stage.GetPrimAtPath(str(parent_path))
    if not parent_prim.IsValid():
        UsdGeom.Xform.Define(stage, str(parent_path))
        parent_prim = stage.GetPrimAtPath(str(parent_path))
    xform = UsdGeom.Xformable(parent_prim)
    ops = xform.GetOrderedXformOps()
    translate_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*map(float, translate_xyz)))


def _print_summary(summary: dict[str, Any]) -> None:
    print("[SceneFactory][RoadCompare] summary", flush=True)
    print(
        f"  metadata_match={summary['metadata_match']} "
        f"point_count={summary['point_count']} "
        f"unique_road_types={summary['unique_road_types']}",
        flush=True,
    )
    for key, item in summary["comparisons"].items():
        line = (
            f"  {key}: match={item['match']} "
            f"lhs_shape={item['lhs_shape']} rhs_shape={item['rhs_shape']}"
        )
        if "max_abs_diff" in item:
            line += f" max_abs_diff={item['max_abs_diff']:.6g}"
        if "mismatch_count" in item:
            line += f" mismatch_count={item['mismatch_count']}"
        print(line, flush=True)


def main() -> None:
    args = _build_parser().parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.sim import SimulationCfg, SimulationContext
    import omni.usd

    sim_device = str(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=sim_device))
    usd_context = omni.usd.get_context()
    usd_context.new_stage()
    stage = usd_context.get_stage()

    scene_factory_cfg_path = Path(args.scene_factory_config).expanduser().resolve()
    scene_factory_cfg = _load_yaml(scene_factory_cfg_path)
    scene_json_path = _resolve_scene_json_path(scene_factory_cfg, args.assignment_index, args.scene_json)

    explicit_cfg = dict(scene_factory_cfg)
    explicit_cfg["road"] = dict(scene_factory_cfg.get("road", {}) or {})
    explicit_cfg["road"]["render_mode"] = "explicit_prims"

    instancer_cfg = dict(scene_factory_cfg)
    instancer_cfg["road"] = dict(scene_factory_cfg.get("road", {}) or {})
    instancer_cfg["road"]["render_mode"] = "point_instancer"

    explicit_world_root = "/World/RoadCompare/Explicit/world_000"
    instancer_world_root = "/World/RoadCompare/PointInstancer/world_000"

    _build_single_world_roads_only(
        stage=stage,
        cfg=explicit_cfg,
        json_path=scene_json_path,
        world_root=explicit_world_root,
    )
    _build_single_world_roads_only(
        stage=stage,
        cfg=instancer_cfg,
        json_path=scene_json_path,
        world_root=instancer_world_root,
    )

    half_offset = 0.5 * float(args.side_by_side_offset_m)
    _offset_parent_xform(stage, "/World/RoadCompare/Explicit", (-half_offset, 0.0, 0.0))
    _offset_parent_xform(stage, "/World/RoadCompare/PointInstancer", (half_offset, 0.0, 0.0))

    explicit_meta = _road_metadata_from_world_root(stage, explicit_world_root)
    instancer_meta = _road_metadata_from_world_root(stage, instancer_world_root)
    summary = {
        "scene_factory_config_path": str(scene_factory_cfg_path),
        "scene_json_path": str(scene_json_path),
        "assignment_index": int(args.assignment_index),
        "explicit_render_mode": "explicit_prims",
        "instancer_render_mode": "point_instancer",
        **_comparison_summary(
            explicit_meta=explicit_meta,
            instancer_meta=instancer_meta,
            atol=float(args.atol),
            rtol=float(args.rtol),
        ),
    }

    _print_summary(summary)

    if args.summary_json:
        summary_json_path = Path(args.summary_json).expanduser().resolve()
        summary_json_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_json_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"[SceneFactory][RoadCompare] wrote summary json: {summary_json_path}", flush=True)

    if args.save_stage_usd:
        save_stage_path = Path(args.save_stage_usd).expanduser().resolve()
        save_stage_path.parent.mkdir(parents=True, exist_ok=True)
        stage.Export(str(save_stage_path))
        print(f"[SceneFactory][RoadCompare] wrote stage usd: {save_stage_path}", flush=True)

    simulation_app.close()
    raise SystemExit(0 if bool(summary["metadata_match"]) else 1)


if __name__ == "__main__":
    main()
