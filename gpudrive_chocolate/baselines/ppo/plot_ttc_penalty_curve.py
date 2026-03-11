from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import yaml


# Matches current vehicle TTC hard clamp in src/chocolate_env.py.
VEHICLE_TTC_HARD_MIN_S = 0.5


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {path}, got {type(data).__name__}")
    return data


def _resolve_curriculum_path(config_path: Path, cfg: Mapping[str, Any]) -> Tuple[Path, Dict[str, Any]]:
    raw = cfg.get("choco_config_path", None)
    if raw is None:
        # Assume caller gave curriculum config directly.
        return config_path, dict(cfg)

    candidate = Path(str(raw)).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (config_path.parent / candidate).resolve()
        if not resolved.exists():
            resolved = (Path.cwd() / candidate).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Could not resolve curriculum config path: {raw}")
    return resolved, _load_yaml(resolved)


def _normalize_fn_name(raw: Any, *, fallback: str = "inverse") -> str:
    fn = str(raw if raw is not None else fallback).strip().lower()
    if fn not in {"inverse", "proximity_zuo"}:
        return fallback
    return fn


def _selected_functions(env: Mapping[str, Any]) -> Tuple[str, str]:
    vehicle_fn = _normalize_fn_name(env.get("ttc_penalty_function", "inverse"), fallback="inverse")
    road_fn = _normalize_fn_name(env.get("road_edge_ttc_penalty_function", vehicle_fn), fallback=vehicle_fn)
    return vehicle_fn, road_fn


def _compute_vehicle_penalty_inverse(ttc_s: np.ndarray, env: Mapping[str, Any]) -> np.ndarray:
    enabled = bool(env.get("ttc_penalty_enable", False))
    alpha = float(env.get("ttc_penalty_alpha", 0.0))
    max_pen = float(env.get("ttc_penalty_max", 0.0))
    min_ttc = max(1e-6, float(env.get("ttc_penalty_min_ttc", 0.2)))

    if not enabled or max_pen <= 0.0 or alpha <= 0.0:
        return np.zeros_like(ttc_s, dtype=np.float64)

    x = np.asarray(ttc_s, dtype=np.float64)
    penalty_abs = np.minimum(max_pen, alpha / np.maximum(x, min_ttc))
    hard_mask = x < VEHICLE_TTC_HARD_MIN_S
    penalty_abs[hard_mask] = max_pen
    return -penalty_abs


def _compute_vehicle_penalty_proximity_zuo(ttc_s: np.ndarray, env: Mapping[str, Any]) -> np.ndarray:
    enabled = bool(env.get("ttc_penalty_enable", False))
    max_pen = float(env.get("ttc_penalty_max", 0.0))
    zuo_a = max(1e-6, float(env.get("ttc_proximity_zuo_a", 0.5)))
    zuo_b = max(1e-6, float(env.get("ttc_proximity_zuo_b", 5.0)))

    if not enabled or max_pen <= 0.0:
        return np.zeros_like(ttc_s, dtype=np.float64)

    x = np.asarray(ttc_s, dtype=np.float64)
    penalty_abs = max_pen * np.exp(-np.power(np.abs(x / zuo_a), zuo_b))
    return -penalty_abs


def _compute_road_penalty_inverse(ttc_s: np.ndarray, env: Mapping[str, Any]) -> np.ndarray:
    enabled = bool(env.get("road_edge_ttc_penalty_enable", False))
    alpha = float(env.get("road_edge_ttc_penalty_alpha", 0.0))
    max_pen = float(env.get("road_edge_ttc_penalty_max", 0.0))
    min_ttc = max(1e-6, float(env.get("road_edge_ttc_penalty_min_ttc", 0.5)))
    hard_min = max(0.0, float(env.get("road_edge_ttc_hard_min_ttc", 0.5)))

    if not enabled or max_pen <= 0.0 or alpha <= 0.0:
        return np.zeros_like(ttc_s, dtype=np.float64)

    x = np.asarray(ttc_s, dtype=np.float64)
    penalty_abs = np.minimum(max_pen, alpha / np.maximum(x, min_ttc))
    hard_mask = x < hard_min
    penalty_abs[hard_mask] = max_pen
    return -penalty_abs


def _compute_road_penalty_proximity_zuo(ttc_s: np.ndarray, env: Mapping[str, Any]) -> np.ndarray:
    enabled = bool(env.get("road_edge_ttc_penalty_enable", False))
    max_pen = float(env.get("road_edge_ttc_penalty_max", 0.0))
    zuo_a = max(
        1e-6,
        float(env.get("road_edge_ttc_proximity_zuo_a", env.get("ttc_proximity_zuo_a", 0.5))),
    )
    zuo_b = max(
        1e-6,
        float(env.get("road_edge_ttc_proximity_zuo_b", env.get("ttc_proximity_zuo_b", 5.0))),
    )

    if not enabled or max_pen <= 0.0:
        return np.zeros_like(ttc_s, dtype=np.float64)

    x = np.asarray(ttc_s, dtype=np.float64)
    penalty_abs = max_pen * np.exp(-np.power(np.abs(x / zuo_a), zuo_b))
    return -penalty_abs


def _default_outputs(config_path: Path, out_dir: Path) -> Tuple[Path, Path]:
    stem = config_path.stem
    return (
        out_dir / f"{stem}_ttc_penalty_curve.png",
        out_dir / f"{stem}_ttc_penalty_curve.csv",
    )


def _write_csv(
    path: Path,
    *,
    ttc_s: np.ndarray,
    vehicle_inverse: np.ndarray,
    vehicle_proximity_zuo: np.ndarray,
    vehicle_selected: np.ndarray,
    road_inverse: np.ndarray,
    road_proximity_zuo: np.ndarray,
    road_selected: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "ttc_s",
                "vehicle_penalty_inverse",
                "vehicle_penalty_proximity_zuo",
                "vehicle_penalty_selected",
                "road_edge_penalty_inverse",
                "road_edge_penalty_proximity_zuo",
                "road_edge_penalty_selected",
            ]
        )
        for x, v_inv, v_zuo, v_sel, r_inv, r_zuo, r_sel in zip(
            ttc_s.tolist(),
            vehicle_inverse.tolist(),
            vehicle_proximity_zuo.tolist(),
            vehicle_selected.tolist(),
            road_inverse.tolist(),
            road_proximity_zuo.tolist(),
            road_selected.tolist(),
        ):
            writer.writerow(
                [float(x), float(v_inv), float(v_zuo), float(v_sel), float(r_inv), float(r_zuo), float(r_sel)]
            )


def _plot(
    *,
    ttc_s: np.ndarray,
    vehicle_inverse: np.ndarray,
    vehicle_proximity_zuo: np.ndarray,
    road_inverse: np.ndarray,
    road_proximity_zuo: np.ndarray,
    vehicle_fn_selected: str,
    road_fn_selected: str,
    out_png: Path,
    show: bool,
    vehicle_hard_min_s: float,
    road_hard_min_s: float,
    title: str,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=160)
    ax_v = axes[0]
    ax_r = axes[1]

    v_inv_lw = 2.8 if vehicle_fn_selected == "inverse" else 1.8
    v_zuo_lw = 2.8 if vehicle_fn_selected == "proximity_zuo" else 1.8
    r_inv_lw = 2.8 if road_fn_selected == "inverse" else 1.8
    r_zuo_lw = 2.8 if road_fn_selected == "proximity_zuo" else 1.8

    ax_v.plot(
        ttc_s,
        vehicle_inverse,
        label="inverse (original)",
        linewidth=v_inv_lw,
        color="#1f77b4",
        alpha=0.95,
    )
    ax_v.plot(
        ttc_s,
        vehicle_proximity_zuo,
        label="proximity_zuo",
        linewidth=v_zuo_lw,
        color="#ff7f0e",
        alpha=0.95,
    )
    ax_v.axvline(
        float(vehicle_hard_min_s),
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label=f"inverse hard clamp @ {vehicle_hard_min_s:.2f}s",
    )
    ax_v.set_xlabel("TTC (seconds)")
    ax_v.set_ylabel("Penalty added to reward")
    ax_v.set_title(f"Vehicle TTC (selected: {vehicle_fn_selected})")
    ax_v.grid(True, linestyle="--", alpha=0.25)
    ax_v.legend(loc="best")

    ax_r.plot(
        ttc_s,
        road_inverse,
        label="inverse (original)",
        linewidth=r_inv_lw,
        color="#1f77b4",
        alpha=0.95,
    )
    ax_r.plot(
        ttc_s,
        road_proximity_zuo,
        label="proximity_zuo",
        linewidth=r_zuo_lw,
        color="#ff7f0e",
        alpha=0.95,
    )
    ax_r.axvline(
        float(road_hard_min_s),
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label=f"inverse hard clamp @ {road_hard_min_s:.2f}s",
    )
    ax_r.set_xlabel("TTC (seconds)")
    ax_r.set_ylabel("Penalty added to reward")
    ax_r.set_title(f"Road-Edge TTC (selected: {road_fn_selected})")
    ax_r.grid(True, linestyle="--", alpha=0.25)
    ax_r.legend(loc="best")

    ax_v.set_xlim(float(ttc_s.min()), float(ttc_s.max()))
    ax_r.set_xlim(float(ttc_s.min()), float(ttc_s.max()))
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    print(f"[ttc-curve] wrote figure: {out_png}")
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot TTC->penalty curves from a PPO config or curriculum config. "
            "Sweeps TTC on a scalar range and applies the same penalty equations used in training."
        )
    )
    p.add_argument("--config", required=True, help="Path to PPO YAML or curriculum YAML.")
    p.add_argument("--ttc-min-s", type=float, default=0.0, help="Sweep range min TTC in seconds.")
    p.add_argument("--ttc-max-s", type=float, default=10.0, help="Sweep range max TTC in seconds.")
    p.add_argument("--num-points", type=int, default=1001, help="Number of samples on TTC axis.")
    p.add_argument(
        "--out-dir",
        default="runs/ttc_penalty_curve",
        help="Output directory for PNG and CSV if explicit output paths are not provided.",
    )
    p.add_argument("--out-png", default=None, help="Optional explicit PNG output path.")
    p.add_argument("--out-csv", default=None, help="Optional explicit CSV output path.")
    p.add_argument("--show", action="store_true", help="Show interactive matplotlib window.")
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = _load_yaml(config_path)
    curriculum_path, curriculum_cfg = _resolve_curriculum_path(config_path, cfg)
    env = curriculum_cfg.get("env", {}) or {}
    if not isinstance(env, Mapping):
        raise ValueError(f"Curriculum env section is invalid in {curriculum_path}")

    ttc_min = float(args.ttc_min_s)
    ttc_max = float(args.ttc_max_s)
    n = int(args.num_points)
    if n < 2:
        raise ValueError("--num-points must be >= 2")
    if ttc_max <= ttc_min:
        raise ValueError("--ttc-max-s must be > --ttc-min-s")

    ttc_s = np.linspace(ttc_min, ttc_max, n, dtype=np.float64)
    vehicle_fn_selected, road_fn_selected = _selected_functions(env)

    vehicle_inverse = _compute_vehicle_penalty_inverse(ttc_s, env)
    vehicle_proximity_zuo = _compute_vehicle_penalty_proximity_zuo(ttc_s, env)
    road_inverse = _compute_road_penalty_inverse(ttc_s, env)
    road_proximity_zuo = _compute_road_penalty_proximity_zuo(ttc_s, env)

    vehicle_selected = (
        vehicle_proximity_zuo if vehicle_fn_selected == "proximity_zuo" else vehicle_inverse
    )
    road_selected = road_proximity_zuo if road_fn_selected == "proximity_zuo" else road_inverse

    out_dir = Path(args.out_dir).expanduser().resolve()
    default_png, default_csv = _default_outputs(curriculum_path, out_dir)
    out_png = Path(args.out_png).expanduser().resolve() if args.out_png else default_png
    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else default_csv

    _write_csv(
        out_csv,
        ttc_s=ttc_s,
        vehicle_inverse=vehicle_inverse,
        vehicle_proximity_zuo=vehicle_proximity_zuo,
        vehicle_selected=vehicle_selected,
        road_inverse=road_inverse,
        road_proximity_zuo=road_proximity_zuo,
        road_selected=road_selected,
    )
    print(f"[ttc-curve] wrote csv: {out_csv}")
    print(
        "[ttc-curve] params "
        f"vehicle(enable={bool(env.get('ttc_penalty_enable', False))}, "
        f"selected_fn={vehicle_fn_selected}, "
        f"alpha={float(env.get('ttc_penalty_alpha', 0.0))}, "
        f"max={float(env.get('ttc_penalty_max', 0.0))}, "
        f"min_ttc={float(env.get('ttc_penalty_min_ttc', 0.2))}, "
        f"hard_min={VEHICLE_TTC_HARD_MIN_S}, "
        f"zuo_a={float(env.get('ttc_proximity_zuo_a', 0.5))}, "
        f"zuo_b={float(env.get('ttc_proximity_zuo_b', 5.0))}) "
        f"road(enable={bool(env.get('road_edge_ttc_penalty_enable', False))}, "
        f"selected_fn={road_fn_selected}, "
        f"alpha={float(env.get('road_edge_ttc_penalty_alpha', 0.0))}, "
        f"max={float(env.get('road_edge_ttc_penalty_max', 0.0))}, "
        f"min_ttc={float(env.get('road_edge_ttc_penalty_min_ttc', 0.5))}, "
        f"hard_min={float(env.get('road_edge_ttc_hard_min_ttc', 0.5))}, "
        f"zuo_a={float(env.get('road_edge_ttc_proximity_zuo_a', env.get('ttc_proximity_zuo_a', 0.5)))}, "
        f"zuo_b={float(env.get('road_edge_ttc_proximity_zuo_b', env.get('ttc_proximity_zuo_b', 5.0)))})"
    )

    _plot(
        ttc_s=ttc_s,
        vehicle_inverse=vehicle_inverse,
        vehicle_proximity_zuo=vehicle_proximity_zuo,
        road_inverse=road_inverse,
        road_proximity_zuo=road_proximity_zuo,
        vehicle_fn_selected=vehicle_fn_selected,
        road_fn_selected=road_fn_selected,
        out_png=out_png,
        show=bool(args.show),
        vehicle_hard_min_s=VEHICLE_TTC_HARD_MIN_S,
        road_hard_min_s=float(env.get("road_edge_ttc_hard_min_ttc", 0.5)),
        title=f"TTC Penalty Curves ({curriculum_path.name})",
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
