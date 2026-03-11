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


def _compute_vehicle_penalty(ttc_s: np.ndarray, env: Mapping[str, Any]) -> np.ndarray:
    enabled = bool(env.get("ttc_penalty_enable", False))
    alpha = float(env.get("ttc_penalty_alpha", 0.0))
    max_pen = float(env.get("ttc_penalty_max", 0.0))
    min_ttc = max(1e-6, float(env.get("ttc_penalty_min_ttc", 0.2)))

    if (not enabled) or alpha <= 0.0 or max_pen <= 0.0:
        return np.zeros_like(ttc_s, dtype=np.float64)

    x = np.asarray(ttc_s, dtype=np.float64)
    penalty_abs = np.minimum(max_pen, alpha / np.maximum(x, min_ttc))
    hard_mask = x < VEHICLE_TTC_HARD_MIN_S
    penalty_abs[hard_mask] = max_pen
    return -penalty_abs


def _compute_road_penalty(ttc_s: np.ndarray, env: Mapping[str, Any]) -> np.ndarray:
    enabled = bool(env.get("road_edge_ttc_penalty_enable", False))
    alpha = float(env.get("road_edge_ttc_penalty_alpha", 0.0))
    max_pen = float(env.get("road_edge_ttc_penalty_max", 0.0))
    min_ttc = max(1e-6, float(env.get("road_edge_ttc_penalty_min_ttc", 0.5)))
    hard_min = max(0.0, float(env.get("road_edge_ttc_hard_min_ttc", 0.5)))

    if (not enabled) or alpha <= 0.0 or max_pen <= 0.0:
        return np.zeros_like(ttc_s, dtype=np.float64)

    x = np.asarray(ttc_s, dtype=np.float64)
    penalty_abs = np.minimum(max_pen, alpha / np.maximum(x, min_ttc))
    hard_mask = x < hard_min
    penalty_abs[hard_mask] = max_pen
    return -penalty_abs


def _default_outputs(config_path: Path, out_dir: Path) -> Tuple[Path, Path]:
    stem = config_path.stem
    return (
        out_dir / f"{stem}_ttc_penalty_curve.png",
        out_dir / f"{stem}_ttc_penalty_curve.csv",
    )


def _write_csv(path: Path, ttc_s: np.ndarray, vehicle_penalty: np.ndarray, road_penalty: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ttc_s", "vehicle_ttc_penalty", "road_edge_ttc_penalty"])
        for x, yv, yr in zip(ttc_s.tolist(), vehicle_penalty.tolist(), road_penalty.tolist()):
            writer.writerow([float(x), float(yv), float(yr)])


def _plot(
    *,
    ttc_s: np.ndarray,
    vehicle_penalty: np.ndarray,
    road_penalty: np.ndarray,
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

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=160)
    ax.plot(ttc_s, vehicle_penalty, label="Vehicle TTC penalty", linewidth=2.2, color="#1f77b4")
    ax.plot(ttc_s, road_penalty, label="Road-edge TTC penalty", linewidth=2.2, color="#d62728")
    ax.axvline(
        float(vehicle_hard_min_s),
        color="#1f77b4",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label=f"Vehicle hard clamp @ {vehicle_hard_min_s:.2f}s",
    )
    ax.axvline(
        float(road_hard_min_s),
        color="#d62728",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label=f"Road hard clamp @ {road_hard_min_s:.2f}s",
    )
    ax.set_xlabel("TTC (seconds)")
    ax.set_ylabel("Penalty added to reward")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best")
    ax.set_xlim(float(ttc_s.min()), float(ttc_s.max()))
    fig.tight_layout()

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
    vehicle_penalty = _compute_vehicle_penalty(ttc_s, env)
    road_penalty = _compute_road_penalty(ttc_s, env)

    out_dir = Path(args.out_dir).expanduser().resolve()
    default_png, default_csv = _default_outputs(curriculum_path, out_dir)
    out_png = Path(args.out_png).expanduser().resolve() if args.out_png else default_png
    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else default_csv

    _write_csv(out_csv, ttc_s, vehicle_penalty, road_penalty)
    print(f"[ttc-curve] wrote csv: {out_csv}")
    print(
        "[ttc-curve] params "
        f"vehicle(enable={bool(env.get('ttc_penalty_enable', False))}, "
        f"alpha={float(env.get('ttc_penalty_alpha', 0.0))}, "
        f"max={float(env.get('ttc_penalty_max', 0.0))}, "
        f"min_ttc={float(env.get('ttc_penalty_min_ttc', 0.2))}, "
        f"hard_min={VEHICLE_TTC_HARD_MIN_S}) "
        f"road(enable={bool(env.get('road_edge_ttc_penalty_enable', False))}, "
        f"alpha={float(env.get('road_edge_ttc_penalty_alpha', 0.0))}, "
        f"max={float(env.get('road_edge_ttc_penalty_max', 0.0))}, "
        f"min_ttc={float(env.get('road_edge_ttc_penalty_min_ttc', 0.5))}, "
        f"hard_min={float(env.get('road_edge_ttc_hard_min_ttc', 0.5))})"
    )

    _plot(
        ttc_s=ttc_s,
        vehicle_penalty=vehicle_penalty,
        road_penalty=road_penalty,
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
