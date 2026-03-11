from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def _parse_sizes(raw: str) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for token in str(raw).split(","):
        item = token.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(
                f"Invalid size token '{token}'. Expected format like '4.5x2.0,5.0x2.2'."
            )
        a, b = item.split("x", 1)
        length_m = float(a.strip())
        width_m = float(b.strip())
        if length_m <= 0.0 or width_m <= 0.0:
            raise ValueError(f"Invalid non-positive size: {length_m}x{width_m}")
        out.append((length_m, width_m))
    if not out:
        raise ValueError("No valid sizes parsed.")
    return out


def _make_grid(
    length_m: float,
    width_m: float,
    centers_xy: np.ndarray,
    radius_m: float,
    step_m: float,
    pad_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x_min = min(-0.5 * length_m, float(np.min(centers_xy[:, 0] - radius_m))) - pad_m
    x_max = max(0.5 * length_m, float(np.max(centers_xy[:, 0] + radius_m))) + pad_m
    y_min = min(-0.5 * width_m, float(np.min(centers_xy[:, 1] - radius_m))) - pad_m
    y_max = max(0.5 * width_m, float(np.max(centers_xy[:, 1] + radius_m))) + pad_m

    nx = int(np.ceil((x_max - x_min) / step_m)) + 1
    ny = int(np.ceil((y_max - y_min) / step_m)) + 1
    xs = np.linspace(x_min, x_max, nx, dtype=np.float32)
    ys = np.linspace(y_min, y_max, ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return X, Y


def _compute_masks(
    length_m: float,
    width_m: float,
    centers_xy: np.ndarray,
    radius_m: float,
    X: np.ndarray,
    Y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    rect_mask = (np.abs(X) <= 0.5 * length_m) & (np.abs(Y) <= 0.5 * width_m)
    circles_union = np.zeros_like(rect_mask, dtype=bool)
    for cx, cy in centers_xy:
        circles_union |= ((X - cx) ** 2 + (Y - cy) ** 2) <= radius_m**2
    return rect_mask, circles_union


def _category_map(rect_mask: np.ndarray, circles_union: np.ndarray) -> np.ndarray:
    # 0: background, 1: overlap(rect & circles), 2: circles outside rect, 3: rect uncovered by circles
    cat = np.zeros(rect_mask.shape, dtype=np.int8)
    overlap = rect_mask & circles_union
    outside = circles_union & (~rect_mask)
    uncovered = rect_mask & (~circles_union)
    cat[overlap] = 1
    cat[outside] = 2
    cat[uncovered] = 3
    return cat


def _compute_area_metrics(
    rect_mask: np.ndarray,
    circles_union: np.ndarray,
    cell_area_m2: float,
) -> dict:
    overlap = rect_mask & circles_union
    outside = circles_union & (~rect_mask)
    uncovered = rect_mask & (~circles_union)

    overlap_area = float(np.sum(overlap)) * cell_area_m2
    outside_area = float(np.sum(outside)) * cell_area_m2
    uncovered_area = float(np.sum(uncovered)) * cell_area_m2
    rect_area = float(np.sum(rect_mask)) * cell_area_m2
    circle_union_area = float(np.sum(circles_union)) * cell_area_m2
    iou = overlap_area / max(rect_area + circle_union_area - overlap_area, 1e-9)
    coverage = overlap_area / max(rect_area, 1e-9)

    return {
        "rect_area_m2": rect_area,
        "circle_union_area_m2": circle_union_area,
        "overlap_area_m2": overlap_area,
        "outside_area_m2": outside_area,
        "uncovered_area_m2": uncovered_area,
        "coverage_ratio": coverage,
        "iou": iou,
    }


def _build_three_circle_model(length_m: float, width_m: float) -> Tuple[np.ndarray, float]:
    radius_m = 0.5 * width_m
    spine_offset_m = max(0.0, 0.5 * length_m - radius_m)
    centers_xy = np.asarray(
        [
            [-spine_offset_m, 0.0],
            [0.0, 0.0],
            [spine_offset_m, 0.0],
        ],
        dtype=np.float32,
    )
    return centers_xy, float(radius_m)


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    rows: Sequence[dict],
    *,
    out_png: Path,
    show: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg" if not show else matplotlib.get_backend())
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib import colors
    from matplotlib.patches import Circle, Rectangle

    n = len(rows)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4.8 * nrows), dpi=150)
    axes_arr = np.asarray(axes, dtype=object).reshape(-1)

    cmap = colors.ListedColormap(
        [
            (1.0, 1.0, 1.0, 0.0),  # background
            (0.24, 0.72, 0.29, 0.55),  # overlap
            (1.00, 0.62, 0.16, 0.55),  # circles outside
            (0.86, 0.26, 0.21, 0.55),  # uncovered
        ]
    )

    for i, row in enumerate(rows):
        ax = axes_arr[i]
        X = row["grid_x"]
        Y = row["grid_y"]
        cat = row["cat"]
        L = row["length_m"]
        W = row["width_m"]
        centers_xy = row["centers_xy"]
        radius_m = row["radius_m"]
        m = row["metrics"]

        ax.imshow(
            cat,
            origin="lower",
            extent=[float(X.min()), float(X.max()), float(Y.min()), float(Y.max())],
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=3,
            aspect="equal",
        )

        rect = Rectangle(
            (-0.5 * L, -0.5 * W),
            L,
            W,
            linewidth=2.0,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(rect)
        for cx, cy in centers_xy:
            ax.add_patch(
                Circle(
                    (float(cx), float(cy)),
                    radius_m,
                    linewidth=2.0,
                    edgecolor="#1f78b4",
                    facecolor="none",
                )
            )

        ax.set_title(f"Car {L:.2f}m x {W:.2f}m (r={radius_m:.2f}m)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(alpha=0.2, linestyle="--", linewidth=0.7)

        text = (
            f"Overlap:   {m['overlap_area_m2']:.3f} m^2 ({100.0*m['coverage_ratio']:.1f}% of car)\n"
            f"Outside:   {m['outside_area_m2']:.3f} m^2 ({100.0*m['outside_area_m2']/max(m['circle_union_area_m2'],1e-9):.1f}% of circles)\n"
            f"Uncovered: {m['uncovered_area_m2']:.3f} m^2 ({100.0*m['uncovered_area_m2']/max(m['rect_area_m2'],1e-9):.1f}% of car)\n"
            f"IoU:       {m['iou']:.3f}"
        )
        ax.text(
            0.02,
            0.02,
            text,
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#444444"},
        )

    for j in range(n, len(axes_arr)):
        axes_arr[j].axis("off")

    fig.suptitle("Three-Circle TTC Approximation Coverage (Top-Down)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    print(f"[ttc-coverage] wrote figure: {out_png}")
    if show:
        plt.show()
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    sizes = _parse_sizes(args.sizes_m)
    step_m = float(args.grid_step_m)
    if step_m <= 0.0:
        raise ValueError("--grid-step-m must be > 0")

    rows_for_plot: List[dict] = []
    rows_for_csv: List[dict] = []

    for length_m, width_m in sizes:
        centers_xy, radius_m = _build_three_circle_model(length_m, width_m)
        X, Y = _make_grid(
            length_m=length_m,
            width_m=width_m,
            centers_xy=centers_xy,
            radius_m=radius_m,
            step_m=step_m,
            pad_m=float(args.pad_m),
        )
        rect_mask, circles_union = _compute_masks(length_m, width_m, centers_xy, radius_m, X, Y)
        cat = _category_map(rect_mask, circles_union)
        metrics = _compute_area_metrics(rect_mask, circles_union, cell_area_m2=step_m * step_m)

        rows_for_plot.append(
            {
                "length_m": float(length_m),
                "width_m": float(width_m),
                "radius_m": float(radius_m),
                "centers_xy": centers_xy,
                "grid_x": X,
                "grid_y": Y,
                "cat": cat,
                "metrics": metrics,
            }
        )
        rows_for_csv.append(
            {
                "length_m": float(length_m),
                "width_m": float(width_m),
                "radius_m": float(radius_m),
                "spine_offset_m": float(max(0.0, 0.5 * length_m - radius_m)),
                **metrics,
            }
        )

    _plot(rows_for_plot, out_png=Path(args.out_png), show=bool(args.show))
    _write_csv(Path(args.out_csv), rows_for_csv)
    print(f"[ttc-coverage] wrote csv: {args.out_csv}")
    print("[ttc-coverage] summary")
    for row in rows_for_csv:
        print(
            "  "
            + f"LxW={row['length_m']:.2f}x{row['width_m']:.2f} "
            + f"coverage={100.0*row['coverage_ratio']:.1f}% "
            + f"outside={row['outside_area_m2']:.3f}m^2 "
            + f"uncovered={row['uncovered_area_m2']:.3f}m^2 "
            + f"iou={row['iou']:.3f}"
        )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Visualize 3-circle TTC approximation on top of vehicle rectangles and "
            "report overlap/outside/uncovered area metrics."
        )
    )
    p.add_argument(
        "--sizes-m",
        type=str,
        default="4.0x1.8,4.5x2.0,5.0x2.2,6.0x2.5",
        help="Comma-separated vehicle sizes as LxW in meters.",
    )
    p.add_argument(
        "--grid-step-m",
        type=float,
        default=0.01,
        help="Grid cell size (m) for area approximation.",
    )
    p.add_argument(
        "--pad-m",
        type=float,
        default=0.8,
        help="Extra plot/grid padding around shape extents.",
    )
    p.add_argument(
        "--out-png",
        type=str,
        default="runs/ttc_circle_coverage/ttc_circle_coverage.png",
        help="Output figure path.",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default="runs/ttc_circle_coverage/ttc_circle_coverage_metrics.csv",
        help="Output CSV path.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show figure interactively (also writes output PNG).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
