#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gpudrive_chocolate.utils.obs_vis import save_obs_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a bar-plot visualization of an obs vector.")
    parser.add_argument("--obs-npy", type=str, default=None, help="Path to .npy file containing an obs vector.")
    parser.add_argument("--out", type=str, default="obs_vis.png", help="Output PNG path.")
    parser.add_argument("--title", type=str, default=None, help="Optional title for the plot.")
    args = parser.parse_args()

    if args.obs_npy:
        obs = np.load(args.obs_npy)
    else:
        obs = np.random.randn(64).astype(np.float32)

    save_obs_png(obs, Path(args.out), title=args.title)


if __name__ == "__main__":
    main()
