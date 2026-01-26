from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def save_obs_png(obs: Iterable[float], path: Path, title: Optional[str] = None) -> None:
    """Save a simple bar-plot visualization of an observation vector."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs_arr = np.asarray(list(obs), dtype=np.float32)
    x = np.arange(obs_arr.size)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(x, obs_arr, color="#2a6f97", width=0.9)
    ax.set_xlabel("obs index")
    ax.set_ylabel("value")
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
