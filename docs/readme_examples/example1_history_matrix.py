"""README Example 1: history design matrix heatmap."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from nstat.compat.matlab import History


def main() -> None:
    np.random.seed(0)

    spike_times = np.sort(np.random.uniform(0.05, 0.95, size=40))
    t_grid = np.linspace(0.0, 1.0, 1001, dtype=float)

    hist = History(np.array([0.0, 0.01, 0.02, 0.05, 0.10], dtype=float))
    H = hist.computeHistory(spike_times, t_grid)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(
        H.T,
        aspect="auto",
        origin="lower",
        extent=[float(t_grid[0]), float(t_grid[-1]), 0, H.shape[1]],
        cmap="viridis",
    )
    ax.set_title("History Design Matrix")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("history window index")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("spike count")
    fig.tight_layout()

    out = Path(__file__).resolve().parents[1] / "images" / "readme_example1_history_matrix.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
