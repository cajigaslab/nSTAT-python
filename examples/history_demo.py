"""History basis demo aligned with MATLAB History window visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nstat.compat.matlab import History


def run_demo(output_path: Path) -> None:
    history = History(bin_edges_s=np.array([0.0, 0.05, 0.10, 0.20]))

    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=120)
    plt.sca(ax)
    history.plot()
    ax.set_title("History Windows (MATLAB-compatible)")
    ax.set_xlabel("Lag [s]")
    ax.set_ylabel("Window Width")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a History demo figure.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output") / "history_demo.png",
        help="Output image path.",
    )
    args = parser.parse_args()
    run_demo(args.output)
