"""Confidence interval plotting demo aligned to MATLAB behavior."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nstat.compat.matlab import ConfidenceInterval


def run_demo(output_path: Path) -> None:
    time = np.linspace(0.0, 1.0, 200)
    mean = 0.5 + 0.2 * np.sin(2.0 * np.pi * 2.0 * time)
    lower = mean - 0.15
    upper = mean + 0.15

    ci = ConfidenceInterval(time=time, lower=lower, upper=upper)
    ci.setColor("g").setValue(0.95)

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=120)
    plt.sca(ax)
    ci.plot("g", 0.2, 1)
    ax.plot(time, mean, color="k", linewidth=1.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Signal")
    ax.set_title("ConfidenceInterval Demo (MATLAB-compatible)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a ConfidenceInterval demo figure.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output") / "confidence_interval_demo.png",
        help="Output image path.",
    )
    args = parser.parse_args()
    run_demo(args.output)
