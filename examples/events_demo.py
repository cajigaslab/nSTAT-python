"""Events class demo aligned to MATLAB Events plotting behavior."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nstat.compat.matlab import Events as MatlabEvents


def run_demo(output_path: Path) -> None:
    time = np.linspace(0.0, 1.0, 500)
    signal = np.sin(2.0 * np.pi * 4.0 * time)

    events = MatlabEvents(times=np.array([0.1, 0.4, 0.9]), labels=["E1", "E2", "E3"])

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=120)
    ax.plot(time, signal, color="k", linewidth=1.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Events Demo (MATLAB-compatible)")

    events.plot(ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an Events demo figure.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output") / "events_demo.png",
        help="Output image path.",
    )
    args = parser.parse_args()
    run_demo(args.output)
