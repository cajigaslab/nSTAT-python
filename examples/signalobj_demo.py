"""SignalObj demo aligned to core MATLAB SignalObj operations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nstat.compat.matlab import SignalObj


def run_demo(output_path: Path) -> None:
    time = np.linspace(0.0, 1.0, 5)
    data = np.column_stack([
        np.array([1.0, 2.0, 4.0, 3.0, 2.0]),
        np.array([2.0, 3.0, 5.0, 4.0, 3.0]),
    ])

    sig = SignalObj(time=time, data=data, name="sig", x_label="time", x_units="s", y_units="unit")
    deriv = sig.derivative()
    merged = sig.merge(
        SignalObj(time=time, data=np.array([10, 20, 30, 40, 50], dtype=float), name="sig2", x_label="time", x_units="s", y_units="unit")
    )

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), dpi=120, sharex=True)
    axes[0].plot(sig.getTime(), sig.dataToMatrix())
    axes[0].set_title("SignalObj: base signal")
    axes[0].set_ylabel("value")

    axes[1].plot(deriv.getTime(), deriv.dataToMatrix())
    axes[1].set_title("SignalObj.derivative")
    axes[1].set_ylabel("d/dt")

    axes[2].plot(merged.getTime(), merged.dataToMatrix())
    axes[2].set_title("SignalObj.merge")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("merged")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a SignalObj demo figure.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output") / "signalobj_demo.png",
        help="Output image path.",
    )
    args = parser.parse_args()
    run_demo(args.output)
