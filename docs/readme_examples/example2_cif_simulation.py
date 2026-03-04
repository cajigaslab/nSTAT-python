"""README Example 2: CIF simulation with lambda curve and spike rasters."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from nstat.compat.matlab import CIF, Covariate


def main() -> None:
    np.random.seed(0)

    t = np.linspace(0.0, 1.5, 1501, dtype=float)
    lam = 10.0 + 6.0 * np.sin(2.0 * np.pi * 2.0 * t) + 2.0 * np.cos(2.0 * np.pi * 5.0 * t)
    lam = np.clip(lam, 0.2, None)

    lambda_sig = Covariate(time=t, data=lam, name="Lambda(t)", labels=["lambda"])
    spike_coll = CIF.simulateCIFByThinningFromLambda(lambda_sig, numRealizations=8)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1.5]}
    )
    ax1.plot(t, lam, color="tab:blue", linewidth=1.8)
    ax1.set_ylabel("rate (spikes/s)")
    ax1.set_title("Conditional Intensity and Simulated Spike Trains")
    ax1.grid(alpha=0.25)

    n_show = min(5, spike_coll.getNumUnits())
    raster_data = [np.asarray(spike_coll.trains[i].spike_times, dtype=float) for i in range(n_show)]
    offsets = np.arange(1, n_show + 1)
    ax2.eventplot(raster_data, lineoffsets=offsets, linelengths=0.75, colors="k")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("unit")
    ax2.set_yticks(offsets)
    ax2.set_ylim(0.5, n_show + 0.5)
    ax2.grid(alpha=0.25)
    fig.tight_layout()

    out = Path(__file__).resolve().parents[1] / "images" / "readme_example2_cif_simulation.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
