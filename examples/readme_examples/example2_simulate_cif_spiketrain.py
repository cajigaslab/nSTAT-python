import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nstat.compat.matlab import CIF, Covariate


def main() -> None:
    np.random.seed(0)

    dt = 0.001
    duration_s = 2.0
    time = np.arange(0.0, duration_s + 0.5 * dt, dt, dtype=float)

    lambda_t = (
        12.0
        + 5.5 * np.sin(2.0 * np.pi * 2.0 * time)
        + 1.5 * np.cos(2.0 * np.pi * 6.0 * time)
    )
    lambda_t = np.clip(lambda_t, 0.1, None)

    lambda_cov = Covariate(
        time=time,
        data=lambda_t,
        name="Lambda(t)",
        units="spikes/s",
        labels=["lambda"],
    )

    coll = CIF.simulateCIFByThinningFromLambda(lambda_cov, 1, dt)
    spike_times = coll.getNST(0).spike_times

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(7.0, 5.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    ax1.plot(time, lambda_t, color="tab:blue", linewidth=1.4)
    ax1.set_ylabel("rate (spikes/s)")
    ax1.set_title("Time-varying CIF")

    ax2.vlines(spike_times, 0.0, 1.0, color="black", linewidth=0.8)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("spikes")
    ax2.set_title("Simulated spike train")

    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "readme_example2_simulate_cif_spiketrain.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
