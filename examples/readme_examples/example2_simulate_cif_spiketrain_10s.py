import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nstat.compat.matlab import CIF, Covariate


def _extract_first_spike_times(spike_obj: object) -> np.ndarray:
    if hasattr(spike_obj, "getNST"):
        try:
            first_train = spike_obj.getNST(0)
        except Exception:
            first_train = spike_obj.getNST(1)
        if hasattr(first_train, "spike_times"):
            return np.asarray(first_train.spike_times, dtype=float).reshape(-1)

    if hasattr(spike_obj, "trains"):
        trains = getattr(spike_obj, "trains")
        if len(trains) > 0:
            first_train = trains[0]
            if hasattr(first_train, "spike_times"):
                return np.asarray(first_train.spike_times, dtype=float).reshape(-1)
            return np.asarray(first_train, dtype=float).reshape(-1)

    if isinstance(spike_obj, (list, tuple)) and len(spike_obj) > 0:
        return np.asarray(spike_obj[0], dtype=float).reshape(-1)

    return np.asarray([], dtype=float)


def main() -> None:
    np.random.seed(0)

    duration_s = 10.0
    dt = 0.001
    t = np.arange(0.0, duration_s + dt, dt, dtype=float)

    f_hz = 0.5
    baseline_hz = 15.0
    amp_hz = 10.0
    lam = np.clip(baseline_hz + amp_hz * np.sin(2.0 * np.pi * f_hz * t), 0.2, None)

    lambda_cov = Covariate(time=t, data=lam, name="Lambda", yunits="spikes/s", dataLabels=["lambda"])
    spikes = CIF.simulateCIFByThinningFromLambda(lambda_cov, 1, dt)
    spike_times = _extract_first_spike_times(spikes)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8.0, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    ax1.plot(t, lam, color="tab:blue", linewidth=1.3)
    ax1.set_ylabel("rate (spikes/s)")
    ax1.set_title("Time-varying CIF over 10 s")

    ax2.vlines(spike_times, 0.0, 1.0, color="black", linewidth=0.8)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_yticks([])
    ax2.set_xlabel("time (s)")
    ax2.set_title("Simulated spike train")

    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "readme_example2_simulate_cif_spiketrain_10s.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
