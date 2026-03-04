import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nstat.compat.matlab import SignalObj


def main() -> None:
    rng = np.random.default_rng(0)
    fs_hz = 1000.0
    dt = 1.0 / fs_hz
    duration_s = 2.0
    time = np.arange(0.0, duration_s, dt, dtype=float)

    signal = (
        1.0 * np.sin(2.0 * np.pi * 10.0 * time)
        + 0.6 * np.sin(2.0 * np.pi * 40.0 * time + 0.3)
        + 0.2 * np.sin(2.0 * np.pi * 75.0 * time)
        + 0.12 * rng.standard_normal(time.size)
    )

    sig_obj = SignalObj(time=time, data=signal, name="synthetic_signal", units="a.u.")
    freq_hz, psd = sig_obj.MTMspectrum()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=False)

    preview_mask = time <= 1.0
    ax1.plot(time[preview_mask], signal[preview_mask], color="black", linewidth=1.0)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("amplitude")
    ax1.set_title("Synthetic signal (first 1 s)")

    ax2.plot(freq_hz, psd, color="tab:blue", linewidth=1.2)
    ax2.set_xlim(0.0, 150.0)
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylabel("power spectral density")
    ax2.set_title("Multi-taper spectrum")

    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "readme_example1_multitaper_spectrum.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
