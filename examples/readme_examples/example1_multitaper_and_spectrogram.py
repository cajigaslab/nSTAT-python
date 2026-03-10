import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from nstat.compat.matlab import SignalObj


def _fallback_multitaper_psd(signal: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal.windows import dpss

    n = signal.size
    tapers = dpss(n, NW=3.5, Kmax=4, sym=False)
    centered = signal - np.mean(signal)
    tapered = tapers * centered[np.newaxis, :]
    fft_vals = np.fft.rfft(tapered, axis=1)
    psd = np.mean(np.abs(fft_vals) ** 2, axis=0) / (fs_hz * n)
    freq = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    return freq, psd


def main() -> None:
    fs_hz = 1000.0
    duration_s = 2.0
    f0_hz = 10.0
    dt = 1.0 / fs_hz
    time = np.arange(0.0, duration_s, dt, dtype=float)

    signal = np.sin(2.0 * np.pi * f0_hz * time)
    sig_obj = SignalObj(time=time, data=signal, name="sine_signal", yunits="a.u.")

    try:
        freq_hz, psd = sig_obj.MTMspectrum()
    except Exception:
        freq_hz, psd = _fallback_multitaper_psd(signal, fs_hz)

    f_spec, t_spec, sxx = spectrogram(
        signal,
        fs=fs_hz,
        nperseg=256,
        noverlap=224,
        scaling="density",
        mode="psd",
    )

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 7.5))

    preview = time <= 1.0
    axes[0].plot(time[preview], signal[preview], color="tab:blue", linewidth=1.4)
    axes[0].set_title("Signal (10 Hz sinusoid)")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("amplitude")

    axes[1].plot(freq_hz, psd, color="tab:orange", linewidth=1.2)
    axes[1].set_xlim(0.0, 100.0)
    axes[1].set_title("Multi-taper spectrum")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("PSD")

    im = axes[2].pcolormesh(t_spec, f_spec, sxx, shading="auto", cmap="magma")
    axes[2].set_ylim(0.0, 100.0)
    axes[2].set_title("Spectrogram")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("frequency (Hz)")
    fig.colorbar(im, ax=axes[2], pad=0.01, label="PSD")

    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "readme_example1_multitaper_and_spectrogram.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
