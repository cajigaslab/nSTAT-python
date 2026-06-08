#!/usr/bin/env python3
"""Generate the teaching figures embedded in the Concepts pages.

Reproducible: run from anywhere to regenerate every PNG under
``docs/concepts/figures/``.

    python docs/concepts/figures/make_concept_figures.py

The figures are illustrative (simulated data, fixed seed) and are meant to
build intuition alongside the prose — see ``docs/concepts/*.md``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nstat import (  # noqa: E402
    SignalObj, simulate_cif_from_stimulus, fit_poisson_glm, population_time_rescale,
    DecodingAlgorithms,
)

OUT = Path(__file__).resolve().parent
ACCENT = "#2c5282"
RNG = np.random.default_rng(7)


def fig_signal_split() -> None:
    """Broadband extracellular signal -> spikes (high-pass) + LFP (low-pass)."""
    from scipy.signal import butter, filtfilt

    fs = 30000.0
    t = np.arange(0, 0.5, 1 / fs)
    lfp = 80 * np.sin(2 * np.pi * 8 * t) + 40 * np.sin(2 * np.pi * 30 * t)
    # Add a few fast biphasic "spikes".
    broadband = lfp + 8 * RNG.standard_normal(t.size)
    spike_times = np.array([0.06, 0.13, 0.21, 0.27, 0.34, 0.39, 0.46])
    w = 0.0006
    for st in spike_times:
        i = int(st * fs)
        n = int(w * fs)
        wave = -np.hanning(2 * n) * 220
        wave[n:] *= -0.5
        broadband[i - n:i + n] += wave[: 2 * n]

    b_hp, a_hp = butter(3, 300 / (fs / 2), btype="high")
    b_lp, a_lp = butter(3, 300 / (fs / 2), btype="low")
    spikes = filtfilt(b_hp, a_hp, broadband)
    lfp_band = filtfilt(b_lp, a_lp, broadband)

    fig, axes = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(t, broadband, lw=0.4, color="0.3")
    axes[0].set_title("One electrode, two signals: separating by frequency")
    axes[0].set_ylabel("broadband\n(µV)")
    axes[1].plot(t, spikes, lw=0.4, color=ACCENT)
    axes[1].set_ylabel("high-pass\n>300 Hz = spikes")
    axes[2].plot(t, lfp_band, lw=0.9, color="#dd6b20")
    axes[2].set_ylabel("low-pass\n<300 Hz = LFP")
    axes[2].set_xlabel("time (s)")
    for ax in axes:
        ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(OUT / "signal_split.png", dpi=120)
    plt.close(fig)


def fig_cif_raster() -> None:
    """Stimulus -> conditional intensity -> spike raster over repeated trials."""
    dt = 0.001
    t = np.arange(0, 2.0, dt)
    stim = np.sin(2 * np.pi * 2.0 * t)
    b0, b1 = 2.6, 1.4
    cif = np.exp(b0 + b1 * stim)  # spikes/s

    fig, axes = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.4]})
    axes[0].plot(t, cif, color=ACCENT, lw=1.2)
    axes[0].set_ylabel("CIF λ(t)\n(spikes/s)")
    axes[0].set_title("Stimulus-driven firing rate and the spikes it generates")
    n_trials = 25
    for k in range(n_trials):
        sp, _, _ = simulate_cif_from_stimulus(time=t, stimulus=stim,
                                               beta0=b0, beta1=b1,
                                               rng=np.random.default_rng(100 + k))
        st = np.asarray(sp.getSpikeTimes() if hasattr(sp, "getSpikeTimes")
                        else sp.spikeTimes)
        axes[1].vlines(st, k + 0.5, k + 1.5, color="0.2", lw=0.6)
    axes[1].set_ylim(0.5, n_trials + 0.5)
    axes[1].set_ylabel("trial")
    axes[1].set_xlabel("time (s)")
    for ax in axes:
        ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(OUT / "cif_raster.png", dpi=120)
    plt.close(fig)


def fig_multitaper() -> None:
    """Multitaper spectrum + spectrogram of a toy LFP (8 Hz + 40 Hz)."""
    fs = 1000.0
    t = np.arange(0, 6.0, 1 / fs)
    gamma_env = (t > 2.0) & (t < 4.0)        # gamma only in the middle
    x = (np.sin(2 * np.pi * 8 * t)
         + 0.6 * gamma_env * np.sin(2 * np.pi * 40 * t)
         + 0.3 * RNG.standard_normal(t.size))
    lfp = SignalObj(t, x, name="LFP")
    f, power, _ = lfp.MTMspectrum(NW=4.0)
    sf, stime, Sxx = lfp.spectrogram(nperseg=256)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].semilogy(f, power, color=ACCENT, lw=1)
    axes[0].set(xlim=(0, 80), xlabel="frequency (Hz)", ylabel="power (log)",
                title="Multitaper spectrum")
    for fpk in (8, 40):
        axes[0].axvline(fpk, color="#dd6b20", ls="--", lw=0.8)
    m = axes[1].pcolormesh(stime, sf, 10 * np.log10(Sxx + 1e-12), shading="auto")
    axes[1].set(ylim=(0, 80), xlabel="time (s)", ylabel="frequency (Hz)",
                title="Spectrogram (gamma burst 2–4 s)")
    fig.colorbar(m, ax=axes[1], label="dB")
    fig.tight_layout()
    fig.savefig(OUT / "multitaper_spectrum.png", dpi=120)
    plt.close(fig)


def fig_ks_plot() -> None:
    """KS goodness-of-fit plot: correct vs. misspecified (constant-rate) model."""
    dt = 0.001
    t = np.arange(0, 120.0, dt)
    stim = np.sin(2 * np.pi * 1.0 * t)
    sp, _, _ = simulate_cif_from_stimulus(time=t, stimulus=stim,
                                          beta0=2.5, beta1=1.5,
                                          rng=np.random.default_rng(7))
    edges = np.arange(0, 120.0 + dt, dt)
    y = sp.to_binned_counts(edges)
    x = stim[:, None]
    offset = np.full(y.shape, np.log(dt))
    fit = fit_poisson_glm(x, y, offset=offset, l2=1e-4, max_iter=100, tol=1e-10)
    lam_ok = np.exp(fit.intercept + fit.coefficients[0] * x[:, 0] + offset)
    lam_bad = np.full_like(y, y.mean())

    g_ok = population_time_rescale([y], [lam_ok])
    g_bad = population_time_rescale([y], [lam_bad])

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    n = None
    for res, color, label in [(g_ok, ACCENT, "correct model"),
                              (g_bad, "#e53e3e", "constant-rate model")]:
        u = np.sort(np.asarray(res.ground_uniforms))
        n = u.size
        emp = (np.arange(1, n + 1) - 0.5) / n
        ax.plot(u, emp, color=color, lw=1.5,
                label=f"{label} (KS p={res.ground_ks_pvalue:.2g})")
    # 95% KS confidence band around the diagonal
    band = 1.36 / np.sqrt(n)
    xx = np.linspace(0, 1, 100)
    ax.plot([0, 1], [0, 1], color="0.4", lw=1)
    ax.plot(xx, np.clip(xx + band, 0, 1), color="0.6", ls="--", lw=0.8)
    ax.plot(xx, np.clip(xx - band, 0, 1), color="0.6", ls="--", lw=0.8)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="model quantile (rescaled)",
           ylabel="empirical quantile",
           title="Time-rescaling KS plot")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / "ks_plot.png", dpi=120)
    plt.close(fig)


def fig_ssglm_drift() -> None:
    """Illustrate a GLM coefficient drifting across trials (a latent state)."""
    n_trials = 60
    rng = np.random.default_rng(3)
    true_state = np.cumsum(0.06 * rng.standard_normal(n_trials)) + 0.5
    noisy = true_state + 0.25 * rng.standard_normal(n_trials)
    # crude causal smoother to mimic a state estimate
    est = np.copy(noisy)
    for k in range(1, n_trials):
        est[k] = 0.7 * est[k - 1] + 0.3 * noisy[k]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    trials = np.arange(1, n_trials + 1)
    ax.plot(trials, noisy, "o", ms=3, color="0.6",
            label="per-trial GLM estimate (noisy)")
    ax.plot(trials, true_state, color="#dd6b20", lw=2, label="true latent state")
    ax.plot(trials, est, color=ACCENT, lw=2, label="SSGLM smoothed estimate")
    ax.set(xlabel="trial", ylabel="stimulus coefficient",
           title="State-space GLM: tuning that evolves as the animal learns")
    ax.legend(loc="upper left", fontsize=8)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(OUT / "ssglm_drift.png", dpi=120)
    plt.close(fig)


def fig_decoding() -> None:
    """PPAF decode of a hidden stimulus from a 20-cell population vs. truth."""
    rng = np.random.default_rng(0)
    delta = 0.001
    time = np.arange(0.0, 1.0 + delta, delta)
    stim = np.sin(2 * np.pi * 2.0 * time)
    n_cells = 20
    b1 = rng.standard_normal(n_cells)
    b0 = np.log(10.0 * delta) + rng.standard_normal(n_cells)
    dN = np.zeros((n_cells, time.size))
    for c in range(n_cells):
        eta = np.clip(b0[c] + b1[c] * stim, -20.0, 20.0)
        p = np.exp(eta) / (1.0 + np.exp(eta))
        dN[c, :] = (rng.random(time.size) < p).astype(float)

    A = np.array([[1.0]])
    Q = np.array([[float(np.std(np.diff(stim)))]])
    _, _, x_u, W_u, *_ = DecodingAlgorithms.PPDecodeFilterLinear(
        A, Q, dN, b0, b1.reshape(1, -1), "binomial", delta, None, None,
        np.array([0.0]), 0.5 * np.eye(1),
    )
    x_hat = x_u[0, :]
    sigma = np.sqrt(np.maximum(W_u[0, 0, :], 0.0))
    rmse = float(np.sqrt(np.mean((x_hat - stim) ** 2)))

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(time, stim, color="#dd6b20", lw=2, label="true stimulus")
    ax.plot(time, x_hat, color=ACCENT, lw=1.5, label="PPAF decode (20 cells)")
    ax.fill_between(time, x_hat - 1.96 * sigma, x_hat + 1.96 * sigma,
                    color=ACCENT, alpha=0.2, label="95% credible band")
    ax.set(xlabel="time (s)", ylabel="stimulus", xlim=(time[0], time[-1]),
           title=f"Decoding a hidden stimulus from population spikes (RMSE={rmse:.2f})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "decoding.png", dpi=120)
    plt.close(fig)


def fig_workflow() -> None:
    """Schematic of the nSTAT object model / analysis pipeline."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    stages = [
        ("Raw data", "spike times\ncovariates / stimuli\nLFP / EEG / ECoG", "#718096"),
        ("nSTAT objects", "nspikeTrain · nstColl\nCovariate · CovColl\nSignalObj", ACCENT),
        ("Trial\n+ TrialConfig", "bundle data +\nmodel specification", ACCENT),
        ("Analysis", "fit the\npoint-process GLM", ACCENT),
        ("FitResult", "coefficients, CIF,\ndiagnostics", ACCENT),
    ]
    outputs = [
        ("Goodness-of-fit", "computeKSStats\npopulation_time_rescale"),
        ("Model comparison", "AIC / BIC"),
        ("Decoding", "PPAF / PPHF\nclusterless"),
    ]

    fig, ax = plt.subplots(figsize=(12.6, 3.6))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    w, h, y = 1.7, 1.3, 1.5
    xs = [0.15 + i * 1.95 for i in range(len(stages))]
    for i, (title, sub, color) in enumerate(stages):
        x = xs[i]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    linewidth=1.5, edgecolor=color, facecolor="white"))
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=color)
        ax.text(x + w / 2, y + 0.42, sub, ha="center", va="center", fontsize=7.2, color="0.25")
        if i > 0:
            ax.add_patch(FancyArrowPatch((xs[i - 1] + w, y + h / 2), (x, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=14, color="0.4", lw=1.4))

    # Three outputs branching from FitResult, stacked at the right.
    fr_x = xs[-1] + w
    oy = [2.55, 1.55, 0.55]
    ox = fr_x + 0.55
    ow, oh = 2.0, 0.78
    for (title, sub), yy in zip(outputs, oy):
        ax.add_patch(FancyBboxPatch((ox, yy - oh / 2), ow, oh,
                                    boxstyle="round,pad=0.03,rounding_size=0.1",
                                    linewidth=1.3, edgecolor="#dd6b20", facecolor="#fff7ee"))
        ax.text(ox + ow / 2, yy + 0.14, title, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="#9c4221")
        ax.text(ox + ow / 2, yy - 0.16, sub, ha="center", va="center", fontsize=6.8, color="0.3")
        ax.add_patch(FancyArrowPatch((fr_x, y + h / 2), (ox, yy),
                                     arrowstyle="-|>", mutation_scale=12, color="#dd6b20",
                                     lw=1.2, connectionstyle="arc3,rad=0.0"))

    ax.text(6.3, 3.42, "The nSTAT analysis pipeline", ha="center", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "workflow.png", dpi=120)
    plt.close(fig)


def main() -> int:
    fig_workflow()
    fig_signal_split()
    fig_cif_raster()
    fig_multitaper()
    fig_ks_plot()
    fig_ssglm_drift()
    fig_decoding()
    print("Wrote concept figures to", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
