from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FigureBuilder = Callable[[dict[str, float], dict[str, object]], plt.Figure]


def default_export_dir(repo_root: Path, example_id: str) -> Path:
    return repo_root.resolve() / "docs" / "figures" / example_id


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def export_figure_set(
    *,
    summary: dict[str, float],
    payload: dict[str, object],
    export_dir: Path,
    builders: list[tuple[str, FigureBuilder]],
) -> list[Path]:
    written: list[Path] = []
    export_dir.mkdir(parents=True, exist_ok=True)
    for filename, builder in builders:
        fig = builder(summary, payload)
        written.append(save_figure(fig, export_dir / filename))
    return written


def _coerce_array(payload: dict[str, object], key: str) -> np.ndarray:
    return np.asarray(payload[key], dtype=float)


def build_example01_constant_mg_summary(summary: dict[str, float], payload: dict[str, object]) -> plt.Figure:
    time_s = _coerce_array(payload, "constant_time_s")
    rate_hz = _coerce_array(payload, "constant_rate_hz")
    acf_lags_s = _coerce_array(payload, "constant_acf_lags_s")
    acf_values = _coerce_array(payload, "constant_acf_values")
    acf_ci = float(summary["constant_acf_ci"])
    ks_ideal = _coerce_array(payload, "constant_ks_ideal")
    ks_empirical = _coerce_array(payload, "constant_ks_empirical")
    ks_ci = _coerce_array(payload, "constant_ks_ci")
    raster_spikes = _coerce_array(payload, "constant_spike_times_s")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    ax_raster, ax_acf, ax_ks, ax_rate = axes.ravel()

    ax_raster.vlines(raster_spikes, 0.0, 1.0, color="black", linewidth=0.55)
    ax_raster.set_xlim(float(time_s[0]), float(time_s[-1]))
    ax_raster.set_ylim(0.0, 1.0)
    ax_raster.set_title("Neural Raster with constant Mg$^{2+}$ Concentration")
    ax_raster.set_xlabel("time [s]")
    ax_raster.set_ylabel("mEPSCs")

    ax_acf.scatter(acf_lags_s, acf_values, s=6, color="tab:blue", alpha=0.85, linewidths=0.0)
    ax_acf.axhline(acf_ci, color="red", linewidth=1.2)
    ax_acf.axhline(-acf_ci, color="red", linewidth=1.2)
    ax_acf.set_title("Autocorrelation Function\nof Rescaled ISIs with 95% CIs")
    ax_acf.set_xlabel(r"$\Delta \tau$ [sec]")
    ax_acf.set_ylabel(r"ACF[$\tilde{b}^{-1}(u)$]")

    ax_ks.plot(ks_ideal, ks_empirical, color="#8a2be2", linewidth=1.5)
    ax_ks.plot([0.0, 1.0], [0.0, 1.0], color="black", linewidth=1.0, alpha=0.7)
    ax_ks.plot(ks_ideal, np.clip(ks_ideal + ks_ci, 0.0, 1.0), color="red", linewidth=1.0, alpha=0.85)
    ax_ks.plot(ks_ideal, np.clip(ks_ideal - ks_ci, 0.0, 1.0), color="red", linewidth=1.0, alpha=0.85)
    ax_ks.set_xlim(0.0, 1.0)
    ax_ks.set_ylim(0.0, 1.0)
    ax_ks.set_title("KS Plot of Rescaled ISIs\nwith 95% Confidence Intervals")
    ax_ks.set_xlabel("Ideal Uniform CDF")
    ax_ks.set_ylabel("Empirical CDF")

    ax_rate.plot(time_s, rate_hz, color="tab:blue", linewidth=1.4)
    ax_rate.set_xlim(float(time_s[0]), float(time_s[-1]))
    ax_rate.set_title(r"$\lambda_{\mathrm{const}}$ baseline")
    ax_rate.set_xlabel("time [s]")
    ax_rate.set_ylabel(r"$\lambda(t)$ [Hz]")

    fig.tight_layout()
    return fig


def build_example01_washout_raster(summary: dict[str, float], payload: dict[str, object]) -> plt.Figure:
    const_spikes = _coerce_array(payload, "constant_spike_times_s")
    washout_spikes = _coerce_array(payload, "washout_spike_times_s")
    const_window = tuple(np.asarray(payload["constant_window_s"], dtype=float).tolist())
    washout_window = tuple(np.asarray(payload["washout_window_s"], dtype=float).tolist())

    fig, (ax_const, ax_washout) = plt.subplots(2, 1, figsize=(11.0, 7.2), sharey=True)

    ax_const.vlines(const_spikes, 0.0, 1.0, color="black", linewidth=0.5)
    ax_const.set_xlim(*const_window)
    ax_const.set_ylim(0.0, 1.0)
    ax_const.set_title("Neural Raster with constant Mg$^{2+}$ Concentration")
    ax_const.set_ylabel("mEPSCs")

    ax_washout.vlines(washout_spikes, 0.0, 1.0, color="black", linewidth=0.45)
    ax_washout.set_xlim(*washout_window)
    ax_washout.set_ylim(0.0, 1.0)
    ax_washout.set_title("Neural Raster with decreasing Mg$^{2+}$ Concentration")
    ax_washout.set_xlabel("time [s]")
    ax_washout.set_ylabel("mEPSCs")

    fig.tight_layout()
    return fig


def build_example01_piecewise_baseline(summary: dict[str, float], payload: dict[str, object]) -> plt.Figure:
    time_s = _coerce_array(payload, "washout_time_s")
    observed_rate_hz = _coerce_array(payload, "washout_observed_rate_hz")
    piecewise_rate_hz = _coerce_array(payload, "washout_piecewise_rate_hz")
    piecewise_hist_rate_hz = _coerce_array(payload, "washout_piecewise_history_rate_hz")
    segment_edges_s = _coerce_array(payload, "washout_segment_edges_s")

    fig, ax = plt.subplots(1, 1, figsize=(11.0, 4.8))
    ax.plot(time_s, observed_rate_hz, color="black", linewidth=1.0, alpha=0.45, label="Observed rate")
    ax.plot(time_s, piecewise_rate_hz, color="tab:orange", linewidth=1.6, label="Piecewise baseline")
    ax.plot(time_s, piecewise_hist_rate_hz, color="tab:red", linewidth=1.4, label="Piecewise + history")
    for edge in segment_edges_s[1:-1]:
        ax.axvline(float(edge), color="tab:blue", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_title("Piecewise Baseline Comparison During Mg$^{2+}$ Washout")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("rate [Hz]")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def build_example02_data_overview(summary: dict[str, float], payload: dict[str, object]) -> plt.Figure:
    time_s = _coerce_array(payload, "time_s")
    spike_indicator = _coerce_array(payload, "spike_indicator")
    stimulus = _coerce_array(payload, "stimulus")
    velocity = _coerce_array(payload, "velocity")

    fig, axes = plt.subplots(3, 1, figsize=(11.2, 7.8), sharex=True)

    spike_times = time_s[spike_indicator > 0.5]
    axes[0].vlines(spike_times, 0.0, 1.0, color="black", linewidth=0.55)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Neural Raster")
    axes[0].set_ylabel("spikes")

    axes[1].plot(time_s, stimulus, color="black", linewidth=0.8)
    axes[1].set_title("Stimulus - Whisker Displacement")
    axes[1].set_ylabel("Displacement [mm]")

    axes[2].plot(time_s, velocity, color="black", linewidth=0.8)
    axes[2].axhline(0.0, color="0.75", linewidth=1.0)
    axes[2].set_title("Displacement Velocity")
    axes[2].set_ylabel("Velocity [mm/s]")
    axes[2].set_xlabel("time [s]")

    fig.tight_layout()
    return fig


def build_example02_model_comparison(summary: dict[str, float], payload: dict[str, object]) -> plt.Figure:
    lags_s = _coerce_array(payload, "xcorr_lags_s")
    xcorr_values = _coerce_array(payload, "xcorr_values")
    peak_lag_s = float(summary["peak_lag_seconds"])
    history_windows = _coerce_array(payload, "history_windows")
    ks_stats = _coerce_array(payload, "ks_stats")
    delta_aic = _coerce_array(payload, "delta_aic")
    delta_bic = _coerce_array(payload, "delta_bic")
    ks_ideal = _coerce_array(payload, "ks_ideal")
    ks_const = _coerce_array(payload, "ks_const_empirical")
    ks_stim = _coerce_array(payload, "ks_stim_empirical")
    ks_hist = _coerce_array(payload, "ks_hist_empirical")
    ks_ci = _coerce_array(payload, "ks_ci")
    coef_names = list(payload["coef_names"])
    coef_values = _coerce_array(payload, "coef_values")
    coef_lower = _coerce_array(payload, "coef_lower")
    coef_upper = _coerce_array(payload, "coef_upper")

    fig = plt.figure(figsize=(13.2, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 1.0], hspace=0.5, wspace=0.32)
    ax_xcorr = fig.add_subplot(gs[0, 0])
    right = gs[0, 1].subgridspec(3, 1, hspace=0.22)
    ax_ks_stat = fig.add_subplot(right[0, 0])
    ax_aic = fig.add_subplot(right[1, 0], sharex=ax_ks_stat)
    ax_bic = fig.add_subplot(right[2, 0], sharex=ax_ks_stat)
    ax_ks = fig.add_subplot(gs[1, 0])
    ax_coef = fig.add_subplot(gs[1, 1])

    ax_xcorr.plot(lags_s, xcorr_values, color="#6aa6d8", linewidth=1.1)
    ax_xcorr.scatter([peak_lag_s], [float(np.interp(peak_lag_s, lags_s, xcorr_values))], color="red", s=35, zorder=3)
    ax_xcorr.set_title(f"Cross Correlation Function - Peak at t={peak_lag_s:.3f} sec")
    ax_xcorr.set_xlabel("Lag [s]")
    ax_xcorr.set_ylabel("xcov")

    for ax, series, ylabel in (
        (ax_ks_stat, ks_stats, "KS statistic"),
        (ax_aic, delta_aic, r"$\Delta$ AIC"),
        (ax_bic, delta_bic, r"$\Delta$ BIC"),
    ):
        ax.plot(history_windows, series, color="#4c97d8", marker=".", linewidth=1.0, markersize=3)
        idx = int(np.argmin(series))
        ax.scatter([history_windows[idx]], [series[idx]], color="red", marker="x", s=24, linewidths=1.0)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    ax_ks_stat.set_title("Model Selection via change\nin KS Statistic, AIC, and BIC")
    ax_bic.set_xlabel("# History Windows, Q")

    ax_ks.plot(ks_ideal, ks_const, color="#7e2de1", linewidth=1.5, label=r"$\lambda_{\mathrm{const}}$")
    ax_ks.plot(ks_ideal, ks_stim, color="tab:green", linewidth=1.3, label=r"$\lambda_{\mathrm{const+stim}}$")
    ax_ks.plot(ks_ideal, ks_hist, color="#39bced", linewidth=1.3, label=r"$\lambda_{\mathrm{const+stim+hist}}$")
    ax_ks.plot([0.0, 1.0], [0.0, 1.0], color="black", linewidth=1.0, alpha=0.7)
    ax_ks.plot(ks_ideal, np.clip(ks_ideal + ks_ci, 0.0, 1.0), color="red", linewidth=1.0, alpha=0.85)
    ax_ks.plot(ks_ideal, np.clip(ks_ideal - ks_ci, 0.0, 1.0), color="red", linewidth=1.0, alpha=0.85)
    ax_ks.set_xlim(0.0, 1.0)
    ax_ks.set_ylim(0.0, 1.0)
    ax_ks.set_title("KS Plot of Rescaled ISIs\nwith 95% Confidence Intervals")
    ax_ks.set_xlabel("Ideal Uniform CDF")
    ax_ks.set_ylabel("Empirical CDF")
    ax_ks.legend(loc="lower right", fontsize=8)

    positions = np.arange(len(coef_names), dtype=float)
    yerr = np.vstack([coef_values - coef_lower, coef_upper - coef_values])
    ax_coef.errorbar(positions, coef_values, yerr=yerr, fmt="o", color="#f0ad2f", ecolor="#f0ad2f", capsize=2)
    sig = (coef_lower > 0.0) | (coef_upper < 0.0)
    ax_coef.scatter(positions[sig], np.full(np.sum(sig), np.max(coef_upper) * 1.05), color="red", marker="x", s=20, linewidths=0.8)
    ax_coef.axhline(0.0, color="0.75", linewidth=1.0)
    ax_coef.set_xticks(positions, coef_names, rotation=90)
    ax_coef.set_title("GLM Coefficients with 95% CIs (* p<0.05)")
    ax_coef.set_ylabel("GLM Fit Coefficients")
    ax_coef.grid(alpha=0.25, axis="y")

    return fig


EXAMPLE_FIGURE_BUILDERS: dict[str, list[tuple[str, FigureBuilder]]] = {
    "example01": [
        ("fig01_constant_mg_summary.png", build_example01_constant_mg_summary),
        ("fig02_washout_raster_overview.png", build_example01_washout_raster),
        ("fig03_piecewise_baseline_comparison.png", build_example01_piecewise_baseline),
    ],
    "example02": [
        ("fig01_data_overview.png", build_example02_data_overview),
        ("fig02_lag_and_model_comparison.png", build_example02_model_comparison),
    ],
}


def export_named_paper_figures(
    example_id: str,
    *,
    summary: dict[str, float],
    payload: dict[str, object],
    export_dir: Path,
) -> list[Path]:
    builders = EXAMPLE_FIGURE_BUILDERS.get(example_id)
    if builders is None:
        raise ValueError(f"No figure builders registered for {example_id}")
    return export_figure_set(summary=summary, payload=payload, export_dir=export_dir, builders=builders)


__all__ = [
    "default_export_dir",
    "export_named_paper_figures",
]
