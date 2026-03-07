from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


FigureBuilder = Callable[[dict[str, object], dict[str, object]], plt.Figure]


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


def build_example03_simulated_and_real_rasters(summary: dict[str, object], payload: dict[str, object]) -> plt.Figure:
    e3 = payload["experiment3"]
    e3b = payload["experiment3b"]
    time_s = np.asarray(e3["time_s"], dtype=float)
    true_rate_hz = np.asarray(e3["true_rate_hz"], dtype=float)
    raster_spike_times = e3["raster_spike_times"]
    stimulus = np.asarray(e3b["stimulus"], dtype=float)
    xk = np.asarray(e3b["xk"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0))
    ax_rate, ax_real, ax_sim, ax_est = axes.ravel()

    ax_rate.plot(time_s, true_rate_hz, color="tab:blue", linewidth=1.6)
    ax_rate.set_title("Simulated Conditional Intensity Function (CIF)")
    ax_rate.set_xlabel("time [s]")
    ax_rate.set_ylabel(r"$\lambda(t)$ [spikes/sec]")

    for row, spikes in enumerate(raster_spike_times[:20], start=1):
        spikes_arr = np.asarray(spikes, dtype=float)
        if spikes_arr.size:
            ax_sim.vlines(spikes_arr, row - 0.4, row + 0.4, color="0.45", linewidth=0.7)
    ax_sim.set_title("20 Simulated Point Process Sample Paths")
    ax_sim.set_xlabel("time [s]")
    ax_sim.set_ylabel("Trial [k]")

    im1 = ax_real.imshow(stimulus, aspect="auto", origin="lower", cmap="gray_r")
    ax_real.set_title("Stimulus Across Trials")
    ax_real.set_xlabel("time bin")
    ax_real.set_ylabel("Trial [k]")
    fig.colorbar(im1, ax=ax_real, fraction=0.046, pad=0.04)

    im2 = ax_est.imshow(xk, aspect="auto", origin="lower", cmap="magma")
    ax_est.set_title("Estimated Trial Dynamics")
    ax_est.set_xlabel("time bin")
    ax_est.set_ylabel("Trial [k]")
    fig.colorbar(im2, ax=ax_est, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


def build_example03_psth_comparison(summary: dict[str, object], payload: dict[str, object]) -> plt.Figure:
    e3 = payload["experiment3"]
    time_s = np.asarray(e3["time_s"], dtype=float)
    true_rate_hz = np.asarray(e3["true_rate_hz"], dtype=float)
    psth_bin_centers_s = np.asarray(e3["psth_bin_centers_s"], dtype=float)
    psth_rate_hz = np.asarray(e3["psth_rate_hz"], dtype=float)
    raster_spike_times = e3["raster_spike_times"]

    fig, (ax_rate, ax_raster) = plt.subplots(2, 1, figsize=(10.8, 7.0), sharex=False)
    ax_rate.plot(time_s, true_rate_hz, color="black", linewidth=1.3, label="True CIF")
    ax_rate.step(psth_bin_centers_s, psth_rate_hz, where="mid", color="tab:blue", linewidth=2.0, label="PSTH")
    ax_rate.set_title("PSTH Comparison")
    ax_rate.set_xlabel("time [s]")
    ax_rate.set_ylabel("rate [Hz]")
    ax_rate.grid(alpha=0.25)
    ax_rate.legend(loc="upper right")

    for row, spikes in enumerate(raster_spike_times[:10], start=1):
        spikes_arr = np.asarray(spikes, dtype=float)
        if spikes_arr.size:
            ax_raster.vlines(spikes_arr, row - 0.4, row + 0.4, color="black", linewidth=0.65)
    ax_raster.set_title("Simulated Raster (first 10 trials)")
    ax_raster.set_xlabel("time [s]")
    ax_raster.set_ylabel("Trial [k]")
    ax_raster.set_ylim(0.5, 10.5)
    ax_raster.grid(alpha=0.15)

    fig.tight_layout()
    return fig


def build_example03_ssglm_simulation_summary(summary: dict[str, object], payload: dict[str, object]) -> plt.Figure:
    e3b = payload["experiment3b"]
    stimulus = np.asarray(e3b["stimulus"], dtype=float)
    xk = np.asarray(e3b["xk"], dtype=float)
    ci_width = np.asarray(e3b["ci_width"], dtype=float)
    qhat = np.asarray(e3b["qhat"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2))
    im0 = axes[0, 0].imshow(stimulus, aspect="auto", origin="lower", cmap="gray_r")
    axes[0, 0].set_title("Observed Stimulus")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(xk, aspect="auto", origin="lower", cmap="viridis")
    axes[0, 1].set_title("Estimated State")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(ci_width, aspect="auto", origin="lower", cmap="magma")
    axes[1, 0].set_title("95% CI Width")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].plot(np.arange(1, qhat.size + 1), qhat, color="tab:blue", linewidth=1.5, marker="o")
    axes[1, 1].set_title("Qhat by Trial")
    axes[1, 1].set_xlabel("Trial [k]")
    axes[1, 1].set_ylabel("Qhat")
    axes[1, 1].grid(alpha=0.25)

    fig.tight_layout()
    return fig


def build_example03_ssglm_fit_diagnostics(summary: dict[str, object], payload: dict[str, object]) -> plt.Figure:
    e3b = payload["experiment3b"]
    stimulus = np.asarray(e3b["stimulus"], dtype=float)
    xk = np.asarray(e3b["xk"], dtype=float)
    qhat_all = np.asarray(e3b["qhat_all"], dtype=float)
    gammahat_all = np.asarray(e3b["gammahat_all"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))
    axes[0, 0].plot(stimulus.mean(axis=0), color="black", linewidth=1.5, label="Observed mean")
    axes[0, 0].plot(xk.mean(axis=0), color="tab:blue", linewidth=1.5, label="Estimated mean")
    axes[0, 0].set_title("Mean Trial Dynamics")
    axes[0, 0].set_xlabel("time bin")
    axes[0, 0].set_ylabel("state")
    axes[0, 0].legend(loc="upper right")

    axes[0, 1].plot(np.arange(1, qhat_all.shape[0] + 1), qhat_all.mean(axis=1), color="tab:orange", linewidth=1.5)
    axes[0, 1].set_title("Mean Qhat Across Fits")
    axes[0, 1].set_xlabel("Trial [k]")
    axes[0, 1].set_ylabel("mean Qhat")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].bar(np.arange(1, gammahat_all.size + 1), gammahat_all, color="tab:green", alpha=0.8)
    axes[1, 0].set_title("Gammahat Across Fits")
    axes[1, 0].set_xlabel("fit index")
    axes[1, 0].set_ylabel("gammahat")

    residual = xk - stimulus
    axes[1, 1].hist(residual.ravel(), bins=40, color="tab:purple", alpha=0.8)
    axes[1, 1].set_title("State Residual Histogram")
    axes[1, 1].set_xlabel("estimate - stimulus")
    axes[1, 1].set_ylabel("count")

    fig.tight_layout()
    return fig


def build_example03_stimulus_effect_surfaces(summary: dict[str, object], payload: dict[str, object]) -> plt.Figure:
    e3b = payload["experiment3b"]
    stimulus = np.asarray(e3b["stimulus"], dtype=float)
    xk = np.asarray(e3b["xk"], dtype=float)
    ci_width = np.asarray(e3b["ci_width"], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    for ax, arr, title, cmap in (
        (axes[0], stimulus, "Stimulus Surface", "gray_r"),
        (axes[1], xk, "Estimated State Surface", "viridis"),
        (axes[2], ci_width, "CI Width Surface", "magma"),
    ):
        im = ax.imshow(arr, aspect="auto", origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("time bin")
        ax.set_ylabel("Trial [k]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


def build_example03_learning_trial_comparison(summary: dict[str, object], payload: dict[str, object]) -> plt.Figure:
    e3b = payload["experiment3b"]
    stimulus = np.asarray(e3b["stimulus"], dtype=float)
    xk = np.asarray(e3b["xk"], dtype=float)
    ci_width = np.asarray(e3b["ci_width"], dtype=float)

    trial_means = xk.mean(axis=1)
    baseline_idx = 0
    learning_idx = int(min(6, xk.shape[0] - 1))
    late_idx = int(xk.shape[0] - 1)

    fig = plt.figure(figsize=(12.4, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], hspace=0.4, wspace=0.28)
    ax_trial = fig.add_subplot(gs[0, 0])
    ax_compare = fig.add_subplot(gs[1, 0])
    ax_heat = fig.add_subplot(gs[:, 1])

    ax_trial.plot(np.arange(1, trial_means.size + 1), trial_means, color="black", linewidth=2.0)
    ax_trial.axvline(learning_idx + 1, color="red", linewidth=1.2)
    ax_trial.set_title(f"Learning Trial:{learning_idx + 1}")
    ax_trial.set_xlabel("Trial [k]")
    ax_trial.set_ylabel("Average Firing Rate [spikes/sec]")

    x_axis = np.linspace(0.0, 1.0, xk.shape[1], dtype=float)
    ax_compare.plot(x_axis, xk[baseline_idx], color="black", linewidth=2.0, label=r"$\lambda_1(t)$")
    ax_compare.plot(x_axis, xk[learning_idx], color="red", linewidth=2.0, label=r"$\lambda_7(t)$")
    ax_compare.plot(x_axis, xk[late_idx], color="0.4", linewidth=1.3, label=r"$\lambda_K(t)$")
    ax_compare.fill_between(
        x_axis,
        xk[learning_idx] - 0.5 * ci_width[learning_idx],
        xk[learning_idx] + 0.5 * ci_width[learning_idx],
        color="red",
        alpha=0.15,
    )
    ax_compare.set_title("Learning Trial Vs. Baseline Trial\nwith 95% CIs")
    ax_compare.set_xlabel("time [s]")
    ax_compare.set_ylabel("Firing Rate [spikes/sec]")
    ax_compare.legend(loc="upper right", fontsize=8)

    im = ax_heat.imshow(np.abs(xk - stimulus), aspect="auto", origin="lower", cmap="gray_r")
    ax_heat.set_title("Trial Comparison Error Map")
    ax_heat.set_xlabel("time bin")
    ax_heat.set_ylabel("Trial Number")
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    return fig


def _placecell_overlay(ax: plt.Axes, animal: dict[str, object], title: str) -> None:
    x_pos = np.asarray(animal["x_pos"], dtype=float)
    y_pos = np.asarray(animal["y_pos"], dtype=float)
    time_s = np.asarray(animal["time_s"], dtype=float)
    spike_times = np.asarray(animal["spike_times"][0], dtype=float)
    spike_x = np.interp(spike_times, time_s, x_pos)
    spike_y = np.interp(spike_times, time_s, y_pos)
    ax.plot(x_pos, y_pos, color="0.25", linewidth=0.5, alpha=0.55, label="Animal Path")
    ax.scatter(spike_x, spike_y, s=7, color="red", alpha=0.7, label="Spike Locations")
    ax.set_title(title)
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_aspect("equal", adjustable="box")


def _placecell_field_grid(fig_title: str, animal: dict[str, object], field_key: str, cmap: str) -> plt.Figure:
    fields = np.asarray(animal[field_key], dtype=float)
    grid_x = np.asarray(animal["grid_x"], dtype=float)
    grid_y = np.asarray(animal["grid_y"], dtype=float)
    indices = np.asarray(animal["selected_indices"], dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2))
    for ax, field, cell_idx in zip(axes.ravel(), fields, indices, strict=False):
        im = ax.imshow(
            field,
            origin="lower",
            extent=[float(grid_x.min()), float(grid_x.max()), float(grid_y.min()), float(grid_y.max())],
            cmap=cmap,
            aspect="auto",
        )
        ax.set_title(f"Cell {int(cell_idx) + 1}")
        ax.set_xlabel("x position")
        ax.set_ylabel("y position")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(fig_title)
    return fig


def build_example04_path_overlay(summary: dict[str, object], payload: dict[str, object]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))
    _placecell_overlay(axes[0], payload["animal1"], "Animal 1, Example Cell Overlay")
    _placecell_overlay(axes[1], payload["animal2"], "Animal 2, Example Cell Overlay")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def build_example04_model_summary(summary: dict[str, object], payload: dict[str, object]) -> plt.Figure:
    animal1 = payload["animal1"]
    animal2 = payload["animal2"]
    indices1 = np.asarray(animal1["selected_indices"], dtype=int) + 1
    indices2 = np.asarray(animal2["selected_indices"], dtype=int) + 1

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axes[0].plot(indices1, np.asarray(animal1["delta_aic"], dtype=float), marker="o", linewidth=1.4, label=r"$\Delta$ AIC")
    axes[0].plot(indices1, np.asarray(animal1["delta_bic"], dtype=float), marker="s", linewidth=1.4, label=r"$\Delta$ BIC")
    axes[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axes[0].set_title("Animal 1 Model Delta")
    axes[0].set_xlabel("Cell index")
    axes[0].set_ylabel("Gaussian - Zernike")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(indices2, np.asarray(animal2["delta_aic"], dtype=float), marker="o", linewidth=1.4, label=r"$\Delta$ AIC")
    axes[1].plot(indices2, np.asarray(animal2["delta_bic"], dtype=float), marker="s", linewidth=1.4, label=r"$\Delta$ BIC")
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axes[1].set_title("Animal 2 Model Delta")
    axes[1].set_xlabel("Cell index")
    axes[1].set_ylabel("Gaussian - Zernike")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    return fig


def build_example04_gaussian_animal1(summary: dict[str, object], payload: dict[str, object]) -> plt.Figure:
    return _placecell_field_grid("Gaussian Place Fields - Animal 1", payload["animal1"], "gaussian_fields", "viridis")


def build_example04_zernike_animal1(summary: dict[str, object], payload: dict[str, object]) -> plt.Figure:
    return _placecell_field_grid("Zernike Place Fields - Animal 1", payload["animal1"], "zernike_fields", "magma")


def build_example04_gaussian_animal2(summary: dict[str, object], payload: dict[str, object]) -> plt.Figure:
    return _placecell_field_grid("Gaussian Place Fields - Animal 2", payload["animal2"], "gaussian_fields", "viridis")


def build_example04_zernike_animal2(summary: dict[str, object], payload: dict[str, object]) -> plt.Figure:
    return _placecell_field_grid("Zernike Place Fields - Animal 2", payload["animal2"], "zernike_fields", "magma")


def build_example04_mesh_comparison(summary: dict[str, object], payload: dict[str, object]) -> plt.Figure:
    mesh = payload["mesh"]
    grid_x = np.asarray(mesh["grid_x"], dtype=float)
    grid_y = np.asarray(mesh["grid_y"], dtype=float)
    gaussian_field = np.asarray(mesh["gaussian_field"], dtype=float)
    zernike_field = np.asarray(mesh["zernike_field"], dtype=float)
    x_pos = np.asarray(mesh["x_pos"], dtype=float)
    y_pos = np.asarray(mesh["y_pos"], dtype=float)
    time_s = np.asarray(mesh["time_s"], dtype=float)
    spike_times = np.asarray(mesh["spike_times"], dtype=float)
    spike_x = np.interp(spike_times, time_s, x_pos)
    spike_y = np.interp(spike_times, time_s, y_pos)

    fig = plt.figure(figsize=(11.0, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(grid_x, grid_y, gaussian_field, rstride=1, cstride=1, color="blue", alpha=0.18, linewidth=0.4)
    ax.plot_wireframe(grid_x, grid_y, zernike_field, rstride=1, cstride=1, color="limegreen", alpha=0.18, linewidth=0.4)
    ax.plot(x_pos, y_pos, zs=0.0, zdir="z", color="black", linewidth=0.4, alpha=0.55)
    ax.scatter(spike_x, spike_y, zs=0.0, zdir="z", color="red", s=4, alpha=0.7)
    ax.set_title(f"Animal#1, Cell#{int(mesh['cell_index']) + 1}")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_zlabel("rate")
    return fig


def build_example05_univariate_setup(summary: dict[str, object], payload: dict[str, object]) -> plt.Figure:
    e5 = payload["experiment5"]
    time_s = np.asarray(e5["time_s"], dtype=float)
    stimulus = np.asarray(e5["stimulus"], dtype=float)
    spikes = np.asarray(e5["spikes"], dtype=float)

    fig, (ax_stim, ax_raster) = plt.subplots(2, 1, figsize=(11.0, 6.6), sharex=True)
    ax_stim.plot(time_s, stimulus, color="tab:blue", linewidth=1.7)
    ax_stim.set_title("Univariate Decoding Setup")
    ax_stim.set_ylabel("stimulus")
    ax_stim.grid(alpha=0.25)

    for row in range(min(8, spikes.shape[1])):
        spike_times = time_s[spikes[:, row] > 0.5]
        if spike_times.size:
            ax_raster.vlines(spike_times, row + 0.6, row + 1.4, color="black", linewidth=0.55)
    ax_raster.set_title("Population Spike Raster (first 8 cells)")
    ax_raster.set_xlabel("time [s]")
    ax_raster.set_ylabel("cell")
    ax_raster.set_ylim(0.5, min(8, spikes.shape[1]) + 0.8)

    fig.tight_layout()
    return fig


def build_example05_univariate_decoding(summary: dict[str, object], payload: dict[str, object]) -> plt.Figure:
    e5 = payload["experiment5"]
    time_s = np.asarray(e5["time_s"], dtype=float)
    stimulus = np.asarray(e5["stimulus"], dtype=float)
    decoded = np.asarray(e5["decoded"], dtype=float)
    ci_low = np.asarray(e5["ci_low"], dtype=float)
    ci_high = np.asarray(e5["ci_high"], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(11.0, 4.6))
    ax.plot(time_s, stimulus, color="black", linewidth=1.7, label="Actual")
    ax.plot(time_s, decoded, color="tab:blue", linewidth=1.5, label="Decoded")
    ax.fill_between(time_s, ci_low, ci_high, color="tab:blue", alpha=0.18, label="95% CI")
    ax.set_title("Univariate Stimulus Decoding")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("stimulus")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def build_example05_reach_setup(summary: dict[str, object], payload: dict[str, object]) -> plt.Figure:
    e5b = payload["experiment5b"]
    spikes = np.asarray(e5b["spikes"], dtype=float)
    x_true = np.asarray(e5b["x_true"], dtype=float)
    y_true = np.asarray(e5b["y_true"], dtype=float)

    fig, (ax_path, ax_pop) = plt.subplots(1, 2, figsize=(11.6, 4.8))
    ax_path.plot(x_true, y_true, color="black", linewidth=1.6)
    ax_path.scatter([x_true[0], x_true[-1]], [y_true[0], y_true[-1]], color=["red", "blue"], s=35)
    ax_path.set_title("Actual Reach Path")
    ax_path.set_xlabel("x [cm]")
    ax_path.set_ylabel("y [cm]")
    ax_path.grid(alpha=0.25)

    im = ax_pop.imshow(spikes.T, aspect="auto", origin="lower", cmap="gray_r")
    ax_pop.set_title("Population Activity")
    ax_pop.set_xlabel("time bin")
    ax_pop.set_ylabel("cell")
    fig.colorbar(im, ax=ax_pop, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def build_example05_goal_vs_free(summary: dict[str, object], payload: dict[str, object]) -> plt.Figure:
    e5b = payload["experiment5b"]
    x_true = np.asarray(e5b["x_true"], dtype=float)
    y_true = np.asarray(e5b["y_true"], dtype=float)
    dx_goal = np.asarray(e5b["dx_goal"], dtype=float)
    dy_goal = np.asarray(e5b["dy_goal"], dtype=float)
    dx_free = np.asarray(e5b["dx_free"], dtype=float)
    dy_free = np.asarray(e5b["dy_free"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.4))
    axes[0, 0].plot(x_true, y_true, color="black", linewidth=1.5, label="Actual")
    axes[0, 0].plot(dx_goal, dy_goal, color="blue", linewidth=1.5, label="PPAF+Goal")
    axes[0, 0].plot(dx_free, dy_free, color="limegreen", linewidth=1.5, label="PPAF")
    axes[0, 0].set_title("Estimated vs. Actual Reach Path")
    axes[0, 0].set_xlabel("x [cm]")
    axes[0, 0].set_ylabel("y [cm]")
    axes[0, 0].legend(loc="lower left", fontsize=8)

    t_idx = np.arange(x_true.size, dtype=float)
    axes[0, 1].plot(t_idx, x_true, color="black", linewidth=1.4, label="Actual")
    axes[0, 1].plot(t_idx, dx_goal, color="blue", linewidth=1.4, label="PPAF+Goal")
    axes[0, 1].plot(t_idx, dx_free, color="limegreen", linewidth=1.4, label="PPAF")
    axes[0, 1].set_title("X Position")

    axes[1, 0].plot(t_idx, y_true, color="black", linewidth=1.4, label="Actual")
    axes[1, 0].plot(t_idx, dy_goal, color="blue", linewidth=1.4, label="PPAF+Goal")
    axes[1, 0].plot(t_idx, dy_free, color="limegreen", linewidth=1.4, label="PPAF")
    axes[1, 0].set_title("Y Position")
    axes[1, 0].set_xlabel("time index")

    axes[1, 1].plot(t_idx, np.gradient(y_true), color="black", linewidth=1.4, label="Actual")
    axes[1, 1].plot(t_idx, np.gradient(dy_goal), color="blue", linewidth=1.4, label="PPAF+Goal")
    axes[1, 1].plot(t_idx, np.gradient(dy_free), color="limegreen", linewidth=1.4, label="PPAF")
    axes[1, 1].set_title("Y Velocity")
    axes[1, 1].set_xlabel("time index")

    for ax in axes.ravel():
        ax.grid(alpha=0.22)
    fig.tight_layout()
    return fig


def build_example05_hybrid_setup(summary: dict[str, object], payload: dict[str, object]) -> plt.Figure:
    e6 = payload["experiment6"]
    time_s = np.asarray(e6["time_s"], dtype=float)
    state_true = np.asarray(e6["state_true"], dtype=float)
    x_pos = np.asarray(e6["x_pos"], dtype=float)
    y_pos = np.asarray(e6["y_pos"], dtype=float)
    spikes = np.asarray(e6["spikes"], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(11.2, 7.8), sharex=True)
    axes[0].plot(time_s, state_true, color="black", linewidth=1.5)
    axes[0].set_title("Hybrid Setup: Hidden State")
    axes[0].set_ylabel("state")

    axes[1].plot(time_s, x_pos, color="tab:blue", linewidth=1.4, label="x")
    axes[1].plot(time_s, y_pos, color="tab:green", linewidth=1.4, label="y")
    axes[1].set_title("Reach Position")
    axes[1].set_ylabel("position")
    axes[1].legend(loc="upper right", fontsize=8)

    for row in range(min(10, spikes.shape[1])):
        spike_times = time_s[spikes[:, row] > 0.5]
        if spike_times.size:
            axes[2].vlines(spike_times, row + 0.6, row + 1.4, color="black", linewidth=0.55)
    axes[2].set_title("Spike Raster (first 10 cells)")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("cell")

    fig.tight_layout()
    return fig


def build_example05_hybrid_summary(summary: dict[str, object], payload: dict[str, object]) -> plt.Figure:
    e6 = payload["experiment6"]
    time_s = np.asarray(e6["time_s"], dtype=float)
    state_true = np.asarray(e6["state_true"], dtype=float)
    state_hat = np.asarray(e6["state_hat"], dtype=float)
    state_prob_1 = np.asarray(e6["state_prob_1"], dtype=float)
    x_pos = np.asarray(e6["x_pos"], dtype=float)
    y_pos = np.asarray(e6["y_pos"], dtype=float)
    decoded_x = np.asarray(e6["decoded_x"], dtype=float)
    decoded_y = np.asarray(e6["decoded_y"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2))
    axes[0, 0].plot(time_s, state_true, color="black", linewidth=1.4, label="Actual")
    axes[0, 0].plot(time_s, state_hat, color="limegreen", linewidth=1.4, label="Estimated")
    axes[0, 0].set_title("Estimated vs. Actual State")
    axes[0, 0].legend(loc="upper right", fontsize=8)

    axes[1, 0].plot(time_s, state_prob_1, color="tab:blue", linewidth=1.4)
    axes[1, 0].set_title("Probability of State 1")
    axes[1, 0].set_xlabel("time [s]")

    axes[0, 1].plot(x_pos, y_pos, color="black", linewidth=1.5, label="Actual")
    axes[0, 1].plot(decoded_x, decoded_y, color="limegreen", linewidth=1.5, label="Decoded")
    axes[0, 1].set_title("Estimated vs. Actual Reach Path")
    axes[0, 1].set_xlabel("x [cm]")
    axes[0, 1].set_ylabel("y [cm]")
    axes[0, 1].legend(loc="lower left", fontsize=8)

    axes[1, 1].plot(time_s, x_pos, color="black", linewidth=1.4, label="Actual x")
    axes[1, 1].plot(time_s, decoded_x, color="tab:blue", linewidth=1.4, label="Decoded x")
    axes[1, 1].plot(time_s, y_pos, color="0.45", linewidth=1.2, label="Actual y")
    axes[1, 1].plot(time_s, decoded_y, color="limegreen", linewidth=1.2, label="Decoded y")
    axes[1, 1].set_title("Decoded State Components")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].legend(loc="lower left", fontsize=8)

    for ax in axes.ravel():
        ax.grid(alpha=0.22)
    fig.tight_layout()
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
    "example03": [
        ("fig01_simulated_and_real_rasters.png", build_example03_simulated_and_real_rasters),
        ("fig02_psth_comparison.png", build_example03_psth_comparison),
        ("fig03_ssglm_simulation_summary.png", build_example03_ssglm_simulation_summary),
        ("fig04_ssglm_fit_diagnostics.png", build_example03_ssglm_fit_diagnostics),
        ("fig05_stimulus_effect_surfaces.png", build_example03_stimulus_effect_surfaces),
        ("fig06_learning_trial_comparison.png", build_example03_learning_trial_comparison),
    ],
    "example04": [
        ("fig01_example_cells_path_overlay.png", build_example04_path_overlay),
        ("fig02_model_summary_statistics.png", build_example04_model_summary),
        ("fig03_gaussian_place_fields_animal1.png", build_example04_gaussian_animal1),
        ("fig04_zernike_place_fields_animal1.png", build_example04_zernike_animal1),
        ("fig05_gaussian_place_fields_animal2.png", build_example04_gaussian_animal2),
        ("fig06_zernike_place_fields_animal2.png", build_example04_zernike_animal2),
        ("fig07_example_cell_mesh_comparison.png", build_example04_mesh_comparison),
    ],
    "example05": [
        ("fig01_univariate_setup.png", build_example05_univariate_setup),
        ("fig02_univariate_decoding.png", build_example05_univariate_decoding),
        ("fig03_reach_and_population_setup.png", build_example05_reach_setup),
        ("fig04_ppaf_goal_vs_free.png", build_example05_goal_vs_free),
        ("fig05_hybrid_setup.png", build_example05_hybrid_setup),
        ("fig06_hybrid_decoding_summary.png", build_example05_hybrid_summary),
    ],
}


def list_named_paper_figures(example_id: str) -> list[str]:
    builders = EXAMPLE_FIGURE_BUILDERS.get(example_id)
    if builders is None:
        raise ValueError(f"No figure builders registered for {example_id}")
    return [filename for filename, _builder in builders]


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
    "list_named_paper_figures",
]
