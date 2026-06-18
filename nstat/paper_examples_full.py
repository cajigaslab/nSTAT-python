from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import correlate, correlation_lags

from .analysis import psth
from .data_manager import ensure_example_data
from .decoding_algorithms import DecodingAlgorithms
from .glm import fit_poisson_glm
from .simulation import simulate_poisson_from_rate
from .zernike import zernike_basis_from_cartesian


def _default_repo_root() -> Path:
    cur = Path(__file__).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "nstat").exists():
            return candidate
    return cur.parents[1]


def _allow_synthetic_data() -> bool:
    return os.environ.get("NSTAT_ALLOW_SYNTHETIC_DATA", "").strip().lower() in {"1", "true", "yes", "on"}


def _is_lfs_pointer(path: Path) -> bool:
    try:
        head = path.read_bytes()[:200]
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def _loadmat_checked(path: Path):
    if path.exists() and not _is_lfs_pointer(path):
        return loadmat(path, squeeze_me=True, struct_as_record=False)
    if _allow_synthetic_data():
        return None
    if not path.exists():
        raise FileNotFoundError(f"Missing MAT file: {path}")
    if _is_lfs_pointer(path):
        raise RuntimeError(
            f"MAT file is a Git LFS pointer, not dataset content: {path}. "
            "Fetch LFS assets or set NSTAT_ALLOW_SYNTHETIC_DATA=1 for synthetic CI fallback."
        )
    raise RuntimeError(f"Unable to load MAT file: {path}")


def _aic_bic(log_likelihood: float, n_obs: int, n_params: int) -> tuple[float, float]:
    aic = 2.0 * n_params - 2.0 * log_likelihood
    bic = np.log(max(n_obs, 1)) * n_params - 2.0 * log_likelihood
    return float(aic), float(bic)


def _history_matrix(y: np.ndarray, lags: list[int]) -> np.ndarray:
    out = np.zeros((y.shape[0], len(lags)), dtype=float)
    for j, lag in enumerate(lags):
        out[lag:, j] = y[:-lag]
    return out


def _autocorrelation(values: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(values, dtype=float).reshape(-1) - float(np.mean(values))
    if centered.size < 2 or float(np.var(centered)) <= 0.0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    corr = np.correlate(centered, centered, mode="full")
    corr = corr[corr.size // 2 :]
    corr = corr / corr[0]
    lags = np.arange(corr.shape[0], dtype=float)
    max_lag = int(min(max_lag, corr.shape[0] - 1))
    return lags[1 : max_lag + 1], corr[1 : max_lag + 1]


def _time_rescaled_uniforms(y: np.ndarray, lam_per_bin: np.ndarray) -> np.ndarray:
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    lam = np.asarray(lam_per_bin, dtype=float).reshape(-1)
    if y_arr.shape != lam.shape:
        raise ValueError("y and lam_per_bin must have the same shape")
    if np.sum(y_arr) <= 1:
        return np.asarray([], dtype=float)

    uniforms: list[float] = []
    accum = 0.0
    for count, lam_i in zip(y_arr, lam, strict=False):
        accum += float(max(lam_i, 1e-12))
        if count >= 1.0:
            for _ in range(int(round(count))):
                uniforms.append(1.0 - np.exp(-accum))
                accum = 0.0
    return np.asarray(uniforms, dtype=float)


def _ks_curve(uniforms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.sort(np.asarray(uniforms, dtype=float).reshape(-1))
    if u.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)
    ideal = np.linspace(1.0 / u.size, 1.0, u.size, dtype=float)
    ci = np.full(u.size, 1.36 / np.sqrt(float(u.size)), dtype=float)
    return ideal, u, ci


def _binned_series(time_s: np.ndarray, values: np.ndarray, bin_width_s: float) -> tuple[np.ndarray, np.ndarray]:
    time_arr = np.asarray(time_s, dtype=float).reshape(-1)
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    if time_arr.shape != values_arr.shape:
        raise ValueError("time_s and values must have the same shape")
    if time_arr.size < 2:
        return time_arr.copy(), values_arr.copy()
    dt = float(np.median(np.diff(time_arr)))
    samples_per_bin = max(int(round(bin_width_s / max(dt, 1e-12))), 1)
    n_bins = values_arr.shape[0] // samples_per_bin
    trimmed_values = values_arr[: n_bins * samples_per_bin]
    trimmed_time = time_arr[: n_bins * samples_per_bin]
    if n_bins == 0:
        return time_arr.copy(), values_arr.copy()
    binned_values = trimmed_values.reshape(n_bins, samples_per_bin).mean(axis=1)
    binned_time = trimmed_time.reshape(n_bins, samples_per_bin)[:, 0]
    return binned_time, binned_values


def _coefficient_intervals(x: np.ndarray, result, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    offset_arr = np.asarray(offset, dtype=float).reshape(-1)
    x_aug = np.column_stack([np.ones(x_arr.shape[0], dtype=float), x_arr])
    beta = np.concatenate([[result.intercept], np.asarray(result.coefficients, dtype=float)])
    lam = np.exp(np.clip(x_aug @ beta + offset_arr, -20.0, 20.0))
    hess = x_aug.T @ (lam[:, None] * x_aug)
    cov = np.linalg.pinv(hess)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    lower = beta - 1.96 * se
    upper = beta + 1.96 * se
    return beta, lower, upper


def _load_mepsc_times_seconds(path: Path) -> np.ndarray:
    if not path.exists():
        if _allow_synthetic_data():
            name = path.name
            if name == "epsc2.txt":
                rng = np.random.default_rng(1001)
                time = np.arange(0.0, 220.0, 0.05, dtype=float)
                rate_hz = np.full(time.shape, 0.55, dtype=float)
            elif name == "washout1.txt":
                rng = np.random.default_rng(1002)
                time = np.arange(0.0, 500.0, 0.05, dtype=float)
                rate_hz = np.where(time < 235.0, 0.70, 1.25)
            elif name == "washout2.txt":
                rng = np.random.default_rng(1003)
                time = np.arange(0.0, 320.0, 0.05, dtype=float)
                rate_hz = 1.75 + 0.20 * np.sin(0.01 * time)
            else:
                raise FileNotFoundError(f"Missing mEPSC file: {path}")
            keep = rng.random(time.shape[0]) < np.clip(rate_hz * 0.05, 1e-6, 0.25)
            return time[keep]
        raise FileNotFoundError(f"Missing mEPSC file: {path}")
    arr = np.loadtxt(path, skiprows=1)
    return np.asarray(arr[:, 1], dtype=float).reshape(-1) / 1000.0


def _bin_spike_times(spikes: np.ndarray, t0: float, t1: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    n_bins = int(np.floor((t1 - t0) / dt)) + 1
    edges = t0 + np.arange(n_bins + 1, dtype=float) * dt
    counts, _ = np.histogram(spikes, bins=edges)
    return edges[:-1], counts.astype(float)


def run_experiment1(data_dir: Path, *, return_payload: bool = False) -> dict[str, float] | tuple[dict[str, float], dict[str, object]]:
    mepsc_dir = data_dir / "mEPSCs"
    epsc2 = _load_mepsc_times_seconds(mepsc_dir / "epsc2.txt")
    washout1 = _load_mepsc_times_seconds(mepsc_dir / "washout1.txt")
    washout2 = _load_mepsc_times_seconds(mepsc_dir / "washout2.txt")

    dt_const = 0.01
    t_const, y_const = _bin_spike_times(epsc2, 0.0, float(np.max(epsc2)), dt_const)
    off_const = np.full(y_const.shape[0], np.log(dt_const), dtype=float)
    m_const = fit_poisson_glm(np.zeros((y_const.shape[0], 0), dtype=float), y_const, offset=off_const, max_iter=80)
    aic_const, bic_const = _aic_bic(m_const.log_likelihood, y_const.shape[0], 1)
    lam_const = m_const.predict_rate(np.zeros((y_const.shape[0], 0), dtype=float), offset=off_const)
    rate_const_hz = lam_const / dt_const
    const_uniforms = _time_rescaled_uniforms(y_const, lam_const)
    const_ideal, const_ks, const_ks_ci = _ks_curve(const_uniforms)
    const_acf_lags, const_acf = _autocorrelation(const_uniforms, max_lag=580)
    const_acf_ci = 1.96 / np.sqrt(max(const_uniforms.shape[0], 1))

    spikes = np.concatenate([260.0 + washout1, np.sort(washout2) + 745.0])
    dt = 0.01
    t, y = _bin_spike_times(spikes, 260.0, float(np.max(spikes)), dt)
    off = np.full(y.shape[0], np.log(dt), dtype=float)

    seg2 = ((t >= 495.0) & (t < 765.0)).astype(float)
    seg3 = (t >= 765.0).astype(float)
    x_piece = np.column_stack([seg2, seg3])
    hist = _history_matrix(y, [1, 2, 3, 4, 5, 7, 10, 14, 20, 30])
    x_piece_hist = np.column_stack([x_piece, hist])

    m_piece = fit_poisson_glm(x_piece, y, offset=off, max_iter=100)
    m_piece_hist = fit_poisson_glm(x_piece_hist, y, offset=off, max_iter=120)
    aic_piece, bic_piece = _aic_bic(m_piece.log_likelihood, y.shape[0], x_piece.shape[1] + 1)
    aic_piece_hist, bic_piece_hist = _aic_bic(m_piece_hist.log_likelihood, y.shape[0], x_piece_hist.shape[1] + 1)
    lam_piece = m_piece.predict_rate(x_piece, offset=off)
    lam_piece_hist = m_piece_hist.predict_rate(x_piece_hist, offset=off)
    time_binned, obs_rate_hz = _binned_series(t, y / dt, 2.0)
    _, piece_rate_hz = _binned_series(t, lam_piece / dt, 2.0)
    _, piece_hist_rate_hz = _binned_series(t, lam_piece_hist / dt, 2.0)

    summary = {
        "const_condition_spikes": float(np.sum(y_const)),
        "const_model_aic": aic_const,
        "const_model_bic": bic_const,
        "constant_acf_ci": float(const_acf_ci),
        "decreasing_condition_spikes": float(np.sum(y)),
        "piecewise_model_aic": aic_piece,
        "piecewise_model_bic": bic_piece,
        "piecewise_history_model_aic": aic_piece_hist,
        "piecewise_history_model_bic": bic_piece_hist,
        "dt_seconds": dt,
    }
    if not return_payload:
        return summary

    payload: dict[str, object] = {
        "constant_spike_times_s": epsc2,
        "constant_time_s": t_const,
        "constant_rate_hz": rate_const_hz,
        "constant_acf_lags_s": const_acf_lags,
        "constant_acf_values": const_acf,
        "constant_ks_ideal": const_ideal,
        "constant_ks_empirical": const_ks,
        "constant_ks_ci": const_ks_ci,
        "constant_window_s": np.asarray([0.0, float(np.max(epsc2))], dtype=float),
        "washout_spike_times_s": spikes,
        "washout_window_s": np.asarray([260.0, float(np.max(spikes))], dtype=float),
        "washout_time_s": time_binned,
        "washout_observed_rate_hz": obs_rate_hz,
        "washout_piecewise_rate_hz": piece_rate_hz,
        "washout_piecewise_history_rate_hz": piece_hist_rate_hz,
        "washout_segment_edges_s": np.asarray([260.0, 495.0, 765.0, float(np.max(spikes))], dtype=float),
    }
    return summary, payload


def run_experiment2(data_dir: Path, *, return_payload: bool = False) -> dict[str, float] | tuple[dict[str, float], dict[str, object]]:
    path = data_dir / "Explicit Stimulus" / "Dir3" / "Neuron1" / "Stim2" / "trngdataBis.mat"
    d = _loadmat_checked(path)
    if d is None:
        rng = np.random.default_rng(2002)
        n = 5000
        t = np.linspace(0.0, 2.0 * np.pi, n, dtype=float)
        stim_raw = np.sin(1.8 * t) + 0.25 * np.sin(0.4 * t + 0.2)
        p = np.clip(0.015 + 0.02 * np.maximum(stim_raw, 0.0), 1e-4, 0.35)
        y = (rng.random(n) < p).astype(float)
    else:
        stim_raw = np.asarray(d["t"], dtype=float).reshape(-1)
        y = np.asarray(d["y"], dtype=float).reshape(-1)

    dt = 0.001
    stim = stim_raw / 10.0
    stim_vel = np.gradient(stim, dt)
    hist = _history_matrix(y, [1, 2, 3, 4, 5])

    x1 = np.zeros((y.shape[0], 0), dtype=float)
    x2 = np.column_stack([stim, stim_vel])
    x3 = np.column_stack([stim, stim_vel, hist])
    offset = np.full(y.shape[0], np.log(dt), dtype=float)

    m1 = fit_poisson_glm(x1, y, offset=offset)
    m2 = fit_poisson_glm(x2, y, offset=offset)
    m3 = fit_poisson_glm(x3, y, offset=offset)

    aic1, bic1 = _aic_bic(m1.log_likelihood, y.shape[0], 1)
    aic2, bic2 = _aic_bic(m2.log_likelihood, y.shape[0], 3)
    aic3, bic3 = _aic_bic(m3.log_likelihood, y.shape[0], 8)
    lam1 = m1.predict_rate(x1, offset=offset)
    lam2 = m2.predict_rate(x2, offset=offset)
    lam3 = m3.predict_rate(x3, offset=offset)

    u1 = _time_rescaled_uniforms(y, lam1)
    u2 = _time_rescaled_uniforms(y, lam2)
    u3 = _time_rescaled_uniforms(y, lam3)
    ks_ideal, ks1_emp, ks_ci = _ks_curve(u1)
    _, ks2_emp, _ = _ks_curve(u2)
    _, ks3_emp, _ = _ks_curve(u3)

    selection_n = int(min(6000, y.shape[0]))
    candidate_q = np.arange(1, 29, dtype=int)
    ks_stats = np.zeros(candidate_q.shape[0], dtype=float)
    aic_series = np.zeros(candidate_q.shape[0], dtype=float)
    bic_series = np.zeros(candidate_q.shape[0], dtype=float)
    y_sel = y[:selection_n]
    stim_sel = stim[:selection_n]
    stim_vel_sel = stim_vel[:selection_n]
    offset_sel = offset[:selection_n]
    full_hist = _history_matrix(y_sel, list(range(1, int(candidate_q[-1]) + 1)))
    for idx, q in enumerate(candidate_q.tolist()):
        x_sel = np.column_stack([stim_sel, stim_vel_sel, full_hist[:, :q]])
        model_q = fit_poisson_glm(x_sel, y_sel, offset=offset_sel, max_iter=45)
        lam_q = model_q.predict_rate(x_sel, offset=offset_sel)
        uq = _time_rescaled_uniforms(y_sel, lam_q)
        ideal_q, emp_q, _ = _ks_curve(uq)
        ks_stats[idx] = float(np.max(np.abs(emp_q - ideal_q))) if ideal_q.size else 0.0
        aic_q, bic_q = _aic_bic(model_q.log_likelihood, y_sel.shape[0], x_sel.shape[1] + 1)
        aic_series[idx] = aic_q
        bic_series[idx] = bic_q

    delta_aic = aic_series - aic_series[0]
    delta_bic = bic_series - bic_series[0]

    # MATLAB Summary.plotSummary: per-window coefficient errorbar panel needs values+CIs for each scan iteration.
    scan_coef_values: list[np.ndarray] = []
    scan_coef_lower: list[np.ndarray] = []
    scan_coef_upper: list[np.ndarray] = []
    scan_coef_names: list[list[str]] = []
    for idx, q in enumerate(candidate_q.tolist()):
        x_sel = np.column_stack([stim_sel, stim_vel_sel, full_hist[:, :q]])
        model_q = fit_poisson_glm(x_sel, y_sel, offset=offset_sel, max_iter=45)
        beta_q, lower_q, upper_q = _coefficient_intervals(x_sel, model_q, offset_sel)
        # Order: history bins [0..q*dt], constant μ, stim, stim_vel
        n_hist = q
        coefs_hist = beta_q[3 : 3 + n_hist]
        low_hist = lower_q[3 : 3 + n_hist]
        up_hist = upper_q[3 : 3 + n_hist]
        names_hist = [f"[{(i)*dt:.3f},{(i+1)*dt:.3f}]" for i in range(n_hist)]
        coefs_full = np.concatenate([coefs_hist, [beta_q[0], beta_q[1], beta_q[2]]])
        low_full = np.concatenate([low_hist, [lower_q[0], lower_q[1], lower_q[2]]])
        up_full = np.concatenate([up_hist, [upper_q[0], upper_q[1], upper_q[2]]])
        names_full = names_hist + ["μ", "stim", "stim_vel"]
        scan_coef_values.append(coefs_full)
        scan_coef_lower.append(low_full)
        scan_coef_upper.append(up_full)
        scan_coef_names.append(names_full)

    xcorr_window = int(min(20000, y.shape[0]))
    xcorr = correlate(y[:xcorr_window] - np.mean(y[:xcorr_window]), stim[:xcorr_window] - np.mean(stim[:xcorr_window]), mode="full", method="fft")
    lags = correlation_lags(xcorr_window, xcorr_window, mode="full")
    keep = np.abs(lags) <= 1000
    xcorr = xcorr[keep]
    lags_s = lags[keep] * dt
    positive = lags_s >= 0.0
    lags_s = lags_s[positive]
    xcorr = xcorr[positive]
    peak_idx = int(np.argmax(xcorr))
    peak_lag_s = float(lags_s[peak_idx])

    coef_beta, coef_lower, coef_upper = _coefficient_intervals(x3, m3, offset)
    coef_names = [f"[{i*dt:.3f},{(i+1)*dt:.3f}]" for i in range(hist.shape[1])]
    coef_names.extend(["μ", "stim"])
    coef_values = np.concatenate([coef_beta[3:], [coef_beta[0], coef_beta[1]]])
    coef_lower_plot = np.concatenate([coef_lower[3:], [coef_lower[0], coef_lower[1]]])
    coef_upper_plot = np.concatenate([coef_upper[3:], [coef_upper[0], coef_upper[1]]])

    # MATLAB plotResults companion fields: ACF of rescaled uniforms, sequential corr u_{j+1} vs u_j,
    # GLM coefficient intervals per model, and point-process residual M(t_k).
    def _seq_corr(u: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
        u_arr = np.asarray(u, dtype=float).reshape(-1)
        if u_arr.size < 3:
            return 0.0, 1.0, u_arr, u_arr
        a, b = u_arr[:-1], u_arr[1:]
        a_mean, b_mean = float(np.mean(a)), float(np.mean(b))
        num = float(np.sum((a - a_mean) * (b - b_mean)))
        den = float(np.sqrt(np.sum((a - a_mean) ** 2) * np.sum((b - b_mean) ** 2)))
        rho = num / den if den > 0 else 0.0
        # approx p-value via Fisher transform
        n = a.size
        if abs(rho) >= 1.0 or n < 4:
            pval = 0.0 if abs(rho) >= 1.0 else 1.0
        else:
            z = 0.5 * np.log((1.0 + rho) / (1.0 - rho)) * np.sqrt(max(n - 3, 1))
            from math import erfc
            pval = float(erfc(abs(z) / np.sqrt(2.0)))
        return float(rho), float(pval), a, b

    def _residual(y_in: np.ndarray, lam_in: np.ndarray) -> np.ndarray:
        # M(t_k) = cumulative (count - intensity) in spikes (dimensionless)
        return np.cumsum(np.asarray(y_in, dtype=float) - np.asarray(lam_in, dtype=float))

    acf_max_lag = 50
    acf_lags1, acf_vals1 = _autocorrelation(u1, acf_max_lag)
    acf_lags2, acf_vals2 = _autocorrelation(u2, acf_max_lag)
    acf_lags3, acf_vals3 = _autocorrelation(u3, acf_max_lag)
    acf_ci_val = 1.96 / np.sqrt(max(u1.size, 1))

    rho1, p1, uj1_a, uj1_b = _seq_corr(u1)
    rho2, p2, uj2_a, uj2_b = _seq_corr(u2)
    rho3, p3, uj3_a, uj3_b = _seq_corr(u3)

    # Per-model coefficient intervals (intercept + covariates). Model 1: intercept only.
    m1_x = np.zeros((y.shape[0], 0), dtype=float)
    beta1, low1, up1 = _coefficient_intervals(m1_x, m1, offset)
    beta2, low2, up2 = _coefficient_intervals(x2, m2, offset)
    beta3, low3, up3 = _coefficient_intervals(x3, m3, offset)
    coef_names_m1 = ["μ"]
    coef_names_m2 = ["μ", "stim", "stim_vel"]
    coef_names_m3 = ["μ", "stim", "stim_vel"] + [f"hist{i+1}" for i in range(hist.shape[1])]

    # Point-process residual restricted to the 0..50s view window (or full range; MATLAB uses analysis span)
    resid_view_n = int(min(int(round(50.0 / dt)), y.shape[0]))
    residual_t = np.arange(resid_view_n, dtype=float) * dt
    residual_m1 = _residual(y[:resid_view_n], lam1[:resid_view_n])
    residual_m2 = _residual(y[:resid_view_n], lam2[:resid_view_n])
    residual_m3 = _residual(y[:resid_view_n], lam3[:resid_view_n])

    summary = {
        "n_samples": float(y.shape[0]),
        "model1_aic": aic1,
        "model2_aic": aic2,
        "model3_aic": aic3,
        "model1_bic": bic1,
        "model2_bic": bic2,
        "model3_bic": bic3,
        "peak_lag_seconds": peak_lag_s,
    }
    if not return_payload:
        return summary

    view_n = int(min(int(round(21.0 / dt)), y.shape[0]))
    # MATLAB nSTATPaperExamples.m: nst2.setMaxTime(21) mutates nst2 only; nst (unclipped)
    # is what nst.plot draws -- so the raster spans the full recording (~50.73 s, ~966 spikes).
    # Expose the full-length spike train and time vector so the notebook can mirror MATLAB.
    # MATLAB plots data.t in raw volts (0..9.953 V); the /10 normalization is GLM-internal only.
    payload = {
        "time_s": np.arange(view_n, dtype=float) * dt,
        "spike_indicator": y[:view_n],
        "stimulus": stim[:view_n],
        "stimulus_raw_v": stim_raw,
        "spike_indicator_full": y,
        "time_s_full": np.arange(y.shape[0], dtype=float) * dt,
        "velocity": stim_vel[:view_n],
        "xcorr_lags_s": lags_s,
        "xcorr_values": xcorr,
        "history_windows": candidate_q.astype(float),
        "ks_stats": ks_stats,
        "delta_aic": delta_aic,
        "delta_bic": delta_bic,
        "ks_ideal": ks_ideal,
        "ks_const_empirical": ks1_emp,
        "ks_stim_empirical": ks2_emp,
        "ks_hist_empirical": ks3_emp,
        "ks_ci": ks_ci,
        "coef_names": coef_names,
        "coef_values": coef_values,
        "coef_lower": coef_lower_plot,
        "coef_upper": coef_upper_plot,
        # ACF of rescaled ISIs (per model)
        "acf_lags": acf_lags1,
        "acf_m1": acf_vals1,
        "acf_m2": acf_vals2,
        "acf_m3": acf_vals3,
        "acf_ci": float(acf_ci_val),
        # Sequential correlation u_{j+1} vs u_j with summary statistics
        "seq_corr_m1": {"uj": uj1_a, "uj1": uj1_b, "rho": rho1, "p": p1},
        "seq_corr_m2": {"uj": uj2_a, "uj1": uj2_b, "rho": rho2, "p": p2},
        "seq_corr_m3": {"uj": uj3_a, "uj1": uj3_b, "rho": rho3, "p": p3},
        # GLM coefficient intervals per model
        "glm_m1": {"names": coef_names_m1, "values": beta1, "lower": low1, "upper": up1},
        "glm_m2": {"names": coef_names_m2, "values": beta2, "lower": low2, "upper": up2},
        "glm_m3": {"names": coef_names_m3, "values": beta3, "lower": low3, "upper": up3},
        # Point-process residual M(t_k)
        "residual_t": residual_t,
        "residual_m1": residual_m1,
        "residual_m2": residual_m2,
        "residual_m3": residual_m3,
        # Per-window scan coefficients (for Summary.plotSummary GLM panel)
        "scan_coef_values": scan_coef_values,
        "scan_coef_lower": scan_coef_lower,
        "scan_coef_upper": scan_coef_upper,
        "scan_coef_names": scan_coef_names,
    }
    return summary, payload


def run_experiment3(seed: int = 7, *, return_payload: bool = False) -> dict[str, float] | tuple[dict[str, float], dict[str, object]]:
    rng = np.random.default_rng(seed)
    dt = 0.001
    tmax = 1.0
    time = np.arange(0.0, tmax + dt, dt)

    linear = np.sin(2.0 * np.pi * 2.0 * time) - 3.0
    p = np.exp(linear)
    p = p / (1.0 + p)
    rate_hz = p / dt

    trains = [simulate_poisson_from_rate(time, rate_hz, rng=rng) for _ in range(20)]
    edges = np.arange(0.0, tmax + 0.05, 0.05)
    psth_rate_hz, counts = psth(trains, edges)
    summary = {
        "num_trials": float(len(trains)),
        "psth_peak_hz": float(np.max(psth_rate_hz)),
        "psth_mean_hz": float(np.mean(psth_rate_hz)),
        "total_spikes": float(np.sum(counts)),
    }
    if not return_payload:
        return summary

    payload = {
        "time_s": time,
        "true_rate_hz": rate_hz,
        "psth_bin_centers_s": 0.5 * (edges[:-1] + edges[1:]),
        "psth_rate_hz": psth_rate_hz,
        "raster_spike_times": [np.asarray(train.spikeTimes, dtype=float) for train in trains],
    }
    return summary, payload


def run_experiment3b(data_dir: Path, *, return_payload: bool = False) -> dict[str, float] | tuple[dict[str, float], dict[str, object]]:
    path = data_dir / "SSGLMExampleData.mat"
    d = _loadmat_checked(path)
    if d is None:
        rng = np.random.default_rng(3003)
        stimulus = rng.normal(0.0, 1.0, size=(15, 250))
        xk = stimulus + rng.normal(0.0, 0.2, size=stimulus.shape)
        ci_half = np.abs(rng.normal(0.35, 0.08, size=stimulus.shape))
        stim_cis = np.stack([xk - ci_half, xk + ci_half], axis=-1)
        qhat = np.abs(rng.normal(0.12, 0.03, size=stimulus.shape[0]))
        qhat_all = np.tile(qhat[:, None], (1, 8))
        gammahat = np.abs(rng.normal(0.08, 0.02, size=3))
        gammahat_all = np.linspace(0.05, 0.11, 8, dtype=float)
        logll = float(-np.mean((xk - stimulus) ** 2) * stimulus.size)
    else:
        stimulus = np.asarray(d["stimulus"], dtype=float)
        xk = np.asarray(d["xK"], dtype=float)
        stim_cis = np.asarray(d["stimCIs"], dtype=float)
        qhat = np.asarray(d["Qhat"], dtype=float).reshape(-1)
        qhat_all = np.asarray(d.get("QhatAll", np.tile(qhat[:, None], (1, 8))), dtype=float)
        gammahat = np.asarray(d["gammahat"], dtype=float).reshape(-1)
        gammahat_all = np.asarray(d.get("gammahatAll", gammahat), dtype=float).reshape(-1)
        logll = float(np.asarray(d["logll"], dtype=float).reshape(-1)[0])

    coverage = np.mean((stimulus >= stim_cis[:, :, 0]) & (stimulus <= stim_cis[:, :, 1]))
    rmse = np.sqrt(np.mean((xk - stimulus) ** 2))
    summary = {
        "num_trials": float(stimulus.shape[0]),
        "num_time_bins": float(stimulus.shape[1]),
        "state_rmse": float(rmse),
        "ci_coverage": float(coverage),
        "mean_qhat": float(np.mean(qhat)),
        "mean_gammahat": float(np.mean(gammahat)),
        "log_likelihood": logll,
    }
    if not return_payload:
        return summary

    payload = {
        "stimulus": stimulus,
        "xk": xk,
        "stim_cis": stim_cis,
        "qhat": qhat,
        "qhat_all": qhat_all,
        "gammahat": gammahat,
        "gammahat_all": gammahat_all,
        "ci_width": stim_cis[:, :, 1] - stim_cis[:, :, 0],
    }
    return summary, payload


def _spike_indicator(time: np.ndarray, spike_times: np.ndarray) -> np.ndarray:
    y = np.zeros(time.shape[0], dtype=float)
    idx = np.searchsorted(time, spike_times, side="left")
    idx = idx[(idx >= 0) & (idx < time.shape[0])]
    if idx.size > 0:
        y[idx] = 1.0
    return y


def _load_placecell_dataset(path: Path):
    d = _loadmat_checked(path)
    if d is None:
        rng = np.random.default_rng(4004)
        time = np.linspace(0.0, 20.0, 2400, dtype=float)
        x = 0.8 * np.sin(0.6 * time) + 0.2 * np.sin(1.7 * time + 0.5)
        y = 0.7 * np.cos(0.5 * time + 0.3)
        radius = np.sqrt(x * x + y * y)
        max_radius = float(np.max(radius)) if radius.size else 0.0
        if max_radius > 0.98:
            scale = 0.98 / max_radius
            x = x * scale
            y = y * scale
        n_cells = 8
        neurons = []
        for _ in range(n_cells):
            field = np.exp(-((x - rng.uniform(-0.5, 0.5)) ** 2 + (y - rng.uniform(-0.5, 0.5)) ** 2) / 0.2)
            p = np.clip(0.001 + 0.03 * field, 1e-6, 0.25)
            spikes = time[rng.random(time.shape[0]) < p]
            neurons.append(type("N", (), {"spikeTimes": spikes})())
        neurons = np.asarray(neurons, dtype=object)
    else:
        x = np.asarray(d["x"], dtype=float).reshape(-1)
        y = np.asarray(d["y"], dtype=float).reshape(-1)
        time = np.asarray(d["time"], dtype=float).reshape(-1)
        neurons = np.asarray(d["neuron"], dtype=object).reshape(-1)
    return x, y, time, neurons


def _evaluate_place_models(x: np.ndarray, y: np.ndarray, time: np.ndarray, neurons: np.ndarray, selected_indices: list[int]):
    dt = float(np.median(np.diff(time)))
    offset = np.full(time.shape[0], np.log(max(dt, 1e-12)), dtype=float)
    x_gauss = np.column_stack([x, y, x * x, y * y, x * y])
    x_zern = zernike_basis_from_cartesian(x, y)

    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 40, dtype=float)
    y_grid = np.linspace(float(np.min(y)), float(np.max(y)), 40, dtype=float)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_gauss = np.column_stack([xx.ravel(), yy.ravel(), xx.ravel() ** 2, yy.ravel() ** 2, xx.ravel() * yy.ravel()])
    grid_zern = zernike_basis_from_cartesian(xx.ravel(), yy.ravel(), fill_value=np.nan)

    d_aic: list[float] = []
    d_bic: list[float] = []
    gaussian_fields: list[np.ndarray] = []
    zernike_fields: list[np.ndarray] = []
    spike_times_all: list[np.ndarray] = []

    for idx in selected_indices:
        spike_times = np.asarray(neurons[idx].spikeTimes, dtype=float).reshape(-1)
        y_spike = _spike_indicator(time, spike_times)
        mg = fit_poisson_glm(x_gauss, y_spike, offset=offset, max_iter=70)
        # Zernike basis already includes the constant Z_0^0 mode, which
        # acts as the intercept — matches MATLAB helpfile
        # ``HippocampalPlaceCellExample.m:83-84`` (``tc{2}`` has only
        # ``{Zernike.z1..z10}`` with no Baseline).  Without
        # ``include_intercept=False`` we double-count and bias the fit.
        mz = fit_poisson_glm(x_zern, y_spike, offset=offset,
                             include_intercept=False, max_iter=70)
        aicg, bicg = _aic_bic(mg.log_likelihood, y_spike.shape[0], x_gauss.shape[1] + 1)
        aicz, bicz = _aic_bic(mz.log_likelihood, y_spike.shape[0], x_zern.shape[1])
        d_aic.append(aicg - aicz)
        d_bic.append(bicg - bicz)
        gaussian_fields.append(mg.predict_rate(grid_gauss).reshape(xx.shape))
        zernike_fields.append(mz.predict_rate(grid_zern).reshape(xx.shape))
        spike_times_all.append(spike_times)

    return {
        "time_s": time,
        "x_pos": x,
        "y_pos": y,
        "selected_indices": np.asarray(selected_indices, dtype=int),
        "delta_aic": np.asarray(d_aic, dtype=float),
        "delta_bic": np.asarray(d_bic, dtype=float),
        "gaussian_fields": np.asarray(gaussian_fields, dtype=float),
        "zernike_fields": np.asarray(zernike_fields, dtype=float),
        "spike_times": spike_times_all,
        "grid_x": xx,
        "grid_y": yy,
    }


def run_experiment4(data_dir: Path, *, return_payload: bool = False) -> dict[str, float] | tuple[dict[str, float], dict[str, object]]:
    path1 = data_dir / "Place Cells" / "PlaceCellDataAnimal1.mat"
    x1, y1, time1, neurons1 = _load_placecell_dataset(path1)

    path2 = data_dir / "Place Cells" / "PlaceCellDataAnimal2.mat"
    x2, y2, time2, neurons2 = _load_placecell_dataset(path2)

    # MATLAB HippocampalPlaceCellExample fits every cell in the dataset and tiles
    # one panel per cell into the place-field grid figures.  Mirror that: include
    # all cells (cap defensively to keep the grid renderable in CI).  The legacy
    # MATLAB-mesh figure (figure(8)/figure(9)) still picks the last cell of the
    # legacy 4-cell selection so its title remains "Cell 25" for parity with the
    # MATLAB published mesh figures.
    _MAX_CELLS_GRID = 36
    selected1 = list(range(min(len(neurons1), _MAX_CELLS_GRID)))
    selected2 = list(range(min(len(neurons2), _MAX_CELLS_GRID)))
    _legacy_mesh_idx = min(24, len(neurons1) - 1)
    if _legacy_mesh_idx not in selected1:
        selected1.append(_legacy_mesh_idx)
    animal1 = _evaluate_place_models(x1, y1, time1, neurons1, selected1)
    animal2 = _evaluate_place_models(x2, y2, time2, neurons2, selected2)

    summary = {
        "num_cells_fit": float(len(selected1) + len(selected2)),
        "mean_delta_aic_gaussian_minus_zernike": float(np.mean(np.concatenate([animal1["delta_aic"], animal2["delta_aic"]]))),
        "mean_delta_bic_gaussian_minus_zernike": float(np.mean(np.concatenate([animal1["delta_bic"], animal2["delta_bic"]]))),
    }
    if not return_payload:
        return summary

    mesh_idx = int(_legacy_mesh_idx)
    # Map mesh_idx onto its position in the fitted selected1 list so we can pull
    # the right gaussian/zernike field for the mesh figures below.
    _mesh_pos = selected1.index(mesh_idx)
    mesh_spike_times = np.asarray(neurons1[mesh_idx].spikeTimes, dtype=float).reshape(-1)
    payload = {
        "animal1": animal1,
        "animal2": animal2,
        "mesh": {
            "cell_index": mesh_idx,
            "gaussian_field": animal1["gaussian_fields"][_mesh_pos],
            "zernike_field": animal1["zernike_fields"][_mesh_pos],
            "grid_x": animal1["grid_x"],
            "grid_y": animal1["grid_y"],
            "x_pos": x1,
            "y_pos": y1,
            "spike_times": mesh_spike_times,
            "time_s": time1,
        },
    }
    return summary, payload


def run_experiment5(
    seed: int = 11, n_cells: int = 20, *, return_payload: bool = False
) -> dict[str, float] | tuple[dict[str, float], dict[str, object]]:
    rng = np.random.default_rng(seed)
    dt = 0.001
    time = np.arange(0.0, 1.0 + dt, dt)
    stim = np.sin(2.0 * np.pi * 2.0 * time)

    n_cells = int(n_cells)
    if n_cells < 1:
        raise ValueError("n_cells must be >= 1")
    spikes = np.zeros((time.shape[0], n_cells), dtype=float)
    cif_rates = np.zeros((time.shape[0], n_cells), dtype=float)
    for i in range(n_cells):
        b1 = rng.normal(1.0, 0.5)
        b0 = np.log(10.0 * dt) + rng.normal(0.0, 0.3)
        eta = b1 * stim + b0
        p = np.exp(eta)
        p = p / (1.0 + p)
        cif_rates[:, i] = p / dt
        spikes[:, i] = (rng.random(time.shape[0]) < p).astype(float)

    decoded = DecodingAlgorithms.linear_decode(spikes, stim)
    rmse = float(np.sqrt(np.mean((decoded["decoded"] - stim) ** 2)))
    summary = {"num_cells": float(n_cells), "decode_rmse": rmse}
    if not return_payload:
        return summary
    payload = {
        "time_s": time,
        "stimulus": stim,
        "spikes": spikes,
        "cif_rates": cif_rates,
        "decoded": np.asarray(decoded["decoded"], dtype=float),
        "ci_low": np.asarray(decoded["ci"][:, 0], dtype=float),
        "ci_high": np.asarray(decoded["ci"][:, 1], dtype=float),
    }
    return summary, payload


def run_experiment5b(
    seed: int = 19, n_cells: int = 30, *, return_payload: bool = False
) -> dict[str, float] | tuple[dict[str, float], dict[str, object]]:
    rng = np.random.default_rng(seed)

    dt = 0.01
    time = np.arange(0.0, 20.0 + dt, dt)
    x_true = 0.25 * np.sin(2.0 * np.pi * 0.15 * time)
    y_true = 0.20 * np.cos(2.0 * np.pi * 0.10 * time)
    vx = np.gradient(x_true, dt)
    vy = np.gradient(y_true, dt)

    n_cells = int(n_cells)
    if n_cells < 1:
        raise ValueError("n_cells must be >= 1")
    spikes = np.zeros((time.shape[0], n_cells), dtype=float)
    for i in range(n_cells):
        wx = rng.normal(0.0, 1.0)
        wy = rng.normal(0.0, 1.0)
        b0 = -3.0 + rng.normal(0.0, 0.2)
        eta = b0 + 3.0 * wx * vx + 3.0 * wy * vy
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -20.0, 20.0)))
        spikes[:, i] = (rng.random(time.shape[0]) < p).astype(float)

    goal_cells = max(n_cells // 2, 1)
    dx_goal = DecodingAlgorithms.linear_decode(spikes[:, :goal_cells], x_true)["decoded"]
    dy_goal = DecodingAlgorithms.linear_decode(spikes[:, :goal_cells], y_true)["decoded"]
    dx_free = DecodingAlgorithms.linear_decode(spikes, x_true)["decoded"]
    dy_free = DecodingAlgorithms.linear_decode(spikes, y_true)["decoded"]
    summary = {
        "num_cells": float(n_cells),
        "num_samples": float(time.shape[0]),
        "decode_rmse_x": float(np.sqrt(np.mean((dx_free - x_true) ** 2))),
        "decode_rmse_y": float(np.sqrt(np.mean((dy_free - y_true) ** 2))),
    }
    if not return_payload:
        return summary
    payload = {
        "time_s": time,
        "x_true": x_true,
        "y_true": y_true,
        "vx_true": vx,
        "vy_true": vy,
        "spikes": spikes,
        "dx_goal": np.asarray(dx_goal, dtype=float),
        "dy_goal": np.asarray(dy_goal, dtype=float),
        "dx_free": np.asarray(dx_free, dtype=float),
        "dy_free": np.asarray(dy_free, dtype=float),
    }
    return summary, payload


def _simulate_hybrid_spikes(x: np.ndarray, mstate: np.ndarray, dt: float, n_cells: int, seed: int):
    rng = np.random.default_rng(seed)
    vx = x[2, :]
    vy = x[3, :]
    wvx = rng.normal(0.0, 1.0, size=n_cells)
    wvy = rng.normal(0.0, 1.0, size=n_cells)
    b1 = rng.normal(-3.8, 0.2, size=n_cells)
    b2 = rng.normal(-2.8, 0.2, size=n_cells)

    spikes = np.zeros((x.shape[1], n_cells), dtype=float)
    for t in range(x.shape[1]):
        base = b1 if int(mstate[t]) == 1 else b2
        eta = base + 1.2 * wvx * vx[t] + 1.2 * wvy * vy[t]
        lam = np.exp(np.clip(eta, -15.0, 15.0))
        p = 1.0 - np.exp(-lam * dt)
        spikes[t, :] = (rng.random(n_cells) < p).astype(float)
    return spikes, wvx, wvy, b1, b2


def _hybrid_state_filter(spikes: np.ndarray, x: np.ndarray, dt: float, p_ij: np.ndarray, wvx: np.ndarray, wvy: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    n_t, _ = spikes.shape
    post = np.zeros((n_t, 2), dtype=float)
    post[0, :] = [0.5, 0.5]
    vx = x[2, :]
    vy = x[3, :]

    for t in range(1, n_t):
        eta1 = b1 + 1.2 * wvx * vx[t] + 1.2 * wvy * vy[t]
        eta2 = b2 + 1.2 * wvx * vx[t] + 1.2 * wvy * vy[t]
        p1 = np.clip(1.0 - np.exp(-np.exp(np.clip(eta1, -15.0, 15.0)) * dt), 1e-6, 1.0 - 1e-6)
        p2 = np.clip(1.0 - np.exp(-np.exp(np.clip(eta2, -15.0, 15.0)) * dt), 1e-6, 1.0 - 1e-6)
        k = spikes[t, :]
        ll1 = np.sum(k * np.log(p1) + (1.0 - k) * np.log(1.0 - p1))
        ll2 = np.sum(k * np.log(p2) + (1.0 - k) * np.log(1.0 - p2))
        pred0 = max(post[t - 1, 0] * p_ij[0, 0] + post[t - 1, 1] * p_ij[1, 0], 1e-15)
        pred1 = max(post[t - 1, 0] * p_ij[0, 1] + post[t - 1, 1] * p_ij[1, 1], 1e-15)
        logs = np.array([np.log(pred0) + ll1, np.log(pred1) + ll2], dtype=float)
        logs = logs - np.max(logs)
        un = np.exp(logs)
        post[t, :] = un / np.sum(un)
    return post


def _load_hybrid_fixture(repo_root: Path) -> dict[str, object]:
    """Load the MATLAB hybrid filter fixture (paperHybridFilterExample.mat).

    Returns a dict with: time, delta, X (6×T), mstate (T,),
    A (list of 2), Q (list of 2), p_ij (2×2), Px0 (list of 2), ind.
    """
    import scipy.io as sio

    candidates = [
        repo_root / "nstat" / "data" / "paperHybridFilterExample.mat",
        repo_root / "helpfiles" / "paperHybridFilterExample.mat",
    ]
    mat_path = None
    for p in candidates:
        if p.exists() and p.stat().st_size > 200:  # skip LFS pointers
            mat_path = p
            break

    if mat_path is None:
        raise FileNotFoundError(
            "Cannot find paperHybridFilterExample.mat fixture. "
            "Ensure it is in nstat/data/ or helpfiles/."
        )

    f = sio.loadmat(str(mat_path))
    time = f["time"].ravel().astype(float)
    delta = float(f["delta"].ravel()[0])
    X = f["X"].astype(float)  # (6, T)
    mstate = f["mstate"].ravel().astype(int)  # (T,), values 1 or 2
    p_ij = f["p_ij"].astype(float)  # (2, 2)

    A_cell = f["A"]
    Q_cell = f["Q"]
    Px0_cell = f["Px0"]
    ind_cell = f["ind"]

    A = [A_cell[0, i].astype(float) for i in range(A_cell.shape[1])]
    Q = [Q_cell[0, i].astype(float) for i in range(Q_cell.shape[1])]
    Px0 = [Px0_cell[0, i].astype(float) for i in range(Px0_cell.shape[1])]
    # ind: MATLAB 1-indexed → Python 0-indexed
    ind = [ind_cell[0, i].ravel().astype(int) - 1 for i in range(ind_cell.shape[1])]

    return {
        "time": time,
        "delta": delta,
        "X": X,
        "mstate": mstate,
        "A": A,
        "Q": Q,
        "p_ij": p_ij,
        "Px0": Px0,
        "ind": ind,
    }


def _hybrid_simulate_binomial_spikes(
    xState: np.ndarray,
    muCoeffs: np.ndarray,
    beta: np.ndarray,
    rng: np.random.Generator,
    delta: float = 0.001,
) -> np.ndarray:
    """Simulate binomial spikes from state, baseline, and tuning coefficients.

    MATLAB-canonical encoder: lambda_c = exp(eta)/(1+exp(eta)) per time bin,
    spikes drawn Bernoulli with p = lambda.  Mirrors
    ``examples/paper/example05_decoding_ppaf_pphf.py::_simulate_binomial_spikes``.

    Parameters
    ----------
    xState : (ns, T) array — kinematic state (position + velocity + accel).
    muCoeffs : (C,) array — baseline log-odds per cell.
    beta : (ns, C) array — per-state-dim tuning per cell.
    rng : numpy Generator.
    delta : float — bin width in seconds.

    Returns
    -------
    dN : (C, T) array of {0,1} spike indicators.
    """
    T = xState.shape[1]
    C = len(muCoeffs)
    dataMat = np.column_stack([np.ones(T), xState.T])  # (T, 1+ns)
    coeffs = np.column_stack([muCoeffs, beta.T])  # (C, 1+ns)
    eta = dataMat @ coeffs.T  # (T, C)
    eta = np.clip(eta, -20.0, 20.0)
    expEta = np.exp(eta)
    p = expEta / (1.0 + expEta)
    return (rng.random(p.shape) < p).astype(float).T  # (C, T)


def run_experiment6(
    repo_root: Path, seed: int = 37, *, return_payload: bool = False
) -> dict[str, float] | tuple[dict[str, float], dict[str, object]]:
    """Hybrid Point Process Filter example — canonical MATLAB port.

    Loads the bundled ``paperHybridFilterExample.mat`` fixture (delta=0.001,
    Tmax=2 s, state-space-driven kinematic reach), simulates a 24-cell
    binomial spike population with velocity tuning per Truccolo 2005,
    and runs PPHybridFilterLinear in both goal-directed (with target)
    and free (no target) modes.

    Matches MATLAB helpfiles/HybridFilterExample.m lines 105-268.
    """
    rng = np.random.default_rng(seed)

    fix = _load_hybrid_fixture(repo_root)
    time = fix["time"]
    delta = fix["delta"]
    X = fix["X"]  # (6, T)
    mstate = fix["mstate"]  # (T,), 1 or 2
    p_ij = fix["p_ij"]
    A_models = fix["A"]  # [A_hold (2×2), A_reach (6×6)]
    Px0 = fix["Px0"]  # [Px0_hold (2×2), Px0_reach (6×6)]
    ind = fix["ind"]  # [[0,1], [0,1,2,3,4,5]]
    T = X.shape[1]

    # Recompute Q from trajectory variance — MATLAB lines 197-202
    nonMovingInd = np.where((X[4, :] == 0) & (X[5, :] == 0))[0]
    movingInd = np.setdiff1d(np.arange(T), nonMovingInd)
    Q_reach = np.diag(np.var(np.diff(X[:, movingInd], axis=1), axis=1))
    Q_reach[:4, :4] = 0.0  # only acceleration carries noise
    varNV = np.diag(np.var(np.diff(X[:, nonMovingInd], axis=1), axis=1))
    Q_hold = varNV[:2, :2]
    Q_models = [Q_hold, Q_reach]

    dim_hold = A_models[0].shape[0]  # 2
    dim_reach = A_models[1].shape[0]  # 6

    # Encoding model — matches MATLAB lines 154-157 / 211-214
    n_cells = 24
    muCoeffs = np.log(10.0 * delta) + rng.standard_normal(n_cells)
    beta_full = np.zeros((6, n_cells), dtype=float)
    beta_full[2, :] = 10.0 * (rng.random(n_cells) - 0.5)  # vx tuning
    beta_full[3, :] = 10.0 * (rng.random(n_cells) - 0.5)  # vy tuning

    dN = _hybrid_simulate_binomial_spikes(X, muCoeffs, beta_full, rng, delta=delta)

    # Initial / target conditions per mode — MATLAB lines 250-259
    x0_list = [X[ind[0], 0], X[ind[1], 0]]
    Pi0_list = Px0
    yT_list = [X[ind[0], -1], X[ind[1], -1]]
    piT_list = [1e-9 * np.eye(dim_hold), 1e-9 * np.eye(dim_reach)]
    beta_hold = beta_full[ind[0], :]  # (2, C)
    beta_reach = beta_full[ind[1], :]  # (6, C)
    mu0 = np.array([0.5, 0.5])

    # Goal-directed hybrid decode (with target)
    S_est, X_est, _, MU_est, _, _, _ = DecodingAlgorithms.PPHybridFilterLinear(
        A_models,
        Q_models,
        p_ij,
        mu0,
        dN,
        [muCoeffs, muCoeffs],
        [beta_hold, beta_reach],
        "binomial",
        delta,
        None,
        None,
        x0_list,
        Pi0_list,
        yT_list,
        piT_list,
    )

    # Free hybrid decode (no target)
    S_estNT, X_estNT, _, MU_estNT, _, _, _ = DecodingAlgorithms.PPHybridFilterLinear(
        A_models,
        Q_models,
        p_ij,
        mu0,
        dN,
        [muCoeffs, muCoeffs],
        [beta_hold, beta_reach],
        "binomial",
        delta,
        None,
        None,
        x0_list,
        Pi0_list,
    )

    summary = {
        "num_samples": float(T),
        "num_cells": float(n_cells),
        "state_accuracy": float(np.mean(S_est == mstate)),
        "decode_rmse_x": float(np.sqrt(np.mean((X_est[0, :] - X[0, :]) ** 2))),
        "decode_rmse_y": float(np.sqrt(np.mean((X_est[1, :] - X[1, :]) ** 2))),
    }
    if not return_payload:
        return summary
    payload = {
        "time_s": time,
        "x_pos": X[0, :],
        "y_pos": X[1, :],
        "x_vel": X[2, :],
        "y_vel": X[3, :],
        "state_true": mstate.astype(float),
        "state_hat": S_est.astype(float),
        "state_prob_1": MU_est[0, :],
        "state_prob_2": MU_est[1, :],
        "decoded_x": X_est[0, :],
        "decoded_y": X_est[1, :],
        "decoded_vx": X_est[2, :],
        "decoded_vy": X_est[3, :],
        "decoded_x_free": X_estNT[0, :],
        "decoded_y_free": X_estNT[1, :],
        "decoded_vx_free": X_estNT[2, :],
        "decoded_vy_free": X_estNT[3, :],
        "state_hat_free": S_estNT.astype(float),
        "state_prob_2_free": MU_estNT[1, :],
        "spikes": dN.T,  # (T, C) for notebook plotting
        "n_cells": n_cells,
        "delta": delta,
    }
    return summary, payload


def run_full_paper_examples(repo_root: Path) -> dict[str, dict[str, float]]:
    data_dir = ensure_example_data(download=True)

    return {
        "experiment1": run_experiment1(data_dir),
        "experiment2": run_experiment2(data_dir),
        "experiment3": run_experiment3(),
        "experiment3b": run_experiment3b(data_dir),
        "experiment4": run_experiment4(data_dir),
        "experiment5": run_experiment5(),
        "experiment5b": run_experiment5b(),
        "experiment6": run_experiment6(repo_root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Native Python approximation of nSTATPaperExamples.m")
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    results = run_full_paper_examples(args.repo_root.resolve())
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
