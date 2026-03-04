"""Shared deterministic performance workloads for nSTAT-python parity tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from nstat.compat.matlab import CIF, Covariate, DecodingAlgorithms, History, nstColl


TIER_ORDER = ("S", "M", "L")
CASE_ORDER = (
    "unit_impulse_basis",
    "covariate_resample",
    "history_design_matrix",
    "simulate_cif_thinning",
    "decoding_spike_rate_cis",
)


@dataclass(frozen=True)
class CaseConfig:
    basis_width_s: float = 0.02
    min_time_s: float = 0.0
    max_time_s: float = 1.0
    sample_rate_hz: float = 500.0
    n_spikes: int = 200
    n_grid: int = 1000
    duration_s: float = 2.0
    n_realizations: int = 5
    max_time_res_s: float = 0.001
    num_basis: int = 4
    num_trials: int = 6
    n_bins: int = 120
    mc_draws: int = 30
    decode_delta_s: float = 0.01


def get_case_config(case: str, tier: str) -> CaseConfig:
    tier = tier.upper()
    if tier not in TIER_ORDER:
        raise ValueError(f"Unknown tier: {tier}")

    vals: dict[str, dict[str, float | int]]
    if case == "unit_impulse_basis":
        vals = {
            "S": dict(max_time_s=1.0, sample_rate_hz=500.0),
            "M": dict(max_time_s=2.0, sample_rate_hz=1000.0),
            "L": dict(max_time_s=4.0, sample_rate_hz=1500.0),
        }
    elif case == "covariate_resample":
        vals = {
            "S": dict(duration_s=2.0, n_grid=2001, sample_rate_hz=500.0),
            "M": dict(duration_s=4.0, n_grid=4001, sample_rate_hz=750.0),
            "L": dict(duration_s=6.0, n_grid=6001, sample_rate_hz=1000.0),
        }
    elif case == "history_design_matrix":
        vals = {
            "S": dict(n_spikes=200, n_grid=1000, duration_s=2.0),
            "M": dict(n_spikes=1000, n_grid=5000, duration_s=2.0),
            "L": dict(n_spikes=3000, n_grid=10000, duration_s=2.0),
        }
    elif case == "simulate_cif_thinning":
        vals = {
            "S": dict(duration_s=1.0, n_realizations=5, max_time_res_s=0.001),
            "M": dict(duration_s=2.0, n_realizations=10, max_time_res_s=0.001),
            "L": dict(duration_s=3.0, n_realizations=20, max_time_res_s=0.001),
        }
    elif case == "decoding_spike_rate_cis":
        vals = {
            "S": dict(num_basis=4, num_trials=6, n_bins=120, mc_draws=30, decode_delta_s=0.01),
            "M": dict(num_basis=6, num_trials=8, n_bins=200, mc_draws=50, decode_delta_s=0.01),
            "L": dict(num_basis=8, num_trials=12, n_bins=320, mc_draws=80, decode_delta_s=0.01),
        }
    else:
        raise ValueError(f"Unknown case: {case}")

    return CaseConfig(**cast(dict[str, Any], vals[tier]))


def _deterministic_spike_times(n_spikes: int, duration_s: float) -> np.ndarray:
    idx = np.arange(1, n_spikes + 1, dtype=float)
    phi = 0.6180339887498949
    spikes = np.mod(idx * phi, 1.0) * float(duration_s)
    return np.sort(spikes)


def _deterministic_decode_inputs(cfg: CaseConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis_idx = np.arange(1, cfg.num_basis + 1, dtype=float)[:, None]
    trial_idx = np.arange(1, cfg.num_trials + 1, dtype=float)[None, :]
    xk = 0.06 * np.sin(0.37 * basis_idx * trial_idx) + 0.04 * np.cos(0.19 * basis_idx * trial_idx)

    wku = np.zeros((cfg.num_basis, cfg.num_basis, cfg.num_trials, cfg.num_trials), dtype=float)
    for r in range(cfg.num_basis):
        wku[r, r, :, :] = 0.05 * np.eye(cfg.num_trials, dtype=float)

    grid = np.arange(cfg.num_trials * cfg.n_bins, dtype=float).reshape(cfg.num_trials, cfg.n_bins)
    d_n = ((np.sin(0.173 * grid) + np.cos(0.037 * grid)) > 1.15).astype(float)
    return xk, wku, d_n


def run_python_workload(case: str, tier: str, seed: int = 20260303) -> dict[str, float]:
    """Execute one deterministic Python workload and return summary metrics."""

    cfg = get_case_config(case=case, tier=tier)

    if case == "unit_impulse_basis":
        basis = nstColl.generateUnitImpulseBasis(
            cfg.basis_width_s,
            cfg.min_time_s,
            cfg.max_time_s,
            cfg.sample_rate_hz,
        )
        mat = basis.data_to_matrix()
        return {
            "rows": float(mat.shape[0]),
            "cols": float(mat.shape[1]),
            "total_mass": float(np.sum(mat)),
        }

    if case == "covariate_resample":
        t = np.linspace(0.0, cfg.duration_s, cfg.n_grid, dtype=float)
        y = np.sin(2.0 * np.pi * 3.0 * t) + 0.2 * np.cos(2.0 * np.pi * 9.0 * t)
        cov = Covariate(t, y, "Stimulus")
        out = cov.resample(cfg.sample_rate_hz)
        mat = out.data_to_matrix()
        return {
            "rows": float(mat.shape[0]),
            "cols": float(mat.shape[1]),
            "signal_energy": float(np.mean(mat[:, 0] ** 2)),
        }

    if case == "history_design_matrix":
        spikes = _deterministic_spike_times(cfg.n_spikes, cfg.duration_s)
        t_grid = np.linspace(0.0, cfg.duration_s, cfg.n_grid, dtype=float)
        hist = History(np.array([0.0, 0.01, 0.02, 0.05, 0.10], dtype=float))
        mat = hist.computeHistory(spikes, t_grid)
        return {
            "rows": float(mat.shape[0]),
            "cols": float(mat.shape[1]),
            "total_count": float(np.sum(mat)),
        }

    if case == "simulate_cif_thinning":
        np.random.seed(seed)
        t = np.linspace(0.0, cfg.duration_s, int(cfg.duration_s * 1000) + 1, dtype=float)
        lam = 12.0 + 8.0 * np.sin(2.0 * np.pi * 3.0 * t)
        lam = np.clip(lam, 0.2, None)
        lam_cov = Covariate(t, lam, "Lambda")
        coll = CIF.simulateCIFByThinningFromLambda(lam_cov, cfg.n_realizations, cfg.max_time_res_s)
        total_spikes = float(sum(train.spike_times.size for train in coll.trains))
        return {
            "num_units": float(coll.getNumUnits()),
            "total_spikes": total_spikes,
            "mean_spikes_per_unit": total_spikes / max(float(coll.getNumUnits()), 1.0),
        }

    if case == "decoding_spike_rate_cis":
        np.random.seed(seed)
        xk, wku, d_n = _deterministic_decode_inputs(cfg)
        t0 = 0.0
        tf = (cfg.n_bins - 1) * cfg.decode_delta_s
        spike_rate_sig, prob_mat, sig_mat = DecodingAlgorithms.computeSpikeRateCIs(
            xk,
            wku,
            d_n,
            t0,
            tf,
            "binomial",
            cfg.decode_delta_s,
            0.0,
            [],
            cfg.mc_draws,
            0.05,
        )
        rate = spike_rate_sig.data_to_matrix()
        return {
            "num_trials": float(prob_mat.shape[0]),
            "prob_mean": float(np.mean(prob_mat)),
            "sig_count": float(np.sum(sig_mat)),
            "rate_mean": float(np.mean(rate)),
        }

    raise ValueError(f"Unhandled workload case: {case}")
