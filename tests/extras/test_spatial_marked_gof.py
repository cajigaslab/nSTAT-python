"""Tests for nstat.extras.spatial.marked_gof — discrete-time-rescaling KS.

Synthetic data only (np.random.default_rng); no patient data.

Contract checks:
- At finite bin width with the TRUE model, the UNCORRECTED KS rejects more
  than nominal (statistic above the band) while the CORRECTED KS passes
  (inside the band) — Haslinger-Pipa-Brown 2010.
- A mark-misspecification case rejects on the mark axis while the time
  axis still passes.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from nstat.extras.spatial import marked_gof
from nstat.extras.spatial.marked_gof import (
    corrected_rescaled,
    marked_time_rescaling,
    uncorrected_rescaled,
)


def _simulate_marked_train(rng, T=6000, Delta=0.020):
    """Finite-bin marked spike train from known place fields on a 1D track."""
    centres = np.array([0.2, 0.5, 0.8])
    mark_means = np.array([40.0, 80.0, 120.0])
    field_sd, mark_sd, peak_rate = 0.07, 8.0, 12.0
    x_true = 0.5 + 0.45 * np.sin(np.linspace(0, 8 * np.pi, T))
    lam0 = np.zeros(T)
    spike_bin, spike_m = [], []
    for k, xk in enumerate(x_true):
        fr = peak_rate * np.exp(-((xk - centres) ** 2) / (2 * field_sd**2))
        lam0[k] = fr.sum()
        if rng.uniform() < lam0[k] * Delta:
            j = rng.choice(3, p=fr / fr.sum())
            spike_bin.append(k)
            spike_m.append(rng.normal(mark_means[j], mark_sd))
    info = dict(
        centres=centres, mark_means=mark_means, mark_sd=mark_sd,
        field_sd=field_sd, peak_rate=peak_rate, x_true=x_true,
    )
    return np.array(spike_bin), np.array(spike_m), lam0 * Delta, info


def test_uncorrected_rejects_corrected_passes_at_finite_bin_width():
    rng = np.random.default_rng(1)
    sb, sm, p_k, info = _simulate_marked_train(rng)
    n = len(sb)
    assert n > 50
    # finite bin width: mean p_k ~ 0.12 makes the discreteness bias visible
    assert 0.05 < p_k[sb].mean() < 0.25

    band = 1.358 / np.sqrt(n)
    u_unc = uncorrected_rescaled(sb, p_k)
    ks_unc = stats.kstest(u_unc, "uniform").statistic

    draws = [
        corrected_rescaled(sb, p_k, np.random.default_rng(100 + d))
        for d in range(30)
    ]
    ks_corr = np.mean([stats.kstest(u, "uniform").statistic for u in draws])

    # The bug: uncorrected statistic is biased UP relative to corrected.
    assert ks_unc > ks_corr
    # Uncorrected false-rejects (outside band); corrected passes (inside).
    assert ks_unc > band
    assert ks_corr < band


def test_marked_time_rescaling_helper_with_conditional_mark_cdf():
    """marked_gof.marked_time_rescaling(sb, spike_m, p_k, mark_cdf, decoded=...)."""
    rng = np.random.default_rng(2)
    sb, sm, p_k, info = _simulate_marked_train(rng)
    centres = info["centres"]
    mark_means = info["mark_means"]
    mark_sd = info["mark_sd"]
    field_sd = info["field_sd"]
    peak_rate = info["peak_rate"]
    x_true = info["x_true"]

    def mark_cdf(m, x):
        w = peak_rate * np.exp(-((x - centres) ** 2) / (2 * field_sd**2))
        w = w / w.sum()
        return float((w * stats.norm.cdf(m, mark_means, mark_sd)).sum())

    res = marked_time_rescaling(
        sb, sm, p_k, mark_cdf, decoded=x_true, n_draws=25,
        rng=np.random.default_rng(7),
    )
    # Time axis: corrected passes, uncorrected rejects.
    assert res.inside_corrected
    assert not res.inside_uncorrected
    # Mark axis: the correct (mixture-aware) mark model passes.
    assert res.ks_mark is not None
    assert res.inside_mark


def test_mark_misspecification_rejects_on_mark_axis_only():
    rng = np.random.default_rng(3)
    sb, sm, p_k, info = _simulate_marked_train(rng)
    n = len(sb)
    band = 1.358 / np.sqrt(n)

    # Misspecified mark model: a single Gaussian where truth is a 3-comp
    # mixture. Time axis is untouched -> still passes (corrected); the
    # mark axis must reject.
    mu_hat, sd_hat = sm.mean(), sm.std()

    def mark_cdf_bad(m, x):
        return float(stats.norm.cdf(m, mu_hat, sd_hat))

    res = marked_time_rescaling(
        sb, sm, p_k, mark_cdf_bad, decoded=info["x_true"], n_draws=25,
        rng=np.random.default_rng(13),
    )
    # Time axis still passes with the discrete-time correction.
    assert res.inside_corrected
    # Mark axis rejects (single-Gaussian ignores the multimodality).
    assert res.ks_mark > band
    assert res.inside_mark is False


def test_multivariate_per_channel_rescaling_runs():
    rng = np.random.default_rng(4)
    # Two independent channels, true model -> both should pass (corrected).
    n_bins = 4000
    Delta = 0.02
    p1 = np.full(n_bins, 6.0 * Delta)
    p2 = np.full(n_bins, 10.0 * Delta)
    sb1 = np.flatnonzero(rng.uniform(size=n_bins) < p1)
    sb2 = np.flatnonzero(rng.uniform(size=n_bins) < p2)
    out = marked_gof.multivariate_time_rescaling(
        [sb1, sb2], [p1, p2], n_draws=20, rng=np.random.default_rng(99),
    )
    assert set(out) == {0, 1}
    for c in (0, 1):
        assert out[c].inside_corrected
