"""Tests for nstat.extras.spatial.residuals.pp_residuals_smoothed.

Kernel-smoothed Pearson residuals on a uniform time grid (Brown et al.
2002; Andersen 1997; Truccolo et al. 2005).  Synthetic data only.
"""
from __future__ import annotations

import numpy as np
import pytest

from nstat.extras.spatial import pp_residuals_smoothed


def test_pp_residuals_smoothed_true_model_is_centered():
    """When the model rate matches the data-generating rate, the
    smoothed residuals hover near zero (Brown et al. 2002)."""
    rng = np.random.default_rng(31)
    T = 4000
    # Time-varying spike probability.
    lam = 0.05 + 0.04 * np.sin(np.linspace(0, 8 * np.pi, T))
    counts = (rng.uniform(size=T) < lam).astype(int)
    spike_bins = np.flatnonzero(counts)
    t_grid, resid = pp_residuals_smoothed(spike_bins, lam, bandwidth=20.0)
    # Under the true model the mean residual is ~ 0 and the bulk stays small.
    assert abs(resid.mean()) < 0.02
    # Smoothed residuals are small in magnitude (bins are 0/1; lam ~ 0.05).
    assert np.percentile(np.abs(resid), 95) < 0.10


def test_pp_residuals_smoothed_misspecified_shows_drift():
    """If the model under-predicts the rate by a constant factor, the
    smoothed residuals are systematically positive (more spikes than
    predicted)."""
    rng = np.random.default_rng(33)
    T = 4000
    lam_true = np.full(T, 0.08)
    counts = (rng.uniform(size=T) < lam_true).astype(int)
    spike_bins = np.flatnonzero(counts)
    # Mis-specify: model thinks the rate is half what it really is.
    lam_model = 0.5 * lam_true
    t_grid, resid = pp_residuals_smoothed(spike_bins, lam_model, bandwidth=20.0)
    # Residual mean = E[N_k] - lam_model = 0.08 - 0.04 = 0.04 > 0.
    assert resid.mean() > 0.02
    # And dominantly positive (not centred at zero).
    assert (resid > 0).mean() > 0.7


def test_pp_residuals_smoothed_shape():
    """Output shapes match the model length T, regardless of how many
    spike bins are supplied."""
    rng = np.random.default_rng(35)
    T = 1000
    lam = np.full(T, 0.05)
    spike_bins = rng.integers(0, T, size=50)
    t_grid, resid = pp_residuals_smoothed(spike_bins, lam, bandwidth=10.0, dt=0.001)
    assert t_grid.shape == (T,)
    assert resid.shape == (T,)
    # dt scaling -> t_grid spans 0 to T * dt.
    assert abs(t_grid[0] - 0.5 * 0.001) < 1e-12
    assert abs(t_grid[-1] - (T - 0.5) * 0.001) < 1e-12
    # No spikes at all -> residual is just -lam smoothed, which is negative.
    t_g2, r2 = pp_residuals_smoothed(np.array([], dtype=int), lam, bandwidth=10.0)
    assert r2.shape == (T,)
    assert (r2 < 0).all()


def test_pp_residuals_smoothed_validation():
    """Reject zero/negative bandwidth, negative rates, and out-of-range
    spike bins."""
    lam = np.full(100, 0.05)
    with pytest.raises(ValueError, match="bandwidth"):
        pp_residuals_smoothed(np.array([0, 1, 2]), lam, bandwidth=0.0)
    with pytest.raises(ValueError, match="bandwidth"):
        pp_residuals_smoothed(np.array([0, 1, 2]), lam, bandwidth=-1.0)
    bad_lam = lam.copy()
    bad_lam[5] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        pp_residuals_smoothed(np.array([0]), bad_lam, bandwidth=5.0)
    # Empty lam_per_bin.
    with pytest.raises(ValueError, match="non-empty"):
        pp_residuals_smoothed(np.array([], dtype=int), np.array([]), bandwidth=5.0)
    # Out-of-range spike bin index.
    with pytest.raises(ValueError, match=r"\[0, T\)"):
        pp_residuals_smoothed(np.array([150]), lam, bandwidth=5.0)
