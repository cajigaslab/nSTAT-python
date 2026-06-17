"""Tests for nstat.extras.spatial.inference — minimum-contrast estimator.

Synthetic data only (np.random.default_rng); no patient data.

Contract checks:
- ``min_contrast_estimator`` recovers known parameters on a synthetic curve.
- NaN samples are filtered before integration (no ``inf`` propagation).
- Validation: q<=0, mismatched lengths, non-1-D theta0 → ValueError.
- ``fit_thomas`` recovers ``sigma`` within 20% and ``mu`` within 30% in
  expectation across a small bank of seeds.
- ``fit_matern_cluster`` likewise on the disc radius.
- Optimizer failure does NOT raise — ``success=False`` propagates.
"""
from __future__ import annotations

import numpy as np
import pytest

from nstat.extras.spatial.cluster_cox import (
    matern_cluster_pair_correlation,
    simulate_matern_cluster,
    simulate_thomas,
    thomas_pair_correlation,
)
from nstat.extras.spatial.inference import (
    MinContrastResult,
    fit_matern_cluster,
    fit_thomas,
    min_contrast_estimator,
)


WINDOW = (0.0, 0.0, 1.0, 1.0)
DOMAIN = ((0.0, 1.0), (0.0, 1.0))
SEED = 20260616


# ----------------------------------------------------------------------
# 1. min_contrast_estimator core behaviour
# ----------------------------------------------------------------------


def test_min_contrast_recovers_thomas_parameters_on_noiseless_curve():
    sigma_true, lam_p_true = 0.04, 25.0
    r_grid = np.linspace(0.005, 0.30, 60)
    g_true = thomas_pair_correlation(r_grid, sigma_true, lam_p_true, 1.0)

    def g_model(r, theta):
        s, lam = float(theta[0]), float(theta[1])
        if s <= 0 or lam <= 0:
            return np.full_like(r, np.nan, dtype=float)
        return thomas_pair_correlation(r, s, lam, 1.0)

    res = min_contrast_estimator(
        g_true,
        g_model,
        r_grid,
        np.array([0.02, 10.0]),
        bounds=[(1e-6, None), (1e-3, None)],
    )
    assert isinstance(res, MinContrastResult)
    assert res.success
    sigma_hat, lam_hat = res.theta_hat
    assert sigma_hat == pytest.approx(sigma_true, rel=0.02)
    assert lam_hat == pytest.approx(lam_p_true, rel=0.05)
    assert res.objective_value < 1e-6
    assert res.g_model_at_theta.shape == r_grid.shape


def test_min_contrast_filters_nan_samples_and_does_not_propagate_inf():
    """NaN-valued empirical samples (typical of the border edge correction
    at small r) must be dropped, not raised on or NaN-propagated."""
    sigma_true, lam_p_true = 0.03, 20.0
    r_grid = np.linspace(0.005, 0.25, 50)
    g_true = thomas_pair_correlation(r_grid, sigma_true, lam_p_true, 1.0)
    # Stamp NaN into the first 10 lags — emulate border-correction behaviour.
    g_with_nan = g_true.copy()
    g_with_nan[:10] = np.nan

    def g_model(r, theta):
        s, lam = float(theta[0]), float(theta[1])
        if s <= 0 or lam <= 0:
            return np.full_like(r, np.nan, dtype=float)
        return thomas_pair_correlation(r, s, lam, 1.0)

    res = min_contrast_estimator(
        g_with_nan,
        g_model,
        r_grid,
        np.array([0.02, 10.0]),
        bounds=[(1e-6, None), (1e-3, None)],
    )
    assert res.success, res.message
    assert np.isfinite(res.objective_value)
    sigma_hat, lam_hat = res.theta_hat
    # Dropping the first 10 lags removes the most-informative small-r tail
    # of g(r), so we relax the recovery tolerance here.  The point of the
    # test is that NaN samples are SILENTLY DROPPED — neither raised on
    # nor allowed to NaN-poison the objective — so we check finiteness
    # and a loose neighbourhood, not bit-level recovery.
    assert sigma_hat > 0 and lam_hat > 0
    assert sigma_hat == pytest.approx(sigma_true, rel=0.5)
    assert lam_hat == pytest.approx(lam_p_true, rel=2.0)


def test_min_contrast_validation_errors():
    r_grid = np.linspace(0.01, 0.2, 20)
    g_emp = np.ones_like(r_grid)

    def g_model(r, theta):
        return thomas_pair_correlation(r, float(theta[0]), float(theta[1]), 1.0)

    # q <= 0 → ValueError.
    with pytest.raises(ValueError, match="q"):
        min_contrast_estimator(g_emp, g_model, r_grid, np.array([0.05, 10.0]), q=0.0)
    # Mismatched lengths → ValueError.
    with pytest.raises(ValueError, match="shape"):
        min_contrast_estimator(
            g_emp[:-1], g_model, r_grid, np.array([0.05, 10.0])
        )
    # theta0 not 1-D → ValueError.
    with pytest.raises(ValueError, match="1-D"):
        min_contrast_estimator(
            g_emp, g_model, r_grid, np.array([[0.05, 10.0]])
        )


def test_min_contrast_too_few_finite_samples_returns_unsuccessful():
    """If <4 samples remain after NaN filtering, the result is unsuccessful
    but the call does NOT raise."""
    r_grid = np.linspace(0.01, 0.2, 20)
    g_emp = np.full_like(r_grid, np.nan)
    g_emp[0] = 1.5  # one finite, three NaN-flanking — well below the 4 floor

    def g_model(r, theta):
        return np.ones_like(r)

    res = min_contrast_estimator(
        g_emp, g_model, r_grid, np.array([0.05, 10.0])
    )
    assert not res.success
    assert "finite" in res.message.lower() or "samples" in res.message.lower()
    assert res.n_iter == 0


# ----------------------------------------------------------------------
# 2. fit_thomas / fit_matern_cluster on simulated patterns
# ----------------------------------------------------------------------


def test_fit_thomas_recovers_parameters_within_tolerance_across_seeds():
    """Architect tolerance: ±20% sigma, ±30% mu in expectation.

    We average over 5 seeds because single-seed Poisson fluctuations on a
    pattern of ~500 points yield estimator variance comparable to the
    tolerance band — the tolerance is on the expectation.
    """
    sigma_true, lam_p_true, mu_true = 0.025, 30.0, 15.0
    area = 1.0
    r_grid = np.linspace(0.005, 0.20, 40)

    sigmas, lam_ps, mus = [], [], []
    for seed in range(SEED, SEED + 5):
        rng = np.random.default_rng(seed)
        pts = simulate_thomas(lam_p_true, mu_true, sigma_true, WINDOW, rng=rng)
        res = fit_thomas(pts, DOMAIN, r_grid)
        assert res.success, res.message
        sigma_hat, lam_p_hat = res.theta_hat
        mu_hat = pts.shape[0] / (lam_p_hat * area)
        sigmas.append(sigma_hat)
        lam_ps.append(lam_p_hat)
        mus.append(mu_hat)

    assert abs(np.mean(sigmas) - sigma_true) / sigma_true < 0.20
    assert abs(np.mean(mus) - mu_true) / mu_true < 0.30


def test_fit_matern_cluster_recovers_radius_within_tolerance_across_seeds():
    R_true, lam_p_true, mu_true = 0.05, 30.0, 15.0
    area = 1.0
    r_grid = np.linspace(0.005, 0.20, 40)

    Rs, mus = [], []
    for seed in range(SEED, SEED + 5):
        rng = np.random.default_rng(seed)
        pts = simulate_matern_cluster(lam_p_true, mu_true, R_true, WINDOW, rng=rng)
        res = fit_matern_cluster(pts, DOMAIN, r_grid)
        assert res.success, res.message
        R_hat, lam_p_hat = res.theta_hat
        mu_hat = pts.shape[0] / (lam_p_hat * area)
        Rs.append(R_hat)
        mus.append(mu_hat)

    assert abs(np.mean(Rs) - R_true) / R_true < 0.20
    assert abs(np.mean(mus) - mu_true) / mu_true < 0.30


def test_min_contrast_respects_bounds_and_returns_message():
    """L-BFGS-B bounds must clamp the estimate to the feasible region, and
    the optimizer's message must propagate (no swallowing of diagnostics)."""
    r_grid = np.linspace(0.005, 0.30, 60)
    g_emp = thomas_pair_correlation(r_grid, 0.04, 25.0, 1.0)

    def g_model(r, theta):
        s, lam = float(theta[0]), float(theta[1])
        if s <= 0 or lam <= 0:
            return np.full_like(r, np.nan, dtype=float)
        return thomas_pair_correlation(r, s, lam, 1.0)

    # Force the optimizer to land at the lower-bound corner of an
    # implausible box by setting an upper bound BELOW the truth.
    res = min_contrast_estimator(
        g_emp,
        g_model,
        r_grid,
        np.array([0.005, 1.0]),
        bounds=[(0.001, 0.01), (0.5, 5.0)],
    )
    assert isinstance(res.message, str) and res.message  # non-empty
    sigma_hat, lam_hat = res.theta_hat
    # Estimates respect the box.
    assert 0.001 <= sigma_hat <= 0.01
    assert 0.5 <= lam_hat <= 5.0


def test_fit_thomas_accepts_custom_theta0_and_rejects_wrong_shape():
    rng = np.random.default_rng(SEED)
    pts = simulate_thomas(30.0, 15.0, 0.03, WINDOW, rng=rng)
    r_grid = np.linspace(0.005, 0.20, 40)
    # Custom theta0 — ok.
    res = fit_thomas(pts, DOMAIN, r_grid, theta0=(0.05, 20.0))
    assert isinstance(res, MinContrastResult)
    assert res.theta_hat.shape == (2,)
    # Wrong-shape theta0 → ValueError.
    with pytest.raises(ValueError, match="shape"):
        fit_thomas(pts, DOMAIN, r_grid, theta0=np.array([0.05, 20.0, 1.0]))
