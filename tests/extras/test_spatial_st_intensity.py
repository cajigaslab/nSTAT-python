"""Tests for nstat.extras.spatial.st_intensity — space-time kernel intensity.

Synthetic data only (np.random.default_rng); no patient data.

Contract checks:
- Mass conservation: the Riemann sum of lambda_hat over W x T approximately
  equals the event count N (Diggle's edge-corrected estimator property).
- Separable recovery: on a KNOWN separable intensity m(x)*mu(t), the fitted
  separable estimator recovers the spatial and temporal marginal shapes
  (correlation > 0.9 with the truth on the grid).
- .evaluate() at the grid centres matches the stored .intensity array.
- Edge/degenerate inputs: single event, empty points/times, mismatched
  shapes, malformed domain/period, unknown kernel, non-positive bandwidth.
"""
from __future__ import annotations

import numpy as np
import pytest

from nstat.extras.spatial.st_intensity import STIntensityResult, intensity_st_kde

DOMAIN = ((0.0, 1.0), (0.0, 1.0))
PERIOD = (0.0, 1.0)


# ----------------------------------------------------------------------
# 1. Mass conservation
# ----------------------------------------------------------------------


def test_mass_conservation_nonseparable_gaussian():
    rng = np.random.default_rng(0)
    n = 500
    # Keep events comfortably inside the window so the query-point edge
    # correction is close to 1 almost everywhere (the estimator's mass
    # conservation is exact only for interior events; see module Notes).
    pts = rng.uniform(0.2, 0.8, size=(n, 2))
    times = rng.uniform(0.2, 0.8, size=n)

    grid = (24, 24, 20)
    res = intensity_st_kde(pts, times, domain=DOMAIN, period=PERIOD, grid=grid)
    assert isinstance(res, STIntensityResult)
    assert res.intensity.shape == (grid[2], grid[0] * grid[1])
    assert np.all(np.isfinite(res.intensity))
    assert np.all(res.intensity >= 0.0)

    dx = 1.0 / grid[0]
    dy = 1.0 / grid[1]
    dt = 1.0 / grid[2]
    mass = float(res.intensity.sum()) * dx * dy * dt
    rel_err = abs(mass - n) / n
    assert rel_err < 0.1, f"mass={mass:.2f} vs N={n}, rel_err={rel_err:.3f}"


def test_mass_conservation_epanechnikov_kernel():
    rng = np.random.default_rng(2)
    n = 400
    pts = rng.uniform(0.25, 0.75, size=(n, 2))
    times = rng.uniform(0.25, 0.75, size=n)

    grid = (20, 20, 16)
    res = intensity_st_kde(
        pts, times, domain=DOMAIN, period=PERIOD, grid=grid, kernel="epanechnikov",
    )
    dx = 1.0 / grid[0]
    dy = 1.0 / grid[1]
    dt = 1.0 / grid[2]
    mass = float(res.intensity.sum()) * dx * dy * dt
    rel_err = abs(mass - n) / n
    assert rel_err < 0.15, f"mass={mass:.2f} vs N={n}, rel_err={rel_err:.3f}"


# ----------------------------------------------------------------------
# 2. Separable recovery
# ----------------------------------------------------------------------


def _m_shape(X: np.ndarray, mu_c: np.ndarray, Sigma_inv: np.ndarray) -> np.ndarray:
    d = X - mu_c
    return np.exp(-0.5 * np.einsum("ni,ij,nj->n", d, Sigma_inv, d))


def _mu_shape(t: np.ndarray) -> np.ndarray:
    # Sinusoid-modulated rate, kept strictly positive on [0, 1]: in [0.4, 1.6].
    return 1.0 + 0.6 * np.sin(2.0 * np.pi * t)


def _simulate_separable(rng: np.random.Generator, n_target: int = 4000):
    """Draw N events from a known separable intensity m(x) * mu(t).

    For a Poisson process with a separable intensity, the (x, t) marks
    are independent draws from the (renormalised) spatial and temporal
    marginals — so each coordinate can be sampled on its own axis.
    """
    mu_c = np.array([0.5, 0.5])
    Sigma = np.array([[0.03, 0.0], [0.0, 0.03]])
    Sigma_inv = np.linalg.inv(Sigma)

    xs = []
    while sum(len(a) for a in xs) < n_target:
        need = n_target - sum(len(a) for a in xs)
        cand = rng.multivariate_normal(mu_c, Sigma, size=need * 2)
        inside = (
            (cand[:, 0] >= 0.0) & (cand[:, 0] <= 1.0)
            & (cand[:, 1] >= 0.0) & (cand[:, 1] <= 1.0)
        )
        xs.append(cand[inside])
    x_pts = np.concatenate(xs)[:n_target]

    peak_mu = 1.6
    ts = []
    while sum(len(a) for a in ts) < n_target:
        need = n_target - sum(len(a) for a in ts)
        cand = rng.uniform(0.0, 1.0, size=int(need / 0.6) + 20)
        accept = rng.uniform(0.0, 1.0, size=cand.shape[0]) < _mu_shape(cand) / peak_mu
        ts.append(cand[accept])
    t_pts = np.concatenate(ts)[:n_target]

    return x_pts, t_pts, mu_c, Sigma_inv


def test_separable_recovers_spatial_and_temporal_marginals():
    rng = np.random.default_rng(0)
    pts, times, mu_c, Sigma_inv = _simulate_separable(rng, n_target=4000)
    assert pts.shape == (4000, 2)
    assert times.shape == (4000,)

    res = intensity_st_kde(
        pts, times, domain=DOMAIN, period=PERIOD, separable=True, grid=(30, 30, 25),
    )
    assert res.separable is True

    # In the separable model, intensity = outer(mu_hat, m_hat) / N, so any
    # single row is proportional to m_hat(x) and any single column is
    # proportional to mu_hat(t) — shape (not scale) is what we validate.
    row = res.intensity[0, :]
    col = res.intensity[:, 0]

    truth_m = _m_shape(res.grid_x, mu_c, Sigma_inv)
    truth_mu = _mu_shape(res.grid_t)

    corr_m = float(np.corrcoef(row, truth_m)[0, 1])
    corr_mu = float(np.corrcoef(col, truth_mu)[0, 1])
    assert corr_m > 0.9, f"spatial marginal corr={corr_m:.3f} below 0.9"
    assert corr_mu > 0.9, f"temporal marginal corr={corr_mu:.3f} below 0.9"


# ----------------------------------------------------------------------
# 3. evaluate() consistency with the stored grid
# ----------------------------------------------------------------------


def test_evaluate_matches_grid_nonseparable():
    rng = np.random.default_rng(3)
    pts = rng.uniform(0.2, 0.8, size=(200, 2))
    times = rng.uniform(0.2, 0.8, size=200)
    grid = (6, 6, 5)
    res = intensity_st_kde(pts, times, domain=DOMAIN, period=PERIOD, grid=grid)

    for k, t in enumerate(res.grid_t):
        vals = res.evaluate(res.grid_x, float(t))
        assert vals.shape == (grid[0] * grid[1],)
        assert np.allclose(vals, res.intensity[k], rtol=1e-8, atol=1e-8)


def test_evaluate_matches_grid_separable():
    rng = np.random.default_rng(4)
    pts = rng.uniform(0.2, 0.8, size=(200, 2))
    times = rng.uniform(0.2, 0.8, size=200)
    grid = (5, 5, 4)
    res = intensity_st_kde(
        pts, times, domain=DOMAIN, period=PERIOD, separable=True, grid=grid,
    )

    for k, t in enumerate(res.grid_t):
        vals = res.evaluate(res.grid_x, float(t))
        assert np.allclose(vals, res.intensity[k], rtol=1e-8, atol=1e-8)


def test_evaluate_single_point_broadcast():
    rng = np.random.default_rng(5)
    pts = rng.uniform(0.2, 0.8, size=(100, 2))
    times = rng.uniform(0.2, 0.8, size=100)
    res = intensity_st_kde(pts, times, domain=DOMAIN, period=PERIOD, grid=(8, 8, 6))

    # Single (x, t) query.
    val = res.evaluate(np.array([0.5, 0.5]), 0.5)
    assert val.shape == (1,)
    assert np.isfinite(val[0]) and val[0] > 0

    # Many x, single scalar t broadcasts across all x.
    xs = rng.uniform(0.2, 0.8, size=(10, 2))
    vals = res.evaluate(xs, 0.5)
    assert vals.shape == (10,)

    # Single x, many t broadcasts across all t.
    vals2 = res.evaluate(np.array([0.5, 0.5]), np.linspace(0.2, 0.8, 7))
    assert vals2.shape == (7,)


def test_evaluate_mismatched_lengths_raise():
    rng = np.random.default_rng(6)
    pts = rng.uniform(0.2, 0.8, size=(50, 2))
    times = rng.uniform(0.2, 0.8, size=50)
    res = intensity_st_kde(pts, times, domain=DOMAIN, period=PERIOD, grid=(6, 6, 5))

    with pytest.raises(ValueError, match="matching length"):
        res.evaluate(rng.uniform(size=(3, 2)), rng.uniform(size=5))


# ----------------------------------------------------------------------
# 4. Edge / degenerate inputs
# ----------------------------------------------------------------------


def test_single_event_does_not_raise():
    res = intensity_st_kde(
        np.array([[0.5, 0.5]]), np.array([0.5]),
        domain=DOMAIN, period=PERIOD, grid=(5, 5, 4),
    )
    assert res.intensity.shape == (4, 25)
    assert np.all(np.isfinite(res.intensity))
    assert np.all(res.intensity >= 0.0)


def test_empty_points_and_times_raises():
    with pytest.raises(ValueError, match="at least one event"):
        intensity_st_kde(
            np.zeros((0, 2)), np.zeros((0,)), domain=DOMAIN, period=PERIOD,
        )


def test_mismatched_points_times_shapes_raises():
    with pytest.raises(ValueError, match="must align"):
        intensity_st_kde(
            np.zeros((5, 2)), np.zeros((3,)), domain=DOMAIN, period=PERIOD,
        )


def test_malformed_domain_raises():
    with pytest.raises(ValueError, match="domain must be"):
        intensity_st_kde(
            np.zeros((2, 2)) + 0.1, np.zeros((2,)) + 0.1,
            domain=((0.0, 1.0),), period=PERIOD,
        )


def test_malformed_period_raises():
    with pytest.raises(ValueError, match="period must be"):
        intensity_st_kde(
            np.zeros((2, 2)) + 0.1, np.zeros((2,)) + 0.1,
            domain=DOMAIN, period=(1.0,),
        )


def test_unknown_kernel_raises():
    with pytest.raises(ValueError, match="kernel must be one of"):
        intensity_st_kde(
            np.zeros((2, 2)) + 0.1, np.zeros((2,)) + 0.1,
            domain=DOMAIN, period=PERIOD, kernel="box",
        )


def test_non_positive_bandwidth_raises():
    with pytest.raises(ValueError, match="bw_space must be positive"):
        intensity_st_kde(
            np.zeros((2, 2)) + 0.1, np.zeros((2,)) + 0.1,
            domain=DOMAIN, period=PERIOD, bw_space=-1.0,
        )
    with pytest.raises(ValueError, match="bw_time must be positive"):
        intensity_st_kde(
            np.zeros((2, 2)) + 0.1, np.zeros((2,)) + 0.1,
            domain=DOMAIN, period=PERIOD, bw_time=0.0,
        )


def test_invalid_grid_raises():
    with pytest.raises(ValueError, match="grid sizes must all be >= 1"):
        intensity_st_kde(
            np.zeros((2, 2)) + 0.1, np.zeros((2,)) + 0.1,
            domain=DOMAIN, period=PERIOD, grid=(0, 5, 5),
        )


def test_period_defaults_to_time_range():
    rng = np.random.default_rng(7)
    pts = rng.uniform(0.2, 0.8, size=(100, 2))
    times = rng.uniform(0.3, 0.7, size=100)
    res = intensity_st_kde(pts, times, domain=DOMAIN, grid=(6, 6, 5))
    assert res.period == (float(times.min()), float(times.max()))
