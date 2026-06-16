"""Tests for nstat.extras.spatial.basis — tensor-product B-spline log-rate bases.

Synthetic data only (np.random.default_rng); no patient data.
"""
from __future__ import annotations

import numpy as np
import pytest

from nstat.extras.spatial.basis import (
    BSplineBasis2D,
    bspline_basis_1d,
    bspline_basis_2d,
)
from nstat.glm import fit_poisson_glm


def test_partition_of_unity_clamped():
    rng = np.random.default_rng(0)
    # 1-D: a clamped uniform B-spline basis is a partition of unity on
    # [grid.min(), grid.max()].
    grid_1d = np.linspace(0.0, 1.0, 21)
    B1 = bspline_basis_1d(grid_1d, n_knots=8, degree=3, clamped=True)
    assert B1.shape == (21, 8)
    assert np.allclose(B1.sum(axis=1), 1.0, atol=1e-12)

    # 2-D: tensor product of two POUs is a POU.
    grid_x = np.linspace(0.0, 1.0, 17)
    grid_y = np.linspace(0.0, 1.0, 13)
    B2 = bspline_basis_2d(grid_x, grid_y, n_knots=6, degree=3, clamped=True)
    assert B2.shape == (17 * 13, 36)
    assert np.allclose(B2.sum(axis=1), 1.0, atol=1e-12)

    # Random row-spot-check: row (i*Ny + j) is the outer product of marginals.
    i, j = int(rng.integers(0, 17)), int(rng.integers(0, 13))
    By_1d = bspline_basis_1d(grid_y, n_knots=6, degree=3, clamped=True)
    Bx_1d = bspline_basis_1d(grid_x, n_knots=6, degree=3, clamped=True)
    expected = np.outer(Bx_1d[i], By_1d[j]).ravel()
    assert np.allclose(B2[i * 13 + j], expected, atol=1e-12)


def test_recovers_log_gaussian_rate_under_glm():
    """A log-quadratic rate on a 30x30 grid is recovered (>0.95 corr) by
    fitting Poisson GLM on the B-spline design matrix as ``x``."""
    rng = np.random.default_rng(1)
    G = 30
    grid_x = np.linspace(0.0, 1.0, G)
    grid_y = np.linspace(0.0, 1.0, G)
    # Ground truth: a log-Gaussian bump.  Width and amplitude chosen so
    # the IRLS solver in fit_poisson_glm stays in its stable basin (the
    # solver clips eta to [-20, 20] and benefits from a mild ridge on a
    # near-singular tensor-product basis).
    mx, my, sigma = 0.5, 0.5, 0.25
    log_rate = np.empty((G, G))
    for i, x in enumerate(grid_x):
        for j, y_ in enumerate(grid_y):
            log_rate[i, j] = 4.0 - 0.5 * ((x - mx) / sigma) ** 2 - 0.5 * ((y_ - my) / sigma) ** 2
    log_rate_flat = log_rate.ravel()
    rate_flat = np.exp(log_rate_flat)

    # Per-cell Poisson counts (cells are unit-area in the dimensionless basis).
    y = rng.poisson(rate_flat)
    B = bspline_basis_2d(grid_x, grid_y, n_knots=8, degree=3, clamped=True)
    assert B.shape == (G * G, 64)

    # Mild ridge for the near-collinear tensor-product basis.
    res = fit_poisson_glm(B, y, include_intercept=True, l2=1e-3, max_iter=300)
    assert res.converged, "fit_poisson_glm did not converge"
    eta_hat = res.intercept + B @ res.coefficients
    corr = float(np.corrcoef(log_rate_flat, eta_hat)[0, 1])
    assert corr > 0.95, f"log-rate recovery corr={corr:.3f} below 0.95"


def test_n_knots_too_small_raises():
    grid = np.linspace(0.0, 1.0, 11)
    with pytest.raises(ValueError, match="n_knots must be >= degree \\+ 1"):
        bspline_basis_1d(grid, n_knots=3, degree=3)
    with pytest.raises(ValueError, match="n_knots must be >= degree \\+ 1"):
        bspline_basis_2d(grid, grid, n_knots=2, degree=3)


def test_circular_domain_not_implemented():
    grid = np.linspace(0.0, 1.0, 11)
    with pytest.raises(NotImplementedError, match="circular domain stub"):
        bspline_basis_2d(grid, grid, n_knots=6, domain="circular")


def test_gram_is_psd_symmetric():
    rng = np.random.default_rng(2)
    grid_x = np.linspace(0.0, 1.0, 14)
    grid_y = np.linspace(0.0, 1.0, 12)
    basis = BSplineBasis2D.from_grid(grid_x, grid_y, n_knots=(7, 6))
    G = basis.gram()
    assert G.shape == (7 * 6, 7 * 6)
    assert np.allclose(G, G.T, atol=1e-12)
    eig = np.linalg.eigvalsh(G)
    assert eig.min() >= -1e-10, f"min eigenvalue {eig.min():.3e} below -1e-10"
    # Quadratic form non-negative on a random coefficient vector.
    beta = rng.standard_normal(G.shape[0])
    assert float(beta @ G @ beta) >= -1e-10
