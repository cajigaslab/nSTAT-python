"""Tests for nstat.extras.spatial.lgcp — Laplace LGCP rate map.

Synthetic data only (np.random.default_rng); no patient data.

Contract checks:
- On a simulated single-bump field, the recovered rate_map mean tracks
  the truth (correlation high; peak located near the true centre).
- The credible band is WIDER in data-sparse cells than dense ones (the
  spec's key property: where W -> 0, posterior var relaxes to the GP prior).
"""
from __future__ import annotations

import numpy as np

from nstat.extras.spatial import LGCPResult, lgcp_fit, lgcp_fit_glm, MaternPrior
from nstat.extras.spatial.basis import BSplineBasis2D

DOMAIN = ((0.0, 1.0), (0.0, 1.0))


def _sim_single_bump(rng):
    mu = np.array([0.45, 0.55])
    Sigma = np.array([[0.045, 0.008], [0.008, 0.035]])
    Sinv = np.linalg.inv(Sigma)
    peak = 900.0

    def log_lambda(X):
        d = X - mu
        return np.log(peak) - 0.5 * np.einsum("ni,ij,nj->n", d, Sinv, d)

    n_prop = rng.poisson(peak)
    prop = rng.uniform(0, 1, size=(n_prop, 2))
    accept = rng.uniform(0, 1, size=n_prop) < np.exp(log_lambda(prop)) / peak
    return prop[accept], mu, log_lambda


def test_lgcp_recovers_single_bump_mean():
    rng = np.random.default_rng(0)
    pts, mu, log_lambda = _sim_single_bump(rng)
    assert len(pts) > 100

    res = lgcp_fit(pts, DOMAIN, grid=20, kernel="matern52", length_scale=0.12)
    assert isinstance(res, LGCPResult)
    assert res.converged

    mean, lo, hi = res.rate_map(level=0.90)
    assert mean.shape == lo.shape == hi.shape == (res.grid.shape[0],)
    assert np.all(hi >= lo)

    # Recovered mean tracks the truth: high rank correlation with the
    # true intensity on the grid, and the argmax cell near the true centre.
    truth = np.exp(log_lambda(res.grid))
    corr = np.corrcoef(np.log(mean), np.log(truth))[0, 1]
    assert corr > 0.8

    peak_cell = res.grid[np.argmax(mean)]
    assert np.linalg.norm(peak_cell - mu) < 0.18


def test_credible_band_wider_in_sparse_cells():
    rng = np.random.default_rng(1)
    pts, _, _ = _sim_single_bump(rng)
    res = lgcp_fit(pts, DOMAIN, grid=20, kernel="matern52", length_scale=0.12)

    empty = res.counts == 0
    occupied = ~empty
    assert empty.sum() > 0 and occupied.sum() > 0

    # KEY PROPERTY: posterior log-rate variance is larger in data-sparse
    # cells (W -> 0 there, so v -> GP prior variance).
    assert res.f_var[empty].mean() > res.f_var[occupied].mean()

    # And that translates into a wider multiplicative band in empty cells.
    mean, lo, hi = res.rate_map(level=0.90)
    width = np.log(hi) - np.log(lo)
    assert width[empty].mean() > width[occupied].mean()


def test_rate_map_level_widens_band():
    rng = np.random.default_rng(2)
    pts, _, _ = _sim_single_bump(rng)
    res = lgcp_fit(pts, DOMAIN, grid=16)
    _, lo90, hi90 = res.rate_map(level=0.90)
    _, lo50, hi50 = res.rate_map(level=0.50)
    # A higher credible level gives a wider band everywhere.
    assert np.all((np.log(hi90) - np.log(lo90)) >= (np.log(hi50) - np.log(lo50)) - 1e-9)


def test_intensity_fn_callable_for_gof_reweighting():
    rng = np.random.default_rng(3)
    pts, _, _ = _sim_single_bump(rng)
    res = lgcp_fit(pts, DOMAIN, grid=16)
    fn = res.intensity_fn()
    vals = fn(pts[:5])
    assert vals.shape == (5,)
    assert np.all(vals > 0)


# ----------------------------------------------------------------------
# Tier B: basis-projected lgcp_fit_glm + MaternPrior
# ----------------------------------------------------------------------


def _basis_and_prior(G: int, n_knots: int = 8, length_scale: float = 0.25):
    grid_x = np.linspace(0.0, 1.0, G)
    grid_y = np.linspace(0.0, 1.0, G)
    basis = BSplineBasis2D.from_grid(grid_x, grid_y, n_knots=n_knots)
    prior = MaternPrior(nu=2.5, length_scale=length_scale, marginal_var=1.0)
    return basis, prior


def test_lgcp_fit_glm_matches_dense_on_small_grid():
    """On a small grid both routines should produce a close log-rate field.

    Acceptance: L2 relative error < 5%.  (Spec allows a 7.5% fallback if
    the 5% bound proved infeasible — not needed in practice.)
    """
    rng = np.random.default_rng(10)
    pts, _, _ = _sim_single_bump(rng)
    G = 32
    # Match length scales between the two prior interpretations (the dense
    # path puts the Matern on cell centres; here it sits on coefficient
    # anchor points, which are slightly more spread out — a moderately
    # longer length scale matches the same effective smoothness).
    res_dense = lgcp_fit(
        pts, DOMAIN, grid=G, kernel="matern52", length_scale=0.18, variance=1.0,
    )
    basis, prior = _basis_and_prior(G, n_knots=10, length_scale=0.18)
    res_glm = lgcp_fit_glm(pts, DOMAIN, basis, prior, grid=G)

    assert isinstance(res_glm, LGCPResult)
    assert res_glm.converged
    assert res_glm.f_mode.shape == res_dense.f_mode.shape
    # lgcp_fit (dense) flattens in xy ordering; lgcp_fit_glm flattens in
    # ij ordering.  Align before the L2 comparison by re-laying out the
    # dense field through its own grid coordinates and the glm field
    # through its: both are functions of the same (x, y) tuples, so we
    # compare them at matched coordinates.
    dense_field = {tuple(c): f for c, f in zip(res_dense.grid, res_dense.f_mode)}
    glm_at_dense_order = np.array(
        [dense_field[tuple(c)] for c in res_glm.grid]
    )
    diff = res_glm.f_mode - glm_at_dense_order
    rel = np.linalg.norm(diff) / max(np.linalg.norm(glm_at_dense_order), 1e-12)
    assert rel < 0.05, f"L2 relative error {rel:.4f} above 5% bound"


def test_lgcp_fit_glm_credible_band_widens_in_empty_region():
    rng = np.random.default_rng(11)
    pts, _, _ = _sim_single_bump(rng)
    G = 32
    basis, prior = _basis_and_prior(G, n_knots=10, length_scale=0.18)
    res = lgcp_fit_glm(pts, DOMAIN, basis, prior, grid=G)

    # The simulator concentrates mass around (0.45, 0.55); the
    # opposite-corner quadrant is data-sparse.
    centres = res.grid
    sparse = (centres[:, 0] > 0.7) & (centres[:, 1] < 0.3)
    dense = (centres[:, 0] > 0.35) & (centres[:, 0] < 0.55) & (
        centres[:, 1] > 0.45
    ) & (centres[:, 1] < 0.65)
    assert sparse.any() and dense.any()
    sparse_band = res.f_var[sparse].mean()
    dense_band = res.f_var[dense].mean()
    assert sparse_band >= 2.0 * dense_band, (
        f"sparse band {sparse_band:.3f} not >= 2x dense band {dense_band:.3f}"
    )


def test_lgcp_fit_glm_recovers_known_log_quadratic_rate():
    rng = np.random.default_rng(12)
    # Log-quadratic ground truth on the unit square — a quadratic in (x, y)
    # is exactly representable by a cubic B-spline tensor product, so this
    # is the canonical sanity check on the basis-projected fit.
    mx, my, sigma = 0.5, 0.5, 0.28

    def log_lambda(X):
        return 5.0 - 0.5 * ((X[:, 0] - mx) / sigma) ** 2 - 0.5 * ((X[:, 1] - my) / sigma) ** 2

    n_prop = rng.poisson(1500)
    prop = rng.uniform(0, 1, size=(n_prop, 2))
    accept = rng.uniform(0, 1, size=n_prop) < np.exp(
        log_lambda(prop) - log_lambda(np.array([[mx, my]]))[0]
    )
    pts = prop[accept]

    G = 32
    basis, prior = _basis_and_prior(G, n_knots=10, length_scale=0.3)
    res = lgcp_fit_glm(pts, DOMAIN, basis, prior, grid=G)
    truth = log_lambda(res.grid)
    corr = float(np.corrcoef(res.f_mode, truth)[0, 1])
    assert corr > 0.95, f"log-rate recovery corr={corr:.3f} below 0.95"


def test_matern_prior_validation_errors():
    import pytest

    with pytest.raises(ValueError, match="nu must be one of"):
        MaternPrior(nu=1.0, length_scale=0.2)
    with pytest.raises(ValueError, match="length_scale must be positive"):
        MaternPrior(nu=2.5, length_scale=0.0)
    with pytest.raises(ValueError, match="length_scale must be positive"):
        MaternPrior(nu=2.5, length_scale=-0.1)
    with pytest.raises(ValueError, match="marginal_var must be positive"):
        MaternPrior(nu=2.5, length_scale=0.2, marginal_var=0.0)
    with pytest.raises(ValueError, match="jitter must be non-negative"):
        MaternPrior(nu=2.5, length_scale=0.2, jitter=-1e-6)


def test_matern_prior_k_psd_symmetric():
    prior = MaternPrior(nu=2.5, length_scale=0.3, marginal_var=2.0, jitter=1e-6)
    rng = np.random.default_rng(13)
    coords = rng.uniform(0, 1, size=(20, 2))
    K = prior.K(coords)
    assert K.shape == (20, 20)
    assert np.allclose(K, K.T, atol=1e-12)
    eig = np.linalg.eigvalsh(K)
    assert eig.min() > 0.0, f"K not PD: min eig = {eig.min():.3e}"

    K_inv = prior.K_inv(coords)
    I = K_inv @ K
    assert np.allclose(I, np.eye(20), atol=1e-8)

    log_det = prior.log_det(coords)
    sign, logdet_ref = np.linalg.slogdet(K)
    assert sign == 1.0
    assert abs(log_det - logdet_ref) < 1e-8


def test_lgcp_fit_glm_completes_on_64x64():
    """End-to-end runtime check on a realistic grid.

    Acceptance: < 30 s wall-clock.  (Spec allows up to 60 s if the basis
    factorization needs more iterations on this seed — has not been
    needed in CI.)
    """
    import time

    rng = np.random.default_rng(14)
    pts, _, _ = _sim_single_bump(rng)
    G = 64
    basis, prior = _basis_and_prior(G, n_knots=10, length_scale=0.18)
    t0 = time.time()
    res = lgcp_fit_glm(pts, DOMAIN, basis, prior, grid=G)
    dt = time.time() - t0
    assert res.converged
    assert res.f_mode.shape == (G * G,)
    assert dt < 30.0, f"lgcp_fit_glm on 64x64 took {dt:.2f}s > 30s"
