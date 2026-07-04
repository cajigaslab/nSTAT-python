"""Tests for nstat.extras.spatial.lgcp_st — spatiotemporal Kronecker LGCP.

Synthetic data only (np.random.default_rng); no patient data.  No code or
test design was ported from any external repository — the model and
solver follow published mathematics (Rasmussen & Williams 2006 Alg. 3.1;
Saatci 2011; Wilson & Nickisch 2015; Hutchinson 1990) cited in
``nstat/extras/spatial/lgcp_st.py``.

Contract checks
----------------
1. The fast Kronecker matrix-vector product (`_kron_matvec`) matches an
   explicit dense ``numpy.kron`` construction to machine precision — the
   load-bearing internal-consistency check for the whole "never form the
   dense matrix" design.
2. An uninformative (large-variance, hence near-zero-precision) prior
   collapses the Newton/IRLS posterior mode to the deterministic
   per-cell Poisson MLE ``log(count / cell_volume)``.  NOTE: the
   technical-brief's validation item literally reads "zero-variance
   limit"; the actual limit that recovers the per-cell MLE is
   variance -> infinity (flat/uninformative prior), not variance -> 0
   (which instead shrinks *toward* the constant prior mean — the
   opposite deterministic limit).  See the CONTRACT SUMMARY for this
   builder invocation for the full explanation; this test implements
   the mathematically correct limit.
3. On a known separable spatial-bump x temporal-sinusoid rate, the
   fitted `rate_map(t)` correlates strongly with the truth at several
   times.
4. The credible band widens in data-sparse cells (same property as
   `lgcp.py`'s spatial-only test, at the group-mean level to absorb the
   Hutchinson stochastic-diagonal estimator's noise).
5. With `Gt=1`, the spatiotemporal fit closely regresses to the plain
   `lgcp_fit` spatial fit on the same (collapsed) data.
"""
from __future__ import annotations

import numpy as np
import pytest

from nstat.extras.spatial.lgcp import lgcp_fit
from nstat.extras.spatial.lgcp_st import LGCPSTResult, _kron_matvec, lgcp_st_fit

DOMAIN = ((0.0, 1.0), (0.0, 1.0))
PERIOD = (0.0, 1.0)


def _sim_separable(rng, peak_g: float = 6000.0):
    """Spatial Gaussian bump x temporal sinusoid, via rejection sampling."""
    mu = np.array([0.5, 0.5])
    Sigma = np.array([[0.04, 0.0], [0.0, 0.04]])
    Sinv = np.linalg.inv(Sigma)

    def log_g(X):
        d = X - mu
        return np.log(peak_g) - 0.5 * np.einsum("ni,ij,nj->n", d, Sinv, d)

    def h(t):
        return 1.0 + 0.7 * np.sin(2.0 * np.pi * t)

    max_h = 1.7
    n_prop = rng.poisson(peak_g * max_h)
    prop_x = rng.uniform(0.0, 1.0, size=(n_prop, 2))
    prop_t = rng.uniform(0.0, 1.0, size=n_prop)
    lam_prop = np.exp(log_g(prop_x)) * h(prop_t)
    accept = rng.uniform(0.0, 1.0, size=n_prop) < lam_prop / (peak_g * max_h)
    return prop_x[accept], prop_t[accept], log_g, h


# ----------------------------------------------------------------------
# 1. Kronecker matvec correctness (the load-bearing internal check)
# ----------------------------------------------------------------------


def test_kron_matvec_matches_dense_kron():
    rng = np.random.default_rng(0)
    Gy, Gx, Gt = 4, 4, 4
    A = rng.normal(size=(Gy, Gy))
    B = rng.normal(size=(Gx, Gx))
    C = rng.normal(size=(Gt, Gt))
    v = rng.normal(size=Gy * Gx * Gt)

    fast = _kron_matvec(v, [A, B, C], (Gy, Gx, Gt))
    dense = np.kron(np.kron(A, B), C) @ v

    assert np.allclose(fast, dense, atol=1e-10, rtol=1e-10)


def test_kron_matvec_nonsquare_axis_sizes():
    """Same check with Gx != Gy != Gt, exercising the general (non-cubic) path."""
    rng = np.random.default_rng(1)
    Gy, Gx, Gt = 3, 5, 2
    A = rng.normal(size=(Gy, Gy))
    B = rng.normal(size=(Gx, Gx))
    C = rng.normal(size=(Gt, Gt))
    v = rng.normal(size=Gy * Gx * Gt)

    fast = _kron_matvec(v, [A, B, C], (Gy, Gx, Gt))
    dense = np.kron(np.kron(A, B), C) @ v

    assert np.allclose(fast, dense, atol=1e-10, rtol=1e-10)


# ----------------------------------------------------------------------
# 2. Flat/uninformative-prior limit recovers the per-cell Poisson MLE
# ----------------------------------------------------------------------


def test_uninformative_prior_recovers_poisson_mle():
    """As the marginal GP variance -> large (precision -> 0 on every axis
    at once), the Newton/IRLS mode collapses to the deterministic
    per-cell Poisson MLE log(count / cell_volume) -- the plain-Newton
    fixed point once the K^-1 term is negligible next to W.

    `prior_mean` is set explicitly near the true log-rate (rather than
    relying on the `None` default, which subtracts `0.5 * variance` and
    would place the very first Newton iterate absurdly far from the
    optimum at this deliberately extreme variance) -- isolating the
    "flat prior" effect from a separate, well-known pitfall of
    undamped Newton/IRLS started far from the optimum.
    """
    rng = np.random.default_rng(0)
    Gx, Gy, Gt = 3, 3, 2
    n = 3000
    pts = rng.uniform(0.0, 1.0, size=(n, 2))
    times = rng.uniform(0.0, 1.0, size=n)
    total_volume = 1.0 * 1.0 * 1.0
    m0 = float(np.log(n / total_volume))

    res = lgcp_st_fit(
        pts, times, domain=DOMAIN, period=PERIOD, grid=(Gx, Gy, Gt),
        length_scale_space=0.2, length_scale_time=0.2,
        variance=1e6, prior_mean=m0, tol=1e-10, max_iter=100,
    )
    assert res.converged

    mle = np.log(res.counts / res.cell_volume)
    assert np.all(res.counts > 0), "test fixture must avoid the y=0 boundary case"
    assert np.allclose(res.f_mode, mle, atol=1e-6)


# ----------------------------------------------------------------------
# 3. Separable recovery on known ground truth
# ----------------------------------------------------------------------


def test_lgcp_st_recovers_separable_bump_times_sinusoid():
    rng = np.random.default_rng(0)
    pts, times, log_g, h = _sim_separable(rng)
    assert len(pts) > 500

    res = lgcp_st_fit(
        pts, times, domain=DOMAIN, period=PERIOD, grid=(7, 7, 4),
        length_scale_space=0.15, length_scale_time=0.2, nu=1.5, variance=1.0,
    )
    assert isinstance(res, LGCPSTResult)
    assert res.converged

    for t in (0.15, 0.35, 0.5, 0.65, 0.85):
        mean, lo, hi = res.rate_map(t)
        assert mean.shape == lo.shape == hi.shape == (res.grid_x.shape[0],)
        assert np.all(hi >= lo)
        truth = np.exp(log_g(res.grid_x)) * h(t)
        corr = np.corrcoef(np.log(mean), np.log(truth))[0, 1]
        assert corr > 0.85, f"t={t}: corr={corr:.3f} below 0.85"


# ----------------------------------------------------------------------
# 4. Credible-band widening in data-sparse cells
# ----------------------------------------------------------------------


def test_credible_band_wider_in_sparse_cells():
    rng = np.random.default_rng(0)
    pts, times, _, _ = _sim_separable(rng)
    res = lgcp_st_fit(
        pts, times, domain=DOMAIN, period=PERIOD, grid=(7, 7, 4),
        length_scale_space=0.15, length_scale_time=0.2, nu=1.5, variance=1.0,
    )

    t_idx = len(res.grid_t) // 2
    t_mid = float(res.grid_t[t_idx])
    empty = res.counts[t_idx] == 0
    occupied = ~empty
    assert empty.sum() > 0 and occupied.sum() > 0

    # Group-mean comparison (as in lgcp.py's spatial-only test) -- the
    # posterior variance is a stochastic (Hutchinson) estimate here, so
    # a per-cell strict inequality would be noise-sensitive; the
    # aggregate relationship is the property being asserted.
    assert res.f_var[t_idx][empty].mean() > res.f_var[t_idx][occupied].mean()

    mean, lo, hi = res.rate_map(t_mid)
    width = np.log(hi) - np.log(lo)
    assert width[empty].mean() > width[occupied].mean()


def test_rate_map_level_widens_band():
    rng = np.random.default_rng(0)
    pts, times, _, _ = _sim_separable(rng)
    res = lgcp_st_fit(
        pts, times, domain=DOMAIN, period=PERIOD, grid=(6, 6, 3),
        length_scale_space=0.15, length_scale_time=0.2,
    )
    t0 = float(res.grid_t[0])
    _, lo90, hi90 = res.rate_map(t0, level=0.90)
    _, lo50, hi50 = res.rate_map(t0, level=0.50)
    assert np.all((np.log(hi90) - np.log(lo90)) >= (np.log(hi50) - np.log(lo50)) - 1e-9)


# ----------------------------------------------------------------------
# 5. Spatial-only (Gt=1) reduction ties to lgcp_fit
# ----------------------------------------------------------------------


def test_lgcp_st_gt1_regresses_to_lgcp_fit():
    """With Gt=1 the fit should closely track the plain 2-D lgcp_fit on
    the same data (a loose regression tie, not a bit-exact one: the
    spatiotemporal prior is a separable K_x (x) K_y product, while
    lgcp_fit uses an isotropic 2-D Matern on Euclidean distance -- two
    genuinely different covariance functions that only have to agree
    where the likelihood, not the prior, dominates the posterior mode).
    """
    rng = np.random.default_rng(0)
    mu = np.array([0.45, 0.55])
    Sigma = np.array([[0.045, 0.008], [0.008, 0.035]])
    Sinv = np.linalg.inv(Sigma)
    peak = 900.0

    def log_lambda(X):
        d = X - mu
        return np.log(peak) - 0.5 * np.einsum("ni,ij,nj->n", d, Sinv, d)

    n_prop = rng.poisson(peak)
    prop = rng.uniform(0.0, 1.0, size=(n_prop, 2))
    accept = rng.uniform(0.0, 1.0, size=n_prop) < np.exp(log_lambda(prop)) / peak
    pts = prop[accept]
    times = np.zeros(len(pts))
    assert len(pts) > 100

    G = 6
    res_st = lgcp_st_fit(
        pts, times, domain=DOMAIN, period=PERIOD, grid=(G, G, 1),
        length_scale_space=0.15, length_scale_time=0.1, nu=1.5, variance=1.0,
    )
    res_2d = lgcp_fit(pts, DOMAIN, grid=G, kernel="matern32", length_scale=0.15, variance=1.0)

    assert res_st.converged and res_2d.converged
    assert res_st.counts.shape == (1, G * G)
    assert np.allclose(res_st.counts[0], res_2d.counts)
    assert np.isclose(res_st.cell_volume, res_2d.cell_area)
    # Both grids happen to share the same y-slow/x-fast flattening
    # convention (nstat.extras.spatial._kernels.make_grid), so a direct
    # coordinate-wise comparison is valid without re-indexing.
    assert np.allclose(res_st.grid_x, res_2d.grid)

    mean_st, _, _ = res_st.rate_map(0.0)
    mean_2d = res_2d.rate_mean()
    rel = np.linalg.norm(np.log(mean_st) - np.log(mean_2d)) / np.linalg.norm(np.log(mean_2d))
    assert rel < 0.05, f"L2 relative error {rel:.4f} above 5% bound"


# ----------------------------------------------------------------------
# Structural / validation checks
# ----------------------------------------------------------------------


def test_lgcp_st_fit_result_shapes():
    rng = np.random.default_rng(2)
    n = 500
    pts = rng.uniform(0.0, 1.0, size=(n, 2))
    times = rng.uniform(0.0, 1.0, size=n)
    Gx, Gy, Gt = 4, 5, 3
    res = lgcp_st_fit(pts, times, domain=DOMAIN, period=PERIOD, grid=(Gx, Gy, Gt))

    assert isinstance(res, LGCPSTResult)
    assert res.grid_x.shape == (Gx * Gy, 2)
    assert res.grid_t.shape == (Gt,)
    assert res.counts.shape == (Gt, Gx * Gy)
    assert res.f_mode.shape == (Gt, Gx * Gy)
    assert res.f_var.shape == (Gt, Gx * Gy)
    assert res.counts.sum() == n
    assert res.cell_volume > 0
    assert isinstance(res.n_iter, int)
    assert isinstance(res.converged, (bool, np.bool_))


def test_intensity_fn_scalar_and_array_time():
    rng = np.random.default_rng(3)
    n = 500
    pts = rng.uniform(0.0, 1.0, size=(n, 2))
    times = rng.uniform(0.0, 1.0, size=n)
    res = lgcp_st_fit(pts, times, domain=DOMAIN, period=PERIOD, grid=(4, 4, 3))
    fn = res.intensity_fn()

    vals_scalar_t = fn(pts[:5], 0.5)
    assert vals_scalar_t.shape == (5,)
    assert np.all(vals_scalar_t > 0)

    vals_array_t = fn(pts[:5], times[:5])
    assert vals_array_t.shape == (5,)
    assert np.all(vals_array_t > 0)

    # A single-point query (1-D X) also works via np.atleast_2d.
    val_single = fn(pts[0], times[0])
    assert val_single.shape == (1,)


def test_rate_map_level_validation():
    rng = np.random.default_rng(4)
    n = 300
    pts = rng.uniform(0.0, 1.0, size=(n, 2))
    times = rng.uniform(0.0, 1.0, size=n)
    res = lgcp_st_fit(pts, times, domain=DOMAIN, period=PERIOD, grid=(3, 3, 2))
    with pytest.raises(ValueError, match="level must be in"):
        res.rate_map(0.5, level=1.5)
    with pytest.raises(ValueError, match="level must be in"):
        res.rate_map(0.5, level=0.0)


def test_lgcp_st_fit_input_validation():
    rng = np.random.default_rng(5)
    pts = rng.uniform(0.0, 1.0, size=(10, 2))
    times = rng.uniform(0.0, 1.0, size=10)

    with pytest.raises(ValueError, match="times"):
        lgcp_st_fit(pts, times[:5], domain=DOMAIN, period=PERIOD, grid=(3, 3, 2))

    with pytest.raises(ValueError, match="domain must be a 2-tuple"):
        lgcp_st_fit(pts, times, domain=((0.0, 1.0),), period=PERIOD, grid=(3, 3, 2))

    with pytest.raises(ValueError, match="grid must be a 3-tuple"):
        lgcp_st_fit(pts, times, domain=DOMAIN, period=PERIOD, grid=(3, 3))

    with pytest.raises(ValueError, match="points must be"):
        lgcp_st_fit(rng.uniform(size=(10, 3)), times, domain=DOMAIN, period=PERIOD, grid=(3, 3, 2))

    with pytest.raises(ValueError, match="variance must be positive"):
        lgcp_st_fit(pts, times, domain=DOMAIN, period=PERIOD, grid=(3, 3, 2), variance=-1.0)
