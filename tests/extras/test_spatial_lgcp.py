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

from nstat.extras.spatial import LGCPResult, lgcp_fit

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
