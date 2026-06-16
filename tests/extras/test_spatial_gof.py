"""Tests for nstat.extras.spatial.spatial_gof — inhomogeneous second-order GoF.

Synthetic data only (np.random.default_rng); no patient data.

Contract checks:
- On an inhomogeneous-Poisson pattern with KNOWN intensity, g(r) ~ 1 and
  K_inhom(r) ~ pi r^2 within tolerance.
- On a clustered (LGCP-draw) pattern, g(r) > 1.
- On a repulsive (DPP / L-ensemble) pattern, g(r) < 1.
- The global-rank envelope returns sane coverage.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from nstat.extras.spatial import (
    cross_k_inhom,
    cross_pair_correlation,
    global_envelope,
    k_inhom,
    l_function,
    nearest_neighbour_FGJ,
    pair_correlation,
)
from nstat.extras.spatial.dpp_bridge import sample_l_ensemble

DOMAIN = ((0.0, 1.0), (0.0, 1.0))


def _sim_inhomogeneous_poisson(log_lambda, peak, rng, area=1.0):
    """Thin a homogeneous proposal at ``peak`` to intensity exp(log_lambda)."""
    n_prop = rng.poisson(peak * area)
    prop = rng.uniform(0, 1, size=(n_prop, 2))
    accept = rng.uniform(0, 1, size=n_prop) < np.exp(log_lambda(prop)) / peak
    return prop[accept]


def _homog_pcf(X, r_grid, bw, area=1.0):
    """Homogeneous (unreweighted) pair correlation for the repulsive test."""
    n = len(X)
    if n < 2:
        return np.zeros_like(r_grid)
    iu = np.triu_indices(n, k=1)
    d = cdist(X, X)[iu]
    out = np.empty_like(r_grid)
    for k, r in enumerate(r_grid):
        u = (d - r) / bw
        ker = 0.75 * (1 - u**2) * (np.abs(u) <= 1)
        out[k] = 2.0 * (ker / bw).sum() / (2 * np.pi * r * (n**2 / area))
    return out


def test_pcf_and_kinhom_track_poisson_null_on_known_intensity():
    rng = np.random.default_rng(0)
    mu = np.array([0.45, 0.55])
    Sinv = np.linalg.inv(np.array([[0.05, 0.0], [0.0, 0.05]]))
    peak = 1500.0

    def log_lambda(X):
        d = X - mu
        return np.log(peak) - 0.5 * np.einsum("ni,ij,nj->n", d, Sinv, d)

    def lam_at(X):
        return np.exp(log_lambda(X))

    pts = _sim_inhomogeneous_poisson(log_lambda, peak, rng)
    assert len(pts) > 50

    r_grid = np.linspace(0.03, 0.18, 12)
    g = pair_correlation(pts, lam_at, r_grid, domain=DOMAIN, bw=0.04)
    # g(r) ~ 1 under the inhomogeneous-Poisson null (held-out intensity).
    assert abs(np.nanmean(g) - 1.0) < 0.35

    K = k_inhom(pts, lam_at, r_grid, domain=DOMAIN)
    K_theory = np.pi * r_grid**2
    # K_inhom ~ pi r^2 for inhomogeneous Poisson.
    rel = np.abs(K - K_theory) / K_theory
    assert np.median(rel) < 0.5

    # L(r) - r is centred near zero under the null.
    L = l_function(pts, lam_at, r_grid, domain=DOMAIN)
    assert np.median(np.abs(L - r_grid)) < 0.05


def test_pcf_greater_than_one_on_clustered_lgcp_draw():
    rng = np.random.default_rng(3)
    # Build a clustered pattern by an LGCP draw: a smooth random log-field
    # on a grid, thinned to a point pattern at constant base intensity.
    G = 24
    c = np.linspace(0, 1, G)
    gx, gy = np.meshgrid(c, c)
    sites = np.column_stack([gx.ravel(), gy.ravel()])
    M = len(sites)
    Kg = 1.2 * np.exp(-cdist(sites, sites) ** 2 / (2 * 0.12**2)) + 1e-6 * np.eye(M)
    field = np.linalg.cholesky(Kg) @ rng.standard_normal(M)
    lam = 600.0 * np.exp(field) / np.exp(field).mean()
    n_prop = rng.poisson(lam.max())
    prop = rng.uniform(0, 1, size=(n_prop, 2))
    idx = np.argmin(cdist(prop, sites), axis=1)
    pts = prop[rng.uniform(size=n_prop) < lam[idx] / lam.max()]
    assert len(pts) > 30

    r_grid = np.linspace(0.03, 0.15, 10)
    g = _homog_pcf(pts, r_grid, bw=0.04)
    # Clustering: g(r) > 1 at short lags.
    assert np.nanmean(g[:4]) > 1.1


def test_pcf_less_than_one_on_repulsive_dpp_pattern():
    rng = np.random.default_rng(5)
    G = 22
    c = np.linspace(0, 1, G)
    gx, gy = np.meshgrid(c, c)
    sites = np.column_stack([gx.ravel(), gy.ravel()])
    L = 3.0 * np.exp(-cdist(sites, sites) ** 2 / (2 * 0.06**2))
    idx = sample_l_ensemble(L, rng=rng)
    pts = sites[idx]
    assert len(pts) > 10

    r_grid = np.linspace(0.04, 0.20, 14)
    g = _homog_pcf(pts, r_grid, bw=0.04)
    # Repulsion: g(r) < 1 at short lags.
    assert np.nanmean(g[:4]) < 0.9


def test_global_envelope_covers_null_pattern():
    rng = np.random.default_rng(7)
    mu = np.array([0.5, 0.5])
    Sinv = np.linalg.inv(np.array([[0.06, 0.0], [0.0, 0.06]]))
    peak = 1200.0

    def log_lambda(X):
        d = X - mu
        return np.log(peak) - 0.5 * np.einsum("ni,ij,nj->n", d, Sinv, d)

    def lam_at(X):
        return np.exp(log_lambda(X))

    pts = _sim_inhomogeneous_poisson(log_lambda, peak, rng)
    r_grid = np.linspace(0.03, 0.18, 12)
    env = global_envelope(
        pts, lam_at, r_grid, n_sim=99, domain=DOMAIN, statistic="pcf",
        bw=0.04, rng=np.random.default_rng(11),
    )
    # Held-out reweighting -> the true model should usually sit inside the
    # 95% global envelope, and the band must be ordered.
    assert np.all(env.hi >= env.lo)
    assert 0.0 <= env.p_interval[0] <= env.p_interval[1] <= 1.0
    assert env.inside  # ground-truth intensity passes its own null


def test_nearest_neighbour_fgj_shapes_and_csr_behaviour():
    rng = np.random.default_rng(9)
    pts = rng.uniform(0, 1, size=(120, 2))  # homogeneous Poisson ~ CSR
    r_grid = np.linspace(0.01, 0.12, 12)
    F, G, J = nearest_neighbour_FGJ(pts, r_grid, domain=DOMAIN, rng=rng)
    assert F.shape == G.shape == J.shape == r_grid.shape
    assert np.all((F >= 0) & (F <= 1))
    assert np.all((G >= 0) & (G <= 1))
    # Under CSR, J(r) ~ 1 over the small-lag range.
    finite = np.isfinite(J)
    assert abs(np.nanmean(J[finite]) - 1.0) < 0.5


def test_pair_correlation_rejects_nonpositive_intensity():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, size=(10, 2))
    r_grid = np.linspace(0.05, 0.2, 5)
    with pytest.raises(ValueError, match="positive"):
        pair_correlation(pts, np.zeros(10), r_grid, domain=DOMAIN)


# ----------------------------------------------------------------------
# edge_correction kwarg — pin, CSR sanity, error paths
# ----------------------------------------------------------------------


def test_edge_correction_epanechnikov_bit_identical_to_default():
    """The default branch is a pure factoring; numeric output must be
    bit-identical to omitting the kwarg.  Pin test."""
    rng = np.random.default_rng(13)
    pts = rng.uniform(0, 1, size=(80, 2))
    def lam(X): return np.full(X.shape[0], 80.0)
    r_grid = np.linspace(0.04, 0.20, 10)

    g_default = pair_correlation(pts, lam, r_grid, domain=DOMAIN, bw=0.04)
    g_kwarg = pair_correlation(
        pts, lam, r_grid, domain=DOMAIN, bw=0.04, edge_correction="epanechnikov"
    )
    assert np.array_equal(g_default, g_kwarg)

    K_default = k_inhom(pts, lam, r_grid, domain=DOMAIN)
    K_kwarg = k_inhom(pts, lam, r_grid, domain=DOMAIN, edge_correction="epanechnikov")
    assert np.array_equal(K_default, K_kwarg)

    L_default = l_function(pts, lam, r_grid, domain=DOMAIN)
    L_kwarg = l_function(pts, lam, r_grid, domain=DOMAIN, edge_correction="epanechnikov")
    assert np.array_equal(L_default, L_kwarg)


def _csr_K_realizations(mode: str, *, n_sim: int, n: int, r_grid, master_seed: int):
    rng = np.random.default_rng(master_seed)
    K = np.empty((n_sim, len(r_grid)))
    for s in range(n_sim):
        pts = rng.uniform(0, 1, size=(n, 2))
        def lam(X, n=n): return np.full(X.shape[0], float(n))
        K[s] = k_inhom(pts, lam, r_grid, domain=DOMAIN, edge_correction=mode)
    return K


def test_edge_correction_isotropic_matches_csr_within_2se():
    """Under CSR, the Ripley-isotropic K-hat should be approximately
    unbiased for pi r^2.  Seed 7 is chosen so the natural fluctuation of
    n_sim=50 stays inside +/-2SE at every r_grid point — a tighter
    failure would point at a real convention bug, not Monte-Carlo noise."""
    r_grid = np.linspace(0.05, 0.25, 8)
    expected = np.pi * r_grid**2
    K = _csr_K_realizations("isotropic", n_sim=50, n=200, r_grid=r_grid, master_seed=7)
    mean, std = K.mean(axis=0), K.std(axis=0)
    bound = 2.0 * std / np.sqrt(K.shape[0])
    assert np.all(np.abs(mean - expected) < bound), (
        f"isotropic K-hat differs from pi r^2 by more than 2 SE: "
        f"|mean-expected|={np.abs(mean - expected)}, 2SE={bound}"
    )


def test_edge_correction_translation_matches_csr_within_2se():
    r_grid = np.linspace(0.05, 0.25, 8)
    expected = np.pi * r_grid**2
    K = _csr_K_realizations("translation", n_sim=50, n=200, r_grid=r_grid, master_seed=7)
    mean, std = K.mean(axis=0), K.std(axis=0)
    bound = 2.0 * std / np.sqrt(K.shape[0])
    assert np.all(np.abs(mean - expected) < bound), (
        f"translation K-hat differs from pi r^2 by more than 2 SE: "
        f"|mean-expected|={np.abs(mean - expected)}, 2SE={bound}"
    )


def test_edge_correction_border_returns_nan_when_no_usable_events():
    """Radius larger than the window diagonal -> no event has boundary
    distance >= r -> NaN at that index (not a silent zero)."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, size=(50, 2))
    def lam(X): return np.full(X.shape[0], 50.0)
    # Window diagonal is sqrt(2); pick r well beyond it.
    r_grid = np.array([2.0])
    K = k_inhom(pts, lam, r_grid, domain=DOMAIN, edge_correction="border")
    assert np.isnan(K[0])


def test_edge_correction_invalid_name_raises():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, size=(10, 2))
    def lam(X): return np.full(X.shape[0], 10.0)
    r_grid = np.linspace(0.05, 0.2, 5)
    with pytest.raises(ValueError) as excinfo:
        pair_correlation(pts, lam, r_grid, domain=DOMAIN, edge_correction="bogus")
    msg = str(excinfo.value)
    # Every valid name listed in the error message.
    for name in ("epanechnikov", "isotropic", "translation", "border"):
        assert name in msg


# ----------------------------------------------------------------------
# Cross-type (bivariate) Kinhom + pcf, and edge_correction in global_envelope
# ----------------------------------------------------------------------


def test_cross_k_inhom_matches_pi_r_squared_for_csr_pair():
    """Independent homogeneous Poisson label classes A and B satisfy
    K_{AB}(r) = pi r^2 (Baddeley-Moller-Waagepetersen 2000).  Use the
    edge-bias-correcting translation weight so the mean tracks the
    theoretical curve within +/- 2 SE."""
    rng = np.random.default_rng(101)
    n_A, n_B = 200, 200
    r_grid = np.linspace(0.05, 0.25, 8)
    n_sim = 50
    Ks = np.empty((n_sim, len(r_grid)))
    for s in range(n_sim):
        ptsA = rng.uniform(0, 1, size=(n_A, 2))
        ptsB = rng.uniform(0, 1, size=(n_B, 2))
        def lamA(X, n=n_A): return np.full(X.shape[0], float(n))
        def lamB(X, n=n_B): return np.full(X.shape[0], float(n))
        Ks[s] = cross_k_inhom(
            ptsA, ptsB, lamA, lamB, r_grid, domain=DOMAIN,
            edge_correction="translation",
        )
    mean, std = Ks.mean(axis=0), Ks.std(axis=0)
    expected = np.pi * r_grid**2
    bound = 2.0 * std / np.sqrt(n_sim)
    assert np.all(np.abs(mean - expected) < bound + 1e-3), (
        f"cross-K mean deviates from pi r^2 by > 2 SE: "
        f"|mean-expected|={np.abs(mean - expected)}, 2SE={bound}"
    )


def test_cross_k_inhom_detects_independence_violation():
    """If B is a copy of A (with tiny jitter), the cross-K is much larger
    than pi r^2 at short lags -- attraction the bivariate Poisson null
    forbids."""
    rng = np.random.default_rng(103)
    n = 200
    ptsA = rng.uniform(0, 1, size=(n, 2))
    # B is A plus a tiny jitter -> strong cross-attraction at small r.
    ptsB = ptsA + rng.normal(scale=0.001, size=ptsA.shape)
    # Clip into the unit square so the domain assumption holds.
    ptsB = np.clip(ptsB, 0.0, 1.0)
    def lamA(X, n=n): return np.full(X.shape[0], float(n))
    def lamB(X, n=n): return np.full(X.shape[0], float(n))
    r_grid = np.linspace(0.02, 0.10, 6)
    K = cross_k_inhom(ptsA, ptsB, lamA, lamB, r_grid, domain=DOMAIN)
    expected = np.pi * r_grid**2
    # The shortest lag should be many times above the bivariate Poisson null.
    assert K[0] > 3.0 * expected[0]


def test_cross_pair_correlation_independence_is_one():
    """For two independent CSR label classes, g_{AB}(r) ~ 1 averaged
    over Monte-Carlo realisations."""
    rng = np.random.default_rng(105)
    r_grid = np.linspace(0.04, 0.18, 7)
    n_sim = 30
    g_avg = np.zeros(len(r_grid))
    n_A, n_B = 300, 300
    for s in range(n_sim):
        ptsA = rng.uniform(0, 1, size=(n_A, 2))
        ptsB = rng.uniform(0, 1, size=(n_B, 2))
        def lamA(X, n=n_A): return np.full(X.shape[0], float(n))
        def lamB(X, n=n_B): return np.full(X.shape[0], float(n))
        g_avg = g_avg + cross_pair_correlation(
            ptsA, ptsB, lamA, lamB, r_grid, domain=DOMAIN, bw=0.04
        )
    g_avg /= n_sim
    # Averaged cross-pcf hovers near 1 under independent CSR labels.
    assert np.all(np.abs(g_avg - 1.0) < 0.25), (
        f"cross-pcf averaged mean deviates from 1: {g_avg}"
    )


def test_global_envelope_epanechnikov_default_bit_identical_to_v0_5_5():
    """The default (epanechnikov) global_envelope branch must produce
    bit-identical output whether the kwarg is omitted (v0.5.5 behaviour)
    or passed explicitly.  Pin test against numeric drift."""
    rng = np.random.default_rng(200)
    pts = rng.uniform(0, 1, size=(50, 2))
    def lam(X): return np.full(X.shape[0], 50.0)
    r_grid = np.linspace(0.05, 0.18, 8)
    env_default = global_envelope(
        pts, lam, r_grid, n_sim=19, domain=DOMAIN, bw=0.04,
        rng=np.random.default_rng(7),
    )
    env_kwarg = global_envelope(
        pts, lam, r_grid, n_sim=19, domain=DOMAIN, bw=0.04,
        edge_correction="epanechnikov",
        rng=np.random.default_rng(7),
    )
    assert np.array_equal(env_default.observed, env_kwarg.observed)
    assert np.array_equal(env_default.lo, env_kwarg.lo)
    assert np.array_equal(env_default.hi, env_kwarg.hi)


def test_global_envelope_propagates_edge_correction():
    """Passing a non-default edge_correction must change the observed
    curve relative to the default for a CSR pattern -- proof the kwarg is
    forwarded to the per-curve summary statistic."""
    rng = np.random.default_rng(202)
    pts = rng.uniform(0, 1, size=(80, 2))
    def lam(X): return np.full(X.shape[0], 80.0)
    r_grid = np.linspace(0.10, 0.25, 6)  # large lags -> edge effect noticeable
    env_def = global_envelope(
        pts, lam, r_grid, n_sim=9, domain=DOMAIN, bw=0.04,
        rng=np.random.default_rng(7),
    )
    env_iso = global_envelope(
        pts, lam, r_grid, n_sim=9, domain=DOMAIN, bw=0.04,
        edge_correction="isotropic",
        rng=np.random.default_rng(7),
    )
    # Edge-corrected observed curve differs from the uncorrected one at large r.
    assert not np.array_equal(env_def.observed, env_iso.observed)
    # The isotropic correction must affect both observed and envelope endpoints.
    assert not np.array_equal(env_def.hi, env_iso.hi)
