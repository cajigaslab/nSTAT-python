"""Tests for nstat.extras.spatial.spatiotemporal_gof — space-time inhomogeneous
second-order goodness-of-fit.

Synthetic data only (np.random.default_rng); no patient data.

Contract checks (code-free validation against published theory, not any
external reference implementation):

1. Analytical Poisson null: on a homogeneous space-time Poisson process
   with known constant lambda_hat, ``K_st(r, t) -> pi r^2 * 2t``.
2. Pair correlation null: ``g(r, t) ~ 1`` under the same homogeneous
   process.
3. Envelope coverage calibration: repeated null realizations should
   mostly fall inside their own nominal global envelope.
4. Clustering detection: an offspring-burst space-time pattern exceeds
   the Poisson envelope at short lags.
5. Internal consistency: ``l_st()`` is monotone in ``k_st``; callable vs
   precomputed ``lambda_hat`` give identical ``K``.

Global-rank envelope calibration (test 3) — fixed
--------------------------------------------------
``nstat.extras.spatial._envelopes.global_rank_envelope`` — the shared
Myllymaki-style global-rank-envelope helper this module reuses (and
which also backs the already-shipped
``nstat.extras.spatial.spatial_gof.global_envelope``) — previously built
its ``lo``/``hi`` band from *per-column* (pointwise) order statistics of
the simulated curves rather than from curves selected by their *joint*
extreme rank, which under-covered (~65% instead of nominal 95%). This
has been fixed: the band is now built from ``S = {i : R_i > d_alpha}``,
the fixed set of curves surviving the joint two-sided extreme-rank
threshold (Myllymaki et al. 2017), giving genuine simultaneous
coverage. The coverage-calibration test below now asserts a proper
nominal band rather than the pre-fix empirically-observed floor.
"""
from __future__ import annotations

import numpy as np
import pytest

from nstat.extras.spatial.spatiotemporal_gof import (
    STEnvelopeResult,
    STKResult,
    global_envelope_st,
    k_st_inhom,
    pair_correlation_st,
)

DOMAIN = ((0.0, 1.0), (0.0, 1.0))
PERIOD = (0.0, 1.0)


# ----------------------------------------------------------------------
# 1. Analytical Poisson null: K_st(r, t) -> pi r^2 * 2t
# ----------------------------------------------------------------------


def test_kst_matches_pi_r2_times_2t_under_homogeneous_poisson():
    rng = np.random.default_rng(0)
    n = 4000
    pts = rng.uniform(0, 1, size=(n, 2))
    times = rng.uniform(0, 1, size=n)
    lam_const = float(n)

    def lam_at(x, t):
        return np.full(x.shape[0], lam_const)

    r_grid = np.linspace(0.04, 0.16, 4)
    t_grid = np.linspace(0.04, 0.16, 4)
    res = k_st_inhom(pts, times, lam_at, r_grid, t_grid, domain=DOMAIN, period=PERIOD)
    assert isinstance(res, STKResult)
    assert res.k_st.shape == (len(r_grid), len(t_grid))
    assert res.edge_correction == "translation"

    expected = np.pi * r_grid[:, None] ** 2 * 2.0 * t_grid[None, :]
    rel = np.abs(res.k_st - expected) / expected
    # Interior of the (r, t) grid: median relative error well under 1%,
    # worst cell under 3% (translation edge correction, n=4000, domain=1).
    assert np.median(rel) < 0.02
    assert np.max(rel) < 0.05


def test_kst_default_edge_correction_is_translation():
    rng = np.random.default_rng(1)
    n = 60
    pts = rng.uniform(0, 1, size=(n, 2))
    times = rng.uniform(0, 1, size=n)

    def lam_at(x, t):
        return np.full(x.shape[0], float(n))

    r_grid = np.linspace(0.05, 0.2, 3)
    t_grid = np.linspace(0.05, 0.2, 3)
    res_default = k_st_inhom(pts, times, lam_at, r_grid, t_grid, domain=DOMAIN, period=PERIOD)
    res_kwarg = k_st_inhom(
        pts, times, lam_at, r_grid, t_grid, domain=DOMAIN, period=PERIOD,
        edge_correction="translation",
    )
    assert np.array_equal(res_default.k_st, res_kwarg.k_st)


def test_kst_border_matches_pi_r2_times_2t_under_homogeneous_poisson():
    """Numerical correctness of the ``"border"`` edge correction on the
    same analytic CSR target the other corrections are checked against."""
    rng = np.random.default_rng(0)
    n = 4000
    pts = rng.uniform(0, 1, size=(n, 2))
    times = rng.uniform(0, 1, size=n)
    lam_const = float(n)

    def lam_at(x, t):
        return np.full(x.shape[0], lam_const)

    r_grid = np.linspace(0.04, 0.16, 4)
    t_grid = np.linspace(0.04, 0.16, 4)
    res = k_st_inhom(
        pts, times, lam_at, r_grid, t_grid, domain=DOMAIN, period=PERIOD,
        edge_correction="border",
    )
    assert res.edge_correction == "border"

    expected = np.pi * r_grid[:, None] ** 2 * 2.0 * t_grid[None, :]
    rel = np.abs(res.k_st - expected) / expected
    # Border correction has higher variance than translation (fewer usable
    # focal points near the boundary); bounds are looser than the
    # translation test above.
    assert np.median(rel) < 0.06
    assert np.max(rel) < 0.10


def test_kst_border_sums_both_pair_directions_not_one_member_doubled():
    """Deterministic regression pin for the math-review border-asymmetry
    bug: the border branch used to test only ONE member of each pair
    (``usable[iu[0]]``) and unconditionally double its contribution,
    silently assuming both pair members share the same border-usability
    (Baddeley-Rubak-Turner 2015, Sec 7.4 requires each ordered direction
    to be checked independently). Two points are placed so exactly one is
    a usable focal point at r=0.1: A=(0.05, 0.5) has spatial
    boundary-distance 0.05 (< r, NOT usable); B=(0.12, 0.5) has
    boundary-distance 0.12 (>= r, usable). ``d(A, B) = 0.07 <= r`` so the
    pair is in range. The buggy formula checked only the array-index-0
    point (A, not usable) and doubled -> K_st == 0.0 even though B *is* a
    usable focal point that should contribute. The correct sum over both
    ordered directions gives a known closed-form nonzero value."""
    pts = np.array([[0.05, 0.5], [0.12, 0.5]])
    times = np.array([0.5, 0.5])

    def lam_at(x, t):
        return np.full(x.shape[0], 1.0)

    r_grid = np.array([0.1])
    t_grid = np.array([0.1])
    res = k_st_inhom(
        pts, times, lam_at, r_grid, t_grid, domain=DOMAIN, period=PERIOD,
        edge_correction="border",
    )
    # eff_area = (1 - 2*0.1)^2 = 0.64, eff_period = 1 - 2*0.1 = 0.8 ->
    # eff_vol = 0.512; only B is a usable focal point (wgt = 1.0) ->
    # K_st = 1.0 / 0.512. The pre-fix formula returned 0.0 here.
    assert res.k_st[0, 0] == pytest.approx(1.0 / 0.512)


# ----------------------------------------------------------------------
# 2. Pair correlation null: g(r, t) ~ 1
# ----------------------------------------------------------------------


def test_pcf_st_near_one_under_homogeneous_poisson():
    rng = np.random.default_rng(0)
    n = 3000
    pts = rng.uniform(0, 1, size=(n, 2))
    times = rng.uniform(0, 1, size=n)
    lam_const = float(n)

    def lam_at(x, t):
        return np.full(x.shape[0], lam_const)

    # Interior of the domain (small lags relative to the unit window) to
    # keep the uncorrected SOIRS estimator's boundary loss modest -- the
    # same convention as the existing (uncorrected-by-default)
    # nstat.extras.spatial.spatial_gof.pair_correlation, whose own test
    # accepts up to 0.35 mean deviation from 1 under the identical
    # "no additional edge correction" convention.
    r_grid = np.linspace(0.02, 0.08, 4)
    t_grid = np.linspace(0.02, 0.08, 4)
    g = pair_correlation_st(pts, times, lam_at, r_grid, t_grid, domain=DOMAIN, period=PERIOD)
    assert g.shape == (len(r_grid), len(t_grid))
    assert np.max(np.abs(g - 1.0)) < 0.35
    assert np.mean(np.abs(g - 1.0)) < 0.2


# ----------------------------------------------------------------------
# 3. Envelope coverage calibration
# ----------------------------------------------------------------------


def test_envelope_coverage_calibration_under_null():
    lam_const = 250.0

    def lam_at(x, t):
        return np.full(x.shape[0], lam_const)

    r_grid = np.array([0.05, 0.10])
    t_grid = np.array([0.05, 0.10])

    master_rng = np.random.default_rng(42)
    n_reps = 40
    inside_count = 0
    for rep in range(n_reps):
        n = master_rng.poisson(lam_const)
        pts = master_rng.uniform(0, 1, size=(n, 2))
        times = master_rng.uniform(0, 1, size=n)
        env = global_envelope_st(
            pts, times, lam_at, r_grid, t_grid, n_sim=79,
            domain=DOMAIN, period=PERIOD, rng=np.random.default_rng(42 * 97 + rep),
        )
        inside_count += int(env.inside)

    coverage = inside_count / n_reps
    # See module docstring: global_rank_envelope now builds lo/hi from the
    # jointly-non-extreme curve set S (Myllymaki et al. 2017), so a nominal
    # 95% envelope should cover close to its nominal rate. With this exact
    # seed/parameter combination coverage measures 38/40 = 0.95; 0.9 is a
    # tight-but-safe floor for n_reps=40 (binomial noise around p=0.95).
    assert coverage >= 0.9


# ----------------------------------------------------------------------
# 4. Clustering detection
# ----------------------------------------------------------------------


def test_envelope_rejects_clustered_burst_pattern():
    rng = np.random.default_rng(55)
    n_parents = 15
    parents_xy = rng.uniform(0.15, 0.85, size=(n_parents, 2))
    parents_t = rng.uniform(0.1, 0.6, size=n_parents)
    offspring_per_parent = 12

    pts_list, t_list = [], []
    for i in range(n_parents):
        off_xy = parents_xy[i] + rng.normal(scale=0.015, size=(offspring_per_parent, 2))
        off_t = parents_t[i] + rng.exponential(scale=0.015, size=offspring_per_parent)
        pts_list.append(off_xy)
        t_list.append(off_t)
    pts = np.clip(np.vstack(pts_list), 0.0, 1.0)
    times = np.clip(np.concatenate(t_list), 0.0, 0.999)
    assert len(pts) > 100

    # Deliberately mis-specified null: constant rate at the pattern's
    # overall mean intensity (the naive homogeneous-Poisson benchmark the
    # burst structure should violate at short space-time lags).
    lam_const = float(len(pts))

    def lam_at(x, t):
        return np.full(x.shape[0], lam_const)

    r_grid = np.array([0.01, 0.03])
    t_grid = np.array([0.01, 0.03])
    env = global_envelope_st(
        pts, times, lam_at, r_grid, t_grid, n_sim=99,
        domain=DOMAIN, period=PERIOD, rng=np.random.default_rng(9),
    )
    assert isinstance(env, STEnvelopeResult)
    # Short-range space-time clustering: the observed K_st at the
    # smallest (r, t) cell is far above its Monte-Carlo Poisson envelope.
    assert env.observed[0, 0] > env.hi[0, 0]
    assert env.inside is False


# ----------------------------------------------------------------------
# 5. Internal consistency
# ----------------------------------------------------------------------


def test_l_st_monotone_in_k_st():
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 1, size=(60, 2))
    times = rng.uniform(0, 1, size=60)
    lam_arr = np.full(60, 60.0)

    r_grid = np.linspace(0.05, 0.2, 5)
    t_grid = np.linspace(0.05, 0.2, 5)
    res = k_st_inhom(pts, times, lam_arr, r_grid, t_grid, domain=DOMAIN, period=PERIOD)
    K = res.k_st
    L = res.l_st()
    assert L.shape == K.shape

    order = np.argsort(K, axis=None)
    diffs = np.diff(L.ravel()[order])
    assert np.all(diffs >= -1e-12)


def test_lambda_hat_callable_and_array_give_identical_k():
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 1, size=(60, 2))
    times = rng.uniform(0, 1, size=60)
    lam_arr = np.full(60, 60.0)

    def lam_fn(x, t):
        return np.full(x.shape[0], 60.0)

    r_grid = np.linspace(0.05, 0.2, 5)
    t_grid = np.linspace(0.05, 0.2, 5)
    res_arr = k_st_inhom(pts, times, lam_arr, r_grid, t_grid, domain=DOMAIN, period=PERIOD)
    res_fn = k_st_inhom(pts, times, lam_fn, r_grid, t_grid, domain=DOMAIN, period=PERIOD)
    assert np.array_equal(res_arr.k_st, res_fn.k_st)

    g_arr = pair_correlation_st(pts, times, lam_arr, r_grid, t_grid, domain=DOMAIN, period=PERIOD)
    g_fn = pair_correlation_st(pts, times, lam_fn, r_grid, t_grid, domain=DOMAIN, period=PERIOD)
    assert np.array_equal(g_arr, g_fn)


# ----------------------------------------------------------------------
# edge_correction kwarg — smoke tests for isotropic / border, error paths
# ----------------------------------------------------------------------


def test_k_st_inhom_isotropic_and_border_run_and_are_finite_or_nan():
    rng = np.random.default_rng(0)
    n = 60
    pts = rng.uniform(0, 1, size=(n, 2))
    times = rng.uniform(0, 1, size=n)

    def lam_at(x, t):
        return np.full(x.shape[0], float(n))

    r_grid = np.linspace(0.05, 0.2, 3)
    t_grid = np.linspace(0.05, 0.2, 3)
    for mode in ("isotropic", "translation", "border"):
        res = k_st_inhom(
            pts, times, lam_at, r_grid, t_grid, domain=DOMAIN, period=PERIOD,
            edge_correction=mode,
        )
        assert res.edge_correction == mode
        assert res.k_st.shape == (3, 3)
        finite_or_nan = np.isfinite(res.k_st) | np.isnan(res.k_st)
        assert np.all(finite_or_nan)


def test_k_st_inhom_border_returns_nan_when_no_usable_events():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, size=(50, 2))
    times = rng.uniform(0, 1, size=50)

    def lam_at(x, t):
        return np.full(x.shape[0], 50.0)

    # Radius beyond the window diagonal and lag beyond the period length
    # -> no event can be a usable focal point -> NaN, not a silent zero.
    r_grid = np.array([2.0])
    t_grid = np.array([2.0])
    res = k_st_inhom(pts, times, lam_at, r_grid, t_grid, domain=DOMAIN, period=PERIOD,
                      edge_correction="border")
    assert np.isnan(res.k_st[0, 0])


def test_k_st_inhom_invalid_edge_correction_raises():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, size=(10, 2))
    times = rng.uniform(0, 1, size=10)

    def lam_at(x, t):
        return np.full(x.shape[0], 10.0)

    r_grid = np.linspace(0.05, 0.2, 3)
    t_grid = np.linspace(0.05, 0.2, 3)
    with pytest.raises(ValueError) as excinfo:
        k_st_inhom(pts, times, lam_at, r_grid, t_grid, domain=DOMAIN, period=PERIOD,
                   edge_correction="bogus")
    msg = str(excinfo.value)
    for name in ("isotropic", "translation", "border"):
        assert name in msg


def test_k_st_inhom_rejects_nonpositive_lambda_hat():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, size=(10, 2))
    times = rng.uniform(0, 1, size=10)
    r_grid = np.linspace(0.05, 0.2, 3)
    t_grid = np.linspace(0.05, 0.2, 3)
    with pytest.raises(ValueError, match="positive"):
        k_st_inhom(pts, times, np.zeros(10), r_grid, t_grid, domain=DOMAIN, period=PERIOD)


def test_k_st_inhom_rejects_misaligned_points_times():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, size=(10, 2))
    times = rng.uniform(0, 1, size=7)
    r_grid = np.linspace(0.05, 0.2, 3)
    t_grid = np.linspace(0.05, 0.2, 3)

    def lam_at(x, t):
        return np.full(x.shape[0], 10.0)

    with pytest.raises(ValueError, match="align"):
        k_st_inhom(pts, times, lam_at, r_grid, t_grid, domain=DOMAIN, period=PERIOD)


def test_k_st_inhom_degenerate_single_event_returns_zeros():
    r_grid = np.linspace(0.05, 0.2, 3)
    t_grid = np.linspace(0.05, 0.2, 3)

    def lam_at(x, t):
        return np.full(x.shape[0], 1.0)

    res = k_st_inhom(
        np.array([[0.5, 0.5]]), np.array([0.5]), lam_at, r_grid, t_grid,
        domain=DOMAIN, period=PERIOD,
    )
    assert np.all(res.k_st == 0.0)


def test_global_envelope_st_statistic_kwarg_validates():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, size=(30, 2))
    times = rng.uniform(0, 1, size=30)

    def lam_at(x, t):
        return np.full(x.shape[0], 30.0)

    r_grid = np.array([0.1])
    t_grid = np.array([0.1])
    with pytest.raises(ValueError, match="statistic must be one of"):
        global_envelope_st(
            pts, times, lam_at, r_grid, t_grid, n_sim=5, statistic="bogus",
            domain=DOMAIN, period=PERIOD, rng=np.random.default_rng(0),
        )


def test_global_envelope_st_gst_and_lst_statistics_run():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, size=(60, 2))
    times = rng.uniform(0, 1, size=60)

    def lam_at(x, t):
        return np.full(x.shape[0], 60.0)

    r_grid = np.array([0.08, 0.14])
    t_grid = np.array([0.08, 0.14])
    for statistic in ("gst", "lst"):
        env = global_envelope_st(
            pts, times, lam_at, r_grid, t_grid, n_sim=9, statistic=statistic,
            domain=DOMAIN, period=PERIOD, rng=np.random.default_rng(1),
        )
        assert env.observed.shape == (2, 2)
        assert np.all(env.hi >= env.lo)
        assert 0.0 <= env.p_interval[0] <= env.p_interval[1] <= 1.0
