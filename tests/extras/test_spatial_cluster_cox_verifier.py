"""Verifier-side independent probes for cluster Cox processes.

Written by the verifier agent (NOT the builder).  These tests close the
loop between the closed-form pair correlations in
``nstat.extras.spatial.cluster_cox`` and the corresponding simulators by
running an independent simulation-vs-theory check:

1. Sample many realizations from the simulator.
2. Compute the empirical SOIRS pair correlation g_hat(r) on each
   realization with ``edge_correction="border"`` (NaN-safe averaging).
3. Average across realizations at fixed lags.
4. Compare to the closed-form g(r) within an architect-set tolerance.

If both halves of the contract (simulator + closed form) are wrong in the
*same* way, this probe would not catch it; but a single-sided bug in
either implementation breaks the identity.  These are intentionally
duplicated outside the builder's test file because the builder's own
tests can hide a bug that the same author would not catch.

Synthetic data only (np.random.default_rng); no patient data.
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
from nstat.extras.spatial.spatial_gof import pair_correlation


WINDOW = (0.0, 0.0, 1.0, 1.0)
DOMAIN = ((0.0, 1.0), (0.0, 1.0))
AREA = 1.0
SEED = 20260617


def _empirical_g_at_lags(
    points: np.ndarray, r_eval: np.ndarray
) -> np.ndarray:
    """Compute empirical g_hat at the requested lags via the border estimator."""
    n = points.shape[0]
    if n < 2:
        return np.full(r_eval.shape, np.nan)
    lam = float(n) / AREA
    lam_arr = np.full(n, lam, dtype=float)
    return np.asarray(
        pair_correlation(
            points,
            lam_arr,
            r_eval,
            domain=DOMAIN,
            edge_correction="border",
        ),
        dtype=float,
    )


def _mean_g_over_realizations(
    simulate_fn, r_eval: np.ndarray, n_realizations: int, seed_base: int
) -> np.ndarray:
    """Sample `n_realizations` patterns, return the per-lag NaN-safe mean."""
    g_stack = np.full((n_realizations, r_eval.size), np.nan, dtype=float)
    for k in range(n_realizations):
        rng = np.random.default_rng(seed_base + k)
        pts = simulate_fn(rng)
        g_stack[k] = _empirical_g_at_lags(pts, r_eval)
    # NaN-safe mean across realizations at each lag.
    with np.errstate(invalid="ignore"):
        return np.nanmean(g_stack, axis=0)


# ----------------------------------------------------------------------
# 1. Thomas — simulator vs closed form
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_thomas_pair_correlation_against_independent_derivation():
    r"""Architect identity: g_Thomas(r) = 1 + exp(-r^2/(4 sigma^2)) /
    (4 pi sigma^2 lambda_p).

    Probe: 100 realizations of simulate_thomas with sigma=0.05, lam_p=50,
    mu=10 on [0,1]^2.  Empirical g_hat(r) at r in {0.02, 0.04, 0.06} must
    be within +/- 15% of the closed form on average.  The 15% tolerance
    accommodates SOIRS kernel-bandwidth bias + 100-realization Monte
    Carlo standard error at this sample size; tighter tolerances would
    require an order-of-magnitude more realizations.
    """
    sigma_true, lam_p_true, mu_true = 0.05, 50.0, 10.0
    r_lags = np.array([0.02, 0.04, 0.06])

    g_theory = thomas_pair_correlation(
        r_lags, sigma_true, lam_p_true, mu_true
    )

    def _sim(rng):
        return simulate_thomas(
            lam_p_true, mu_true, sigma_true, WINDOW, rng=rng
        )

    g_mean = _mean_g_over_realizations(
        _sim, r_lags, n_realizations=100, seed_base=SEED
    )

    assert np.all(np.isfinite(g_mean)), (
        f"empirical g_hat had NaN at requested lags: {g_mean}"
    )
    rel_err = np.abs(g_mean - g_theory) / np.abs(g_theory)
    assert np.all(rel_err < 0.15), (
        f"closed-form vs empirical disagree beyond 15%: "
        f"theory={g_theory}, empirical={g_mean}, rel_err={rel_err}"
    )


# ----------------------------------------------------------------------
# 2. Matern cluster — simulator vs closed form (including the r>2R cliff)
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_matern_cluster_pair_correlation_against_independent_derivation():
    r"""Architect identity: g_Matern(r) = 1 + h(r;R)/(pi R^2 lam_p) for
    r <= 2R, g(r) = 1 for r > 2R.

    Probe: 100 realizations of simulate_matern_cluster with R=0.05,
    lam_p=50, mu=10.  Lags r in {0.02, 0.05, 0.08} are <= 2R = 0.10 and
    must be within +/- 15% of the closed form.  Lag r = 0.12 is > 2R and
    must collapse to 1 within +/- 15% (i.e. the discontinuity at 2R is
    honored end-to-end).
    """
    R_true, lam_p_true, mu_true = 0.05, 50.0, 10.0
    r_in_support = np.array([0.02, 0.05, 0.08])
    r_above_support = np.array([0.12])
    r_lags = np.concatenate([r_in_support, r_above_support])

    g_theory = matern_cluster_pair_correlation(
        r_lags, R_true, lam_p_true, mu_true
    )

    # Sanity-check the closed form against the brief: g(0.12) must be 1.
    assert g_theory[-1] == pytest.approx(1.0, abs=1e-12)

    def _sim(rng):
        return simulate_matern_cluster(
            lam_p_true, mu_true, R_true, WINDOW, rng=rng
        )

    g_mean = _mean_g_over_realizations(
        _sim, r_lags, n_realizations=100, seed_base=SEED
    )

    assert np.all(np.isfinite(g_mean)), (
        f"empirical g_hat had NaN at requested lags: {g_mean}"
    )
    rel_err = np.abs(g_mean - g_theory) / np.abs(g_theory)
    assert np.all(rel_err < 0.15), (
        "closed-form vs empirical disagree beyond 15%: "
        f"theory={g_theory}, empirical={g_mean}, rel_err={rel_err}"
    )
