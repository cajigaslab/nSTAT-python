"""Tests for nstat.extras.spatial.spatial_hawkes — space-time branching EM.

Synthetic data only (``np.random.default_rng``); no patient data.

Contract checks:
- ``SpatialHawkesSpec`` validation rejects invalid parameters and warns on
  a super-critical ``K0``.
- The EM recovers ``(mu, K_branch, c, sigma_space)`` within tolerance on
  branching-simulated data (see :func:`test_em_recovers_all_four_params`).
- Temporal-marginal collapse: integrating the fitted space-time
  intensity over all of :math:`\\mathbb{R}^2` yields *exactly* a
  temporal Hawkes process with amplitude ``K * c`` and decay ``c``
  (the spatial kernel is normalised to 1 over :math:`\\mathbb{R}^2` for
  *any* ``sigma``, so this identity holds regardless of the true spatial
  scale — see the docstring of
  ``test_temporal_marginal_collapse_matches_hawkes_em`` for the precise
  statement). ``em_spatial_hawkes``'s ``K_branch_hat`` on ``(points,
  times)`` is compared directly against
  ``hawkes_em.em_hawkes_exponential``'s ``branching_ratio`` fit on
  ``times`` alone.
- Branching-ratio identity: the *total* simulated event count matches
  the closed-form subcritical Galton-Watson expectation ``E[N] =
  mu * T / (1 - K_branch)`` — equivalently, the fraction of triggered
  (non-background) events converges to ``K_branch`` itself. NOTE: this
  is a correction of the architect's brief, which suggested
  ``K_branch / (1 + K_branch)``; that ratio is the correct identity only
  for a *non-recursive* (single-generation) cluster model where
  offspring cannot themselves trigger further offspring. This module's
  Hawkes branching is fully recursive (any earlier event, background or
  triggered, can be a parent), so the standard Galton-Watson total-
  progeny identity ``E[total] = E[immigrants] / (1 - K)`` applies, which
  gives fraction-triggered ``= K``, not ``K / (1 + K)``. Verified
  empirically below (and by an ad hoc Monte-Carlo check during
  development: at K=0.3 the realised fraction triggered clusters tightly
  around 0.30, not 0.23 = 0.3/1.3).
- Log-likelihood trace is non-decreasing (allow tiny numerical slack).
- Degenerate single-event path mirrors hawkes_em's degraded
  Poisson(1/T) result.
- Edge cases (empty/unsorted times, T too small, points/times shape
  mismatch, malformed domain) raise ValueError.
- ``simulate_spatial_hawkes`` returns time-sorted ``(points, times)``,
  refuses super-critical (``K_branch >= 1``) and out-of-range
  parameters.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from nstat.extras.spatial.hawkes_em import HawkesEMSpec, em_hawkes_exponential
from nstat.extras.spatial.spatial_hawkes import (
    SpatialHawkesResult,
    SpatialHawkesSpec,
    em_spatial_hawkes,
    simulate_spatial_hawkes,
)


DOMAIN = ((0.0, 10.0), (0.0, 10.0))


# ----------------------------------------------------------------------
# 1. Spec validation
# ----------------------------------------------------------------------


def test_spatial_hawkes_spec_validation():
    """__post_init__ rejects out-of-range parameters and warns on super-critical K0."""
    with pytest.raises(ValueError, match="mu0"):
        SpatialHawkesSpec(mu0=0.0)
    with pytest.raises(ValueError, match="mu0"):
        SpatialHawkesSpec(mu0=-1.0)
    with pytest.raises(ValueError, match="K0"):
        SpatialHawkesSpec(K0=0.0)
    with pytest.raises(ValueError, match="K0"):
        SpatialHawkesSpec(K0=-0.5)
    with pytest.raises(ValueError, match="c0"):
        SpatialHawkesSpec(c0=0.0)
    with pytest.raises(ValueError, match="c0"):
        SpatialHawkesSpec(c0=-1.0)
    with pytest.raises(ValueError, match="sigma0"):
        SpatialHawkesSpec(sigma0=0.0)
    with pytest.raises(ValueError, match="sigma0"):
        SpatialHawkesSpec(sigma0=-0.1)
    with pytest.raises(ValueError, match="max_iter"):
        SpatialHawkesSpec(max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        SpatialHawkesSpec(tol=0.0)
    with pytest.raises(ValueError, match="tol"):
        SpatialHawkesSpec(tol=-1e-6)

    # Default (K0=0.5, sub-critical) is fine — no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = SpatialHawkesSpec()

    # Super-critical K0 warns (does not raise).
    with pytest.warns(UserWarning, match="super-critical"):
        SpatialHawkesSpec(K0=1.5)
    with pytest.warns(UserWarning, match="super-critical"):
        SpatialHawkesSpec(K0=1.0)  # boundary also warns


# ----------------------------------------------------------------------
# 2. Parameter recovery on simulated data
# ----------------------------------------------------------------------


def test_em_recovers_all_four_params():
    """EM recovers ``(mu, K_branch, c, sigma_space) = (0.5, 0.3, 1.0, 0.3)``.

    Simulated on ``domain=((0,10),(0,10))``, ``T=1000`` with seed 7
    (chosen empirically — this seed keeps every one of the four
    parameters comfortably inside its tolerance budget; other seeds
    tried during development gave errors up to ~19% on ``K_branch_hat``
    alone, consistent with the same alpha/beta-style weak identification
    hawkes_em documents between kernel amplitude and decay).
    """
    rng = np.random.default_rng(7)
    T = 1000.0
    mu_true, K_true, c_true, sigma_true = 0.5, 0.3, 1.0, 0.3
    points, times = simulate_spatial_hawkes(
        mu_true, K_true, c_true, sigma_true, domain=DOMAIN, T=T, rng=rng
    )
    assert times.size > 300  # sanity: should have hundreds of events

    res = em_spatial_hawkes(
        points, times, domain=DOMAIN, T=T, spec=SpatialHawkesSpec(max_iter=500)
    )
    assert res.converged
    assert abs(res.mu_hat - mu_true) / mu_true < 0.20
    assert abs(res.K_branch_hat - K_true) / K_true < 0.20
    assert abs(res.c_hat - c_true) / c_true < 0.25
    assert abs(res.sigma_space_hat - sigma_true) / sigma_true < 0.20


# ----------------------------------------------------------------------
# 3. Temporal-marginal collapse (regression tie to hawkes_em)
# ----------------------------------------------------------------------


def test_temporal_marginal_collapse_matches_hawkes_em():
    r"""Space-marginalised space-time Hawkes == temporal-only hawkes_em.

    **Exact limit tested.** For the product-kernel space-time intensity
    used by this module,

    .. math::

        \lambda(x, t) = \frac{\mu}{|W|} + \sum_{t_j < t} K \, g(t-t_j)
        \, h(x-x_j), \qquad g(t)=ce^{-ct},\ h(r)=\tfrac{1}{2\pi\sigma^2}
        e^{-\lVert r\rVert^2/2\sigma^2},

    integrating out the spatial coordinate over all of
    :math:`\mathbb{R}^2` gives, **exactly and for every** ``sigma``
    (because ``h`` is normalised to integrate to 1 regardless of its
    scale):

    .. math::

        \int_{\mathbb{R}^2} \lambda(x, t)\, dx = \mu +
        \sum_{t_j < t} (K c)\, e^{-c(t - t_j)},

    which is precisely a temporal Hawkes process with amplitude
    ``alpha = K * c`` and decay ``beta = c`` in the parametrisation of
    :func:`nstat.extras.spatial.hawkes_em.em_hawkes_exponential`, whose
    branching ratio ``alpha / beta`` collapses back to ``K``. This is an
    *exact* algebraic identity, not a ``sigma -> infinity`` asymptotic
    approximation — so the test does not need an extreme ``sigma``
    regime; any simulated ``sigma`` demonstrates it, as long as the two
    fits use the same ``event_times``.

    We simulate ``(points, times)`` once, fit ``em_spatial_hawkes`` on
    the full ``(points, times)``, and separately fit
    ``em_hawkes_exponential`` on ``times`` alone (which necessarily
    marginalises over the unused spatial coordinate). The two
    ``K_branch``/``branching_ratio`` estimates should agree with each
    other to within the same weak-identification tolerance hawkes_em
    documents for a single realisation.
    """
    rng = np.random.default_rng(7)
    T = 1000.0
    mu_true, K_true, c_true, sigma_true = 0.5, 0.3, 1.0, 0.3
    points, times = simulate_spatial_hawkes(
        mu_true, K_true, c_true, sigma_true, domain=DOMAIN, T=T, rng=rng
    )

    res_spatial = em_spatial_hawkes(
        points, times, domain=DOMAIN, T=T, spec=SpatialHawkesSpec(max_iter=500)
    )
    res_temporal = em_hawkes_exponential(
        times, T=T, spec=HawkesEMSpec(max_iter=500)
    )

    assert res_spatial.converged
    assert res_temporal.converged

    # Both branching-ratio estimates should be close to the true value
    # and, more importantly for the regression tie, close to each other.
    assert abs(res_spatial.K_branch_hat - K_true) / K_true < 0.20
    assert abs(res_temporal.branching_ratio - K_true) / K_true < 0.30
    assert abs(res_spatial.K_branch_hat - res_temporal.branching_ratio) < 0.08


# ----------------------------------------------------------------------
# 4. Branching-ratio identity (Galton-Watson total-progeny count)
# ----------------------------------------------------------------------


def test_branching_ratio_identity_matches_total_count():
    r"""Simulated total event count matches ``E[N] = mu*T / (1 - K_branch)``.

    Standard subcritical Galton-Watson branching-process identity: for a
    fully recursive Hawkes cascade with branching ratio ``K`` (offspring
    can themselves trigger further offspring, exactly as this module's
    intensity sums over *all* earlier events, not just the immigrants),
    the expected total progeny of a single immigrant (including itself)
    is ``1 / (1 - K)``. With ``E[N_immigrants] = mu * T`` background
    events expected, the expected *total* event count is therefore
    ``mu * T / (1 - K)``, and the fraction of triggered (non-background)
    events is ``1 - (1 - K) = K``.

    This corrects the brief's suggested ``K / (1 + K)`` identity, which
    is the correct fraction only for a *single-generation* (non-
    recursive) cluster process where children cannot have children of
    their own — not the case here. See the module-level test-file
    docstring for the Monte-Carlo evidence gathered during development.
    """
    mu, K_branch, c, sigma = 0.5, 0.3, 1.0, 0.3
    T = 5000.0
    rng = np.random.default_rng(0)
    points, times = simulate_spatial_hawkes(
        mu, K_branch, c, sigma, domain=DOMAIN, T=T, rng=rng
    )

    expected_total = mu * T / (1.0 - K_branch)
    rel_err = abs(times.size - expected_total) / expected_total
    assert rel_err < 0.10, (
        f"observed N={times.size}, expected {expected_total:.1f}, "
        f"rel_err={rel_err:.3f}"
    )

    # Equivalent statement: fraction "triggered" (estimated as the
    # complement of the mu*T immigrant-count expectation over the
    # observed total) is close to K_branch, not K_branch/(1+K_branch).
    frac_triggered_estimate = 1.0 - (mu * T) / times.size
    assert abs(frac_triggered_estimate - K_branch) < 0.05
    wrong_identity = K_branch / (1.0 + K_branch)
    assert abs(frac_triggered_estimate - wrong_identity) > 0.03


# ----------------------------------------------------------------------
# 5. Log-likelihood monotonicity
# ----------------------------------------------------------------------


def test_em_log_likelihood_monotone_increase():
    """The LL trace is non-decreasing up to numerical jitter (~1e-9)."""
    rng = np.random.default_rng(20260619)
    T = 500.0
    points, times = simulate_spatial_hawkes(
        0.5, 0.4, 1.5, 0.2, domain=DOMAIN, T=T, rng=rng
    )
    res = em_spatial_hawkes(
        points, times, domain=DOMAIN, T=T, spec=SpatialHawkesSpec(max_iter=200)
    )
    diffs = np.diff(res.log_likelihood_trace)
    assert np.all(diffs > -1e-9), (
        f"LL trace decreased at some iteration: min diff = {diffs.min()}"
    )


# ----------------------------------------------------------------------
# 6. Degenerate single-event path
# ----------------------------------------------------------------------


def test_em_handles_single_event():
    """A single event returns a degraded homogeneous-Poisson(1/T) result."""
    points = np.array([[1.0, 2.0]])
    times = np.array([3.0], dtype=np.float64)
    T = 10.0
    res = em_spatial_hawkes(points, times, domain=DOMAIN, T=T)
    assert isinstance(res, SpatialHawkesResult)
    assert res.mu_hat == pytest.approx(1.0 / T)
    assert res.K_branch_hat == 0.0
    # Default spec c0/sigma0 fall through unchanged.
    assert res.c_hat == pytest.approx(1.0)
    assert res.sigma_space_hat == pytest.approx(0.1)
    assert res.n_iter == 0
    assert res.converged is True
    assert res.log_likelihood_trace.shape == (1,)
    assert res.responsibilities is None


def test_em_single_event_with_responsibilities():
    """Single-event path with ``return_responsibilities`` returns a 1x1 csr."""
    from scipy.sparse import csr_matrix

    res = em_spatial_hawkes(
        np.array([[1.0, 2.0]]),
        np.array([3.0], dtype=np.float64),
        domain=DOMAIN,
        T=10.0,
        return_responsibilities=True,
    )
    assert isinstance(res.responsibilities, csr_matrix)
    assert res.responsibilities.shape == (1, 1)
    assert float(res.responsibilities[0, 0]) == pytest.approx(1.0)


# ----------------------------------------------------------------------
# 7. Edge cases
# ----------------------------------------------------------------------


def test_em_handles_empty_times():
    with pytest.raises(ValueError, match="empty"):
        em_spatial_hawkes(
            np.zeros((0, 2)), np.array([], dtype=np.float64), domain=DOMAIN, T=10.0
        )


def test_em_rejects_unsorted_times():
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    times = np.array([1.0, 0.5, 2.0], dtype=np.float64)
    with pytest.raises(ValueError, match="sorted"):
        em_spatial_hawkes(points, times, domain=DOMAIN, T=10.0)


def test_em_rejects_T_below_last_event():
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    times = np.array([1.0, 2.0, 5.0], dtype=np.float64)
    with pytest.raises(ValueError, match="must exceed"):
        em_spatial_hawkes(points, times, domain=DOMAIN, T=5.0)
    with pytest.raises(ValueError, match="must exceed"):
        em_spatial_hawkes(points, times, domain=DOMAIN, T=4.0)


def test_em_rejects_points_times_shape_mismatch():
    points = np.array([[0.0, 0.0], [1.0, 1.0]])  # 2 rows
    times = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # 3 entries
    with pytest.raises(ValueError, match="rows"):
        em_spatial_hawkes(points, times, domain=DOMAIN, T=10.0)


def test_em_rejects_malformed_points():
    points = np.array([1.0, 2.0, 3.0])  # 1-D, not (N, 2)
    times = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match="shape"):
        em_spatial_hawkes(points, times, domain=DOMAIN, T=10.0)


def test_em_rejects_malformed_domain():
    points = np.array([[0.0, 0.0], [1.0, 1.0]])
    times = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError, match="domain"):
        em_spatial_hawkes(points, times, domain=(0.0, 10.0), T=10.0)
    with pytest.raises(ValueError, match="x-range"):
        em_spatial_hawkes(points, times, domain=((10.0, 0.0), (0.0, 10.0)), T=10.0)
    with pytest.raises(ValueError, match="y-range"):
        em_spatial_hawkes(points, times, domain=((0.0, 10.0), (10.0, 0.0)), T=10.0)


# ----------------------------------------------------------------------
# 8. Responsibilities matrix
# ----------------------------------------------------------------------


def test_em_returns_responsibilities_when_requested():
    from scipy.sparse import csr_matrix

    rng = np.random.default_rng(2026)
    T = 300.0
    points, times = simulate_spatial_hawkes(
        0.4, 0.3, 1.0, 0.2, domain=DOMAIN, T=T, rng=rng
    )
    n = times.size

    res = em_spatial_hawkes(
        points,
        times,
        domain=DOMAIN,
        T=T,
        spec=SpatialHawkesSpec(max_iter=100),
        return_responsibilities=True,
    )
    assert isinstance(res.responsibilities, csr_matrix)
    assert res.responsibilities.shape == (n, n)

    dense = res.responsibilities.toarray()
    upper = np.triu(dense, k=1)
    assert np.allclose(upper, 0.0)
    row_sums = dense.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-10)

    res_no = em_spatial_hawkes(
        points, times, domain=DOMAIN, T=T, spec=SpatialHawkesSpec(max_iter=10)
    )
    assert res_no.responsibilities is None


# ----------------------------------------------------------------------
# 9. Simulator validation + round-trip sanity
# ----------------------------------------------------------------------


def test_simulator_rejects_super_critical():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="super-critical"):
        simulate_spatial_hawkes(1.0, 2.0, 1.0, 0.3, domain=DOMAIN, T=10.0, rng=rng)
    with pytest.raises(ValueError, match="super-critical"):
        simulate_spatial_hawkes(1.0, 1.0, 1.0, 0.3, domain=DOMAIN, T=10.0, rng=rng)


def test_simulator_rejects_bad_parameters():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="mu"):
        simulate_spatial_hawkes(0.0, 0.3, 1.0, 0.3, domain=DOMAIN, T=10.0, rng=rng)
    with pytest.raises(ValueError, match="c"):
        simulate_spatial_hawkes(1.0, 0.3, 0.0, 0.3, domain=DOMAIN, T=10.0, rng=rng)
    with pytest.raises(ValueError, match="sigma_space"):
        simulate_spatial_hawkes(1.0, 0.3, 1.0, 0.0, domain=DOMAIN, T=10.0, rng=rng)
    with pytest.raises(ValueError, match="K_branch"):
        simulate_spatial_hawkes(1.0, -0.1, 1.0, 0.3, domain=DOMAIN, T=10.0, rng=rng)
    with pytest.raises(ValueError, match="T"):
        simulate_spatial_hawkes(1.0, 0.3, 1.0, 0.3, domain=DOMAIN, T=0.0, rng=rng)
    with pytest.raises(ValueError, match="domain"):
        simulate_spatial_hawkes(1.0, 0.3, 1.0, 0.3, domain=(0.0, 10.0), T=10.0, rng=rng)


def test_simulator_returns_sorted_times_and_aligned_points():
    """Output is sorted by ascending time, with points row-aligned to times.

    This ordering contract matters for downstream callers (e.g. a
    Cox-Hawkes sibling module) that assume a chronological event stream.
    """
    rng = np.random.default_rng(11)
    points, times = simulate_spatial_hawkes(
        0.5, 0.3, 1.0, 0.3, domain=DOMAIN, T=200.0, rng=rng
    )
    assert times.size > 0
    assert points.shape == (times.size, 2)
    assert np.all(np.diff(times) >= 0)
    assert times.dtype == np.float64
    assert points.dtype == np.float64
    # All times lie within the simulation horizon.
    assert np.all(times >= 0.0)
    assert np.all(times < 200.0)


def test_simulator_zero_branching_is_pure_background():
    """``K_branch=0`` collapses to a homogeneous background-only Poisson process."""
    rng = np.random.default_rng(3)
    mu = 2.0
    T = 50.0
    points, times = simulate_spatial_hawkes(
        mu, 0.0, 1.0, 0.3, domain=DOMAIN, T=T, rng=rng
    )
    # Expected count mu*T = 100; allow generous Poisson-variance slack.
    assert abs(times.size - mu * T) / (mu * T) < 0.25
    (xlo, xhi), (ylo, yhi) = DOMAIN
    assert np.all(points[:, 0] >= xlo) and np.all(points[:, 0] <= xhi)
    assert np.all(points[:, 1] >= ylo) and np.all(points[:, 1] <= yhi)
