"""Tests for nstat.extras.spatial.cluster_cox — cluster Cox processes.

Synthetic data only (np.random.default_rng); no patient data.

Contract checks:
- Parameter classes reject non-positive parameters.
- Closed-form Thomas / Matérn-cluster ``g(r)`` match analytic identities.
- Simulators produce points strictly inside the window.
- Empirical mean intensity tracks ``lambda_parent * mu_offspring``.
- ``NeymanScottCox`` with ``pad=0`` warns; ``return_parents=True`` returns
  the parent locations.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from nstat.extras.spatial.cluster_cox import (
    MaternClusterProcess,
    NeymanScottCox,
    ThomasProcess,
    matern_cluster_pair_correlation,
    simulate_matern_cluster,
    simulate_neyman_scott,
    simulate_thomas,
    thomas_pair_correlation,
)


WINDOW = (0.0, 0.0, 1.0, 1.0)
SEED = 20260616


# ----------------------------------------------------------------------
# 1. Parameter validation
# ----------------------------------------------------------------------


def test_thomas_process_rejects_non_positive_parameters():
    with pytest.raises(ValueError, match="intensity_parent"):
        ThomasProcess(0.0, 1.0, 0.05)
    with pytest.raises(ValueError, match="mu_offspring"):
        ThomasProcess(10.0, -1.0, 0.05)
    with pytest.raises(ValueError, match="sigma"):
        ThomasProcess(10.0, 1.0, 0.0)
    # Valid construction does not raise.
    _ = ThomasProcess(10.0, 1.0, 0.05)


def test_matern_cluster_process_rejects_non_positive_parameters():
    with pytest.raises(ValueError, match="intensity_parent"):
        MaternClusterProcess(-1.0, 1.0, 0.05)
    with pytest.raises(ValueError, match="mu_offspring"):
        MaternClusterProcess(10.0, 0.0, 0.05)
    with pytest.raises(ValueError, match="radius"):
        MaternClusterProcess(10.0, 1.0, 0.0)


def test_neyman_scott_rejects_non_callable_kernel_and_negative_pad():
    with pytest.raises(ValueError, match="offspring_kernel"):
        NeymanScottCox(
            10.0, 1.0, offspring_kernel="not-callable", pad=0.1  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="pad"):
        NeymanScottCox(
            10.0, 1.0, offspring_kernel=lambda n, r: np.zeros((n, 2)), pad=-1.0
        )


# ----------------------------------------------------------------------
# 2. Closed-form pair correlation identities
# ----------------------------------------------------------------------


def test_thomas_pair_correlation_matches_analytic_identities():
    sigma, lam_p = 0.03, 25.0
    r = np.array([0.0, 0.01, 0.05, 0.5, 2.0])
    g = thomas_pair_correlation(r, sigma, lam_p, 1.0)
    # g(0) = 1 + 1/(4 pi sigma^2 lambda_p)
    expected_at_zero = 1.0 + 1.0 / (4.0 * np.pi * sigma**2 * lam_p)
    assert g[0] == pytest.approx(expected_at_zero, rel=1e-12)
    # g monotone decreasing in r, approaches 1 as r -> infinity.
    assert np.all(np.diff(g) <= 0.0)
    assert g[-1] == pytest.approx(1.0, abs=1e-12)
    # Validation: non-positive sigma / lambda raises ValueError.
    with pytest.raises(ValueError, match="sigma"):
        thomas_pair_correlation(np.array([0.1]), -1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="intensity_parent"):
        thomas_pair_correlation(np.array([0.1]), 0.05, 0.0, 1.0)


def test_matern_cluster_pair_correlation_supports_at_2R():
    R, lam_p = 0.05, 25.0
    # g(0) = 1 + h(0;R) / (pi R^2 lam_p) with h(0;R) = 1.
    # g(2R) = 1 exactly; g(r > 2R) = 1.
    r = np.array([0.0, 0.5 * R, R, 1.5 * R, 2.0 * R, 3.0 * R])
    g = matern_cluster_pair_correlation(r, R, lam_p, 1.0)
    assert g[0] == pytest.approx(1.0 + 1.0 / (np.pi * R**2 * lam_p), rel=1e-12)
    # Continuity at r = 2R from below: limit -> 1.
    assert g[-2] == pytest.approx(1.0, abs=1e-10)
    assert g[-1] == pytest.approx(1.0, abs=1e-12)
    # Monotone-decreasing on the support.
    in_support = g[:-1]
    assert np.all(np.diff(in_support) <= 1e-12)


# ----------------------------------------------------------------------
# 3. Simulators — points inside window, parametric mean intensity
# ----------------------------------------------------------------------


def test_simulate_thomas_keeps_points_in_window_and_recovers_mean_intensity():
    rng = np.random.default_rng(SEED)
    lam_p, mu, sigma = 30.0, 15.0, 0.03
    # Average n over several seeds — single-seed Poisson fluctuations are big.
    ns = []
    for seed in range(SEED, SEED + 10):
        r = np.random.default_rng(seed)
        pts = simulate_thomas(lam_p, mu, sigma, WINDOW, rng=r)
        assert pts.dtype == np.float64
        assert pts.shape[1] == 2
        # Strict containment.
        assert (pts[:, 0] >= 0.0).all() and (pts[:, 0] <= 1.0).all()
        assert (pts[:, 1] >= 0.0).all() and (pts[:, 1] <= 1.0).all()
        ns.append(pts.shape[0])
    # Expected mean count = lam_p * mu * area = 450.  Sample mean should be
    # within 30% of expected on 10 seeds.
    assert abs(np.mean(ns) - lam_p * mu) / (lam_p * mu) < 0.3
    # And one explicit seed mention to silence "unused" linters.
    _ = rng


def test_simulate_matern_cluster_keeps_points_in_window():
    rng = np.random.default_rng(SEED)
    pts = simulate_matern_cluster(30.0, 15.0, 0.05, WINDOW, rng=rng)
    assert pts.dtype == np.float64
    assert pts.shape[1] == 2
    assert (pts[:, 0] >= 0.0).all() and (pts[:, 0] <= 1.0).all()
    assert (pts[:, 1] >= 0.0).all() and (pts[:, 1] <= 1.0).all()
    assert pts.shape[0] > 100  # Non-trivial pattern.


def test_simulate_thomas_rejects_invalid_window():
    rng = np.random.default_rng(SEED)
    with pytest.raises(ValueError, match="window"):
        simulate_thomas(10.0, 5.0, 0.05, (0.0, 0.0, -1.0, 1.0), rng=rng)
    with pytest.raises(ValueError, match="window"):
        simulate_thomas(10.0, 5.0, 0.05, (0.0, 0.0, 1.0), rng=rng)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# 4. Neyman-Scott generic dispatcher
# ----------------------------------------------------------------------


def test_neyman_scott_pad_zero_warns_and_return_parents_works():
    rng = np.random.default_rng(SEED)

    def gauss(n, r):
        return r.normal(0.0, 0.03, size=(n, 2))

    proc_padded = NeymanScottCox(
        intensity_parent=30.0, mu_offspring=10.0, offspring_kernel=gauss, pad=0.1
    )
    # With explicit pad, return_parents=True yields (offspring, parents).
    offspring, parents = simulate_neyman_scott(
        proc_padded, WINDOW, rng=rng, return_parents=True
    )
    assert offspring.dtype == np.float64
    assert parents.shape[1] == 2
    assert offspring.shape[0] > 0
    # Offspring all inside the window.
    assert (offspring[:, 0] >= 0.0).all() and (offspring[:, 0] <= 1.0).all()

    # pad=0 must warn.
    proc_unpadded = NeymanScottCox(
        intensity_parent=30.0, mu_offspring=10.0, offspring_kernel=gauss, pad=0.0
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = simulate_neyman_scott(proc_unpadded, WINDOW, rng=rng)
        assert any("pad" in str(rec.message).lower() for rec in w)


def test_neyman_scott_dispatch_matches_thomas_when_kernel_is_gaussian():
    """A NeymanScottCox configured as a Gaussian kernel produces a pattern
    whose empirical clustering profile (count vs distance) is statistically
    indistinguishable from simulate_thomas; we only check both produce
    non-empty patterns from the same generator state."""
    lam_p, mu, sigma = 30.0, 12.0, 0.03

    def gauss(n, r):
        return r.normal(0.0, sigma, size=(n, 2))

    r1 = np.random.default_rng(SEED)
    pts_thomas = simulate_thomas(lam_p, mu, sigma, WINDOW, rng=r1)

    r2 = np.random.default_rng(SEED)
    proc = NeymanScottCox(
        intensity_parent=lam_p,
        mu_offspring=mu,
        offspring_kernel=gauss,
        pad=3.0 * sigma,
    )
    pts_ns = simulate_neyman_scott(proc, WINDOW, rng=r2)
    # Same seed, same RNG-call order ⇒ identical patterns.  This pins the
    # dispatcher to the simulator's exact RNG protocol; a future refactor
    # that changes draw order will break here loudly.
    assert pts_thomas.shape == pts_ns.shape
    np.testing.assert_allclose(pts_thomas, pts_ns)
