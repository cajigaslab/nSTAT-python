"""Correctness tests for the time-rescaling KS-test oracle.

The oracle in ``tests/parity/_third_party/time_rescale_oracle.py`` is a
clean-room implementation of the Brown / Barbieri / Ventura / Kass /
Frank (Neural Computation 2002) time-rescaling theorem KS test, used
as a second-opinion reference for :meth:`nstat.FitResult.computeKSStats`.

These tests validate the oracle stands on its own — the well-specified
case rejects rarely, the mis-specified case rejects always — so it can
be trusted as a cross-validation reference.  A separate PR will wire
it into a full nstat.fit_result.computeKSStats comparison harness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add the _third_party sub-tree to sys.path so the oracle is importable
# without polluting the public nstat namespace.
_REPO_ROOT = Path(__file__).resolve().parent
_THIRD_PARTY = _REPO_ROOT / "parity" / "_third_party"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

from time_rescale_oracle import (  # noqa: E402
    TimeRescaleResult,
    time_rescaling_ks_test,
)


# ----------------------------------------------------------------------
# Mathematical correctness
# ----------------------------------------------------------------------


def _simulate_homogeneous_poisson(
    rate_hz: float, duration_s: float, dt: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Bernoulli thinning at small dt approximates a Poisson process."""
    n_bins = int(round(duration_s / dt))
    p_per_bin = rate_hz * dt
    spike_indicator = (rng.uniform(size=n_bins) < p_per_bin).astype(int)
    intensity = np.full(n_bins, rate_hz, dtype=float)
    return intensity, spike_indicator


def test_oracle_returns_dataclass_with_expected_fields() -> None:
    """Basic structural assertion on the return type."""
    rng = np.random.default_rng(0)
    intensity, spikes = _simulate_homogeneous_poisson(50.0, 5.0, 1e-3, rng)
    result = time_rescaling_ks_test(intensity, spikes, dt=1e-3)
    assert isinstance(result, TimeRescaleResult)
    assert result.rescaled_isis.ndim == 1
    assert result.uniform_z.ndim == 1
    assert result.rescaled_isis.shape == result.uniform_z.shape
    assert 0.0 <= result.ks_stat <= 1.0
    assert 0.0 <= result.ks_pvalue <= 1.0


def test_well_specified_model_does_not_reject_under_repeated_draws() -> None:
    """When the intensity matches the data-generating process, the KS
    test should produce uniformly-distributed p-values across many runs
    (well-known consequence of the time-rescaling theorem).

    On 20 independent simulations at α=0.05, expect ~1 rejection on
    average; we allow up to 6 to absorb seed variance (binomial(20,
    0.05) > 6 has p ≈ 0.0003 — well below test-flake threshold).
    """
    rate_hz, duration_s, dt = 50.0, 5.0, 1e-3
    rng = np.random.default_rng(42)
    n_rejections = 0
    n_trials = 20
    for _ in range(n_trials):
        intensity, spikes = _simulate_homogeneous_poisson(rate_hz, duration_s, dt, rng)
        result = time_rescaling_ks_test(intensity, spikes, dt=dt)
        if result.ks_pvalue < 0.05:
            n_rejections += 1
    assert n_rejections <= 6, (
        f"Well-specified model rejected {n_rejections}/{n_trials} times at "
        f"α=0.05.  Expected ~1; >6 is statistically suspicious."
    )


def test_mis_specified_intensity_rejects_strongly() -> None:
    """When the intensity is wrong by a factor of 10, the KS test
    should reject decisively (p < 0.01 effectively always)."""
    rate_hz, duration_s, dt = 50.0, 5.0, 1e-3
    rng = np.random.default_rng(7)
    intensity, spikes = _simulate_homogeneous_poisson(rate_hz, duration_s, dt, rng)

    # Pretend the model said the rate was 5 Hz (10x too low).
    wrong_intensity = np.full_like(intensity, 5.0)
    result = time_rescaling_ks_test(wrong_intensity, spikes, dt=dt)
    assert result.ks_pvalue < 1e-3, (
        f"Mis-specified intensity (10x too low) didn't reject: "
        f"ks_stat={result.ks_stat:.3f}, p={result.ks_pvalue:.3e}"
    )


def test_inhomogeneous_intensity_well_specified() -> None:
    """The theorem applies to inhomogeneous Poisson too.  Use a
    time-varying sinusoidal rate and verify the well-specified case
    doesn't reject."""
    dt = 1e-3
    duration_s = 10.0
    t = np.arange(0.0, duration_s, dt)
    intensity = 30.0 + 20.0 * np.sin(2.0 * np.pi * 1.0 * t)  # 1 Hz modulation
    rng = np.random.default_rng(100)
    spikes = (rng.uniform(size=t.size) < intensity * dt).astype(int)

    result = time_rescaling_ks_test(intensity, spikes, dt=dt)
    assert result.ks_pvalue > 0.01, (
        f"Well-specified inhomogeneous case rejected at α=0.01: "
        f"ks_stat={result.ks_stat:.3f}, p={result.ks_pvalue:.3e}"
    )


# ----------------------------------------------------------------------
# Edge cases / input validation
# ----------------------------------------------------------------------


def test_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        time_rescaling_ks_test(np.ones(100), np.zeros(99), dt=1e-3)


def test_rejects_negative_intensity() -> None:
    spikes = np.zeros(100, dtype=int)
    spikes[[10, 50, 90]] = 1
    intensity = np.full(100, 5.0)
    intensity[20] = -0.1  # negative!
    with pytest.raises(ValueError, match="non-negative"):
        time_rescaling_ks_test(intensity, spikes, dt=1e-3)


def test_rejects_fewer_than_two_spikes() -> None:
    """K=1 spike gives 0 ISIs — can't run KS test."""
    intensity = np.full(100, 5.0)
    spikes = np.zeros(100, dtype=int)
    spikes[50] = 1
    with pytest.raises(ValueError, match="at least 2 spikes"):
        time_rescaling_ks_test(intensity, spikes, dt=1e-3)


def test_uniform_z_in_unit_interval() -> None:
    """Sanity: the z = 1 - exp(-xi) transform must land in [0, 1)."""
    rng = np.random.default_rng(0)
    intensity, spikes = _simulate_homogeneous_poisson(50.0, 5.0, 1e-3, rng)
    result = time_rescaling_ks_test(intensity, spikes, dt=1e-3)
    assert np.all(result.uniform_z >= 0.0)
    assert np.all(result.uniform_z < 1.0)


# ----------------------------------------------------------------------
# Cross-check against scipy stats (the oracle uses scipy for the KS
# distribution; this guards against a regression in scipy or in the
# oracle's call to it).
# ----------------------------------------------------------------------


def test_oracle_ks_matches_scipy_ks_on_synthetic_uniform_data() -> None:
    """If we hand-craft a sample of uniform-distributed z values, the
    oracle's reported ks_stat must match scipy.stats.ks_1samp directly."""
    from scipy import stats

    rng = np.random.default_rng(0)
    z_samples = rng.uniform(0.0, 1.0, size=500)

    # Reverse-engineer: xi = -log(1 - z); spike_indicator + intensity
    # such that the cumulative-intensity differences reproduce xi.
    xi = -np.log(1.0 - z_samples)
    dt = 1e-3
    # Place spikes at cumulative xi positions; intensity = 1 everywhere.
    # rescaled_ISI = sum(1 * dt) over the interval = (n_bins) * dt
    # So we need each interval to have xi_k / dt bins.
    cumulative_bins = np.cumsum(np.round(xi / dt).astype(int)) + 1
    n_bins = int(cumulative_bins[-1]) + 10
    spike_indicator = np.zeros(n_bins, dtype=int)
    spike_indicator[0] = 1  # anchor spike at t=0
    for idx in cumulative_bins:
        if idx < n_bins:
            spike_indicator[idx] = 1
    intensity = np.ones(n_bins, dtype=float)

    result = time_rescaling_ks_test(intensity, spike_indicator, dt=dt)
    expected_stat, expected_p = stats.ks_1samp(
        result.uniform_z, stats.uniform(loc=0.0, scale=1.0).cdf
    )
    assert result.ks_stat == pytest.approx(float(expected_stat), abs=1e-12)
    assert result.ks_pvalue == pytest.approx(float(expected_p), abs=1e-12)
