"""Tests for multivariate_gof_with_coupling — the wrapper that bundles the
per-channel discrete-time test (Haslinger-Pipa-Brown 2010) with the
population marked-region :math:`\\chi^2` (Tao et al. 2018).

Synthetic data only; no patient data.
"""
from __future__ import annotations

import numpy as np
import pytest

from nstat import PopulationTimeRescaleResult, population_time_rescale
from nstat.extras.spatial.marked_gof import (
    CoupledMarkedGOFResult,
    MarkedGOFResult,
    multivariate_gof_with_coupling,
    multivariate_time_rescaling,
)


def _independent_population(rng, n_channels=3, n_bins=80000, Delta=0.001):
    """A correctly-specified, independent multi-channel binned Poisson stream.

    Bins are 1 ms wide so per-bin probabilities stay :math:`\\lesssim 0.02`,
    keeping the bin-summed ground-process compensator close to the
    continuous integral — the regime in which the
    :func:`nstat.population_time_rescale` ground KS is well-calibrated.
    Each channel has a time-VARYING sinusoidal rate; constant rates
    would quantize the rescaled inter-event distribution and over-reject
    for a reason unrelated to model fit.
    """
    base = np.array([4.0, 6.0, 9.0])[:n_channels]
    t = np.arange(n_bins) * Delta
    p_per_channel = []
    for c, r0 in enumerate(base):
        modulation = 1.0 + 0.6 * np.sin(2.0 * np.pi * (0.5 + 0.2 * c) * t)
        p_per_channel.append(np.clip(r0 * modulation * Delta, 0.0, 1.0))
    spike_bins_per_channel = [
        np.flatnonzero(rng.uniform(size=n_bins) < p) for p in p_per_channel
    ]
    return spike_bins_per_channel, p_per_channel


def test_wrapper_returns_bundled_per_channel_and_population():
    rng = np.random.default_rng(7)
    sb, pk = _independent_population(rng)

    res = multivariate_gof_with_coupling(
        sb, pk, n_draws=20, n_tau_bins=4, rng=np.random.default_rng(101),
    )

    assert isinstance(res, CoupledMarkedGOFResult)
    assert set(res.per_channel) == {0, 1, 2}
    for c in res.per_channel:
        assert isinstance(res.per_channel[c], MarkedGOFResult)
    assert isinstance(res.population, PopulationTimeRescaleResult)


def test_wrapper_agrees_with_separate_calls_on_per_channel():
    """The wrapper's per-channel results match a direct call (same rng path)."""
    rng = np.random.default_rng(11)
    sb, pk = _independent_population(rng)

    direct_rng = np.random.default_rng(303)
    direct = multivariate_time_rescaling(sb, pk, n_draws=20, rng=direct_rng)

    bundled_rng = np.random.default_rng(303)
    bundled = multivariate_gof_with_coupling(sb, pk, n_draws=20, rng=bundled_rng)

    for c in direct:
        # The corrected/uncorrected stats are deterministic given the rng
        # seed, so identical seeds must yield identical numbers.
        assert direct[c].ks_corrected == bundled.per_channel[c].ks_corrected
        assert direct[c].ks_uncorrected == bundled.per_channel[c].ks_uncorrected


def test_wrapper_population_matches_manual_conversion():
    """The bundled population result equals a hand-converted direct call."""
    rng = np.random.default_rng(13)
    sb, pk = _independent_population(rng)

    bundled = multivariate_gof_with_coupling(
        sb, pk, n_draws=10, n_tau_bins=3, rng=np.random.default_rng(0),
    )

    T = len(pk[0])
    counts_list = [
        np.bincount(np.asarray(s, dtype=int), minlength=T).astype(float)
        for s in sb
    ]
    lam_per_bin_list = [np.asarray(p, dtype=float) for p in pk]
    direct_pop = population_time_rescale(counts_list, lam_per_bin_list, n_tau_bins=3)

    assert bundled.population.ground_ks_stat == direct_pop.ground_ks_stat
    assert bundled.population.mark_chi2_stat == direct_pop.mark_chi2_stat
    np.testing.assert_array_equal(
        bundled.population.expected_counts, direct_pop.expected_counts
    )
    np.testing.assert_array_equal(
        bundled.population.observed_counts, direct_pop.observed_counts
    )


def test_wrapper_true_model_passes_both_tests():
    """Independent correctly-specified channels with fine bins: per-channel
    KS *and* population KS / mark-:math:`\\chi^2` all pass."""
    rng = np.random.default_rng(17)
    sb, pk = _independent_population(rng)

    # alpha=0.01 (wider band) to keep the test stable across rng draws —
    # at alpha=0.05 a single per-channel KS can land just outside the band
    # ~5% of the time even under the true model.
    res = multivariate_gof_with_coupling(
        sb, pk, n_draws=20, n_tau_bins=4, alpha=0.01,
        rng=np.random.default_rng(19),
    )

    for c, r in res.per_channel.items():
        assert r.inside_corrected, (
            f"channel {c}: discrete-time-corrected KS rejected the TRUE model "
            f"(ks_corrected={r.ks_corrected:.4f}, band={r.ks_band:.4f})"
        )
    assert res.population.ground_ks_pvalue > 0.01, (
        f"ground KS rejected the TRUE model (p={res.population.ground_ks_pvalue:.3g})"
    )
    assert res.population.mark_chi2_pvalue > 0.01


def test_wrapper_rejects_mismatched_lengths():
    rng = np.random.default_rng(2)
    sb, pk = _independent_population(rng, n_channels=2, n_bins=1000)

    with pytest.raises(ValueError, match="equal length"):
        multivariate_gof_with_coupling(sb[:1], pk)

    pk_bad = [pk[0], pk[1][:500]]
    with pytest.raises(ValueError, match="same time grid"):
        multivariate_gof_with_coupling(sb, pk_bad)


def test_wrapper_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one channel"):
        multivariate_gof_with_coupling([], [])
