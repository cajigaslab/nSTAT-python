"""Tests for the multivariate (marked) time-rescaling GOF.

``nstat.population_time_rescale`` implements the Tao, Weber, Arai & Eden
(2018) marked point-process time-rescaling framework.  The headline
property — and the reason it exists alongside the per-neuron univariate
KS test — is that it catches inter-neuron *coupling* misfit that the
univariate test misses.  These tests pin:

1. a correctly-specified population is not rejected;
2. an overall-rate misspecification is caught by the ground-process KS;
3. **the canonical case**: a model with correct per-neuron marginals but
   missing dependency passes the univariate KS for every neuron yet is
   rejected by the population ground-process KS;
4. a relative-proportion misfit is caught by the marked chi-square (where
   the ground KS, with the correct total rate, passes);
5. for a single neuron the ground process reduces to the univariate
   rescaling;
6. input validation.
"""
from __future__ import annotations

import numpy as np
import pytest

from nstat.fit import _time_rescaled_uniforms, population_time_rescale
from tests.parity._third_party.time_rescale_oracle import time_rescaling_ks_test


def _sim_inhomogeneous_population(n_neurons=4, n_bins=6000, seed=0):
    """Correctly-specified inhomogeneous-Poisson population + its λ."""
    rng = np.random.default_rng(seed)
    counts, lams = [], []
    for k in range(n_neurons):
        lam = 0.01 * (1.0 + 0.5 * np.sin(2 * np.pi * (np.arange(n_bins) / n_bins) * (k + 1)))
        counts.append((rng.random(n_bins) < lam).astype(float))
        lams.append(lam)
    return counts, lams


def test_population_time_rescale_correct_model_not_rejected() -> None:
    """A correctly-specified population fails to reject on both statistics."""
    counts, lams = _sim_inhomogeneous_population(seed=0)
    r = population_time_rescale(counts, lams, n_tau_bins=5)
    assert np.isfinite(r.ground_ks_pvalue) and np.isfinite(r.mark_chi2_pvalue)
    assert r.ground_ks_pvalue > 0.01, f"ground KS spuriously rejected: {r.ground_ks_pvalue}"
    assert r.mark_chi2_pvalue > 0.01, f"mark chi2 spuriously rejected: {r.mark_chi2_pvalue}"


def test_ground_ks_rejects_wrong_overall_rate() -> None:
    """Scaling every intensity down breaks the ground-process rescaling.

    The marked chi-square is (correctly) invariant to a global rate
    scaling — it tests relative allocation — so only the ground KS fires.
    """
    counts, lams = _sim_inhomogeneous_population(seed=0)
    r = population_time_rescale(counts, [l * 0.33 for l in lams], n_tau_bins=5)
    assert r.ground_ks_pvalue < 1e-6, f"ground KS missed wrong rate: {r.ground_ks_pvalue}"


def test_population_catches_coupling_that_univariate_misses() -> None:
    """The canonical case: correct marginals, missing dependency.

    Two neurons fire near-synchronously (a shared drive, B offset 2 bins
    after A).  The model uses each neuron's *correct marginal* rate but
    treats them as independent.  Each neuron's univariate KS passes — yet
    the population ground-process KS rejects, because the pooled process
    is not Poisson under the independence assumption.
    """
    rng = np.random.default_rng(7)
    n_bins = 20000
    p_drive, p_noise = 0.01, 0.004
    drive = rng.random(n_bins) < p_drive
    a = (drive | (rng.random(n_bins) < p_noise)).astype(float)
    b_drive = np.zeros(n_bins, dtype=bool)
    idx = np.flatnonzero(drive)
    b_drive[np.minimum(idx + 2, n_bins - 1)] = True
    b = (b_drive | (rng.random(n_bins) < p_noise)).astype(float)
    lam_a = np.full(n_bins, a.mean())   # correct marginal
    lam_b = np.full(n_bins, b.mean())   # correct marginal

    # Per-neuron univariate KS (independent reference oracle) — passes.
    ks_a = time_rescaling_ks_test(lam_a, a, dt=1.0)
    ks_b = time_rescaling_ks_test(lam_b, b, dt=1.0)
    assert ks_a.ks_pvalue > 0.05 and ks_b.ks_pvalue > 0.05, (
        f"univariate unexpectedly rejected: A={ks_a.ks_pvalue}, B={ks_b.ks_pvalue}"
    )

    # Population ground-process KS — catches the unmodeled dependency.
    r = population_time_rescale([a, b], [lam_a, lam_b], n_tau_bins=5)
    assert r.ground_ks_pvalue < 1e-3, (
        f"population KS missed the coupling: {r.ground_ks_pvalue}"
    )


def test_mark_chi2_catches_proportion_misfit_ground_ks_does_not() -> None:
    """Wrong relative allocation across neurons (right total rate).

    Neuron 0 truly fires ~8x neuron 1, but the model splits the rate
    equally.  The marked chi-square rejects; the ground KS — whose total
    intensity is still correct — does not.  Complementary coverage.
    """
    rng = np.random.default_rng(11)
    n_bins = 20000
    r0, r1 = 0.04, 0.005
    n0 = (rng.random(n_bins) < r0).astype(float)
    n1 = (rng.random(n_bins) < r1).astype(float)
    lam_equal = np.full(n_bins, (r0 + r1) / 2.0)
    r = population_time_rescale([n0, n1], [lam_equal, lam_equal], n_tau_bins=1)
    assert r.mark_chi2_pvalue < 1e-6, f"mark chi2 missed proportion misfit: {r.mark_chi2_pvalue}"
    assert r.ground_ks_pvalue > 0.01, f"ground KS should pass (right total): {r.ground_ks_pvalue}"
    assert r.mark_chi2_dof == 1  # 2 cells (n_tau_bins=1, 2 neurons) - 1


def test_single_neuron_ground_process_equals_univariate_rescaling() -> None:
    """With one neuron the ground process IS that neuron's rescaling."""
    counts, lams = _sim_inhomogeneous_population(n_neurons=1, seed=2)
    r = population_time_rescale(counts, lams)
    expected = _time_rescaled_uniforms(counts[0], lams[0])
    assert np.allclose(r.ground_uniforms, expected)
    assert np.isclose(r.observed_counts[0], counts[0].sum())
    assert np.isclose(r.expected_counts[0], lams[0].sum())


def test_population_time_rescale_input_validation() -> None:
    counts, lams = _sim_inhomogeneous_population(n_neurons=2, n_bins=100, seed=0)
    with pytest.raises(ValueError, match="equal length"):
        population_time_rescale(counts, lams[:1])
    with pytest.raises(ValueError, match="at least one neuron"):
        population_time_rescale([], [])
    with pytest.raises(ValueError, match="n_tau_bins must be >= 1"):
        population_time_rescale(counts, lams, n_tau_bins=0)
    with pytest.raises(ValueError, match="same number of time bins"):
        population_time_rescale([counts[0], counts[1][:50]], [lams[0], lams[1][:50]])
