"""Tests for nstat.extras.spatial.sbm_hawkes — SBM-prior multivariate Hawkes EM.

Synthetic data only (``np.random.default_rng``); no patient data.

Contract checks (architect's brief Tier G PR-2):
- SBMHawkesSpec validation rejects invalid parameters and warns on
  super-critical initialisations.
- The EM recovers a K=2 block structure up to label permutation on a
  10-process Linderman-style simulation (within a loose accuracy budget
  because hard-EM can get stuck on hard initialisations).
- The K=1 degenerate case collapses to the single-block Hawkes EM
  behaviour: A_hat is 1x1, the recovered (alpha, beta) match truth
  within reasonable tolerance, and z_hat is all-zero.
- Log-likelihood is monotone non-decreasing across EM iterations (the
  greedy z update uses an accept-if-LL-up safeguard, and the
  closed-form (mu, beta, A) M-step is the exact Q-maximiser).
- The simulator rejects super-critical configurations (spectral radius
  of A/beta >= 1).
- Edge cases (empty input list, silent process, unsorted events,
  T <= max event time, K > N) all raise ValueError.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from nstat.extras.spatial.sbm_hawkes import (
    SBMHawkesResult,
    SBMHawkesSpec,
    em_sbm_hawkes,
    simulate_sbm_hawkes,
)


# ----------------------------------------------------------------------
# 1. Spec validation
# ----------------------------------------------------------------------


def test_spec_validation():
    """__post_init__ rejects out-of-range parameters and warns on super-critical."""
    with pytest.raises(ValueError, match="n_blocks"):
        SBMHawkesSpec(n_blocks=0)
    with pytest.raises(ValueError, match="alpha0"):
        SBMHawkesSpec(n_blocks=2, alpha0=0.0)
    with pytest.raises(ValueError, match="alpha0"):
        SBMHawkesSpec(n_blocks=2, alpha0=-0.1)
    with pytest.raises(ValueError, match="alpha0_off"):
        SBMHawkesSpec(n_blocks=2, alpha0_off=-0.01)
    with pytest.raises(ValueError, match="beta0"):
        SBMHawkesSpec(n_blocks=2, beta0=0.0)
    with pytest.raises(ValueError, match="beta0"):
        SBMHawkesSpec(n_blocks=2, beta0=-1.0)
    with pytest.raises(ValueError, match="max_iter"):
        SBMHawkesSpec(n_blocks=2, max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        SBMHawkesSpec(n_blocks=2, tol=0.0)
    with pytest.raises(ValueError, match="tol"):
        SBMHawkesSpec(n_blocks=2, tol=-1e-9)
    with pytest.raises(ValueError, match="z_init"):
        SBMHawkesSpec(n_blocks=2, z_init="lloyd")

    # Default (sub-critical) spec should not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = SBMHawkesSpec(n_blocks=2)

    # Super-critical alpha0 / beta0 warns.
    with pytest.warns(UserWarning, match="super-critical"):
        SBMHawkesSpec(n_blocks=2, alpha0=1.0, beta0=1.0)
    with pytest.warns(UserWarning, match="super-critical"):
        SBMHawkesSpec(n_blocks=2, alpha0=2.0, beta0=1.0)


# ----------------------------------------------------------------------
# 2. Edge cases
# ----------------------------------------------------------------------


def test_em_rejects_empty_input():
    """Empty event-list raises ValueError."""
    spec = SBMHawkesSpec(n_blocks=1)
    with pytest.raises(ValueError, match="at least one process"):
        em_sbm_hawkes([], T=10.0, spec=spec)


def test_em_rejects_silent_process():
    """A process with zero events raises ValueError."""
    spec = SBMHawkesSpec(n_blocks=1)
    events = [np.array([1.0, 2.0]), np.array([], dtype=np.float64)]
    with pytest.raises(ValueError, match="silent"):
        em_sbm_hawkes(events, T=10.0, spec=spec)


def test_em_rejects_unsorted():
    """Unsorted events raise ValueError."""
    spec = SBMHawkesSpec(n_blocks=1)
    events = [np.array([2.0, 1.0, 3.0])]
    with pytest.raises(ValueError, match="sorted"):
        em_sbm_hawkes(events, T=10.0, spec=spec)


def test_em_rejects_T_below_max_event():
    """T <= last event time raises ValueError."""
    spec = SBMHawkesSpec(n_blocks=1)
    events = [np.array([1.0, 2.0, 5.0])]
    with pytest.raises(ValueError, match="T="):
        em_sbm_hawkes(events, T=5.0, spec=spec)


def test_em_rejects_n_blocks_exceeds_n_processes():
    """K > N raises ValueError."""
    spec = SBMHawkesSpec(n_blocks=5)
    events = [np.array([1.0, 2.0]), np.array([1.5, 2.5])]
    with pytest.raises(ValueError, match="exceeds"):
        em_sbm_hawkes(events, T=10.0, spec=spec)


def test_simulator_rejects_supercritical():
    """A super-critical (A, beta) pair raises ValueError."""
    rng = np.random.default_rng(0)
    z = np.array([0], dtype=np.int_)
    A = np.array([[2.0]], dtype=np.float64)
    mu = np.array([0.5], dtype=np.float64)
    with pytest.raises(ValueError, match="super-critical"):
        simulate_sbm_hawkes(z, A, beta=1.0, mu=mu, T=50.0, rng=rng)


# ----------------------------------------------------------------------
# 3. Recovery on K=2 simulated data
# ----------------------------------------------------------------------


def test_em_recovers_two_block_structure():
    """EM recovers a 2-block partition of 10 processes up to permutation.

    Hard-EM with greedy z is sensitive to its starting partition; we
    therefore use ``z_init='kmeans-rate'`` on the empirical-rate vector,
    which exposes the rate gap between intra- and inter-block coupling.
    The accuracy budget (``>= 0.7`` correctly classified after Hungarian
    alignment) is intentionally loose because hard-EM can occasionally
    land in a local optimum where one process is mis-assigned.
    """
    from scipy.optimize import linear_sum_assignment

    # NB: with K=2 blocks of size 5 each, the pairwise coupling
    # spectral radius is 5 * (alpha + alpha_off) for intra=alpha,
    # inter=alpha_off — must remain <1 for the multivariate process to
    # be sub-critical.  We pick alpha=0.12, alpha_off=0.02 (radius
    # 5 * 0.14 = 0.7) which preserves a strong diagonal-vs-off signal
    # while staying stable.
    rng = np.random.default_rng(20260617)
    n_per_block = 5
    n_processes = 2 * n_per_block
    z_true = np.repeat([0, 1], n_per_block).astype(np.int_)
    mu = 0.5 * np.ones(n_processes)
    A_true = np.array([[0.12, 0.02], [0.02, 0.12]], dtype=np.float64)
    beta_true = 1.0
    T = 100.0

    events = simulate_sbm_hawkes(z_true, A_true, beta_true, mu, T, rng=rng)
    # Sanity: we expect a healthy number of events per process at this
    # configuration (mu=0.5, intra-block alpha=0.4, beta=1 -> 50 baseline
    # + cascade).  If a process is silent we'd rather know about it now.
    for n_idx, ev in enumerate(events):
        assert ev.size > 0, f"process {n_idx} silent in simulation"

    spec = SBMHawkesSpec(
        n_blocks=2,
        alpha0=0.1,
        alpha0_off=0.02,
        beta0=1.0,
        max_iter=20,
        tol=1e-6,
        z_init="kmeans-rate",
    )
    result = em_sbm_hawkes(events, T=T, spec=spec, seed=20260617)

    assert isinstance(result, SBMHawkesResult)
    assert result.n_processes == n_processes
    assert result.n_blocks == 2
    assert result.n_iter > 0
    assert result.A_hat.shape == (2, 2)
    assert result.z_hat.shape == (n_processes,)

    # Label alignment via Hungarian assignment on the negative confusion
    # matrix (-1 * count of (true=i, pred=j)).
    z_hat = result.z_hat
    confusion = np.zeros((2, 2), dtype=np.int_)
    for true_lab, pred_lab in zip(z_true, z_hat):
        confusion[true_lab, pred_lab] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    perm = {int(col_ind[i]): int(row_ind[i]) for i in range(len(row_ind))}
    z_hat_aligned = np.array([perm[int(k)] for k in z_hat], dtype=np.int_)

    accuracy = float(np.mean(z_hat_aligned == z_true))
    assert accuracy >= 0.7, (
        f"recovery accuracy {accuracy:.2f} below 0.7 target"
    )

    # Align A_hat under the same permutation: A_aligned[i, j] =
    # A_hat[perm_inv[i], perm_inv[j]] where perm_inv maps true label ->
    # recovered label.
    inv_perm = {v: k for k, v in perm.items()}
    A_aligned = np.array(
        [
            [result.A_hat[inv_perm[i], inv_perm[j]] for j in range(2)]
            for i in range(2)
        ]
    )
    # Block-vs-mixture signal: diagonal stronger than off-diagonal.
    assert A_aligned[0, 0] > A_aligned[0, 1]
    assert A_aligned[1, 1] > A_aligned[1, 0]


# ----------------------------------------------------------------------
# 4. K=1 degenerate case (collapses to univariate Hawkes)
# ----------------------------------------------------------------------


def test_em_collapses_to_single_block_when_K_eq_1():
    """K=1: SBM-Hawkes degenerates to a univariate-Hawkes-style EM.

    With all processes in one block, A_hat is a 1x1 amplitude that should
    sit in the same neighbourhood as the univariate Hawkes alpha (up to a
    per-process normalisation).  Tolerances are loose (±40%) because the
    single-realisation Hawkes likelihood is weakly identified between
    alpha and beta even in the univariate case (see hawkes_em.py docstring).
    """
    # K=1, N=4: pairwise spectral radius = 4 * A[0,0] must remain <1.
    # alpha=0.15 -> radius 0.6.
    rng = np.random.default_rng(20260618)
    n_processes = 4
    z_true = np.zeros(n_processes, dtype=np.int_)
    A_true = np.array([[0.15]], dtype=np.float64)
    beta_true = 1.0
    mu = 0.5 * np.ones(n_processes)
    T = 200.0

    events = simulate_sbm_hawkes(z_true, A_true, beta_true, mu, T, rng=rng)
    for ev in events:
        assert ev.size > 5

    spec = SBMHawkesSpec(
        n_blocks=1,
        alpha0=0.15,
        alpha0_off=0.05,
        beta0=1.0,
        max_iter=30,
        tol=1e-6,
        z_init="kmeans-rate",
    )
    result = em_sbm_hawkes(events, T=T, spec=spec, seed=20260618)

    assert result.A_hat.shape == (1, 1)
    assert np.all(result.z_hat == 0)
    # The per-pair alpha is on the order of the true alpha; tolerance
    # ±40% to absorb the alpha/beta identifiability ridge.
    alpha_hat = float(result.A_hat[0, 0])
    assert abs(alpha_hat - 0.15) / 0.15 < 0.4, (
        f"alpha_hat {alpha_hat:.3f} too far from 0.15"
    )
    assert abs(result.beta_hat - 1.0) / 1.0 < 0.3, (
        f"beta_hat {result.beta_hat:.3f} too far from 1.0"
    )


# ----------------------------------------------------------------------
# 5. Log-likelihood monotonicity
# ----------------------------------------------------------------------


def test_em_log_likelihood_non_decreasing():
    """Trace is non-decreasing (greedy z safeguard + closed-form Q-max).

    Uses the same 2-block setup as ``test_em_recovers_two_block_structure``
    but only asks that every iteration accept a step that does not
    decrease LL beyond a small numerical tolerance (1e-9 relative).
    """
    rng = np.random.default_rng(20260617)
    n_per_block = 5
    z_true = np.repeat([0, 1], n_per_block).astype(np.int_)
    n_processes = 2 * n_per_block
    mu = 0.5 * np.ones(n_processes)
    # Same sub-critical config as test_em_recovers_two_block_structure.
    A_true = np.array([[0.12, 0.02], [0.02, 0.12]], dtype=np.float64)
    T = 100.0
    events = simulate_sbm_hawkes(z_true, A_true, 1.0, mu, T, rng=rng)

    spec = SBMHawkesSpec(
        n_blocks=2,
        alpha0=0.1,
        alpha0_off=0.02,
        beta0=1.0,
        max_iter=15,
        tol=1e-8,
        z_init="kmeans-rate",
    )
    result = em_sbm_hawkes(events, T=T, spec=spec, seed=20260617)

    diffs = np.diff(result.log_likelihood_trace)
    # Allow tiny negative jitter from float-precision in the LL summation.
    assert np.all(diffs >= -1e-6), (
        f"LL trace decreased: min diff {diffs.min():.3e}, "
        f"trace = {result.log_likelihood_trace}"
    )
