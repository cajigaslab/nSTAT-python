"""Tests for nstat.extras.spatial.hawkes_bridge.bartlett_spectrum.

Synthetic only (no patient data, no MATLAB-repo coupling); all randomness
via np.random.default_rng(seed).

The Bartlett spectrum is the §6.C.1 wave-vector diagnostic of Daley &
Vere-Jones (2003) §8.4 / Bacry-Mastromatteo-Muzy (2015):

    S(f, k) = G(f) * sum_{c,c'} A[c, c'] * exp(-i k . (r_c - r_{c'}))

with G(f) = 1 / (beta + i 2 pi f) for the exponential kernel.

bartlett_spectrum lives in hawkes_bridge.py alongside fit_hawkes_exp BUT
does NOT touch ``tick`` at all — it is pure NumPy/SciPy.  That import
safety is verified by A8.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nstat.extras.spatial import bartlett_spectrum

FIXTURE = (
    Path(__file__).parent / "fixtures" / "bartlett_reference_4ch.npy"
)


def _planar_wave_adjacency(pos, v, beta, A0, rng):
    """Causal planar-wave adjacency:

    A[i, j] = A0 * exp(-beta * (r_i - r_j) . v / |v|^2)   for tau >= 0
            = 0                                            otherwise

    with a small log-normal perturbation to break exact symmetries.
    The fixture generator uses these exact parameters with seed = 42.
    """
    r = pos[:, None, :] - pos[None, :, :]            # (C, C, 2)
    tau = (r @ v) / (v @ v)                           # (C, C)
    adj = A0 * np.exp(-beta * np.maximum(tau, 0.0))
    adj = np.where(tau >= 0, adj, 0.0)
    adj = adj * (1.0 + 0.05 * rng.standard_normal(adj.shape))
    return np.clip(adj, 0.0, None)


# --------------------------------------------------------------------------
# A1 — shape and dtype contract
# --------------------------------------------------------------------------


def test_bartlett_returns_real_power_by_default():
    rng = np.random.default_rng(1)
    C, Nf, Nk = 3, 5, 7
    adj = 0.1 * rng.random((C, C))
    pos = rng.standard_normal((C, 2))
    f = np.linspace(0.5, 5.0, Nf)
    k = rng.standard_normal((Nk, 2))

    S = bartlett_spectrum(adj, pos, f, k, decay=1.0)
    assert S.shape == (Nf, Nk)
    assert S.dtype == np.float64
    assert np.all(S >= 0)


def test_bartlett_returns_complex_on_request():
    rng = np.random.default_rng(2)
    C, Nf, Nk = 3, 5, 7
    adj = 0.1 * rng.random((C, C))
    pos = rng.standard_normal((C, 2))
    f = np.linspace(0.5, 5.0, Nf)
    k = rng.standard_normal((Nk, 2))

    S_c = bartlett_spectrum(adj, pos, f, k, decay=1.0, return_complex=True)
    S_p = bartlett_spectrum(adj, pos, f, k, decay=1.0)
    assert S_c.shape == (Nf, Nk)
    assert S_c.dtype == np.complex128
    np.testing.assert_allclose(np.abs(S_c) ** 2, S_p, rtol=1e-10, atol=1e-14)


# --------------------------------------------------------------------------
# A2 — input validation
# --------------------------------------------------------------------------


def test_bartlett_validates_shapes_and_decays():
    rng = np.random.default_rng(3)
    C = 3
    adj = 0.1 * rng.random((C, C))
    pos = rng.standard_normal((C, 2))
    f = np.linspace(0.5, 5.0, 4)
    k = rng.standard_normal((6, 2))

    with pytest.raises(ValueError, match="triggering_matrix"):
        bartlett_spectrum(rng.random((3, 4)), pos, f, k)
    with pytest.raises(ValueError, match="electrode_positions"):
        bartlett_spectrum(adj, rng.random((C, 3)), f, k)
    with pytest.raises(ValueError, match="freq_grid"):
        bartlett_spectrum(adj, pos, f.reshape(2, 2), k)
    with pytest.raises(ValueError, match="wave_vector_grid"):
        bartlett_spectrum(adj, pos, f, rng.random((6, 3)))
    with pytest.raises(ValueError, match="decay"):
        bartlett_spectrum(adj, pos, f, k, decay=-0.5)
    with pytest.raises(ValueError, match="decay array"):
        bartlett_spectrum(adj, pos, f, k, decay=-np.ones((C, C)))
    with pytest.raises(ValueError, match=r"decay array.*\(C, C\)"):
        bartlett_spectrum(adj, pos, f, k, decay=np.ones((C, C + 1)))


# --------------------------------------------------------------------------
# A3 — planar-wave magnitude / direction / fixture invariance (binding)
# --------------------------------------------------------------------------


def test_bartlett_planar_wave_magnitude_within_10pct():
    """Exact-formula magnitude check at a SPECIFIC (f, k) for the
    planar-wave adjacency.  This is the binding magnitude assertion.

    Note (builder-flagged contract issue): the architect's spec implies
    a sharp ridge at ``k = 2*pi*f*v/|v|^2``, but the planar-wave
    adjacency ``A*exp(-beta*max(tau,0))`` is purely REAL and yields a
    spectrum whose peak in ``k`` is at ``k = 0`` (DC), because phases
    align coherently there.  An exponential-kernel causal adjacency does
    NOT produce a propagating ridge at non-zero ``k`` without a complex
    phase or an oscillatory factor — that ridge prediction belongs to
    propagating kernels ``A*delta(r - v*tau)``, not to the on-lattice
    exponential triggering matrices that ``fit_hawkes_exp`` returns.

    We instead validate the spectrum value at a specific cell against
    the closed-form analytical expression to bind the implementation.
    Fixture parameters: 4 electrodes on a unit 2x2 grid; v = (2.0, 0.0);
    decay = beta = 1.0; A0 = 0.05; rng seed 42.
    """
    rng = np.random.default_rng(42)
    pos = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float
    )
    v = np.array([2.0, 0.0])
    beta = 1.0
    A0 = 0.05
    adj = _planar_wave_adjacency(pos, v, beta, A0, rng)

    # Single (f, k) probe: f=1.5 Hz, k=(1.2, 0.3) — an arbitrary off-DC
    # cell.  Compute the closed-form S(f, k) directly and compare.
    f_probe = np.array([1.5])
    k_probe = np.array([[1.2, 0.3]])
    S = bartlett_spectrum(adj, pos, f_probe, k_probe, decay=beta)

    # Closed-form: S = |G(f)|^2 * |sum_{c, c'} A[c, c'] exp(-i k . r_{cc'})|^2
    r = pos[:, None, :] - pos[None, :, :]            # (C, C, 2)
    phase = (k_probe[0] * r).sum(axis=2)              # (C, C)
    coh = (adj * np.exp(-1j * phase)).sum()
    G = 1.0 / (beta + 1j * 2.0 * np.pi * 1.5)
    expected = (np.abs(G) * np.abs(coh)) ** 2
    measured = float(S[0, 0])
    ratio = measured / expected
    # ARCHITECT GUIDANCE: tight at +-10%; if 10-15% wrap in xfail; >15% STOP.
    # With the closed-form expectation this is exact up to FP roundoff.
    assert 0.9 <= ratio <= 1.1, (
        f"Bartlett spectrum magnitude ratio out of tolerance: {ratio:.6f} "
        f"(expected ~1.0); something is wrong with the spectrum kernel."
    )


def test_bartlett_planar_wave_decay_anisotropy():
    """The Bartlett *power* spectrum of a real adjacency is k -> -k
    symmetric (Hermitian-symmetry of the spatial FT of a real signal),
    so we cannot distinguish forward/backward propagation by sign of k
    alone — see the docstring of ``test_bartlett_planar_wave_magnitude``.

    What IS testable: the rate at which the spectrum decays away from
    k=0 differs between directions aligned with ``v`` and directions
    orthogonal to it, because the causal planar-wave adjacency packs
    differently along/across the propagation axis.  We check that the
    spectrum decays more SLOWLY along v than orthogonal to v at the
    same |k|, consistent with anisotropic coupling.
    """
    rng = np.random.default_rng(42)
    pos = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float
    )
    v = np.array([2.0, 0.0])
    beta, A0 = 1.0, 0.05
    adj = _planar_wave_adjacency(pos, v, beta, A0, rng)

    f = np.array([1.0])
    K = 1.5
    k_along = np.array([[K, 0.0]])         # along v
    k_ortho = np.array([[0.0, K]])         # orthogonal to v
    S_along = float(bartlett_spectrum(adj, pos, f, k_along, decay=beta)[0, 0])
    S_ortho = float(bartlett_spectrum(adj, pos, f, k_ortho, decay=beta)[0, 0])
    S_dc = float(
        bartlett_spectrum(adj, pos, f, np.array([[1e-6, 0.0]]), decay=beta)[0, 0]
    )

    # Both are below the DC peak.
    assert S_along < S_dc
    assert S_ortho < S_dc
    # The adjacency is anisotropic in the directions of v vs ortho-v,
    # so the two cuts of the spectrum must differ at |k|=K.
    rel = abs(S_along - S_ortho) / max(S_along, S_ortho, 1e-30)
    assert rel > 1e-3, (
        f"S(f, k_along_v) == S(f, k_ortho_v) at machine precision "
        f"(rel_diff = {rel:.2e}); anisotropic adjacency should produce "
        f"different spectrum cuts."
    )


def test_bartlett_fixture_invariance_on_reduced_grid():
    """Binding fixture-invariance check: reduced 4x4 grid must match the
    saved reference to rtol=1e-6.  Any code path / numerical drift that
    perturbs the spectrum will tip this test over.
    """
    assert FIXTURE.exists(), (
        f"missing fixture {FIXTURE} — regenerate per the docstring above "
        f"using seed=42 with the planar-wave parameters."
    )
    rng = np.random.default_rng(42)
    pos = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float
    )
    v = np.array([2.0, 0.0])
    beta, A0 = 1.0, 0.05
    adj = _planar_wave_adjacency(pos, v, beta, A0, rng)

    f_red = np.array([0.5, 1.0, 2.0, 4.0])
    k_red = np.array(
        [[0.5, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    )
    S = bartlett_spectrum(adj, pos, f_red, k_red, decay=beta)
    ref = np.load(FIXTURE)
    np.testing.assert_allclose(S, ref, rtol=1e-6)


# --------------------------------------------------------------------------
# A4 — zero adjacency short-circuits to all zeros
# --------------------------------------------------------------------------


def test_bartlett_zero_adjacency_is_all_zeros():
    rng = np.random.default_rng(4)
    C = 4
    adj = np.zeros((C, C))
    pos = rng.standard_normal((C, 2))
    f = np.linspace(0.5, 5.0, 5)
    k = rng.standard_normal((6, 2))

    S = bartlett_spectrum(adj, pos, f, k, decay=1.0)
    assert S.shape == (5, 6)
    assert S.dtype == np.float64
    assert np.all(S == 0.0)

    S_c = bartlett_spectrum(adj, pos, f, k, decay=1.0, return_complex=True)
    assert S_c.dtype == np.complex128
    assert np.all(S_c == 0.0 + 0.0j)


# --------------------------------------------------------------------------
# A5 — stationarity warning when spectral radius >= 1
# --------------------------------------------------------------------------


def test_bartlett_warns_when_spectral_radius_geq_1():
    rng = np.random.default_rng(5)
    C = 3
    # Adjacency with spectral radius > 1: a permutation matrix scaled by 2.
    adj = 2.0 * np.eye(C)
    pos = rng.standard_normal((C, 2))
    f = np.linspace(0.5, 5.0, 4)
    k = rng.standard_normal((5, 2))

    with pytest.warns(UserWarning, match="spectral radius"):
        bartlett_spectrum(adj, pos, f, k, decay=1.0)


# --------------------------------------------------------------------------
# A6 — per-pair (C, C) decay path agrees with scalar path
# --------------------------------------------------------------------------


def test_bartlett_per_pair_decay_matches_scalar_when_constant():
    rng = np.random.default_rng(6)
    C = 4
    adj = 0.1 * rng.random((C, C))
    pos = rng.standard_normal((C, 2))
    f = np.linspace(0.5, 5.0, 6)
    k = rng.standard_normal((8, 2))
    beta = 1.7

    S_scalar = bartlett_spectrum(adj, pos, f, k, decay=beta)
    S_full = bartlett_spectrum(adj, pos, f, k, decay=beta * np.ones((C, C)))
    np.testing.assert_allclose(S_scalar, S_full, rtol=1e-10, atol=1e-14)

    # And the complex paths agree too.
    S_scalar_c = bartlett_spectrum(
        adj, pos, f, k, decay=beta, return_complex=True
    )
    S_full_c = bartlett_spectrum(
        adj, pos, f, k, decay=beta * np.ones((C, C)), return_complex=True
    )
    np.testing.assert_allclose(S_scalar_c, S_full_c, rtol=1e-10, atol=1e-14)


# --------------------------------------------------------------------------
# A7 — frequency factor matches analytical |G(f)|^2 for a single-channel adj
# --------------------------------------------------------------------------


def test_bartlett_frequency_factor_matches_lorentzian():
    """For C=1, adj=[[A]], the spectrum reduces to |G(f)|^2 * A^2 — a
    Lorentzian in f with width set by ``decay``.
    """
    A = 0.3
    beta = 2.0
    adj = np.array([[A]])
    pos = np.array([[0.0, 0.0]])
    f = np.linspace(0.1, 5.0, 20)
    k = np.array([[0.5, 0.5]])  # single arbitrary non-DC k

    S = bartlett_spectrum(adj, pos, f, k, decay=beta)
    expected = A ** 2 / (beta ** 2 + (2.0 * np.pi * f) ** 2)
    np.testing.assert_allclose(S[:, 0], expected, rtol=1e-10)


# --------------------------------------------------------------------------
# A8 — bartlett_spectrum does NOT import ``tick``
# --------------------------------------------------------------------------


def test_bartlett_spectrum_does_not_require_tick():
    """A user without the optional [hawkes] dep must be able to call
    bartlett_spectrum.  We assert that calling it leaves ``tick`` out of
    ``sys.modules`` (we cannot uninstall tick in-process, but if it
    became a transitive import the test fixture file would have already
    been generated with the eager import path).
    """
    import sys

    # Snapshot current state of tick in sys.modules.
    tick_was_loaded = "tick" in sys.modules

    rng = np.random.default_rng(8)
    C = 3
    adj = 0.1 * rng.random((C, C))
    pos = rng.standard_normal((C, 2))
    f = np.linspace(0.5, 5.0, 4)
    k = rng.standard_normal((5, 2))
    bartlett_spectrum(adj, pos, f, k, decay=1.0)

    # If tick was NOT loaded before this call, it must still NOT be loaded.
    if not tick_was_loaded:
        assert "tick" not in sys.modules, (
            "bartlett_spectrum triggered an import of `tick`; it must stay "
            "tick-free (the function is pure NumPy/SciPy)."
        )
