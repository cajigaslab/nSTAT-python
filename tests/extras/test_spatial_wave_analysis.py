"""Tests for nstat.extras.spatial.wave_analysis.

Synthetic only; np.random.default_rng for any randomness.
"""
from __future__ import annotations

import numpy as np
import pytest

from nstat.extras.spatial import (
    WaveAnalysisResult,
    bartlett_spectrum,
    detect_wave_peaks,
    reconstruct_kernel,
)


# --------------------------------------------------------------------------
# B1 — reconstruct_kernel shape / values / scalar-vs-(C,C) decay equivalence
# --------------------------------------------------------------------------


def test_reconstruct_kernel_shape_and_values():
    rng = np.random.default_rng(1)
    C = 3
    adj = 0.1 * rng.random((C, C))
    tau = np.linspace(0.0, 2.0, 11)
    phi = reconstruct_kernel(adj, 1.5, tau)
    assert phi.shape == (C, C, len(tau))
    # At tau=0, phi = adj exactly.
    np.testing.assert_allclose(phi[:, :, 0], adj, atol=1e-14)
    # phi >= 0 everywhere (adj is non-negative, decay non-negative).
    assert np.all(phi >= 0)


def test_reconstruct_kernel_decay_scalar_matches_constant_matrix():
    rng = np.random.default_rng(2)
    C = 3
    adj = 0.1 * rng.random((C, C))
    tau = np.linspace(0.0, 2.0, 7)
    beta = 1.3
    phi_scalar = reconstruct_kernel(adj, beta, tau)
    phi_full = reconstruct_kernel(adj, beta * np.ones((C, C)), tau)
    np.testing.assert_allclose(phi_scalar, phi_full, rtol=1e-12, atol=1e-14)


# --------------------------------------------------------------------------
# B2 — reconstruct_kernel input validation
# --------------------------------------------------------------------------


def test_reconstruct_kernel_validates_inputs():
    rng = np.random.default_rng(3)
    C = 3
    adj = 0.1 * rng.random((C, C))
    tau_pos = np.linspace(0.0, 1.0, 5)

    # Non-square adjacency.
    with pytest.raises(ValueError, match="adjacency"):
        reconstruct_kernel(np.zeros((3, 4)), 1.0, tau_pos)
    # 2-D tau_grid.
    with pytest.raises(ValueError, match="tau_grid"):
        reconstruct_kernel(adj, 1.0, tau_pos.reshape(-1, 1))
    # Negative tau.
    with pytest.raises(ValueError, match="non-negative"):
        reconstruct_kernel(adj, 1.0, np.array([0.0, -0.1, 0.2]))
    # Negative scalar decay.
    with pytest.raises(ValueError, match="non-negative"):
        reconstruct_kernel(adj, -0.5, tau_pos)
    # Negative array decay.
    bad_decays = np.ones((C, C))
    bad_decays[0, 0] = -1.0
    with pytest.raises(ValueError, match="negative"):
        reconstruct_kernel(adj, bad_decays, tau_pos)
    # Wrong-shape decay array.
    with pytest.raises(ValueError, match=r"\(C, C\)"):
        reconstruct_kernel(adj, np.ones((C, C + 1)), tau_pos)


# --------------------------------------------------------------------------
# B3 — reconstruct_kernel integrated norm
# --------------------------------------------------------------------------


def test_reconstruct_kernel_integrated_norm_matches_adjacency_over_beta():
    """The integral over tau of an exponential kernel exp(-beta tau)
    is 1/beta, so the integrated phi[c1, c2] -> adj[c1, c2] / beta.
    Trapz on a fine grid should recover this to ~1%.
    """
    rng = np.random.default_rng(4)
    C = 3
    adj = 0.1 * rng.random((C, C))
    beta = 2.0
    tau = np.linspace(0.0, 20.0, 2001)  # dense, far past 1/beta
    phi = reconstruct_kernel(adj, beta, tau)
    # NumPy <2.0 uses ``trapz``; NumPy >=2.0 renamed to ``trapezoid``.
    trap = getattr(np, "trapezoid", None) or np.trapz
    integrated = trap(phi, tau, axis=2)
    expected = adj / beta
    np.testing.assert_allclose(integrated, expected, rtol=0.005)


# --------------------------------------------------------------------------
# B4 — detect_wave_peaks dataclass + sort order
# --------------------------------------------------------------------------


def test_detect_wave_peaks_returns_dataclass_in_descending_power_order():
    # Hand-crafted spectrum with known peaks.
    f = np.linspace(0.5, 5.0, 5)
    k = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 2.0]]
    )
    S = np.zeros((5, 5))
    # Top peak at (f_idx=3, k_idx=4)
    S[3, 4] = 10.0
    # Second at (1, 2)
    S[1, 2] = 5.0
    # Third at (0, 1)
    S[0, 1] = 2.0
    S[0, 0] = 1.0  # DC k=(0,0) — should be masked even though non-trivial

    res = detect_wave_peaks(S, f, k, n_peaks=3, min_separation_bins=1)
    assert isinstance(res, WaveAnalysisResult)
    assert len(res.freq) == 3
    # Descending order:
    assert res.power[0] > res.power[1] > res.power[2]
    np.testing.assert_allclose(res.power, [10.0, 5.0, 2.0])
    # Top peak: f[3] = 0.5 + 3*(5-0.5)/4 = 3.875, k=(1, 2).
    assert res.freq[0] == pytest.approx(f[3])
    assert (res.kx[0], res.ky[0]) == (pytest.approx(1.0), pytest.approx(2.0))


# --------------------------------------------------------------------------
# B5 — detect_wave_peaks DC mask
# --------------------------------------------------------------------------


def test_detect_wave_peaks_masks_dc_wave_vector():
    f = np.linspace(0.5, 5.0, 4)
    # First k is exactly (0, 0) — DC; it should be excluded.
    k = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    S = np.full((4, 3), 1.0)
    S[2, 0] = 100.0  # huge value at DC — must be masked
    res = detect_wave_peaks(S, f, k, n_peaks=2)
    # The DC value 100 must NOT appear in the result.
    assert 100.0 not in res.power.tolist()
    # All accepted kx, ky come from non-DC rows.
    assert np.all(res.kx ** 2 + res.ky ** 2 > 0)


# --------------------------------------------------------------------------
# B6 — detect_wave_peaks non-maximum suppression
# --------------------------------------------------------------------------


def test_detect_wave_peaks_nonmax_suppression_drops_adjacent_cells():
    """With min_separation_bins=2, the second-largest cell adjacent to
    the top peak must be suppressed, and the next valid candidate
    farther away must be accepted instead.
    """
    f = np.linspace(0.0, 4.0, 5)
    # 5 distinct non-DC k's:
    k = np.array(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 3.0]]
    )
    S = np.zeros((5, 5))
    S[2, 2] = 10.0   # top peak at (f_idx=2, k_idx=2)
    S[2, 3] = 9.0    # adjacent in k (k_idx=3 vs 2) — suppressed at sep=2
    S[2, 1] = 8.5    # adjacent in k (k_idx=1 vs 2) — suppressed at sep=2
    S[4, 4] = 8.0    # far away, accepted
    S[0, 0] = 7.0    # far away, accepted

    res = detect_wave_peaks(S, f, k, n_peaks=3, min_separation_bins=2)
    assert len(res.freq) == 3
    # Top is the (f=2, k=2) cell.
    assert res.power[0] == pytest.approx(10.0)
    # Suppressed values 9, 8.5 should NOT appear.
    assert 9.0 not in res.power.tolist()
    assert 8.5 not in res.power.tolist()
    # The next two accepted are (4, 4) and (0, 0).
    np.testing.assert_allclose(sorted(res.power[1:].tolist(), reverse=True), [8.0, 7.0])


# --------------------------------------------------------------------------
# B7 — detect_wave_peaks speed/direction formulas
# --------------------------------------------------------------------------


def test_detect_wave_peaks_speed_and_direction_formulas():
    f = np.array([1.0])
    k = np.array([[3.0, 4.0]])  # |k| = 5
    S = np.array([[1.0]])
    res = detect_wave_peaks(S, f, k, n_peaks=1)
    # speed = 2 pi f / |k| = 2 pi / 5
    assert res.speed[0] == pytest.approx(2.0 * np.pi / 5.0)
    # direction = atan2(4, 3)
    assert res.direction[0] == pytest.approx(np.arctan2(4.0, 3.0))


# --------------------------------------------------------------------------
# B8 — detect_wave_peaks input validation
# --------------------------------------------------------------------------


def test_detect_wave_peaks_validates_inputs():
    f = np.array([1.0, 2.0])
    k = np.array([[1.0, 0.0], [0.0, 1.0]])
    S = np.zeros((2, 2))
    # Spectrum not 2-D
    with pytest.raises(ValueError, match="2-D"):
        detect_wave_peaks(np.zeros((2, 2, 2)), f, k)
    # Wrong-shape freq_grid
    with pytest.raises(ValueError, match="freq_grid"):
        detect_wave_peaks(S, np.array([1.0]), k)
    # Wrong-shape wave_vector_grid
    with pytest.raises(ValueError, match="wave_vector_grid"):
        detect_wave_peaks(S, f, np.array([[1.0]]))
    # n_peaks invalid
    with pytest.raises(ValueError, match="n_peaks"):
        detect_wave_peaks(S, f, k, n_peaks=0)
    # min_separation_bins invalid
    with pytest.raises(ValueError, match="min_separation_bins"):
        detect_wave_peaks(S, f, k, min_separation_bins=-1)


# --------------------------------------------------------------------------
# B9 — detect_wave_peaks gracefully handles all-DC grid
# --------------------------------------------------------------------------


def test_detect_wave_peaks_empty_when_all_k_are_dc():
    f = np.linspace(1.0, 4.0, 3)
    k = np.zeros((4, 2))   # ALL wave vectors are DC
    S = np.full((3, 4), 1.0)
    res = detect_wave_peaks(S, f, k, n_peaks=3)
    # All k's masked => empty result.
    assert len(res.freq) == 0
    assert len(res.kx) == 0
    assert len(res.power) == 0


# --------------------------------------------------------------------------
# B10 — end-to-end: bartlett_spectrum + detect_wave_peaks coherent
# --------------------------------------------------------------------------


def test_end_to_end_bartlett_into_detect_wave_peaks():
    rng = np.random.default_rng(10)
    C = 4
    pos = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float
    )
    adj = 0.05 * rng.random((C, C))
    f = np.linspace(0.5, 5.0, 16)
    kx = np.linspace(-2.0, 2.0, 9)
    ky = np.linspace(-2.0, 2.0, 9)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k = np.stack([KX.ravel(), KY.ravel()], axis=1)

    S = bartlett_spectrum(adj, pos, f, k, decay=1.0)
    res = detect_wave_peaks(S, f, k, n_peaks=5)
    # Up to 5 peaks; speeds finite and positive.
    assert 1 <= len(res.freq) <= 5
    assert np.all(np.isfinite(res.speed))
    assert np.all(res.speed > 0)
    # Powers are strictly descending.
    diffs = np.diff(res.power)
    assert np.all(diffs <= 0)


# --------------------------------------------------------------------------
# B11 — WaveAnalysisResult is frozen / immutable
# --------------------------------------------------------------------------


def test_wave_analysis_result_is_frozen():
    arr = np.array([1.0])
    res = WaveAnalysisResult(
        freq=arr, kx=arr, ky=arr, power=arr, speed=arr, direction=arr
    )
    with pytest.raises((AttributeError, Exception)):
        res.freq = np.array([2.0])  # type: ignore[misc]
