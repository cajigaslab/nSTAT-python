r"""Spatiotemporal wave analysis for Hawkes triggering kernels.

Diagnostics for a fitted multivariate Hawkes process with exponential
triggering kernels — namely, the Python-only companion to the §6.C wave
analysis on top of :func:`nstat.extras.spatial.hawkes_bridge.bartlett_spectrum`.

The kernel reconstruction
:math:`\hat\varphi_{cc'}(\tau) = A_{cc'}\,e^{-\beta\,\tau}\,1_{\tau\ge 0}`
is the parametric form returned by :func:`fit_hawkes_exp`; the wave-peak
detector locates ridges :math:`|\mathbf{k}\,\|\mathbf{v}\|| = 2\pi f` of the
Bartlett spectrum at which the process exhibits a coherent propagating
component (Bacry-Mastromatteo-Muzy 2015; Daley & Vere-Jones 2003 §8.4).

Pure NumPy/SciPy — no optional dependency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WaveAnalysisResult:
    """Top-:math:`N` wave-vector peaks of a Bartlett spectrum.

    All fields are 1-D arrays of length ``n`` (the number of peaks
    successfully accepted by the greedy non-maximum-suppression).

    Attributes
    ----------
    freq
        Frequencies of the accepted peaks (Hz).
    kx, ky
        Wave-vector components in rad / (position-unit).
    power
        Peak power at the accepted ``(freq, k)`` cells (matches the
        ``return_complex=False`` output of ``bartlett_spectrum``).
    speed
        Phase speed ``2*pi*freq / |k|`` (position-unit / s).
    direction
        Propagation direction ``atan2(ky, kx)`` in radians.
    """

    freq: np.ndarray
    kx: np.ndarray
    ky: np.ndarray
    power: np.ndarray
    speed: np.ndarray
    direction: np.ndarray


def reconstruct_kernel(
    adjacency: np.ndarray,
    decays: float | np.ndarray,
    tau_grid: np.ndarray,
) -> np.ndarray:
    r"""Reconstruct the exponential-family pairwise triggering kernel.

    For the exponential parametric form,

    .. math:: \varphi_{c c'}(\tau) = A_{cc'}\,e^{-\beta_{cc'}\,\tau}\,1_{\tau \ge 0}.

    Non-exponential families are **out of scope** for this builder; if
    you need a non-parametric reconstruction, use the ``tick``
    ``HawkesBasisKernels`` estimator and pass the basis output instead.

    Parameters
    ----------
    adjacency
        ``(C, C)`` integrated kernel norms :math:`A_{cc'}`.
    decays
        Scalar :math:`\beta` (shared across pairs) or ``(C, C)`` array
        of per-pair decays.  Negative entries raise ``ValueError``.
    tau_grid
        ``(Nt,)`` non-negative lags (seconds).

    Returns
    -------
    np.ndarray
        ``(C, C, Nt)`` kernel values.

    Raises
    ------
    ValueError
        On shape mismatch, negative ``tau_grid``, or negative decays.
    """
    adj = np.asarray(adjacency, dtype=float)
    tau = np.asarray(tau_grid, dtype=float)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(
            f"adjacency must be (C, C); got shape {adj.shape}"
        )
    C = adj.shape[0]
    if tau.ndim != 1:
        raise ValueError(f"tau_grid must be 1-D; got shape {tau.shape}")
    if np.any(tau < 0):
        raise ValueError("tau_grid must be non-negative for a causal kernel")

    decays_arr = np.asarray(decays, dtype=float)
    if decays_arr.ndim == 0:
        if float(decays_arr) < 0:
            raise ValueError("decays must be non-negative")
        decays_full = np.broadcast_to(decays_arr, (C, C)).astype(float)
    elif decays_arr.shape == (C, C):
        if np.any(decays_arr < 0):
            raise ValueError("decays array contains negative entries")
        decays_full = decays_arr
    else:
        raise ValueError(
            f"decays must be scalar or (C, C) with C={C}; got shape {decays_arr.shape}"
        )

    # phi[c1, c2, t] = adj[c1, c2] * exp(-decays_full[c1, c2] * tau[t])
    return adj[:, :, None] * np.exp(
        -decays_full[:, :, None] * tau[None, None, :]
    )


def detect_wave_peaks(
    spectrum: np.ndarray,
    freq_grid: np.ndarray,
    wave_vector_grid: np.ndarray,
    *,
    n_peaks: int = 3,
    min_separation_bins: int = 1,
) -> WaveAnalysisResult:
    r"""Locate the top-:math:`N` wave-vector peaks of a Bartlett spectrum.

    Uses a greedy descending-power sort with a Chebyshev-style
    non-maximum suppression: a candidate ``(f_idx, k_idx)`` is accepted
    only if its frequency index AND its (masked) wave-vector index are
    each at least ``min_separation_bins`` away from every previously
    accepted peak (this excludes both close-by harmonics in :math:`f`
    and the antipodal :math:`\mathbf{k} \mapsto -\mathbf{k}` redundancy
    when the grid is symmetric).

    Wave vectors with :math:`|\mathbf{k}| = 0` are masked out (the
    spatial DC component carries no propagation information).

    Parameters
    ----------
    spectrum
        ``(Nf, Nk)`` non-negative real power (e.g. the
        ``return_complex=False`` output of
        :func:`~nstat.extras.spatial.bartlett_spectrum`).
    freq_grid
        ``(Nf,)`` frequencies in Hz.
    wave_vector_grid
        ``(Nk, 2)`` wave vectors in rad / (position-unit).
    n_peaks
        Number of peaks to return.
    min_separation_bins
        Minimum index-separation along EACH axis between accepted peaks.

    Returns
    -------
    WaveAnalysisResult
        Up to ``n_peaks`` accepted peaks; arrays may be shorter than
        ``n_peaks`` if non-maximum suppression exhausted the grid.

    Raises
    ------
    ValueError
        On shape mismatch or ``n_peaks < 1``.
    """
    S = np.asarray(spectrum, dtype=float)
    f = np.asarray(freq_grid, dtype=float)
    k_full = np.asarray(wave_vector_grid, dtype=float)

    if S.ndim != 2:
        raise ValueError(f"spectrum must be 2-D (Nf, Nk); got shape {S.shape}")
    if f.ndim != 1 or f.shape[0] != S.shape[0]:
        raise ValueError(
            f"freq_grid must be 1-D of length Nf={S.shape[0]}; "
            f"got shape {f.shape}"
        )
    if k_full.ndim != 2 or k_full.shape[1] != 2 or k_full.shape[0] != S.shape[1]:
        raise ValueError(
            f"wave_vector_grid must be (Nk, 2) with Nk={S.shape[1]}; "
            f"got shape {k_full.shape}"
        )
    if n_peaks < 1:
        raise ValueError(f"n_peaks must be >= 1; got {n_peaks}")
    if min_separation_bins < 0:
        raise ValueError(
            f"min_separation_bins must be >= 0; got {min_separation_bins}"
        )

    # Mask DC (|k| == 0) wave vectors.
    k_norm = np.linalg.norm(k_full, axis=1)
    keep_k = k_norm > 0.0
    if not np.any(keep_k):
        empty = np.empty((0,), dtype=float)
        return WaveAnalysisResult(
            freq=empty, kx=empty, ky=empty,
            power=empty, speed=empty, direction=empty,
        )

    S_masked = S[:, keep_k]                          # (Nf, Nk_kept)
    k_kept = k_full[keep_k]                           # (Nk_kept, 2)
    k_norm_kept = k_norm[keep_k]                      # (Nk_kept,)

    # Greedy descending-power sort over the full (freq x k_kept) grid.
    flat = S_masked.ravel()
    order = np.argsort(-flat, kind="stable")          # descending power

    Nf, Nk_kept = S_masked.shape
    accepted_f: list[int] = []
    accepted_k: list[int] = []

    for idx in order:
        if len(accepted_f) >= n_peaks:
            break
        f_idx = int(idx // Nk_kept)
        k_idx = int(idx % Nk_kept)
        # Non-maximum suppression: reject if too close to any accepted peak
        # on BOTH axes.
        too_close = False
        for af, ak in zip(accepted_f, accepted_k):
            if (
                abs(f_idx - af) < min_separation_bins
                and abs(k_idx - ak) < min_separation_bins
            ):
                too_close = True
                break
        if too_close:
            continue
        accepted_f.append(f_idx)
        accepted_k.append(k_idx)

    if not accepted_f:
        empty = np.empty((0,), dtype=float)
        return WaveAnalysisResult(
            freq=empty, kx=empty, ky=empty,
            power=empty, speed=empty, direction=empty,
        )

    accepted_f_arr = np.asarray(accepted_f, dtype=int)
    accepted_k_arr = np.asarray(accepted_k, dtype=int)
    freq_out = f[accepted_f_arr]
    k_out = k_kept[accepted_k_arr]
    kx = k_out[:, 0]
    ky = k_out[:, 1]
    power_out = S_masked[accepted_f_arr, accepted_k_arr]
    # speed = phase speed |omega| / |k| = 2 pi f / |k|.
    speed = 2.0 * np.pi * freq_out / k_norm_kept[accepted_k_arr]
    direction = np.arctan2(ky, kx)

    return WaveAnalysisResult(
        freq=freq_out,
        kx=kx,
        ky=ky,
        power=power_out,
        speed=speed,
        direction=direction,
    )


__all__ = ["WaveAnalysisResult", "reconstruct_kernel", "detect_wave_peaks"]
