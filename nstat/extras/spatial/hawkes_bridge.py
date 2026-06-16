r"""Multivariate Hawkes process bridge via ``tick`` (optional).

Lazy bridge to the Hawkes estimators of `tick
<https://github.com/X-DataInitiative/tick>`_ (Bacry et al. 2018, JMLR 18;
BSD-3) for self / mutually-exciting temporal point processes —
functional-connectivity and triggering-kernel estimation for a
multi-electrode ensemble.

Install
-------

.. code-block:: bash

    pip install nstat-toolbox[hawkes]

The fitted :math:`C \times C` triggering matrix is the cell-averaged
continuous triggering kernel: a propagating kernel
:math:`\varphi(\tau, r) = \psi(\tau)\,\delta(r - v\tau)` gives
off-diagonal entries peaked at :math:`\tau \approx \|r_{cc'}\|/\|v\|`,
the signature of a traveling wave at speed :math:`v`.  This bridge is
intentionally thin: it wraps ``tick``'s exponential-kernel MLE and
returns plain NumPy.

.. note::

   Estimating inhibition (negative triggering) is hard and the
   exponential-kernel MLE here assumes excitation; see Bonnet (2023) for
   the inhibition-kernel difficulty.  *Confidence: high for excitatory
   exponential kernels; lower for inhibition.*
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nstat.extras._lazy import require_optional


@dataclass(frozen=True)
class HawkesFit:
    """Fitted multivariate Hawkes parameters (plain NumPy).

    Attributes
    ----------
    baseline
        ``(C,)`` background rates :math:`\\mu_c`.
    adjacency
        ``(C, C)`` branching / triggering matrix (integrated kernel
        norms) — entry ``[c, c']`` is the expected number of events in
        channel ``c`` triggered by one event in channel ``c'``.
    decays
        The exponential decay(s) used in the parametric kernel.
    n_channels
        Number of channels ``C``.
    """

    baseline: np.ndarray
    adjacency: np.ndarray
    decays: np.ndarray
    n_channels: int


def fit_hawkes_exp(
    event_times,
    decay: float | np.ndarray = 1.0,
    *,
    penalty: str = "l2",
    C: float = 1e3,
    max_iter: int = 200,
) -> HawkesFit:
    r"""Fit a multivariate Hawkes process with exponential kernels via ``tick``.

    Parameters
    ----------
    event_times
        Sequence of length ``C`` — one increasing 1-D array of event
        times (seconds) per channel.
    decay
        Exponential decay :math:`\beta` (scalar, shared across channels).
    penalty
        Regularization passed to ``tick`` (``"l2"`` default; ``"l1"`` for
        sparse connectivity à la ADM4).
    C
        Inverse regularization strength (``tick`` convention).
    max_iter
        Solver iterations.

    Returns
    -------
    HawkesFit

    Raises
    ------
    ImportError
        If ``tick`` is not installed (with the ``[hawkes]`` install hint).
    """
    tick_hawkes = require_optional("tick.hawkes", install_key="hawkes")
    HawkesExpKern = tick_hawkes.HawkesExpKern

    events = [np.asarray(e, dtype=float) for e in event_times]
    n_channels = len(events)
    learner = HawkesExpKern(
        decays=float(np.asarray(decay).ravel()[0]),
        penalty=penalty,
        C=C,
        max_iter=max_iter,
    )
    learner.fit(events)
    return HawkesFit(
        baseline=np.asarray(learner.baseline, dtype=float),
        adjacency=np.asarray(learner.adjacency, dtype=float),
        decays=np.atleast_1d(np.asarray(decay, dtype=float)),
        n_channels=n_channels,
    )


__all__ = ["HawkesFit", "fit_hawkes_exp"]


# --------------------------------------------------------------------------
# Bartlett spectrum — pure NumPy/SciPy (no ``tick`` dependency).
# --------------------------------------------------------------------------
# Pure-NumPy/SciPy implementation: kept in this module because the input
# (a fitted ``triggering_matrix``) is the natural output of ``fit_hawkes_exp``
# and the §6.C.1 Bartlett spectrum is the canonical wave-vector diagnostic
# for a propagating triggering kernel.  This function does NOT require
# ``tick`` and is safe to eagerly re-export from
# ``nstat.extras.spatial.__init__`` (see comments there).
import warnings  # noqa: E402  -- intentionally local to the bartlett block


def bartlett_spectrum(
    triggering_matrix: np.ndarray,
    electrode_positions: np.ndarray,
    freq_grid: np.ndarray,
    wave_vector_grid: np.ndarray,
    *,
    decay: float | np.ndarray = 1.0,
    return_complex: bool = False,
) -> np.ndarray:
    r"""Bartlett (frequency × wave-vector) spectrum of a Hawkes triggering kernel.

    Implements the §8.4 Bartlett spectrum of Daley & Vere-Jones (2003) for a
    multivariate Hawkes process whose pairwise triggering kernel is
    :math:`\varphi_{cc'}(\tau) = A_{cc'} \beta\,e^{-\beta\tau}\,1_{\tau\ge 0}`
    (the parametric form returned by :func:`fit_hawkes_exp`).  The spectral
    density at frequency :math:`f` and wave vector :math:`\mathbf{k}` is

    .. math::

        S(f, \mathbf{k}) = \frac{1}{\beta + i\,2\pi f}\,
        \sum_{c,c'} A_{cc'}\,e^{-i\,\mathbf{k}\cdot(\mathbf{r}_c - \mathbf{r}_{c'})}.

    A coherent propagating component
    :math:`A_{cc'} \propto \exp(-\beta\,\mathbf{r}_{cc'}\!\cdot\!\mathbf{v}/\|\mathbf{v}\|^2)`
    produces a ridge :math:`|\mathbf{k}\,\|\mathbf{v}\|| = 2\pi f` whose
    direction is :math:`\hat{\mathbf{v}}` (Bacry-Mastromatteo-Muzy 2015).

    Parameters
    ----------
    triggering_matrix
        ``(C, C)`` adjacency / integrated triggering matrix (e.g.
        :attr:`HawkesFit.adjacency`).
    electrode_positions
        ``(C, 2)`` planar electrode coordinates.
    freq_grid
        ``(Nf,)`` frequencies in Hz.
    wave_vector_grid
        ``(Nk, 2)`` wave vectors in rad / (position-unit).  Speed is
        recovered as ``2*pi*f / |k|`` with units ``position-unit / s``.
    decay
        Exponential decay :math:`\beta` (scalar) or per-pair ``(C, C)``
        array of decays.  Negative entries raise ``ValueError``.
    return_complex
        If ``False`` (default), return the **real, non-negative power**
        :math:`|S(f, \mathbf{k})|^2`.  If ``True``, return the complex
        spectrum so callers can take phases / cross-spectra.

    Returns
    -------
    np.ndarray
        ``(Nf, Nk)`` power (real, ``float``) if ``return_complex=False``,
        else ``(Nf, Nk)`` complex spectrum.

    Raises
    ------
    ValueError
        On shape mismatch or negative ``decay``.

    Warns
    -----
    UserWarning
        If the adjacency matrix has spectral radius :math:`\ge 1` (the
        Hawkes process is not stationary; the Bartlett formula assumes
        stationarity).

    References
    ----------
    Daley DJ, Vere-Jones D (2003). *An Introduction to the Theory of Point
    Processes*, Vol. I, §8.4.  Bacry E, Mastromatteo I, Muzy J-F (2015).
    *Hawkes processes in finance.* Market Microstructure and Liquidity
    1(1):1550005.  Hansen NR, Reynaud-Bouret P, Rivoirard V (2015).
    *Lasso and probabilistic inequalities for multivariate point processes.*
    Bernoulli 21(1):83.
    """
    adj = np.asarray(triggering_matrix, dtype=float)
    pos = np.asarray(electrode_positions, dtype=float)
    f = np.asarray(freq_grid, dtype=float)
    k = np.asarray(wave_vector_grid, dtype=float)

    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(
            f"triggering_matrix must be (C, C); got shape {adj.shape}"
        )
    C = adj.shape[0]
    if pos.shape != (C, 2):
        raise ValueError(
            f"electrode_positions must be (C, 2) with C={C}; got shape {pos.shape}"
        )
    if f.ndim != 1:
        raise ValueError(f"freq_grid must be 1-D; got shape {f.shape}")
    if k.ndim != 2 or k.shape[1] != 2:
        raise ValueError(
            f"wave_vector_grid must be (Nk, 2); got shape {k.shape}"
        )

    decay_arr = np.asarray(decay, dtype=float)
    if decay_arr.ndim == 0:
        decay_scalar: float | None = float(decay_arr)
        if decay_scalar < 0:
            raise ValueError(f"decay must be non-negative; got {decay_scalar}")
        decay_full = None
    else:
        if decay_arr.shape != (C, C):
            raise ValueError(
                f"decay array must be (C, C) with C={C}; got shape {decay_arr.shape}"
            )
        if np.any(decay_arr < 0):
            raise ValueError("decay array contains negative entries")
        decay_scalar = None
        decay_full = decay_arr

    Nf = f.shape[0]
    Nk = k.shape[0]

    if not np.any(adj):
        dtype = complex if return_complex else float
        return np.zeros((Nf, Nk), dtype=dtype)

    rho = float(np.max(np.abs(np.linalg.eigvals(adj))))
    if rho >= 1.0:
        warnings.warn(
            f"Hawkes adjacency has spectral radius {rho:.3f} >= 1.0; "
            "stationarity assumption violated",
            UserWarning,
            stacklevel=2,
        )

    # r[i, j, d] = pos[i, d] - pos[j, d]  (so r[i, j] = r_i - r_j)
    r = pos[:, None, :] - pos[None, :, :]                # (C, C, 2)
    phase = np.einsum("kd,ijd->kij", k, r)                # (Nk, C, C)
    exp_phase = np.exp(-1j * phase)                        # (Nk, C, C) complex

    if decay_scalar is not None:
        # Factorized path: kernel = adj * exp(-i k . r)  is freq-independent;
        # the frequency factor G(f) = 1 / (decay + i 2 pi f) multiplies across.
        G = 1.0 / (decay_scalar + 1j * 2.0 * np.pi * f)   # (Nf,) complex
        S_k = np.einsum("ij,kij->k", adj, exp_phase)      # (Nk,) complex
        complex_spec = G[:, None] * S_k[None, :]          # (Nf, Nk) complex
    else:
        # Loop path: per-pair decay couples to frequency; chunk to bound memory.
        complex_spec = np.empty((Nf, Nk), dtype=complex)
        # Memory estimate for the full (Nf, C, C) G_mat plus (Nf, Nk) outputs:
        # 16 bytes/complex * Nf * (C*C + Nk).  Slab over freq to keep < 32 MB.
        bytes_per_freq = 16 * (C * C + Nk)
        slab = max(1, min(Nf, (32 * 1024 * 1024) // max(bytes_per_freq, 1)))
        # Pre-flatten exp_phase against adj weights once per slab.
        for start in range(0, Nf, slab):
            stop = min(start + slab, Nf)
            f_slab = f[start:stop]                          # (Ns,)
            # G_mat[s, c1, c2] = 1 / (decay[c1, c2] + i 2 pi f_slab[s])
            G_mat = 1.0 / (
                decay_full[None, :, :] + 1j * 2.0 * np.pi * f_slab[:, None, None]
            )                                                # (Ns, C, C)
            weighted = adj[None, :, :] * G_mat               # (Ns, C, C) complex
            # S_fk[s, k] = sum_{ij} weighted[s, i, j] * exp_phase[k, i, j]
            S_fk = np.einsum("sij,kij->sk", weighted, exp_phase)
            complex_spec[start:stop, :] = S_fk

    if return_complex:
        return complex_spec.astype(complex)
    return (np.abs(complex_spec) ** 2).astype(float)


__all__ = ["HawkesFit", "fit_hawkes_exp", "bartlett_spectrum"]
