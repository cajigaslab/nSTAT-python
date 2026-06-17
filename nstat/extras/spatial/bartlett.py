r"""Bartlett (spectral) density of a 2-D stationary point process.

Hankel transform of the pair correlation deviation :math:`g(r) - 1` —
the spatial analogue of the temporal Bartlett spectrum already provided
by :func:`nstat.extras.spatial.hawkes_bridge.bartlett_spectrum`.  For
an isotropic stationary point process in the plane, the *reduced
second-moment Bartlett density* is the Hankel-zero transform

.. math::

    S(k) = 2\pi \int_0^\infty r\,\bigl(g(r) - 1\bigr)\,
           J_0(kr)\,\mathrm{d}r ,

which is the Fourier transform of the radial reduced second moment.
For a log-Gaussian Cox process (LGCP) driven by a Gaussian log-rate
field of covariance :math:`C(r)`, the Bartlett density is the
Hankel transform of :math:`\exp(C(r)) - 1` (Møller, Syversveen &
Waagepetersen 1998).

This estimator takes the *empirical* pair correlation
:math:`\hat g(r)` on a finite lag grid (typically from
:func:`nstat.extras.spatial.spatial_gof.pair_correlation`) and returns
the discretised Hankel transform.  The estimate is accurate over the
*body* of the wavenumber grid; the top decile near
:math:`k_{\max} \sim \pi / \Delta r` is sensitive to the lag-grid
truncation (Stein 1999 §3) and is excluded by default for diagnostic
comparisons.

References
----------
- Møller J, Syversveen AR, Waagepetersen RP (1998). *Log Gaussian Cox
  processes.* Scandinavian Journal of Statistics 25(3):451-482.
- Stein ML (1999). *Interpolation of Spatial Data: Some Theory for
  Kriging.* Springer, §3 (spectral methods for stationary processes).
- Bartlett MS (1964). *The spectral analysis of two-dimensional point
  processes.* Biometrika 51(3-4):299-311.
"""
from __future__ import annotations

import numpy as np
from scipy.special import j0


def _trapezoid(y, x):
    """NumPy 1.x/2.x portability shim for the trapezoid rule.

    NumPy >= 2.0 renamed ``np.trapz`` to ``np.trapezoid``.  Older
    versions only have ``np.trapz``.  This shim avoids the deprecation
    warning on NumPy 2.x without breaking 1.x.
    """
    trap = getattr(np, "trapezoid", None) or np.trapz
    return trap(y, x)


def bartlett_density_from_pcf(
    r_grid: np.ndarray,
    g_of_r: np.ndarray,
    k_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Bartlett spectral density from an empirical pair correlation.

    Discretised Hankel-zero transform of :math:`(g(r) - 1)` on the lag
    grid ``r_grid``,

    .. math::

        \hat S(k) = 2\pi \sum_m r_m\,(g(r_m) - 1)\,J_0(k\,r_m)\,
                    \Delta r_m ,

    evaluated by :func:`numpy.trapezoid` (NumPy 2.x) /
    :func:`numpy.trapz` (NumPy 1.x).

    Parameters
    ----------
    r_grid
        Strictly positive, strictly increasing lag values, shape
        ``(L,)``.
    g_of_r
        Empirical pair correlation at each lag, shape ``(L,)``.
    k_grid
        Wavenumbers at which to evaluate :math:`\hat S(k)`.  Defaults
        to 64 log-spaced wavenumbers from :math:`\pi / r_{\max}` to
        :math:`\pi / \Delta r_{\min}` — the Nyquist-like body of the
        accessible spectrum given the lag grid.

    Returns
    -------
    k_grid : np.ndarray
        The wavenumbers (the supplied grid, or the default 64-point
        log-spaced one).  Shape ``(N_k,)``.
    S : np.ndarray
        The Bartlett spectral density at each wavenumber.  Shape
        ``(N_k,)``.

    Raises
    ------
    ValueError
        If ``r_grid`` is not strictly positive / strictly increasing,
        if ``g_of_r`` does not align with ``r_grid``, or if any
        wavenumber in ``k_grid`` is non-positive.

    Notes
    -----
    *Confidence: high* on the body of the wavenumber grid (matches the
    LGCP closed-form within a few percent).  The top decile near
    :math:`k_{\max}` is sensitive to the finite-lag truncation (Stein
    1999 §3); avoid comparing to a closed form there.
    """
    r = np.asarray(r_grid, dtype=float).ravel()
    g = np.asarray(g_of_r, dtype=float).ravel()
    if r.shape != g.shape:
        raise ValueError(
            f"r_grid and g_of_r must align; got {r.shape} vs {g.shape}"
        )
    if r.size < 2:
        raise ValueError("r_grid must have at least 2 points")
    if np.any(r <= 0):
        raise ValueError("r_grid must be strictly positive")
    if np.any(np.diff(r) <= 0):
        raise ValueError("r_grid must be strictly increasing")

    if k_grid is None:
        k_lo = float(np.pi / r[-1])
        # Smallest accessible wavelength ~ 2 * dr_min; k_hi = pi / dr_min.
        dr_min = float(np.diff(r).min())
        k_hi = float(np.pi / max(dr_min, 1e-12))
        if k_hi <= k_lo:
            k_hi = 2.0 * k_lo
        k = np.logspace(np.log10(k_lo), np.log10(k_hi), 64)
    else:
        k = np.asarray(k_grid, dtype=float).ravel()
        if k.size == 0:
            raise ValueError("k_grid must be non-empty")
        if np.any(k <= 0):
            raise ValueError("k_grid must be strictly positive")

    # S(k) = 2 pi int_0^inf r (g(r) - 1) J0(k r) dr, vectorised over k
    # via the (N_k, L) outer-product integrand.
    integrand = (k[:, None] * r[None, :])  # (N_k, L)
    integrand = j0(integrand)
    integrand = integrand * (r * (g - 1.0))[None, :]
    S = 2.0 * np.pi * _trapezoid(integrand, r)
    return k, np.asarray(S, dtype=float)


__all__ = [
    "bartlett_density_from_pcf",
]
