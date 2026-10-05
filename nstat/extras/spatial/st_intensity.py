r"""Space-time kernel intensity estimation :math:`\hat\lambda(x,t)`.

Pure NumPy/SciPy.  A boundary-corrected product-kernel estimator of the
first-order intensity of a spatio-temporal point process, following the
kernel + edge-correction machinery of Diggle (2013), Chapter 7.  This is
original code implemented directly from the published equations below —
it is **not** a port of any external implementation.

Non-separable product-kernel estimator
---------------------------------------

At a query point :math:`(x, t)` with :math:`x \in \mathbb R^2`:

.. math::

    \hat\lambda(x, t) = \sum_i
        \frac{K_{h_s}(x - x_i)\,K_{h_t}(t - t_i)}
             {c_s(x)\,c_t(t)},

where :math:`K_{h_s}` is a 2-D product (Gaussian or Epanechnikov) spatial
kernel with bandwidth :math:`h_s`, :math:`K_{h_t}` a 1-D temporal kernel
with bandwidth :math:`h_t`, and :math:`c_s(x)`, :math:`c_t(t)` are
Diggle's edge-correction denominators — the mass of the kernel, centred
at the *query* point, that falls inside the observation window / interval
(Diggle 1985; Diggle 2013 §7.2.3):

.. math::

    c_s(x) = \int_W K_{h_s}(x - u)\,du, \qquad
    c_t(t) = \int_T K_{h_t}(t - s)\,ds.

For a rectangular window :math:`W = [x_{lo}, x_{hi}] \times [y_{lo},
y_{hi}]` and a Gaussian kernel, :math:`K_{h_s}` factorises into
independent per-axis Gaussians, so :math:`c_s(x)` is a *product of
standard-normal CDF differences*:

.. math::

    c_s(x) = \left[\Phi\!\Big(\tfrac{x_{hi}-x_1}{h_s}\Big)
                  - \Phi\!\Big(\tfrac{x_{lo}-x_1}{h_s}\Big)\right]
             \left[\Phi\!\Big(\tfrac{y_{hi}-x_2}{h_s}\Big)
                  - \Phi\!\Big(\tfrac{y_{lo}-x_2}{h_s}\Big)\right],

and :math:`c_t(t)` is the analogous single CDF difference on
:math:`T = [t_{lo}, t_{hi}]`.  The Epanechnikov kernel is handled the
same way with its own (polynomial, closed-form) antiderivative in place
of :math:`\Phi`.  Because the query-point denominator does not depend on
:math:`i`, it factors out of the sum over events — this is the
*uniform* edge-corrected kernel intensity estimator (spatstat's
``correction="uniform"``), a query-point-evaluated variant of Diggle's
(1985) per-event edge correction, generalised to the product space-time
kernel of Diggle (2013) Ch. 7.  Unlike Diggle's per-event correction it
is only *approximately* mass-conserving (:math:`\int \hat\lambda \approx
n`), as the Notes below and the companion test document.

Separable estimator
--------------------

When the process is assumed *separable*, :math:`\lambda(x,t) =
m(x)\,\mu(t)/N`, the marginals are estimated independently with the same
edge correction,

.. math::

    \hat m(x) = \sum_i \frac{K_{h_s}(x - x_i)}{c_s(x)}, \qquad
    \hat\mu(t) = \sum_i \frac{K_{h_t}(t - t_i)}{c_t(t)},

and combined as :math:`\hat\lambda(x,t) = \hat m(x)\,\hat\mu(t) / N`
(Diggle 2013, §7.2.1-7.2.2).

Bandwidths
----------

When ``bw_space``/``bw_time`` are not supplied, Silverman's (1986) normal
-reference rule of thumb is used: :math:`h = \bar\sigma\, n^{-1/6}` in 2-D
(the standard normal-reference constant is exactly 1 at :math:`d=2`) and
:math:`h = 1.06\,\sigma\, n^{-1/5}` in 1-D.  Fuentes-Santos, Gonzalez
-Manteiga & Mateu (2018) discuss the bias/variance trade-off of this
choice for first-order structure comparisons in point processes and
motivate cross-validated alternatives when the normal-reference rule is
too coarse for a given field.

This module produces the ``lambda_hat`` grid/callable consumed by the
sibling space-time :math:`K`-function estimator (SOIRS reweighting).

References
----------
- Diggle PJ (2013). *Statistical Analysis of Spatial and Spatio-Temporal
  Point Patterns*, 3rd ed. CRC Press, Chapter 7.
- Diggle PJ (1985). *A kernel method for smoothing point process data.*
  J. Royal Statistical Society, Series C 34(2):138-147.
- Fuentes-Santos I, Gonzalez-Manteiga W, Mateu J (2018). *A nonparametric
  test for the comparison of first-order structures of spatial point
  processes.* Spatial Statistics 25:44-63.
- Silverman BW (1986). *Density Estimation for Statistics and Data
  Analysis.* Chapman & Hall.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nstat.extras.spatial._kernels import epanechnikov

__all__ = ["STIntensityResult", "intensity_st_kde"]

_VALID_KERNELS = ("gaussian", "epanechnikov")


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------


def _validate_domain(domain) -> tuple[tuple[float, float], tuple[float, float]]:
    try:
        (xlo, xhi), (ylo, yhi) = domain
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"domain must be ((xlo, xhi), (ylo, yhi)); got {domain!r}"
        ) from exc
    xlo, xhi, ylo, yhi = float(xlo), float(xhi), float(ylo), float(yhi)
    if not (xhi > xlo):
        raise ValueError(f"domain x-range must have xhi > xlo; got ({xlo}, {xhi})")
    if not (yhi > ylo):
        raise ValueError(f"domain y-range must have yhi > ylo; got ({ylo}, {yhi})")
    return (xlo, xhi), (ylo, yhi)


def _validate_period(period) -> tuple[float, float]:
    try:
        tlo, thi = period
    except (TypeError, ValueError) as exc:
        raise ValueError(f"period must be (tlo, thi); got {period!r}") from exc
    tlo, thi = float(tlo), float(thi)
    if not (thi > tlo):
        raise ValueError(f"period must have thi > tlo; got ({tlo}, {thi})")
    return tlo, thi


def _validate_kernel(kernel: str) -> None:
    if kernel not in _VALID_KERNELS:
        raise ValueError(f"kernel must be one of {_VALID_KERNELS}; got {kernel!r}")


# ----------------------------------------------------------------------
# Kernel + edge-mass primitives (per-axis; combined by the caller)
# ----------------------------------------------------------------------


def _kernel_pdf_1d(kernel: str, u: np.ndarray) -> np.ndarray:
    """Standardised 1-D kernel density (integrates to 1 over its support)."""
    if kernel == "gaussian":
        return np.exp(-0.5 * u**2) / np.sqrt(2.0 * np.pi)
    return epanechnikov(u)


def _edge_mass_1d(
    kernel: str, centres: np.ndarray, lo: float, hi: float, h: float
) -> np.ndarray:
    r"""Diggle edge mass :math:`\int_{lo}^{hi} \tfrac1h K((s-c)/h)\,ds`.

    Vectorised over ``centres``.  For the Gaussian kernel this is a
    standard-normal CDF difference; for Epanechnikov it is the closed
    -form antiderivative of the compact-support polynomial kernel,
    :math:`F(u) = \tfrac12 + \tfrac34 u - \tfrac14 u^3` on :math:`[-1,1]`
    (and 0 / 1 outside).
    """
    centres = np.asarray(centres, dtype=float)
    u_lo = (lo - centres) / h
    u_hi = (hi - centres) / h
    if kernel == "gaussian":
        from scipy.stats import norm

        return norm.cdf(u_hi) - norm.cdf(u_lo)

    def _F(u: np.ndarray) -> np.ndarray:
        u = np.clip(u, -1.0, 1.0)
        return 0.5 + 0.75 * u - 0.25 * u**3

    return _F(u_hi) - _F(u_lo)


def _spatial_kernel_weights(
    kernel: str, query_xy: np.ndarray, events_xy: np.ndarray, h: float
) -> np.ndarray:
    """``(M, N)`` product-kernel density :math:`K_{h_s}(x_m - x_i)`."""
    dx = query_xy[:, 0:1] - events_xy[None, :, 0]
    dy = query_xy[:, 1:2] - events_xy[None, :, 1]
    kx = _kernel_pdf_1d(kernel, dx / h) / h
    ky = _kernel_pdf_1d(kernel, dy / h) / h
    return kx * ky


def _temporal_kernel_weights(
    kernel: str, query_t: np.ndarray, events_t: np.ndarray, h: float
) -> np.ndarray:
    """``(M, N)`` kernel density :math:`K_{h_t}(t_m - t_i)`."""
    dt = query_t[:, None] - events_t[None, :]
    return _kernel_pdf_1d(kernel, dt / h) / h


def _spatial_edge_mass(
    kernel: str,
    query_xy: np.ndarray,
    domain: tuple[tuple[float, float], tuple[float, float]],
    h: float,
) -> np.ndarray:
    """``(M,)`` edge-correction denominator :math:`c_s(x_m)`."""
    (xlo, xhi), (ylo, yhi) = domain
    cx = _edge_mass_1d(kernel, query_xy[:, 0], xlo, xhi, h)
    cy = _edge_mass_1d(kernel, query_xy[:, 1], ylo, yhi, h)
    return cx * cy


def _temporal_edge_mass(
    kernel: str, query_t: np.ndarray, period: tuple[float, float], h: float
) -> np.ndarray:
    """``(M,)`` edge-correction denominator :math:`c_t(t_m)`."""
    tlo, thi = period
    return _edge_mass_1d(kernel, query_t, tlo, thi, h)


# ----------------------------------------------------------------------
# Bandwidth defaults (Silverman 1986)
# ----------------------------------------------------------------------


def _silverman_bandwidth_2d(points: np.ndarray) -> float:
    r"""Silverman (1986) normal-reference bandwidth for an isotropic 2-D
    kernel: :math:`h = \bar\sigma\, n^{-1/6}` (the normal-reference
    constant :math:`(4/(d+2))^{1/(d+4)}` equals 1 at :math:`d=2`).  Uses
    the mean of the per-axis sample standard deviations as the isotropic
    scale :math:`\bar\sigma`.
    """
    n = points.shape[0]
    if n > 1:
        sigma = float(np.mean(np.std(points, axis=0, ddof=1)))
    else:
        sigma = 1.0
    sigma = float(max(sigma, np.finfo(float).eps))
    return sigma * n ** (-1.0 / 6.0)


def _silverman_bandwidth_1d(times: np.ndarray) -> float:
    r"""Classic Silverman (1986) 1-D rule of thumb:
    :math:`h = 1.06\,\sigma\, n^{-1/5}`.
    """
    n = times.shape[0]
    sigma = float(np.std(times, ddof=1)) if n > 1 else 1.0
    sigma = float(max(sigma, np.finfo(float).eps))
    return 1.06 * sigma * n ** (-1.0 / 5.0)


# ----------------------------------------------------------------------
# Grid construction
# ----------------------------------------------------------------------


def _cell_centres(lo: float, hi: float, G: int) -> np.ndarray:
    edges = np.linspace(lo, hi, G + 1)
    return 0.5 * (edges[:-1] + edges[1:])


# ----------------------------------------------------------------------
# Core evaluators
# ----------------------------------------------------------------------


def _evaluate_pairs(
    points: np.ndarray,
    times: np.ndarray,
    domain: tuple[tuple[float, float], tuple[float, float]],
    period: tuple[float, float],
    bw_space: float,
    bw_time: float,
    separable: bool,
    kernel: str,
    x: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Evaluate :math:`\\hat\\lambda` at paired ``(x_m, t_m)`` query points.

    ``x`` is broadcast to ``t``'s length (or vice versa) when one of the
    two has length 1; otherwise their lengths must match.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    t = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"x must be (m, 2) or (2,); got shape {x.shape}")

    if x.shape[0] == 1 and t.shape[0] > 1:
        x = np.repeat(x, t.shape[0], axis=0)
    elif t.shape[0] == 1 and x.shape[0] > 1:
        t = np.repeat(t, x.shape[0])
    if x.shape[0] != t.shape[0]:
        raise ValueError(
            "x and t must have matching length (or one of length 1); got "
            f"{x.shape[0]} and {t.shape[0]}"
        )

    n = points.shape[0]
    eps = np.finfo(float).eps
    Ws = _spatial_kernel_weights(kernel, x, points, bw_space)  # (M, N)
    Wt = _temporal_kernel_weights(kernel, t, times, bw_time)  # (M, N)
    c_s = _spatial_edge_mass(kernel, x, domain, bw_space)  # (M,)
    c_t = _temporal_edge_mass(kernel, t, period, bw_time)  # (M,)

    if separable:
        m_hat = Ws.sum(axis=1) / np.maximum(c_s, eps)
        mu_hat = Wt.sum(axis=1) / np.maximum(c_t, eps)
        return m_hat * mu_hat / n

    raw = np.einsum("mi,mi->m", Ws, Wt)
    return raw / np.maximum(c_s * c_t, eps)


def _evaluate_product_grid(
    points: np.ndarray,
    times: np.ndarray,
    grid_x: np.ndarray,
    grid_t: np.ndarray,
    domain: tuple[tuple[float, float], tuple[float, float]],
    period: tuple[float, float],
    bw_space: float,
    bw_time: float,
    separable: bool,
    kernel: str,
) -> np.ndarray:
    """Evaluate :math:`\\hat\\lambda` on the full ``grid_x x grid_t`` product.

    Vectorised via two smaller ``(G, N)`` weight matrices combined by a
    matrix product (non-separable) or an outer product (separable),
    avoiding the ``O(Gx*Gy*Gt*N)`` dense intermediate a naive triple loop
    would materialise.
    """
    n = points.shape[0]
    eps = np.finfo(float).eps
    Ws = _spatial_kernel_weights(kernel, grid_x, points, bw_space)  # (Gxy, N)
    Wt = _temporal_kernel_weights(kernel, grid_t, times, bw_time)  # (Gt, N)
    c_s = _spatial_edge_mass(kernel, grid_x, domain, bw_space)  # (Gxy,)
    c_t = _temporal_edge_mass(kernel, grid_t, period, bw_time)  # (Gt,)

    if separable:
        m_hat = Ws.sum(axis=1) / np.maximum(c_s, eps)  # (Gxy,)
        mu_hat = Wt.sum(axis=1) / np.maximum(c_t, eps)  # (Gt,)
        return np.outer(mu_hat, m_hat) / n  # (Gt, Gxy)

    raw = Ws @ Wt.T  # (Gxy, Gt)
    denom = np.outer(c_s, c_t)  # (Gxy, Gt)
    return (raw / np.maximum(denom, eps)).T  # (Gt, Gxy)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class STIntensityResult:
    """Fitted space-time kernel intensity estimate (plain NumPy).

    Attributes
    ----------
    grid_x
        ``(Gx*Gy, 2)`` spatial grid centres (row-major, ``x`` fastest —
        matches :func:`nstat.extras.spatial._kernels.make_grid`).
    grid_t
        ``(Gt,)`` time grid centres.
    intensity
        ``(Gt, Gx*Gy)`` :math:`\\hat\\lambda` evaluated on the product
        grid; row ``k`` is the spatial rate map at ``grid_t[k]``.
    bw_space
        Spatial bandwidth :math:`h_s` used (given or Silverman default).
    bw_time
        Temporal bandwidth :math:`h_t` used (given or Silverman default).
    separable
        Whether the separable ``m(x) mu(t) / N`` factorisation was used.
    domain
        ``((xlo, xhi), (ylo, yhi))`` spatial window (stored for
        :meth:`evaluate`'s edge correction).
    period
        ``(tlo, thi)`` temporal interval (stored for :meth:`evaluate`'s
        edge correction).
    kernel
        ``"gaussian"`` or ``"epanechnikov"``.
    points
        The original event coordinates/times, retained so
        :meth:`evaluate` can be called at arbitrary query points (not
        just grid interpolation).
    times
        See ``points``.
    """

    grid_x: np.ndarray
    grid_t: np.ndarray
    intensity: np.ndarray
    bw_space: float
    bw_time: float
    separable: bool
    domain: tuple[tuple[float, float], tuple[float, float]]
    period: tuple[float, float]
    kernel: str
    points: np.ndarray
    times: np.ndarray

    def evaluate(self, x: np.ndarray, t: np.ndarray | float) -> np.ndarray:
        r"""Evaluate :math:`\hat\lambda` at arbitrary ``(x, t)`` query points.

        Parameters
        ----------
        x
            ``(m, 2)`` spatial query points, or ``(2,)`` for a single
            point (broadcast against ``t``).
        t
            ``(m,)`` query times, or a scalar / length-1 array
            (broadcast against ``x``).  When both ``x`` and ``t`` have
            more than one row, their lengths must match — the pair
            ``(x[k], t[k])`` is evaluated for each ``k``.

        Returns
        -------
        np.ndarray
            ``(m,)`` :math:`\hat\lambda(x_k, t_k)`.

        Notes
        -----
        This is the exact estimator formula re-evaluated at the query
        points (not a grid interpolation of :attr:`intensity`), so it
        agrees with :attr:`intensity` only up to floating-point
        round-off when queried at the stored grid centres.  This is the
        method the sibling space-time :math:`K`-function estimator calls
        to reweight observed event pairs by
        :math:`1/(\hat\lambda(x_i,t_i)\hat\lambda(x_j,t_j))`.
        """
        return _evaluate_pairs(
            self.points,
            self.times,
            self.domain,
            self.period,
            self.bw_space,
            self.bw_time,
            self.separable,
            self.kernel,
            x,
            t,
        )


def intensity_st_kde(
    points: np.ndarray,
    times: np.ndarray,
    *,
    domain: tuple[tuple[float, float], tuple[float, float]],
    period: tuple[float, float] | None = None,
    bw_space: float | None = None,
    bw_time: float | None = None,
    separable: bool = False,
    grid: tuple[int, int, int] = (40, 40, 40),
    kernel: str = "gaussian",
) -> STIntensityResult:
    r"""Boundary-corrected space-time kernel intensity estimate.

    Implements the non-separable product-kernel estimator or, when
    ``separable=True``, the marginal-product estimator of Diggle (2013)
    Chapter 7 — see the module docstring for the full equations.

    Parameters
    ----------
    points
        ``(n, 2)`` event coordinates.
    times
        ``(n,)`` event times, aligned with ``points``.
    domain
        ``((xlo, xhi), (ylo, yhi))`` rectangular spatial observation
        window.
    period
        ``(tlo, thi)`` temporal observation interval.  ``None``
        (default) uses ``(times.min(), times.max())``.  Despite the
        parameter name (kept for interface compatibility), this is the
        temporal analogue of ``domain`` — an observation interval used
        for the boundary correction and the default grid range — **not**
        a periodic (wrap-around) boundary condition; no cyclic smoothing
        is applied.
    bw_space, bw_time
        Kernel bandwidths :math:`h_s`, :math:`h_t`.  ``None`` (default)
        uses Silverman's (1986) rule of thumb (see module docstring).
    separable
        If ``True``, fit the marginal spatial/temporal intensities
        independently and combine as :math:`\hat m(x)\hat\mu(t)/N`.  If
        ``False`` (default), fit the non-separable product-kernel
        estimator.
    grid
        ``(Gx, Gy, Gt)`` number of evaluation cells per axis.
    kernel
        ``"gaussian"`` (default, isotropic 2-D / standard-normal 1-D) or
        ``"epanechnikov"`` (compact-support product kernel).

    Returns
    -------
    STIntensityResult

    Raises
    ------
    ValueError
        If ``points``/``times`` are empty or misaligned, ``domain``/
        ``period`` are malformed, ``kernel`` is not recognised, a
        supplied bandwidth is non-positive, or a ``grid`` entry is
        ``< 1``.

    Notes
    -----
    *Confidence: high* on the estimator algebra (Diggle 1985; Diggle
    2013 Ch. 7 is the standard reference); the edge-corrected estimator
    is only *approximately* mass-conserving
    (:math:`\iint_{W\times T} \hat\lambda \approx n`) because the
    correction denominator is evaluated at the query point rather than
    at each event — exact for events comfortably inside the window,
    approximate near the boundary (asserted in the companion tests).

    References
    ----------
    - Diggle PJ (2013). *Statistical Analysis of Spatial and
      Spatio-Temporal Point Patterns*, 3rd ed. CRC Press, Chapter 7.
    - Diggle PJ (1985). *A kernel method for smoothing point process
      data.* JRSS-C 34(2):138-147.
    - Fuentes-Santos I, Gonzalez-Manteiga W, Mateu J (2018). *A
      nonparametric test for the comparison of first-order structures of
      spatial point processes.* Spatial Statistics 25:44-63.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    times = np.asarray(times, dtype=float).ravel()

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must be (n, 2); got shape {points.shape}")
    if points.shape[0] != times.shape[0]:
        raise ValueError(
            "points and times must align; got "
            f"{points.shape[0]} points and {times.shape[0]} times"
        )
    n = points.shape[0]
    if n == 0:
        raise ValueError("points/times must contain at least one event")

    _validate_kernel(kernel)
    domain = _validate_domain(domain)
    period = _validate_period(period) if period is not None else (
        float(times.min()), float(times.max())
    )

    try:
        Gx, Gy, Gt = (int(g) for g in grid)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"grid must be a 3-tuple (Gx, Gy, Gt); got {grid!r}") from exc
    if Gx < 1 or Gy < 1 or Gt < 1:
        raise ValueError(f"grid sizes must all be >= 1; got {grid!r}")

    if bw_space is None:
        bw_space = _silverman_bandwidth_2d(points)
    else:
        bw_space = float(bw_space)
        if not (bw_space > 0):
            raise ValueError(f"bw_space must be positive; got {bw_space}")

    if bw_time is None:
        bw_time = _silverman_bandwidth_1d(times)
    else:
        bw_time = float(bw_time)
        if not (bw_time > 0):
            raise ValueError(f"bw_time must be positive; got {bw_time}")

    (xlo, xhi), (ylo, yhi) = domain
    x_centres = _cell_centres(xlo, xhi, Gx)
    y_centres = _cell_centres(ylo, yhi, Gy)
    t_centres = _cell_centres(period[0], period[1], Gt)

    mesh_x, mesh_y = np.meshgrid(x_centres, y_centres, indexing="xy")
    grid_x = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])

    intensity = _evaluate_product_grid(
        points, times, grid_x, t_centres, domain, period,
        bw_space, bw_time, separable, kernel,
    )

    return STIntensityResult(
        grid_x=grid_x,
        grid_t=t_centres,
        intensity=intensity,
        bw_space=bw_space,
        bw_time=bw_time,
        separable=bool(separable),
        domain=domain,
        period=period,
        kernel=kernel,
        points=points,
        times=times,
    )
