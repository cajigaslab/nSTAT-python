r"""Marked-pattern second-order goodness-of-fit (pure NumPy/SciPy).

Diagnostics that test whether the marks attached to a spatial point
pattern are independent of the geometry of the pattern — the standard
mark correlation function and the mark variogram.

Two complementary summaries are provided:

- :func:`mark_correlation` — the kernel-smoothed mark correlation
  function :math:`k_f(r)`, the conditional expectation of a symmetric
  bivariate mark function :math:`f(m_i, m_j)` given that an event pair is
  separated by :math:`r`, normalised by its global (independent-mark)
  reference value.  Under the *random-marks* null (mark independent of
  the pattern and i.i.d. across events), :math:`k_f(r) \equiv 1`;
  :math:`k_f(r) > 1` indicates *mark clustering* (marks at nearby events
  covary positively), :math:`k_f(r) < 1` mark repulsion.  The default
  kernel is the Schlather (2001) product kernel
  :math:`f(m_i, m_j) = m_i m_j`, normalised by :math:`\bar m^2` —
  identical to ``spatstat::markcorr`` with ``correction = "isotropic"``
  off by default.
- :func:`mark_variogram` — the kernel-smoothed mark variogram
  :math:`\gamma_m(r) = \tfrac12 E[(m_i - m_j)^2 \mid \|x_i - x_j\| = r]`
  (Cressie-Hawkins 1980; Stoyan-Stoyan 1994 §13).  Increases from
  zero (perfect mark dependence at zero lag) toward the variance of the
  marks as the marks decorrelate; flat at the mark variance under the
  random-marks null.

Both estimators are Nadaraya-Watson smooths over the pair distances of
events, using the Epanechnikov kernel from
:mod:`nstat.extras.spatial._kernels`.  Lags at which the kernel weight
sums to zero (no pairs in support) return ``NaN`` — never a silent
zero.

References
----------
- Schlather M (2001). *On the second-order characteristics of marked
  point processes.* Bernoulli 7(1):99-117.
- Stoyan D, Stoyan H (1994). *Fractals, Random Shapes and Point Fields:
  Methods of Geometrical Statistics.* Wiley, §13.
- Cressie N, Hawkins DM (1980). *Robust estimation of the variogram, I.*
  Journal of the IAMG 12(2):115-125.
- Illian J, Penttinen A, Stoyan H, Stoyan D (2008). *Statistical Analysis
  and Modelling of Spatial Point Patterns.* Wiley, Chap. 5.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

from nstat.extras.spatial._kernels import epanechnikov


_VALID_KERNELS = ("schlather", "isham", "none")


def _validate_mark_kernel(name: str) -> None:
    if name not in _VALID_KERNELS:
        raise ValueError(
            "kernel must be one of {'schlather','isham','none'}; "
            f"got {name!r}"
        )


def _stoyan_bw(r_grid: np.ndarray) -> float:
    """Stoyan-style default bandwidth from the lag grid."""
    r_grid = np.asarray(r_grid, dtype=float)
    if r_grid.size <= 1:
        return 0.05
    return float(np.mean(np.diff(r_grid)))


def _pair_mark_arrays(marks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the (m_i, m_j) arrays for the upper-triangle of the pair set."""
    marks = np.asarray(marks, dtype=float).ravel()
    n = marks.shape[0]
    iu = np.triu_indices(n, k=1)
    return marks[iu[0]], marks[iu[1]]


def mark_correlation(
    points: np.ndarray,
    marks: np.ndarray,
    r_grid: np.ndarray,
    *,
    kernel: str = "schlather",
    bw: float | None = None,
) -> np.ndarray:
    r"""Kernel mark correlation function :math:`k_f(r)`.

    Nadaraya-Watson estimator with the Epanechnikov kernel

    .. math::

        \hat k_f(r) =
            \frac{\sum_{i<j} k_{bw}(r - \|x_i - x_j\|)\, f(m_i, m_j)}
                 {c_f \sum_{i<j} k_{bw}(r - \|x_i - x_j\|)}

    where :math:`c_f` is the global (independent-mark) reference
    value: :math:`c_f = \bar m^2` for the Schlather product kernel
    :math:`f(m_i, m_j) = m_i m_j`, :math:`c_f = \mathrm{Var}(m)` for the
    Isham covariance kernel :math:`f(m_i, m_j) = (m_i - \bar m)(m_j -
    \bar m)`, and :math:`c_f = 1` for the unnormalised ``"none"`` mode.

    Parameters
    ----------
    points
        ``(n, 2)`` event coordinates.
    marks
        ``(n,)`` per-event mark values.
    r_grid
        Lags at which to evaluate :math:`k_f`.
    kernel
        ``"schlather"`` (default): product kernel normalised by
        :math:`\bar m^2` (the standard *mark correlation function*).
        ``"isham"``: centred-product kernel normalised by
        :math:`\mathrm{Var}(m)` (the *mark covariance function*).
        ``"none"``: raw product kernel with normalisation 1.
    bw
        Smoothing bandwidth.  Defaults to the mean lag spacing
        (Stoyan-style rule of thumb).

    Returns
    -------
    np.ndarray
        :math:`\hat k_f(r)` at each lag in ``r_grid``.  Lags at which the
        denominator is zero return ``np.nan`` (no pairs in kernel
        support); never a silent zero.

    Notes
    -----
    *Confidence: high* on the random-marks null limit (:math:`k_f \to 1`
    under independent marks) and the clustering / repulsion sign.  The
    absolute scale is sensitive to the bandwidth in sparse-lag regimes,
    in the same way as the pair correlation function.
    """
    _validate_mark_kernel(kernel)
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    marks = np.asarray(marks, dtype=float).ravel()
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            "points must be (n, 2) for the planar mark correlation; got "
            f"shape {pts.shape}"
        )
    if marks.shape[0] != pts.shape[0]:
        raise ValueError(
            "marks must align with points; got "
            f"{marks.shape[0]} marks vs {pts.shape[0]} points"
        )
    r_grid = np.asarray(r_grid, dtype=float)
    if pts.shape[0] < 2:
        return np.full_like(r_grid, np.nan)

    if bw is None:
        bw = _stoyan_bw(r_grid)
    bw = float(bw)
    if bw <= 0:
        raise ValueError(f"bw must be positive; got {bw!r}")

    d = pdist(pts)
    mi, mj = _pair_mark_arrays(marks)

    if kernel == "schlather":
        f = mi * mj
        mbar = float(marks.mean())
        c_f = mbar * mbar
    elif kernel == "isham":
        mbar = float(marks.mean())
        f = (mi - mbar) * (mj - mbar)
        c_f = float(marks.var())
    else:  # "none"
        f = mi * mj
        c_f = 1.0

    out = np.empty_like(r_grid)
    for k, r in enumerate(r_grid):
        ker = epanechnikov((d - r) / bw)
        den = ker.sum()
        if den <= 0 or c_f <= 0:
            out[k] = np.nan
            continue
        num = float((ker * f).sum())
        out[k] = num / (den * c_f)
    return out


def mark_variogram(
    points: np.ndarray,
    marks: np.ndarray,
    r_grid: np.ndarray,
    *,
    bw: float | None = None,
) -> np.ndarray:
    r"""Kernel mark variogram :math:`\gamma_m(r)` (Cressie-Hawkins 1980).

    Nadaraya-Watson estimator

    .. math::

        \hat\gamma_m(r) = \frac{1}{2}\,
            \frac{\sum_{i<j} k_{bw}(r - \|x_i - x_j\|)\,(m_i - m_j)^2}
                 {\sum_{i<j} k_{bw}(r - \|x_i - x_j\|)} .

    Under the random-marks null, :math:`\gamma_m(r) \to \mathrm{Var}(m)`
    at every lag (modulo finite-sample noise); a non-constant
    :math:`\gamma_m(r)` flags mark non-stationarity / clustering.  For a
    constant mark field the variogram is identically zero.

    Parameters
    ----------
    points
        ``(n, 2)`` event coordinates.
    marks
        ``(n,)`` per-event mark values.
    r_grid
        Lags at which to evaluate :math:`\gamma_m`.
    bw
        Smoothing bandwidth.  Defaults to the mean lag spacing.

    Returns
    -------
    np.ndarray
        :math:`\hat\gamma_m(r)` at each lag in ``r_grid``.  Lags at
        which the kernel weight sums to zero return ``np.nan``.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    marks = np.asarray(marks, dtype=float).ravel()
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            "points must be (n, 2) for the planar mark variogram; got "
            f"shape {pts.shape}"
        )
    if marks.shape[0] != pts.shape[0]:
        raise ValueError(
            "marks must align with points; got "
            f"{marks.shape[0]} marks vs {pts.shape[0]} points"
        )
    r_grid = np.asarray(r_grid, dtype=float)
    if pts.shape[0] < 2:
        return np.full_like(r_grid, np.nan)

    if bw is None:
        bw = _stoyan_bw(r_grid)
    bw = float(bw)
    if bw <= 0:
        raise ValueError(f"bw must be positive; got {bw!r}")

    d = pdist(pts)
    mi, mj = _pair_mark_arrays(marks)
    sq = (mi - mj) ** 2

    out = np.empty_like(r_grid)
    for k, r in enumerate(r_grid):
        ker = epanechnikov((d - r) / bw)
        den = ker.sum()
        if den <= 0:
            out[k] = np.nan
            continue
        out[k] = 0.5 * float((ker * sq).sum()) / float(den)
    return out


__all__ = [
    "mark_correlation",
    "mark_variogram",
]
