"""Shared kernel / smoothing helpers for the spatial point-process module.

Pure NumPy/SciPy.  These back the GP prior in :mod:`nstat.extras.spatial.lgcp`
and the second-order estimators in :mod:`nstat.extras.spatial.spatial_gof`.

The functions here are deliberately small and dependency-free — Matern
covariances, the Epanechnikov edge kernel, and pairwise distances —
shared between the LGCP and the inhomogeneous second-order estimators.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def epanechnikov(u: np.ndarray) -> np.ndarray:
    r"""Epanechnikov kernel :math:`0.75\,(1-u^2)\mathbf{1}[|u|\le 1]`.

    The standard second-order smoothing kernel used in the pair-correlation
    estimator (Stoyan & Stoyan 1994; Baddeley et al. 2015).  Compact
    support keeps the edge-correction local.
    """
    u = np.asarray(u, dtype=float)
    out = 0.75 * (1.0 - u**2)
    out[np.abs(u) > 1.0] = 0.0
    return out


def matern_covariance(
    coords: np.ndarray,
    *,
    length_scale: float,
    variance: float = 1.0,
    nu: float = 2.5,
    jitter: float = 1e-6,
) -> np.ndarray:
    r"""Dense Matern covariance matrix on a set of points.

    Parameters
    ----------
    coords
        ``(M, d)`` array of point coordinates.
    length_scale
        The Matern length-scale :math:`\ell` (range parameter).
    variance
        Marginal variance :math:`\sigma^2`.
    nu
        Smoothness.  Only the half-integer values ``0.5`` (exponential),
        ``1.5``, and ``2.5`` are implemented in closed form — these are
        the practically-used Matern orders for LGCP priors.
    jitter
        Added to the diagonal for numerical positive-definiteness.

    Returns
    -------
    np.ndarray
        The ``(M, M)`` covariance matrix.

    Notes
    -----
    Matern-5/2 is the default — a smooth-but-not-too-smooth GP prior
    that is the standard choice for LGCP rate maps.  *Confidence:
    high — standard Matern algebra (Rasmussen & Williams 2006, §4.2).*
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    d = cdist(coords, coords)
    ell = float(length_scale)
    if ell <= 0:
        raise ValueError("length_scale must be positive")
    if nu == 0.5:
        k = np.exp(-d / ell)
    elif nu == 1.5:
        s = np.sqrt(3.0) * d / ell
        k = (1.0 + s) * np.exp(-s)
    elif nu == 2.5:
        s = np.sqrt(5.0) * d / ell
        k = (1.0 + s + 5.0 * d**2 / (3.0 * ell**2)) * np.exp(-s)
    else:
        raise ValueError(
            "nu must be one of {0.5, 1.5, 2.5} for the closed-form Matern; "
            f"got {nu!r}"
        )
    K = variance * k
    if jitter:
        K = K + jitter * np.eye(K.shape[0])
    return K


def make_grid(domain: tuple[tuple[float, float], ...], n_per_dim: int):
    """Build a regular cell-centre grid over a rectangular domain.

    Parameters
    ----------
    domain
        Tuple of ``(lo, hi)`` per spatial dimension, e.g.
        ``((0.0, 1.0), (0.0, 1.0))`` for the unit square.
    n_per_dim
        Number of cells along each axis (``G`` in the worked example).

    Returns
    -------
    centres : np.ndarray
        ``(G**d, d)`` cell-centre coordinates in row-major (meshgrid)
        order — matching ``np.histogramdd`` after a transpose.
    cell_area : float
        The area (or volume) of one cell.
    edges : list[np.ndarray]
        The bin edges per dimension (length ``G + 1`` each), suitable for
        :func:`numpy.histogramdd`.
    """
    domain = tuple((float(lo), float(hi)) for lo, hi in domain)
    G = int(n_per_dim)
    edges = [np.linspace(lo, hi, G + 1) for lo, hi in domain]
    centre_axes = [0.5 * (e[:-1] + e[1:]) for e in edges]
    mesh = np.meshgrid(*centre_axes, indexing="xy")
    centres = np.column_stack([m.ravel() for m in mesh])
    cell_area = float(np.prod([(hi - lo) / G for lo, hi in domain]))
    return centres, cell_area, edges


def bin_counts(points: np.ndarray, edges) -> np.ndarray:
    """Exact integer cell counts on the grid defined by ``edges``.

    Row-major (meshgrid ``indexing='xy'``) flattening to match
    :func:`make_grid`.  Cell counts of a Poisson point process on a
    partition are themselves independent Poisson with mean equal to
    the cell-integrated intensity (Møller-Waagepetersen 2003, Prop. 3.1).
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.shape[0] == 0:
        n_cells = int(np.prod([len(e) - 1 for e in edges]))
        return np.zeros(n_cells, dtype=float)
    H, _ = np.histogramdd(points, bins=edges)
    # meshgrid 'xy' transposes the first two axes relative to histogramdd.
    if H.ndim >= 2:
        H = np.swapaxes(H, 0, 1)
    return H.ravel().astype(float)


__all__ = [
    "epanechnikov",
    "matern_covariance",
    "make_grid",
    "bin_counts",
]
