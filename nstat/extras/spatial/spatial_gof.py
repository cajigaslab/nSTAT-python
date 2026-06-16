r"""Inhomogeneous second-order spatial goodness-of-fit (pure NumPy/SciPy).

A Python-only inhomogeneous second-order goodness-of-fit suite for
spatial point processes.  It provides the intensity-reweighted
second-order summary statistics that test what a
fitted intensity :math:`\hat\lambda(\mathbf{x})` leaves over — the
diagnostics a homogeneous :math:`K`-function cannot give for a
non-stationary neural field:

- :func:`pair_correlation` — the SOIRS-reweighted pair correlation
  :math:`g(r)`; ``g(r) > 1`` indicates clustering, ``g(r) < 1`` repulsion,
  ``g(r) = 1`` the (inhomogeneous) Poisson null.
- :func:`k_inhom` — the inhomogeneous :math:`K`-function of
  Baddeley, Moller & Waagepetersen (2000); for an inhomogeneous Poisson
  process ``K_inhom(r) = pi r^2`` in 2-D.
- :func:`l_function` — the variance-stabilized
  :math:`L(r) = \sqrt{K(r)/\pi}`, so ``L(r) - r = 0`` under the null.
- :func:`nearest_neighbour_FGJ` — the empty-space :math:`F`, nearest-
  neighbour :math:`G`, and :math:`J = (1-G)/(1-F)` functions.
- :func:`global_envelope` — a Monte-Carlo global-rank envelope test
  (Myllymaki et al. 2017) built by thinning the fitted inhomogeneous
  Poisson intensity.

.. warning::

   **Plug-in bias.**  Reweighting by an intensity :math:`\hat\lambda`
   estimated from the *same* pattern deflates the variance of the
   estimator and shrinks the envelope below nominal coverage (the fitted
   field has already absorbed some of the clustering).  Pass a
   *held-out* intensity for ``lambda_hat`` (e.g. a smoother fit to a
   disjoint fold, or a global-reweighting estimator; Shaw, Moller &
   Waagepetersen 2021) or treat the resulting coverage as optimistic.
   Every estimator below carries this caveat.

This is the static-spatial analogue of the time-rescaling KS test in
:mod:`nstat.extras.spatial.marked_gof`; the SOIRS precondition (second-
order intensity-reweighted stationarity — :math:`\lambda` bounded away
from zero, :math:`g` a function of lag only) is the spatial mirror of
the time-rescaling null.

References
----------
- Baddeley AJ, Moller J, Waagepetersen R (2000). *Non- and semi-parametric
  estimation of interaction in inhomogeneous point patterns.*
  Statistica Neerlandica 54(3):329.
- Myllymaki M, Mrkvicka T, Grabarnik P, Seijo H, Hahn U (2017). *Global
  envelope tests for spatial processes.* JRSS-B 79(2):381.
- Shaw T, Moller J, Waagepetersen R (2021). *Globally intensity-reweighted
  estimators for K- and pair correlation functions.* ANZJS 63(1):93.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist

from nstat.extras.spatial._envelopes import EnvelopeResult, global_rank_envelope
from nstat.extras.spatial._kernels import epanechnikov


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _as_points(points: np.ndarray) -> np.ndarray:
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.ndim != 2:
        raise ValueError(f"points must be (n, d); got shape {points.shape}")
    return points


def _intensity_at(lambda_hat, points: np.ndarray) -> np.ndarray:
    """Evaluate the reweighting intensity at the given points.

    ``lambda_hat`` may be (a) a callable ``X -> lambda(X)``, or (b) a
    1-D array aligned with ``points`` giving lambda at each point.
    """
    if callable(lambda_hat):
        lam = np.asarray(lambda_hat(points), dtype=float).ravel()
    else:
        lam = np.asarray(lambda_hat, dtype=float).ravel()
        if lam.shape[0] != points.shape[0]:
            raise ValueError(
                "lambda_hat array must align with points; got "
                f"{lam.shape[0]} vs {points.shape[0]}"
            )
    if np.any(lam <= 0):
        raise ValueError(
            "lambda_hat must be strictly positive (SOIRS requires lambda "
            "bounded away from zero); got a non-positive value."
        )
    return lam


def _domain_measure(domain) -> float:
    return float(np.prod([hi - lo for lo, hi in domain]))


# ----------------------------------------------------------------------
# Pair correlation g(r)
# ----------------------------------------------------------------------


def pair_correlation(
    points: np.ndarray,
    lambda_hat,
    r_grid: np.ndarray,
    *,
    bw: float | None = None,
    domain=None,
) -> np.ndarray:
    r"""SOIRS-reweighted inhomogeneous pair correlation :math:`g(r)`.

    Kernel estimator with an Epanechnikov edge kernel and the SOIRS
    reweighting :math:`1/(\hat\lambda_i\,\hat\lambda_j)`:

    .. math::

        \hat g(r) = \frac{1}{2\pi r\,|D|}
            \sum_{i\neq j} \frac{k_{bw}(r - \|x_i - x_j\|)}
                {\hat\lambda(x_i)\,\hat\lambda(x_j)} .

    Under the inhomogeneous Poisson null, :math:`g(r) = 1` at all lags;
    ``g(r) > 1`` is clustering and ``g(r) < 1`` repulsion.

    Parameters
    ----------
    points
        ``(n, d)`` event coordinates (``d = 2`` for the planar estimator).
    lambda_hat
        The reweighting intensity — a callable ``X -> lambda(X)`` or a
        per-point array.  **Use a held-out estimate** (see module warning).
    r_grid
        Lags at which to evaluate :math:`g`.
    bw
        Kernel bandwidth.  Defaults to a Stoyan-style rule of thumb
        based on the lag spacing.
    domain
        ``((lo, hi), ...)`` per dim.  Defaults to the bounding box of
        ``points`` — pass the true analysis window when known.

    Returns
    -------
    np.ndarray
        :math:`\hat g(r)` at each lag in ``r_grid``.

    Notes
    -----
    *Confidence: high* on the homogeneous limit (g → 1 for Poisson) and
    the clustering/repulsion sign; the absolute scale is sensitive to the
    bandwidth and the plug-in caveat in the module docstring.
    """
    pts = _as_points(points)
    r_grid = np.asarray(r_grid, dtype=float)
    if pts.shape[0] < 2:
        return np.zeros_like(r_grid)
    if pts.shape[1] != 2:
        raise ValueError("pair_correlation is implemented for d=2 (planar).")

    if domain is None:
        domain = tuple((pts[:, k].min(), pts[:, k].max()) for k in range(pts.shape[1]))
    area = _domain_measure(domain)
    if bw is None:
        bw = float(0.1 / np.sqrt(pts.shape[0] / area)) if area > 0 else 0.05
        bw = max(bw, float(np.mean(np.diff(r_grid))) if len(r_grid) > 1 else 0.05)

    lam = _intensity_at(lambda_hat, pts)
    iu = np.triu_indices(pts.shape[0], k=1)
    d = cdist(pts, pts)[iu]
    wgt = 1.0 / (lam[iu[0]] * lam[iu[1]])

    g = np.empty_like(r_grid)
    for k, r in enumerate(r_grid):
        ring = 2.0 * np.pi * r * area
        if ring <= 0:
            g[k] = 0.0
            continue
        g[k] = 2.0 * np.sum(epanechnikov((d - r) / bw) / bw * wgt) / ring
    return g


# ----------------------------------------------------------------------
# Inhomogeneous K and L
# ----------------------------------------------------------------------


def k_inhom(
    points: np.ndarray,
    lambda_hat,
    r_grid: np.ndarray,
    *,
    domain=None,
) -> np.ndarray:
    r"""Inhomogeneous :math:`K`-function (Baddeley-Moller-Waagepetersen 2000).

    .. math::

        \hat K_{\text{inhom}}(r) = \frac{1}{|D|}
            \sum_{i\neq j}
            \frac{\mathbf{1}[\|x_i - x_j\| \le r]}
                {\hat\lambda(x_i)\,\hat\lambda(x_j)} ,

    the standard estimator that reduces to the ordinary :math:`K` when
    :math:`\hat\lambda` is constant.  For an inhomogeneous Poisson
    process, ``K_inhom(r) = pi r^2`` in 2-D (the CSR benchmark).

    Parameters
    ----------
    points, lambda_hat, domain
        See :func:`pair_correlation`.  **Held-out** ``lambda_hat``.
    r_grid
        Upper radii at which to evaluate the cumulative statistic.

    Returns
    -------
    np.ndarray
        :math:`\hat K_{\text{inhom}}(r)` at each lag.
    """
    pts = _as_points(points)
    r_grid = np.asarray(r_grid, dtype=float)
    if pts.shape[0] < 2:
        return np.zeros_like(r_grid)
    if domain is None:
        domain = tuple((pts[:, k].min(), pts[:, k].max()) for k in range(pts.shape[1]))
    area = _domain_measure(domain)
    lam = _intensity_at(lambda_hat, pts)

    iu = np.triu_indices(pts.shape[0], k=1)
    d = pdist(pts)
    wgt = 1.0 / (lam[iu[0]] * lam[iu[1]])

    K = np.empty_like(r_grid)
    for k, r in enumerate(r_grid):
        # factor 2: the (i, j) and (j, i) ordered pairs both contribute.
        K[k] = 2.0 * np.sum(wgt[d <= r]) / area
    return K


def l_function(
    points: np.ndarray,
    lambda_hat,
    r_grid: np.ndarray,
    *,
    domain=None,
) -> np.ndarray:
    r"""Variance-stabilized inhomogeneous :math:`L`-function.

    :math:`L(r) = \sqrt{K_{\text{inhom}}(r)/\pi}` (2-D), so the centred
    statistic :math:`L(r) - r` is flat at zero under the inhomogeneous
    Poisson null.  Returns ``L(r)`` (subtract ``r_grid`` for the centred
    form).
    """
    K = k_inhom(points, lambda_hat, r_grid, domain=domain)
    return np.sqrt(np.maximum(K, 0.0) / np.pi)


# ----------------------------------------------------------------------
# Nearest-neighbour F / G / J
# ----------------------------------------------------------------------


def nearest_neighbour_FGJ(
    points: np.ndarray,
    r_grid: np.ndarray,
    *,
    domain=None,
    n_dummy: int | None = None,
    rng: np.random.Generator | None = None,
):
    r"""Empty-space :math:`F`, nearest-neighbour :math:`G`, and :math:`J`.

    - :math:`G(r)` — CDF of the nearest-neighbour distance from each event
      to the nearest other event.
    - :math:`F(r)` — empty-space function: CDF of the distance from a grid
      of reference (dummy) locations to the nearest event.
    - :math:`J(r) = (1 - G(r))/(1 - F(r))` — equals 1 under CSR;
      ``J < 1`` clustering, ``J > 1`` regularity.

    These are *homogeneous* (unreweighted) summaries; they characterize
    inter-event spacing and are most useful as a complement to the
    reweighted :math:`g`/:math:`K` on roughly-stationary patterns.

    Returns
    -------
    F, G, J : tuple[np.ndarray, np.ndarray, np.ndarray]
        Each aligned with ``r_grid``.
    """
    pts = _as_points(points)
    r_grid = np.asarray(r_grid, dtype=float)
    rng = np.random.default_rng() if rng is None else rng
    if domain is None:
        domain = tuple((pts[:, k].min(), pts[:, k].max()) for k in range(pts.shape[1]))

    n = pts.shape[0]
    if n < 2:
        z = np.zeros_like(r_grid)
        return z, z, np.ones_like(r_grid)

    # G: nearest-neighbour distances among events.
    dmat = cdist(pts, pts)
    np.fill_diagonal(dmat, np.inf)
    nn = dmat.min(axis=1)
    G = np.array([np.mean(nn <= r) for r in r_grid])

    # F: empty-space distances from dummy points to nearest event.
    if n_dummy is None:
        n_dummy = max(100, 4 * n)
    lows = np.array([lo for lo, _ in domain])
    highs = np.array([hi for _, hi in domain])
    dummies = lows + (highs - lows) * rng.uniform(size=(n_dummy, pts.shape[1]))
    de = cdist(dummies, pts).min(axis=1)
    F = np.array([np.mean(de <= r) for r in r_grid])

    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where((1.0 - F) > 1e-12, (1.0 - G) / (1.0 - F), np.nan)
    return F, G, J


# ----------------------------------------------------------------------
# Global-rank envelope (Monte-Carlo)
# ----------------------------------------------------------------------


def global_envelope(
    points: np.ndarray,
    lambda_hat,
    r_grid: np.ndarray,
    *,
    n_sim: int = 199,
    domain=None,
    statistic: str = "pcf",
    bw: float | None = None,
    alpha: float = 0.05,
    lambda_grid=None,
    grid_points: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> EnvelopeResult:
    r"""Monte-Carlo global-rank envelope test against the inhomogeneous null.

    Simulates ``n_sim`` realizations of the fitted inhomogeneous Poisson
    process (by thinning a dominating homogeneous process at the maximum
    intensity over a reference grid), computes the chosen summary
    statistic on each, and builds the global-rank envelope of
    Myllymaki et al. (2017).  The observed pattern's statistic is tested
    for global containment.

    Parameters
    ----------
    points, lambda_hat, r_grid, domain, bw
        See :func:`pair_correlation`.  ``lambda_hat`` (held-out) is used
        both to reweight the statistic and as the simulation intensity.
    n_sim
        Number of Monte-Carlo null simulations (199 → 95% global level
        with ``alpha = 0.05``).
    statistic
        ``"pcf"`` for :math:`g(r)` (default), ``"kinhom"`` for
        :math:`K_{\text{inhom}}(r)`, or ``"linhom"`` for :math:`L(r)`.
    alpha
        Global type-I error level.
    lambda_grid, grid_points
        Optional precomputed intensity values and their coordinates,
        used to find the dominating intensity for thinning.  If omitted,
        a default reference grid over ``domain`` is built and
        ``lambda_hat`` evaluated on it.
    rng
        NumPy random generator for the simulations.

    Returns
    -------
    EnvelopeResult
        Carries ``observed``, ``lo``, ``hi``, ``inside``, ``p_interval``.

    Notes
    -----
    Honours the **plug-in caveat**: if ``lambda_hat`` was fit to the same
    pattern, the envelope coverage is optimistic.  Prefer a held-out
    intensity.  *Confidence: high on the mechanics; coverage validity is
    conditional on the held-out reweighting.*
    """
    pts = _as_points(points)
    r_grid = np.asarray(r_grid, dtype=float)
    rng = np.random.default_rng() if rng is None else rng
    if domain is None:
        domain = tuple((pts[:, k].min(), pts[:, k].max()) for k in range(pts.shape[1]))
    area = _domain_measure(domain)

    stat_fns = {
        "pcf": lambda X: pair_correlation(X, lambda_hat, r_grid, bw=bw, domain=domain),
        "kinhom": lambda X: k_inhom(X, lambda_hat, r_grid, domain=domain),
        "linhom": lambda X: l_function(X, lambda_hat, r_grid, domain=domain),
    }
    if statistic not in stat_fns:
        raise ValueError(f"statistic must be one of {list(stat_fns)}; got {statistic!r}")
    stat_fn = stat_fns[statistic]

    # Dominating intensity for the thinning simulation.
    if grid_points is None or lambda_grid is None:
        n_ref = 40
        lows = np.array([lo for lo, _ in domain])
        highs = np.array([hi for _, hi in domain])
        axes = [np.linspace(lo, hi, n_ref) for lo, hi in domain]
        mesh = np.meshgrid(*axes, indexing="xy")
        grid_points = np.column_stack([m.ravel() for m in mesh])
        lambda_grid = _intensity_at(lambda_hat, grid_points)
    else:
        grid_points = _as_points(grid_points)
        lambda_grid = np.asarray(lambda_grid, dtype=float).ravel()
    lam_max = float(np.max(lambda_grid))

    observed = stat_fn(pts)
    sims = np.empty((n_sim, len(r_grid)))
    lows = np.array([lo for lo, _ in domain])
    highs = np.array([hi for _, hi in domain])
    for s in range(n_sim):
        n_prop = rng.poisson(lam_max * area)
        cand = lows + (highs - lows) * rng.uniform(size=(n_prop, pts.shape[1]))
        lam_cand = _intensity_at(lambda_hat, cand) if n_prop > 0 else np.empty(0)
        keep = rng.uniform(size=n_prop) < (lam_cand / lam_max) if n_prop > 0 else np.array([], bool)
        Xs = cand[keep]
        sims[s] = stat_fn(Xs)

    return global_rank_envelope(observed, sims, r_grid, alpha=alpha)


__all__ = [
    "pair_correlation",
    "k_inhom",
    "l_function",
    "nearest_neighbour_FGJ",
    "global_envelope",
    "EnvelopeResult",
]
