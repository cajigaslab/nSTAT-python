r"""Minimum-contrast estimation for spatial cluster Cox processes.

Implements the minimum-contrast (MC) estimator of Diggle (2013) §6.2.1
for fitting closed-form pair correlations to the empirical SOIRS
:math:`\hat g(r)` returned by
:func:`nstat.extras.spatial.spatial_gof.pair_correlation`.

The MC objective is

.. math::

    S(\theta) = \int_{r_{\min}}^{r_{\max}}
        \Bigl[\hat g(r)^q - g(r;\theta)^q\Bigr]^2 \, dr,

with :math:`q = 0.25` the default contrast power (Diggle 2013 §6.2.1;
Møller-Waagepetersen 2003 §4.2).  The integral is evaluated by
Simpson's rule over the lag grid, NaN samples (typical of the
``"border"`` edge correction at small ``r``) are dropped, and the
search is run with SciPy's bounded L-BFGS-B.

Convenience wrappers :func:`fit_thomas` and :func:`fit_matern_cluster`
plug in the closed-form pair correlations from
:mod:`~nstat.extras.spatial.cluster_cox` and return a
:class:`MinContrastResult`.

References
----------
- Diggle, P. J. (2013). *Statistical Analysis of Spatial and
  Spatio-Temporal Point Patterns* (3rd ed.). CRC §6.2.1.
- Møller, J. & Waagepetersen, R. P. (2003). *Statistical Inference and
  Simulation for Spatial Point Processes.* Chapman & Hall §4.2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import minimize

from nstat.extras.spatial.cluster_cox import (
    matern_cluster_pair_correlation,
    thomas_pair_correlation,
)
from nstat.extras.spatial.spatial_gof import pair_correlation


@dataclass(frozen=True)
class MinContrastResult:
    r"""Output of a minimum-contrast estimator.

    Attributes
    ----------
    theta_hat
        Parameter estimate (1-D ndarray).
    objective_value
        Final value of :math:`S(\theta)` at ``theta_hat``.
    g_model_at_theta
        Model :math:`g(r; \hat\theta)` evaluated on the lag grid.
    n_iter
        Optimizer iteration count (0 if optimization did not run).
    success
        ``True`` when the optimizer converged AND the empirical curve
        had enough finite samples (>=4 after NaN filtering).
    message
        Human-readable status string.
    """
    theta_hat: np.ndarray
    objective_value: float
    g_model_at_theta: np.ndarray
    n_iter: int
    success: bool
    message: str


def min_contrast_estimator(
    g_emp: np.ndarray,
    g_model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    r_grid: np.ndarray,
    theta0: np.ndarray,
    *,
    bounds: list[tuple[float | None, float | None]] | None = None,
    q: float = 0.25,
) -> MinContrastResult:
    r"""Diggle (2013) minimum-contrast estimator.

    Minimises

    .. math::

        S(\theta) = \int \bigl[\hat g(r)^q - g(r; \theta)^q\bigr]^2 \, dr

    over a closed-form pair-correlation model ``g_model_fn(r, theta)``.
    Samples where either ``g_emp(r)`` or the model evaluates to NaN are
    dropped before integration; if fewer than 4 samples remain the
    optimisation is skipped and ``success=False`` is returned.

    Parameters
    ----------
    g_emp
        Empirical pair correlation aligned with ``r_grid``.
    g_model_fn
        Callable ``(r, theta) -> g(r; theta)`` returning an array
        aligned with ``r``.
    r_grid
        1-D lag grid (strictly increasing).
    theta0
        1-D initial parameter vector.
    bounds
        L-BFGS-B box bounds, ``[(lo, hi), ...]``; ``None`` for
        unconstrained.
    q
        Contrast power (default 0.25; Diggle 2013 §6.2.1).

    Returns
    -------
    MinContrastResult
    """
    if q <= 0:
        raise ValueError(f"q must be positive; got {q}")
    g_emp = np.asarray(g_emp, dtype=float).ravel()
    r_grid = np.asarray(r_grid, dtype=float).ravel()
    theta0_arr = np.asarray(theta0, dtype=float)
    if theta0_arr.ndim != 1:
        raise ValueError(
            f"theta0 must be 1-D; got shape {theta0_arr.shape}"
        )
    theta0 = theta0_arr
    if g_emp.shape != r_grid.shape:
        raise ValueError(
            f"g_emp shape {g_emp.shape} must match r_grid shape {r_grid.shape}"
        )

    # Build a closure that returns S(theta) on the finite, non-negative
    # subset of (r, g_emp).  Samples where g_emp <= 0 cannot be raised
    # to a fractional power; they are also dropped.
    def _objective(theta: np.ndarray) -> float:
        g_mod = np.asarray(g_model_fn(r_grid, theta), dtype=float).ravel()
        finite = (
            np.isfinite(g_emp)
            & np.isfinite(g_mod)
            & (g_emp > 0.0)
            & (g_mod > 0.0)
        )
        if finite.sum() < 4:
            return np.inf
        r = r_grid[finite]
        e = g_emp[finite] ** q - g_mod[finite] ** q
        return float(simpson(e**2, x=r))

    # Pre-flight: confirm we have enough finite samples at theta0 to
    # run the optimiser at all.
    g_mod0 = np.asarray(g_model_fn(r_grid, theta0), dtype=float).ravel()
    finite0 = (
        np.isfinite(g_emp)
        & np.isfinite(g_mod0)
        & (g_emp > 0.0)
        & (g_mod0 > 0.0)
    )
    if finite0.sum() < 4:
        return MinContrastResult(
            theta_hat=np.asarray(theta0, dtype=float).copy(),
            objective_value=float("inf"),
            g_model_at_theta=g_mod0,
            n_iter=0,
            success=False,
            message=(
                f"too few finite (g_emp, g_model) samples ({int(finite0.sum())}) "
                "after filtering; optimisation skipped"
            ),
        )

    try:
        res = minimize(
            _objective,
            theta0,
            method="L-BFGS-B",
            bounds=bounds,
        )
    except Exception as exc:  # pragma: no cover - defensive
        g_mod_final = np.asarray(
            g_model_fn(r_grid, theta0), dtype=float
        ).ravel()
        return MinContrastResult(
            theta_hat=np.asarray(theta0, dtype=float).copy(),
            objective_value=float("inf"),
            g_model_at_theta=g_mod_final,
            n_iter=0,
            success=False,
            message=f"optimizer raised: {exc!r}",
        )

    g_mod_final = np.asarray(g_model_fn(r_grid, res.x), dtype=float).ravel()
    return MinContrastResult(
        theta_hat=np.asarray(res.x, dtype=float),
        objective_value=float(res.fun),
        g_model_at_theta=g_mod_final,
        n_iter=int(res.nit),
        success=bool(res.success),
        message=str(res.message),
    )


# ----------------------------------------------------------------------
# Convenience wrappers — Thomas / Matérn-cluster
# ----------------------------------------------------------------------


def _domain_extent(domain: tuple[tuple[float, float], tuple[float, float]]):
    """Return ``(xmin, xmax, ymin, ymax, area, diam)`` from a 2-tuple-of-2-tuples."""
    if not (
        isinstance(domain, tuple)
        and len(domain) == 2
        and all(isinstance(d, tuple) and len(d) == 2 for d in domain)
    ):
        raise ValueError(
            "domain must be ((xmin, xmax), (ymin, ymax)); got "
            f"{domain!r}"
        )
    (xmin, xmax), (ymin, ymax) = domain
    area = float((xmax - xmin) * (ymax - ymin))
    diam = float(np.hypot(xmax - xmin, ymax - ymin))
    return float(xmin), float(xmax), float(ymin), float(ymax), area, diam


def _empirical_pair_correlation(
    points: np.ndarray,
    domain: tuple[tuple[float, float], tuple[float, float]],
    r_grid: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Estimate ``lambda_hat = n / area`` and return ``(g_emp, lambda_hat)``.

    The SOIRS estimator is honest under SOIRS — homogeneous limit is fine
    here because the cluster processes have constant mean intensity.  We
    use ``edge_correction="border"`` (Baddeley-Rubak-Turner 2015 §7.4),
    matching the architect's brief; it returns ``NaN`` at radii where no
    event has boundary-distance >= r, which the min-contrast NaN filter
    drops automatically.
    """
    pts = np.asarray(points, dtype=float)
    _, _, _, _, area, _ = _domain_extent(domain)
    if area <= 0:
        raise ValueError(f"domain area must be positive; got {area}")
    n = pts.shape[0]
    if n < 2:
        raise ValueError(
            f"need at least 2 points to estimate g(r); got {n}"
        )
    lam = float(n) / area
    lam_arr = np.full(n, lam, dtype=float)
    g_emp = pair_correlation(
        pts, lam_arr, r_grid, domain=domain, edge_correction="border"
    )
    return np.asarray(g_emp, dtype=float), lam


def fit_thomas(
    points: np.ndarray,
    domain: tuple[tuple[float, float], tuple[float, float]],
    r_grid: np.ndarray,
    theta0: tuple[float, float] | np.ndarray | None = None,
    *,
    grid_search_fallback: bool = False,  # noqa: ARG001 (reserved for future use)
) -> MinContrastResult:
    r"""Fit a :class:`~.cluster_cox.ThomasProcess` by minimum contrast.

    Estimates ``(sigma, intensity_parent)`` from a pattern by matching
    the SOIRS empirical pair correlation to the closed-form Thomas
    :func:`~.cluster_cox.thomas_pair_correlation`.  ``mu_offspring``
    does not enter :math:`g(r)` and is therefore not estimable from
    second-order statistics alone — recover it post-hoc from
    :math:`\hat\mu = n / (\hat\lambda_p\,|W|)`.

    Parameters
    ----------
    points
        ``(n, 2)`` observed pattern.
    domain
        ``((xmin, xmax), (ymin, ymax))``.
    r_grid
        1-D lag grid.
    theta0
        Optional initial ``(sigma, intensity_parent)``.  Default:
        ``(0.1 * diam, 10.0)``.
    grid_search_fallback
        Reserved — currently unused (the L-BFGS-B optimiser converges
        reliably for the Thomas closed form within the brief's
        tolerance).  Kept in the signature so a future enhancement can
        switch on a coarse grid pre-scan without breaking callers.

    Returns
    -------
    MinContrastResult
        ``theta_hat = (sigma_hat, lambda_p_hat)``.
    """
    pts = np.asarray(points, dtype=float)
    r_grid = np.asarray(r_grid, dtype=float).ravel()
    _, _, _, _, _, diam = _domain_extent(domain)
    g_emp, _ = _empirical_pair_correlation(pts, domain, r_grid)

    if theta0 is None:
        theta0 = np.array([0.1 * diam, 10.0], dtype=float)
    else:
        theta0 = np.asarray(theta0, dtype=float).ravel()
        if theta0.shape != (2,):
            raise ValueError(
                f"theta0 must have shape (2,); got {theta0.shape}"
            )

    bounds: list[tuple[float | None, float | None]] = [
        (1e-6, None),  # sigma > 0
        (1e-3, None),  # intensity_parent > 0
    ]

    def _g_thomas(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        sigma, lam_p = float(theta[0]), float(theta[1])
        if sigma <= 0 or lam_p <= 0:
            return np.full_like(r, np.nan, dtype=float)
        return thomas_pair_correlation(r, sigma, lam_p, 1.0)

    return min_contrast_estimator(
        g_emp,
        _g_thomas,
        r_grid,
        theta0,
        bounds=bounds,
    )


def fit_matern_cluster(
    points: np.ndarray,
    domain: tuple[tuple[float, float], tuple[float, float]],
    r_grid: np.ndarray,
    theta0: tuple[float, float] | np.ndarray | None = None,
    *,
    grid_search_fallback: bool = False,  # noqa: ARG001 (reserved for future use)
) -> MinContrastResult:
    r"""Fit a :class:`~.cluster_cox.MaternClusterProcess` by minimum contrast.

    Estimates ``(radius, intensity_parent)``; see :func:`fit_thomas` for
    the ``mu_offspring`` caveat.

    Default ``theta0 = (0.1 * diam, 10.0)``; bounds
    ``[(1e-6, None), (1e-3, None)]``.
    """
    pts = np.asarray(points, dtype=float)
    r_grid = np.asarray(r_grid, dtype=float).ravel()
    _, _, _, _, _, diam = _domain_extent(domain)
    g_emp, _ = _empirical_pair_correlation(pts, domain, r_grid)

    if theta0 is None:
        theta0 = np.array([0.1 * diam, 10.0], dtype=float)
    else:
        theta0 = np.asarray(theta0, dtype=float).ravel()
        if theta0.shape != (2,):
            raise ValueError(
                f"theta0 must have shape (2,); got {theta0.shape}"
            )

    bounds: list[tuple[float | None, float | None]] = [
        (1e-6, None),  # radius > 0
        (1e-3, None),  # intensity_parent > 0
    ]

    def _g_matern(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        radius, lam_p = float(theta[0]), float(theta[1])
        if radius <= 0 or lam_p <= 0:
            return np.full_like(r, np.nan, dtype=float)
        return matern_cluster_pair_correlation(r, radius, lam_p, 1.0)

    return min_contrast_estimator(
        g_emp,
        _g_matern,
        r_grid,
        theta0,
        bounds=bounds,
    )


__all__ = [
    "MinContrastResult",
    "min_contrast_estimator",
    "fit_thomas",
    "fit_matern_cluster",
]
