r"""Space-time inhomogeneous second-order goodness-of-fit (pure NumPy/SciPy).

The space-time analogue of :mod:`nstat.extras.spatial.spatial_gof`: the
intensity-reweighted second-order summary statistics that test what a
fitted space-time intensity :math:`\hat\lambda(x,t)` (e.g. the output of
:func:`nstat.extras.spatial.st_intensity.intensity_st_kde`) leaves over.

- :func:`k_st_inhom` — the inhomogeneous space-time :math:`K`-function
  (Diggle, Chetwynd, Haggkvist & Morris 1995; Gabriel & Diggle 2009). For
  an inhomogeneous space-time Poisson process in 2-D + time,
  ``K_st(r, t) = pi r^2 * 2t`` (the product of the spatial disc area and
  the two-sided temporal interval length).
- :func:`pair_correlation_st` — the space-time pair correlation
  :math:`g(r, t)` of Moller & Ghorbani (2012); ``g(r, t) = 1`` under the
  inhomogeneous space-time Poisson null, ``> 1`` short-range space-time
  clustering, ``< 1`` inhibition.
- :func:`global_envelope_st` — a Monte-Carlo global-rank envelope test
  (Myllymaki et al. 2017) over the flattened ``(r, t)`` surface, built by
  thinning the fitted space-time inhomogeneous Poisson intensity.

This is original code implemented directly from the published equations
cited below (Diggle et al. 1995; Gabriel & Diggle 2009; Moller & Ghorbani
2012; Myllymaki et al. 2017); it is **not** a port of any external
implementation.

Derivation of the estimated pair correlation
---------------------------------------------

For a second-order intensity-reweighted stationary (SOIRS) space-time
process, the reduced second moment measure satisfies, in polar-plus-lag
form (isotropic in the spatial offset, symmetric in the signed temporal
offset),

.. math::

    K_{st}(r, t) = \int_{\|h\|\le r} \int_{|\tau| \le t} g(\|h\|, |\tau|)
        \, d\tau\, dh
        = \int_0^r\!\int_0^t g(u, v)\, (2\pi u)\, (2)\, dv\, du
        = 4\pi \int_0^r\!\int_0^t u\, g(u, v)\, dv\, du,

where the factor :math:`2\pi u` is the usual area element of a 2-D
annulus and the factor :math:`2` comes from the two-sided temporal lag
interval :math:`[-t, t]` folding onto :math:`|\tau|\in[0,t]`.  Under the
inhomogeneous-Poisson null (:math:`g\equiv 1`), :math:`K_{st}(r,t) = 2\pi
r^2 t = \pi r^2\cdot 2t` — the benchmark used by :func:`k_st_inhom`.
Differentiating,

.. math::

    g(r, t) = \frac{1}{4\pi r}\, \frac{\partial^2 K_{st}}{\partial r\,
        \partial t},

which :func:`pair_correlation_st` estimates directly by a product-kernel
smoothed count (the space-time analogue of the Epanechnikov-kernel
pair-correlation estimator in :func:`nstat.extras.spatial.spatial_gof.pair_correlation`,
whose ring normalisation :math:`2\pi r` is here replaced by :math:`4\pi
r` to account for the extra temporal fold).

Edge correction
----------------

Each event pair ``(i, j)`` is reweighted by :math:`w_{ij} = w^{\mathrm
space}_{ij} \cdot w^{\mathrm time}_{ij}`, the product of a spatial and a
temporal edge-correction weight (Gabriel & Diggle 2009, Sec. 2.2, describe
exactly this product-correction strategy for the space-time case). The
spatial factor reuses the three rectangular-window corrections already
implemented in :mod:`nstat.extras.spatial._edge` for
:func:`nstat.extras.spatial.spatial_gof.k_inhom` — Ripley's (1976, 1977)
isotropic correction, Ohser's (1983) translation correction, and the
border method of Baddeley, Rubak & Turner (2015).  The temporal factor is
the natural 1-D analogue for an observation interval :math:`T=[t_{lo},
t_{hi}]` of length :math:`|T|`:

- **translation**: :math:`|T| / (|T| - |t_i - t_j|)` (Ohser 1983's
  translation correction specialised to a 1-D interval).
- **border**: restrict the focal event to those with temporal
  boundary-distance :math:`\ge t` (Baddeley-Rubak-Turner 2015, eq. 7.5,
  specialised to 1-D), analogous to the spatial border restriction.

The Ripley isotropic correction has no natural 1-D analogue (there is no
boundary "arc" in one dimension), so the ``"isotropic"`` mode pairs the
spatial Ripley weight with the temporal translation weight.

.. warning::

   **Plug-in bias.**  As in the static-spatial module, reweighting by an
   intensity :math:`\hat\lambda` fit to the *same* pattern deflates the
   estimator's variance and shrinks the envelope below nominal coverage.
   Pass a *held-out* :math:`\hat\lambda` (e.g. from
   :func:`nstat.extras.spatial.st_intensity.intensity_st_kde` fit to a
   disjoint fold) where possible.

References
----------
- Diggle PJ, Chetwynd AG, Haggkvist R, Morris SE (1995). *Second-order
  analysis of space-time clustering.* Statistical Methods in Medical
  Research / J. Royal Statistical Society, Series C 44(1):71-86.
- Gabriel E, Diggle PJ (2009). *Second-order analysis of inhomogeneous
  spatio-temporal point process data.* Statistica Neerlandica 63(1):43-51.
- Moller J, Ghorbani M (2012). *Aspects of second-order analysis of
  structured inhomogeneous spatio-temporal point processes.* Statistica
  Neerlandica 66(4):472-491.
- Myllymaki M, Mrkvicka T, Grabarnik P, Seijo H, Hahn U (2017). *Global
  envelope tests for spatial processes.* JRSS-B 79(2):381-404.
- Ripley BD (1976, 1977); Ohser J (1983); Baddeley A, Rubak E, Turner R
  (2015) — the rectangular spatial edge corrections reused here, see
  :mod:`nstat.extras.spatial._edge`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist

from nstat.extras.spatial._edge import border_usable_mask, frac_disc_in_rect
from nstat.extras.spatial._envelopes import global_rank_envelope
from nstat.extras.spatial._kernels import epanechnikov

__all__ = [
    "STKResult",
    "STEnvelopeResult",
    "k_st_inhom",
    "pair_correlation_st",
    "global_envelope_st",
]

_VALID_ST_EDGE_CORRECTIONS = ("isotropic", "translation", "border")


# ----------------------------------------------------------------------
# Result dataclasses
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class STKResult:
    """Result of :func:`k_st_inhom`.

    Attributes
    ----------
    r_grid
        Spatial radii at which :math:`K_{st}` was evaluated, shape ``(nr,)``.
    t_grid
        Temporal lags at which :math:`K_{st}` was evaluated, shape ``(nt,)``.
    k_st
        The estimated :math:`\\hat K_{st}(r, t)`, shape ``(nr, nt)``; row
        ``i`` / column ``j`` correspond to ``r_grid[i]`` / ``t_grid[j]``.
    edge_correction
        The edge-correction mode used (``"isotropic"``, ``"translation"``,
        or ``"border"``).
    """

    r_grid: np.ndarray
    t_grid: np.ndarray
    k_st: np.ndarray
    edge_correction: str

    def l_st(self) -> np.ndarray:
        r"""Variance-stabilised transform of :math:`\hat K_{st}`.

        Extends the classical spatial :math:`L(r) = \sqrt{K(r)/\pi}`
        transform (Besag 1977) using the space-time null constant
        :math:`2\pi` in place of :math:`\pi` (since :math:`K_{st}(r,t) =
        2\pi r^2 t` under the inhomogeneous-Poisson null; see the module
        docstring derivation):

        .. math::

            L_{st}(r, t) = \sqrt{K_{st}(r, t) / (2\pi)}.

        Monotone non-decreasing in :math:`K_{st}` (the defining property
        of a variance-stabilising square-root transform); ``NaN`` entries
        in ``k_st`` (e.g. from the ``"border"`` correction with no usable
        focal events) propagate through unchanged.
        """
        K = self.k_st
        return np.where(
            np.isnan(K), np.nan, np.sqrt(np.maximum(K, 0.0) / (2.0 * np.pi))
        )


@dataclass(frozen=True)
class STEnvelopeResult:
    """Result of :func:`global_envelope_st`.

    Attributes
    ----------
    r_grid, t_grid
        The lag grids the summary statistic was evaluated on.
    observed
        The observed summary surface, shape ``(nr, nt)``.
    lo, hi
        Lower / upper global-rank envelope at the requested level, each
        shape ``(nr, nt)``.
    inside
        ``True`` iff the observed surface lies inside ``[lo, hi]`` at
        every ``(r, t)`` cell — i.e. the global test does NOT reject the
        null.
    p_interval
        Conservative / liberal global-rank p-value interval
        ``(p_lo, p_hi)`` (Myllymaki et al. 2017).
    """

    r_grid: np.ndarray
    t_grid: np.ndarray
    observed: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    inside: bool
    p_interval: tuple[float, float]


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _as_points(points: np.ndarray) -> np.ndarray:
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.ndim != 2:
        raise ValueError(f"points must be (n, 2); got shape {points.shape}")
    return points


def _as_times(times: np.ndarray) -> np.ndarray:
    return np.atleast_1d(np.asarray(times, dtype=float)).ravel()


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


def _validate_st_edge_correction(name: str) -> None:
    if name not in _VALID_ST_EDGE_CORRECTIONS:
        raise ValueError(
            "edge_correction must be one of "
            "{'isotropic','translation','border'}; "
            f"got {name!r}"
        )


def _domain_measure(domain) -> float:
    return float(np.prod([hi - lo for lo, hi in domain]))


def _intensity_at(lambda_hat, points: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Evaluate the reweighting intensity at the given events.

    ``lambda_hat`` may be (a) a callable ``(x, t) -> lambda(x, t)`` — the
    exact signature of
    :meth:`nstat.extras.spatial.st_intensity.STIntensityResult.evaluate` —
    or (b) a 1-D array aligned with ``points``/``times`` giving lambda at
    each event.
    """
    if callable(lambda_hat):
        lam = np.asarray(lambda_hat(points, times), dtype=float).ravel()
    else:
        lam = np.asarray(lambda_hat, dtype=float).ravel()
        if lam.shape[0] != points.shape[0]:
            raise ValueError(
                "lambda_hat array must align with points/times; got "
                f"{lam.shape[0]} vs {points.shape[0]}"
            )
    if np.any(lam <= 0):
        raise ValueError(
            "lambda_hat must be strictly positive (SOIRS requires lambda "
            "bounded away from zero); got a non-positive value."
        )
    return lam


def _frac_translation_interval(dt: np.ndarray, period: tuple[float, float]) -> np.ndarray:
    r"""1-D translation edge correction (Ohser 1983, specialised to an interval).

    :math:`|T| / (|T| - |dt|)`, ``+inf`` if ``|dt| >= |T|``.  Vectorised
    over ``dt``.
    """
    tlo, thi = period
    L = float(thi - tlo)
    a = np.abs(np.asarray(dt, dtype=float))
    overlap = L - a
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(overlap > 0, L / overlap, np.inf)
    return w


def _border_usable_mask_1d(
    times: np.ndarray, period: tuple[float, float], t: float
) -> np.ndarray:
    """Boolean ``(n,)`` mask of times with boundary-distance ``>= t``."""
    tlo, thi = period
    bdist = np.minimum(times - tlo, thi - times)
    return bdist >= float(t)


# ----------------------------------------------------------------------
# Inhomogeneous space-time K
# ----------------------------------------------------------------------


def k_st_inhom(
    points: np.ndarray,
    times: np.ndarray,
    lambda_hat,
    r_grid: np.ndarray,
    t_grid: np.ndarray,
    *,
    domain,
    period,
    edge_correction: str = "translation",
) -> STKResult:
    r"""Inhomogeneous space-time :math:`K`-function.

    .. math::

        \hat K_{st}(r, t) = \frac{1}{|W\times T|}
            \sum_{i\neq j}
            \frac{\mathbf 1[\|x_i-x_j\|\le r]\,
                  \mathbf 1[|t_i-t_j|\le t]\, w_{ij}}
                 {\hat\lambda(x_i,t_i)\,\hat\lambda(x_j,t_j)},

    the SOIRS-reweighted estimator of Gabriel & Diggle (2009), building on
    Diggle, Chetwynd, Haggkvist & Morris (1995).  For an inhomogeneous
    space-time Poisson process, ``K_st(r, t) = pi * r**2 * 2*t`` — the
    2-D-plus-time CSR benchmark (see the module docstring for the
    derivation).

    Parameters
    ----------
    points
        ``(n, 2)`` event spatial coordinates.
    times
        ``(n,)`` event times, aligned with ``points``.
    lambda_hat
        The reweighting intensity — a callable ``(x, t) -> lambda`` (e.g.
        :meth:`nstat.extras.spatial.st_intensity.STIntensityResult.evaluate`)
        or a per-event array.  **Use a held-out estimate** (see module
        warning).
    r_grid, t_grid
        Spatial radii / temporal lags at which to evaluate the cumulative
        statistic.
    domain
        ``((xlo, xhi), (ylo, yhi))`` rectangular spatial observation
        window.
    period
        ``(tlo, thi)`` temporal observation interval.
    edge_correction
        One of ``"isotropic"`` (Ripley 1976/1977 spatial factor paired
        with the temporal translation factor), ``"translation"``
        (Ohser 1983 spatial x 1-D translation temporal — the default),
        or ``"border"`` (Baddeley-Rubak-Turner 2015 spatial x 1-D border
        temporal).  ``"border"`` returns ``NaN`` at ``(r, t)`` cells where
        no event has both spatial and temporal boundary-distance
        sufficient to be a usable focal point, or where the eroded
        window/interval is empty.

    Returns
    -------
    STKResult

    Notes
    -----
    *Confidence: high* on the CSR benchmark (``K_st -> pi r^2 * 2t`` for
    an inhomogeneous space-time Poisson process; asserted in the
    companion tests to a few percent at moderate ``n``); same plug-in
    caveat as the static-spatial estimator.
    """
    _validate_st_edge_correction(edge_correction)
    pts = _as_points(points)
    tms = _as_times(times)
    if pts.shape[1] != 2:
        raise ValueError("k_st_inhom is implemented for d=2 (planar) spatial coordinates.")
    if pts.shape[0] != tms.shape[0]:
        raise ValueError(
            "points and times must align; got "
            f"{pts.shape[0]} points and {tms.shape[0]} times"
        )
    r_grid = np.asarray(r_grid, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    domain = _validate_domain(domain)
    period = _validate_period(period)

    n = pts.shape[0]
    if n < 2:
        return STKResult(
            r_grid, t_grid, np.zeros((len(r_grid), len(t_grid))), edge_correction
        )

    area = _domain_measure(domain)
    period_len = period[1] - period[0]
    vol = area * period_len

    lam = _intensity_at(lambda_hat, pts, tms)
    iu = np.triu_indices(n, k=1)
    d = pdist(pts)
    dtau = np.abs(tms[iu[0]] - tms[iu[1]])
    wgt = 1.0 / (lam[iu[0]] * lam[iu[1]])

    K = np.empty((len(r_grid), len(t_grid)))

    if edge_correction in ("isotropic", "translation"):
        pi_i, pj = pts[iu[0]], pts[iu[1]]
        if edge_correction == "isotropic":
            fi = np.array(
                [frac_disc_in_rect(pi_i[m], d[m], domain) for m in range(d.size)]
            )
            fj = np.array(
                [frac_disc_in_rect(pj[m], d[m], domain) for m in range(d.size)]
            )
            w_sp = 0.5 * (1.0 / fi + 1.0 / fj)
        else:  # translation — fully vectorised (Ohser 1983 closed form).
            offs = pi_i - pj
            (xlo, xhi), (ylo, yhi) = domain
            Lx, Ly = float(xhi - xlo), float(yhi - ylo)
            ax, ay = np.abs(offs[:, 0]), np.abs(offs[:, 1])
            overlap = np.maximum(Lx - ax, 0.0) * np.maximum(Ly - ay, 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                w_sp = np.where(overlap > 0, (Lx * Ly) / overlap, np.inf)

        raw_dt = tms[iu[0]] - tms[iu[1]]
        w_t = _frac_translation_interval(raw_dt, period)
        w_pair = w_sp * w_t

        for ri, r in enumerate(r_grid):
            mask_r = d <= r
            for ti, t in enumerate(t_grid):
                mask = mask_r & (dtau <= t)
                K[ri, ti] = 2.0 * np.sum(wgt[mask] * w_pair[mask]) / vol
        return STKResult(r_grid, t_grid, K, edge_correction)

    # edge_correction == "border": restrict the focal points per (r, t) cell.
    (xlo, xhi), (ylo, yhi) = domain
    usable_t_list = [_border_usable_mask_1d(tms, period, t) for t in t_grid]
    for ri, r in enumerate(r_grid):
        usable_sp = border_usable_mask(pts, domain, r)
        eff_area = max(0.0, (xhi - xlo) - 2 * r) * max(0.0, (yhi - ylo) - 2 * r)
        for ti, t in enumerate(t_grid):
            eff_period = max(0.0, period_len - 2 * t)
            if eff_area <= 0 or eff_period <= 0:
                K[ri, ti] = np.nan
                continue
            usable = usable_sp & usable_t_list[ti]
            if not usable.any():
                K[ri, ti] = np.nan
                continue
            base = (d <= r) & (dtau <= t)
            # Each ordered direction of a pair contributes independently:
            # (i, j) counts iff i is a usable focal point, (j, i) counts
            # iff j is (Baddeley-Rubak-Turner 2015, Sec. 7.4) -- the two
            # members of a pair are not, in general, both usable or both
            # unusable, so this must be checked per-direction rather than
            # testing one member and doubling.
            keep_i = usable[iu[0]] & base
            keep_j = usable[iu[1]] & base
            if not (keep_i.any() or keep_j.any()):
                K[ri, ti] = 0.0
                continue
            eff_vol = eff_area * eff_period
            K[ri, ti] = (np.sum(wgt[keep_i]) + np.sum(wgt[keep_j])) / eff_vol
    return STKResult(r_grid, t_grid, K, edge_correction)


# ----------------------------------------------------------------------
# Space-time pair correlation
# ----------------------------------------------------------------------


def pair_correlation_st(
    points: np.ndarray,
    times: np.ndarray,
    lambda_hat,
    r_grid: np.ndarray,
    t_grid: np.ndarray,
    *,
    bw_r: float | None = None,
    bw_t: float | None = None,
    domain,
    period,
) -> np.ndarray:
    r"""SOIRS-reweighted space-time pair correlation :math:`g(r, t)`.

    Direct product-kernel estimator (Moller & Ghorbani 2012 form; see the
    module docstring for the ``4 pi r`` differential normalisation
    derived from :math:`K_{st}`):

    .. math::

        \hat g(r, t) = \frac{1}{4\pi r\,|W\times T|}
            \sum_{i\neq j}
            \frac{k_{bw_r}(r - \|x_i-x_j\|)\, k_{bw_t}(t - |t_i-t_j|)}
                 {\hat\lambda(x_i,t_i)\,\hat\lambda(x_j,t_j)} .

    Under the inhomogeneous space-time Poisson null, :math:`g(r,t) = 1`
    at every ``(r, t)``; ``> 1`` indicates space-time clustering, ``< 1``
    inhibition.  No additional geometric edge correction is applied here
    (matching the default, uncorrected ``"epanechnikov"`` branch of
    :func:`nstat.extras.spatial.spatial_gof.pair_correlation` — the SOIRS
    reweighting is the only correction) — pass an ``edge_correction`` to
    :func:`k_st_inhom` instead if the cumulative statistic near the
    domain/period boundary matters.

    Parameters
    ----------
    points, times, lambda_hat, domain, period
        See :func:`k_st_inhom`.
    r_grid, t_grid
        Spatial radii / temporal lags at which to evaluate :math:`g`.
    bw_r, bw_t
        Kernel bandwidths.  ``None`` (default) uses a Stoyan-style rule
        of thumb per dimension, following the convention of
        :func:`nstat.extras.spatial.spatial_gof.pair_correlation`.

    Returns
    -------
    np.ndarray
        :math:`\hat g(r, t)`, shape ``(len(r_grid), len(t_grid))``.

    Notes
    -----
    *Confidence: high* on the homogeneous limit (:math:`g \to 1` for the
    Poisson null) and the sign of clustering/inhibition; the absolute
    scale is sensitive to ``bw_r``/``bw_t`` and the plug-in caveat in the
    module docstring.
    """
    pts = _as_points(points)
    tms = _as_times(times)
    if pts.shape[1] != 2:
        raise ValueError(
            "pair_correlation_st is implemented for d=2 (planar) spatial coordinates."
        )
    if pts.shape[0] != tms.shape[0]:
        raise ValueError(
            "points and times must align; got "
            f"{pts.shape[0]} points and {tms.shape[0]} times"
        )
    r_grid = np.asarray(r_grid, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    domain = _validate_domain(domain)
    period = _validate_period(period)

    n = pts.shape[0]
    if n < 2:
        return np.zeros((len(r_grid), len(t_grid)))

    area = _domain_measure(domain)
    period_len = period[1] - period[0]
    vol = area * period_len

    if bw_r is None:
        bw_r = float(0.1 / np.sqrt(n / area)) if area > 0 else 0.05
        bw_r = max(bw_r, float(np.mean(np.diff(r_grid))) if len(r_grid) > 1 else 0.05)
    if bw_t is None:
        bw_t = float(0.1 / np.sqrt(n / period_len)) if period_len > 0 else 0.05
        bw_t = max(bw_t, float(np.mean(np.diff(t_grid))) if len(t_grid) > 1 else 0.05)

    lam = _intensity_at(lambda_hat, pts, tms)
    iu = np.triu_indices(n, k=1)
    d = pdist(pts)
    dtau = np.abs(tms[iu[0]] - tms[iu[1]])
    wgt = 1.0 / (lam[iu[0]] * lam[iu[1]])

    g = np.empty((len(r_grid), len(t_grid)))
    for ri, r in enumerate(r_grid):
        norm_r = 4.0 * np.pi * r * vol
        if norm_r <= 0:
            g[ri, :] = 0.0
            continue
        ker_r = epanechnikov((d - r) / bw_r) / bw_r
        for ti, t in enumerate(t_grid):
            ker_t = epanechnikov((dtau - t) / bw_t) / bw_t
            g[ri, ti] = 2.0 * np.sum(ker_r * ker_t * wgt) / norm_r
    return g


# ----------------------------------------------------------------------
# Global-rank envelope (Monte-Carlo)
# ----------------------------------------------------------------------


def global_envelope_st(
    points: np.ndarray,
    times: np.ndarray,
    lambda_hat,
    r_grid: np.ndarray,
    t_grid: np.ndarray,
    *,
    n_sim: int = 199,
    statistic: str = "kst",
    alpha: float = 0.05,
    domain,
    period,
    rng: np.random.Generator | None = None,
    edge_correction: str = "translation",
) -> STEnvelopeResult:
    r"""Monte-Carlo global-rank envelope test against the space-time inhomogeneous null.

    Simulates ``n_sim`` realizations of the fitted inhomogeneous
    space-time Poisson process (by thinning a dominating homogeneous
    space-time process at the maximum intensity over a reference grid),
    computes the chosen summary statistic on the flattened ``(r, t)``
    surface of each, and builds the global-rank envelope of Myllymaki et
    al. (2017) — reusing
    :func:`nstat.extras.spatial._envelopes.global_rank_envelope`, the
    same machinery :func:`nstat.extras.spatial.spatial_gof.global_envelope`
    uses for the static-spatial case.

    Parameters
    ----------
    points, times, lambda_hat, domain, period
        See :func:`k_st_inhom`.  ``lambda_hat`` (held-out) is used both to
        reweight the statistic and as the simulation intensity.
    r_grid, t_grid
        Spatial radii / temporal lags defining the summary surface.
    n_sim
        Number of Monte-Carlo null simulations (199 -> 95% global level
        with ``alpha = 0.05``).
    statistic
        ``"kst"`` for :math:`K_{st}(r,t)` (default), ``"lst"`` for the
        variance-stabilised :math:`L_{st}(r,t)` (:meth:`STKResult.l_st`),
        or ``"gst"`` for the pair correlation :math:`g(r,t)`
        (:func:`pair_correlation_st` — always uncorrected; ``edge_correction``
        is ignored for this choice).
    alpha
        Global type-I error level.
    rng
        NumPy random generator for the simulations.
    edge_correction
        Forwarded to :func:`k_st_inhom` when ``statistic`` is ``"kst"`` or
        ``"lst"``.

    Returns
    -------
    STEnvelopeResult

    Notes
    -----
    Honours the **plug-in caveat**: if ``lambda_hat`` was fit to the same
    pattern, the envelope coverage is optimistic.  *Confidence: high* on
    the mechanics (direct reuse of the validated static-spatial rank
    -envelope machinery on a flattened surface); coverage validity is
    conditional on the held-out reweighting.
    """
    pts = _as_points(points)
    tms = _as_times(times)
    if pts.shape[1] != 2:
        raise ValueError(
            "global_envelope_st is implemented for d=2 (planar) spatial coordinates."
        )
    if pts.shape[0] != tms.shape[0]:
        raise ValueError(
            "points and times must align; got "
            f"{pts.shape[0]} points and {tms.shape[0]} times"
        )
    r_grid = np.asarray(r_grid, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    domain = _validate_domain(domain)
    period = _validate_period(period)
    rng = np.random.default_rng() if rng is None else rng

    area = _domain_measure(domain)
    period_len = period[1] - period[0]
    vol = area * period_len
    nr, nt = len(r_grid), len(t_grid)

    def _kst(X, T):
        return k_st_inhom(
            X, T, lambda_hat, r_grid, t_grid,
            domain=domain, period=period, edge_correction=edge_correction,
        ).k_st

    stat_fns = {
        "kst": _kst,
        "lst": lambda X, T: k_st_inhom(
            X, T, lambda_hat, r_grid, t_grid,
            domain=domain, period=period, edge_correction=edge_correction,
        ).l_st(),
        "gst": lambda X, T: pair_correlation_st(
            X, T, lambda_hat, r_grid, t_grid, domain=domain, period=period,
        ),
    }
    if statistic not in stat_fns:
        raise ValueError(f"statistic must be one of {list(stat_fns)}; got {statistic!r}")
    stat_fn = stat_fns[statistic]

    observed = stat_fn(pts, tms)

    # Reference grid for the dominating intensity used to thin the
    # simulation proposal (mirrors spatial_gof.global_envelope's approach,
    # extended with a time axis).
    n_ref = 12
    (xlo, xhi), (ylo, yhi) = domain
    tlo, thi = period
    ax = np.linspace(xlo, xhi, n_ref)
    ay = np.linspace(ylo, yhi, n_ref)
    at = np.linspace(tlo, thi, n_ref)
    mesh_x, mesh_y = np.meshgrid(ax, ay, indexing="xy")
    ref_xy = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    ref_xy_rep = np.repeat(ref_xy, n_ref, axis=0)
    ref_t_rep = np.tile(at, ref_xy.shape[0])
    lam_ref = _intensity_at(lambda_hat, ref_xy_rep, ref_t_rep)
    lam_max = float(np.max(lam_ref))

    sims = np.empty((n_sim, nr, nt))
    for s in range(n_sim):
        n_prop = rng.poisson(lam_max * vol)
        cand_xy = np.column_stack([
            xlo + (xhi - xlo) * rng.uniform(size=n_prop),
            ylo + (yhi - ylo) * rng.uniform(size=n_prop),
        ])
        cand_t = tlo + (thi - tlo) * rng.uniform(size=n_prop)
        if n_prop > 0:
            lam_cand = _intensity_at(lambda_hat, cand_xy, cand_t)
            keep = rng.uniform(size=n_prop) < (lam_cand / lam_max)
        else:
            keep = np.array([], dtype=bool)
        Xs, Ts = cand_xy[keep], cand_t[keep]
        sims[s] = stat_fn(Xs, Ts)

    observed_flat = observed.ravel()
    sims_flat = sims.reshape(n_sim, nr * nt)
    env = global_rank_envelope(
        observed_flat, sims_flat, np.arange(nr * nt, dtype=float), alpha=alpha
    )

    return STEnvelopeResult(
        r_grid=r_grid,
        t_grid=t_grid,
        observed=observed.reshape(nr, nt),
        lo=env.lo.reshape(nr, nt),
        hi=env.hi.reshape(nr, nt),
        inside=env.inside,
        p_interval=env.p_interval,
    )
