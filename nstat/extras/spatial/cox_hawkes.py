r"""Cox-Hawkes: an LGCP background modulating spatial-Hawkes self-excitation.

Pure NumPy/SciPy.  Composes the two sibling modules built alongside this
one — :mod:`nstat.extras.spatial.lgcp_st` (a Kronecker-Laplace
spatiotemporal log-Gaussian Cox process) and
:mod:`nstat.extras.spatial.spatial_hawkes` (a space-time branching-EM
Hawkes fit) — into the doubly-stochastic model of Miscouridou, Bhatt,
Mohler, Flaxman & Bhamidi (2022), *Cox-Hawkes: doubly stochastic
spatiotemporal Poisson processes*, Transactions on Machine Learning
Research:

.. math::

    \lambda(x, t) = \mu(x, t) + \sum_{t_j < t} K \, g(t - t_j) \, h(x - x_j),

where :math:`\mu(x, t)` is an inhomogeneous log-Gaussian Cox process
background (rather than the homogeneous :math:`\mu / |W|` used by
:mod:`~nstat.extras.spatial.spatial_hawkes`) and :math:`g, h` are the
same normalised exponential-temporal x isotropic-Gaussian-spatial
offspring kernel used there.

There is no public Python implementation of Cox-Hawkes to port from —
the estimator below was derived directly from the model's generative
description in Miscouridou et al. (2022) plus the branching/declustering
machinery already established in this package (Veen & Schoenberg 2008;
Zhuang, Ogata & Vere-Jones 2002), **not** from that paper's own (Numpyro
full-joint-MCMC) inference code, which this module does not attempt to
reproduce.  This is the tractable *alternating* estimator: an outer loop
that alternates a background LGCP refit with a spatial-Hawkes-style
closed-form M-step, rather than joint MCMC.

Algorithm
---------

1. **Init.**  Fit :func:`~nstat.extras.spatial.lgcp_st.lgcp_st_fit`
   treating every event as background, giving an initial
   :math:`\hat\mu(x, t)`.
2. **E-step (declustering).**  For each event :math:`i`, soft
   responsibilities

   .. math::

      p_{i,\text{bg}} \propto \hat\mu(x_i, t_i), \qquad
      p_{ij} \propto K \, g(t_i - t_j) \, h(x_i - x_j) \quad (j < i),

   normalised so each row sums to 1 — the same responsibility structure
   as :func:`~nstat.extras.spatial.spatial_hawkes.em_spatial_hawkes`'s
   E-step, with the homogeneous :math:`\mu / |W|` term replaced by the
   inhomogeneous :math:`\hat\mu(x_i, t_i)`.
3. **M-step, background.**  Refit the LGCP on the *exact* expected
   sufficient statistic rather than a Monte-Carlo surrogate: bin each
   event's background responsibility mass :math:`p_{i,\text{bg}}` into
   its grid cell to form a **fractional count histogram**
   :math:`y_{\text{bg}}[m] = \sum_{i \in \text{cell }m} p_{i,\text{bg}}`,
   then run the same Newton/IRLS + Kronecker-CG Laplace-mode solve
   :func:`~nstat.extras.spatial.lgcp_st.lgcp_st_fit` uses internally,
   but against that fractional histogram instead of an integer one
   (:func:`_fit_weighted_background`).  See "Weighted-histogram
   background M-step" below for why this deterministic, zero-Monte
   -Carlo-variance refit is used instead of the classical stochastic
   -declustering *thinning* estimator of Zhuang, Ogata & Vere-Jones
   (2002).
4. **M-step, excitation.**  Closed-form weighted MLE for
   :math:`(K, c, \sigma)` from the triggered responsibilities
   :math:`p_{ij}`, using the *same* derivations as
   :func:`~nstat.extras.spatial.spatial_hawkes.em_spatial_hawkes`'s
   M-step (re-implemented directly here, since the background term is
   no longer a scalar the closed forms can absorb algebraically — see
   that module's docstring for the derivation of each update).
5. Iterate 2-4 until the log-likelihood changes by less than ``tol`` or
   ``max_outer`` is reached.

Weighted-histogram background M-step
--------------------------------------

:func:`~nstat.extras.spatial.lgcp_st.lgcp_st_fit`'s Poisson data term is
a per-cell **count histogram**,
:math:`y_m \sim \mathrm{Poisson}(v\, e^{f_m})`.  Given the E-step's
background responsibilities :math:`p_{i,\text{bg}} \in (0, 1)`, the
*exact* expected sufficient statistic for the background sub-problem's
complete-data log-likelihood is the fractional count
:math:`y_{\text{bg}}[m] = \sum_{i \in \text{cell }m} p_{i,\text{bg}}` —
binning each event's responsibility *mass*, not the event itself.
:func:`_fit_weighted_background` builds exactly that histogram (a
weighted ``numpy.histogramdd``, mirroring
:func:`~nstat.extras.spatial._kernels.bin_counts`'s axis convention)
and reuses :mod:`~nstat.extras.spatial.lgcp_st`'s public
:class:`~nstat.extras.spatial.lgcp_st.LGCPSTResult` result type and its
private Kronecker/Newton building blocks (``_axis_grid``,
``_make_kinv_matvec``, ``_spatial_grid``, ``_to_txy`` — imported, not
duplicated; the same cross-module private-helper reuse convention
:mod:`~nstat.extras.spatial.gibbs` already uses for
:mod:`~nstat.extras.spatial.cluster_cox`'s ``_validate_window`` /
``_window_area``) to run the *same* Newton/IRLS + Kronecker-CG
Laplace-mode solve that function uses, just against a fractional
rather than integer count vector — so :func:`_fit_weighted_background`
re-orchestrates that numerics with different input data rather than
reimplementing it.

This strictly dominates two alternatives considered and rejected during
development:

- **Stochastic-declustering thinning** (Zhuang, Ogata & Vere-Jones
  2002) — one Bernoulli keep/drop draw per event at probability
  :math:`p_{i,\text{bg}}`, then an *integer*-count refit on the
  survivors.  ``E[bin_count]`` of the thinned stream equals the desired
  weighted grid count, but only *in expectation*: a single draw is
  noisy, injecting Monte-Carlo variance into every outer iteration's
  M-step and making the scheme a stochastic-EM variant (Celeux &
  Diebolt 1985) whose log-likelihood trace is not guaranteed
  non-decreasing.  The weighted histogram above uses the
  responsibilities directly — zero Monte-Carlo variance, no distortion
  of the LGCP's prior-vs-likelihood balance.
- **Deterministic quantized replication** — duplicate event :math:`i`
  ``round(p_i * Q)`` times for some large integer :math:`Q`.  This
  multiplies the effective sample size feeding the Poisson likelihood
  by :math:`Q` **without** rescaling the fixed Matern-GP prior
  precision :math:`K^{-1}` to match, so the replicated-and-then
  -corrected posterior mode maximizes ``prior_term(f) + Q *
  likelihood_term(f)``, not ``prior_term(f) + likelihood_term(f)`` — a
  real, non-cosmetic under-regularization bias (worse the larger
  :math:`Q`), unlike the exact weighted histogram, which needs no such
  correction.

Because both M-steps here (this background refit, and the closed-form
excitation MLE below) are now exact maximizers of the same
declustering-EM's expected complete-data log-likelihood given the
current E-step responsibilities, the standard (generalized) EM
monotonicity argument applies to the *observed* log-likelihood trace,
up to the residual numerical tolerance of the background refit's own
inner Newton/CG loop (``tol=1e-8`` on the Newton step, ``rtol=1e-10``
per CG solve) — see the companion test's documented (and now much
tighter) slack.

Excitation-side compensator note
----------------------------------

The full-model compensator is
:math:`\int\!\!\int \mu(x,t)\,dx\,dt + K \sum_j (1 - e^{-c(T - t_j)})`
— the background term is the LGCP's total posterior-mean mass over the
grid (no closed form needed since the grid quadrature is already
computed by :func:`~nstat.extras.spatial.lgcp_st.lgcp_st_fit`), and the
excitation term is *exactly* the same right-censoring compensator as
:mod:`~nstat.extras.spatial.spatial_hawkes` (the spatial kernel
integrates to 1 over all of :math:`\mathbb{R}^2`, contributing no
boundary term).

References
----------
Miscouridou X, Bhatt S, Mohler G, Flaxman S, Bhamidi S (2022).
*Cox-Hawkes: doubly stochastic spatiotemporal Poisson processes.*
Transactions on Machine Learning Research.

Veen A, Schoenberg FP (2008). *Estimation of space-time branching
process models in seismology using an EM-type algorithm.* Journal of
the American Statistical Association 103(482):614-624.  (Source of the
excitation M-step closed forms re-implemented in step 4 above; see
:mod:`nstat.extras.spatial.spatial_hawkes` for the full derivation.)

Zhuang J, Ogata Y, Vere-Jones D (2002). *Stochastic declustering of
space-time earthquake occurrences.* Journal of the American Statistical
Association 97(458):369-380.  (Source of the declustering
responsibility framework generalized by the E-step in step 2 above;
this module replaces their original stochastic-thinning background
M-step with the exact weighted-histogram refit of step 3 — see
"Weighted-histogram background M-step" above.)

Celeux G, Diebolt J (1985). *The SEM algorithm: a probabilistic teacher
algorithm derived from the EM algorithm for the mixture problem.*
Computational Statistics Quarterly 2:73-82.  (Stochastic-EM framing for
why the thinning-based M-step considered and rejected above would not
have given a strictly monotone log-likelihood trace.)

Møller J, Rasmussen JG (2005). *Perfect simulation of Hawkes processes.*
Advances in Applied Probability 37(3):629-646.  (Branching/Poisson
-cluster offspring-cascade representation reused by
:func:`simulate_cox_hawkes`, exactly as in
:func:`~nstat.extras.spatial.spatial_hawkes.simulate_spatial_hawkes`.)

Lewis PAW, Shedler GS (1979). *Simulation of nonhomogeneous Poisson
processes by thinning.* Naval Research Logistics Quarterly
26(3):403-413.  (Dominating-rate thinning used by
:func:`simulate_cox_hawkes` to draw background immigrants from an
arbitrary intensity function.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, cg

from nstat.extras.spatial._kernels import matern_covariance
from nstat.extras.spatial.lgcp_st import (
    LGCPSTResult,
    lgcp_st_fit,
    _CG_MAXITER,
    _CG_RTOL,
    _ETA_CLIP,
    _N_VARIANCE_PROBES,
    _VARIANCE_RNG_SEED,
    _axis_grid,
    _make_kinv_matvec,
    _spatial_grid,
    _to_txy,
)
from nstat.extras.spatial.spatial_hawkes import SpatialHawkesSpec

__all__ = [
    "CoxHawkesResult",
    "fit_cox_hawkes",
    "simulate_cox_hawkes",
]

_TINY = np.finfo(np.float64).tiny


# ----------------------------------------------------------------------
# Domain / period validation (local convention: each spatial module
# carries its own small copy rather than importing a sibling's private
# helper — see e.g. spatial_hawkes.py's and st_intensity.py's own
# `_validate_domain`).
# ----------------------------------------------------------------------


def _validate_domain(
    domain: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Validate a rectangular ``((xlo, xhi), (ylo, yhi))`` domain; return it as floats.

    Accepts either a tuple or a list of two ``(lo, hi)`` pairs — the
    sibling spatial modules (``st_intensity``, ``spatiotemporal_gof``,
    ``spatial_hawkes``) accept lists via flexible unpacking, so a value
    valid for one must be valid for all.  Coerced to a tuple-of-tuples
    of floats on return regardless of the input container type.
    """
    valid_shape = (
        isinstance(domain, (tuple, list))
        and len(domain) == 2
        and all(isinstance(d, (tuple, list)) and len(d) == 2 for d in domain)
    )
    if not valid_shape:
        raise ValueError(
            "domain must be a rectangular ((xlo, xhi), (ylo, yhi)) sequence; "
            f"got {domain!r}"
        )
    (xlo, xhi), (ylo, yhi) = domain
    xlo, xhi, ylo, yhi = float(xlo), float(xhi), float(ylo), float(yhi)
    if not (xhi > xlo):
        raise ValueError(f"domain x-range must satisfy xhi > xlo; got ({xlo}, {xhi})")
    if not (yhi > ylo):
        raise ValueError(f"domain y-range must satisfy yhi > ylo; got ({ylo}, {yhi})")
    return ((xlo, xhi), (ylo, yhi))


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CoxHawkesResult:
    """Fitted Cox-Hawkes model: an LGCP background plus spatial-Hawkes excitation.

    Attributes
    ----------
    background : LGCPSTResult
        The final-iteration fitted spatiotemporal LGCP background (see
        :mod:`nstat.extras.spatial.lgcp_st`); call
        ``background.intensity_fn()`` for the bare background rate, or
        ``background.rate_map(t)`` for a credible-band spatial slice.
    K_branch_hat : float
        Fitted branching ratio (expected direct offspring per event) of
        the spatial-Hawkes excitation term.
    c_hat : float
        Fitted temporal decay of the exponential offspring kernel.
    sigma_space_hat : float
        Fitted spatial scale of the isotropic-Gaussian offspring kernel.
    background_fraction : float
        Mean of the final E-step's background responsibilities
        :math:`p_{i,\\text{bg}}` across all events — the estimated
        fraction of events attributable to the background rather than
        to self-excitation.
    log_likelihood_trace : np.ndarray
        Per-outer-iteration log-likelihood, including the initial value
        at index 0 (length ``n_outer + 1``).
    n_outer : int
        Number of outer (declustering) iterations performed.
    converged : bool
        True iff the relative log-likelihood change fell below ``tol``
        before ``max_outer`` was reached.
    event_points, event_times : np.ndarray
        The fitted event stream (row-aligned), stored so
        :meth:`intensity_fn` can evaluate the *exact* excitation term
        (summed over the observed triggering history) rather than a
        mean-field approximation.  Not part of the architect's minimal
        field list; added because the brief's ``intensity_fn()``
        contract ("background + expected excitation") needs the
        observed history to be exact rather than a stationary
        :math:`1/(1-K)` mean-field surrogate.
    """

    background: LGCPSTResult
    K_branch_hat: float
    c_hat: float
    sigma_space_hat: float
    background_fraction: float
    log_likelihood_trace: np.ndarray
    n_outer: int
    converged: bool
    event_points: np.ndarray
    event_times: np.ndarray

    def intensity_fn(self) -> Callable[[np.ndarray, "float | np.ndarray"], np.ndarray]:
        r"""Return a callable ``fn(X, t) -> total rate`` (background + excitation).

        Parameters of the returned callable
        ------------------------------------
        X : array_like, shape ``(n, 2)`` (or ``(2,)`` for one point)
            Spatial query coordinates.
        t : float or array_like, shape ``(n,)``
            Query time(s).  A scalar broadcasts to every row of ``X``.

        Returns
        -------
        np.ndarray
            ``(n,)`` total conditional intensity
            :math:`\hat\mu(x, t) + \sum_{t_j < t} \hat K\, g(t - t_j)\,
            h(x - x_j)`, summed over the *observed* fitted event stream
            (:attr:`event_points`, :attr:`event_times`) — the exact
            fitted conditional intensity, not a mean-field average.
        """
        bg_fn = self.background.intensity_fn()
        ev_x = self.event_points
        ev_t = self.event_times
        K = float(self.K_branch_hat)
        c = float(self.c_hat)
        sigma = float(self.sigma_space_hat)

        def _fn(X, t):
            X = np.atleast_2d(np.asarray(X, dtype=float))
            n_q = X.shape[0]
            t_arr = np.asarray(t, dtype=float)
            if t_arr.ndim == 0:
                t_arr = np.full(n_q, float(t_arr))
            else:
                t_arr = np.broadcast_to(t_arr.reshape(-1), (n_q,))

            bg_vals = bg_fn(X, t_arr)

            dt = t_arr[:, None] - ev_t[None, :]
            mask = dt > 0.0
            dxy = X[:, None, :] - ev_x[None, :, :]
            dr2 = np.sum(dxy * dxy, axis=-1)
            g = np.where(mask, c * np.exp(-c * np.where(mask, dt, 0.0)), 0.0)
            h = np.exp(-dr2 / (2.0 * sigma * sigma)) / (2.0 * np.pi * sigma * sigma)
            trig = K * np.sum(g * h, axis=1)
            return bg_vals + trig

        return _fn


# ----------------------------------------------------------------------
# Shared E-step / log-likelihood machinery
# ----------------------------------------------------------------------


def _background_total_mass(bg: LGCPSTResult) -> float:
    """Total posterior-mean background mass over the whole grid (``int mu dx dt``)."""
    mean_rate = np.exp(bg.f_mode + 0.5 * np.clip(bg.f_var, 0.0, None))
    return float(np.sum(mean_rate) * bg.cell_volume)


def _e_step(
    points: NDArray[np.float64],
    times: NDArray[np.float64],
    mask: NDArray[np.bool_],
    dt_full: NDArray[np.float64],
    dr2_full: NDArray[np.float64],
    bg: LGCPSTResult,
    K_branch: float,
    c: float,
    sigma: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Soft background/triggered responsibilities at the current parameters.

    Mirrors :mod:`nstat.extras.spatial.spatial_hawkes`'s E-step, with the
    homogeneous ``mu / area`` term replaced by the inhomogeneous LGCP
    background evaluated pointwise at each event.
    """
    mu_events = bg.intensity_fn()(points, times)
    g = np.where(mask, c * np.exp(-c * dt_full), 0.0)
    h = np.where(
        mask,
        np.exp(-dr2_full / (2.0 * sigma * sigma)) / (2.0 * np.pi * sigma * sigma),
        0.0,
    )
    triggering = K_branch * g * h
    lam = mu_events + triggering.sum(axis=1)
    lam_safe = np.maximum(lam, _TINY)
    p_diag = mu_events / lam_safe
    p_offdiag = triggering / lam_safe[:, None]
    return p_diag, p_offdiag, lam_safe


def _log_likelihood(
    lam_safe: NDArray[np.float64],
    bg: LGCPSTResult,
    K_branch: float,
    c: float,
    T: float,
    times: NDArray[np.float64],
) -> float:
    """Full Cox-Hawkes log-likelihood at the current fit (see module docstring)."""
    ll_events = float(np.sum(np.log(lam_safe)))
    ll_compensator = _background_total_mass(bg) + K_branch * float(
        np.sum(1.0 - np.exp(-c * (T - times)))
    )
    return ll_events - ll_compensator


# ----------------------------------------------------------------------
# Weighted-histogram background M-step (see module docstring,
# "Weighted-histogram background M-step").
# ----------------------------------------------------------------------


def _bin_weighted_counts(
    points3d: NDArray[np.float64],
    edges: list[NDArray[np.float64]],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fractional per-cell counts: a *weighted* 3-D histogram.

    Same axis-order convention as
    :func:`nstat.extras.spatial._kernels.bin_counts` (``numpy.histogramdd``
    plus a leading-axes ``swapaxes`` to match the ``meshgrid('xy')``
    row-major layout that :mod:`~nstat.extras.spatial.lgcp_st` uses
    internally), but with each event contributing its ``weights`` mass
    to its cell instead of a unit count.  ``bin_counts`` itself has no
    ``weights`` parameter (out of this builder's scope to add), so this
    is a minimal weighted sibling rather than a modification of it.
    """
    if points3d.shape[0] == 0:
        n_cells = int(np.prod([len(e) - 1 for e in edges]))
        return np.zeros(n_cells, dtype=float)
    H, _ = np.histogramdd(points3d, bins=edges, weights=weights)
    if H.ndim >= 2:
        H = np.swapaxes(H, 0, 1)
    return H.ravel().astype(float)


def _fit_weighted_background(
    points: NDArray[np.float64],
    times: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    domain: tuple[tuple[float, float], tuple[float, float]],
    period: tuple[float, float],
    grid: tuple[int, int, int],
    length_scale_space: float = 0.12,
    length_scale_time: float = 0.1,
    nu: float = 1.5,
    variance: float = 1.0,
    prior_mean: float | None = None,
    max_iter: int = 50,
    tol: float = 1e-8,
    jitter: float = 1e-6,
) -> LGCPSTResult:
    r"""Exact weighted-histogram LGCP background M-step.

    Re-orchestrates :func:`~nstat.extras.spatial.lgcp_st.lgcp_st_fit`'s
    Newton/IRLS + Kronecker-CG Laplace-mode solve against a *fractional*
    count histogram :math:`y_{\text{bg}}[m] = \sum_{i \in \text{cell }m}
    \text{weights}_i` instead of an integer one, so a per-event soft
    background responsibility can drive the refit exactly (zero
    Monte-Carlo variance) rather than via stochastic thinning — see the
    module docstring's "Weighted-histogram background M-step" for the
    derivation and the rejected alternatives.

    Every keyword argument not documented below (``length_scale_space``
    through ``jitter``) has exactly the same meaning and default as the
    identically-named parameter of
    :func:`~nstat.extras.spatial.lgcp_st.lgcp_st_fit`; only the data
    term differs (fractional vs. integer counts).

    Parameters
    ----------
    points : (n, 2) float64 ndarray
        Event locations.
    times : (n,) float64 ndarray
        Event times, row-aligned with ``points``.
    weights : (n,) float64 ndarray
        Per-event background responsibility :math:`p_{i,\text{bg}} \in
        [0, 1]`; binned as fractional mass rather than thinned.

    Returns
    -------
    LGCPSTResult
        Same result type :func:`~nstat.extras.spatial.lgcp_st.lgcp_st_fit`
        returns, so callers (:func:`fit_cox_hawkes`,
        :class:`CoxHawkesResult`) need not distinguish the two.

    Notes
    -----
    Not part of the public API (leading underscore, not in
    :data:`__all__`) — an internal building block of
    :func:`fit_cox_hawkes`'s M-step.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    t_arr = np.asarray(times, dtype=float).reshape(-1)
    w_arr = np.asarray(weights, dtype=float).reshape(-1)

    (xlo, xhi), (ylo, yhi) = domain
    tlo, thi = period
    grid_norm = tuple(int(g) for g in grid)
    Gx, Gy, Gt = grid_norm

    cx, ex, dx = _axis_grid(xlo, xhi, Gx)
    cy, ey, dy = _axis_grid(ylo, yhi, Gy)
    ct, et, dt = _axis_grid(tlo, thi, Gt)
    cell_volume = float(dx * dy * dt)

    grid_x = _spatial_grid(cx, cy)
    grid_t = ct

    # Bin events onto the 3-D (x, y, t) grid, weighted by their current
    # background responsibility -- the exact expected sufficient
    # statistic for this M-step (see module docstring).
    pts3d = np.column_stack([pts[:, 0], pts[:, 1], t_arr])
    y_flat = _bin_weighted_counts(pts3d, [ex, ey, et], w_arr)
    shape = (Gy, Gx, Gt)
    M = Gy * Gx * Gt

    axis_variance = float(variance) ** (1.0 / 3.0)
    Kx = matern_covariance(cx.reshape(-1, 1), length_scale=length_scale_space,
                            variance=axis_variance, nu=nu, jitter=jitter)
    Ky = matern_covariance(cy.reshape(-1, 1), length_scale=length_scale_space,
                            variance=axis_variance, nu=nu, jitter=jitter)
    Kt = matern_covariance(ct.reshape(-1, 1), length_scale=length_scale_time,
                            variance=axis_variance, nu=nu, jitter=jitter)

    ly, Qy = np.linalg.eigh(Ky)
    lx, Qx = np.linalg.eigh(Kx)
    lt, Qt = np.linalg.eigh(Kt)
    inv_eig = 1.0 / (ly[:, None, None] * lx[None, :, None] * lt[None, None, :])
    inv_eig_flat = inv_eig.reshape(-1)
    kinv_matvec = _make_kinv_matvec(Qy, Qx, Qt, inv_eig_flat, shape)

    total_volume = float((xhi - xlo) * (yhi - ylo) * (thi - tlo))
    if prior_mean is None:
        # Same log-normal mean correction as lgcp_st_fit's default, but
        # driven by the *effective* (weighted) event count sum(weights)
        # rather than the raw event count, since y_flat sums to the
        # former, not the latter.
        n_eff = max(float(np.sum(w_arr)), 1.0)
        m0 = float(np.log(n_eff / total_volume) - 0.5 * variance)
    else:
        m0 = float(prior_mean)

    # ----- Newton/IRLS Laplace-mode solve (identical control flow to
    # lgcp_st_fit's, replicated here per the module docstring's
    # "Weighted-histogram background M-step" -- the numerics it calls
    # are imported, not duplicated). -----
    f = np.full(M, m0)
    x_prev = np.zeros(M)
    converged = False
    n_iter = max_iter
    for it in range(max_iter):
        f_clip = np.clip(f, -_ETA_CLIP, _ETA_CLIP)
        lam = cell_volume * np.exp(f_clip)
        rhs = lam * (f - m0) + (y_flat - lam)

        def _precision_matvec(v, _lam=lam):
            return kinv_matvec(v) + _lam * v

        op = LinearOperator((M, M), matvec=_precision_matvec, dtype=float)
        x, _info = cg(op, rhs, x0=x_prev, rtol=_CG_RTOL, maxiter=_CG_MAXITER)
        f_new = m0 + x
        delta = np.max(np.abs(f_new - f))
        x_prev = x
        f = f_new
        if delta < tol:
            converged = True
            n_iter = it + 1
            break

    # Posterior variance diagonal at the converged mode: same matrix-free
    # Hutchinson stochastic-diagonal estimator lgcp_st_fit uses (fixed
    # seed -- see lgcp_st.py's _VARIANCE_RNG_SEED).
    f_clip = np.clip(f, -_ETA_CLIP, _ETA_CLIP)
    lam_final = cell_volume * np.exp(f_clip)

    def _precision_matvec_final(v, _lam=lam_final):
        return kinv_matvec(v) + _lam * v

    op_final = LinearOperator((M, M), matvec=_precision_matvec_final, dtype=float)
    var_rng = np.random.default_rng(_VARIANCE_RNG_SEED)
    diag_acc = np.zeros(M)
    for _ in range(_N_VARIANCE_PROBES):
        z = var_rng.choice(np.array([-1.0, 1.0]), size=M)
        xz, _info = cg(op_final, z, rtol=1e-8, maxiter=_CG_MAXITER)
        diag_acc += z * xz
    v_flat = np.clip(diag_acc / _N_VARIANCE_PROBES, 0.0, None)

    counts_out = _to_txy(y_flat, Gy, Gx, Gt)
    f_mode_out = _to_txy(f, Gy, Gx, Gt)
    f_var_out = _to_txy(v_flat, Gy, Gx, Gt)

    return LGCPSTResult(
        grid_x=grid_x,
        grid_t=grid_t,
        counts=counts_out,
        f_mode=f_mode_out,
        f_var=f_var_out,
        cell_volume=cell_volume,
        n_iter=n_iter,
        converged=converged,
    )


# ----------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------


def fit_cox_hawkes(
    points: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    domain: tuple[tuple[float, float], tuple[float, float]],
    period: tuple[float, float],
    grid: tuple[int, int, int] = (24, 24, 24),
    length_scale_space: float = 0.12,
    length_scale_time: float = 0.1,
    max_outer: int = 20,
    tol: float = 1e-4,
    hawkes_spec: SpatialHawkesSpec | None = None,
) -> CoxHawkesResult:
    r"""Alternating estimator for an LGCP-background Cox-Hawkes process.

    See the module docstring for the full algorithm (init -> E-step
    declustering -> background LGCP refit -> excitation closed-form
    M-step, iterated to convergence).

    Deterministic: the background M-step below is an exact
    weighted-histogram LGCP refit (see the module docstring's
    "Weighted-histogram background M-step"), not the stochastic
    -declustering thinning of an earlier design, so ``fit_cox_hawkes``
    has no random-number-generator parameter — repeated calls on the
    same inputs return bit-identical results.

    Parameters
    ----------
    points : (n, 2) float64 ndarray
        Event locations, row-aligned with ``times``.
    times : (n,) float64 ndarray
        Sorted ascending event times.
    domain : ((xlo, xhi), (ylo, yhi))
        Rectangular spatial analysis window, forwarded to
        :func:`~nstat.extras.spatial.lgcp_st.lgcp_st_fit`.
    period : (tlo, thi)
        Analysis time window; ``thi`` is also used as the Hawkes
        observation horizon ``T`` (must exceed ``times[-1]``).
    grid : (Gx, Gy, Gt)
        LGCP grid resolution, forwarded unchanged to
        :func:`~nstat.extras.spatial.lgcp_st.lgcp_st_fit` on *every*
        outer iteration (the background is refit from scratch each
        time — see the "Runtime caveat" in the companion contract
        summary for the resulting cost).
    length_scale_space, length_scale_time : float
        Matern range parameters for the LGCP background prior, forwarded
        to every background refit.  The defaults (0.12, 0.1) are
        calibrated for a unit-square ``domain``; for a physically-sized
        domain (e.g. an 8 mm array) scale these to the domain extent, or
        fit on a normalised ``((0, 1), (0, 1))`` domain and rescale the
        rate afterward — otherwise the background is over- or
        under-smoothed relative to the data scale.
    max_outer : int
        Maximum number of outer (declustering) iterations.
    tol : float
        Relative log-likelihood convergence tolerance.
    hawkes_spec : SpatialHawkesSpec or None
        Reused from :mod:`nstat.extras.spatial.spatial_hawkes` purely
        for its ``K0``/``c0``/``sigma0`` initial-guess fields (its
        ``mu0``/``max_iter``/``tol`` fields are ignored here — this
        module's background has no scalar ``mu``, and the outer-loop
        iteration budget/tolerance are governed by ``max_outer``/``tol``
        above instead).  Defaults to ``SpatialHawkesSpec()``.

    Returns
    -------
    CoxHawkesResult

    Raises
    ------
    ValueError
        If ``points``/``times`` are malformed, fewer than 2 events are
        given, ``times`` is not sorted ascending, or ``period``'s upper
        bound does not exceed the last event time.
    """
    points = np.asarray(points, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    if times.ndim != 1:
        raise ValueError(f"times must be 1-D; got shape {times.shape}")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (n, 2); got {points.shape}")
    if points.shape[0] != times.shape[0]:
        raise ValueError(
            f"points has {points.shape[0]} rows but times has "
            f"{times.shape[0]} entries"
        )

    n = times.size
    if n < 2:
        raise ValueError(
            f"need at least 2 events to fit a Cox-Hawkes model; got {n}"
        )
    if np.any(np.diff(times) < 0):
        raise ValueError("times must be sorted ascending")

    domain = _validate_domain(domain)
    period = (float(period[0]), float(period[1]))
    if not (period[1] > period[0]):
        raise ValueError(f"period must satisfy thi > tlo; got {period}")
    T = period[1]
    if not (T > times[-1]):
        raise ValueError(
            f"period upper bound {T} must exceed the last event time {times[-1]}"
        )

    grid = tuple(int(g) for g in grid)
    if len(grid) != 3:
        raise ValueError("grid must be a 3-tuple (Gx, Gy, Gt)")

    if max_outer < 1:
        raise ValueError(f"max_outer must be >= 1; got {max_outer}")
    if tol <= 0:
        raise ValueError(f"tol must be positive; got {tol}")

    if hawkes_spec is None:
        hawkes_spec = SpatialHawkesSpec()

    # ----- Init: LGCP background fit treating every event as background. -----
    bg = lgcp_st_fit(
        points, times, domain=domain, period=period, grid=grid,
        length_scale_space=length_scale_space, length_scale_time=length_scale_time,
    )
    K_branch = float(hawkes_spec.K0)
    c = float(hawkes_spec.c0)
    sigma = float(hawkes_spec.sigma0)

    # Dense pairwise scaffolding, fixed across outer iterations (only the
    # background and (K, c, sigma) change) -- same precompute-once
    # discipline as spatial_hawkes.em_spatial_hawkes.
    mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
    dt_full = np.where(mask, times[:, None] - times[None, :], 0.0)
    diff_xy = points[:, None, :] - points[None, :, :]
    dr2_full = np.where(mask, np.sum(diff_xy * diff_xy, axis=-1), 0.0)

    p_diag, p_offdiag, lam_safe = _e_step(
        points, times, mask, dt_full, dr2_full, bg, K_branch, c, sigma
    )
    ll_trace: list[float] = [_log_likelihood(lam_safe, bg, K_branch, c, T, times)]

    converged = False
    n_outer = 0
    for outer_idx in range(max_outer):
        # ----- M-step: background, via the exact weighted-histogram LGCP
        # refit (see module docstring, "Weighted-histogram background
        # M-step") -- deterministic, zero Monte-Carlo variance. -----
        bg = _fit_weighted_background(
            points, times, p_diag, domain=domain, period=period, grid=grid,
            length_scale_space=length_scale_space, length_scale_time=length_scale_time,
        )

        # ----- M-step: excitation, closed-form weighted MLE (see module
        # docstring; formulas re-implemented from
        # spatial_hawkes.em_spatial_hawkes's derivation). -----
        sum_p_offdiag = float(p_offdiag.sum())
        sum_p_dt = float((p_offdiag * dt_full).sum())
        sum_p_dr2 = float((p_offdiag * dr2_full).sum())

        if sum_p_dt > 0.0:
            c_new = sum_p_offdiag / sum_p_dt
        else:
            c_new = c

        boundary_T = T - times
        K_denom = float(np.sum(1.0 - np.exp(-c_new * boundary_T)))
        if K_denom > _TINY:
            K_new = sum_p_offdiag / K_denom
        else:
            K_new = K_branch

        if sum_p_offdiag > 0.0:
            sigma2_new = sum_p_dr2 / (2.0 * sum_p_offdiag)
            sigma_new = float(np.sqrt(max(sigma2_new, _TINY)))
        else:
            sigma_new = sigma

        K_branch = max(K_new, 0.0)
        c = max(c_new, _TINY)
        sigma = max(sigma_new, _TINY)

        # ----- Refresh E-step + log-likelihood at the new parameters. -----
        p_diag, p_offdiag, lam_safe = _e_step(
            points, times, mask, dt_full, dr2_full, bg, K_branch, c, sigma
        )
        ll_curr = _log_likelihood(lam_safe, bg, K_branch, c, T, times)
        ll_trace.append(ll_curr)
        n_outer = outer_idx + 1

        denom = max(abs(ll_trace[-2]), _TINY)
        if abs(ll_curr - ll_trace[-2]) / denom < tol:
            converged = True
            break

    background_fraction = float(np.mean(p_diag))

    return CoxHawkesResult(
        background=bg,
        K_branch_hat=float(K_branch),
        c_hat=float(c),
        sigma_space_hat=float(sigma),
        background_fraction=background_fraction,
        log_likelihood_trace=np.asarray(ll_trace, dtype=np.float64),
        n_outer=n_outer,
        converged=converged,
        event_points=points.copy(),
        event_times=times.copy(),
    )


# ----------------------------------------------------------------------
# Simulator: LGCP-background immigrants + Hawkes offspring cascade
# ----------------------------------------------------------------------


def simulate_cox_hawkes(
    background_intensity_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    K_branch: float,
    c: float,
    sigma_space: float,
    *,
    domain: tuple[tuple[float, float], tuple[float, float]],
    T: float,
    rng: np.random.Generator,
    bg_max: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Simulate a Cox-Hawkes process: LGCP-background immigrants + a Hawkes cascade.

    Two-stage simulation:

    1. **Background immigrants** — an inhomogeneous Poisson process with
       intensity ``background_intensity_fn(x, t)`` on ``W x [0, T]``,
       drawn by dominating-rate thinning (Lewis & Shedler 1979): propose
       a homogeneous Poisson process at rate ``bg_max`` and keep each
       candidate ``(x, t)`` with probability
       ``background_intensity_fn(x, t) / bg_max``.
    2. **Offspring cascade** — every immigrant (and every one of its
       descendants) independently spawns ``Poisson(K_branch)`` direct
       offspring with an ``Exp(c)`` temporal delay and an isotropic
       ``N(0, sigma_space^2 I)`` spatial offset, exactly the
       branching/Poisson-cluster representation (Møller & Rasmussen
       2005) used by
       :func:`~nstat.extras.spatial.spatial_hawkes.simulate_spatial_hawkes`
       — reimplemented here rather than called, because that function's
       immigrant-generation step is hard-coded to a *homogeneous*
       background.  Offspring born at or after ``T`` are discarded and do
       not themselves spawn further offspring (right-censoring).

    Parameters
    ----------
    background_intensity_fn : callable ``(X, t) -> rate``
        ``X`` is ``(n, 2)``, ``t`` is ``(n,)``; returns ``(n,)``
        non-negative rates.  Typically
        ``LGCPSTResult.intensity_fn()`` from a prior
        :func:`~nstat.extras.spatial.lgcp_st.lgcp_st_fit` call, or any
        hand-written ground-truth background for a simulation study.
    K_branch : float
        Branching ratio, ``0 <= K_branch < 1`` (subcritical).
    c : float
        Temporal decay of the offspring kernel, ``c > 0``.
    sigma_space : float
        Spatial scale of the offspring kernel, ``sigma_space > 0``.
    domain : ((xlo, xhi), (ylo, yhi))
        Rectangular window immigrants are drawn from.  Offspring are
        **not** clipped to this window (matching
        ``simulate_spatial_hawkes``'s convention).
    T : float
        Simulation horizon, ``T > 0``.
    rng : numpy.random.Generator
        Random number generator (``np.random.default_rng(seed)``).
    bg_max : float or None
        Dominating rate for the immigrant thinning step.  If ``None``,
        estimated by probing ``background_intensity_fn`` at 5000 random
        ``(x, t)`` points over ``W x [0, T]`` and taking
        ``1.25 * max(probed values)`` as a safety-padded bound; this is
        a heuristic, not a guaranteed bound, so pass an explicit
        ``bg_max`` if the true peak of ``background_intensity_fn`` is
        known and might exceed the probed estimate.

    Returns
    -------
    points : (M, 2) float64 ndarray
        Event locations, sorted by ascending event time.
    times : (M,) float64 ndarray
        Sorted event times in ``[0, T)``.

    Raises
    ------
    ValueError
        If any parameter is out of range, ``K_branch >= 1``
        (super-critical), ``domain`` is malformed, or a probed/proposed
        background value exceeds ``bg_max``.
    """
    if not (K_branch >= 0):
        raise ValueError(f"K_branch must be non-negative; got {K_branch}")
    if K_branch >= 1.0:
        raise ValueError(
            f"super-critical process: K_branch = {K_branch} >= 1; "
            "expected event count is infinite"
        )
    if not (c > 0):
        raise ValueError(f"c must be positive; got {c}")
    if not (sigma_space > 0):
        raise ValueError(f"sigma_space must be positive; got {sigma_space}")
    if not (T > 0):
        raise ValueError(f"T must be positive; got {T}")

    domain = _validate_domain(domain)
    (xlo, xhi), (ylo, yhi) = domain
    area = (xhi - xlo) * (yhi - ylo)

    if bg_max is None:
        n_probe = 5000
        probe_x = rng.uniform(xlo, xhi, n_probe)
        probe_y = rng.uniform(ylo, yhi, n_probe)
        probe_t = rng.uniform(0.0, T, n_probe)
        probe_pts = np.column_stack([probe_x, probe_y])
        probe_vals = np.asarray(
            background_intensity_fn(probe_pts, probe_t), dtype=np.float64
        )
        bg_max = float(np.max(probe_vals)) * 1.25
        if not (bg_max > 0):
            raise ValueError(
                "background_intensity_fn was non-positive at every probed "
                "point; cannot estimate a dominating rate for simulation"
            )
    else:
        bg_max = float(bg_max)
        if not (bg_max > 0):
            raise ValueError(f"bg_max must be positive; got {bg_max}")

    # Immigrants: Lewis-Shedler thinning of a homogeneous rate-bg_max
    # proposal against the true inhomogeneous background.
    n_prop = int(rng.poisson(bg_max * area * T))
    prop_x = rng.uniform(xlo, xhi, n_prop)
    prop_y = rng.uniform(ylo, yhi, n_prop)
    prop_t = rng.uniform(0.0, T, n_prop)
    prop_pts = np.column_stack([prop_x, prop_y])
    prop_vals = np.asarray(
        background_intensity_fn(prop_pts, prop_t), dtype=np.float64
    )
    if prop_vals.size and np.any(prop_vals > bg_max):
        raise ValueError(
            f"background_intensity_fn exceeded bg_max={bg_max}; supply a "
            "larger bg_max"
        )
    accept = rng.uniform(0.0, 1.0, n_prop) < prop_vals / bg_max
    frontier_x = prop_x[accept]
    frontier_y = prop_y[accept]
    frontier_t = prop_t[accept]

    all_t: list[NDArray[np.float64]] = [frontier_t]
    all_x: list[NDArray[np.float64]] = [frontier_x]
    all_y: list[NDArray[np.float64]] = [frontier_y]

    # Offspring cascade -- same branching loop as
    # simulate_spatial_hawkes, started from the inhomogeneous immigrants
    # above instead of a homogeneous-Poisson frontier.
    while frontier_t.size > 0:
        n_off = rng.poisson(K_branch, size=frontier_t.size)
        total_off = int(n_off.sum())
        if total_off == 0:
            break

        parent_idx = np.repeat(np.arange(frontier_t.size), n_off)
        delays = rng.exponential(1.0 / c, size=total_off)
        child_t = frontier_t[parent_idx] + delays

        keep = child_t < T
        if not np.any(keep):
            break
        parent_idx = parent_idx[keep]
        child_t = child_t[keep]

        dx = rng.normal(0.0, sigma_space, size=child_t.size)
        dy = rng.normal(0.0, sigma_space, size=child_t.size)
        child_x = frontier_x[parent_idx] + dx
        child_y = frontier_y[parent_idx] + dy

        all_t.append(child_t)
        all_x.append(child_x)
        all_y.append(child_y)

        frontier_t, frontier_x, frontier_y = child_t, child_x, child_y

    times_out = np.concatenate(all_t) if all_t else np.empty(0, dtype=np.float64)
    xs_out = np.concatenate(all_x) if all_x else np.empty(0, dtype=np.float64)
    ys_out = np.concatenate(all_y) if all_y else np.empty(0, dtype=np.float64)

    order = np.argsort(times_out, kind="stable")
    times_sorted = times_out[order].astype(np.float64, copy=False)
    points_sorted = np.stack([xs_out[order], ys_out[order]], axis=1).astype(
        np.float64, copy=False
    )
    return points_sorted, times_sorted
