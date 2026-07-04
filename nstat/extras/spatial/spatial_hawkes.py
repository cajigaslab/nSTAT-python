r"""Space-time ETAS-style self-exciting Hawkes process via branching EM.

Pure-NumPy extension of :mod:`nstat.extras.spatial.hawkes_em` (Veen &
Schoenberg 2008) that adds an isotropic Gaussian spatial triggering
kernel to the exponential temporal kernel, giving a full space-time
epidemic-type aftershock sequence (ETAS) conditional intensity with a
*homogeneous* background:

.. math::

    \lambda(x, t) = \frac{\mu}{|W|} +
        \sum_{t_j < t} K \, g(t - t_j) \, h(x - x_j)

with

.. math::

    g(t) = c \, e^{-c t}, \qquad
    h(r) = \frac{1}{2 \pi \sigma^2} \exp\!\left(-\frac{\lVert r
    \rVert^2}{2 \sigma^2}\right)

``g`` and ``h`` are each normalised to integrate to 1 over their full
domain (``[0, \infty)`` and :math:`\mathbb{R}^2` respectively), so ``K``
*is* the branching ratio (expected number of direct offspring per
event) with no separate amplitude/decay entanglement in the spatial
dimension. The background rate is homogeneous over the rectangular
window ``W`` — an inhomogeneous (log-Gaussian Cox process) background
is a separate concern and belongs in
:mod:`nstat.extras.spatial.lgcp`, not here.

This module has **no MATLAB counterpart** (space-time ETAS is outside
the scope of the original nSTAT toolbox) and therefore lives in the
opt-in ``nstat.extras`` namespace with no ``parity/manifest.yml`` entry,
per the MATLAB-independence rule in ``CLAUDE.md``. All mathematics below
was derived directly from the cited papers; no third-party ETAS
implementation was read, ported, or consulted.

Branching-EM structure
-----------------------

Mirrors :func:`nstat.extras.spatial.hawkes_em.em_hawkes_exponential`:
soft parent responsibilities in the E-step, closed-form weighted MLE in
the M-step, a dense ``O(N^2)`` implementation (materialising the full
time-difference *and* squared-distance matrices), and the same
right-boundary compensator-correction discipline for the temporal decay
``c`` / branching ratio ``K``. The spatial kernel ``h`` is normalised
over all of :math:`\mathbb{R}^2` (not clipped to ``W``), so — by
construction — it needs **no** spatial edge correction; only the
temporal compensator has a boundary term (offspring born before ``T``
but whose kernel mass extends past it).

References
----------
Veen A & Schoenberg FP (2008). *Estimation of space-time branching
process models in seismology using an EM-type algorithm.* Journal of
the American Statistical Association 103(482):614-624.

Zhuang J, Ogata Y & Vere-Jones D (2002). *Stochastic declustering of
space-time earthquake occurrences.* Journal of the American Statistical
Association 97(458):369-380.

Ogata Y (1998). *Space-time point-process models for earthquake
occurrences.* Annals of the Institute of Statistical Mathematics
50(2):379-402.

Møller J & Rasmussen JG (2005). *Perfect simulation of Hawkes
processes.* Advances in Applied Probability 37(3):629-646. (Branching /
Poisson-cluster representation used by :func:`simulate_spatial_hawkes`.)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


__all__ = [
    "SpatialHawkesSpec",
    "SpatialHawkesResult",
    "em_spatial_hawkes",
    "simulate_spatial_hawkes",
]


# ----------------------------------------------------------------------
# Configuration + result dataclasses
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialHawkesSpec:
    """Initial guesses + convergence config for space-time Hawkes EM.

    Parameters
    ----------
    mu0 : float or None
        Initial baseline rate (total background count rate over the
        whole window and horizon; NOT a density). If None, auto-infer
        as ``N / T`` at fit time.
    K0 : float
        Initial branching ratio (expected direct offspring per event).
        Default 0.5. Must satisfy ``K0 < 1`` for the process to be
        sub-critical (a value >= 1 warns but is still accepted as a
        starting point).
    c0 : float
        Initial temporal decay of the exponential offspring kernel
        ``g(t) = c exp(-c t)``. Default 1.0.
    sigma0 : float
        Initial spatial scale of the isotropic Gaussian offspring
        kernel. Default 0.1.
    max_iter : int
        Maximum EM iterations. Default 200.
    tol : float
        Relative log-likelihood tolerance for convergence. Default 1e-6.
    """

    mu0: float | None = None
    K0: float = 0.5
    c0: float = 1.0
    sigma0: float = 0.1
    max_iter: int = 200
    tol: float = 1e-6

    def __post_init__(self) -> None:
        if self.mu0 is not None and not (self.mu0 > 0):
            raise ValueError(f"mu0 must be positive or None; got {self.mu0}")
        if not (self.K0 > 0):
            raise ValueError(f"K0 must be positive; got {self.K0}")
        if not (self.c0 > 0):
            raise ValueError(f"c0 must be positive; got {self.c0}")
        if not (self.sigma0 > 0):
            raise ValueError(f"sigma0 must be positive; got {self.sigma0}")
        if not (self.max_iter >= 1):
            raise ValueError(f"max_iter must be >= 1; got {self.max_iter}")
        if not (self.tol > 0):
            raise ValueError(f"tol must be positive; got {self.tol}")
        if self.K0 >= 1.0:
            warnings.warn(
                "initialisation is super-critical (K0 >= 1); EM may not "
                "converge to a sub-critical process",
                UserWarning,
                stacklevel=2,
            )


@dataclass(frozen=True)
class SpatialHawkesResult:
    """Fitted space-time Hawkes parameters from branching EM.

    Attributes
    ----------
    mu_hat : float
        Fitted total background rate (events per unit time, summed over
        the whole window ``W``; the background *density* is
        ``mu_hat / |W|``).
    K_branch_hat : float
        Fitted branching ratio (expected direct offspring per event).
        Because both the temporal kernel ``g`` and spatial kernel ``h``
        are individually normalised to integrate to 1, ``K_branch_hat``
        *is* the branching ratio directly — no further division needed
        (contrast with :attr:`nstat.extras.spatial.hawkes_em.HawkesEMResult.branching_ratio`,
        which divides an unnormalised amplitude by a decay).
    c_hat : float
        Fitted temporal decay of the exponential offspring kernel.
    sigma_space_hat : float
        Fitted spatial scale (standard deviation) of the isotropic
        Gaussian offspring kernel.
    log_likelihood_trace : (n_iter+1,) ndarray
        Per-iteration log-likelihood, including the initial value at
        index 0.
    n_iter : int
        Number of EM iterations performed (0 if converged immediately or
        on the trivial single-event path).
    converged : bool
        True iff convergence criterion was met before max_iter.
    responsibilities : scipy.sparse.csr_matrix or None
        Optional ``(N, N)`` lower-triangular sparse matrix of posterior
        parent assignments, laid out exactly as in
        :class:`nstat.extras.spatial.hawkes_em.HawkesEMResult`: row
        ``i``'s diagonal cell is the probability event ``i`` is a
        background event; cell ``[i, j]`` (``j < i``) is the probability
        that event ``j`` triggered event ``i``. Each row sums to 1.0.
        ``None`` unless ``return_responsibilities=True`` was passed.
    """

    mu_hat: float
    K_branch_hat: float
    c_hat: float
    sigma_space_hat: float
    log_likelihood_trace: np.ndarray
    n_iter: int
    converged: bool
    # ``object | None`` (not ``scipy.sparse.csr_matrix | None``) so that
    # importing this module does not pull scipy.sparse at import time.
    responsibilities: object | None


# ----------------------------------------------------------------------
# Domain helper
# ----------------------------------------------------------------------


def _validate_domain(
    domain: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    """Validate a rectangular ``((xlo, xhi), (ylo, yhi))`` domain; return its area."""
    # Accept either a tuple or a list of two (lo, hi) pairs — the sibling
    # spatial modules (st_intensity, spatiotemporal_gof) accept lists via
    # flexible unpacking, so a value valid for one must be valid for all.
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
    if not (xhi > xlo):
        raise ValueError(
            f"domain x-range must satisfy xhi > xlo; got ({xlo}, {xhi})"
        )
    if not (yhi > ylo):
        raise ValueError(
            f"domain y-range must satisfy yhi > ylo; got ({ylo}, {yhi})"
        )
    return float((xhi - xlo) * (yhi - ylo))


# ----------------------------------------------------------------------
# EM algorithm
# ----------------------------------------------------------------------


def _spatial_hawkes_log_likelihood(
    times: NDArray[np.float64],
    T: float,
    mu: float,
    K_branch: float,
    c: float,
    sigma: float,
    area: float,
    dt: NDArray[np.float64],  # (N, N) lower-triangular t_i - t_j (j < i) else 0
    dr2: NDArray[np.float64],  # (N, N) lower-triangular ||x_i - x_j||^2 (j < i) else 0
    mask: NDArray[np.bool_],  # (N, N) True where j < i
) -> float:
    r"""Space-time Hawkes log-likelihood for exponential x isotropic-Gaussian kernel.

    .. math::

        LL = \sum_i \log \lambda(x_i, t_i) - \mu T -
             K \sum_j \left(1 - e^{-c (T - t_j)}\right)

    where ``lambda(x_i, t_i) = mu / area + K * sum_{j<i} c * exp(-c *
    (t_i - t_j)) * h(x_i - x_j)``. The spatial kernel integrates to 1
    over all of :math:`\mathbb{R}^2` by construction, so it contributes
    no boundary term to the compensator — only the temporal kernel's
    right-censoring at ``T`` does.
    """
    g = np.where(mask, c * np.exp(-c * dt), 0.0)
    h = np.where(
        mask,
        np.exp(-dr2 / (2.0 * sigma * sigma)) / (2.0 * np.pi * sigma * sigma),
        0.0,
    )
    triggering = K_branch * g * h
    lam = mu / area + triggering.sum(axis=1)
    lam_safe = np.maximum(lam, np.finfo(np.float64).tiny)
    ll_events = np.sum(np.log(lam_safe))
    ll_compensator = mu * T + K_branch * np.sum(1.0 - np.exp(-c * (T - times)))
    return float(ll_events - ll_compensator)


def em_spatial_hawkes(
    points: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    domain: tuple[tuple[float, float], tuple[float, float]],
    T: float,
    spec: SpatialHawkesSpec | None = None,
    return_responsibilities: bool = False,
) -> SpatialHawkesResult:
    r"""Branching EM for a space-time Hawkes / ETAS-style process.

    Extends :func:`nstat.extras.spatial.hawkes_em.em_hawkes_exponential`
    with an isotropic Gaussian spatial offspring kernel. Each event is
    either a background event (probability proportional to ``mu /
    |W|``) or triggered by an earlier event ``j`` (probability
    proportional to ``K * g(t_i - t_j) * h(x_i - x_j)``). EM alternates
    between soft parent responsibilities (E-step) and closed-form
    weighted-MLE re-estimation of ``(mu, K, c, sigma)`` (M-step).

    Parameters
    ----------
    points : (N, 2) float64 ndarray
        Event locations, row-aligned with ``times``.
    times : (N,) float64 ndarray
        Sorted event times in ``[0, T]``.
    domain : ((xlo, xhi), (ylo, yhi))
        Rectangular background window ``W`` (same convention as
        :mod:`nstat.extras.spatial.spatial_gof`). Only its area enters
        the fit (via the homogeneous background density ``mu / |W|``);
        the offspring kernel is not clipped to it.
    T : float
        Observation horizon. Must satisfy ``T > times[-1]``.
    spec : SpatialHawkesSpec or None
        Configuration. Defaults to ``SpatialHawkesSpec()``.
    return_responsibilities : bool
        If True, the returned result includes a CSR sparse
        responsibility matrix (lazy-imports ``scipy.sparse``).

    Returns
    -------
    SpatialHawkesResult

    Notes
    -----
    The implementation materialises the full ``(N, N)`` time-difference
    and squared-distance matrices, so memory cost is ``O(N^2)`` — same
    caveat as :func:`~nstat.extras.spatial.hawkes_em.em_hawkes_exponential`
    (acceptable dense NumPy up to ``N`` ~ a few thousand).

    M-step closed forms (derived by setting the gradient of the
    expected complete-data log-likelihood ``Q`` to zero):

    - ``mu = sum(p_ii) / T`` — identical in form to the purely-temporal
      case; ``|W|`` cancels because the background compensator is
      ``mu/|W| * |W| * T = mu * T``.
    - ``sigma^2 = sum(p_ij * ||x_i - x_j||^2) / (2 * sum(p_ij))`` — the
      2-D isotropic-Gaussian weighted MLE (the factor of 2 comes from
      the 2 spatial degrees of freedom).
    - ``c`` uses the same simplified closed form as
      :func:`~nstat.extras.spatial.hawkes_em.em_hawkes_exponential`'s
      ``beta`` update, ignoring the compensator's dependence on ``c``
      (exact only in the ``T -> infinity`` limit; the standard
      Veen-Schoenberg practical compromise).
    - ``K`` (the branching ratio) is then the **exact** conditional
      maximiser of ``Q`` at ``c = c_new``:
      ``K = sum(p_ij) / sum_j(1 - exp(-c_new * (T - t_j)))``. Unlike
      :func:`~nstat.extras.spatial.hawkes_em.em_hawkes_exponential`'s
      ``alpha`` update, there is **no** extra multiplicative factor of
      ``c_new`` here: because our temporal kernel ``g`` is already
      normalised to integrate to 1 (``g(t) = c e^{-ct}``), ``K`` is
      *itself* the branching ratio, whereas hawkes_em's ``alpha`` is an
      unnormalised amplitude with ``alpha/beta`` as the branching ratio.
      Substituting ``alpha := K*c, beta := c`` into hawkes_em's exact
      ``alpha`` update and solving for ``K`` shows the ``c_new`` factors
      cancel exactly, confirming this simpler form.

    As in hawkes_em, a single-realisation fit only weakly identifies
    the individual amplitude/decay-style parameters; if you need a
    single well-determined summary, ``K_branch_hat`` here is already
    the branching ratio (no further ratio needed), but the *joint*
    ``(K_branch_hat, c_hat)`` pair is still less precisely determined
    than either alone would be under repeated-realisation averaging.

    References
    ----------
    Veen A & Schoenberg FP (2008). *Estimation of space-time branching
    process models in seismology using an EM-type algorithm.* JASA
    103(482):614-624.

    Zhuang J, Ogata Y & Vere-Jones D (2002). *Stochastic declustering of
    space-time earthquake occurrences.* JASA 97(458):369-380.
    """
    points = np.asarray(points, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    if times.ndim != 1:
        raise ValueError(f"times must be 1-D; got shape {times.shape}")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2); got {points.shape}")
    if points.shape[0] != times.shape[0]:
        raise ValueError(
            f"points has {points.shape[0]} rows but times has "
            f"{times.shape[0]} entries"
        )

    if spec is None:
        spec = SpatialHawkesSpec()

    n = times.size

    if n == 0:
        raise ValueError(
            "times is empty; cannot identify spatial-Hawkes parameters"
        )

    if n >= 2 and np.any(np.diff(times) < 0):
        raise ValueError("times must be sorted ascending")

    if not (T > times[-1]):
        raise ValueError(f"T={T} must exceed last event time {times[-1]}")

    area = _validate_domain(domain)

    # Trivial single-event path: mirrors hawkes_em's degenerate case —
    # observationally indistinguishable from a homogeneous Poisson
    # process over W x [0, T] at this sample size.
    if n == 1:
        mu_hat = 1.0 / T
        ll0 = float(np.log(mu_hat / area) - mu_hat * T)
        resp: object | None = None
        if return_responsibilities:
            from scipy.sparse import csr_matrix

            resp = csr_matrix(np.array([[1.0]], dtype=np.float64))
        return SpatialHawkesResult(
            mu_hat=mu_hat,
            K_branch_hat=0.0,
            c_hat=spec.c0,
            sigma_space_hat=spec.sigma0,
            log_likelihood_trace=np.array([ll0], dtype=np.float64),
            n_iter=0,
            converged=True,
            responsibilities=resp,
        )

    # Pre-compute the lower-triangular time-difference and squared
    # spatial-distance matrices once. `mask` is the boolean stencil;
    # `dt`/`dr2` hold values only where the mask is True.
    diffs_t = times[:, None] - times[None, :]
    mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
    dt = np.where(mask, diffs_t, 0.0)

    diff_xy = points[:, None, :] - points[None, :, :]  # (N, N, 2)
    dr2_full = np.sum(diff_xy * diff_xy, axis=-1)  # (N, N)
    dr2 = np.where(mask, dr2_full, 0.0)

    # Initialisation.
    mu_current = float(spec.mu0) if spec.mu0 is not None else float(n) / float(T)
    K_current = float(spec.K0)
    c_current = float(spec.c0)
    sigma_current = float(spec.sigma0)

    ll_history: list[float] = []
    ll_prev = _spatial_hawkes_log_likelihood(
        times, T, mu_current, K_current, c_current, sigma_current, area, dt, dr2, mask
    )
    ll_history.append(ll_prev)

    converged = False
    n_iter = 0
    tiny = np.finfo(np.float64).tiny

    for iteration_idx in range(spec.max_iter):
        iteration = iteration_idx + 1

        # ----- E-step: soft parent assignments. -----
        g = np.where(mask, c_current * np.exp(-c_current * dt), 0.0)
        h = np.where(
            mask,
            np.exp(-dr2 / (2.0 * sigma_current * sigma_current))
            / (2.0 * np.pi * sigma_current * sigma_current),
            0.0,
        )
        triggering = K_current * g * h  # (N, N), zero except j < i
        lam_per_event = mu_current / area + triggering.sum(axis=1)  # (N,)
        lam_safe = np.maximum(lam_per_event, tiny)

        p_diag = (mu_current / area) / lam_safe  # (N,)
        p_offdiag = triggering / lam_safe[:, None]  # (N, N)

        # ----- M-step: closed-form weighted MLE. -----
        sum_p_diag = float(p_diag.sum())
        sum_p_offdiag = float(p_offdiag.sum())
        sum_p_dt = float((p_offdiag * dt).sum())
        sum_p_dr2 = float((p_offdiag * dr2).sum())

        mu_new = sum_p_diag / T

        # c update: same simplified closed form as hawkes_em's beta
        # (ignores the compensator's dependence on c).
        if sum_p_dt > 0.0:
            c_new = sum_p_offdiag / sum_p_dt
        else:
            c_new = c_current

        # K update: exact maximiser of Q(theta) w.r.t. K at c = c_new.
        # See the "Notes" docstring above for why there is no extra
        # c_new multiplier (unlike hawkes_em's alpha update) — g is
        # already normalised so K is directly the branching ratio.
        boundary_T = T - times  # (N,)
        K_c = float(np.sum(1.0 - np.exp(-c_new * boundary_T)))
        if K_c > tiny:
            K_new = sum_p_offdiag / K_c
        else:
            K_new = K_current

        # sigma^2 update: 2-D isotropic-Gaussian weighted MLE.
        if sum_p_offdiag > 0.0:
            sigma2_new = sum_p_dr2 / (2.0 * sum_p_offdiag)
            sigma_new = float(np.sqrt(max(sigma2_new, tiny)))
        else:
            sigma_new = sigma_current

        # Numerical guards: keep iterates strictly positive.
        mu_new = max(mu_new, tiny)
        K_new = max(K_new, 0.0)
        c_new = max(c_new, tiny)
        sigma_new = max(sigma_new, tiny)

        mu_current, K_current, c_current, sigma_current = (
            mu_new,
            K_new,
            c_new,
            sigma_new,
        )

        ll_curr = _spatial_hawkes_log_likelihood(
            times, T, mu_current, K_current, c_current, sigma_current, area, dt, dr2, mask
        )
        ll_history.append(ll_curr)
        n_iter = iteration

        denom = max(abs(ll_prev), tiny)
        if abs(ll_curr - ll_prev) / denom < spec.tol:
            converged = True
            ll_prev = ll_curr
            break
        ll_prev = ll_curr

    responsibilities_out: object | None = None
    if return_responsibilities:
        # Recompute one last E-step at the final parameters so the
        # returned matrix matches (mu_hat, K_branch_hat, c_hat,
        # sigma_space_hat) — the M-step inside the loop overwrites the
        # previous responsibilities before convergence is tested.
        g = np.where(mask, c_current * np.exp(-c_current * dt), 0.0)
        h = np.where(
            mask,
            np.exp(-dr2 / (2.0 * sigma_current * sigma_current))
            / (2.0 * np.pi * sigma_current * sigma_current),
            0.0,
        )
        triggering = K_current * g * h
        lam_per_event = mu_current / area + triggering.sum(axis=1)
        lam_safe = np.maximum(lam_per_event, tiny)
        p_diag_final = (mu_current / area) / lam_safe
        p_offdiag_final = triggering / lam_safe[:, None]

        dense = np.zeros((n, n), dtype=np.float64)
        diag_idx = np.arange(n)
        dense[diag_idx, diag_idx] = p_diag_final
        dense += np.where(mask, p_offdiag_final, 0.0)

        from scipy.sparse import csr_matrix

        responsibilities_out = csr_matrix(dense)

    return SpatialHawkesResult(
        mu_hat=float(mu_current),
        K_branch_hat=float(K_current),
        c_hat=float(c_current),
        sigma_space_hat=float(sigma_current),
        log_likelihood_trace=np.asarray(ll_history, dtype=np.float64),
        n_iter=n_iter,
        converged=converged,
        responsibilities=responsibilities_out,
    )


# ----------------------------------------------------------------------
# Branching (Poisson-cluster) simulator
# ----------------------------------------------------------------------


def simulate_spatial_hawkes(
    mu: float,
    K_branch: float,
    c: float,
    sigma_space: float,
    *,
    domain: tuple[tuple[float, float], tuple[float, float]],
    T: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Simulate a space-time Hawkes/ETAS process via branching.

    Uses the Poisson-cluster / branching representation of a
    self-exciting process (Møller & Rasmussen 2005): generation-0
    ("background") events form a homogeneous Poisson process on
    ``W x [0, T]`` with rate ``mu / |W|``; every event (of any
    generation) independently spawns ``Poisson(K_branch)`` direct
    offspring, each with a temporal delay drawn from ``Exp(c)`` (the
    density ``g(t) = c e^{-ct}``) and a spatial offset drawn from
    ``N(0, sigma_space^2 I)`` (the density ``h``). Offspring with a
    birth time ``>= T`` are discarded (right-censored) and do not
    themselves spawn further offspring, since events outside ``[0, T]``
    are unobserved. The recursion terminates almost surely for
    ``K_branch < 1`` (subcritical Galton-Watson branching).

    Parameters
    ----------
    mu : float
        Total background rate (events per unit time over the whole
        window ``W``), ``mu > 0``.
    K_branch : float
        Branching ratio (expected direct offspring per event),
        ``0 <= K_branch < 1`` (subcritical).
    c : float
        Temporal decay of the offspring kernel, ``c > 0``.
    sigma_space : float
        Spatial scale of the offspring kernel, ``sigma_space > 0``.
    domain : ((xlo, xhi), (ylo, yhi))
        Rectangular background window ``W``.
    T : float
        Simulation horizon, ``T > 0``.
    rng : numpy.random.Generator
        Random number generator (``np.random.default_rng(seed)``).

    Returns
    -------
    points : (M, 2) float64 ndarray
        Event locations, sorted by ascending event time (row-aligned
        with ``times``).
    times : (M,) float64 ndarray
        Sorted event times in ``[0, T)``.

    Raises
    ------
    ValueError
        If any parameter is out of range, or if ``K_branch >= 1``
        (super-critical: expected total event count is infinite).
    """
    if not (mu > 0):
        raise ValueError(f"mu must be positive; got {mu}")
    if not (K_branch >= 0):
        raise ValueError(f"K_branch must be non-negative; got {K_branch}")
    if not (c > 0):
        raise ValueError(f"c must be positive; got {c}")
    if not (sigma_space > 0):
        raise ValueError(f"sigma_space must be positive; got {sigma_space}")
    if not (T > 0):
        raise ValueError(f"T must be positive; got {T}")
    if K_branch >= 1.0:
        raise ValueError(
            f"super-critical process: K_branch = {K_branch} >= 1; "
            "expected event count is infinite"
        )

    _validate_domain(domain)
    (xlo, xhi), (ylo, yhi) = domain

    # Generation 0: homogeneous background events on W x [0, T].
    n_bg = int(rng.poisson(mu * T))
    frontier_t = rng.uniform(0.0, T, size=n_bg)
    frontier_x = rng.uniform(xlo, xhi, size=n_bg)
    frontier_y = rng.uniform(ylo, yhi, size=n_bg)

    all_t: list[NDArray[np.float64]] = [frontier_t]
    all_x: list[NDArray[np.float64]] = [frontier_x]
    all_y: list[NDArray[np.float64]] = [frontier_y]

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
