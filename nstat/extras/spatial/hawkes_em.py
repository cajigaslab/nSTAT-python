r"""Hawkes EM declustering via branching-structure estimation.

Pure-NumPy implementation of the Veen-Schoenberg (2008) EM algorithm for
self-exciting temporal point processes with exponential triggering kernels.
Recovers baseline rate mu, branching intensity alpha, and decay beta from
spike times via iterative parent-assignment declustering. No optional
dependencies.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


__all__ = [
    "HawkesEMSpec",
    "HawkesEMResult",
    "em_hawkes_exponential",
    "simulate_hawkes_exponential",
]


# ----------------------------------------------------------------------
# Configuration + result dataclasses
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class HawkesEMSpec:
    """Initial guesses + convergence config for Hawkes EM.

    Parameters
    ----------
    mu0 : float or None
        Initial baseline rate. If None, auto-infer as N/T at fit time.
    alpha0 : float
        Initial branching intensity. Default 0.5. Must satisfy alpha0 / beta0 < 1
        for the process to be sub-critical.
    beta0 : float
        Initial decay parameter. Default 1.0.
    max_iter : int
        Maximum EM iterations. Default 100.
    tol : float
        Relative log-likelihood tolerance for convergence. Default 1e-6.
    """

    mu0: float | None = None
    alpha0: float = 0.5
    beta0: float = 1.0
    max_iter: int = 100
    tol: float = 1e-6

    def __post_init__(self) -> None:
        if not (self.alpha0 > 0):
            raise ValueError(f"alpha0 must be positive; got {self.alpha0}")
        if not (self.beta0 > 0):
            raise ValueError(f"beta0 must be positive; got {self.beta0}")
        if self.mu0 is not None and not (self.mu0 > 0):
            raise ValueError(
                f"mu0 must be positive or None; got {self.mu0}"
            )
        if not (self.max_iter >= 1):
            raise ValueError(f"max_iter must be >= 1; got {self.max_iter}")
        if not (self.tol > 0):
            raise ValueError(f"tol must be positive; got {self.tol}")
        if self.alpha0 / self.beta0 >= 1.0:
            warnings.warn(
                "initialisation is super-critical; EM may not converge to a "
                "sub-critical process",
                UserWarning,
                stacklevel=2,
            )


@dataclass(frozen=True)
class HawkesEMResult:
    """Fitted Hawkes(mu, alpha exp(-beta t)) parameters from EM.

    Attributes
    ----------
    mu_hat : float
        Fitted baseline rate.
    alpha_hat : float
        Fitted branching intensity.
    beta_hat : float
        Fitted exponential decay.
    log_likelihood_trace : (n_iter+1,) ndarray
        Per-iteration log-likelihood, including the initial value at index 0.
    n_iter : int
        Number of EM iterations performed (0 if converged immediately or
        on the trivial single-event path).
    converged : bool
        True iff convergence criterion was met before max_iter.
    responsibilities : scipy.sparse.csr_matrix or None
        Optional ``(N, N)`` lower-triangular sparse matrix of posterior
        parent assignments.  Row ``i`` carries event ``i``'s parent
        distribution: cell ``[i, i]`` is the probability event ``i`` is
        spontaneous, and cell ``[i, j]`` (``j < i``) is the probability
        that event ``j`` triggered event ``i``.  Each row sums to 1.0.
        ``None`` unless ``return_responsibilities=True`` was passed.
    """

    mu_hat: float
    alpha_hat: float
    beta_hat: float
    log_likelihood_trace: np.ndarray
    n_iter: int
    converged: bool
    # The annotation is intentionally ``object | None`` rather than
    # ``scipy.sparse.csr_matrix | None`` so importing this module does
    # not pull scipy.sparse at import time.
    responsibilities: object | None

    @property
    def branching_ratio(self) -> float:
        return self.alpha_hat / self.beta_hat


# ----------------------------------------------------------------------
# EM algorithm
# ----------------------------------------------------------------------


def _hawkes_log_likelihood(
    event_times: NDArray[np.float64],
    T: float,
    mu: float,
    alpha: float,
    beta: float,
    dt: NDArray[np.float64],  # (N, N) lower-triangular t_i - t_j (j < i) else 0
    mask: NDArray[np.bool_],  # (N, N) True where j < i
) -> float:
    r"""Hawkes log-likelihood for exponential triggering.

    .. math::

        LL = \sum_i \log \lambda(t_i) - \mu T -
             \frac{\alpha}{\beta} \sum_i (1 - e^{-\beta (T - t_i)})

    where ``lambda(t_i) = mu + alpha * sum_{j<i} exp(-beta * (t_i - t_j))``.
    """
    # Triggering contributions exp(-beta * (t_i - t_j)) for j < i, zeroed elsewhere.
    kernel = np.where(mask, np.exp(-beta * dt), 0.0)
    triggering_sum = alpha * kernel.sum(axis=1)
    lam = mu + triggering_sum
    # Guard against floating-point underflow before taking the log.
    lam_safe = np.maximum(lam, np.finfo(np.float64).tiny)
    ll_events = np.sum(np.log(lam_safe))
    # Integrated intensity: mu*T plus aggregated exponential-tail decay
    # of every event's triggering kernel from its arrival time to T.
    ll_compensator = mu * T + (alpha / beta) * np.sum(
        1.0 - np.exp(-beta * (T - event_times))
    )
    return float(ll_events - ll_compensator)


def em_hawkes_exponential(
    event_times: NDArray[np.float64],
    T: float,
    *,
    spec: HawkesEMSpec | None = None,
    return_responsibilities: bool = False,
) -> HawkesEMResult:
    r"""Veen-Schoenberg 2008 EM for Hawkes(mu, alpha exp(-beta t)).

    The branching-structure interpretation: each event is either spontaneous
    (probability proportional to mu) or triggered by an earlier event
    (probability proportional to alpha * exp(-beta * (t_i - t_j))). EM
    alternates between assigning soft responsibilities (E-step) and
    re-estimating (mu, alpha, beta) by closed-form weighted MLE (M-step).

    Parameters
    ----------
    event_times : (N,) float64 ndarray
        Sorted event times in ``[0, T]``.
    T : float
        Observation horizon. Must satisfy ``T > event_times[-1]``.
    spec : HawkesEMSpec or None
        Configuration. Defaults to ``HawkesEMSpec()``.
    return_responsibilities : bool
        If True, the returned result includes a CSR sparse responsibility
        matrix (lazy-imports ``scipy.sparse``).

    Returns
    -------
    HawkesEMResult

    Notes
    -----
    The implementation materialises the full ``(N, N)`` time-difference
    matrix, so memory cost is ``O(N^2)``. For ``N`` up to ~5000 this is
    acceptable in dense NumPy; beyond that prefer a multivariate MLE via
    :func:`nstat.extras.spatial.hawkes_bridge.fit_hawkes_exp` (the
    ``tick``-backed bridge), which uses a stochastic-gradient inner loop.

    The M-step uses the Veen-Schoenberg (2008) closed form for
    ``mu`` and ``beta`` and the exact maximiser of ``Q(alpha; beta_new)``
    for ``alpha`` — the latter includes the right-boundary correction
    ``alpha = beta * sum p_ij / sum_j (1 - exp(-beta * (T - t_j)))``.
    The simpler ``alpha = sum p_ij / N`` form sometimes cited drops the
    boundary correction and is not a true M-step (``Q`` can decrease at
    finite ``T``).  Even so this single-realisation EM is only
    weakly identified between ``alpha`` and ``beta``; if you only need
    the branching ratio ``alpha / beta`` use
    :attr:`HawkesEMResult.branching_ratio` — it is far better
    determined than the individual amplitude and decay.

    References
    ----------
    Veen A & Schoenberg FP (2008). *Estimation of space-time branching
    process models in seismology using an EM-type algorithm.* JASA
    103(482):614-624.

    Marsan D & Lengliné O (2008). *Extending earthquakes' reach through
    cascading.* Science 319(5866):1076-1079.
    """
    event_times = np.asarray(event_times, dtype=np.float64)
    if event_times.ndim != 1:
        raise ValueError(
            f"event_times must be 1-D; got shape {event_times.shape}"
        )

    if spec is None:
        spec = HawkesEMSpec()

    n = event_times.size

    if n == 0:
        raise ValueError(
            "event_times is empty; cannot identify Hawkes parameters"
        )

    if n >= 2 and np.any(np.diff(event_times) < 0):
        raise ValueError("event_times must be sorted ascending")

    if not (T > event_times[-1]):
        raise ValueError(
            f"T={T} must exceed last event time {event_times[-1]}"
        )

    # Trivial single-event path: a Hawkes process with one event is
    # observationally indistinguishable from Poisson(1/T) at this sample
    # size — return a degraded result with zero branching.
    if n == 1:
        mu_hat = 1.0 / T
        ll0 = float(np.log(mu_hat) - 1.0)
        resp: object | None = None
        if return_responsibilities:
            from scipy.sparse import csr_matrix

            resp = csr_matrix(np.array([[1.0]], dtype=np.float64))
        return HawkesEMResult(
            mu_hat=mu_hat,
            alpha_hat=0.0,
            beta_hat=spec.beta0,
            log_likelihood_trace=np.array([ll0], dtype=np.float64),
            n_iter=0,
            converged=True,
            responsibilities=resp,
        )

    # Pre-compute the lower-triangular time-difference matrix once: the
    # (i, j) entry is t_i - t_j for j < i, zero on/above the diagonal.
    # `mask` is the boolean stencil; `dt` holds the time differences only
    # where the mask is True (other cells are set to a finite value so the
    # exp() call does not produce surprises — it is gated by `mask` again).
    diffs = event_times[:, None] - event_times[None, :]
    mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
    dt = np.where(mask, diffs, 0.0)

    # Initialisation: auto-infer mu0 as N/T when not supplied.
    mu_current = float(spec.mu0) if spec.mu0 is not None else float(n) / float(T)
    alpha_current = float(spec.alpha0)
    beta_current = float(spec.beta0)

    ll_history: list[float] = []
    ll_prev = _hawkes_log_likelihood(
        event_times, T, mu_current, alpha_current, beta_current, dt, mask
    )
    ll_history.append(ll_prev)

    converged = False
    n_iter = 0
    final_responsibilities_dense: NDArray[np.float64] | None = None

    tiny = np.finfo(np.float64).tiny

    for iteration_idx in range(spec.max_iter):
        iteration = iteration_idx + 1
        # ----- E-step: soft parent assignments. -----
        # Triggering kernel exp(-beta (t_i - t_j)) for j < i.
        kernel = np.where(mask, np.exp(-beta_current * dt), 0.0)
        triggering = alpha_current * kernel  # (N, N), zero except j < i
        lam_per_event = mu_current + triggering.sum(axis=1)  # (N,)
        lam_safe = np.maximum(lam_per_event, tiny)

        # p_ii: probability event i is spontaneous.
        p_diag = mu_current / lam_safe  # (N,)
        # p_ij: probability event j triggered event i (j < i).
        p_offdiag = triggering / lam_safe[:, None]  # (N, N)

        # ----- M-step: closed-form weighted MLE. -----
        sum_p_diag = float(p_diag.sum())
        # Sum of off-diagonal responsibilities = expected count of
        # triggered events.
        sum_p_offdiag = float(p_offdiag.sum())
        # Sum of p_ij * (t_i - t_j) over j < i — the weighted-mean
        # parent-child delay.
        sum_p_dt = float((p_offdiag * dt).sum())

        mu_new = sum_p_diag / T
        # beta update: Veen-Schoenberg (2008) closed form, ignoring the
        # right-boundary correction to the compensator.
        if sum_p_dt > 0.0:
            beta_new = sum_p_offdiag / sum_p_dt
        else:
            # No triggering mass — leave beta untouched.
            beta_new = beta_current
        # alpha update: maximise Q(theta) w.r.t. alpha at beta = beta_new.
        # The boundary correction matters here because the integrated
        # kernel of every event j contributes alpha/beta * (1 -
        # exp(-beta(T - t_j))) to the compensator.  Without this
        # correction (the naive ``alpha = sum p_ij / N`` form sometimes
        # cited) the M-step is not a true M-step and Q can decrease.
        boundary_T = T - event_times  # (N,)
        K_beta = float(np.sum(1.0 - np.exp(-beta_new * boundary_T)))
        if K_beta > tiny:
            alpha_new = beta_new * sum_p_offdiag / K_beta
        else:
            alpha_new = alpha_current

        # Numerical guards: keep iterates strictly positive so the next
        # log-likelihood evaluation does not feed log(0).
        mu_new = max(mu_new, tiny)
        alpha_new = max(alpha_new, 0.0)
        beta_new = max(beta_new, tiny)

        mu_current, alpha_current, beta_current = mu_new, alpha_new, beta_new

        ll_curr = _hawkes_log_likelihood(
            event_times, T, mu_current, alpha_current, beta_current, dt, mask
        )
        ll_history.append(ll_curr)
        n_iter = iteration

        denom = max(abs(ll_prev), tiny)
        if abs(ll_curr - ll_prev) / denom < spec.tol:
            converged = True
            ll_prev = ll_curr
            break
        ll_prev = ll_curr

    # If responsibilities were requested, recompute one last E-step at the
    # final parameters so the returned matrix matches (mu_hat, alpha_hat,
    # beta_hat) — the M-step inside the loop overwrites the previous
    # responsibilities before convergence is tested.
    if return_responsibilities:
        kernel = np.where(mask, np.exp(-beta_current * dt), 0.0)
        triggering = alpha_current * kernel
        lam_per_event = mu_current + triggering.sum(axis=1)
        lam_safe = np.maximum(lam_per_event, tiny)
        p_diag_final = mu_current / lam_safe
        p_offdiag_final = triggering / lam_safe[:, None]
        # Lower-triangular layout: row i = event i's parent distribution.
        # Diagonal carries spontaneous mass p_ii; below-diagonal cell
        # (i, j), j < i, carries p_ij (= prob event j triggered i).
        dense = np.zeros((n, n), dtype=np.float64)
        diag_idx = np.arange(n)
        dense[diag_idx, diag_idx] = p_diag_final
        # p_offdiag_final is already organised as rows=i, cols=j with
        # zeros above the diagonal — copy the lower-triangular block.
        dense += np.where(mask, p_offdiag_final, 0.0)
        final_responsibilities_dense = dense

    responsibilities_out: object | None = None
    if return_responsibilities:
        from scipy.sparse import csr_matrix

        assert final_responsibilities_dense is not None
        responsibilities_out = csr_matrix(final_responsibilities_dense)

    return HawkesEMResult(
        mu_hat=float(mu_current),
        alpha_hat=float(alpha_current),
        beta_hat=float(beta_current),
        log_likelihood_trace=np.asarray(ll_history, dtype=np.float64),
        n_iter=n_iter,
        converged=converged,
        responsibilities=responsibilities_out,
    )


# ----------------------------------------------------------------------
# Ogata thinning simulator
# ----------------------------------------------------------------------


def simulate_hawkes_exponential(
    mu: float,
    alpha: float,
    beta: float,
    T: float,
    *,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    r"""Simulate a Hawkes(mu, alpha exp(-beta t)) process on ``[0, T]``
    via Ogata's thinning algorithm.

    The intensity at time ``t`` is

    .. math::

        \lambda(t) = \mu + \alpha \sum_{t_i < t} e^{-\beta (t - t_i)}.

    Parameters
    ----------
    mu, alpha, beta : float
        Hawkes parameters with ``mu > 0``, ``alpha >= 0``, ``beta > 0``.
    T : float
        Simulation horizon.
    rng : numpy.random.Generator
        Random number generator (``np.random.default_rng(seed)``).

    Returns
    -------
    NDArray[float64]
        Sorted event times in ``[0, T]``.

    Raises
    ------
    ValueError
        If ``alpha / beta >= 1`` (super-critical: expected event count is
        infinite).
    """
    if not (mu > 0):
        raise ValueError(f"mu must be positive; got {mu}")
    if not (alpha >= 0):
        raise ValueError(f"alpha must be non-negative; got {alpha}")
    if not (beta > 0):
        raise ValueError(f"beta must be positive; got {beta}")
    if not (T > 0):
        raise ValueError(f"T must be positive; got {T}")
    if alpha / beta >= 1.0:
        raise ValueError(
            f"super-critical process: alpha/beta = {alpha / beta} >= 1; "
            "expected event count is infinite"
        )

    events: list[float] = []
    t = 0.0
    while True:
        # Intensity upper bound at the current time t: the cluster decay
        # is non-increasing in time, so lambda(t+) is the largest the
        # intensity can attain on [t, next_event).
        if events:
            ev_arr = np.asarray(events, dtype=np.float64)
            lam_bar = mu + alpha * float(np.sum(np.exp(-beta * (t - ev_arr))))
        else:
            lam_bar = mu

        # Sample inter-arrival from Exp(lam_bar).
        tau = float(rng.exponential(1.0 / lam_bar))
        t = t + tau
        if t > T:
            break

        # Acceptance probability: ratio of true intensity at t to lam_bar.
        if events:
            ev_arr = np.asarray(events, dtype=np.float64)
            lam_t = mu + alpha * float(np.sum(np.exp(-beta * (t - ev_arr))))
        else:
            lam_t = mu

        if rng.uniform() <= lam_t / lam_bar:
            events.append(t)

    return np.asarray(events, dtype=np.float64)
