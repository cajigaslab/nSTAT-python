r"""SBM-prior multivariate Hawkes EM.

Latent-block extension of the Veen-Schoenberg branching-responsibilities EM.
N processes are partitioned into K latent communities; intra- and
inter-community coupling is governed by a K x K alpha matrix with a shared
exponential decay. Closed-form M-step for (alpha, beta, baseline rates);
greedy hard-assignment update for z (Linderman, Adams & Pillow 2014).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


__all__ = [
    "SBMHawkesSpec",
    "SBMHawkesResult",
    "em_sbm_hawkes",
    "simulate_sbm_hawkes",
]


# ----------------------------------------------------------------------
# Configuration + result dataclasses
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SBMHawkesSpec:
    """Initial guesses and convergence config for SBM-Hawkes EM.

    Parameters
    ----------
    n_blocks : int
        Number of latent communities K. Must be >= 1.
    alpha0 : float
        Initial intra-block coupling amplitude. Default 0.3.
    alpha0_off : float
        Initial inter-block coupling amplitude. Default 0.05.
    beta0 : float
        Initial exponential decay (shared across all (k, k') pairs). Default 1.0.
    max_iter : int
        Maximum EM iterations. Default 100.
    tol : float
        Relative log-likelihood tolerance for convergence. Default 1e-6.
    z_init : str
        Block-assignment initialisation. One of {"random", "kmeans-rate"}.
        "kmeans-rate" runs k-means on each process's empirical rate;
        "random" assigns uniformly.  Default "kmeans-rate".
    """

    n_blocks: int
    alpha0: float = 0.3
    alpha0_off: float = 0.05
    beta0: float = 1.0
    max_iter: int = 100
    tol: float = 1e-6
    z_init: str = "kmeans-rate"

    def __post_init__(self) -> None:
        if not (self.n_blocks >= 1):
            raise ValueError(f"n_blocks must be >= 1; got {self.n_blocks}")
        if not (self.alpha0 > 0):
            raise ValueError(f"alpha0 must be positive; got {self.alpha0}")
        if not (self.alpha0_off >= 0):
            raise ValueError(
                f"alpha0_off must be non-negative; got {self.alpha0_off}"
            )
        if not (self.beta0 > 0):
            raise ValueError(f"beta0 must be positive; got {self.beta0}")
        if not (self.max_iter >= 1):
            raise ValueError(f"max_iter must be >= 1; got {self.max_iter}")
        if not (self.tol > 0):
            raise ValueError(f"tol must be positive; got {self.tol}")
        if self.z_init not in {"random", "kmeans-rate"}:
            raise ValueError(
                f"z_init must be 'random' or 'kmeans-rate'; got {self.z_init!r}"
            )
        if self.alpha0 / self.beta0 >= 1.0:
            warnings.warn(
                "intra-block branching ratio super-critical at init; "
                "convergence may stall",
                UserWarning,
                stacklevel=2,
            )


@dataclass(frozen=True)
class SBMHawkesResult:
    """Fitted SBM-Hawkes parameters from hard-EM.

    Attributes
    ----------
    z_hat : (N,) int ndarray
        Recovered hard block assignments (block labels in {0..K-1};
        recovery is only meaningful up to label permutation).
    A_hat : (K, K) ndarray
        Recovered block-coupling matrix. A_hat[k, k'] is the amplitude
        for events in block k triggering events in block k'.
    beta_hat : float
        Shared exponential decay.
    mu_hat : (N,) ndarray
        Per-process baseline rates.
    log_likelihood_trace : (n_iter + 1,) ndarray
        Per-iteration LL.
    n_iter : int
        EM iterations performed.
    converged : bool
        Whether convergence tolerance was met before max_iter.
    n_processes : int
        Echoed N.
    n_blocks : int
        Echoed K.
    """

    z_hat: np.ndarray
    A_hat: np.ndarray
    beta_hat: float
    mu_hat: np.ndarray
    log_likelihood_trace: np.ndarray
    n_iter: int
    converged: bool
    n_processes: int
    n_blocks: int


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _validate_events(
    event_times_per_process: list[NDArray[np.float64]], T: float
) -> tuple[list[NDArray[np.float64]], int]:
    """Validate input event lists; return (typed list, N)."""
    if len(event_times_per_process) == 0:
        raise ValueError("at least one process required")
    n = len(event_times_per_process)
    typed: list[NDArray[np.float64]] = []
    max_event = -np.inf
    for idx, events in enumerate(event_times_per_process):
        arr = np.asarray(events, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(
                f"process {idx}: event_times must be 1-D; got shape {arr.shape}"
            )
        if arr.size == 0:
            raise ValueError(
                f"process {idx} has no events; SBM-Hawkes does not handle "
                "silent processes — drop them before fitting"
            )
        if arr.size >= 2 and np.any(np.diff(arr) < 0):
            raise ValueError(
                f"process {idx}: event_times must be sorted ascending"
            )
        if arr[-1] >= T:
            raise ValueError(
                f"process {idx}: T={T} must exceed last event time {arr[-1]}"
            )
        max_event = max(max_event, float(arr[-1]))
        typed.append(arr)
    if not (T > max_event):
        # Defensive: covered by per-process check above, but keeps the
        # invariant explicit for the caller.
        raise ValueError(
            f"T={T} must exceed the maximum event time {max_event}"
        )
    return typed, n


def _kmeans_1d_rate_init(
    rates: NDArray[np.float64],
    k: int,
    rng: np.random.Generator,
    *,
    n_init: int = 5,
    max_iter: int = 50,
) -> NDArray[np.int_]:
    """Tiny k-means on the (N,) empirical-rate vector.

    Pure NumPy; vectorised over the N-by-k assignment-distance matrix.
    Returns an ``(N,)`` integer assignment in ``{0, ..., k-1}``.  When
    ``k == 1`` returns all-zero; when ``N == k`` returns ``range(k)``
    (every process is its own block).
    """
    n = rates.size
    if k == 1:
        return np.zeros(n, dtype=np.int_)
    if k == n:
        # Each process its own block — the rank-order assignment makes
        # the labels deterministic across seeds.
        return np.argsort(np.argsort(rates))

    best_z: NDArray[np.int_] | None = None
    best_inertia = np.inf
    rates_2d = rates[:, None]  # (N, 1)
    for _ in range(n_init):
        # k-means++ style initialisation: first centre uniform, others
        # weighted by squared distance to nearest existing centre.
        # ``j_minus_one`` counts how many centres have been seeded; the
        # j-th iteration places the (j_minus_one + 1)-th centre.  The
        # loop runs k-1 times to seed the remaining k-1 centres.
        centres = np.empty(k, dtype=np.float64)
        centres[0] = float(rng.choice(rates))
        for j_minus_one in range(k - 1):
            j = j_minus_one + 1
            d2 = np.min(
                (rates[:, None] - centres[None, :j]) ** 2, axis=1
            )
            total = float(d2.sum())
            if total <= 0.0:
                centres[j] = float(rng.choice(rates))
            else:
                probs = d2 / total
                centres[j] = float(rng.choice(rates, p=probs))

        z_local = np.zeros(n, dtype=np.int_)
        for _it in range(max_iter):
            # Distance to each centre.
            dists = (rates_2d - centres[None, :]) ** 2  # (N, k)
            z_new = np.argmin(dists, axis=1).astype(np.int_)
            if np.array_equal(z_new, z_local) and _it > 0:
                z_local = z_new
                break
            z_local = z_new
            # Update centres; handle empty clusters by reseeding to a
            # random point.
            for j in range(k):
                members = rates[z_local == j]
                if members.size == 0:
                    centres[j] = float(rng.choice(rates))
                else:
                    centres[j] = float(members.mean())

        # Inertia = within-cluster sum of squared distance.
        inertia = 0.0
        for j in range(k):
            members = rates[z_local == j]
            if members.size > 0:
                inertia += float(np.sum((members - members.mean()) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_z = z_local.copy()

    assert best_z is not None
    return best_z


# ----------------------------------------------------------------------
# Cached interaction structure: R[n][m, i] = sum_{j: t_j^m < t_i^n}
# exp(-beta * (t_i^n - t_j^m)).
#
# Once R is built (cost O(sum_n n_n * sum_m n_m)) the per-process
# intensity is `lam_n(t_i^n) = mu_n + sum_m A[z_m, z_n] * R[n][m, i]`,
# which factorises block-coupling from event timings.  Every subsequent
# LL evaluation under a relabelling of z is O(sum_n n_n * N) — fast
# enough to drive the greedy z update without ballooning.
# ----------------------------------------------------------------------


def _build_kernel_response(
    events_list: list[NDArray[np.float64]],
    beta: float,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Build per-receiver kernel-response and delay matrices.

    Returns
    -------
    R : list of (M_m, n_n_receiver) ndarrays
        Wait — the natural shape is per receiver n, an (N_n, N) array
        where ``R[n][i, m] = sum_{j: t_j^m < t_i^n} exp(-beta (t_i^n
        - t_j^m))``.  So a list of length N, each (n_n, N).
    Rdt : list of (N_n, N) ndarrays
        Same layout but weighted by the delay: ``Rdt[n][i, m] = sum_j
        (t_i^n - t_j^m) exp(-beta (t_i^n - t_j^m))``.  Used by the
        M-step beta update.
    """
    n_proc = len(events_list)
    R: list[NDArray[np.float64]] = []
    Rdt: list[NDArray[np.float64]] = []
    for events_n in events_list:
        n_n = events_n.size
        R_n = np.zeros((n_n, n_proc), dtype=np.float64)
        Rdt_n = np.zeros((n_n, n_proc), dtype=np.float64)
        for m_idx, events_m in enumerate(events_list):
            # dt[i, j] = t_i^n - t_j^m
            dt = events_n[:, None] - events_m[None, :]
            mask = dt > 0.0
            kernel = np.where(mask, np.exp(-beta * dt), 0.0)
            R_n[:, m_idx] = kernel.sum(axis=1)
            Rdt_n[:, m_idx] = (kernel * np.where(mask, dt, 0.0)).sum(axis=1)
        R.append(R_n)
        Rdt.append(Rdt_n)
    return R, Rdt


def _sender_boundary(
    events_list: list[NDArray[np.float64]],
    T: float,
    beta: float,
) -> NDArray[np.float64]:
    """Per-process boundary correction ``sum_j (1 - exp(-beta(T - t_j^m)))``."""
    n_proc = len(events_list)
    out = np.zeros(n_proc, dtype=np.float64)
    for m_idx, events_m in enumerate(events_list):
        out[m_idx] = float(np.sum(1.0 - np.exp(-beta * (T - events_m))))
    return out


def _log_likelihood_from_R(
    R: list[NDArray[np.float64]],
    sender_bnd: NDArray[np.float64],
    z: NDArray[np.int_],
    A: NDArray[np.float64],
    beta: float,
    mu: NDArray[np.float64],
    T: float,
) -> float:
    """LL given precomputed kernel responses and sender-boundary terms."""
    tiny = np.finfo(np.float64).tiny
    n_blocks = A.shape[0]
    # Event-side: sum_n sum_i log(mu_n + sum_m A[z_m, z_n] R[n][i, m]).
    ll_events = 0.0
    n_proc = len(R)
    for n_idx in range(n_proc):
        k_recv = int(z[n_idx])
        # Block-weighted coupling: for each sender m, weight A[z_m, k_recv].
        weights = A[z, k_recv]  # (N,)
        lam = mu[n_idx] + R[n_idx] @ weights  # (N_n,)
        lam_safe = np.maximum(lam, tiny)
        ll_events += float(np.sum(np.log(lam_safe)))

    # Compensator-side: mu sum + sum_m sum_{k'} A[z_m, k'] K_{k'} *
    # sender_bnd[m] / beta.
    K_counts = np.bincount(z, minlength=n_blocks).astype(np.float64)  # (K,)
    # For each sender m the row A[z_m, :] dot K_counts gives the total
    # outgoing receiver-block coupling.
    row_totals = A[z, :] @ K_counts  # (N,)
    ll_comp = float(mu.sum() * T) + float(np.sum(row_totals * sender_bnd) / beta)
    return ll_events - ll_comp


# ----------------------------------------------------------------------
# EM algorithm
# ----------------------------------------------------------------------


def em_sbm_hawkes(
    event_times_per_process: list[NDArray[np.float64]],
    T: float,
    *,
    spec: SBMHawkesSpec,
    seed: int | None = None,
) -> SBMHawkesResult:
    """Fit N processes to a K-block Hawkes mixture via hard-EM.

    Parameters
    ----------
    event_times_per_process : list of (n_i,) ndarrays
        Sorted event times per process; length N gives the number of processes.
    T : float
        Observation horizon. Must satisfy ``T > max(t for all events)``.
    spec : SBMHawkesSpec
        Configuration.
    seed : int or None
        Reproducibility seed for k-means init or random init.

    Returns
    -------
    SBMHawkesResult

    Notes
    -----
    Block labels are arbitrary (only the partition matters); recovery
    tests must compare ``z_hat`` to ground truth up to label permutation
    (e.g., via :func:`scipy.optimize.linear_sum_assignment` on the
    confusion matrix).

    The implementation precomputes per-receiver kernel-response
    matrices ``R[n][i, m] = sum_{j: t_j^m < t_i^n} exp(-beta(t_i^n -
    t_j^m))`` once per beta value.  This decouples block-coupling
    weights ``A[z_m, z_n]`` from event timings: every subsequent LL
    evaluation under a z relabelling is ``O(sum_n n_n * N)`` rather than
    ``O(sum_n n_n * sum_m n_m)``.  Memory is ``O(sum_n n_n * N)`` — fine
    for ``N <= ~30`` processes and total events ``<= ~5000``.

    References
    ----------
    Linderman SW, Adams RP, Pillow JW (2014). *Discovering latent network
    structure in point process data.* NeurIPS.

    Veen A, Schoenberg FP (2008). *Estimation of space-time branching
    process models in seismology using an EM-type algorithm.* JASA
    103(482):614-624.
    """
    events_list, n_processes = _validate_events(event_times_per_process, T)
    n_blocks = spec.n_blocks
    if n_blocks > n_processes:
        raise ValueError(
            f"n_blocks={n_blocks} exceeds n_processes={n_processes}; "
            "drop blocks or add processes"
        )

    rng = np.random.default_rng(seed)
    tiny = np.finfo(np.float64).tiny

    # ---- Initialise z ----
    rates = np.array(
        [events.size / T for events in events_list], dtype=np.float64
    )
    if spec.z_init == "kmeans-rate":
        z = _kmeans_1d_rate_init(rates, n_blocks, rng)
    else:
        # "random" assignment, but guarantee every block is non-empty
        # so the M-step doesn't divide by zero.
        perm = rng.permutation(n_processes)
        z = np.empty(n_processes, dtype=np.int_)
        z[perm[:n_blocks]] = np.arange(n_blocks)
        if n_processes > n_blocks:
            z[perm[n_blocks:]] = rng.integers(0, n_blocks, n_processes - n_blocks)

    # ---- Initialise A, beta, mu ----
    A = np.full((n_blocks, n_blocks), spec.alpha0_off, dtype=np.float64)
    np.fill_diagonal(A, spec.alpha0)
    beta = float(spec.beta0)
    mu = np.maximum(rates.copy(), tiny)

    # ---- Build initial kernel response cache ----
    R, Rdt = _build_kernel_response(events_list, beta)
    sender_bnd = _sender_boundary(events_list, T, beta)

    ll_history: list[float] = []
    ll_prev = _log_likelihood_from_R(
        R, sender_bnd, z, A, beta, mu, T
    )
    ll_history.append(ll_prev)

    converged = False
    n_iter = 0

    for iteration_idx in range(spec.max_iter):
        n_iter = iteration_idx + 1

        # =============================================================
        # E-step accumulators
        # =============================================================
        # For each receiver n with current k_recv = z[n]:
        #   lam_i = mu_n + sum_m A[z_m, k_recv] R[n][i, m]
        #   contrib_im = A[z_m, k_recv] R[n][i, m] / lam_i  (after summing
        #                                                    over i: total
        #                                                    mass from m)
        # We need:
        #   sum_p_diag_per_n[n]              for mu update
        #   S[k_send, k_recv]                for A update
        #   sum_p_off_total, sum_p_dt_total  for beta update
        sum_p_diag_per_n = np.zeros(n_processes, dtype=np.float64)
        S = np.zeros((n_blocks, n_blocks), dtype=np.float64)
        sum_p_off_total = 0.0
        sum_p_dt_total = 0.0

        for n_idx in range(n_processes):
            k_recv = int(z[n_idx])
            weights = A[z, k_recv]  # (N,)
            lam = mu[n_idx] + R[n_idx] @ weights  # (N_n,)
            lam_safe = np.maximum(lam, tiny)
            # Spontaneous responsibility, summed over events of process n.
            sum_p_diag_per_n[n_idx] = float(np.sum(mu[n_idx] / lam_safe))
            # Per-(m) off-diagonal responsibility, summed over events i:
            # p_off_sum[m] = sum_i A[z_m, k_recv] R[n][i, m] / lam_safe[i].
            inv_lam = 1.0 / lam_safe  # (N_n,)
            # Numerator weights = A[z_m, k_recv] * R[n][:, m] for each m;
            # multiplied row-wise by inv_lam, summed over i.
            #   per_m[m] = weights[m] * sum_i R[n][i, m] * inv_lam[i]
            R_inv = R[n_idx] * inv_lam[:, None]  # (N_n, N)
            per_m_sum = weights * R_inv.sum(axis=0)  # (N,)
            sum_p_off_total += float(per_m_sum.sum())
            # Delay-weighted: per_m_dt[m] = weights[m] * sum_i Rdt[n][i, m] * inv_lam[i]
            Rdt_inv = Rdt[n_idx] * inv_lam[:, None]
            per_m_dt = weights * Rdt_inv.sum(axis=0)
            sum_p_dt_total += float(per_m_dt.sum())
            # Accumulate into S[z_m, k_recv].
            np.add.at(S[:, k_recv], z, per_m_sum)

        # ---- M-step ----
        mu_new = sum_p_diag_per_n / T
        mu_new = np.maximum(mu_new, tiny)

        if sum_p_dt_total > 0.0 and sum_p_off_total > 0.0:
            beta_new = sum_p_off_total / sum_p_dt_total
        else:
            beta_new = beta
        beta_new = max(beta_new, tiny)

        # Sender boundary at the new beta.
        sender_bnd_new = _sender_boundary(events_list, T, beta_new)
        # Block-sender boundary: sum_{m: z_m = k_send} sender_bnd_new[m].
        block_sender_bnd = np.zeros(n_blocks, dtype=np.float64)
        np.add.at(block_sender_bnd, z, sender_bnd_new)

        # Receiver block sizes |{n : z_n = k'}|.
        K_counts = np.bincount(z, minlength=n_blocks).astype(np.float64)

        A_new = np.zeros_like(A)
        for k_send in range(n_blocks):
            denom_send = block_sender_bnd[k_send]
            if denom_send <= tiny:
                A_new[k_send, :] = A[k_send, :]
                continue
            for k_recv in range(n_blocks):
                K_recv = K_counts[k_recv]
                if K_recv <= 0.0:
                    A_new[k_send, k_recv] = A[k_send, k_recv]
                    continue
                A_new[k_send, k_recv] = (
                    beta_new * S[k_send, k_recv] / (K_recv * denom_send)
                )
        A_new = np.maximum(A_new, 0.0)

        # Commit (mu, beta, A); rebuild R / sender_bnd if beta changed.
        beta_changed = abs(beta_new - beta) > 1e-12
        mu = mu_new
        beta = beta_new
        A = A_new
        if beta_changed:
            R, Rdt = _build_kernel_response(events_list, beta)
            sender_bnd = sender_bnd_new
        else:
            sender_bnd = sender_bnd_new

        # =============================================================
        # Greedy z update (accept-if-LL-up safeguard)
        # =============================================================
        # The closed-form (A, beta, mu) update above was conditioned on
        # the old z; updating z afterwards is generally NOT monotone in
        # the joint LL.  The accept-only-if-not-worse safeguard makes
        # the z step monotone in its own right.
        current_ll = _log_likelihood_from_R(R, sender_bnd, z, A, beta, mu, T)
        for n_idx in range(n_processes):
            best_k = int(z[n_idx])
            best_ll = current_ll
            for k_cand in range(n_blocks):
                if k_cand == best_k:
                    continue
                z_trial = z.copy()
                z_trial[n_idx] = k_cand
                ll_trial = _log_likelihood_from_R(
                    R, sender_bnd, z_trial, A, beta, mu, T
                )
                if ll_trial > best_ll:
                    best_ll = ll_trial
                    best_k = k_cand
            if best_k != int(z[n_idx]):
                z[n_idx] = best_k
                current_ll = best_ll

        # ---- Convergence check ----
        ll_curr = current_ll
        ll_history.append(ll_curr)

        denom = max(abs(ll_prev), tiny)
        if abs(ll_curr - ll_prev) / denom < spec.tol:
            converged = True
            ll_prev = ll_curr
            break
        ll_prev = ll_curr

    return SBMHawkesResult(
        z_hat=z.copy(),
        A_hat=A.copy(),
        beta_hat=float(beta),
        mu_hat=mu.copy(),
        log_likelihood_trace=np.asarray(ll_history, dtype=np.float64),
        n_iter=n_iter,
        converged=converged,
        n_processes=n_processes,
        n_blocks=n_blocks,
    )


# ----------------------------------------------------------------------
# Multivariate Ogata thinning simulator
# ----------------------------------------------------------------------


def simulate_sbm_hawkes(
    z: NDArray[np.int_],
    A: NDArray[np.float64],
    beta: float,
    mu: NDArray[np.float64],
    T: float,
    *,
    rng: np.random.Generator,
) -> list[NDArray[np.float64]]:
    r"""Simulate N multivariate Hawkes processes with SBM coupling on ``[0, T]``.

    Multivariate Ogata thinning.  Per-process accumulators
    ``S_m(t) = sum_{t_j^m < t} exp(-beta(t - t_j^m))`` are advanced
    multiplicatively at every step (``S_m(t + dt) = S_m(t) * exp(-beta
    dt)``), which keeps the per-event cost ``O(N)`` rather than
    ``O(total_events)``.  Rejects super-critical configurations
    (spectral radius of ``A / beta`` exceeds 1) for which the expected
    event count diverges.

    Parameters
    ----------
    z : (N,) int ndarray
        Per-process block assignments.
    A : (K, K) ndarray
        Block-coupling matrix.
    beta : float
        Shared decay.
    mu : (N,) ndarray
        Baselines.
    T : float
        Horizon.
    rng : numpy.random.Generator
        ``np.random.default_rng(seed)``.

    Returns
    -------
    list of (n_i,) sorted event-time arrays
    """
    z_arr = np.asarray(z, dtype=np.int_)
    A_arr = np.asarray(A, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)

    if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
        raise ValueError(f"A must be square; got shape {A_arr.shape}")
    n_blocks = A_arr.shape[0]
    n = z_arr.size
    if mu_arr.shape != (n,):
        raise ValueError(
            f"mu shape {mu_arr.shape} does not match number of processes {n}"
        )
    if np.any(z_arr < 0) or np.any(z_arr >= n_blocks):
        raise ValueError(
            f"z entries must lie in [0, {n_blocks}); got range "
            f"[{z_arr.min()}, {z_arr.max()}]"
        )
    if not (beta > 0):
        raise ValueError(f"beta must be positive; got {beta}")
    if np.any(mu_arr <= 0):
        raise ValueError("mu entries must all be positive")
    if np.any(A_arr < 0):
        raise ValueError("A entries must be non-negative")
    if not (T > 0):
        raise ValueError(f"T must be positive; got {T}")

    # Spectral-radius stability check on the *pairwise* coupling matrix
    # ``A_pair[m, n] = A[z_m, z_n]`` divided by beta.  This is equivalent
    # to checking the spectral radius of ``A * diag(K_counts) / beta``
    # where ``K_counts[k]`` is the size of block k (the non-zero
    # eigenvalues of the N x N pair matrix are exactly the eigenvalues
    # of the K x K weighted block matrix).
    K_counts_pre = np.bincount(z_arr, minlength=n_blocks).astype(np.float64)
    B = (A_arr * K_counts_pre[None, :]) / beta
    eigvals = np.linalg.eigvals(B)
    spectral_radius = float(np.max(np.abs(eigvals)))
    if spectral_radius >= 1.0:
        raise ValueError(
            f"super-critical: spectral radius of A/beta = {spectral_radius} "
            ">= 1; expected event count diverges"
        )

    # Pre-extract A[z, z] -> the (N, N) per-pair coupling matrix.  This
    # lets the per-event intensity update be a simple matvec.
    A_pair = A_arr[np.ix_(z_arr, z_arr)]  # (N, N): A_pair[m, n] = A[z_m, z_n]

    events: list[list[float]] = [[] for _ in range(n)]
    # S[m] = sum_{t_j^m < t} exp(-beta(t - t_j^m)) — advances by *= exp(-beta dt).
    S = np.zeros(n, dtype=np.float64)
    t = 0.0

    while True:
        # Intensity at the current t (upper bound for the next interval
        # because the cluster decay is non-increasing between events):
        #   lam_n(t+) = mu_n + sum_m A_pair[m, n] * S[m]
        lam = mu_arr + A_pair.T @ S  # (N,)
        lam_total = float(lam.sum())
        if lam_total <= 0.0:
            break
        tau = float(rng.exponential(1.0 / lam_total))
        t_new = t + tau
        if t_new >= T:
            break
        # Advance S to t_new — multiplicative decay.
        decay = float(np.exp(-beta * tau))
        S = S * decay
        t = t_new

        lam_t = mu_arr + A_pair.T @ S
        lam_t_total = float(lam_t.sum())
        accept_prob = min(1.0, lam_t_total / lam_total)
        if rng.uniform() <= accept_prob:
            probs = lam_t / lam_t_total
            n_chosen = int(rng.choice(n, p=probs))
            events[n_chosen].append(t)
            # The new event contributes +1 to S[n_chosen] going forward
            # (its kernel value at t is exp(0) = 1).
            S[n_chosen] += 1.0

    return [np.asarray(ev, dtype=np.float64) for ev in events]
