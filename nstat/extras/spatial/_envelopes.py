"""Monte-Carlo global-rank envelope helpers (Myllymaki et al. 2017).

Pure NumPy/SciPy.  Backs :func:`nstat.extras.spatial.spatial_gof.global_envelope`.

The global-rank envelope is the multiple-testing-correct way to turn a
functional summary statistic (here the pair correlation or the
inhomogeneous K-function evaluated on a grid of lags) into a single
test with exact nominal coverage under the simulated null — unlike a
pointwise envelope, which over-rejects because of the many implicit
simultaneous tests across lags.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnvelopeResult:
    """Result of a global-rank envelope test.

    Attributes
    ----------
    r_grid
        The lag values at which the summary statistic was evaluated.
    observed
        The observed summary curve, shape ``(len(r_grid),)``.
    lo, hi
        Lower / upper global-rank envelope at the requested level, each
        shape ``(len(r_grid),)``.
    inside
        ``True`` iff the observed curve lies inside ``[lo, hi]`` at every
        lag — i.e. the global test does NOT reject the null.
    p_interval
        Conservative / liberal global-rank p-value interval
        ``(p_lo, p_hi)`` (Myllymaki et al. 2017, eq. for the rank
        envelope test).
    n_sim
        Number of Monte-Carlo simulations used.
    """

    r_grid: np.ndarray
    observed: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    inside: bool
    p_interval: tuple[float, float]
    n_sim: int


def global_rank_envelope(
    observed: np.ndarray,
    simulated: np.ndarray,
    r_grid: np.ndarray,
    *,
    alpha: float = 0.05,
) -> EnvelopeResult:
    r"""Global-rank (extreme-rank) envelope from simulated curves.

    Implements the extreme-rank length / global-rank envelope of
    Myllymaki et al. (2017, JRSS-B 79(2):381).  Given the observed curve
    and ``n_sim`` curves drawn from the null, the two-sided extreme rank
    of each curve is its smallest pointwise rank from either tail; the
    :math:`100(1-\alpha)\%` envelope is the band traced by the
    :math:`\lfloor \alpha (n_{\text{sim}}+1)\rfloor`-th most extreme
    simulated curve.

    Parameters
    ----------
    observed
        Observed summary statistic, shape ``(L,)``.
    simulated
        Null-simulated curves, shape ``(n_sim, L)``.
    r_grid
        The lags, shape ``(L,)`` (carried through to the result).
    alpha
        Global type-I error level (``0.05`` → 95% envelope).

    Returns
    -------
    EnvelopeResult
    """
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    r_grid = np.asarray(r_grid, dtype=float)
    n_sim = simulated.shape[0]
    if simulated.shape[1] != observed.shape[0]:
        raise ValueError(
            "observed and simulated must share the lag dimension; got "
            f"{observed.shape[0]} vs {simulated.shape[1]}"
        )

    # Stack observed as curve 0 so ranks are computed jointly.
    all_curves = np.vstack([observed[None, :], simulated])  # (n_sim+1, L)
    m = n_sim + 1

    # Pointwise two-sided ranks: rank from below and from above at each lag.
    # 'min' ties so the extreme rank is conservative.
    order_lo = all_curves.argsort(axis=0, kind="stable")
    rank_lo = np.empty_like(all_curves)
    rank_hi = np.empty_like(all_curves)
    # Compute average-free 1-based ranks with min-tie handling per column.
    for j in range(all_curves.shape[1]):
        col = all_curves[:, j]
        # rank from below (1 = smallest)
        order = np.argsort(col, kind="stable")
        r_asc = np.empty(m, dtype=float)
        r_asc[order] = np.arange(1, m + 1)
        # min-tie: equal values get the smallest rank in the group
        _apply_min_tie(col, r_asc)
        rank_lo[:, j] = r_asc
        # rank from above (1 = largest)
        r_desc = (m + 1) - r_asc
        # recompute min-tie for descending direction
        r_desc2 = np.empty(m, dtype=float)
        order_d = np.argsort(-col, kind="stable")
        r_desc2[order_d] = np.arange(1, m + 1)
        _apply_min_tie(-col, r_desc2)
        rank_hi[:, j] = r_desc2
    del order_lo

    # Two-sided extreme rank of each curve = min over lags of (rank from
    # either tail).
    extreme_rank = np.minimum(rank_lo.min(axis=1), rank_hi.min(axis=1))  # (m,)

    # Critical extreme-rank index for the global level alpha (0-based):
    # ``floor(alpha * m)`` counts the curves allowed to fall outside the
    # band, so the lower envelope is that order statistic (0-based) and the
    # upper is its mirror from the top.
    k0 = max(int(np.floor(alpha * m)) - 1, 0)
    # Build the envelope from the simulated curves only (exclude observed,
    # curve index 0) at the k-th smallest / largest order statistics.
    sim_sorted = np.sort(simulated, axis=0)
    lo = sim_sorted[k0]
    hi = sim_sorted[n_sim - 1 - k0]

    inside = bool(np.all((observed >= lo) & (observed <= hi)))

    # Global-rank p-value interval (Myllymaki 2017).
    r_obs = extreme_rank[0]
    p_lo = float(np.sum(extreme_rank < r_obs) / m)
    p_hi = float(np.sum(extreme_rank <= r_obs) / m)

    return EnvelopeResult(
        r_grid=r_grid,
        observed=observed,
        lo=lo,
        hi=hi,
        inside=inside,
        p_interval=(p_lo, p_hi),
        n_sim=n_sim,
    )


def _apply_min_tie(values: np.ndarray, ranks: np.ndarray) -> None:
    """In-place: collapse ranks of tied values to the group minimum."""
    order = np.argsort(values, kind="stable")
    sorted_vals = values[order]
    sorted_ranks = ranks[order]
    i = 0
    n = len(values)
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            sorted_ranks[i:j] = sorted_ranks[i:j].min()
        i = j
    ranks[order] = sorted_ranks


__all__ = ["EnvelopeResult", "global_rank_envelope"]
