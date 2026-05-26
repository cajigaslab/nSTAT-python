"""Clean-room time-rescaling KS-test oracle.

Independent reference implementation used by
``tests/test_time_rescale_oracle.py`` to triangulate
:meth:`nstat.FitResult.computeKSStats` against a separately-written
computation of the same Brown / Barbieri / Ventura / Kass / Frank
"time-rescaling theorem" diagnostic.

Reference
---------
Brown EN, Barbieri R, Ventura V, Kass RE, Frank LM (2002).  "The
time-rescaling theorem and its application to neural spike train data
analysis."  Neural Computation 14:325–346.

Algorithm
---------
Given an estimated conditional intensity :math:`\\lambda(t)` and a
spike train with inter-spike intervals :math:`u_1, u_2, \\dots, u_K`,
the time-rescaling theorem says the rescaled ISIs

.. math::

    \\xi_k \\;=\\; \\int_{t_{k-1}}^{t_k} \\lambda(t)\\,dt

are i.i.d. exponential(1) under the null hypothesis that
:math:`\\lambda` is correctly specified.  Equivalently, the transformed
quantities

.. math::

    z_k \\;=\\; 1 - \\exp(-\\xi_k)

are i.i.d. uniform on [0, 1].  The Kolmogorov-Smirnov statistic

.. math::

    D_K \\;=\\; \\sup_k \\left| \\hat F_K(z_k) - z_k \\right|

quantifies the deviation between the empirical CDF :math:`\\hat F_K` of
the :math:`z` values and the theoretical uniform CDF.  Under the null,
:math:`\\sqrt K \\cdot D_K` follows the Kolmogorov distribution.

The implementation below evaluates the rescaled-ISI integral by trapezoidal
quadrature on a discretized intensity series — the same scheme used in
all published ports of the algorithm — and uses ``scipy.stats.ks_1samp``
for the standard 1-sample KS statistic and asymptotic p-value.

Independence note
-----------------
This file does NOT depend on, copy from, or adapt any specific
existing Python implementation.  It is written from the Brown 2002
paper directly.  See ``tests/parity/_third_party/README.md`` for the
clean-room policy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TimeRescaleResult:
    """KS-test result from the time-rescaling oracle.

    Attributes
    ----------
    rescaled_isis
        :math:`\\xi_k` — integral of the CIF over each ISI.  Length K
        (number of spikes).
    uniform_z
        :math:`z_k = 1 - e^{-\\xi_k}` — i.i.d. uniform under the null.
    ks_stat
        Kolmogorov-Smirnov statistic :math:`D_K`.
    ks_pvalue
        Asymptotic two-sided p-value under the null.
    """

    rescaled_isis: np.ndarray
    uniform_z: np.ndarray
    ks_stat: float
    ks_pvalue: float


def time_rescaling_ks_test(
    intensity: np.ndarray,
    spike_indicator: np.ndarray,
    *,
    dt: float = 1e-3,
) -> TimeRescaleResult:
    """Run the Brown 2002 time-rescaling KS test.

    Parameters
    ----------
    intensity
        Estimated conditional intensity :math:`\\lambda(t)`, sampled at
        uniform spacing ``dt``.  Shape ``(N,)``.  Values must be
        non-negative.
    spike_indicator
        Binary indicator of spike occurrence on the same grid.  Shape
        ``(N,)``.  Non-zero entries are interpreted as spikes.
    dt
        Bin width in the same time units as ``1/intensity`` (default 1
        ms — matches nstat's per-millisecond intensity convention).

    Returns
    -------
    TimeRescaleResult

    Raises
    ------
    ValueError
        If ``intensity`` and ``spike_indicator`` differ in length, if
        ``intensity`` contains negative entries, or if fewer than 2
        spikes are present (KS test requires K >= 2 inter-spike
        intervals; with only 1 spike there are 0 ISIs).
    """
    intensity = np.asarray(intensity, dtype=float).ravel()
    spike_indicator = np.asarray(spike_indicator).ravel()

    if intensity.shape != spike_indicator.shape:
        raise ValueError(
            f"intensity and spike_indicator length mismatch: "
            f"{intensity.shape} vs {spike_indicator.shape}"
        )
    if np.any(intensity < 0):
        raise ValueError("intensity must be non-negative everywhere")

    spike_idx = np.flatnonzero(spike_indicator).astype(int)
    if spike_idx.size < 2:
        raise ValueError(
            f"Need at least 2 spikes to form K >= 1 ISI; got {spike_idx.size}"
        )

    # Trapezoidal integral of intensity over each closed inter-spike
    # interval [t_{k-1}, t_k].
    cumulative = np.concatenate(([0.0], np.cumsum(intensity) * dt))
    rescaled_isis = cumulative[spike_idx[1:] + 1] - cumulative[spike_idx[:-1] + 1]

    # Time-rescaled ISIs should be i.i.d. exponential(1) under H0.
    # The transform z = 1 - exp(-xi) maps to uniform(0, 1).
    uniform_z = 1.0 - np.exp(-rescaled_isis)

    ks_stat, ks_pvalue = stats.ks_1samp(
        uniform_z, stats.uniform(loc=0.0, scale=1.0).cdf
    )

    return TimeRescaleResult(
        rescaled_isis=rescaled_isis,
        uniform_z=uniform_z,
        ks_stat=float(ks_stat),
        ks_pvalue=float(ks_pvalue),
    )


__all__ = ["TimeRescaleResult", "time_rescaling_ks_test"]
