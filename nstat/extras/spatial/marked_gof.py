r"""Marked / discrete-time-rescaling goodness-of-fit (pure NumPy/SciPy).

This module is the Python-only companion to the curriculum's Chapter 6
(*Spatiotemporal Point Processes*) goodness-of-fit pipeline.  Its centre
of gravity is the **discrete-time-rescaling correction** of
Haslinger, Pipa & Brown (2010), which fixes a real bug in the naive KS
test at finite bin width.

The problem (Thm 6.B.2)
-----------------------
Continuous time-rescaling says the rescaled inter-event times
:math:`z_j = \Lambda(t_j) - \Lambda(t_{j-1})` are i.i.d. Exp(1), so a KS
test against Exp(1) (equivalently Unif(0,1) after the CDF transform) is
exact.  But on **discretized** time with bin width :math:`\Delta` and
per-bin spike probabilities :math:`p_k`, the naive bin-summed compensator
is **not** Exp(1) even for the *true* model — the discreteness biases the
statistic by :math:`O(\bar\lambda\Delta)`, so the naive KS *false-rejects*
a correct model.

The fix (the randomized PIT)
----------------------------
With i.i.d. :math:`r_j \sim \text{Unif}(0,1)`, the corrected variate

.. math::

   u_j = \Bigl[\prod_{k=k_{j-1}+1}^{k_j - 1}(1 - p_k)\Bigr]
         \,(1 - r_j\, p_{k_j})

is **exactly** Unif(0,1) under the true model.  A KS test on
:math:`\{u_j\}` (averaged over several :math:`r_j` draws, or compared to a
Monte-Carlo reference band) has the correct size.

Mark spaces (Prop. 6.B.5)
-------------------------
Marked rescaling factors as (rescaled time) × (conditional mark).  A
**finite** mark space (counting reference measure → mark = channel) is
the multivariate-rescaling case; a **continuous** mark space (Lebesgue
reference measure → e.g. a waveform amplitude) is rescaled through the
conditional mark CDF :math:`F(m \mid \cdot)`.  A mis-specified
:math:`p(m\mid\cdot)` shows up as a **mark-axis** KS departure while the
time axis still passes — the signature the curriculum's optional cell
B6.2 demonstrates.

All functions are pure NumPy/SciPy.

References
----------
- Haslinger R, Pipa G, Brown E (2010). *Discrete time rescaling theorem:
  determining goodness of fit for discrete time statistical models of
  neural spiking.* Neural Computation 22(10):2477.
- Gerhard F, Haslinger R, Pipa G (2011). *Applying the multivariate
  time-rescaling theorem to neural population models.* Neural Computation
  23(6):1452.
- Tao L et al. (2018). Marked time-rescaling for population goodness-of-fit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class MarkedGOFResult:
    """Result of a discrete-time / marked goodness-of-fit run.

    Attributes
    ----------
    ks_uncorrected
        KS statistic of the naive (bin-summed) rescaled times vs Unif(0,1)
        — biased upward at finite bin width; expected to *false-reject*.
    ks_corrected
        KS statistic of the discrete-time-corrected times (averaged over
        the randomization draws), vs Unif(0,1) — should pass for the true
        model.
    ks_mark
        KS statistic on the mark axis (rescaled marks through the
        conditional mark CDF), vs Unif(0,1).  ``None`` if no marks given.
    ks_band
        The two-sided KS critical value at ``alpha`` for ``n`` events
        (Kolmogorov asymptotic ``c(alpha)/sqrt(n)``).
    inside_uncorrected, inside_corrected, inside_mark
        Whether each statistic falls inside the band (``< ks_band``).
        ``inside_*`` True ⇒ the test does NOT reject.
    n_events
        Number of inter-event intervals used.
    u_corrected
        The corrected variates from the first randomization draw (for
        plotting the empirical CDF).
    u_uncorrected
        The uncorrected variates (for the side-by-side plot).
    """

    ks_uncorrected: float
    ks_corrected: float
    ks_mark: float | None
    ks_band: float
    inside_uncorrected: bool
    inside_corrected: bool
    inside_mark: bool | None
    n_events: int
    u_corrected: np.ndarray
    u_uncorrected: np.ndarray


# ----------------------------------------------------------------------
# The discrete-time-rescaling variates (Thm 6.B.2)
# ----------------------------------------------------------------------


def uncorrected_rescaled(spike_bins: np.ndarray, p_k: np.ndarray) -> np.ndarray:
    r"""Naive bin-summed rescaled times :math:`1 - e^{-\sum p_k}`.

    This is the statistic that **false-rejects** at finite bin width — the
    bug the discrete-time correction fixes.  Provided so a demo can show
    the uncorrected curve straying outside the KS band.

    Parameters
    ----------
    spike_bins
        Sorted integer bin indices at which events occurred.
    p_k
        Per-bin spike (hazard) probabilities :math:`p_k = \lambda_k\Delta`,
        length = number of time bins.
    """
    sb = np.sort(np.asarray(spike_bins, dtype=int))
    p_k = np.asarray(p_k, dtype=float)
    u, prev = [], -1
    for kj in sb:
        u.append(1.0 - np.exp(-p_k[prev + 1: kj + 1].sum()))
        prev = int(kj)
    return np.asarray(u, dtype=float)


def corrected_rescaled(
    spike_bins: np.ndarray,
    p_k: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""Discrete-time-corrected uniform variates (Haslinger-Pipa-Brown 2010).

    .. math::

        u_j = \Bigl[\prod_{k=k_{j-1}+1}^{k_j-1}(1-p_k)\Bigr]
              (1 - r_j\, p_{k_j}), \quad r_j \sim \text{Unif}(0,1).

    Exactly Unif(0,1) under the true model.

    Parameters
    ----------
    spike_bins
        Sorted integer event-bin indices.
    p_k
        Per-bin spike probabilities.
    rng
        Random generator for the per-interval randomization draw :math:`r_j`.
    """
    sb = np.sort(np.asarray(spike_bins, dtype=int))
    p_k = np.asarray(p_k, dtype=float)
    rng = np.random.default_rng() if rng is None else rng
    u, prev = [], -1
    for kj in sb:
        kj = int(kj)
        inner = np.prod(1.0 - p_k[prev + 1: kj])      # survival over the gap
        u.append(inner * (1.0 - rng.uniform() * p_k[kj]))
        prev = kj
    return np.asarray(u, dtype=float)


def _ks_band(n: int, alpha: float) -> float:
    """Two-sided KS critical value (Kolmogorov asymptotic)."""
    c = {0.10: 1.224, 0.05: 1.358, 0.01: 1.628}.get(round(alpha, 2))
    if c is None:
        # Solve the asymptotic equation approximately for arbitrary alpha.
        c = np.sqrt(-0.5 * np.log(alpha / 2.0))
    return float(c / np.sqrt(max(n, 1)))


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def marked_time_rescaling(
    spike_bins: np.ndarray,
    marks: np.ndarray | None,
    p_k: np.ndarray,
    mark_cdf=None,
    *,
    decoded=None,
    n_draws: int = 25,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> MarkedGOFResult:
    r"""Discrete-time + marked goodness-of-fit in one call.

    Runs the time axis **both** the naive (uncorrected) way and the
    discrete-time-corrected way (Thm 6.B.2), so the bin-width bias is
    visible side by side, and — if marks and a conditional mark CDF are
    supplied — the mark axis via :math:`F(m \mid \cdot)`.

    This is the helper the curriculum Chapter 6 worked example calls as::

        from nstat.extras.spatial import marked_gof
        ks = marked_gof.marked_time_rescaling(sb, spike_m, p_k, x_hat)

    where ``x_hat`` is the decoded position at each event (passed as
    ``decoded``); the conditional mark CDF is then ``mark_cdf(m, x)``.

    Parameters
    ----------
    spike_bins
        Sorted integer event-bin indices.
    marks
        Per-event mark values (continuous-mark case), or ``None`` to run
        only the time axis.  For a *finite* (channel) mark space, see
        :func:`multivariate_time_rescaling`.
    p_k
        Per-bin spike probabilities :math:`p_k = \lambda_k \Delta`.
    mark_cdf
        Callable giving the conditional mark CDF.  Either ``mark_cdf(m)``
        (mark-only) or ``mark_cdf(m, x)`` when ``decoded`` is supplied —
        the latter mirrors the curriculum's ``F(m | x_hat)``.
    decoded
        Optional per-event covariate (e.g. decoded position) forwarded as
        the second argument to ``mark_cdf``.
    n_draws
        Number of randomization draws to average the corrected KS over.
    alpha
        KS test level (sets the band).
    rng
        Random generator.

    Returns
    -------
    MarkedGOFResult
    """
    sb = np.sort(np.asarray(spike_bins, dtype=int))
    p_k = np.asarray(p_k, dtype=float)
    rng = np.random.default_rng() if rng is None else rng
    n = len(sb)
    band = _ks_band(n, alpha)

    u_unc = uncorrected_rescaled(sb, p_k)
    draws = [corrected_rescaled(sb, p_k, np.random.default_rng(rng.integers(1 << 31)))
             for _ in range(max(n_draws, 1))]
    ks_unc = float(stats.kstest(u_unc, "uniform").statistic)
    ks_corr = float(np.mean([stats.kstest(u, "uniform").statistic for u in draws]))

    ks_mark = None
    inside_mark = None
    if marks is not None and mark_cdf is not None:
        marks = np.asarray(marks, dtype=float)
        # Marks are aligned with the ORIGINAL (unsorted) event order; the
        # curriculum passes spike_bins/marks in event order and sorts only
        # the bins.  Evaluate the mark CDF per event.
        order = np.argsort(np.asarray(spike_bins, dtype=int))
        marks_sorted = marks[order]
        if decoded is not None:
            dec = np.asarray(decoded, dtype=float)
            # decoded may be indexed by bin (length n_bins) or by event.
            if dec.shape[0] == len(p_k):
                dec_at = dec[sb]
            else:
                dec_at = dec[order] if dec.shape[0] == n else dec
            u_mark = np.array([float(mark_cdf(m, x))
                               for m, x in zip(marks_sorted, dec_at)])
        else:
            u_mark = np.array([float(mark_cdf(m)) for m in marks_sorted])
        ks_mark = float(stats.kstest(u_mark, "uniform").statistic)
        inside_mark = bool(ks_mark < band)

    return MarkedGOFResult(
        ks_uncorrected=ks_unc,
        ks_corrected=ks_corr,
        ks_mark=ks_mark,
        ks_band=band,
        inside_uncorrected=bool(ks_unc < band),
        inside_corrected=bool(ks_corr < band),
        inside_mark=inside_mark,
        n_events=n,
        u_corrected=draws[0],
        u_uncorrected=u_unc,
    )


def multivariate_time_rescaling(
    spike_bins_per_channel,
    p_k_per_channel,
    *,
    n_draws: int = 25,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict[int, MarkedGOFResult]:
    r"""Per-channel discrete-time rescaling for a finite (channel) mark space.

    The finite-mark / multivariate case (Prop. 6.B.5; Gerhard-Haslinger-
    Pipa 2011): rescale each channel by its own compensator computed under
    the *joint* history.  Returns one :class:`MarkedGOFResult` per channel.

    .. note::

       A coupling-blind fit can pass every per-channel KS while the joint
       model is wrong (Thm 6.B.3).  Per-channel passing is **necessary,
       not sufficient**.

    Parameters
    ----------
    spike_bins_per_channel
        Sequence of per-channel sorted event-bin index arrays.
    p_k_per_channel
        Sequence of per-channel per-bin spike-probability arrays.
    n_draws, alpha, rng
        As in :func:`marked_time_rescaling`.

    Returns
    -------
    dict[int, MarkedGOFResult]
        Channel index → result.
    """
    rng = np.random.default_rng() if rng is None else rng
    out: dict[int, MarkedGOFResult] = {}
    for c, (sb, pk) in enumerate(zip(spike_bins_per_channel, p_k_per_channel)):
        out[c] = marked_time_rescaling(
            sb, None, pk, None, n_draws=n_draws, alpha=alpha, rng=rng,
        )
    return out


__all__ = [
    "MarkedGOFResult",
    "uncorrected_rescaled",
    "corrected_rescaled",
    "marked_time_rescaling",
    "multivariate_time_rescaling",
]
