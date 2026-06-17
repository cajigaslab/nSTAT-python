r"""Marked / discrete-time-rescaling goodness-of-fit (pure NumPy/SciPy).

A Python-only companion to the discrete-time / marked
goodness-of-fit framework.  Its centre of gravity is the
**discrete-time-rescaling correction** of Haslinger, Pipa & Brown
(2010), which fixes a real bug in the naive KS test at finite bin
width.

The problem
-----------
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

Mark spaces
-----------
Marked rescaling factors as (rescaled time) × (conditional mark).  A
**finite** mark space (counting reference measure → mark = channel) is
the multivariate-rescaling case (Gerhard-Haslinger-Pipa 2011); a
**continuous** mark space (Lebesgue
reference measure → e.g. a waveform amplitude) is rescaled through the
conditional mark CDF :math:`F(m \mid \cdot)`.  A mis-specified
:math:`p(m\mid\cdot)` shows up as a **mark-axis** KS departure while the
time axis still passes.

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
# The discrete-time-rescaling variates (Haslinger-Pipa-Brown 2010)
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
    discrete-time-corrected way (Haslinger-Pipa-Brown 2010), so the
    bin-width bias is visible side by side, and — if marks and a
    conditional mark CDF are supplied — the mark axis via
    :math:`F(m \mid \cdot)`.

    Typical call::

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
        the latter is the conditional mark CDF ``F(m | x_hat)``.
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
        # Marks are aligned with the ORIGINAL (unsorted) event order;
        # callers pass spike_bins / marks in event order and only the
        # bins get sorted.  Evaluate the mark CDF per event.
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


# ----------------------------------------------------------------------
# Rescaled-time autocorrelation (independence diagnostic)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class RescaledACFResult:
    r"""Autocorrelation of the rescaled-time variates with a Bartlett band.

    The discrete-time-rescaling KS test of
    :func:`marked_time_rescaling` checks the *marginal* distribution of
    the rescaled variates (Unif(0,1)) but is blind to serial dependence
    (Brown et al. 2002).  This dataclass carries the lag-1..``n_lags``
    autocorrelation of the normal-score-transformed uniforms,
    :math:`z_j = \Phi^{-1}(u_j)`, together with the asymptotic
    :math:`\pm 1.96/\sqrt{n}` Bartlett band (Andersen 1997; Truccolo
    et al. 2005) and a per-lag in-band flag.

    Attributes
    ----------
    lags
        Integer lags ``1..n_lags``, shape ``(n_lags,)``.
    acf
        Sample autocorrelation of :math:`z_j` at each lag, shape
        ``(n_lags,)``.
    band
        Two-sided :math:`\pm 1.96 / \sqrt{n}` Bartlett band — a scalar,
        applied symmetrically about zero.
    inside_band
        Boolean ``(n_lags,)`` mask: ``True`` where ``|acf| < band``.
    """

    lags: np.ndarray
    acf: np.ndarray
    band: float
    inside_band: np.ndarray


def rescaled_acf(
    u_rescaled: np.ndarray,
    *,
    n_lags: int = 20,
) -> RescaledACFResult:
    r"""Autocorrelation of the rescaled-time variates with a Bartlett band.

    Steps:

    1. Apply the normal-score transform :math:`z_j = \Phi^{-1}(u_j)`
       (clamped to :math:`[10^{-12},\,1 - 10^{-12}]` so the tails are
       finite).  Under the true model, :math:`u_j \sim \text{Unif}(0,1)`
       i.i.d., so :math:`z_j \sim \mathcal{N}(0,1)` i.i.d.
    2. Compute the centred sample autocorrelation of :math:`z_j` at lags
       1 through ``n_lags`` via :func:`numpy.correlate`.
    3. Flag in-band lags using the asymptotic
       :math:`\pm 1.96 / \sqrt{n}` Bartlett band.

    Parameters
    ----------
    u_rescaled
        The rescaled-time uniforms, typically
        :attr:`MarkedGOFResult.u_corrected`.  Must contain at least
        ``n_lags + 2`` values.
    n_lags
        Number of positive lags to report.

    Returns
    -------
    RescaledACFResult

    Raises
    ------
    ValueError
        If ``u_rescaled`` has fewer than ``n_lags + 2`` values.

    Notes
    -----
    *Confidence: high* on the in-band behaviour under the true model
    (the asymptotic Bartlett band is exact in the i.i.d. limit) and on
    the band-violation behaviour under serial dependence.  Use as a
    complement to the marginal KS test, not a replacement.
    """
    u = np.asarray(u_rescaled, dtype=float).ravel()
    if n_lags < 1:
        raise ValueError(f"n_lags must be >= 1; got {n_lags!r}")
    if u.size < n_lags + 2:
        raise ValueError(
            f"u_rescaled must contain at least n_lags + 2 = {n_lags + 2} "
            f"values; got {u.size}"
        )
    u_clamped = np.clip(u, 1e-12, 1.0 - 1e-12)
    z = stats.norm.ppf(u_clamped)
    z = z - z.mean()
    full = np.correlate(z, z, mode="full")
    mid = full.size // 2
    var = full[mid]
    if var <= 0:
        acf = np.zeros(n_lags, dtype=float)
    else:
        acf = full[mid + 1: mid + 1 + n_lags] / var
    band = float(1.96 / np.sqrt(u.size))
    lags = np.arange(1, n_lags + 1, dtype=int)
    inside = np.abs(acf) < band
    return RescaledACFResult(
        lags=lags,
        acf=acf,
        band=band,
        inside_band=inside,
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

    The finite-mark / multivariate case (Gerhard-Haslinger-Pipa 2011):
    rescale each channel by its own compensator computed under the
    *joint* history.  Returns one :class:`MarkedGOFResult` per channel.

    .. note::

       A coupling-blind fit can pass every per-channel KS while the
       joint model is wrong (Gerhard-Haslinger-Pipa 2011, §4):
       per-channel passing is **necessary, not sufficient**.  The
       population-level diagnostic that closes that gap is
       :func:`nstat.population_time_rescale` (Tao, Weber, Arai & Eden
       2018); see :func:`multivariate_gof_with_coupling` for a single
       call that runs both tests on the same data.

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

    See Also
    --------
    nstat.population_time_rescale :
        Population-level (Tao et al. 2018) coupling diagnostic that
        catches misfit invisible to per-channel KS.
    multivariate_gof_with_coupling :
        Convenience wrapper that runs this function *and*
        :func:`nstat.population_time_rescale` on the same data and
        returns both results.
    """
    rng = np.random.default_rng() if rng is None else rng
    out: dict[int, MarkedGOFResult] = {}
    for c, (sb, pk) in enumerate(zip(spike_bins_per_channel, p_k_per_channel)):
        out[c] = marked_time_rescaling(
            sb, None, pk, None, n_draws=n_draws, alpha=alpha, rng=rng,
        )
    return out


@dataclass(frozen=True)
class CoupledMarkedGOFResult:
    """Bundled result of :func:`multivariate_gof_with_coupling`.

    Attributes
    ----------
    per_channel
        Mapping ``channel_index -> MarkedGOFResult`` returned by
        :func:`multivariate_time_rescaling` — the discrete-time-corrected
        KS test for each channel in isolation.
    population
        :class:`~nstat.PopulationTimeRescaleResult` returned by
        :func:`nstat.population_time_rescale` — the ground-process KS and
        marked-region Pearson :math:`\\chi^2` over the joint
        :math:`(\\tau, k)` space, which catches cross-channel coupling
        misfit invisible to the per-channel test.
    """

    per_channel: "dict[int, MarkedGOFResult]"
    population: "object"  # nstat.PopulationTimeRescaleResult; quoted to avoid a hard import


def multivariate_gof_with_coupling(
    spike_bins_per_channel,
    p_k_per_channel,
    *,
    n_draws: int = 25,
    alpha: float = 0.05,
    n_tau_bins: int = 4,
    rng: np.random.Generator | None = None,
) -> CoupledMarkedGOFResult:
    r"""Discrete-time per-channel GoF *and* population coupling test, in one call.

    Per-channel rescaling (Gerhard-Haslinger-Pipa 2011) catches finite-bin
    bias in each channel's marginal fit but is blind to inter-channel
    coupling: a model can pass every per-channel KS and still get the
    joint distribution wrong.  The population marked-region
    :math:`\chi^2` of Tao, Weber, Arai & Eden (2018) — implemented by
    :func:`nstat.population_time_rescale` — closes that gap.  This
    wrapper runs both on the same data so a caller does not have to
    convert the inputs by hand.

    Conversion of the per-channel inputs to the population-test format:

    - ``counts_list[c] = np.bincount(spike_bins_per_channel[c],
      minlength=T)`` where :math:`T` is the length of ``p_k_per_channel[c]``.
    - ``lam_per_bin_list[c] = p_k_per_channel[c]``.  This treats the
      per-bin spike *probability* :math:`p_k = 1 - e^{-\lambda_k\Delta}`
      as the per-bin model-expected count :math:`\lambda_k\Delta`,
      which is exact in the small-:math:`p_k` regime that the
      discrete-time correction itself assumes.

    Parameters
    ----------
    spike_bins_per_channel
        Sequence of per-channel sorted event-bin index arrays.
    p_k_per_channel
        Sequence of per-channel per-bin spike-probability arrays; all
        arrays must have the same length :math:`T`.
    n_draws, alpha, rng
        Forwarded to :func:`multivariate_time_rescaling`.
    n_tau_bins
        Number of equal-:math:`\tau` cells per channel in the population
        :math:`\chi^2` partition (forwarded to
        :func:`nstat.population_time_rescale`).  ``> 1`` resolves
        within-channel timing and is what makes the test sensitive to
        coupling.  Default 4 is a reasonable coupling-sensitive choice
        for typical recordings; raise / lower to trade coupling
        sensitivity against the expected-count-per-cell requirement of
        the asymptotic :math:`\chi^2`.

    Returns
    -------
    CoupledMarkedGOFResult
        ``per_channel`` (dict) plus ``population``
        (:class:`nstat.PopulationTimeRescaleResult`).

    Raises
    ------
    ValueError
        If ``spike_bins_per_channel`` and ``p_k_per_channel`` have
        unequal length or if the per-channel ``p_k`` arrays disagree
        in length.

    See Also
    --------
    multivariate_time_rescaling : per-channel discrete-time test alone.
    nstat.population_time_rescale : population coupling test alone.

    References
    ----------
    Gerhard F, Haslinger R, Pipa G (2011), Neural Computation 23(6):1452.
    Tao L, Weber KM, Arai K, Eden UT (2018), J Comput Neurosci 45:147.
    """
    from nstat.fit import population_time_rescale

    spike_bins_per_channel = list(spike_bins_per_channel)
    p_k_per_channel = list(p_k_per_channel)
    if len(spike_bins_per_channel) != len(p_k_per_channel):
        raise ValueError(
            "spike_bins_per_channel and p_k_per_channel must have equal length"
        )
    if not p_k_per_channel:
        raise ValueError("need at least one channel")
    T = len(p_k_per_channel[0])
    for c, pk in enumerate(p_k_per_channel):
        if len(pk) != T:
            raise ValueError(
                f"p_k_per_channel[{c}] has length {len(pk)}, expected {T} "
                "(all channels must share the same time grid)"
            )

    per_channel = multivariate_time_rescaling(
        spike_bins_per_channel,
        p_k_per_channel,
        n_draws=n_draws,
        alpha=alpha,
        rng=rng,
    )

    counts_list = [
        np.bincount(np.asarray(sb, dtype=int), minlength=T).astype(float)
        for sb in spike_bins_per_channel
    ]
    lam_per_bin_list = [np.asarray(pk, dtype=float) for pk in p_k_per_channel]
    population = population_time_rescale(
        counts_list, lam_per_bin_list, n_tau_bins=int(n_tau_bins)
    )

    return CoupledMarkedGOFResult(per_channel=per_channel, population=population)


__all__ = [
    "MarkedGOFResult",
    "CoupledMarkedGOFResult",
    "RescaledACFResult",
    "uncorrected_rescaled",
    "corrected_rescaled",
    "marked_time_rescaling",
    "multivariate_time_rescaling",
    "multivariate_gof_with_coupling",
    "rescaled_acf",
]
