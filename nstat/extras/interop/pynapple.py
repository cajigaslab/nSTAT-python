"""pynapple ↔ nstat converters (Tier-B interop).

pynapple (https://github.com/pynapple-org/pynapple) is the modern
systems-neuroscience time-series library — its ``Ts`` / ``Tsd`` /
``TsdFrame`` containers plus ``IntervalSet`` epoch math are exactly the
kind of trial-window operations users repeatedly hand-roll on top of
nstat's :class:`Trial`.  This module exposes the conversion so users
can keep epoch math in pynapple and do MATLAB-style GLM / point-process
analysis in nstat.

Install:
    pip install nstat-toolbox[pynapple]
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nstat import nspikeTrain, SpikeTrainCollection

if TYPE_CHECKING:
    import pynapple as nap


_IMPORT_ERROR_MSG = (
    "nstat.extras.interop.pynapple requires the 'pynapple' package. "
    "Install with: pip install nstat-toolbox[pynapple]"
)


def _require_pynapple() -> "type[nap.Ts]":
    try:
        import pynapple as nap
    except ImportError as e:
        raise ImportError(_IMPORT_ERROR_MSG) from e
    return nap.Ts


def to_pynapple_ts(nst: nspikeTrain) -> "nap.Ts":
    """Convert an :class:`nstat.nspikeTrain` to a :class:`pynapple.Ts`.

    The resulting ``Ts`` carries timestamps in seconds (pynapple's
    canonical unit, matching nstat's convention).  The pynapple ``Ts``
    has no notion of ``minTime`` / ``maxTime`` outside its own
    timestamps, so use :func:`to_pynapple_with_support` if you need to
    preserve nstat's recording-window bounds.

    Parameters
    ----------
    nst
        nstat spike train.

    Returns
    -------
    pynapple.Ts
    """
    _require_pynapple()
    import pynapple as nap

    return nap.Ts(t=np.asarray(nst.spikeTimes, dtype=float), time_units="s")


def to_pynapple_with_support(
    nst: nspikeTrain,
) -> "tuple[nap.Ts, nap.IntervalSet]":
    """Convert an :class:`nstat.nspikeTrain` plus its recording window.

    Returns a ``(Ts, IntervalSet)`` pair so the downstream pynapple
    workflow knows the full recording window — important for tuning
    curves, perievent histograms, and rate calculations that need to
    know "how much time was observed" (not just when spikes happened).

    Returns
    -------
    ts : pynapple.Ts
    support : pynapple.IntervalSet
        Single-interval set ``[nst.minTime, nst.maxTime]``.
    """
    _require_pynapple()
    import pynapple as nap

    ts = nap.Ts(t=np.asarray(nst.spikeTimes, dtype=float), time_units="s")
    support = nap.IntervalSet(start=float(nst.minTime), end=float(nst.maxTime))
    return ts, support


def from_pynapple_ts(
    ts: "nap.Ts",
    *,
    name: str = "",
    sample_rate: float = 1000.0,
    support: "nap.IntervalSet | None" = None,
) -> nspikeTrain:
    """Convert a :class:`pynapple.Ts` to an :class:`nstat.nspikeTrain`.

    Parameters
    ----------
    ts
        pynapple timestamps (seconds).
    name
        nstat spike-train name.
    sample_rate
        Recording sample rate in Hz.  Default 1000 Hz to match nstat's
        default; override to match your acquisition.
    support
        Optional pynapple ``IntervalSet`` providing the recording window.
        If provided, its ``[start, end]`` is used for ``minTime`` /
        ``maxTime``.  If absent, the window defaults to the
        ``[min, max]`` of the timestamps (no padding).

    Returns
    -------
    nspikeTrain
    """
    _require_pynapple()

    times_s = np.asarray(ts.times(), dtype=float)
    if support is not None:
        min_t = float(np.asarray(support.start).min())
        max_t = float(np.asarray(support.end).max())
    elif times_s.size:
        min_t, max_t = float(times_s.min()), float(times_s.max())
    else:
        raise ValueError(
            "Cannot infer recording window from an empty pynapple Ts. "
            "Pass support=<pynapple.IntervalSet> so the nspikeTrain knows "
            "its observation window (otherwise downstream rate / ISI "
            "computations silently corrupt)."
        )

    return nspikeTrain(
        spikeTimes=times_s,
        name=name,
        sampleRate=sample_rate,
        minTime=min_t,
        maxTime=max_t,
    )


def to_pynapple_tsgroup(
    spike_collection: SpikeTrainCollection,
) -> "nap.TsGroup":
    """Convert a :class:`nstat.SpikeTrainCollection` to a :class:`pynapple.TsGroup`.

    A pynapple ``TsGroup`` is the natural container for a per-neuron
    population (the same role nstat's ``SpikeTrainCollection`` plays).
    """
    _require_pynapple()
    import pynapple as nap

    trains = list(spike_collection)
    ts_dict = {
        i: nap.Ts(t=np.asarray(tr.spikeTimes, dtype=float), time_units="s")
        for i, tr in enumerate(trains)
    }
    # Use the first train's support as the group support (collections in
    # nstat are constructed from trains that share a common time base).
    if trains:
        support = nap.IntervalSet(
            start=float(trains[0].minTime), end=float(trains[0].maxTime)
        )
        return nap.TsGroup(ts_dict, time_support=support)
    return nap.TsGroup(ts_dict)


__all__ = [
    "to_pynapple_ts",
    "to_pynapple_with_support",
    "from_pynapple_ts",
    "to_pynapple_tsgroup",
]
