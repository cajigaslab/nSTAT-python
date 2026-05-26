"""Neo ↔ nstat converters (Tier-B interop, scope: spike trains).

Neo (https://github.com/NeuralEnsemble/python-neo) is the de-facto
interchange object for the Elephant / SpikeInterface / Brian2 ecosystem,
with readers for Spike2 / NEX / AlphaOmega / Axon / Blackrock / Plexon /
TDT formats.  Adding these converters lets nstat consume data from any
of those file formats without nstat owning a reader.

The bridge is intentionally narrow: we convert spike-train objects only.
Analog signals, events, and segments are left for the user to manage
on the Neo side — nstat's :class:`Trial` doesn't carry the same
hierarchical metadata that Neo's :class:`neo.core.Block` /
:class:`neo.core.Segment` do.

Install:
    pip install nstat-toolbox[neo]
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nstat import nspikeTrain, SpikeTrainCollection
from nstat.extras._lazy import require_optionals

if TYPE_CHECKING:
    import neo


def _require_neo() -> "type[neo.SpikeTrain]":
    """Lazy-import neo + quantities, raising a clear error if either is absent."""
    neo_mod, _pq = require_optionals("neo", "quantities", install_key="neo")
    return neo_mod.core.SpikeTrain


def to_neo_spiketrain(nst: nspikeTrain) -> "neo.SpikeTrain":
    """Convert an :class:`nstat.nspikeTrain` to a :class:`neo.SpikeTrain`.

    Parameters
    ----------
    nst
        nstat spike train.  ``nst.spikeTimes`` is interpreted in seconds
        (the nstat-wide convention; see CLAUDE.md "Time and units").

    Returns
    -------
    neo.SpikeTrain
        Neo spike train with ``t_start`` = ``nst.minTime`` s,
        ``t_stop`` = ``nst.maxTime`` s, and the original name and
        sampling rate preserved in ``annotations``.

    Notes
    -----
    Neo requires explicit ``pq.Quantity`` units; this converter assumes
    seconds throughout (matching nstat's convention).
    """
    NeoSpikeTrain = _require_neo()
    import quantities as pq

    return NeoSpikeTrain(
        times=np.asarray(nst.spikeTimes, dtype=float) * pq.s,
        t_start=float(nst.minTime) * pq.s,
        t_stop=float(nst.maxTime) * pq.s,
        name=nst.name or None,
        sampling_rate=float(nst.sampleRate) * pq.Hz,
    )


def from_neo_spiketrain(
    neo_spiketrain: "neo.SpikeTrain",
    *,
    name: str | None = None,
    sample_rate: float | None = None,
) -> nspikeTrain:
    """Convert a :class:`neo.SpikeTrain` to an :class:`nstat.nspikeTrain`.

    Parameters
    ----------
    neo_spiketrain
        Source Neo spike train.  Must carry time units convertible to
        seconds (Neo's default).
    name
        Override the train name.  If ``None``, uses ``neo_spiketrain.name``.
    sample_rate
        Override the sampling rate (Hz).  If ``None``, uses
        ``neo_spiketrain.sampling_rate`` converted to Hz, or 1000.0 if
        the source carries no sampling rate.

    Returns
    -------
    nspikeTrain
    """
    _require_neo()  # validates neo is installed before reading attributes
    import quantities as pq

    times_s = np.asarray(neo_spiketrain.rescale(pq.s).magnitude, dtype=float)
    t_start = float(neo_spiketrain.t_start.rescale(pq.s).magnitude)
    t_stop = float(neo_spiketrain.t_stop.rescale(pq.s).magnitude)

    if sample_rate is None:
        sr_attr = getattr(neo_spiketrain, "sampling_rate", None)
        sample_rate = float(sr_attr.rescale(pq.Hz).magnitude) if sr_attr is not None else 1000.0

    return nspikeTrain(
        spikeTimes=times_s,
        name=name if name is not None else (neo_spiketrain.name or ""),
        sampleRate=sample_rate,
        minTime=t_start,
        maxTime=t_stop,
    )


def to_neo_segment(spike_collection: SpikeTrainCollection) -> "neo.Segment":
    """Convert a :class:`nstat.SpikeTrainCollection` to a :class:`neo.Segment`.

    A Neo Segment groups spike trains that share a common time base —
    the natural mapping for nstat's per-trial spike collections.
    """
    _require_neo()
    from neo.core import Segment as NeoSegment

    segment = NeoSegment(name=getattr(spike_collection, "name", None))
    for train in list(spike_collection):
        segment.spiketrains.append(to_neo_spiketrain(train))
    return segment


__all__ = ["to_neo_spiketrain", "from_neo_spiketrain", "to_neo_segment"]
