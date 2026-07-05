"""Demo: bug-resistant clinical epoch bookkeeping via nstat <-> pynapple.

Real recordings are rarely one clean, analysis-ready segment. An iEEG
session mixes interictal (seizure-free) stretches with ictal/post-ictal
stretches that a baseline-rate analysis must exclude; a BCI session is a
stream of individual go-cue-to-target trial windows cut out of one
continuous acquisition. Restricting analysis to only the clinically
relevant epochs -- interictal-only windows, or a single BCI-trial window
-- is exactly the kind of bookkeeping that is easy to get subtly wrong by
hand: spikes, LFP, and behavior often stream at different sample rates,
so ad hoc index arithmetic (``np.searchsorted`` applied to one array but
not another, off-by-one boundary slicing) is a common source of silent
alignment bugs that corrupt downstream statistics without raising an
error.

:mod:`pynapple`'s :class:`~pynapple.IntervalSet` epoch algebra (Viejo
et al. 2023) exists to make this bookkeeping robust: epochs are
first-class objects, and ``.restrict()`` clips any time series to them
consistently, so one epoch definition governs spikes, LFP, and behavior
alike -- no hand-rolled index math to get wrong. This demo exercises
:mod:`nstat.extras.interop.pynapple` as the bridge that lets an
:class:`nstat.nspikeTrain` participate in that epoch algebra:

- Convert :class:`nstat.nspikeTrain` to :class:`pynapple.Ts` plus its
  recording-window :class:`pynapple.IntervalSet`.
- Restrict to a sub-epoch -- standing in for an interictal-only window
  in an iEEG recording, or a single BCI-trial window -- using
  pynapple's ``IntervalSet.restrict`` epoch math.
- Convert back to :class:`nstat.nspikeTrain` with the sub-epoch as
  support, and confirm restrict-then-convert agrees with
  convert-then-restrict.

Reference: Viejo G et al. (2023). Pynapple, a toolbox for data analysis
in neuroscience. eLife 12:e85786.

Run::

    pip install nstat-toolbox[pynapple]
    python examples/extras/interop_pynapple_demo.py
"""
from __future__ import annotations

import numpy as np

from nstat import nspikeTrain


def main() -> int:
    try:
        import pynapple as nap
        from nstat.extras.interop.pynapple import (
            to_pynapple_with_support,
            from_pynapple_ts,
        )
    except ImportError as exc:
        print(f"Install required: {exc}")
        return 1

    # --- Build an nstat spike train over a 10 s recording ----------------
    # Stand-in for one channel/unit's spikes across a longer session that
    # a clinician or BCI pipeline will later carve into epochs.
    rng = np.random.default_rng(42)
    spikes = np.sort(rng.uniform(0.0, 10.0, size=50))
    nst = nspikeTrain(
        spikeTimes=spikes,
        name="demo",
        sampleRate=30_000.0,
        minTime=0.0,
        maxTime=10.0,
    )
    print(f"nstat input  : {len(nst.spikeTimes)} spikes over "
          f"[{nst.minTime}, {nst.maxTime}] s")

    # --- Convert to pynapple, then restrict to the clinical epoch -------
    # [2, 5] s stands in for the epoch that matters clinically -- e.g. an
    # interictal-only window in an iEEG recording, or a single BCI-trial
    # window -- carved out with IntervalSet.restrict rather than by
    # hand-rolled index arithmetic on the spike-time array.
    ts, support = to_pynapple_with_support(nst)
    sub_window = nap.IntervalSet(start=2.0, end=5.0)
    ts_sub = ts.restrict(sub_window)
    print(f"pynapple sub : {len(ts_sub)} spikes inside [2, 5] s window "
          f"(via IntervalSet.restrict)")

    # --- Round-trip the epoch back to an nstat train ---------------------
    nst_sub = from_pynapple_ts(ts_sub, name="demo_sub",
                                sample_rate=30_000.0, support=sub_window)
    print(f"nstat sub    : {len(nst_sub.spikeTimes)} spikes, "
          f"window=[{nst_sub.minTime}, {nst_sub.maxTime}] s")

    # Bookkeeping check: restricting before vs. after the nstat<->pynapple
    # conversion must agree -- exactly the class of silent alignment bug
    # that hand-rolled index arithmetic risks introducing.
    inside = (spikes >= 2.0) & (spikes <= 5.0)
    matches = len(nst_sub.spikeTimes) == int(inside.sum())
    print(f"agreement    : nstat restrict-then-convert matches "
          f"pynapple convert-then-restrict = {matches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
