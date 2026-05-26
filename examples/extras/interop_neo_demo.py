"""Demo: nstat ↔ Neo round-trip and Neo Segment construction.

Demonstrates :mod:`nstat.extras.interop.neo`:

- Convert an :class:`nstat.nspikeTrain` to :class:`neo.core.SpikeTrain`.
- Round-trip back to :class:`nstat.nspikeTrain` (spike times, window, name preserved).
- Build a :class:`neo.core.Segment` from a :class:`nstat.SpikeTrainCollection`
  for downstream consumption by Elephant / SpikeInterface / Brian2.

Run::

    pip install nstat-toolbox[neo]
    python examples/extras/interop_neo_demo.py
"""
from __future__ import annotations

import numpy as np

from nstat import nspikeTrain, SpikeTrainCollection


def main() -> int:
    try:
        from nstat.extras.interop.neo import (
            to_neo_spiketrain,
            from_neo_spiketrain,
            to_neo_segment,
        )
    except ImportError as exc:
        print(f"Install required: {exc}")
        return 1

    # --- Build a small nstat spike train (3 spikes over 2 s window) -----
    nst = nspikeTrain(
        spikeTimes=[0.1, 0.5, 1.2],
        name="demo_unit",
        sampleRate=30_000.0,
        minTime=0.0,
        maxTime=2.0,
    )
    print(f"nstat input  : {len(nst.spikeTimes)} spikes, "
          f"window=[{nst.minTime:.2f}, {nst.maxTime:.2f}] s, name={nst.name!r}")

    # --- Convert to neo.SpikeTrain --------------------------------------
    neo_st = to_neo_spiketrain(nst)
    print(f"neo output   : {len(neo_st)} spikes, "
          f"t_start={float(neo_st.t_start):.2f}, t_stop={float(neo_st.t_stop):.2f}")

    # --- Round-trip back -------------------------------------------------
    nst2 = from_neo_spiketrain(neo_st)
    matches = np.allclose(nst.spikeTimes, nst2.spikeTimes)
    print(f"round-trip   : spike-times agree = {matches}")

    # --- Build a neo.Segment from a 3-train collection ------------------
    coll = SpikeTrainCollection(
        [
            nspikeTrain([0.10, 0.55, 1.30], minTime=0, maxTime=2, name="A"),
            nspikeTrain([0.12, 0.61, 1.34], minTime=0, maxTime=2, name="B"),
            nspikeTrain([0.15, 0.59, 1.40], minTime=0, maxTime=2, name="C"),
        ]
    )
    segment = to_neo_segment(coll)
    print(f"neo segment  : {len(segment.spiketrains)} spike trains "
          f"({[st.name for st in segment.spiketrains]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
