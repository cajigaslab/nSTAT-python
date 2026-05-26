"""Demo: nstat ↔ pynapple round-trip and IntervalSet epoch math.

Demonstrates :mod:`nstat.extras.interop.pynapple`:

- Convert :class:`nstat.nspikeTrain` to :class:`pynapple.Ts` plus its
  recording-window :class:`pynapple.IntervalSet`.
- Restrict the train to a sub-epoch using pynapple's epoch math.
- Convert back to :class:`nstat.nspikeTrain` with the sub-epoch as support.

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

    # --- Build an nstat spike train over a 10 s window ------------------
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

    # --- Convert to pynapple, then restrict to [2, 5] s -----------------
    ts, support = to_pynapple_with_support(nst)
    sub_window = nap.IntervalSet(start=2.0, end=5.0)
    ts_sub = ts.restrict(sub_window)
    print(f"pynapple sub : {len(ts_sub)} spikes inside [2, 5] s window "
          f"(via IntervalSet.restrict)")

    # --- Round-trip the sub-epoch back to an nstat train ---------------
    nst_sub = from_pynapple_ts(ts_sub, name="demo_sub",
                                sample_rate=30_000.0, support=sub_window)
    print(f"nstat sub    : {len(nst_sub.spikeTimes)} spikes, "
          f"window=[{nst_sub.minTime}, {nst_sub.maxTime}] s")

    inside = (spikes >= 2.0) & (spikes <= 5.0)
    matches = len(nst_sub.spikeTimes) == int(inside.sum())
    print(f"agreement    : nstat restrict-then-convert matches "
          f"pynapple convert-then-restrict = {matches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
