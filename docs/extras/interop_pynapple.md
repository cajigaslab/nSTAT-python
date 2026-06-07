# `nstat.extras.interop.pynapple` — pynapple bridge

Convert between [`nstat.nspikeTrain`](../api.rst) /
[`nstat.SpikeTrainCollection`](../api.rst) and the
[pynapple](https://github.com/pynapple-org/pynapple) data model.
pynapple's `Ts` / `Tsd` / `TsdFrame` containers plus `IntervalSet`
epoch math are exactly the trial-window operations users repeatedly
hand-roll on top of nstat's `Trial`.

## Install

```bash
pip install nstat-toolbox[pynapple]
```

Pulls `pynapple>=0.7`.

## API

| Function | Direction | Notes |
|---|---|---|
| `to_pynapple_ts(nst)` | nstat → pynapple | Bare `Ts` — discards recording window |
| `to_pynapple_with_support(nst)` | nstat → pynapple | Returns `(Ts, IntervalSet)` preserving the window |
| `from_pynapple_ts(ts, *, name, sample_rate, support)` | pynapple → nstat | Requires `support=` for empty trains |
| `to_pynapple_tsgroup(collection)` | nstat → pynapple | Returns a `TsGroup` (per-neuron population) |

## Recipe

```python
import pynapple as nap
from nstat import nspikeTrain
from nstat.extras.interop.pynapple import (
    to_pynapple_with_support, from_pynapple_ts,
)

# nstat → pynapple with epoch math
nst = nspikeTrain(
    spikeTimes=[0.1, 0.5, 1.2, 1.8],
    minTime=0.0, maxTime=2.0, name="cell_7",
    sampleRate=30_000.0,
)
ts, support = to_pynapple_with_support(nst)

# Use pynapple's IntervalSet to restrict to [1, 2] s
sub = nap.IntervalSet(start=1.0, end=2.0)
ts_sub = ts.restrict(sub)

# Round-trip the restricted train back to nstat
nst_sub = from_pynapple_ts(ts_sub, name="cell_7_sub",
                           sample_rate=30_000.0, support=sub)
```

## Gotchas

- **Empty trains.** `from_pynapple_ts` with an empty `Ts` and no
  `support` raises `ValueError` — without an explicit support window,
  the bridge would silently produce `minTime=maxTime=0.0`, corrupting
  downstream rate / ISI computations. **Always pass `support=`** when
  converting populations where some neurons may be silent on a trial.
- **Sample rate.** pynapple doesn't track sample rate (it's a
  time-series library, not a hardware-acquisition wrapper). Pass
  `sample_rate=` explicitly to preserve it across the round-trip;
  default is 1000 Hz.
- **`TsGroup` support.** `to_pynapple_tsgroup` assumes all trains in
  the collection share a common time base (uses the first train's
  `[minTime, maxTime]` as the group support). Trains with divergent
  windows should be converted individually.

## End-to-end demo

[`examples/extras/interop_pynapple_demo.py`](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/extras/interop_pynapple_demo.py)
runs the convert → restrict → convert-back chain with a 50-spike
fixture.

## Upstream references

- pynapple: https://github.com/pynapple-org/pynapple
- License: MIT (GPL-2 compatible)
- Sibling library: [NeMoS](https://github.com/flatironinstitute/nemos)
  (same Flatiron group; pynapple-native GLM toolkit)
