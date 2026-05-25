# `nstat.extras.interop.neo` — Neo bridge

Convert between [`nstat.nspikeTrain`](../api.html) /
[`nstat.SpikeTrainCollection`](../api.html) and the
[Neo](https://github.com/NeuralEnsemble/python-neo) data model. Neo is
the de-facto interchange object for the Elephant / SpikeInterface /
Brian2 ecosystem, with readers for Spike2, NEX, AlphaOmega, Axon,
Blackrock, Plexon, and TDT formats.

## Install

```bash
pip install nstat-toolbox[neo]
```

Pulls `neo>=0.13` + `quantities>=0.15`.

## API

| Function | Direction | Notes |
|---|---|---|
| `to_neo_spiketrain(nst)` | nstat → Neo | Spike times in seconds; preserves `name`, `sampleRate` |
| `from_neo_spiketrain(neo_st, *, name=None, sample_rate=None)` | Neo → nstat | Auto-converts units via `quantities.Quantity.rescale` |
| `to_neo_segment(collection)` | nstat → Neo | Builds a `neo.Segment` from a `SpikeTrainCollection` |

## Recipe

```python
from nstat import nspikeTrain, SpikeTrainCollection
from nstat.extras.interop.neo import (
    to_neo_spiketrain, from_neo_spiketrain, to_neo_segment,
)

# nstat → Neo
nst = nspikeTrain(
    spikeTimes=[0.1, 0.5, 1.2],
    name="unit_42",
    sampleRate=30_000.0,
    minTime=0.0,
    maxTime=2.0,
)
neo_st = to_neo_spiketrain(nst)

# Round-trip back (units preserved)
nst_back = from_neo_spiketrain(neo_st)
assert (nst_back.spikeTimes == nst.spikeTimes).all()

# Collection → Neo Segment (one row per train)
coll = SpikeTrainCollection([nst, nst, nst])
segment = to_neo_segment(coll)
assert len(segment.spiketrains) == 3
```

## Gotchas

- **Units.** Neo requires `quantities.Quantity` units. The bridge
  assumes seconds throughout (matching nstat's `spikeTimes` convention
  — see [CLAUDE.md "Time and units"](https://github.com/cajigaslab/nSTAT-python/blob/main/CLAUDE.md)).
  If you pass a Neo `SpikeTrain` whose times are in `ms`, the
  `rescale(pq.s)` conversion is automatic.
- **Window.** `t_start` / `t_stop` map to `minTime` / `maxTime`.  If
  the source Neo train doesn't carry `sampling_rate`, the round-trip
  defaults to 1000 Hz — pass `sample_rate=` explicitly to override.
- **Segment scope.** `to_neo_segment` only populates `segment.spiketrains` —
  not analog signals or events. nstat's `SpikeTrainCollection` doesn't
  carry the hierarchical metadata Neo's `Block` / `Segment` track.

## End-to-end demo

[`examples/extras/interop_neo_demo.py`](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/extras/interop_neo_demo.py)
runs the full round-trip + Segment construction with `python examples/extras/interop_neo_demo.py`.

## Upstream references

- Neo: https://github.com/NeuralEnsemble/python-neo
- License: BSD-3-Clause (GPL-2 compatible)
- Used downstream by: [Elephant](https://github.com/NeuralEnsemble/elephant),
  [SpikeInterface](https://github.com/SpikeInterface/spikeinterface),
  [Brian2](https://github.com/brian-team/brian2)
