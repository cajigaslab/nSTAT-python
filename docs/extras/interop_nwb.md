# `nstat.extras.interop.nwb` — NWB:N reader

Read [NWB](https://github.com/NeurodataWithoutBorders/pynwb) files into
[`nstat.SpikeTrainCollection`](../api.html). NWB:N is the
BRAIN-Initiative-standard format for neurophysiology data; this bridge
lets analyses written for nstat run directly on NWB-formatted datasets.

## Install

```bash
pip install nstat-toolbox[nwb]
```

Pulls `pynwb>=2.8` (and `hdmf` transitively).

## API

| Function | Notes |
|---|---|
| `nwb_units_to_collection(nwbfile, *, sample_rate, name_prefix, time_window)` | Reads the `units` table from an in-memory `NWBFile` |
| `read_nwb_path(path, *, sample_rate, name_prefix)` | File-path convenience wrapper |

### Recording-window resolution order

For each unit, `minTime` / `maxTime` are resolved most-specific to least-specific:

1. **Explicit** `time_window=(t0, t1)` parameter — applied uniformly to every unit.
2. **Per-unit** `obs_intervals` column (NWB-standard for "intervals during which the unit was observed") — uses the outer envelope.
3. **Fallback**: per-unit `[min(spike_times), max(spike_times)]` — emits a `UserWarning` because this **silently understates** the observation window for sparsely-firing units, breaking PSTH / rate denominators.

## Recipe

```python
from pynwb import NWBHDF5IO
from nstat.extras.interop.nwb import nwb_units_to_collection, read_nwb_path

# Path-based convenience
coll = read_nwb_path("/path/to/session.nwb", sample_rate=30_000.0)

# Or open the file yourself for richer control
with NWBHDF5IO("/path/to/session.nwb", mode="r") as io:
    nwb = io.read()
    coll = nwb_units_to_collection(
        nwb,
        sample_rate=30_000.0,
        # Override per-unit obs_intervals — applied uniformly:
        time_window=(0.0, 600.0),
    )

print(f"Loaded {len(list(coll))} units")
```

## Gotchas

- **Window discipline.** If you ingest an NWB file without
  `obs_intervals` and without `time_window=`, you get a `UserWarning`
  on the first sparse unit. **Treat this as an error in production**
  — silent understatement of denominators is the #1 source of
  rate-comparison bugs across trials.
- **Sample rate.** NWB stores spike times as seconds, not raw samples,
  so `sample_rate=` is metadata-only for nstat's primitives — it does
  not affect the spike times themselves. Defaults to 30 kHz.
- **Writer omitted.** No nstat → NWB writer ships today. A faithful
  NWB file needs mandatory metadata (subject, session_description,
  experimenter, …) that nstat doesn't track. Construct NWB files via
  [`pynwb`](https://pynwb.readthedocs.io/) directly.

## End-to-end demo

[`examples/extras/interop_nwb_demo.py`](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/extras/interop_nwb_demo.py)
builds an in-memory NWB file with `obs_intervals`, then demonstrates
both the implicit-window path and the explicit `time_window=` override.

## Upstream references

- pynwb: https://github.com/NeurodataWithoutBorders/pynwb
- License: BSD-3-Clause (GPL-2 compatible)
- Maintained by: LBNL / Allen Institute / NIH
