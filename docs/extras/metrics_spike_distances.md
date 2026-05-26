# `nstat.extras.metrics.spike_distances` — PySpike spike-train distances

Modern, parameter-free spike-train distance and synchrony metrics that
have no counterpart in MATLAB nSTAT. Thin wrappers around
[PySpike](https://github.com/mariomulansky/PySpike), the BSD-2 library
that implements the Kreuz / Mulansky family of time-resolved
spike-train distances with C/Cython acceleration.

Three metrics ship today:

- **ISI-distance** (Kreuz 2007) — instantaneous dissimilarity in
  inter-spike intervals.
- **SPIKE-distance** (Kreuz 2013) — instantaneous dissimilarity in
  spike timing. Bounded in [0, 1].
- **SPIKE-synchronization** (Kreuz 2015) — fraction of "synchronous"
  spikes; symmetric in [0, 1].

All three are **parameter-free** — no kernel bandwidth, no binning.
That's their main advantage over rate-based or kernel-density measures.

## Install

```bash
pip install nstat-toolbox[metrics]
```

Pulls `pyspike>=0.8`.

## API

| Function | Returns | Notes |
|---|---|---|
| `isi_distance(a, b)` | `float` | Kreuz 2007 |
| `spike_distance(a, b)` | `float ∈ [0, 1]` | Kreuz 2013 |
| `spike_synchronization(a, b)` | `float ∈ [0, 1]` | Kreuz 2015 |
| `pairwise_spike_distance_matrix(trains)` | `np.ndarray (N, N)` | Symmetric, zero diagonal |

## Recipe

```python
import numpy as np
from nstat import nspikeTrain
from nstat.extras.metrics.spike_distances import (
    isi_distance, spike_distance, spike_synchronization,
    pairwise_spike_distance_matrix,
)

# Two trains sharing a common 1 s window
a = nspikeTrain([0.10, 0.50, 0.90], minTime=0.0, maxTime=1.0)
b = nspikeTrain([0.15, 0.55, 0.95], minTime=0.0, maxTime=1.0)

d_isi   = isi_distance(a, b)              # ~0.04 (low — similar ISI structure)
d_spike = spike_distance(a, b)            # ~0.05 (low — similar timing)
s_sync  = spike_synchronization(a, b)     # ~1.0  (high — coincident)

# Population-level pairwise matrix
trains = [a, b, nspikeTrain([0.20, 0.60, 1.00], minTime=0, maxTime=1)]
D = pairwise_spike_distance_matrix(trains)
assert D.shape == (3, 3)
assert np.allclose(np.diag(D), 0.0)
assert np.allclose(D, D.T)
```

## Gotchas

- **Shared time window required.** All metrics assume the two (or N)
  trains share a common recording window. The bridge passes
  `nst.minTime` / `nst.maxTime` as PySpike's `edges=` — if your trains
  have divergent windows you should crop them first.
- **Empty trains.** Both functions handle empty trains without
  crashing (PySpike returns ISI-distance and SPIKE-distance of 0.0 or
  NaN depending on the case). Population-level matrices remain
  well-defined; check for NaN if you have all-zero rows.
- **No Victor-Purpura.** PySpike doesn't implement the Victor-Purpura
  metric (1996, which requires a cost parameter). If you need it, look
  at [`elephant.spike_train_dissimilarity`](https://elephant.readthedocs.io/)
  — but Elephant is Neo-typed end-to-end, so use the
  [`interop.neo` bridge](interop_neo.md) first.

## End-to-end demo

[`examples/extras/metrics_spike_distances_demo.py`](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/extras/metrics_spike_distances_demo.py)
generates a 5-train population with shared sinusoidal rate modulation,
then computes pairwise scalars and the full distance matrix.

## Upstream references

- PySpike: https://github.com/mariomulansky/PySpike
- License: BSD-2-Clause (GPL-2 compatible)
- Papers:
  [Kreuz 2007](https://doi.org/10.1016/j.jneumeth.2007.02.005) (ISI-distance),
  [Kreuz 2013](https://doi.org/10.1162/NECO_a_00407) (SPIKE-distance),
  [Kreuz 2015](https://doi.org/10.3389/fnsys.2015.00007) (synchronization)
