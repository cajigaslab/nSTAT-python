# `nstat.extras.decoding.clusterless_bridge` — Clusterless point-process decoding

Wraps
[`replay_trajectory_classification`](https://github.com/Eden-Kramer-Lab/replay_trajectory_classification)
(Denovellis et al. 2021, eLife; MIT) to bring **clusterless** marked
point-process state-space decoding and **trajectory-type classification**
into the `nstat.extras` namespace.

This closes the **Tier 2.1** gap on
[`parity/methods_roadmap.md`](https://github.com/cajigaslab/nSTAT-python/blob/main/parity/methods_roadmap.md)
— the modern descendant of nSTAT's exact PPAF / PPHF filters that adds:

1. **Clusterless observations** — the model consumes each spike's
   waveform features (the "marks") instead of requiring a spike-sorted
   unit, removing cluster-quality dependence (Kloosterman et al. 2014;
   Deng et al. 2015).
2. **Trajectory classification** — a discrete latent state
   (e.g. *local*, *forward replay*, *reverse replay*, *fragmented*)
   sits on top of the continuous position decode (Denovellis 2021).

## Install

```bash
pip install nstat-toolbox[clusterless]
```

Pulls `replay_trajectory_classification` and its JAX-based numerical
stack (~200 MB).  Like `[dynamax]`, this group is **not** rolled into
`[all-extras]`; install it explicitly when you need clusterless
decoding.

## API

### Single-state continuous decoder

| Symbol | Notes |
|---|---|
| `fit_clusterless_decoder(position, multiunits, *, place_bin_size=2.0, movement_var=None, is_training=None, is_compute_acausal=True)` | Wraps `ClusterlessDecoder` with sensible defaults (one `Environment`, one `RandomWalk` transition) → `ClusterlessDecoderResult` |

### Multi-state classifier

| Symbol | Notes |
|---|---|
| `fit_clusterless_classifier(position, multiunits, *, place_bin_size=2.0, state_names=None, discrete_diagonal=0.98, is_training=None, is_compute_acausal=True)` | Wraps `ClusterlessClassifier` with a default 2-state setup (`continuous` = random walk, `fragmented` = uniform) and a diagonal discrete transition → `ClusterlessClassifierResult` |

### Result dataclasses (plain NumPy — no xarray exposed)

| Dataclass | Fields |
|---|---|
| `ClusterlessDecoderResult` | `posterior` `(T, *position_bins)`, `map_position` `(T, n_position_dims)`, `position_bin_centers`, `causal_posterior` |
| `ClusterlessClassifierResult` | `posterior` `(T, n_states, *position_bins)`, `state_probabilities` `(T, n_states)`, `state_names`, `position_bin_centers` |

## Data conventions

Both entry points take **plain NumPy** inputs on a shared time grid:

- `position` — animal position, shape `(n_time, n_position_dims)`.
  1-D arrays are reshaped automatically.
- `multiunits` — mark cube, shape `(n_time, n_marks, n_electrodes)`.
  Use `NaN` for "no spike on this electrode at this time" (the
  upstream convention).

Results are plain NumPy — downstream code never needs xarray to
consume them.

## Recipe

```python
import numpy as np
from nstat.extras.decoding.clusterless_bridge import fit_clusterless_decoder

# Synthetic: a 1-D back-and-forth trajectory plus a sparse multiunit cube.
rng = np.random.default_rng(0)
n_time, n_marks, n_electrodes = 200, 4, 3
t = np.arange(n_time)
position = (50.0 + 45.0 * np.sin(2 * np.pi * t / n_time)).reshape(-1, 1)
multiunits = np.full((n_time, n_marks, n_electrodes), np.nan)
for t_i in np.flatnonzero(rng.random(n_time) < 0.3):
    for e in range(n_electrodes):
        if rng.random() < 0.5:
            multiunits[t_i, :, e] = rng.normal(loc=position[t_i, 0] / 20.0, size=n_marks)

result = fit_clusterless_decoder(position, multiunits, place_bin_size=5.0)
print(result.posterior.shape, result.map_position.shape)
```

Live runnable demo (decoder + classifier):
[`examples/extras/decoding_clusterless_demo.py`](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/extras/decoding_clusterless_demo.py).

## Scope

| Feature | Status |
|---|---|
| Wrap `ClusterlessDecoder` (single-state) | shipped |
| Wrap `ClusterlessClassifier` (multi-state) | shipped |
| Wrap sorted-spike variants (`SortedSpikes{Decoder,Classifier}`) | deferred — these duplicate `nstat.DecodingAlgorithms` (PPAF/PPHF) for sorted spikes, so the bridge focuses on the clusterless additions |
| Track-linearization helpers / GUI / movie utilities | not in scope |

The bridge is intentionally **thin**: data validation, sensible
defaults, plain-NumPy outputs.  All state-space inference, fitting, and
likelihood machinery is delegated to
`replay_trajectory_classification`.  When that library evolves, the
bridge's surface stays stable (the dataclass fields above).

## References

- Denovellis EL, Frank LM, Eden UT (2021).  *State space models for
  tracking neural representations of dynamic experimental variables.*
  eLife. https://elifesciences.org/articles/64505
- Kloosterman F, Layton SP, Chen Z, Wilson MA (2014).  *Bayesian
  decoding using unsorted spikes in the rat hippocampus.*
  J Neurophysiol. https://pmc.ncbi.nlm.nih.gov/articles/PMC4805376/
- Deng X, Liu DF, Karlsson MP, Frank LM, Eden UT (2015).  *Rapid
  classification of hippocampal replay content for real-time
  applications.*  J Neurophysiol.
