# `nstat.extras.decoding.place_field_decoder` — Place-cell encoding + 2-D PPAF decoding

A one-call wrapper around the canonical 2-D place-cell encoding and
decoding pattern from
[`examples/paper/example08_real_place_cells.py`](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/paper/example08_real_place_cells.py).
The example08 script spells out the full pipeline — tensor-product
B-spline Poisson GLM (encoding), per-cell quadratic conditional
intensity refit, and `DecodingAlgorithms.PPDecodeFilterLinear` (the
history-free O(C·T) fast path) on the held-out trajectory.  This
wrapper packages that pipeline behind one function:
`fit_place_field_decoder(trial, position)`.

This is a pure-core module — no opt-deps beyond the numpy + scipy
stack that `nstat` already requires.

## Install

```bash
pip install nstat-toolbox
```

No extras group needed.  The wrapper imports
`nstat.extras.spatial.basis.bspline_basis_2d`,
`nstat.glm.fit_poisson_glm`, `nstat.glm.fit_binomial_glm`,
`nstat.CIF`, and `nstat.DecodingAlgorithms.PPDecodeFilter*` — all of
which ship in the core toolbox.

## API

| Symbol | Notes |
|---|---|
| `PlaceFieldDecoderConfig(bin_width_s=0.02, n_basis_per_dim=8, spline_order=4, cif_kind="quadratic", decode_filter="linear", min_n_spikes_per_cell=10)` | Frozen dataclass.  Validates field values in `__post_init__`. |
| `fit_place_field_decoder(trial, position, *, config=None)` | Fit the encoding + decoding pipeline on a single `Trial` plus `(n_time, 2)` position array aligned to the trial covariate grid.  Returns `PlaceFieldDecoderResult`. |
| `PlaceFieldDecoderResult` | Frozen dataclass holding `decoded_position` `(T, 2)`, `decoded_covariance` `(T, 2, 2)`, `decoding_error` `(T,)`, `mean_decoding_error` (float), `cell_indices_kept`, `cell_indices_skipped`, `spline_coefs`, `quadratic_coefs`, `n_basis_per_dim`, `bin_width_s`. |

### Choosing `decode_filter`

- `"linear"` (default) calls `PPDecodeFilterLinear` — the O(C·T)
  fast path documented in PR #198.  When `cif_kind="quadratic"`, the
  per-cell coefficients are Taylor-linearised at the trajectory mean
  so the filter consumes the linear `(mu, beta)` form.  Stable on
  smooth long-running trajectories where the trajectory mean is a
  good operating point; can diverge on short or atypically-shaped
  walks.
- `"nonlinear"` calls `PPDecodeFilter` (CIF-object branch) — slower
  but evaluates the full quadratic CIF analytically each step, so
  the decode is robust to off-mean excursions of the state.

### Choosing `cif_kind`

- `"quadratic"` matches example08: six coefficients per cell
  `[1, x, y, x^2, y^2, x*y]` — captures the asymmetric falloff of a
  real place field.
- `"linear"` is the lighter `[1, x, y]` form — sometimes cleaner for
  sparsely-spiking cells where the quadratic terms over-fit.

## Recipe

```python
import numpy as np
from nstat import Covariate, CovariateCollection, SpikeTrainCollection, Trial, nspikeTrain
from nstat.extras.decoding import PlaceFieldDecoderConfig, fit_place_field_decoder

# Build a synthetic single-cell trial — see the demo for the full
# multi-cell version with smooth OU position.
T, fs = 30.0, 50.0
t = np.arange(int(T * fs)) / fs
x_pos = 0.5 + 0.3 * np.sin(2 * np.pi * t / T)
y_pos = 0.5 + 0.3 * np.cos(2 * np.pi * t / T)

# 200 Poisson spikes concentrated near (0.25, 0.5).
rng = np.random.default_rng(0)
rates = 30.0 * np.exp(-((x_pos - 0.25) ** 2 + (y_pos - 0.5) ** 2) / (2 * 0.15 ** 2))
spike_count = rng.poisson(rates * (1.0 / fs))
spike_times = np.sort(np.concatenate([
    [t[k] + rng.uniform(0, 1.0 / fs)] for k in range(t.size) for _ in range(int(spike_count[k]))
])) if spike_count.sum() else np.array([])

trial = Trial(
    spike_collection=SpikeTrainCollection(
        [nspikeTrain(spike_times, minTime=0.0, maxTime=T)]
    ),
    covariate_collection=CovariateCollection([
        Covariate(t, x_pos, "x", "time", "s", "m", ["x"]),
        Covariate(t, y_pos, "y", "time", "s", "m", ["y"]),
    ]),
)
position = np.column_stack([
    np.asarray(trial.covarColl.getCov(0).data).reshape(-1),
    np.asarray(trial.covarColl.getCov(1).data).reshape(-1),
])

cfg = PlaceFieldDecoderConfig(decode_filter="nonlinear")
result = fit_place_field_decoder(trial, position, config=cfg)
print(result.decoded_position.shape, result.mean_decoding_error)
```

Live runnable demo (3-cell synthetic trial with the same pipeline):
[`examples/extras/decoding_place_field_demo.py`](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/extras/decoding_place_field_demo.py).

## Notes on the underlying pipeline

The unwrapped reference workflow lives in
[`examples/paper/example08_real_place_cells.py`](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/paper/example08_real_place_cells.py)
(lines 270-420).  Read it when you need to customise the basis,
expose the held-out spatial GoF diagnostics, or alter the
quadratic-CIF refit (this wrapper hard-codes the example08 choices).

## Scope

| Feature | Status |
|---|---|
| B-spline Poisson encoder per cell | shipped |
| Quadratic / linear CIF refit per cell | shipped |
| PPAF decoding via `PPDecodeFilterLinear` (fast path) | shipped |
| PPAF decoding via `PPDecodeFilter` (CIF branch) | shipped |
| Per-cell silent-cell skipping + UserWarning | shipped |
| Spatial goodness-of-fit (held-out g(r), rescaled ACF) | not in scope — see `nstat.extras.spatial` and example08 |
| Cross-validated bandwidth selection | not in scope |

## References

- Brown EN, Frank LM, Tang D, Quirk MC, Wilson MA (1998).
  *A statistical paradigm for neural spike train decoding applied to
  position prediction from ensemble firing patterns of rat
  hippocampal place cells.*  J Neurosci 18(18):7411.
- Eden UT, Frank LM, Barbieri R, Solo V, Brown EN (2004).
  *Dynamic analysis of neural encoding by point process adaptive
  filtering.*  Neural Comput 16(5):971.
- `nstat-python` PR #194 (Animal-1 demo / example08).
- `nstat-python` PR #198 (history-free O(C·T) fast path that this
  wrapper exercises via `decode_filter="linear"`).
