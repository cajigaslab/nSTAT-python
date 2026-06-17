# `nstat.extras.latents.gpfa_bridge`

Gaussian-Process Factor Analysis (Yu et al. 2009, *J. Neurophysiol.*
102(1)) for inferring smooth low-dimensional latent trajectories from
simultaneous spike trains.  Bridges
[`elephant.gpfa.GPFA`](https://elephant.readthedocs.io/en/latest/reference/gpfa.html);
pure Python-only extension, no MATLAB nSTAT counterpart.

## Install

```bash
pip install nstat-toolbox[latents]
```

Pulls Elephant (>=1.2) plus Neo and quantities (~50 MB combined).

## API

| Symbol | Notes |
|---|---|
| `GPFAConfig(x_dim, bin_size_s=0.02, em_max_iter=500, em_tol=1e-8, min_var_frac=0.01)` | Fit configuration; frozen dataclass with `__post_init__` validation. |
| `fit_gpfa(spike_trains, *, config=None, seed=None) -> GPFAResult` | Main entry point.  Accepts `list[Trial]` or `list[list[neo.SpikeTrain]]`; requires >=2 trials. |
| `GPFAResult` | Frozen dataclass: `latent_trajectories: list[(n_bins, x_dim) ndarray]`, `x_dim`, `bin_size_s`, `n_trials`, `log_likelihood`, `elephant_model`. |

## Worked example

[see examples/extras/latents_gpfa_demo.py](../../examples/extras/latents_gpfa_demo.py)

## Notes

- **Multi-trial requirement.** GPFA's EM covariance estimation is
  degenerate on a single trial; the bridge raises `ValueError` if
  `len(spike_trains) < 2`.
- **Axis order.** Latent trajectories are returned as `(n_bins, x_dim)`
  — nstat's `(time, feature)` convention.  Elephant's native order is
  `(x_dim, n_bins)`, transposed inside the bridge.
- **Log-likelihood reporting.** Elephant evaluates the log-likelihood
  only every `freq_ll=5` iterations; intermediate trace entries are
  `NaN`.  `GPFAResult.log_likelihood` returns the last finite entry, or
  `None` if elephant produced no finite value.
- **Seed semantics.** Elephant's GPFA uses module-level numpy random.
  Passing `seed` to `fit_gpfa` temporarily sets the legacy seed and
  restores the previous state on exit, so the caller's RNG context is
  preserved.  This is the sole legacy-`np.random.seed` use in nstat —
  the `default_rng`-only convention applies everywhere else.

## References

- Yu BM, Cunningham JP, Santhanam G, Ryu SI, Shenoy KV, Sahani M.
  (2009). *Gaussian-process factor analysis for low-dimensional
  single-trial analysis of neural population activity.*
  J. Neurophysiol. 102(1): 614–635.
