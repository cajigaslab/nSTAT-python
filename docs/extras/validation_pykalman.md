# `nstat.extras.validation.pykalman_bridge` — pykalman Kalman cross-validation

Cross-validate
[`nstat.DecodingAlgorithms.kalman_filter`](../api.rst) and
`kalman_fixedIntervalSmoother` against
[pykalman](https://github.com/pykalman/pykalman) — a community-
resurrected BSD-3 pure-NumPy Kalman implementation. Because pykalman
has no JAX dependency, it's the *lowest-friction* cross-validation
reference for the Gaussian-only Kalman path.

This bridge directly addresses [AUDIT D3](https://github.com/cajigaslab/nSTAT-python/blob/main/AUDIT_REPORT.md)
(the known smoother-index approximation gap in
`kalman_fixedIntervalSmoother`) by providing an independent reference
implementation users can call from the same script that uses nstat's
Kalman primitives.

## Install

```bash
pip install nstat-toolbox[test-parity]   # bundles pykalman + nemos + statsmodels + nitime
```

Or `pip install pykalman>=0.11` alone.

## API

| Symbol | Notes |
|---|---|
| `cross_validate_kalman(observations, A, C, Q, R, x0, P0, *, compute_smoother=True)` | Fits both, returns `KalmanComparison` |
| `KalmanComparison` (dataclass) | `nstat_filtered_means`, `pykalman_filtered_means`, `nstat_smoothed_means`, `pykalman_smoothed_means`, `filtered_inf_norm`, `smoothed_inf_norm` |
| `KalmanComparison.assert_filtered_agree(atol=1e-2)` | Filter regression guard |
| `KalmanComparison.assert_smoothed_agree(atol=1.0)` | Smoother regression guard (intentionally very loose — see AUDIT D3 below) |

## Recipe

```python
import numpy as np
from nstat.extras.validation.pykalman_bridge import cross_validate_kalman

T, Dx, Dy = 100, 2, 2
A = np.eye(Dx) * 0.95
C = np.eye(Dy)
Q = np.eye(Dx) * 0.01
R = np.eye(Dy) * 0.1
x0 = np.zeros(Dx)
P0 = np.eye(Dx)

# Simulate a linear-Gaussian process
rng = np.random.default_rng(0)
x = np.zeros((T, Dx))
y = np.zeros((T, Dy))
x[0] = rng.multivariate_normal(x0, P0)
y[0] = C @ x[0] + rng.multivariate_normal(np.zeros(Dy), R)
for t in range(1, T):
    x[t] = A @ x[t - 1] + rng.multivariate_normal(np.zeros(Dx), Q)
    y[t] = C @ x[t] + rng.multivariate_normal(np.zeros(Dy), R)

cmp = cross_validate_kalman(y, A, C, Q, R, x0, P0)
cmp.assert_filtered_agree(atol=1e-2)   # filter agrees
# Smoother gap is documented & known — see below.
```

## Gotchas — the empirical baselines

The default tolerances on the `assert_*_agree` methods **document the
current state**, not a claim of exact agreement:

- **Filter: `atol=1e-2`.** Empirical baseline ~2.6e-3 on a 100×2 LG
  fixture, dominated by the t=0 initialization convention difference
  between nstat (prior at t=0) and pykalman (posterior at t=0).
  Tighten to `1e-8` once nstat's filter is patched.
- **Smoother: `atol=1.0`** (intentionally very loose). Empirical
  baseline ~0.4. nstat's `kalman_fixedIntervalSmoother` is an
  *augmented-state smoother* with a finite lag, while pykalman uses a
  full-window RTS smoother. The ~0.4 disagreement is **expected** —
  this is exactly what AUDIT_REPORT.md §3.1 calls out. Tighten to
  `1e-6` only after the AUDIT D3 closure work lands.

Use both `assert_*_agree` methods as **regression guards** (catch new
disagreements), not as claims of correctness.

## End-to-end demo

[`examples/extras/validation_pykalman_demo.py`](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/extras/validation_pykalman_demo.py)
runs the full filter + smoother comparison and prints the empirical
baselines.

## Upstream references

- pykalman: https://github.com/pykalman/pykalman
- License: BSD-3-Clause (GPL-2 compatible)
- Alternative (heavier, JAX-backed): [Dynamax](https://github.com/probml/dynamax)
  — planned `nstat.extras.em.dynamax` bridge for EM-trained variants
  (KF_EM / PP_EM / mPPCO_EM)
