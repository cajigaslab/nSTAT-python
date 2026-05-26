# `nstat.extras.em.dynamax_bridge` — EM-trained state-space models via Dynamax

Wraps [Dynamax](https://github.com/probml/dynamax) (JAX-based,
MIT-licensed) to provide EM-trained linear-Gaussian state-space models
without nstat having to re-implement the EM machinery.

This is the **foundation** for closing the unported MATLAB nSTAT
`KF_EM` / `PP_EM` / `mPPCO_EM` families documented in
[`AUDIT_REPORT.md`](https://github.com/cajigaslab/nSTAT-python/blob/main/AUDIT_REPORT.md)
§3.2 (19 methods, ~7,500 LOC of MATLAB if ported verbatim).

## Install

```bash
pip install nstat-toolbox[dynamax]   # pulls Dynamax (~50 MB) + JAX (~200 MB)
```

## API

| Symbol | Notes |
|---|---|
| `fit_linear_gaussian_em(observations, state_dim, *, n_iter=50, seed=0)` | Fit a discrete-time LG state-space model via Dynamax EM |
| `LinearGaussianEMResult` (dataclass) | `transition_matrix`, `observation_matrix`, `transition_covariance`, `observation_covariance`, `initial_state_mean`, `initial_state_covariance`, `log_likelihoods`, `n_iter` |

All result arrays are plain NumPy — callers don't need to know about JAX or pytrees.

## Recipe

```python
import numpy as np
from nstat.extras.em.dynamax_bridge import fit_linear_gaussian_em

# Simulate a 2-state linear-Gaussian process
rng = np.random.default_rng(0)
T, state_dim, emission_dim = 300, 2, 2
A_true = np.eye(state_dim) * 0.9
Q_true = np.eye(state_dim) * 0.02
R_true = np.eye(emission_dim) * 0.1
x = np.zeros((T, state_dim))
y = np.zeros((T, emission_dim))
x[0] = rng.multivariate_normal(np.zeros(state_dim), np.eye(state_dim))
y[0] = x[0] + rng.multivariate_normal(np.zeros(emission_dim), R_true)
for t in range(1, T):
    x[t] = A_true @ x[t - 1] + rng.multivariate_normal(np.zeros(state_dim), Q_true)
    y[t] = x[t] + rng.multivariate_normal(np.zeros(emission_dim), R_true)

# Fit via EM
result = fit_linear_gaussian_em(y, state_dim=2, n_iter=30)
print(f"Final log-likelihood: {result.log_likelihoods[-1]:.2f}")
print(f"Learned Â:\n{result.transition_matrix}")
```

## Scope (initial release)

| Feature | Status |
|---|---|
| `fit_linear_gaussian_em` | shipped — KF_EM equivalent |
| `fit_point_process_em` (PP_EM equivalent) | deferred — needs Dynamax `PoissonHMM` bridge |
| `fit_hybrid_em` (mPPCO_EM equivalent) | deferred — needs Dynamax `GeneralizedGaussianSSM` bridge |

## Gotchas

- **JAX install footprint** is ~200 MB. The `[dynamax]` extra is
  intentionally not in `[all-extras]` until a CI-functional run
  validates Dynamax compatibility with the rest of the test matrix.
- **Pytree → NumPy conversion.** Dynamax represents parameters as
  nested JAX pytrees (`ParamsLGSSMDynamics`, `ParamsLGSSMEmissions`,
  `ParamsLGSSMInitial`). The bridge unpacks these into plain NumPy
  arrays so users don't need to learn the pytree convention.
- **EM monotonicity.** EM log-likelihood is theoretically non-decreasing.
  The bridge tests assert this with a tolerance of `-1e-6` to absorb
  floating-point noise; if Dynamax's optimizer produces a decrement
  larger than that, the test will surface it.

## End-to-end demo

[`examples/extras/em_dynamax_demo.py`](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/extras/em_dynamax_demo.py)
fits a 2-state LG model on 300 samples and prints the parameter
estimates + EM log-likelihood trace.

## Upstream references

- Dynamax: https://github.com/probml/dynamax
- License: MIT (GPL-2 compatible)
- Active development (v1.0, 1,688 commits as of audit)
- Maintained by: Kevin Murphy's probml group
