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

### EM trainers

| Symbol | MATLAB counterpart | Notes |
|---|---|---|
| `fit_linear_gaussian_em(observations, state_dim, *, n_iter=50, seed=0)` | `KF_EM` | LG state-space EM via Dynamax `LinearGaussianSSM.fit_em` (thin wrapper) → `LinearGaussianEMResult` |
| `fit_point_process_em(observations, state_dim, *, n_iter=30, n_newton_iter=5, seed=0)` | `PP_EM` | Poisson-LGSSM EM (CMGF E-step + closed-form/Newton M-step) → `PointProcessEMResult` |
| `fit_hybrid_em(poisson_observations, gaussian_observations, state_dim, *, n_iter=30, n_newton_iter=3, seed=0)` | `mPPCO_EM` | Mixed Poisson + Gaussian EM (IRLS-pseudo-obs augmented smoother E-step) → `HybridEMResult` |

### Point-process inference (known model)

| Symbol | MATLAB counterpart | Notes |
|---|---|---|
| `cmgf_poisson_filter(y, A, C, Q, x0, P0)` | `PPDecodeFilter` | CMGF (EKF-integration) point-process filter → `CMGFPoissonFilterResult` |
| `cmgf_poisson_smoother(y, A, C, Q, x0, P0)` | `PP_fixedIntervalSmoother` | CMGF forward-backward smoother → `CMGFPoissonFilterResult` |

### Result dataclasses

| Dataclass | Fields |
|---|---|
| `LinearGaussianEMResult` | `transition_matrix`, `observation_matrix`, `transition_covariance`, `observation_covariance`, `initial_state_mean`, `initial_state_covariance`, `log_likelihoods`, `n_iter` |
| `PointProcessEMResult` | `transition_matrix`, `observation_matrix`, `transition_covariance`, `initial_state_mean`, `initial_state_covariance`, `marginal_log_likelihoods`, `n_iter` |
| `HybridEMResult` | `transition_matrix`, `poisson_observation_matrix`, `gaussian_observation_matrix`, `transition_covariance`, `gaussian_observation_covariance`, `initial_state_mean`, `initial_state_covariance`, `marginal_log_likelihoods`, `n_iter` |
| `CMGFPoissonFilterResult` | `state_means`, `state_covariances`, `marginal_log_likelihood` |

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

## Scope

| Feature | Status |
|---|---|
| `fit_linear_gaussian_em` | shipped — KF_EM equivalent |
| `cmgf_poisson_filter` | shipped — point-process Kalman filter under Gaussian approximation |
| `cmgf_poisson_smoother` | shipped — point-process forward-backward smoother |
| `fit_point_process_em` | shipped — **PP_EM equivalent** (CMGF E-step + closed-form/Newton M-step, Smith & Brown 2003 PPLDS) |
| `fit_hybrid_em` | shipped — **mPPCO_EM equivalent** (IRLS-pseudo-obs augmented LG smoother E-step + closed-form / Newton M-step) |

### PP_EM and mPPCO_EM caveats

These are **first-pass** implementations following the algorithmic
description in Smith & Brown 2003 (PPLDS).  Known approximations:

- **Lag-one cross-covariance** in the M-step is approximated by the
  outer product of smoothed means (Dynamax CMGF doesn't expose lag-one
  covs).  Introduces ~1–3% bias in `Q` on stationary fixtures, larger
  on highly transient ones.
- **Laplace approximation** of `E[exp(C x_t)]` uses the diagonal
  quadratic correction.  Sufficient for moderate-rate processes; may
  underestimate variance at high rates.
- **mPPCO_EM uses fixed-R pseudo-observation covariance** (time-averaged
  `1/mean(λ)`).  Full time-varying R would require a custom Kalman
  smoother; deferred.
- **Marginal log-likelihood trace is not strictly monotonic** under
  the Laplace approximation (the approximation changes each iteration).
  Expect mostly-monotonic behavior with occasional small dips early
  in training; substantial decreases indicate a bug.

## CMGF Poisson recipe

For *inference* on a known model (filter or smoother), the bridge is a
thin wrapper around Dynamax's
:func:`conditional_moments_gaussian_filter` / :func:`smoother` for the
Poisson-LGSSM:

```python
import numpy as np
from nstat.extras.em.dynamax_bridge import (
    cmgf_poisson_filter, cmgf_poisson_smoother,
)

# Known model: x_t = A x_{t-1} + w_t,  y_t ~ Poisson(exp(C x_t))
A = np.eye(2) * 0.95
C = np.eye(2) * 0.3
Q = np.eye(2) * 0.05
x0 = np.zeros(2)
P0 = np.eye(2) * 0.1

# y is a (T, emission_dim) integer-valued spike-count array.
filtered = cmgf_poisson_filter(y, A, C, Q, x0, P0)
smoothed = cmgf_poisson_smoother(y, A, C, Q, x0, P0)

print(filtered.state_means.shape)      # (T, 2)
print(smoothed.marginal_log_likelihood)
```

Counterpart to MATLAB nSTAT's `PPDecodeFilter` / `PP_fixedIntervalSmoother`.

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
