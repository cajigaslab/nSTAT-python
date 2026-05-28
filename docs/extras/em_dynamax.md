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

### Held-out predictive log-likelihood (quality diagnostic)

| Symbol | Notes |
|---|---|
| `point_process_predictive_ll(y, A, C, Q, x0, P0, *, n_quad=15)` | True one-step-ahead predictive log-likelihood of a Poisson-LGSSM → `PredictiveLogLik` |
| `hybrid_predictive_ll(yp, yg, A, C_p, C_g, Q, R, x0, P0, *, n_quad=15)` | Same for the hybrid model; `total = poisson + gaussian` |

These are **pure NumPy** and do **not** require dynamax — they score the
*true* Poisson (and Gaussian) likelihood of observations under the
one-step-ahead predictive state, the right metric for convergence
checks, model/restart comparison, and held-out scoring (pass a test
segment + train-fitted parameters).  They replace the surrogate
Gaussian-smoother `marginal_log_likelihoods` trace, which is **not** a
valid objective (it re-linearizes each iteration).

### Result dataclasses

| Dataclass | Fields |
|---|---|
| `LinearGaussianEMResult` | `transition_matrix`, `observation_matrix`, `transition_covariance`, `observation_covariance`, `initial_state_mean`, `initial_state_covariance`, `log_likelihoods`, `n_iter` |
| `PointProcessEMResult` | `transition_matrix`, `observation_matrix`, `transition_covariance`, `initial_state_mean`, `initial_state_covariance`, `marginal_log_likelihoods`, `n_iter` |
| `HybridEMResult` | `transition_matrix`, `poisson_observation_matrix`, `gaussian_observation_matrix`, `transition_covariance`, `gaussian_observation_covariance`, `initial_state_mean`, `initial_state_covariance`, `marginal_log_likelihoods`, `n_iter` |
| `CMGFPoissonFilterResult` | `state_means`, `state_covariances`, `marginal_log_likelihood` |
| `PredictiveLogLik` | `total`, `per_timestep`, `poisson`, `gaussian` |

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
| `point_process_predictive_ll` / `hybrid_predictive_ll` | shipped — true one-step-ahead held-out predictive log-likelihood (pure NumPy, no dynamax) |

### PP_EM and mPPCO_EM — experimental status & caveats

> ⚠️ **`fit_point_process_em` and `fit_hybrid_em` are EXPERIMENTAL.**
> They fit the observation model (firing rates, and for the hybrid the
> Gaussian noise `R`) correctly.  As of the Tier 0.1 identifiability
> pass the latent `A`, `C` are now returned in a **canonical gauge**
> (whiten + SVD-rotate + sign-fix), removing the scale/rotation drift —
> but EM can still converge to **different local optima** across seeds,
> so a single fit's `A`/`C` should be interpreted with care.  Both
> functions emit a `UserWarning` to this effect.

**The gauge freedom, and how it is now pinned.**  A Poisson LDS has a
gauge freedom: the transform `(A, C, x) → (T A T⁻¹, C T⁻¹, T x)` leaves
the observable log-rate `C x` — and hence the likelihood — exactly
invariant for any invertible `T` (the full `GL(d)` group, `d²` degrees
of freedom).  EM has no reason to prefer any point on this orbit, so an
unconstrained fit lets the absolute scale and rotation of `A`/`C` drift
freely (the original PR showed `|C|` of 5–100 on fits whose *rates*
were perfectly sensible).

This is the MATLAB `PP_EMCreateConstraints` role.  The Python port pins
the gauge to the standard LDS canonical form (cf. Macke et al. 2011;
Buesing et al. 2012) **once after EM convergence** — never per
iteration, which fights the Newton trust-region and destabilizes the
fit:

1. **Whiten** the latent so the empirical state second moment becomes
   the identity (`T = M^{-1/2}`) — removes the symmetric gauge DOF.
2. **SVD-rotate** so the stacked emission matrix has orthogonal columns
   ordered by descending singular value — removes the residual `O(d)`.
3. **Sign-fix** each axis so the largest-magnitude entry of each
   emission column is positive — removes the `2^d` sign flips.

The returned emission matrix therefore satisfies `CᵀC = diag(S²)`
(a machine-precision-exact, seed-stable invariant the tests assert).
What remains is *local-optima* multiplicity — distinct fits with
genuinely different likelihoods — not gauge freedom; pinning that would
require multi-restart model selection (tracked separately).

**What IS reliable** (use these):

- The fitted **firing rate** `exp(C x)` / smoothed log-rate — the
  identifiable observable.  Re-smooth at the returned parameters with
  `cmgf_poisson_smoother` to obtain it.
- For `fit_hybrid_em`, the **Gaussian observation noise `R`** (recovers
  the true value within a small factor — it lives in observation space
  and is gauge-invariant).
- The **held-out predictive log-likelihood** (`point_process_predictive_ll`
  / `hybrid_predictive_ll`) — a true, gauge-invariant quality score; see
  below.

**Checking fit quality — use the predictive log-likelihood, not the EM
trace.**  The `marginal_log_likelihoods` returned by the trainers is a
*surrogate* (the Gaussian-smoother likelihood of the re-linearized IRLS
pseudo-observations); it changes basis every iteration and is **not**
monotonic or comparable across fits.  For a real metric, score the
observations with the one-step-ahead predictive log-likelihood:

```python
import numpy as np
from nstat.extras.em.dynamax_bridge import (
    fit_point_process_em, point_process_predictive_ll,
)

y_train, y_test = y[:800], y[800:]                 # held-out split
fit = fit_point_process_em(y_train, state_dim=3, n_iter=30, seed=0)
score = point_process_predictive_ll(
    y_test, fit.transition_matrix, fit.observation_matrix,
    fit.transition_covariance, fit.initial_state_mean,
    fit.initial_state_covariance,
)
print(score.total)            # higher = better; compare seeds/state_dims
print(score.per_timestep)     # locate where a fit predicts poorly
```

Because it is gauge-invariant and pure NumPy, it is the right tool to
pick `state_dim`, compare EM restarts, or detect a bad fit.

> ⚠️ **Observability caveat (a real limitation the diagnostic exposes).**
> PP_EM's held-out predictive performance depends strongly on how much
> the spikes constrain the latent.  With **weak observability** (few
> neurons and/or small loadings) PP_EM tends to converge to a degenerate
> solution — dynamics `A → 0`, inflated `C`/`Q` — that tracks the
> in-sample mean rate but generalizes *worse than a constant-rate model*
> (the predictive LL can be sharply negative). With **strong
> observability** (many informative neurons) `A` is recovered and the
> held-out predictive LL improves over the initialization.  The practical
> recommendation: always check `*_predictive_ll` on held-out data, prefer
> more neurons / informative loadings, and use multi-restart selection.
> Hardening PP_EM convergence under weak observability is tracked in
> [`parity/methods_roadmap.md`](https://github.com/cajigaslab/nSTAT-python/blob/main/parity/methods_roadmap.md)
> (Tier 0).

**What was fixed in the deep-dive pass** (improvements over the initial
PR):

- **Lag-one cross-covariance** is now exact: the E-step uses an IRLS
  pseudo-observation linearization + a purpose-built **time-varying-R
  RTS smoother** (`_kalman_rts_smoother_tv`) that returns the lag-one
  smoothed cross-covariances.  This stopped the previous A→0 collapse
  (the moment-matching approximation that dropped the cross-cov term
  biased `A` toward zero).
- **Time-varying pseudo-observation noise** `R_t = 1/λ_t`: substituting
  a fixed `R` (forced by a batched smoother) breaks the IRLS weight
  cancellation and was numerically unstable (SVD non-convergence) at
  low rates.  The new smoother accepts per-timestep `R_t`.
- **Gaussian `R` M-step trace correction**: now includes the
  `C_g Σ_t C_g'` latent-uncertainty term, without which `R` collapsed
  toward zero over iterations.
- **Gauge + step bounding**: a cheap per-iteration unit-RMS *diagonal*
  scale pin plus a Newton trust-region keep `|C|` finite during the
  fit, and a single **full canonical-gauge transform after convergence**
  (whiten + SVD-rotate + sign-fix) pins the remaining rotational and
  sign freedom.  This is the Tier 0.1 identifiability pass — the
  `PP_EMCreateConstraints` equivalent; `A`/`C` are now returned in a
  unique canonical frame (`CᵀC` diagonal, descending).

**Still approximate / deferred to a future release:**

- **Multi-restart model selection** — the canonical gauge makes a single
  fit's `A`/`C` well-defined, but EM can still reach different local
  optima across seeds (genuinely different likelihoods, not gauge
  copies).  Picking the best of several restarts is the remaining step
  toward fully reproducible `A`/`C`.
- The **Laplace `E[exp(C x_t)]`** uses the diagonal quadratic
  correction; sufficient for moderate rates, may underestimate
  variance at high rates.
- The reported `marginal_log_likelihoods` are the **surrogate
  Gaussian smoother log-likelihoods** (the IRLS pseudo-observations
  are re-linearized each iteration), not the true Poisson marginal
  likelihood — do **not** use the trace as a convergence diagnostic.

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
