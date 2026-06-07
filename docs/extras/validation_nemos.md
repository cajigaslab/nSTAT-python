# `nstat.extras.validation.nemos_bridge` — NeMoS GLM cross-validation

Cross-validate [`nstat.fit_poisson_glm`](../api.rst) against
[NeMoS](https://github.com/flatironinstitute/nemos), Flatiron's
JAX-backed Poisson/Gamma GLM toolbox. NeMoS uses the *same*
raised-cosine / B-spline basis families that nstat does (and that the
2012 paper assumes), with an MIT license that is GPL-2 compatible.

This is the strongest available Python-side cross-check on nstat's
GLM fitting path short of running MATLAB-engine itself.

## Install

```bash
pip install nstat-toolbox[nemos]   # JAX install is ~200 MB
```

Pulls `nemos>=0.2`.

## API

| Symbol | Notes |
|---|---|
| `cross_validate_poisson_glm(X, y, *, include_intercept=True)` | Fits both, returns `GLMComparison` |
| `GLMComparison` (dataclass) | `nstat_coef`, `nemos_coef`, `coef_inf_norm`, `coef_rel_inf_norm` |
| `GLMComparison.assert_agree(atol, rtol)` | Regression-guard hook for parity tests |

## Recipe

```python
import numpy as np
from nstat.extras.validation.nemos_bridge import cross_validate_poisson_glm

# Generate a synthetic Poisson GLM fixture
rng = np.random.default_rng(0)
X = rng.standard_normal((1000, 3))
beta_true = np.array([0.2, -0.4, 0.1])
y = rng.poisson(np.exp(0.5 + X @ beta_true))

# Cross-validate
cmp = cross_validate_poisson_glm(X, y)
print(f"nstat: {cmp.nstat_coef}")
print(f"NeMoS: {cmp.nemos_coef}")
print(f"|Δβ|_∞: {cmp.coef_inf_norm:.3e}")

# Use in a parity test
cmp.assert_agree(atol=5e-2, rtol=5e-2)
```

## Gotchas

- **JAX install footprint.** The `[nemos]` extra pulls JAX (~200 MB).
  For CI environments, prefer `[test-parity]` which bundles NeMoS +
  pykalman + statsmodels + nitime in one install.
- **Optimizer divergence.** nstat uses IRLS with a tiny ridge
  (`l2=1e-6`); NeMoS uses an optax-driven first-order optimizer with
  its own stopping criteria. Coefficients typically agree to ~5e-3 on
  well-conditioned problems; the bridge's `assert_agree(atol=5e-2)`
  default is intentionally loose enough to absorb this.
- **Intercept layout.** When `include_intercept=True`, both
  coefficient vectors are returned intercept-first (length `p+1`).

## End-to-end demo

[`examples/extras/validation_nemos_demo.py`](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/extras/validation_nemos_demo.py)
runs the full fit-and-compare on a 1000×3 Poisson fixture.

## Upstream references

- NeMoS: https://github.com/flatironinstitute/nemos
- License: MIT (GPL-2 compatible — MIT can be redistributed under GPL)
- Sibling library: [pynapple](https://github.com/pynapple-org/pynapple)
  (same Flatiron group; pynapple is NeMoS's native time-series container)
- Maintained by: Center for Computational Neuroscience, Flatiron
  Institute (NIH BRAIN funded)
