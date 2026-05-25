# `nstat.extras.validation.statsmodels_bridge` — statsmodels GLM cross-validation

Cross-validate [`nstat.fit_poisson_glm`](../api.html) against
[`statsmodels.genmod.GLM`](https://www.statsmodels.org/) — the third
independent Poisson-GLM oracle in `nstat.extras.validation` (alongside
NeMoS and nstat's own IRLS).

Because **both nstat and statsmodels use IRLS**, they should agree to
**near machine precision** (~1e-9 on well-conditioned synthetic
fixtures). This is the **tightest** cross-validation oracle available
in the extras namespace — much tighter than NeMoS (~5e-3, different
optimizer) or pykalman (~1e-2, different paradigm).

## Install

```bash
pip install nstat-toolbox[test-parity]   # bundles statsmodels + nemos + pykalman + nitime
```

statsmodels is already in most SciPy installations — install footprint
is trivial.

## API

| Symbol | Notes |
|---|---|
| `cross_validate_poisson_glm(X, y, *, include_intercept=True)` | Fits both, returns `StatsmodelsGLMComparison` |
| `StatsmodelsGLMComparison` (dataclass) | `nstat_coef`, `statsmodels_coef`, `coef_inf_norm`, `coef_rel_inf_norm` |
| `StatsmodelsGLMComparison.assert_agree(atol=1e-3, rtol=1e-3)` | Regression-guard hook for parity tests |

## Recipe

```python
import numpy as np
from nstat.extras.validation.statsmodels_bridge import cross_validate_poisson_glm

rng = np.random.default_rng(0)
X = rng.standard_normal((1000, 3))
beta_true = np.array([0.2, -0.4, 0.1])
y = rng.poisson(np.exp(0.5 + X @ beta_true))

cmp = cross_validate_poisson_glm(X, y)
print(f"|Δβ|_∞: {cmp.coef_inf_norm:.3e}")   # expect ~1e-9

# Tight regression guard — both use IRLS, so machine-precision agreement.
cmp.assert_agree(atol=1e-6, rtol=1e-6)
```

## Triangulation pattern

The three GLM oracles together form the strongest cross-validation
matrix for `nstat.fit_poisson_glm`:

| Oracle | Algorithm | Expected agreement |
|---|---|---|
| **statsmodels** | IRLS (same as nstat) | ~1e-9 (machine precision) |
| **NeMoS** | optax first-order (independent) | ~5e-3 |
| **MATLAB gold fixtures** | MATLAB's `glmfit` (IRLS) | exact (by design) |

A regression that loosens the statsmodels agreement beyond ~1e-6
likely indicates a real bug in nstat's IRLS path — much more sensitive
signal than the NeMoS bridge.

## Gotchas

- **Intercept layout.** `include_intercept=True` (default) prepends an
  intercept column for statsmodels via `sm.add_constant(X)`, and
  returns intercept-first coefficient vectors from both libraries.
- **Tolerance philosophy.** The default `atol=1e-3` is intentionally
  loose enough to absorb real-data conditioning issues. For synthetic
  well-conditioned fixtures, **tighten to `1e-6`** to surface
  meaningful deviations.

## End-to-end demo

[`examples/extras/validation_statsmodels_demo.py`](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/extras/validation_statsmodels_demo.py)
runs the full fit-and-compare on a 1000×3 Poisson fixture.

## Upstream references

- statsmodels: https://www.statsmodels.org
- License: BSD-3-Clause (GPL-2 compatible)
- Algorithm: IRLS via `statsmodels.genmod.families.Poisson()` with the
  canonical log link
