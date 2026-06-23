# `nstat.extras.matlab_rng` — MATLAB-aligned Mersenne Twister RNG

Provides a thin wrapper around NumPy's legacy `numpy.random.RandomState`
(MT19937) seeded the same way MATLAB's `rng(N)` seeds its default
`twister` generator. Used by recipes that compare Python output against
MATLAB Monte Carlo outputs where reproducible cross-language streams
matter (`PPLFP_MStep`, `PPLFP_EM`, `v9_PPSS_EM` Case C drift entries).

## Install

No optional dependency — `matlab_rng` ships with the core
`nstat-toolbox` package. The module has zero runtime dependencies beyond
NumPy, which `nstat-toolbox` already requires.

```bash
pip install nstat-toolbox
```

## Usage

```python
from nstat.extras.matlab_rng import MatlabRNG, seeded_global_rng

# Bit-equivalent to MATLAB rand under matching seed:
r = MatlabRNG(42)
u = r.rand(3)                     # matches MATLAB rand(1,3) under rng(42)
n = r.randn(3)                    # Box-Muller from same MT stream (deterministic)

# Context manager: seeds NumPy global state + monkey-patches default_rng
# so any nstat function inside the block draws deterministically.
import numpy as np
with seeded_global_rng(42) as rng:
    a = np.random.randn(3)        # deterministic
    g = np.random.default_rng()   # also deterministic
    b = g.standard_normal(3)
```

## Bit-equivalence guarantees

| Operation | MATLAB equivalent | Bit-equivalent? |
|---|---|---|
| `MatlabRNG(seed).rand(n)` | `rng(seed); rand(1, n)` | **yes** |
| `MatlabRNG(seed).randn(n)` | `rng(seed); randn(1, n)` | **no** (statistically equivalent) |
| `MatlabRNG(seed).normrnd(mu, sigma, n)` | `rng(seed); normrnd(mu, sigma, 1, n)` | **no** (statistically equivalent) |

### Why `randn` isn't bit-equivalent

MATLAB's `randn` uses the Marsaglia-Tsang **Ziggurat** algorithm (R14sp1
and later) to convert uniform draws into standard normals. NumPy's
legacy `RandomState.randn` and `MatlabRNG.randn` both use **Box-Muller**
instead. Both draw from the same MT19937 stream and produce the same
`N(0,1)` distribution, but they consume different numbers of uint32
words per output sample.

A proper Ziggurat port would close the gap; deferred to a future
release if any caller needs strict cross-language MC parity.

## When to use

| Goal | Use |
|---|---|
| Reproducible Python-side MC across runs | `MatlabRNG` or `seeded_global_rng` |
| Bit-equivalent uniform stream to MATLAB | `MatlabRNG(seed).rand(...)` |
| Bit-equivalent normal stream to MATLAB | **Not available** — use `MatlabRNG` for statistical equivalence |
| Default Python RNG (PCG64, faster, not MATLAB-compatible) | `numpy.random.default_rng(seed)` directly — don't wrap |

## Related

- `parity/numerical_drift_spec.yml` entries `PPLFP_MStep`, `PPLFP_EM`,
  `v9_PPSS_EM` route through `seeded_global_rng(42)` for deterministic
  drift comparison.
- v13 iter 60 introduced the wiring; iter-63 fixup tightened the
  resulting Case C tolerances by 2-10× on `atol`.
