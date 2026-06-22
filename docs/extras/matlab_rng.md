# `nstat.extras.matlab_rng` — MATLAB-aligned MT19937 RNG

A thin wrapper around NumPy's legacy `RandomState` (MT19937) seeded the
same way MATLAB's `rng(N)` seeds its default `twister` generator.  Use
this module when you need Python-side Monte Carlo draws that are
run-to-run deterministic and comparable to MATLAB output.

Introduced in **v0.5.6** as part of the MATLAB-parity closure work.

## Install

```bash
pip install nstat-toolbox
```

No extras group required. `nstat.extras.matlab_rng` ships with the core
package and depends only on NumPy.

## API

| Symbol | Notes |
|---|---|
| `MatlabRNG(seed)` | Constructor. `seed` must be a non-negative integer. |
| `MatlabRNG.rand(*shape)` | Uniform `[0, 1)` draws. **Bit-equivalent to MATLAB `rand`** under the same seed. |
| `MatlabRNG.randn(*shape)` | Standard-normal draws via Box-Muller from the same MT19937 stream. Deterministic and machine-independent; *not* bit-equivalent to MATLAB's Ziggurat `randn`. |
| `MatlabRNG.legacy_randn(*shape)` | NumPy's legacy `RandomState.randn` (polar Marsaglia). Bit-equivalent to NumPy's `np.random.randn` under the same seed; not equivalent to MATLAB. |
| `MatlabRNG.normrnd(mu, sigma, *shape)` | MATLAB-style `normrnd(mu, sigma, ...)` using `randn`. |
| `MatlabRNG.standard_normal(size)` | NumPy-Generator-style alias for `randn`. |
| `MatlabRNG.seed` | The seed used to initialise the generator (read-only property). |
| `MatlabRNG.random_state` | The underlying `numpy.random.RandomState` instance (read-only property). |
| `seeded_global_rng(seed)` | Context manager: seeds `np.random.seed(seed)` and monkey-patches `np.random.default_rng` for the with-block, then restores both on exit. Yields a `MatlabRNG`. |

## What is (and is not) bit-equivalent to MATLAB

| Draw | Bit-equivalent? |
|---|---|
| Uniform `rand` | **Yes** — both use the same MT19937 state initialisation |
| Normal `randn` via `MatlabRNG.randn` | No — Box-Muller vs. MATLAB's Ziggurat. Same distribution, different bit stream. |
| Normal via `MatlabRNG.legacy_randn` | No — NumPy polar Marsaglia vs. MATLAB Ziggurat. |

## Quick start

```python
from nstat.extras.matlab_rng import MatlabRNG, seeded_global_rng

# Uniform draws — bit-equivalent to MATLAB rand(1,3) under rng(42)
r = MatlabRNG(42)
u = r.rand(3)
print(u)  # same as MATLAB: rand(1,3) after rng(42)

# Reproducibility check
r2 = MatlabRNG(42)
assert (r2.rand(3) == u).all()

# Seed the global NumPy state for a whole recipe block
import numpy as np
with seeded_global_rng(42) as rng:
    a = np.random.randn(5)          # legacy global path, deterministic
    g = np.random.default_rng()     # patched: returns PCG64(42)
    b = g.standard_normal(5)
```

## Using `seeded_global_rng` in parity recipes

`seeded_global_rng` is the recommended way to make an nstat Monte Carlo
recipe deterministic without modifying the recipe function itself:

```python
from nstat.extras.matlab_rng import seeded_global_rng
from nstat import Trial  # or any nstat function that draws internally

with seeded_global_rng(0):
    result = my_simulation(n_trials=100)   # all random draws are seeded
```

On entry the context manager saves the current `np.random` global state,
seeds it with `seed`, and patches `np.random.default_rng` to return a
deterministically-seeded `Generator(PCG64(seed))`.  On exit both are
restored, so the caller's RNG context is unaffected.

## Notes

- **One `MatlabRNG` per recipe run.** Do not share a `MatlabRNG` instance
  between concurrent threads; the underlying `RandomState` is not
  thread-safe.
- **`randn` stream diverges from MATLAB after the first draw.** MATLAB's
  Ziggurat consumes a variable number of MT words per sample; Box-Muller
  consumes exactly 2 uniforms per pair.  The streams desynchronise
  immediately.  For full bit-exact normal equivalence a Ziggurat port
  would be required (out of scope).
- **`seeded_global_rng` patches `default_rng` globally.** Successive
  calls to `np.random.default_rng()` inside the block all return a fresh
  `Generator(PCG64(seed))` — matching the intent of "run-to-run
  determinism" for the MC paths it targets.  Callers that pass an
  explicit seed to `default_rng` are passed through unmodified.

## References

- Matsumoto M, Nishimura T (1998). *Mersenne twister: a 623-dimensionally
  equidistributed uniform pseudo-random number generator.* ACM TOMACS
  8(1):3–30.
- The MathWorks. *`rng` — Control random number generation* (MATLAB
  documentation). Describes the `twister` (MT19937) seeding convention.
