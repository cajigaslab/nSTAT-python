# MATLAB defects and Python improvements ledger

This file records every place where Python's behavior intentionally diverges
from MATLAB nSTAT. Per AGENT_GUIDE.md §0, three reasons justify divergence:

1. **Defect fix** — MATLAB has a bug (off-by-one, wrong sign, instability)
2. **Stability improvement** — Python uses a more numerically robust algorithm
3. **Efficiency improvement** — Python uses a faster algorithm with bit-equivalent output

Schema for each entry:

```
## Defect: <one-line title>
- **MATLAB location:** `<file>:<line>` in `cajigaslab/nSTAT@<sha-or-tag>`
- **Defect class:** Bug | Stability | Efficiency
- **MATLAB behavior:** <what the original code does>
- **Correct behavior:** <what the science demands; cite reference>
- **Python implementation:** `<file>:<line>` in this repo
- **Fixture impact:** `tests/parity/fixtures/matlab_gold/<file>.mat` refreshed in commit `<sha>` (or "no fixture impact")
- **Discovered:** <iter # / date>
```

---

## Open entries

### Stability: point-process thinning uses -expm1(-x) instead of 1 - exp(-x)

- **MATLAB location:** `+nstat/simulatePointProcess.m` and analogous
  thinning paths in `cajigaslab/nSTAT@7798a14` use the direct form
  `p = 1 - exp(-lambda*dt)`.
- **Defect class:** Stability
- **MATLAB behavior:** For small `lambda*dt` (low firing rates or fine
  bins) the subtraction `1 - exp(-x)` suffers catastrophic
  cancellation: `exp(-x) ≈ 1 - x + x²/2 - …`, and double-precision
  evaluation loses up to ~half the significant digits of `p`. The
  effect biases per-bin spike probabilities slightly toward zero,
  most visible at sub-1 Hz rates with millisecond bins.
- **Correct behavior:** Use the IEEE-754 primitive `expm1`, which
  computes `exp(x) - 1` with full precision near zero. Then
  `p = -expm1(-lambda*dt)` carries the full mantissa down to
  `lambda*dt ≈ 1e-16`. Same algorithm, no fixture impact at any
  realistic rate but mathematically tighter.
- **Python implementation:**
  - `nstat/simulation.py:47-50` (`simulate_poisson_from_rate`)
  - `nstat/simulators.py:75-78` (`simulate_point_process`)
  - `nstat/fit.py:121-131` (`_pp_uniforms_from_lambda` KS helper)
- **Fixture impact:** no fixture impact — all 40
  `tests/test_*_fidelity.py` + `tests/test_matlab_gold_fixtures.py`
  + `tests/parity/` tests pass unchanged at default tolerance.
- **Discovered:** iter 3 / 2026-06-18

---

### Stability: UKF Kalman gain via linear solve instead of explicit inverse

- **MATLAB location:** `@DecodingAlgorithms/ukf.m` in
  `cajigaslab/nSTAT@7798a14` computes `K = P12 / P2` (MATLAB's
  mrdivide, which internally calls LAPACK solve — already correct in
  MATLAB), but our prior Python port translated it as
  `K = P12 @ np.linalg.inv(P2)`.
- **Defect class:** Stability
- **MATLAB behavior:** MATLAB's mrdivide is stable. Python's port
  formed the explicit inverse, which loses ~one order of conditioning
  vs. a direct solve and is the canonical "don't do this" pattern in
  numerical linear algebra (Trefethen & Bau, *Numerical Linear
  Algebra*, Lec. 20).
- **Correct behavior:** `K @ P2 = P12  ⇒  P2.T @ K.T = P12.T  ⇒
  K = solve(P2.T, P12.T).T`. Identical output for well-conditioned
  P2; meaningfully more accurate when P2 is near-singular (large
  measurement-noise / weak-update regime).
- **Python implementation:** `nstat/decoding_algorithms.py:2530-2533`
  inside `DecodingAlgorithms.ukf`.
- **Fixture impact:** no fixture impact — UKF gold fixtures pass
  unchanged.
- **Discovered:** iter 3 / 2026-06-18

---

### Stability: Events.plot label x-coordinate uses data-axis transform

- **MATLAB location:** `@Events/plot.m:97` in `cajigaslab/nSTAT@7798a14`
- **Defect class:** Stability
- **MATLAB behavior:** Event labels are placed using axes-normalized
  coordinates with a fixed `-0.02` nudge from the event line. With tight
  `xlim` (e.g. `[0.55, 0.71]`), the `-0.02` shift in axes-fraction units
  no longer tracks the event line because it's independent of data width.
- **Correct behavior:** Labels should sit directly above each event line
  regardless of `xlim`. Solution: use `ax.get_xaxis_transform()` so the
  x-coordinate is in data space (anchored to the event time, tracks the
  line) and y is in axes coordinates (just above the axis). The
  `ha='center'`, `va='bottom'` alignment replaces the `-0.02` nudge.
- **Python implementation:** `nstat/events.py:138-152` (iter 2 of parity push)
- **Fixture impact:** `tests/parity/fixtures/matlab_gold/events_exactness.mat`
  `plot_label_positions` field refreshed in iter 4 to reflect the new data
  coordinates (old: `[0.18, 1.03, 0.68, 1.03]`, new: `[0.2, 1.02, 0.7, 1.02]`).
- **Discovered:** iter 2 / 2026-06-18

---

## Reviewer checklist for parity-affecting PRs

- [ ] Every modified gold fixture has a defects-ledger entry
- [ ] Every "MATLAB does X but I changed Python to do Y" claim has a citation
- [ ] No silent fixture refresh (every `.mat` change has a commit message)
- [ ] No reverting a MATLAB-style convention thinking it's a bug — when in
      doubt, ask the maintainer
