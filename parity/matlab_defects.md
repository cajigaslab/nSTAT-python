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

- **MATLAB location:** `+nstat/simulatePointProcess.m` and analogous thinning paths in `cajigaslab/nSTAT@7798a14`
- **Defect class:** Stability
- **MATLAB behavior:** Uses the direct form `p = 1 - exp(-lambda*dt)`.  For small `lambda*dt`
  (low firing rates or fine bins) the subtraction `1 - exp(-x)` suffers
  catastrophic cancellation: `exp(-x) ≈ 1 - x + x²/2 - …`, and
  double-precision evaluation loses up to ~half the significant digits
  of `p`.  The effect biases per-bin spike probabilities slightly toward
  zero, most visible at sub-1 Hz rates with millisecond bins.
- **Correct behavior:** Use the IEEE-754 primitive `expm1`, which computes `exp(x) - 1` with
  full precision near zero.  Then `p = -expm1(-lambda*dt)` carries the
  full mantissa down to `lambda*dt ≈ 1e-16`.  Same algorithm, no
  fixture impact at any realistic rate but mathematically tighter.
- **Python implementation:**
  - `nstat/simulation.py:47-50` (`simulate_poisson_from_rate`)
  - `nstat/simulators.py:75-78` (`simulate_point_process`)
  - `nstat/fit.py:121-131` (`_pp_uniforms_from_lambda` KS helper)
- **Fixture impact:** no fixture impact — all 40 `tests/test_*_fidelity.py` +
  `tests/test_matlab_gold_fixtures.py` + `tests/parity/` tests pass
  unchanged at default tolerance.
- **Discovered:** iter 3 / 2026-06-18

---

### Stability: UKF Kalman gain via linear solve instead of explicit inverse

- **MATLAB location:** `@DecodingAlgorithms/ukf.m` in `cajigaslab/nSTAT@7798a14`
- **Defect class:** Stability
- **MATLAB behavior:** `@DecodingAlgorithms/ukf.m` computes `K = P12 / P2` (MATLAB's
  mrdivide, which internally calls LAPACK solve — already correct in
  MATLAB), but our prior Python port translated it as
  `K = P12 @ np.linalg.inv(P2)`.  MATLAB's mrdivide is stable.
  Python's port formed the explicit inverse, which loses ~one order of
  conditioning vs. a direct solve and is the canonical "don't do this"
  pattern in numerical linear algebra (Trefethen & Bau, *Numerical
  Linear Algebra*, Lec. 20).
- **Correct behavior:** `K @ P2 = P12  ⇒  P2.T @ K.T = P12.T  ⇒  K = solve(P2.T, P12.T).T`.
  Identical output for well-conditioned P2; meaningfully more accurate
  when P2 is near-singular (large measurement-noise / weak-update
  regime).
- **Python implementation:** `nstat/decoding_algorithms.py:2530-2533` inside `DecodingAlgorithms.ukf`
- **Fixture impact:** no fixture impact — UKF gold fixtures pass unchanged.
- **Discovered:** iter 3 / 2026-06-18

---

### Stability: Events.plot label x-coordinate uses data-axis transform

- **MATLAB location:** `@Events/plot.m:97` in `cajigaslab/nSTAT@7798a14`
- **Defect class:** Stability
- **MATLAB behavior:** Event labels are placed using axes-normalized coordinates with a
  fixed `-0.02` nudge from the event line.  With tight `xlim`
  (e.g. `[0.55, 0.71]`), the `-0.02` shift in axes-fraction units no
  longer tracks the event line because it's independent of data width.
- **Correct behavior:** Labels should sit directly above each event line regardless of
  `xlim`.  Solution: use `ax.get_xaxis_transform()` so the
  x-coordinate is in data space (anchored to the event time, tracks
  the line) and y is in axes coordinates (just above the axis).  The
  `ha='center'`, `va='bottom'` alignment replaces the `-0.02` nudge.
- **Python implementation:** `nstat/events.py:138-152` (iter 2 of parity push)
- **Fixture impact:** `tests/parity/fixtures/matlab_gold/events_exactness.mat`
  `plot_label_positions` field refreshed in iter 4 to reflect the new
  data coordinates (old: `[0.18, 1.03, 0.68, 1.03]`, new:
  `[0.2, 1.02, 0.7, 1.02]`).
- **Discovered:** iter 2 / 2026-06-18

---

### Bug (preserved for exact-mirror parity, not "fixed"): Documented MATLAB quirk preserved: nst2.setMaxTime(21) mutates nst2 but not nst

- **MATLAB location:** `nSTATPaperExamples.m` lines 121-145 (Experiment 2, Explicit Stimulus / Whisker Data)
- **Defect class:** Bug (preserved for exact-mirror parity, not "fixed")
- **MATLAB behavior:** The script copies `nst2 = nst.copy()`, calls `nst2.setMaxTime(21)`,
  then later plots `nst.plot` for the spike raster — the unclipped
  `nst` object is what's drawn, so MATLAB's figure 2 top panel shows
  the full ~50.73 s record with ~966 spikes despite the script
  appearing to clip to 21 s.  The stimulus side IS clipped via
  `stim.getSigInTimeWindow(0, 21)`.
- **Correct behavior:** Per the exact-mirror rule, reproduce MATLAB's actual output.  The
  previous Python port "fixed" what it read as a MATLAB bug by
  clipping the spike train to 21 s in `payload['spike_indicator']`,
  which produced ~360 spikes over 0..21 s and diverged from MATLAB.
- **Python implementation:**
  - `nstat/paper_examples_full.py:421-432` now exposes both `spike_indicator_full`/`time_s_full` (matching MATLAB's `nst.plot`) and the original 21 s-clipped arrays (used elsewhere by the GLM fits).
  - `notebooks/ExplicitStimulusWhiskerData.ipynb` cell 4 was updated to draw the figure-1 raster and figure-2 top panel from the full-length arrays.  Stimulus axes in MATLAB are raw volts (0..9.953 V); the GLM-internal `/10` normalization is now kept in `stim` and the raw signal is exposed as `payload['stimulus_raw_v']` for plotting.
- **Fixture impact:** No `.mat` gold fixture change — this affects notebook figures only.
- **Discovered:** iter 14 / 2026-06-18

---

### Stability (no behavioural change; intentional architectural divergence): RNG-stream divergence: MATLAB Mersenne Twister vs NumPy default_rng

- **MATLAB location:** Multiple paper-example scripts: `nSTATPaperExamples.m` Experiment 5 (StimulusDecode2D) draws coefficients `coeffs = -|randn(80,5)|` and innovations `r = 0.01*randn(2,N)`; `TrialExamples.m:33` draws spike times via `sort(rand(1,100))*lengthTrial`.
- **Defect class:** Stability (no behavioural change; intentional architectural divergence)
- **MATLAB behavior:** Uses MATLAB's Mersenne Twister via `randn`/`rand`, seeded with
  `rng(seed)`.
- **Correct behavior:** Python uses `np.random.default_rng(seed)` (PCG64), the modern
  NumPy-recommended generator.  Per the porting guideline that random
  sequences need not be bit-identical across language runtimes, we do
  not try to replicate MT19937 bit-stream in Python.

  Evidence of formula equivalence: when MATLAB's `dataMat` and
  `coeffs` are piped into the Python `_simulate_decode` formula
  `exp(eta)/(1+exp(eta))/delta`, output matches MATLAB to ~1e-13 —
  the pipeline is structurally identical; only the random draws
  differ.
- **Python implementation:**
  - `notebooks/StimulusDecode2D.ipynb` and `notebooks/TrialExamples.ipynb`.
  - `TrialExamples` previously used a deterministic quasi-uniform `np.linspace` grid (CV<<1); iter 14 fixed this to `np.sort(rng.uniform(0, length_trial, 100))` to recover the Poisson-like ISI character (CV~1) MATLAB produces.  `StimulusDecode2D` RNG choice is intentional and not changed.
- **Fixture impact:** No `.mat` gold fixture change.
- **Discovered:** iter 14 / 2026-06-18

---

## Reviewer checklist for parity-affecting PRs

- [ ] Every modified gold fixture has a defects-ledger entry
- [ ] Every "MATLAB does X but I changed Python to do Y" claim has a citation
- [ ] No silent fixture refresh (every `.mat` change has a commit message)
- [ ] No reverting a MATLAB-style convention thinking it's a bug — when in
      doubt, ask the maintainer
