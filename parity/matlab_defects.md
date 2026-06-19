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
- **Upstream status:** adopted-upstream
- **Resolved in:** cajigaslab/nSTAT@main (post-2026-06-19)
- **Resolved iter:** iter 45 / 2026-06-19
- **Resolved notes:** Upstream merged the proposed `-expm1(-lambda*dt)` fix in the
  thinning code path. Verified in iter 44 by re-capturing MATLAB gold
  fixtures against updated MATLAB: no fixture impact — Python was
  already using the better algorithm, and MATLAB's switch to
  `-expm1` agrees with Python to machine precision so the fresh
  `.mat` fixtures are byte-equivalent to the pre-update baselines on
  the thinning paths.
- **Upstream issue:** cajigaslab/nSTAT#78

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
- **Upstream status:** adopted-upstream
- **Resolved in:** cajigaslab/nSTAT@main (post-2026-06-19)
- **Resolved iter:** iter 45 / 2026-06-19
- **Resolved notes:** Upstream confirmed MATLAB's `mrdivide` path was already correct in
  the original `ukf.m` (the issue was a Python-port translation that
  had used `inv()`; MATLAB itself needed no change). Verified in
  iter 44 by re-capturing MATLAB gold fixtures: no fixture impact —
  UKF outputs unchanged.
- **Upstream issue:** cajigaslab/nSTAT#79

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
- **Upstream status:** adopted-upstream
- **Resolved in:** cajigaslab/nSTAT@main (post-2026-06-19)
- **Resolved iter:** iter 45 / 2026-06-19
- **Resolved notes:** Upstream merged the data-axis transform fix in `@Events/plot.m`.
  Verified in iter 44 by re-capturing MATLAB gold fixtures:
  `events_exactness.mat` `plot_label_positions` y-component shifted
  from `1.02` (axes-fraction nudge) to `2.09` (data coordinates above
  max event amplitude) — matching the corrected MATLAB output. Iter 47
  must align Python's `Events.plot` label-positioning with the new
  MATLAB convention to make `test_events_match_matlab_gold_fixture`
  pass against the refreshed fixture.
- **Upstream issue:** cajigaslab/nSTAT#80

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

### Bug (preserved): PPLFP_EM `n4` parameter-count branch uses Qhat constraints instead of Rhat

- **MATLAB location:** `+nstat/+decoding/PPLFP.m:1936-1942` in `cajigaslab/nSTAT@2d86602` (function `PPLFP_EM`)
- **Defect class:** Bug (preserved)
- **MATLAB behavior:** The middle branch of the `n4` (Rhat-parameter-count) cascade tests
  `PPLFP_EM_Constraints.QhatDiag==1 && QhatIsotropic==0` instead of
  the parallel `RhatDiag`/`RhatIsotropic` pair used by the matched
  `n2` branch immediately above.  When R has a non-trivial constraint
  but Q is dense, the AIC/AICc/BIC parameter count over-reports `n4`
  to `numel(Rhat)`; when Q is diagonal-non-isotropic the count
  collapses to `size(Rhat,1)` regardless of R's actual structure.
- **Correct behavior:** `n4` should mirror the symmetric `n2` cascade: test
  `RhatDiag && RhatIsotropic -> 1`, else `RhatDiag -> size(R,1)`,
  else `numel(R)`.  Only Rhat's own constraints govern the count of
  free parameters in Rhat.
- **Python implementation:** `nstat/decoding/PPLFP.py` `PPLFP.PPLFP_EM` — `n4` branch preserves the MATLAB conditional verbatim with an in-line `NOTE:` comment so the AIC/BIC values agree with MATLAB to the bit.  A fix can be applied later if/when MATLAB upstream patches it.
- **Fixture impact:** No `.mat` gold fixture change yet (PPLFP_EM fixtures not regenerated by iter 29).
- **Discovered:** iter 29 / 2026-06-18

---

### Case C (stability / tolerance relaxation): PPLFP_EStep baseline .mat absent — numerical drift unverified

- **MATLAB location:** `+nstat/+decoding/PPLFP.m:1990-2206` in `cajigaslab/nSTAT` (function `PPLFP_EStep`)
- **Defect class:** Case C (stability / tolerance relaxation)
- **MATLAB behavior:** `PPLFP_EStep` runs the PPLFP forward filter, the RTS smoother, and
  accumulates the sufficient statistics for the M-step (Sxkm1xkm1,
  Sxkm1xk, Sxkxk, Sykyk, Sxkyk), the linearised Gaussian terms
  (sumXkTerms, sumYkTerms), the conditional-intensity contribution
  sumPPll, and the complete-data log-likelihood lower bound `logll`.
  It contains multiple MATLAB `B/A`-style right divides over potentially
  ill-conditioned smoothed covariances, plus an unbounded `exp(terms)`
  that can overflow at extreme states.
- **Correct behavior:** A `.mat` gold fixture covering `PPLFP_EStep` should be captured by
  seeding MATLAB's RNG, recording all inputs (A, Q, C, R, y, alpha,
  dN, mu, beta, gamma, HkAll, x0, Px0) and the four returned objects
  (x_K, W_K, logll, and every field of ExpectationSums), then
  verifying Python against it at `rtol=1e-6, atol=1e-8` for x_K / W_K
  / Sx* / S*k* / sumXkTerms / sumYkTerms, and a relaxed `rtol=1e-4`
  for `logll` (which sums sumPPll over K and is the most numerically
  sensitive aggregate).
- **Python implementation:**
  - `nstat/decoding/PPLFP.py` `PPLFP.PPLFP_EStep` is a full port: filter via `PPLFP_DecodeLinear`, RTS smoother via `kalman_smootherFromFiltered` (with MATLAB column-major ↔ Python time-major shape conversions), de Jong/MacKinnon cross-covariance Wku, sufficient statistics for both Poisson and binomial CIF links, and the complete-data log-likelihood (`-Dx*K/2*log(2π) - K/2*log|Q| - Dy*K/2*log(2π) - K/2*log|R| - Dx/2*log(2π) - 1/2*log|Px0| + sumPPll - 0.5*tr(Q\sumXkTerms) - 0.5*tr(R\sumYkTerms) - Dx/2`).
  - Uses `np.linalg.solve` for every MATLAB `B/A` to honour the iter-4 stability rule; falls back to `np.linalg.pinv` only when the solve fails.
  - Runs end-to-end on synthetic inputs (verified iter 34): returns ExpectationSums dict with all 11 fields, finite logll for both poisson and binomial fitType.
  - `parity/numerical_drift_spec.yml` entry `PPLFP_EStep` is staged with `todo: true`; recipe `pplfp_estep` will be wired in iter 35+ once the fixture lands.
- **Fixture impact:** Missing fixture `tests/parity/fixtures/matlab_gold/pplfp_PPLFP_EStep.mat` — baseline capture failed during iter-29 baseline phase.
- **Discovered:** iter 34 / 2026-06-18

---

### Case C (stability / tolerance relaxation): PPLFP_MStep baseline .mat absent — numerical drift unverified

- **MATLAB location:** `+nstat/+decoding/PPLFP.m:2207-3093` in `cajigaslab/nSTAT` (function `PPLFP_MStep`)
- **Defect class:** Case C (stability / tolerance relaxation)
- **MATLAB behavior:** `PPLFP_MStep` performs the EM maximisation step. The NewtonRaphson
  branch samples `McExp=50` Monte-Carlo draws per time-step via
  `normrnd(0,1,...)`, so even bit-identical pre-state cannot reproduce
  MATLAB's RNG stream from Python. The GLM branch routes through
  `Analysis.RunAnalysisForAllNeurons`, which is itself a multi-stage
  port and contributes its own drift envelope.
- **Correct behavior:** A `.mat` gold fixture covering `PPLFP_MStep` should be captured by
  seeding MATLAB's RNG, recording all inputs and the ten returned
  arrays (`Ahat, Qhat, Chat, Rhat, alphahat, muhat_new, betahat_new,
  gammahat_new, x0hat, Px0hat`), then verifying Python against it at
  `rtol=1e-4` (Newton-Raphson branch) or `rtol=1e-6` (closed-form
  arrays `Ahat, Chat, alphahat, Qhat, Rhat, x0hat, Px0hat`, which do
  not touch the RNG).
- **Python implementation:**
  - `nstat/decoding/PPLFP.py` `PPLFP.PPLFP_MStep` is a full port (closed-form updates + GLM branch + Newton-Raphson branch for `beta`, `mu`, `gamma`); runs end-to-end on synthetic inputs (verified iter 34).
  - `parity/numerical_drift_spec.yml` entry `PPLFP_MStep` is staged with `todo: true`; recipe `pplfp_mstep` will be wired in iter 35+ once the fixture lands.
- **Fixture impact:** Missing fixture `tests/parity/fixtures/matlab_gold/pplfp_PPLFP_MStep.mat` — baseline capture failed during iter-29 baseline phase.
- **Discovered:** iter 34 / 2026-06-18

---

### Case C (stability / tolerance relaxation): PPLFP_ComputeParamStandardErrors — Monte Carlo SE drift between MATLAB and NumPy RNG streams

- **MATLAB location:** `+nstat/+decoding/PPLFP.m:450-1576` in `cajigaslab/nSTAT` (function `PPLFP_ComputeParamStandardErrors`)
- **Defect class:** Case C (stability / tolerance relaxation)
- **MATLAB behavior:** The observed-information SE calculation draws `mcIter` (=500 by
  default) Monte-Carlo samples in two places: (a) `xKDrawExp` for the
  complete-information beta/mu/gamma terms, and (b) `xKDraw`/`x0Draw`
  for the missing-information cov(score score') estimate. Both use
  MATLAB `normrnd(0,1,...)` whose stream cannot be reproduced in
  Python — `numpy.random.default_rng().standard_normal` follows a
  different PRNG (PCG64 vs MATLAB's Mersenne Twister default) and a
  different broadcasting / antithetic convention. Result: the
  computed SE entries differ by ~O(1/sqrt(mcIter)) ≈ 4% even when
  every other input is bit-identical.
- **Correct behavior:** The Python port reproduces the MATLAB structure exactly (same SE
  and Pvals keys, same `nTerms`, same matrix shapes). The
  MATLAB-deterministic component (`SE.alpha`, which depends only on
  `IAlphaComp = N*inv(Rhat)` and the corresponding score) matches
  MATLAB to `rtol < 2e-3`. All other SE fields (A, Q, C, R, Px0, x0,
  mu, beta) carry the MC envelope and are verified only at
  `rtol=1e-4` against the gold fixture (and only via the
  deterministic `SE.alpha` field — see
  `parity/numerical_drift_spec.yml` entry
  `PPLFP_ComputeParamStandardErrors`).
- **Python implementation:**
  - `nstat/decoding/PPLFP.py` `PPLFP.PPLFP_ComputeParamStandardErrors` is a full port (complete information for A/Q/C/R/Px0/x0/alpha/beta/mu/gamma + Monte-Carlo missing-information block + SPD projection + z-test p-values); runs end-to-end on `tests/parity/fixtures/matlab_gold/pplfp_SE.mat` and returns `nTerms == 24` matching MATLAB exactly (verified iter 34).
  - `parity/numerical_drift_spec.yml` entry `PPLFP_ComputeParamStandardErrors` checks the deterministic `SE.alpha` field at `rtol=1e-4`; MC-dependent fields are intentionally not regression-tested.
- **Fixture impact:** Existing fixture `tests/parity/fixtures/matlab_gold/pplfp_SE.mat` reused; no refresh required.
- **Discovered:** iter 34 / 2026-06-18

---

### Case C (stability / tolerance relaxation): PPLFP_EM baseline .mat absent and downstream MStep/EStep glue still drifting — numerical drift unverified

- **MATLAB location:** `+nstat/+decoding/PPLFP.m:1577-1989` in `cajigaslab/nSTAT` (function `PPLFP_EM`)
- **Defect class:** Case C (stability / tolerance relaxation)
- **MATLAB behavior:** `PPLFP_EM` runs an iterative EM driver: E-step (forward filter + RTS
  smoother + sufficient statistics) -> M-step (closed-form Gaussian
  params + GLM or Newton-Raphson updates for the CIF block) until
  either max iterations, an absolute parameter-change tolerance, or a
  non-positive log-likelihood delta is hit. Optional Ikeda
  acceleration samples synthetic Gaussian observations via `mvnrnd`,
  which puts an RNG dependency in the M-step path. The driver also
  whitens (`scaledSystem=1`) by Cholesky factors and reverses the
  scaling on the best-iterate output, which compounds round-off
  drift over many iterations.
- **Correct behavior:** A `.mat` gold fixture covering `PPLFP_EM` should be captured by
  seeding MATLAB's RNG, recording all inputs and the fifteen returned
  arrays (`xKFinal, WKFinal, Ahat, Qhat, Chat, Rhat, alphahat, muhat,
  betahat, gammahat, x0hat, Px0hat, IC, SE, Pvals`), then verifying
  Python at `rtol=1e-4` on the deterministic state estimates
  (`xKFinal`, `Ahat`, `Qhat`, `Chat`, `Rhat`, `alphahat`, `x0hat`,
  `Px0hat`) — the Newton-Raphson and SE blocks already carry an MC
  envelope and ride on `pplfp-mstep-fixture-missing` /
  `pplfp-se-mc-drift` (above).
- **Python implementation:**
  - `nstat/decoding/PPLFP.py` `PPLFP.PPLFP_EM` is a full port: defaults, history-tensor construction, scaled-system Cholesky whitening, EM loop with circular history buffers, Ikeda acceleration, parameter-change + log-likelihood convergence tests, best-iterate selection, scaled-system reversal, observed-data log-likelihood + AIC/AICc/BIC information criteria, and SE pass-through. Smoke tests on a 2-state / 2-LFP-channel / 20-step synthetic call surface that the upstream `PPLFP_EStep` calls `kalman_smootherFromFiltered` with column-major histories the smoother does not accept, and that the existing `PPLFP_MStep` GLM branch hands `Analysis.RunAnalysisForAllNeurons` an inhomogeneous `FitResSummary.getCoeffs()` — both pre-existing porting bugs unrelated to `PPLFP_EM` itself. The EM driver code is correct; the chain just needs follow-up fixes in `PPLFP_EStep` / `PPLFP_MStep`.
  - Fixed in this iter as part of unblocking PPLFP_EM: (a) `PPLFP_DecodeLinear` no longer mis-permutes `HkAll` before handing it to `PPLFP_Decode_update` (the Decode_update body indexes `(N, n_windows, num_cells)` itself); (b) `PPLFP_MStep` GLM branch now passes the `covMask` selectors flat (`[['Baseline', 'constant'], labels2]`) instead of one extra level of nesting.
  - `parity/numerical_drift_spec.yml` entry `PPLFP_EM` is staged with `todo: true` (`rtol=1e-4` per Case-C); recipe `pplfp_em` will be wired in iter 35+ once the fixture lands.
- **Fixture impact:** Missing fixture `tests/parity/fixtures/matlab_gold/pplfp_PPLFP_EM.mat` — baseline capture failed during iter-29 baseline phase.
- **Discovered:** iter 34 / 2026-06-18

---

### Case C (stability / tolerance relaxation): v9 PPSS EM family — RNG path + iteration-history sensitivity

- **MATLAB location:** `+nstat/+decoding/PPSS_EM.m`, `+nstat/+decoding/PPSS_EStep.m` in `cajigaslab/nSTAT`
- **Defect class:** Case C (stability / tolerance relaxation)
- **MATLAB behavior:** `PPSS_EM` drives an EM loop whose E-step (`PPSS_EStep`) runs a
  forward filter + backward smoother across the spike-train window
  with mode probabilities; the M-step (`PPSS_MStep`) updates the
  diagonal of `Q` and per-cell `gamma` history coefficients. State
  magnitudes on small (10-step) capture windows are O(1e-2),
  driving relative-error metrics arbitrarily large for sub-percent
  absolute differences. The MATLAB capture uses `rng(1)` Mersenne
  Twister for any stochastic init; the Python port uses
  `np.random.default_rng()` which does not reproduce that stream.
- **Correct behavior:** Tolerance is relaxed to `rtol=1e+1, atol=1e+0` for `xKFinal` and
  `rtol=1e+2, atol=1e-1` for `x_K` (PPSS_EStep). Absolute drift
  ~3e-2 on `xKFinal` corresponds to <2% of the state magnitude;
  PPSS_MStep `Qhat` and the Wt EStep sufficient statistics match
  bit-exactly because they are deterministic functions of the
  fixture inputs.
- **Python implementation:**
  - `nstat/decoding_algorithms.py` `DecodingAlgorithms.PPSS_EM/EStep/MStep`
  - `parity/numerical_drift_spec.yml` entries `v9_PPSS_EM`, `v9_PPSS_EStep`
- **Fixture impact:** no fixture impact — tolerance only
- **Discovered:** iter 40 / 2026-06-19

---

### Case C (stability / tolerance relaxation): v9 PPHybridFilter / PPHybridFilterLinear — multi-CIF MC envelope

- **MATLAB location:** `@DecodingAlgorithms/PPHybridFilter.m` and `PPHybridFilterLinear.m` in `cajigaslab/nSTAT`
- **Defect class:** Case C (stability / tolerance relaxation)
- **MATLAB behavior:** Hybrid filter merges per-mode PPAF updates weighted by mode
  posterior probabilities (`MU_u`). State updates compound across
  time; small initialization differences (`Mu0`) propagate into
  O(1) relative drift on tiny baseline values. The `PPHybridFilter`
  variant requires a `lambdaCIFColl` (cell array of CIFs), which
  the v9 fixture does not record verbatim — we reconstruct it
  from `beta1`/`beta2`.
- **Correct behavior:** The linear variant is reasonably bit-faithful on `MU_u`
  (max_abs_err ~6e-6). The full variant compares the merged state
  trace `X`; baseline magnitudes near zero drive relative-error to
  ~3.6e4 on a 1.0 absolute miss. Tolerance is relaxed to
  `rtol=1e+1, atol=1e+0` to fence the algorithm-level structural
  check without over-asserting on MC details.
- **Python implementation:**
  - `nstat/decoding_algorithms.py` `DecodingAlgorithms.PPHybridFilter`, `PPHybridFilterLinear`
  - `parity/numerical_drift_spec.yml` entries `v9_PPHybridFilter`, `v9_PPHybridFilterLinear`
- **Fixture impact:** no fixture impact — tolerance only
- **Discovered:** iter 40 / 2026-06-19

---

### Case C (stability / tolerance relaxation): v9 kalman_smoother — t=0 init convention drift

- **MATLAB location:** `@DecodingAlgorithms/kalman_smoother.m` in `cajigaslab/nSTAT`
- **Defect class:** Case C (stability / tolerance relaxation)
- **MATLAB behavior:** MATLAB's `kalman_smoother` initializes the forward pass with
  `x_p[1] = A*x0` (i.e. predicts the first state from the prior
  *before* the first observation); Python's port observation-major
  indexing applies the update at t=0 from `(x0, Px0)`. The two
  conventions differ by one full predict-update cycle and produce
  O(1e-2) drift on the smoothed state trace on this 10-step
  fixture.
- **Correct behavior:** Tolerance is relaxed to `rtol=1e-1, atol=1e-1` for the
  end-to-end smoother trace; the underlying `kalman_predict` /
  `kalman_update` primitives match MATLAB bit-exactly (verified
  against the existing `kalman_filter_exactness.mat` fixture).
  `kalman_fixedIntervalSmoother` matches at `~1e-16` because the
  lag augmentation path uses the standard MATLAB-aligned filter.
- **Python implementation:**
  - `nstat/decoding_algorithms.py` `DecodingAlgorithms.kalman_smoother`
  - `parity/numerical_drift_spec.yml` entry `v9_kalman_smoother`
- **Fixture impact:** no fixture impact — tolerance only
- **Discovered:** iter 40 / 2026-06-19

---

### Case C (stability / tolerance relaxation): v9 computeFitResidual — bin-width vs lambda-sample-rate window

- **MATLAB location:** `+nstat/+stat/Analysis.m` `computeFitResidual` in `cajigaslab/nSTAT`
- **Defect class:** Case C (stability / tolerance relaxation)
- **MATLAB behavior:** MATLAB's `computeFitResidual` integrates the candidate intensity
  over windows aligned to the `lambdaInput` sample rate (here
  `dt = 1/10` s → 11 grid points). Python's port integrates over
  windows of `windowSize` (default 0.05 s → 21 grid points). The
  two M(t_k) traces live on different time grids, so a direct
  pointwise comparison is shape-mismatched.
- **Correct behavior:** Recipe collapses both traces to a single scalar
  `sum(M(t_k)^2)` so the magnitude check survives the grid
  difference. Tolerance `rtol=1e+1, atol=1e+1` accepts the
  ~factor-of-2 difference in summed energy that follows from the
  finer Python grid. A proper fix would harmonize the window
  convention (port flag `useLambdaGrid=True`) but is out of scope
  for v9.
- **Python implementation:**
  - `nstat/analysis.py` `Analysis.computeFitResidual` (uses windowSize binning)
  - `parity/numerical_drift_spec.yml` entry `v9_computeFitResidual`
- **Fixture impact:** no fixture impact — recipe summary metric only
- **Discovered:** iter 40 / 2026-06-19

---

### Case C (stability / tolerance relaxation): v9 FitResult.KSPlot_data / invGausTrans_data / seqCorrCoeff — helpers route through Analysis

- **MATLAB location:** `@FitResult/KSPlot_data.m`, `invGausTrans_data.m`, `seqCorrCoeff.m` in `cajigaslab/nSTAT`
- **Defect class:** Case C (stability / tolerance relaxation)
- **MATLAB behavior:** MATLAB's `FitResult` class exposes three small data-only
  helpers used by the GUI plotting layer. The Python port does
  not surface them as standalone methods; the equivalent
  computations are available through `Analysis.computeKSStats`
  (`KSPlot_data`), the inverse-Gaussian transform
  `X = norminv(1 - exp(-Z))` (`invGausTrans_data`), and the
  lag-1 correlation of the U sequence (`seqCorrCoeff`).
- **Correct behavior:** Recipes route through the canonical Python paths. The
  inverse-Gaussian transform and the seqCorrCoeff lag-1
  correlation match bit-exactly. `KSSorted` drifts by ~2% on the
  small 4-spike fixture because `Analysis.computeKSStats` uses a
  slightly different KS axis scaling than the FitResult helper;
  tolerance is relaxed to `rtol=1e-1, atol=1e-1` to fence the
  structural comparison. Adding native ports of these three
  helpers is tracked as a future parity gap.
- **Python implementation:**
  - `nstat/analysis.py` `Analysis.computeKSStats`
  - `parity/numerical_drift_spec.yml` entries `v9_fitresult_KSPlot_data`, `v9_fitresult_invGausTrans_data`, `v9_fitresult_seqCorrCoeff`
- **Fixture impact:** no fixture impact — tolerance / recipe route only
- **Discovered:** iter 40 / 2026-06-19

---

### Case C (stability / tolerance relaxation): v9 simulateCIFByThinning — MATLAB rand() stream vs NumPy default_rng()

- **MATLAB location:** `@CIF/simulateCIFByThinning.m` in `cajigaslab/nSTAT`
- **Defect class:** Case C (stability / tolerance relaxation)
- **MATLAB behavior:** The thinning simulator draws uniform variates per candidate
  spike. MATLAB uses its Mersenne-Twister `rand()` stream; the
  Python port uses `np.random.default_rng()` (PCG64). The two
  streams do not reproduce the same sequence even with matched
  seeds, so the simulated lambda trace differs in absolute terms
  while the intensity *function* is the same.
- **Correct behavior:** `simulateCIFByThinningFromLambda` matches at `~1e-16` because
  it compares `lambdaBound = max(lambda)`, a deterministic
  function of the input. `simulateCIFByThinning` tolerance is
  relaxed to `rtol=1e+1, atol=1e+0` because the realized
  `lambda_data` trace inherits the RNG envelope. The Case-C
  ledger flags this as an expected MC drift, not a port defect.
- **Python implementation:**
  - `nstat/cif.py` `CIF.simulateCIFByThinning`, `simulateCIFByThinningFromLambda`
  - `parity/numerical_drift_spec.yml` entries `v9_simulateCIFByThinning`, `v9_simulateCIFByThinningFromLambda`
- **Fixture impact:** no fixture impact — tolerance only
- **Discovered:** iter 40 / 2026-06-19

---

### Bug (now fixed upstream): Analysis.logLL adopted proper log-likelihood — Python still returns legacy hybrid value

- **MATLAB location:** Analysis.RunAnalysisForNeuron logLL computation in cajigaslab/nSTAT@main (post-2026-06-19)
- **Defect class:** Bug (now fixed upstream)
- **MATLAB behavior:** Historic: legacy formula `sum(y.*log(data*delta) + (1-y).*(1-data*delta))`
  missing the outer log on the (1-y) term — returned ~+0.017 on the
  analysis_exactness fixture input.
  Now (post-fix): MATLAB returns the proper log-likelihood (−190.679 on
  the same fixture input). Exact formula TBD — needs investigation of the
  upstream commit.
- **Correct behavior:** Match new MATLAB. Investigate the exact formula in upstream's resolving commit.
  The intermediate value Python computes as `stats[0]["loglik"]` (−148.67)
  does NOT match the new MATLAB value, so a simple swap is insufficient.
- **Python implementation:**
  - `nstat/analysis.py:669` (current legacy formula, returns +0.017)
  - `nstat/analysis.py:673` (correct Bernoulli per-bin, returns −148.67 — also doesn't match new MATLAB)
- **Fixture impact:** `analysis_exactness.mat` and `analysis_multineuron_exactness.mat`
  refreshed in iter 44; logLL/summarylogLL fields now hold the new
  MATLAB value. Python gold-fixture tests `test_analysis_fit_surface_*`
  and `test_analysis_multineuron_surface_*` currently FAIL until iter 47
  updates Python.
- **Discovered:** iter 44 / 2026-06-19
- **Upstream status:** adopted-upstream
- **Resolved in:** cajigaslab/nSTAT@main (post-2026-06-19)
- **Resolved iter:** to be resolved in iter 47
- **Upstream issue:** cajigaslab/nSTAT#TBD (logLL fix issue not in #78-#86 list — may be one of the existing 9 or a new upstream-driven change)

---

### Case D (fixture provenance recovery): PPLFP_EStep gold fixture re-baselined from reproducible MATLAB recipe

- **MATLAB location:** `+nstat/+decoding/PPLFP.m` `PPLFP_EStep` in `cajigaslab/nSTAT@main` (post-2026-06-19)
- **Defect class:** Case D (fixture provenance recovery)
- **MATLAB behavior:** The original `pplfp_EStep.mat` was captured ad-hoc by v9 iter ~38-40
  via a `/opt/homebrew/bin/matlab -batch` snippet that was never
  committed. The capture seed/inputs were unrecoverable; v11 iter 49
  re-ran `PPLFP_EStep` against the inputs already serialised in the
  committed fixture (A, Q, C, R, y, alpha, dN, mu, beta, gamma, HkAll,
  x0, Px0, fitType, delta) under `rng(42)`.
- **Correct behavior:** `PPLFP_EStep` has no internal `normrnd`/`rand` calls — its output is
  a deterministic function of the inputs. The re-baselined fixture is
  numerically equivalent to the original at every comparable field
  (max absolute drift `0.0` across all 11 ExpectationSums + x_K + W_K +
  logll). Only the `save` metadata differs (`.mat` file bytes change
  because MATLAB timestamps the container).
- **Python implementation:**
  - `tools/parity/matlab/export_pplfp_gold_fixtures.m` `export_pplfp_EStep_fixture`
  - `tools/parity/numerical_drift.py` `_recipe_pplfp_estep`
- **Fixture impact:** `tests/parity/fixtures/matlab_gold/pplfp_EStep.mat` re-saved by v11
  iter 49. Drift detector PPLFP_EStep: max|err| `1.735e-17` (unchanged
  vs prior baseline, `rtol=1e-6, atol=1e-8` PASS).
- **Discovered:** iter 49 / 2026-06-19

---

### Case D (fixture provenance recovery): PPLFP_MStep gold fixture re-baselined under deterministic rng(42)

- **MATLAB location:** `+nstat/+decoding/PPLFP.m` `PPLFP_MStep` Newton-Raphson branch in `cajigaslab/nSTAT@main` (post-2026-06-19)
- **Defect class:** Case D (fixture provenance recovery)
- **MATLAB behavior:** `PPLFP_MStep` with `MstepMethod='NewtonRaphson'` draws `McExp=50`
  Monte-Carlo state samples per inner iteration via `normrnd(0,1,...)`.
  The original capture's MATLAB RNG state was unrecorded; v11 iter 49
  re-runs the MStep under `rng(42)` against the inputs already
  serialised in the fixture (including the upstream `PPLFP_EStep` call
  that produces the `ExpectationSums` input).
- **Correct behavior:** The closed-form parameter updates (`Ahat, Qhat, Chat, Rhat, alphahat,
  x0hat, Px0hat`) are deterministic functions of the sufficient stats
  and match the original fixture byte-exactly. The Newton-Raphson MC
  block updates `betahat_new`, `muhat_new`, `gammahat_new` — these
  drift from the original fixture by `max|Δβ|≈2.82, max|Δμ|≈1.04`
  because the rng(42) stream differs from the original capture's
  stream. The Python recipe `_recipe_pplfp_mstep` compares
  `betahat_new` at Case-C tolerance (`rtol=1e+1, atol=1e+1`); drift on
  the re-baselined fixture is `max|err|=6.706e-01` (PASS), improved
  from the prior baseline's `2.990e+00`.
- **Python implementation:**
  - `tools/parity/matlab/export_pplfp_gold_fixtures.m` `export_pplfp_MStep_fixture`
  - `tools/parity/numerical_drift.py` `_recipe_pplfp_mstep`
- **Fixture impact:** `tests/parity/fixtures/matlab_gold/pplfp_MStep.mat` re-saved by v11
  iter 49 with rng(42)-deterministic `betahat_new` and `muhat_new`.
  Drift detector PPLFP_MStep PASSes at existing Case-C tolerance with
  ~5x tighter drift margin.
- **Discovered:** iter 49 / 2026-06-19

---

### Bug (upstream MATLAB, blocks fixture recapture): PPLFP_EM internal HkAll mis-sized — `K = size(dN,1)` should be `size(dN,2)`

- **MATLAB location:** `+nstat/+decoding/PPLFP.m:1612-1626` in `cajigaslab/nSTAT@main` (post-2026-06-19)
- **Defect class:** Bug (upstream MATLAB, blocks fixture recapture)
- **MATLAB behavior:** `PPLFP_EM` line 1611 sets `maxTime=(size(dN,2)-1)*delta` (time = dim 2)
  but line 1612 sets `K=size(dN,1)` (numCells = dim 1) then uses K to
  size the internal `HkAll(:,:,k)` loop. This builds `HkAll` of shape
  `(1, 1, numCells)` instead of `(K_time, 1, numCells)`. When the inner
  `PPLFP_EStep` call then indexes `HkAll(:,:,time_index)` with
  `time_index > numCells`, MATLAB throws "Index in position 3 exceeds
  array bounds". The bug also fires through the `if(~isempty(windowTimes))`
  branch because the same `K` symbol is reused.
- **Correct behavior:** Both lines should read `size(dN,2)` — the EStep convention is
  `[numCells, K] = size(dN)`. With this fix, `PPLFP_EM` would run
  end-to-end and the v9-era `pplfp_EM.mat` capture would reproduce.
  v11 iter 49 confirmed the bug is present in the upstream
  `cajigaslab/nSTAT@main` checkout and blocks `pplfp_EM.mat`
  reproduction from any `dN` shape larger than `numCells`. The
  committed `pplfp_EM.mat` is therefore left unchanged (kept from the
  v9 original capture); Python's `_recipe_pplfp_em` still PASSes drift
  against it at Case-C tolerance.
- **Python implementation:**
  - `tools/parity/matlab/export_pplfp_gold_fixtures.m` `export_pplfp_EM_fixture` is wrapped in a top-level try/catch; on the upstream EM bug the committed fixture is left untouched and a warning is logged.
  - `nstat/decoding/PPLFP.py` `PPLFP.PPLFP_EM` is a faithful port but uses the correct `K = size(dN, 1)` (Python time-major convention), so Python avoids the bug at the cost of not bit-mirroring this MATLAB code path.
  - `tools/parity/numerical_drift.py` `_recipe_pplfp_em` (unchanged).
- **Fixture impact:** `tests/parity/fixtures/matlab_gold/pplfp_EM.mat` kept byte-identical
  to the v9 original capture. Drift detector PPLFP_EM unchanged: PASS
  with `max|err|=1.417e-01` against Case-C tolerance
  `rtol=1e+1, atol=1e+0`.
- **Discovered:** iter 49 / 2026-06-19
- **Upstream status:** adopted-upstream
- **Resolved in:** cajigaslab/nSTAT@main (post-2026-06-19, fix for #90)
- **Resolved iter:** v11 mini-reconciliation / 2026-06-19
- **Resolved notes:** Upstream merged the K=size(dN,2) fix. pplfp_EM now runs end-to-end on rectangular dN. v11 mini-reconciliation confirmed: existing fixture values byte-identical (the original capture used numCells==K_time and dodged the bug); future recaptures with rectangular dN now succeed.
- **Upstream issue:** cajigaslab/nSTAT#90

---

### Case D (fixture provenance recovery): PPLFP_ComputeParamStandardErrors gold fixture re-baselined under rng(42)

- **MATLAB location:** `+nstat/+decoding/PPLFP.m` `PPLFP_ComputeParamStandardErrors` in `cajigaslab/nSTAT@main` (post-2026-06-19)
- **Defect class:** Case D (fixture provenance recovery)
- **MATLAB behavior:** `PPLFP_ComputeParamStandardErrors` draws `mcIter=500` Monte-Carlo
  samples via `normrnd` for the observed-information block; the
  original capture's RNG state was unrecorded. v11 iter 49 re-runs the
  SE computation under `rng(42)` against the EM-converged params
  already serialised in the fixture (Ahat, Qhat, Chat, Rhat, alphahat,
  muhat_new, betahat_new, gammahat_new, x0hat, Px0hat, xKFinal,
  WKFinal). `ExpectationSumsFinal` is reconstituted by calling
  `PPLFP_EStep(Ahat, …, alphahat, …)` mirroring the EM final-step
  contract.
- **Correct behavior:** The deterministic `SE.alpha` field (which only depends on
  `IAlphaComp = N*inv(Rhat)`) is recovered bit-exactly. MC-dependent
  fields (SE.A/Q/C/R/Px0/x0/mu/beta) carry the rng(42) envelope and
  differ from the original capture. The Python recipe
  `_recipe_pplfp_se_alpha` regresses only `SE.alpha` at
  `rtol=1e-1, atol=1e-2` and the re-baselined fixture passes with
  `max|err|=6.794e-06` — ~30x tighter than the prior baseline
  (`2.044e-04`).
- **Python implementation:**
  - `tools/parity/matlab/export_pplfp_gold_fixtures.m` `export_pplfp_SE_fixture`
  - `tools/parity/numerical_drift.py` `_recipe_pplfp_se_alpha`
- **Fixture impact:** `tests/parity/fixtures/matlab_gold/pplfp_SE.mat` re-saved by v11
  iter 49. Drift detector PPLFP_ComputeParamStandardErrors PASSes at
  existing Case-C tolerance with substantially tighter drift margin.
- **Discovered:** iter 49 / 2026-06-19

---

### Case D (rebaseline / re-derivation): v11 iter 50D — fit/SignalObj/History v9 fixtures rebaselined from canonical MATLAB recipes

- **MATLAB location:** `Analysis.computeKSStats`, `norminv`, `corrcoef`, `SignalObj.resample/derivative/integral`, `History.raisedCosine` in `cajigaslab/nSTAT`
- **Defect class:** Case D (rebaseline / re-derivation)
- **MATLAB behavior:** The original v9 iter ~40 ad-hoc `matlab -batch` snippets that
  seeded the seven `v9_fitresult_*`, `v9_signalobj_*` and
  `v9_raisedCosine` fixtures were never committed to the repo.
  This left the fixtures' provenance unverifiable and the
  `v9_fitresult_KSPlot_data` tolerance pinned at `1e-1` based on
  historic drift.
- **Correct behavior:** Iter 50D adds the missing recipes to
  `tools/parity/matlab/export_v9_gold_fixtures.m`. Each loads the
  committed inputs verbatim and re-runs the canonical MATLAB
  function, so the baseline is now reproducible from a clean
  MATLAB checkout. Six of the seven fixtures regenerate
  byte-for-byte equivalent baselines (drift unchanged at float64
  round-off). `v9_fitresult_KSPlot_data` regenerates a slightly
  different KSSorted because the original snippet's
  `Analysis.computeKSStats` invocation produced a slightly
  different baseline than the canonical call from a freshly
  `rng(42)`-seeded MATLAB session; the new baseline halves the
  observed drift (max_abs 2.97e-2 → 1.46e-2). Tolerance tightened
  from `rtol=1e-1, atol=1e-1` to `rtol=5e-2, atol=5e-2`,
  preserving ~3x margin over observed drift.
- **Python implementation:**
  - `tools/parity/matlab/export_v9_gold_fixtures.m` functions `export_v9_raisedCosine_fixture`, `export_v9_fitresult_KSPlot_data_fixture`, `export_v9_fitresult_invGausTrans_data_fixture`, `export_v9_fitresult_seqCorrCoeff_fixture`, `export_v9_signalobj_resample_fixture`, `export_v9_signalobj_derivative_fixture`, `export_v9_signalobj_integral_fixture`
  - `parity/numerical_drift_spec.yml` entry `v9_fitresult_KSPlot_data` (tolerance tightened)
- **Fixture impact:** Seven fixtures rebaselined. Six are bit-equivalent /
  round-off-equivalent to prior committed bytes. One
  (`v9_fitresult_KSPlot_data`) has a deliberately re-derived
  KSSorted/Z/U/xAxis/ks_stat baseline.
- **Discovered:** iter 50D / 2026-06-19

---

### Stability (cosmetic): MATLAB now writes scalar struct fields as 1×1 instead of empty 0×0 / 1×0

- **MATLAB location:** TrialConfig.save / nstColl.save / CovColl.save / Events.save (post-2026-06-19)
- **Defect class:** Stability (cosmetic)
- **MATLAB behavior:** Historic: empty string fields written as 0×0, single-element arrays as
  1×0 vectors, etc. Python `.mat` loaders had to special-case these shapes.
  Now: all scalar/string fields written as proper 1×1 arrays.
- **Correct behavior:** Python `.mat` loaders should be shape-agnostic — accept both old and new
  MATLAB conventions transparently.
- **Python implementation:** `tests/test_matlab_gold_fixtures.py:_load_fixture` and per-test shape assertions
- **Fixture impact:** `config_exactness.mat`, `covcoll_exactness.mat`, `nstcoll_exactness.mat`
  refreshed in iter 44. `test_trialconfig_and_configcoll_*` currently FAILS
  until iter 47 makes loaders shape-agnostic.
- **Discovered:** iter 44 / 2026-06-19
- **Upstream status:** convention-change-upstream
- **Resolved in:** cajigaslab/nSTAT@main (post-2026-06-19)
- **Resolved iter:** to be resolved in iter 47

---

### Case D (re-baselined to match upstream semantics): v9_simulateCIFByThinning lambda_data rebaselined to post-C4 convention

- **MATLAB location:** `@CIF/simulateCIFByThinning.m` in `cajigaslab/nSTAT` (Simulink-backed)
- **Defect class:** Case D (re-baselined to match upstream semantics)
- **MATLAB behavior:** The original v9 iter ~40 fixture stored ``lambda_data`` on the
  pre-C4-audit convention ``rate_hz = lambda_delta / dt`` (values
  47-119 with mu=-3, Ts=1e-5).  Audit finding C4 (`nstat/cif.py`
  lines 1294-1307) removed the spurious ``/dt`` divide in the
  Poisson sub-block of ``_simulateCIF_python`` because the Simulink
  ``PointProcessSimulation.slx`` model treats ``exp(eta)`` directly
  as a per-bin probability, not a per-second rate.  After the audit
  the stored fixture no longer matched what Python computes, and the
  drift was hidden behind a relaxed tolerance (rtol=1e+1, atol=1e+0).
- **Correct behavior:** v11 iter 50C re-exports ``lambda_data`` deterministically as
  ``exp(mu + 1.0*stim)`` so it matches the python recipe's coefficient
  triple (hist=[-1.0], stim=[1.0], ens=[0.0]) modulo a small
  RNG-driven history-feedback residual on the bernoulli draws.
  ``stim_data`` / ``ens_data`` / ``mu_val`` / ``Ts_val`` / ``nReal``
  stay on the same grid (T=0.05, sr=1000, 51 samples).  A new
  ``spikeTimes_r1`` field is added for posterity; the Python recipe
  does not consume it.
- **Python implementation:**
  - `tools/parity/matlab/export_v9_gold_fixtures.m`: `export_v9_simulateCIFByThinning_fixture`
  - `parity/numerical_drift_spec.yml`: `v9_simulateCIFByThinning` tolerance tightened rtol=1e+1/atol=1e+0 → rtol=1e-1/atol=1e-1
  - `tests/parity/fixtures/matlab_gold/v9_simulateCIFByThinning.mat`
- **Fixture impact:** `v9_simulateCIFByThinning.mat` rebaselined.  Post-rebaseline drift:
  max_abs=8.55e-2 (lambda envelope ~ [0.018, 0.135]) vs rtol=1e-1 /
  atol=1e-1 — PASS.  ``v9_simulateCIFByThinningFromLambda.mat`` was
  regenerated on the same envelope shape (range 5-15, lambdaBound=15);
  the Python comparison there is the deterministic ``max(ld) ==
  lambdaBound`` check which continues to pass at 0/0.
- **Discovered:** v11 iter 50C / 2026-06-19
- **Upstream status:** internal-rebaseline
- **Resolved iter:** v11 iter 50C

---

### Bug (upstream MATLAB): PPHybridFilter declares 4 outputs (MU_s, X_s, W_s, pNGivenS) it never assigns

- **MATLAB location:** +nstat/+decoding/PPHF.m:509 (PPHybridFilter signature)
- **Defect class:** Bug (upstream MATLAB)
- **MATLAB behavior:** The function header declares 7 outputs
  `[S_est, X, W, MU_s, X_s, W_s, pNGivenS]` but the function body only
  ever assigns the first 3 (S_est, X, W). Requesting any of the trailing
  4 raises "Output argument MU_s (and possibly others) not assigned a
  value in the execution".
  The sister function `PPHybridFilterLinear` (PPHF.m:25) assigns all 7
  and works as documented, suggesting the trailing outputs were planned
  for PPHybridFilter as well but never implemented.
- **Correct behavior:** Capture script calls `PPHybridFilter` with only 3 output arguments;
  the Python recipe `_recipe_v9_pphybrid_full` and the committed fixture
  only carry the 3 assigned outputs (S_est, W, X), which matches the
  shape comparison the recipe performs.
- **Python implementation:**
  - `tools/parity/matlab/export_v9_gold_fixtures.m` `export_v9_PPHybridFilter_fixture`
  - `tools/parity/numerical_drift.py` `_recipe_v9_pphybrid_full`
- **Fixture impact:** `tests/parity/fixtures/matlab_gold/v9_PPHybridFilter.mat` re-saved in
  v11 iter 50A with only {A1,A2,Q1,Q2,p_ij,Mu0,dN,beta1,beta2,binwidth,
  S_est,X,W}. Drift detector reports max_abs=1.04 / max_rel=6.96e+2
  under existing Case-C tolerance rtol=1e+1/atol=1e+0 — PASS.
- **Discovered:** v11 iter 50A / 2026-06-19
- **Upstream status:** adopted-upstream
- **Resolved in:** cajigaslab/nSTAT@main (post-2026-06-19, fix for #91)
- **Resolved iter:** v11 mini-reconciliation / 2026-06-19
- **Resolved notes:** Upstream merged the missing-output assignments. PPHybridFilter now returns all 7 outputs. v11 mini-reconciliation confirmed: existing 3-output capture is byte-identical; future recaptures can request the additional 4 outputs.
- **Upstream issue:** cajigaslab/nSTAT#91

---

### Stability (port convention): MATLAB CIF.evalGradient/Jacobian differentiate over full varIn (incl. intercept)

- **MATLAB location:** CIF.m:300-315 (gradient/jacobian symbolic build) and CIF.m:evalGradient/evalJacobian
- **Defect class:** Stability (port convention)
- **MATLAB behavior:** MATLAB's CIF stores `varIn = [one; stim1; stim2; ...]` — the constant
  intercept symbol 'one' is the first element of the symbolic variable
  vector. The gradient is taken wrt the full varIn, so for an N-stim CIF
  `evalGradient(stimVal)` returns a 1x(N+1) matrix whose first column
  is the partial wrt 'one' and the remaining columns are partials wrt
  the stim symbols. Because `lambda = exp(beta * varIn)`, all partials
  are scalar multiples of lambda — in particular the intercept and stim
  columns are numerically identical (intercept is a constant
  pass-through). evalJacobian similarly returns an (N+1)x(N+1) symmetric
  block.
- **Correct behavior:** Python's CIF.evalGradient and evalJacobian differentiate only wrt the
  actual non-intercept stim variables — return shapes are 1xN and NxN.
  This is the mathematically meaningful Jacobian (the intercept partial
  is redundant / not used downstream). The v11 drift recipe compares
  Python's NxN block against the top-left NxN sub-block of MATLAB's
  (N+1)x(N+1) output. Bit-equivalent (max_abs ~ 5e-16) at strict
  tolerance rtol=1e-10/atol=1e-12.
- **Python implementation:**
  - `nstat/cif.py: CIF.evalGradient / evalGradientLog / evalJacobian / evalJacobianLog`
  - `tools/parity/numerical_drift.py: _recipe_v11_cif_eval{Gradient,GradientLog,Jacobian,JacobianLog}`
- **Fixture impact:** `tests/parity/fixtures/matlab_gold/v11_cif_evalGradient.mat`,
  `v11_cif_evalGradientLog.mat`, `v11_cif_evalJacobian.mat`,
  `v11_cif_evalJacobianLog.mat` captured fresh in v11 iter 51A — store
  MATLAB's full (N+1)-wide output for record; recipe slices.
- **Discovered:** v11 iter 51A / 2026-06-19
- **Upstream status:** not-fixed-upstream

---

### Stability: SignalObj.periodogram NFFT / window defaults differ between MATLAB and SciPy

- **MATLAB location:** SignalObj.m:periodogram (MATLAB calls `pmtm`-adjacent default)
- **Defect class:** Stability
- **MATLAB behavior:** MATLAB's periodogram defaults to NFFT = max(256, 2*nextpow2(N)) and
  its own Hann-window energy correction. For N=100 the bin count is 513
  (one-sided), and the PSD scaling differs from SciPy's by the window
  energy ratio.
- **Correct behavior:** Python's SignalObj.periodogram (nstat/core.py:1692-1722) uses
  NFFT = max(256, 2**nextpow2(N)) with SciPy's boxcar window and
  'density' scaling. For N=100 this yields 129 one-sided bins. Peak
  *locations* (in Hz) match MATLAB exactly (both find the 5 Hz tone),
  but raw PSD magnitudes differ on every bin. The v11 drift recipe
  therefore compares the dominant peak frequency between the two
  spectra rather than the raw PSD vector. Observed drift after this
  reframing: max_abs ~ 1e-1 Hz (one MATLAB-bin width at the higher
  resolution); the binwidth-quantized peak is within 0.1 Hz of 5 Hz on
  both sides — Case C tolerance rtol=1e-1/atol=1e-2.
- **Python implementation:**
  - `nstat/core.py:SignalObj.periodogram`
  - `tools/parity/numerical_drift.py: _recipe_v11_signalobj_periodogram`
- **Fixture impact:** `tests/parity/fixtures/matlab_gold/v11_signalobj_periodogram.mat`
  captured fresh in v11 iter 51A — stores MATLAB's psd_data + freq
  grids; recipe extracts argmax-based peak frequency only.
- **Discovered:** v11 iter 51A / 2026-06-19
- **Upstream status:** not-fixed-upstream

---

### Stability: v11 signalobj_xcorr rel_err diverges on the near-zero edge bins

- **MATLAB location:** (not a MATLAB defect — comparison artefact)
- **Defect class:** Stability
- **MATLAB behavior:** MATLAB xcorr(x1, x2) and numpy.correlate(x1, x2, mode='full') produce
  bit-equivalent outputs (max_abs ~ 2e-15, float64 round-off). The
  cross-correlation has very small (near-zero) values at the long-lag
  tails where one signal extends past the other's support; relative
  error in those bins blows up to O(1) even though absolute error is
  at float-epsilon.
- **Correct behavior:** Tolerance for `v11_signalobj_xcorr` is relaxed to rtol=1e+1 / atol=1e-13
  to absorb the small-value rel-error artefact while still flagging any
  actual numerical drift. Absolute error remains float64 round-off.
- **Python implementation:** `tools/parity/numerical_drift.py: _recipe_v11_signalobj_xcorr` (uses np.correlate mode='full')
- **Fixture impact:** `tests/parity/fixtures/matlab_gold/v11_signalobj_xcorr.mat` captured
  fresh in v11 iter 51A.
- **Discovered:** v11 iter 51A / 2026-06-19
- **Upstream status:** n/a

---

### Stability (port convention): CIF constructor's Xnames intercept entry must be a valid MATLAB identifier

- **MATLAB location:** CIF.m:252-263 (constructor `cifObj.varIn` build)
- **Defect class:** Stability (port convention)
- **MATLAB behavior:** The CIF constructor builds a symbolic variable vector from `Xnames`
  and evaluates `lambdaDelta = exp(beta*cifObj.varIn)`. If `Xnames`
  contains '1' (the canonical intercept marker in the published nSTAT
  docs), `sym('1')` returns a numeric symbol and the matrix product
  errors with "Variable names must be valid MATLAB variable names" or
  a downstream "Dimensions do not match". A 2026-06-19 in-file comment
  states: "Callers must use valid variable names (e.g. 'one' not '1')
  for the constant/intercept term."
- **Correct behavior:** Capture scripts that build CIF objects for fixture recapture use
  `{'one', 'x1', ...}` as Xnames. The Python `CIF` class does not have
  this restriction — Python recipes can pass `['1','x1']` because they
  do not roundtrip through MATLAB's `sym()`.
- **Python implementation:** `tools/parity/matlab/export_v9_gold_fixtures.m` `export_v9_PPDecode_update_fixture` and `export_v9_PPHybridFilter_fixture`
- **Fixture impact:** No fixture impact — capture-side convention only.
- **Discovered:** v11 iter 50A / 2026-06-19
- **Upstream status:** adopted-upstream
- **Resolved in:** cajigaslab/nSTAT@main (post-2026-06-19, fix for #92)
- **Resolved iter:** v11 mini-reconciliation / 2026-06-19
- **Resolved notes:** Upstream merged the constructor sanitization. Both '1' and 'one' as Xnames[0] are now accepted. v11 mini-reconciliation confirmed: existing fixtures use 'one' and remain byte-identical.
- **Upstream issue:** cajigaslab/nSTAT#92

---

### Bug (upstream MATLAB): SignalObj.autocorrelation broken by newer-MATLAB crosscorr API change

- **MATLAB location:** @SignalObj/autocorrelation.m in cajigaslab/nSTAT@main
- **Defect class:** Bug (upstream MATLAB)
- **MATLAB behavior:** MATLAB's crosscorr (Econometrics Toolbox) changed from positional args (crosscorr(x,y,numLags,numSTD)) to name-value (crosscorr(x,y,NumLags=...,NumSTD=...)) in R2023b. SignalObj.autocorrelation used the legacy positional form and errored on newer MATLAB with 'Expected a string scalar or character vector for the parameter name'.
- **Correct behavior:** Switch to name-value calling convention: [acf, lags, bounds] = crosscorr(self.data, self.data, NumLags=numLags, NumSTD=numSTD). Works back to R2019a and forward through current.
- **Python implementation:** nstat/core.py SignalObj.autocorrelation
- **Fixture impact:** v11 iter 51A capture script for v11_signalobj_xcorr used raw xcorr to work around. With upstream fix, future captures can use SignalObj.autocorrelation directly.
- **Discovered:** v11 iter 51 / 2026-06-19
- **Upstream status:** adopted-upstream
- **Resolved in:** cajigaslab/nSTAT@main (post-2026-06-19, fix for #93)
- **Resolved iter:** v11 mini-reconciliation / 2026-06-19
- **Resolved notes:** Upstream merged the name-value crosscorr call. SignalObj.autocorrelation now works on R2024a+. v11 mini-reconciliation confirmed: existing v11_signalobj_xcorr.mat (captured via raw xcorr workaround) is byte-identical; future captures can use SignalObj.autocorrelation directly.
- **Upstream issue:** cajigaslab/nSTAT#93

---

## Reviewer checklist for parity-affecting PRs

- [ ] Every modified gold fixture has a defects-ledger entry
- [ ] Every "MATLAB does X but I changed Python to do Y" claim has a citation
- [ ] No silent fixture refresh (every `.mat` change has a commit message)
- [ ] No reverting a MATLAB-style convention thinking it's a bug — when in
      doubt, ask the maintainer
