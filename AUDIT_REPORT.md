# nSTAT Cross-Toolbox Audit Report

**Date:** 2026-03-10 (initial); refreshed 2026-06-11 with Phase 1-3 findings.
**Scope:** Full method-by-method comparison of Matlab nSTAT vs Python nSTAT-python
**Gold standard:** Matlab (cajigaslab/nSTAT)
**Python version audited:** v0.2.0 (initial) → v0.4.x (Phase 1-3 audit)

---

## Executive Summary

| Category | Count |
|---|---|
| **Matlab bugs found** (M1-M13 initial + M14-M21 Phase 1-3) | 21 |
| **Python bugs found** | 9 |
| **Missing Python methods (high priority)** | 5 |
| **Missing Python methods (medium priority)** | 28 |
| **Missing Python methods (low priority)** | 15+ |
| **Behavioral differences (documented)** | 18 |

---

## 1. Matlab Bugs

### 1.1 Already Filed (GitHub Issues)

| # | Class | Bug | Issue |
|---|---|---|---|
| M1 | SignalObj | `findPeaks` minima logic inverted | [#12](https://github.com/cajigaslab/nSTAT/issues/12) |
| M2 | SignalObj | `findGlobalPeak` typo `sortedData` vs `sortedPeaks` | [#13](https://github.com/cajigaslab/nSTAT/issues/13) |
| M3 | SignalObj | `shiftMe` returns value instead of mutating handle | [#14](https://github.com/cajigaslab/nSTAT/issues/14) |

### 1.2 New Bugs to File

| # | Class | File:Line | Description | Severity |
|---|---|---|---|---|
| M4 | Analysis | Analysis.m:1051 | `ensCovMaskTemp(neighbors(j), neuronNum)=0` zeroes ALL columns in Granger causality mask instead of just `neuronNum(i)` | HIGH |
| M5 | Analysis | Analysis.m:1063-1066 | `phiMat` via `strfind`/`~isempty` always indexes only the first coefficient in Granger causality | HIGH |
| M6 | Analysis | Analysis.m:246,364 | Typo `tObj.sampeRate` (missing 'l' in sampleRate) | MEDIUM |
| M7 | CovColl | CovColl.m:377-378 | `findMaxTime` applies `covShift` twice | MEDIUM |
| M8 | CovColl | CovColl.m (isCovPresent) | Off-by-one: `cov < numCov` should be `cov <= numCov` (also in Python) | LOW |
| M9 | TrialConfig | TrialConfig.m (fromStructure) | Omits `ensCovMask` parameter when reconstructing from struct | LOW |
| M10 | SignalObj | SignalObj.m | `autocorrelation` typo: uses `crosscor` variable name | LOW |
| M11 | SignalObj | SignalObj.m | `times`/`rdivide` operator aliasing bug | LOW |
| M12 | DecodingAlgorithms | DA.m:496-501,664 | Gamma broadcasting: loop variable `c` reused after loop exit; `gammaNew(:,c)` writes only to last column | MEDIUM |
| M13 | nstColl | nstColl.m:~1484 | `getSpikeTimes()`: `count` variable uninitialized when mask excludes neuron 1 | LOW |

### 1.3 New Bugs from 2026-06-11 Audit (Phase 1-3 session)

Found by the 6 parallel audit agents (core data classes, collections,
CIF/LinearCIF/History, Fitting/Analysis, DecodingAlgorithms, examples)
plus the live Simulink-model inspection that resolved C4.

| # | Class | File:Line | Description | Severity |
|---|---|---|---|---|
| M14 | SignalObj | SignalObj.m:1342-1353 | `resample` skips when `sampleRate` unchanged but `setMinTime`/`setMaxTime` can mutate the time window post-construction, making the equality check stale.  Length mismatches surface in `makeCompatible` chains. | MEDIUM |
| M15 | nstColl | nstColl.m:178-180 | `getFieldVal`: pre-increments `cnt` before `neuronNumbers(cnt) = ...`, so the index-array is off by one relative to `fieldVal`. | HIGH |
| M16 | nstColl | nstColl.m:397-399 | `getNSTnameFromInd`: bounds check `ind > 0 && nstCollObj.numSpikeTrains` lacks the `ind <= numSpikeTrains` clause — author forgot to write the comparison.  Out-of-bounds indices fall through silently. | MEDIUM |
| M17 | Events | Events.m:87 | (was already fixed in MATLAB; Python `events.py:84` regressed to hardcoded `'r'` instead of `self.eventColor`).  Cross-port consistency fix: Python should match MATLAB's corrected behavior. | LOW |
| M18 | DecodingAlgorithms | DA.m:479,546 | `estimateInfoMat`: `Ic(1:R,1:R) = ...` assigned twice; first formula at line 479 is silently overwritten at line 546.  Refactor leftover; harmless but misleading. | LOW |
| M19 | TrialConfig | TrialConfig.m:162-164 | `fromStructure` passes `structure.covLag` as the 5th positional arg (interpreted as `ensCovMask`) and `structure.name` as the 6th (interpreted as `covLag`).  Argument shift silently corrupts every `.mat` round-trip involving `TrialConfig` serialisation.  (Python ``_trial_config_impl.py:190-197`` reproduces this bug; fix should land in both ports simultaneously.) | HIGH |
| M20 | DecodingAlgorithms | +nstat/+decoding/SSGLM.m:373 | `PPSS_EStep` binomial `JacobianLD`: MATLAB writes `lambdaDelta.*(1-lambdaDelta).*(1 - 2*lambdaDelta.^2)`.  Derivation: for `p = σ(η)`, `dp/dη = p(1-p)` and `d²p/dη² = p(1-p)(1-2p)` — the third factor is **linear**, not squared.  Confirmed typo: the other three binomial sites (`DecodingAlgorithms.m:533, 603`; and Python lines 2681/2773/3291) all use the linear `(1-2*lambdaDelta)` form; SSGLM.m:373 is the lone outlier.  **Python `decoding_algorithms.py:2641` is correct as-is** (uses `(1.0 - 2.0 * lambdaDelta)`); MATLAB has a bug.  Resolved 2026-06-11. | CONFIRMED |
| M21 | CovColl | CovColl.m:202-208 | `maskAwayAllExcept` correctly calls `maskAwayOnlyCov → resetMask` first.  Python's Phase 2 fix (PR #169) brought the Python port into line with this MATLAB behavior — the bug was Python-side (didn't reset).  Logged here as the diagnostic anchor for the NetworkTutorial CRITICAL pedagogical bug. | INFO |
| M22 | DecodingAlgorithms | DA.m `PPHybridFilterLinear` time recursion (Python `decoding_algorithms.py:2059, 2277`) | Time recursion reads `MU_u[:, time_index]` (current step, always 0 since not yet written) when it should read `MU_u[:, time_index - 1]` (previous step).  Effect: model probability column sums are NOT 1; only the first column gets a non-zero update.  Python preserves MATLAB-faithful (bug-affected) behavior to maintain gold-fixture parity at `tests/parity/fixtures/matlab_gold/decoder_*.mat`; standalone correctness test `test_pphybridfilterlinear_returns_model_probabilities_and_state_banks` is marked `xfail` with reference to this entry.  Discovered 2026-06-11 during v0.5.0 subsystem 4 deep-dive. | CONFIRMED (held for MATLAB-faithful parity) |

### 1.4 Simulink Model Behavior (C4 verification, 2026-06-11)

Resolved by opening `PointProcessSimulation.mdl.r2013a` (textual MDL,
ASCII) and `PointProcessSimulation.slx` (zip+XML).  Both formats
confirm:

| Sub-block | Annotation | Math |
|---|---|---|
| `Poisson` (SID 53) | `\lambda \cdot \Delta = e^{X \cdot \beta}` | `lambdaDelta = exp(eta)` directly (per-bin probability) |
| `Binomial` (SID 48) | `\lambda \cdot \Delta = e^{X \cdot \beta}/(1+e^{X\cdot \beta})` | `lambdaDelta = sigmoid(eta)` |
| `Switch` (SID 51) | `simTypeSelect = 1 → Poisson port` | `simType='poisson'` routes Poisson sub-block; `'binomial'` routes Binomial |

Python's pre-PR-#167 `_simulateCIF_python` Poisson path applied
`1 - exp(-exp(eta)*dt)` which double-counted `dt` and produced spike
rates ~1/dt slow.  Fixed in PR #167 (commit `25b404d`).  The continuous-
time companion `PointProcessSimulationCont.slx` uses the same math.

---

## 2. Python Bugs

| # | Class | File:Line | Description | Severity | Fix |
|---|---|---|---|---|---|
| P1 | SignalObj | core.py:697 | `std()` uses `ddof=0` but Matlab `std(x)` uses `ddof=1` | MEDIUM | Change to `np.std(..., ddof=1)` |
| P2 | SignalObj | core.py:971 | `alignTime()` shifts unconditionally; Matlab only shifts if `minTime <= marker <= maxTime` | MEDIUM | Add bounds check |
| ~~P3~~ | ~~nspikeTrain~~ | ~~core.py (setMinTime/setMaxTime)~~ | ~~Not a bug: `computeStatistics(-1)` called as method runs full computation; `-1` only skips plotting~~ | ~~N/A~~ | ~~None needed~~ |
| ~~P4~~ | ~~nspikeTrain~~ | ~~core.py (setSigRep)~~ | ~~Not a bug: same as P3~~ | ~~N/A~~ | ~~None needed~~ |
| P5 | SpikeTrainCollection | trial.py (psthGLM) | Returns `(self.psth(binwidth), None, None)` stub instead of full GLM PSTH | HIGH | Wire up `_psth_glm_coeffs()` |
| P6 | SpikeTrainCollection | trial.py (getNSTnames) | Ignores `selectorArray` parameter; always returns all names | MEDIUM | Honor the filter argument |
| P7 | SpikeTrainCollection | trial.py (getUniqueNSTnames) | Same issue: ignores `selectorArray` | MEDIUM | Honor the filter argument |
| P8 | SpikeTrainCollection | trial.py (getNST) | Does not resample to collection `sampleRate` on retrieval (Matlab does) | MEDIUM | Add resample check |
| P9 | CovColl | trial.py (isCovPresent) | Off-by-one: same as Matlab bug M8 | LOW | Fix `<` to `<=` |

---

## 3. Missing Python Methods

### 3.1 HIGH Priority (blocks paper examples or common workflows)

| Class | Method | Matlab Lines | Notes |
|---|---|---|---|
| SpikeTrainCollection | `psthGLM()` (full impl) | ~200 | Stub exists; internal `_psth_glm_coeffs()` has logic but not wired |
| DecodingAlgorithms | `KF_EM` | 3356-3659 | Gaussian state-space EM — commonly needed |
| DecodingAlgorithms | `KF_EStep` | 4401-4498 | E-step for KF_EM |
| DecodingAlgorithms | `KF_MStep` | 4499-4586 | M-step for KF_EM |
| DecodingAlgorithms | `KF_ComputeParamStandardErrors` | 3660-4400 | Parameter inference for KF_EM |

### 3.2 MEDIUM Priority

| Class | Method | Matlab Lines | Notes |
|---|---|---|---|
| DecodingAlgorithms | `KF_EMCreateConstraints` | 3295-3355 | Constraint builder for KF_EM |
| DecodingAlgorithms | `PP_EM` | 8593-8952 | PP state-space EM (no basis) |
| DecodingAlgorithms | `PP_EStep` | 9265-9402 | E-step for PP_EM |
| DecodingAlgorithms | `PP_MStep` | 9657-10362 | M-step for PP_EM |
| DecodingAlgorithms | `PP_ComputeParamStandardErrors` | 7712-8592 | Info matrix for PP_EM |
| DecodingAlgorithms | `PP_EMCreateConstraints` | 7663-7711 | Constraint builder for PP_EM |
| DecodingAlgorithms | `mPPCODecodeLinear` | 4689-4845 | Mixed PP+Gaussian filter |
| DecodingAlgorithms | `mPPCODecode_predict` | 4846-4854 | Predict sub-step |
| DecodingAlgorithms | `mPPCODecode_update` | 4855-4944 | Update sub-step |
| DecodingAlgorithms | `mPPCO_fixedIntervalSmoother` | 4587-4688 | Smoother for mPPCO |
| DecodingAlgorithms | `mPPCO_EMCreateConstraints` | 4945-5005 | Constraint builder |
| DecodingAlgorithms | `mPPCO_EM` | 6139-6554 | Full EM for mPPCO |
| DecodingAlgorithms | `mPPCO_EStep` | 6555-6772 | E-step for mPPCO |
| DecodingAlgorithms | `mPPCO_MStep` | 6773-7662 | M-step for mPPCO |
| DecodingAlgorithms | `mPPCO_ComputeParamStandardErrors` | 5006-6138 | Info matrix for mPPCO |
| DecodingAlgorithms | `computeSpikeRateCIs` | 3087-3186 | Monte Carlo spike rate CIs |
| DecodingAlgorithms | `computeSpikeRateDiffCIs` | 3189-3292 | Spike rate difference CIs |
| DecodingAlgorithms | PPDecodeFilterLinear target estimation | ~200 lines | Srinivasan 2006; silently disabled in Python |
| FitResSummary | `plotCoeffsWithoutHistory()` | ~300 | Coefficient plot excluding history terms |
| FitResSummary | `plotHistCoeffs()` | ~300 | History coefficient specialized plot |
| Trial | `toStructure()` / `fromStructure()` | ~100 | Serialization round-trip |
| Trial | `getNumHist()` | ~20 | History count accessor |
| Trial | `findMinSampleRate()` | ~20 | Min sample rate across components |
| Analysis | Validation lambda computation | ~50 | `fitResults.validation` always None in Python |

### 3.3 LOW Priority

| Class | Method | Notes |
|---|---|---|
| SignalObj | `alignToMax` | Align signal to its maximum |
| SignalObj | `normWindowedSignal` | Normalize windowed signal |
| SignalObj | `windowedSignal` | Extract windowed segment |
| SignalObj | `getSubSignalsWithinNStd` | Filter by N-std criterion |
| SignalObj | `plotVariability` / `plotAllVariability` | Variability visualization |
| SignalObj | `clearPlotProps` / `plotPropsSet` | Plot property management |
| SignalObj | `convertNamesToIndices` | Name-to-index mapping |
| SignalObj | `areDataLabelsEmpty` / `isLabelPresent` | Label utilities |
| SignalObj | `ctranspose` / `transpose` / `ldivide` / `mtimes` | Matrix operator overloads |
| SpikeTrainCollection | `reverseOrderPlot` parameter | Reverse raster plot order |
| Events | `colorString` override in plot | Custom color per event |

---

## 4. Behavioral Differences

### 4.1 Significant (may affect numerical results)

| # | Class | Difference | Impact |
|---|---|---|---|
| D1 | SignalObj | Python arithmetic doesn't call `makeCompatible` to auto-align time grids; raises ValueError instead | Users must manually align grids |
| D2 | Analysis | Different GLM solvers: Matlab CG with rrflag=0; Python Newton-Raphson with L2=1e-6 | Slight numerical differences in coefficients |
| D3 | DecodingAlgorithms | `kalman_fixedIntervalSmoother`: **CORRECTED 2026-05-26.** Original entry claimed a Python-side approximation vs MATLAB. Investigation (driven by the new `nstat.extras.validation.pykalman_bridge`) shows the Python implementation IS a faithful port of MATLAB's augmented-state fixed-lag smoother — both use the same Anderson-Moore construction. The previously-reported ~0.4 disagreement vs pykalman's RTS smoother was a fixed-lag-vs-RTS comparison, not a parity bug. nstat's full RTS path lives in `kalman_smoother` (forward filter + RTS backward pass) and agrees with pykalman to ~1.6e-4 (dominated by the t=0 init convention difference). No code change to `kalman_fixedIntervalSmoother` itself. | No parity bug; bridge was fixed in PR-E to call `kalman_smoother` instead. |
| D4 | DecodingAlgorithms | `ComputeStimulusCIs`: Matlab uses Monte Carlo; Python public API uses Gaussian approximation | Different CI widths for non-Gaussian posteriors |
| D5 | DecodingAlgorithms | `PPHybridFilter`: Python delegates to linear version, losing nonlinear CIF support | Cannot handle nonlinear CIF models |
| D6 | nspikeTrain | Burst statistics go stale after setMinTime/setMaxTime in Python | Stale burst data if accessed post-window-change |

### 4.2 Minor / By Design

| # | Class | Difference | Impact |
|---|---|---|---|
| D7 | nspikeTrain | Python always sorts spike times on construction; Matlab preserves insertion order | Python more correct |
| D8 | SpikeTrainCollection | `addSingleSpikeToColl` uses value semantics (copy); Matlab uses handle (shared ref) | Pythonic; mutations don't propagate |
| D9 | SpikeTrainCollection | `shiftTime` returns new collection; Matlab mutates in place | Pythonic |
| D10 | SpikeTrainCollection | `psthBars()` uses deterministic smoothing; Matlab uses BARS package | By design (no BARS dependency) |
| D11 | SpikeTrainCollection | `nstColl.plot()` y-tick labels use indices; Matlab uses neuron names | Cosmetic |
| D12 | DecodingAlgorithms | `kalman_filter` has different API signature and data layout (time-first vs state-first) | API difference, correct logic |
| D13 | DecodingAlgorithms | `PPSS_EStep`: Python uses `eigh` (symmetric); Matlab uses `eig` (general) | Python more correct |
| D14 | DecodingAlgorithms | `PPSS_MStep` gamma clamping: Matlab [-1e1,1e1]; Python [-1e2,1e2] | Minor numerical difference |
| D15 | DecodingAlgorithms | `PPDecode_updateLinear` fallback: Matlab rcond check; Python try/except LinAlgError | Same intent, different mechanism |
| D16 | FitResult | Matlab pre-computes all diagnostics at once; Python computes on demand | Lazy vs eager evaluation |
| D17 | ConfidenceInterval | Python arithmetic operators more correct than Matlab for CI propagation | Python improvement |
| D18 | nspikeTrain | `computeRate()` fully implemented in Python; Matlab has only stub | Python extension |

---

## 5. Python-Only Extensions (no Matlab counterpart)

These are enhancements in Python with no Matlab equivalent:

- `nspikeTrain.to_binned_counts()` — binned spike count array
- `nspikeTrain.times`, `n_spikes`, `duration`, `firing_rate_hz` — convenience properties
- `SpikeTrainCollection.__iter__`, `__len__` — Pythonic iteration
- `SpikeTrainCollection.ssglmFB()` — collection-level SSGLM convenience
- `nspikeTrain.computeRate()` — full implementation (Matlab is stub)
- `nspikeTrain.originalSpikeTimes` — robust restore-to-original
- Deterministic `psthBars()` fallback (no external BARS dependency)
- `_ComputeStimulusCIs_MC` — separate Monte Carlo CI method
- `linear_decode` — linear decoding utility

---

## 6. Recommended Fix Priority

### Immediate (v0.2.1 patch)

1. **P1**: Fix `SignalObj.std()` ddof → `ddof=1` ✅
2. **P2**: Fix `SignalObj.alignTime()` bounds check ✅
3. ~~**P3/P4**: Not bugs — `computeStatistics(-1)` runs full computation~~
4. **P5**: Wire up `psthGLM()` to full GLM implementation ✅
5. **P6/P7**: Fix `getNSTnames`/`getUniqueNSTnames` to honor selectorArray ✅
6. **P8**: Add resample check in `getNST()` ✅
7. **P9**: Fix `isCovPresent` off-by-one ✅

### Next Release (v0.3.0)

1. Port `KF_EM` family (5 methods, ~1,300 lines)
2. Port `PP_EM` family (5 methods, ~2,700 lines)
3. Port `mPPCO` family (9 methods, ~3,500 lines)
4. Implement `FitResSummary.plotCoeffsWithoutHistory()` and `plotHistCoeffs()`
5. Implement `Trial.toStructure()` / `fromStructure()`
6. Add validation lambda computation to Analysis
7. Implement PPDecodeFilterLinear target estimation branch

### Matlab Bug Reports Filed

- M4: Analysis Granger causality ensCovMask zeroing → [nSTAT#15](https://github.com/cajigaslab/nSTAT/issues/15)
- M5: Analysis phiMat strfind indexing → included in [nSTAT#15](https://github.com/cajigaslab/nSTAT/issues/15)
- M6: Analysis sampeRate typo → [nSTAT#16](https://github.com/cajigaslab/nSTAT/issues/16)
- M7: CovColl findMaxTime double shift → [nSTAT#18](https://github.com/cajigaslab/nSTAT/issues/18)
- M8: CovColl isCovPresent off-by-one → [nSTAT#17](https://github.com/cajigaslab/nSTAT/issues/17)
- M9: TrialConfig fromStructure missing ensCovMask → [nSTAT#19](https://github.com/cajigaslab/nSTAT/issues/19)
- M10: SignalObj autocorrelation typo → low priority, not filed
- M11: SignalObj times/rdivide aliasing → low priority, not filed
- M12: DecodingAlgorithms gamma broadcasting → [nSTAT#20](https://github.com/cajigaslab/nSTAT/issues/20)
- M13: nstColl getSpikeTimes uninitialized count → [nSTAT#21](https://github.com/cajigaslab/nSTAT/issues/21)
