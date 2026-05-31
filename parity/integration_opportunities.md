# Integration Opportunities — Python neural-data ecosystem cross-reference

> Generated: 2026-05-24
> Purpose: identify Python toolboxes that can cross-validate, complement,
> or extend `nstat-python` (v0.3.1).  Independence rule (CLAUDE.md)
> preserved — every recommendation here is a Python-side library; no
> coupling to the MATLAB `cajigaslab/nSTAT` repo is introduced.
>
> **See also** [`methods_roadmap.md`](methods_roadmap.md) — the
> prioritized *methods/algorithms* incorporation plan (2026 review):
> what new capabilities to add next, vs. this document's *library*
> survey.

---

## Executive summary

1. **NeMoS (Flatiron) is the single highest-value cross-validation target.**  It is a JAX-backed Poisson/Gamma GLM toolbox with the *exact* raised-cosine and B-spline basis families nstat-python uses, MIT-licensed, Python 3.10–3.12, actively maintained (v0.2.7, Feb 2026).  Adding it as a test-only extra and writing a parity harness for `Analysis.GLMFit` vs `nemos.glm.GLM` is the cheapest high-confidence win available.
2. **Dynamax (probml) is the right home for the missing EM families** (`KF_EM`, `PP_EM`, `mPPCO_EM` — 19 unported MATLAB methods from AUDIT_REPORT.md §3.2).  Rather than porting 7,500 lines of MATLAB EM code, wrap dynamax's `LinearGaussianSSM` / `PoissonLDS` / `GeneralizedGaussianSSM` behind the `DecodingAlgorithms` namespace and document the API delta.
3. **Eden-Kramer Lab's `time_rescale` package is a drop-in second opinion** for `FitResult.computeKSStats` — MIT, pure NumPy/SciPy, but unmaintained since 2021.  Vendor or pin, do not depend on PyPI.
4. **pynapple + Neo + pynwb form the import/interchange layer nstat-python is missing.**  No analysis overlap; everything is data-model glue.  Adding an `nstat.interop` subpackage that converts `nspikeTrain` ↔ `pynapple.Ts` / `neo.SpikeTrain` / `pynwb.Units` would let nstat slot into the modern systems-neuro workflow without touching analysis code.
5. **Avoid the historical PyGLM forks** (slinderman, memming, theano_pyglm).  All abandoned 5+ years; theano backend is dead; superseded by NeMoS and Dynamax.

---

## Capability gap matrix

| Capability | nstat-python today | Best-in-ecosystem | Recommendation |
|---|---|---|---|
| Spike sorting | none (assumed pre-sorted) | SpikeInterface (v0.103.x, MIT) | optional `[sorting]` extra; never required |
| Standardized I/O (NWB, Plexon, Blackrock, Spike2) | none | Neo 0.14.4 (BSD-3) | optional `[io]` extra; `nstat.interop.neo` |
| Time-series container (epoch math, restrict-to-trial) | partial (`Trial`) | pynapple 0.11.2 (MIT) | optional `[interop]` extra; convert at boundary |
| Poisson GLM, raised-cosine / B-spline basis | `Analysis.GLMFit` + `History`/`HistoryBasis` | NeMoS 0.2.7 (MIT, JAX) | **cross-validation target**; add `[test]`-only extra |
| Generalized additive Poisson | partial (B-spline only) | pyGAM 0.9 (Apache-2.0) | reference for tensor-product splines if pursued |
| Multitaper PSD / DPSS | `SignalObj.multitaper` (basic) | nitime 0.12 (BSD-3) + `scipy.signal.windows.dpss` | cross-validate; optionally delegate |
| Wavelet / synchrosqueezing | none | ssqueezepy 0.6.6 (MIT) | optional `[wavelet]` extra; closes AUDIT_REPORT gap |
| Spectral Granger / coherence | `Analysis.computeGrangerCausalityMatrix` (time-domain VAR) | spectral_connectivity 2.0.1 (**GPL-3**) + mne-connectivity 0.8.1 (BSD-3) | cross-validate vs mne-connectivity; avoid spectral_connectivity (license downgrade) |
| Kalman filter / smoother | `DecodingAlgorithms.kalman_*` | pykalman 0.11.2 (BSD-3), filterpy (MIT), dynamax 1.0 (MIT) | cross-validate against pykalman (matches AUDIT D3); dynamax for EM |
| State-space EM (KF_EM / PP_EM / mPPCO_EM) | **missing** (19 methods unported) | dynamax 1.0 (MIT, JAX) | wrap as opt-dep; supersedes porting effort |
| Time-rescaling KS test | `FitResult.computeKSStats` | Eden-Kramer-Lab/time_rescale 0.2.1 (MIT) | cross-validate; vendor — upstream stale since 2021 |
| Spike-train distance metrics (ISI, SPIKE, Victor-Purpura) | none | PySpike 0.8.0 (BSD-2) + Elephant | optional `[metrics]`; PySpike has C-accel |
| HMM / SLDS / rSLDS | none | ssm-jax (deprecated) → dynamax 1.0 | dynamax only |
| PSID (behaviorally-relevant subspace) | none | PyPSID 1.2.6 (custom USC license) | reference only — license is restrictive |
| Behavioral data pipelines / provenance | none | DataJoint 2.2.2 (Apache-2.0) | out of scope; document for users |
| Calcium imaging | none | CaImAn 1.13.1 (**GPL-2.0** — compatible) | out of scope; mention only |
| Pre-built point-process / hippocampal decoder | partial (PPAF/PPHF) | replay_trajectory_classification 1.4.1 (MIT) | optional reference for Example 04 (place cells) |

---

## Toolbox profiles

### Tier-A: Direct cross-validation candidates

#### NeMoS — https://github.com/flatironinstitute/nemos

- **Version / date:** 0.2.7 (2026-02-27)
- **License:** MIT (GPL-2 compatible — MIT can be redistributed under GPL)
- **Python:** 3.10, 3.11, 3.12
- **Maintainer:** Center for Computational Neuroscience, Flatiron Institute (NIH BRAIN funded)
- **Capabilities:** Poisson `GLM`, Gamma `GLM`, `PopulationGLM`, `ClassifierGLM`.  Basis families that mirror nstat's design choices precisely: `RaisedCosineLinearConv`, `RaisedCosineLogConv` (Pillow basis), `BSplineConv`/`MSplineConv`/`CyclicBSplineConv`, plus `AdditiveBasis` / `MultiplicativeBasis` composition.  JAX-backed; GPU-capable; pynapple-integrated.
- **Overlap with nstat:** Heavy.  `nstat.Analysis.GLMFit` + `nstat.History.HistoryBasis` + `nstat.cif.LinearCIF` cover essentially the same modeling surface.
- **Integration suggestion:** Add as a `[test]` extra under `pyproject.toml [project.optional-dependencies]`; write `tests/test_nemos_parity.py` that fits the same Poisson GLM in both libraries on a small synthetic dataset and asserts coefficient agreement to ~1e-4.
- **Risks:** JAX install footprint is large (~200 MB).  Keep it strictly test-only — do not import from `nstat/`.

#### Dynamax — https://github.com/probml/dynamax

- **Version / date:** 1.0 (2025-03-25, 1,688 commits, active)
- **License:** MIT
- **Python:** 3.10+ (JAX)
- **Capabilities:** `LinearGaussianSSM` (Kalman + RTS smoother + EM), `PoissonHMM`, `LinearAutoregressiveHMM`, `GeneralizedGaussianSSM` with the Conditional-Moments-Gaussian-Smoother — the closest published equivalent of MATLAB nSTAT's `KF_EM` / `PP_EM` / `mPPCO_EM` families.
- **Overlap with nstat:** Fills exactly the AUDIT_REPORT §3.1 / §3.2 gaps (5 + 14 missing methods).
- **Integration suggestion:** Two-phase.  (a) Add `tests/test_dynamax_kalman_parity.py` to cross-validate `DecodingAlgorithms.kalman_filter` and `kalman_fixedIntervalSmoother` (the latter has the known smoother-index approximation gap, AUDIT D3 — dynamax is the gold standard here).  (b) Create `nstat.interop.dynamax` thin adapters so users can request EM-trained models without nstat owning the EM code.
- **Risks:** JAX dependency; Dynamax's API is parameter-tuple-based (Pytrees), nstat's is object-based.  An adapter layer is needed.

#### pykalman — https://github.com/pykalman/pykalman

- **Version / date:** 0.11.2 (2026-01-31, community-resurrected; 1.3k stars)
- **License:** BSD-3-Clause
- **Python:** 3.10–3.14
- **Capabilities:** `KalmanFilter`, `UnscentedKalmanFilter`, EM for KF parameters, robust square-root form.
- **Overlap:** Direct equivalent of `DecodingAlgorithms.kalman_filter` and `kalman_smoother`.  No point-process support.
- **Integration suggestion:** Pure-NumPy, no JAX — use it as the *low-friction* cross-validation reference for the Gaussian-only Kalman path.  Lighter test dep than dynamax.
- **Risks:** Older API conventions (state-first axis layout); minor.

#### Eden-Kramer-Lab/time_rescale — https://github.com/Eden-Kramer-Lab/time_rescale

- **Version / date:** 0.2.1 (2021-10-12, **stale**)
- **License:** MIT
- **Capabilities:** `TimeRescaling` class with `plot_ks`, autocorrelation independence test, trial-aware censoring correction.
- **Overlap:** Direct counterpart of `FitResult.computeKSStats`.
- **Integration suggestion:** Use as a parity oracle for `computeKSStats`.  Because upstream is unmaintained, do **not** add as a runtime or test dep — vendor the ~200 lines into `tests/parity/_third_party/time_rescale.py` with attribution, or pin to `==0.2.1` in a `[test-parity]` extra.

#### nitime — https://github.com/nipy/nitime

- **Version / date:** 0.12 (2025-11-06)
- **License:** BSD-3-Clause
- **Capabilities:** Multitaper PSD, multitaper coherence, time-domain coherence — the most rigorous DPSS implementation in the Python ecosystem outside MNE.
- **Overlap:** `SignalObj.multitaper` cross-validation; nstat's current implementation is basic (AGENT_GUIDE.md §5.6 explicitly notes "for serious spectral analysis prefer dedicated libraries").
- **Integration suggestion:** Cross-validate `SignalObj.multitaper` against `nitime.algorithms.spectral.multi_taper_psd` in `tests/test_signalobj_spectral_parity.py`.

#### mne-connectivity — https://github.com/mne-tools/mne-connectivity

- **Version / date:** 0.8.1 (2026-04-11)
- **License:** BSD-3-Clause
- **Capabilities:** Spectral Granger causality, partial directed coherence, phase-locking value, weighted PLI.
- **Overlap:** Cross-reference for `Analysis.computeGrangerCausalityMatrix` (which is *time-domain* VAR; mne-connectivity is *spectral*, so the comparison is at the conceptual level, not bit-level).
- **Integration suggestion:** Note in `parity/report.md` as a recommended second-opinion tool for users; do not add as a dep.

### Tier-B: Interop / data-model

#### Neo — https://github.com/NeuralEnsemble/python-neo

- **Version / date:** 0.14.4 (2026-03-12)
- **License:** BSD-3-Clause
- **Capabilities:** Hierarchical data model (`Block`/`Segment`/`SpikeTrain`/`AnalogSignal`/`Event`), readers for Spike2/NEX/AlphaOmega/Axon/Blackrock/Plexon/TDT.  Standard interchange object for the Elephant / SpikeInterface / Brian ecosystems.
- **Integration suggestion:** Add `nstat.interop.neo` module with `to_neo_spiketrain(nspikeTrain)` and `from_neo_spiketrain(neo.SpikeTrain) -> nspikeTrain` converters.  Quantities-aware (preserves units).
- **Risks:** None; pure data conversion.

#### pynapple — https://github.com/pynapple-org/pynapple

- **Version / date:** 0.11.2 (2026-05-13)
- **License:** MIT
- **Python:** 3.11 demonstrated; supports modern Python.
- **Capabilities:** `Ts`/`Tsd`/`TsdFrame` time series, `IntervalSet` epoch math, tuning curves, cross-correlograms, perievent histograms, Morlet wavelets, NWB-native I/O.
- **Integration suggestion:** `nstat.interop.pynapple` with `to_pynapple_ts(nspikeTrain)` and inverse.  Pynapple's `IntervalSet` math is what users repeatedly hand-roll on top of `Trial` — exposing the conversion would defer that whole API surface to pynapple cleanly.
- **Risks:** None; same independence-rule status as Neo.

#### pynwb — https://github.com/NeurodataWithoutBorders/pynwb

- **Version / date:** 3.1.3 (2025-12-09)
- **License:** BSD-3-Clause
- **Capabilities:** Reference NWB:N reader/writer maintained by LBNL/AllenInst/NIH.
- **Integration suggestion:** `nstat.interop.nwb` reader: `nwb_to_trial(nwbfile, electrical_series_name="lfp", units_table="units") -> Trial`.  The figshare paper-example dataset could optionally be republished as NWB for forward compatibility.
- **Risks:** Adds `hdmf` transitive dep; keep optional.

### Tier-C: Reference / inspiration only

#### Elephant — https://github.com/NeuralEnsemble/elephant

- **Version / date:** 1.2.0 (2026-03-04)
- **License:** BSD-3-Clause
- **Capabilities:** Spike train statistics (rates, ISI/CV), correlation (cross-correlogram, sttc), synchrony detection (CAD/UE/ASSET/SPADE/CuBIC), spike-triggered average, spectral analysis, causality measures, surrogate generation, point-process generation (homogeneous/inhomogeneous Poisson, gamma).
- **Integration:** Cross-validate (a) ISI/CV statistics in `nspikeTrain.computeStatistics` against `elephant.statistics`, and (b) `simulate_poisson_from_rate` against `elephant.spike_train_generation.inhomogeneous_poisson_process`.  Test-only.
- **Caveat:** Elephant is Neo-typed end-to-end — interop layer required first.

#### pyGAM — https://pygam.readthedocs.io

- **Version / date:** active, Python 3.10+
- **License:** Apache-2.0
- **Capabilities:** Generalized additive models with Poisson family, tensor-product splines, penalties.
- **Integration:** Reference architecture only.  If nstat ever extends `LinearCIF` beyond 1-D B-splines to tensor-product splines, pyGAM is the cleanest existing implementation to learn from.

#### statsmodels — https://www.statsmodels.org

- **Version / date:** 0.14.6 (2025-12-05)
- **License:** BSD-3-Clause
- **Capabilities:** `statsmodels.genmod.GLM` (Poisson/Binomial/Gamma families with all standard link functions), state-space framework, GEE.
- **Integration:** A third independent reference for `Analysis.GLMFit` coefficients.  Already in the SciPy stack; zero install cost.  Use in the same `test_glmfit_parity.py` as NeMoS — triangulating against two unrelated implementations is much stronger than against one.

#### ssqueezepy — https://github.com/OverLordGoldDragon/ssqueezepy

- **Version / date:** 0.6.6 (2025-08-02)
- **License:** MIT
- **Capabilities:** CWT, STFT, synchrosqueezing (CWT + STFT), generalized Morse wavelets, ridge extraction, GPU via CuPy/PyTorch.
- **Integration:** Closes AGENT_GUIDE.md §5.6 admission that nstat lacks "adaptive multitaper, wavelet/synchrosqueeze."  Plug as optional `[wavelet]` extra; expose `SignalObj.synchrosqueeze()` convenience that delegates.

#### multitaper_toolbox (Prerau lab) — https://github.com/preraulab/multitaper_toolbox

- **Version / date:** v1, multi-language (Python/MATLAB/Rust/R), occasional updates
- **License:** BSD-3-Clause
- **Capabilities:** State-of-the-art multitaper spectrogram (Prerau lab's "fast multitaper") with Rust acceleration (10–21× speedup).
- **Integration:** Reference for a future high-performance `SignalObj.spectrogram` path; lower priority than nitime cross-validation.

#### filterpy — https://github.com/rlabbe/filterpy

- **License:** MIT
- **Capabilities:** Kalman/EKF/UKF/particle filter, with a beautifully documented companion textbook.
- **Integration:** Pedagogical reference; pykalman is a better runtime cross-check target.

#### PySpike — https://github.com/mariomulansky/PySpike

- **Version / date:** 0.8.0 (2023-10-13, moderate maintenance)
- **License:** BSD-2-Clause
- **Capabilities:** ISI-distance, SPIKE-distance, SPIKE-Synchronization with C/Cython acceleration.
- **Integration:** Currently no spike-train-distance functionality in nstat; if added, PySpike is the obvious dependency rather than rolling your own.

#### replay_trajectory_classification — https://github.com/Eden-Kramer-Lab/replay_trajectory_classification

- **Version / date:** 1.4.1 (2024-09-06)
- **License:** MIT
- **Capabilities:** Clusterless and sorted-spike decoding of position from hippocampus; richer than the PPAF/PPHF currently in nstat.
- **Integration:** Reference for a more sophisticated `Example 04` (place cells) extension.  Loosely complementary, not overlapping.

### Tier-D: Out-of-scope but worth noting for users

- **SpikeInterface** v0.103.x (MIT) — the de-facto Python spike-sorting framework.  AGENT_GUIDE.md correctly states nstat assumes pre-sorted data; point users at SpikeInterface in the README "Related projects" section.
- **CaImAn** 1.13.1 (GPL-2.0 — *same license as nstat*) — calcium imaging; out of nstat scope but license-compatible if any fusion is ever attempted.
- **DataJoint** 2.2.2 (Apache-2.0) — relational data pipelines; complementary to nstat's analysis surface, not overlapping.
- **MNE-Python** 1.12.1 (BSD-3) — heavy EEG/MEG focus.  Useful as a deep reference for time-frequency, but full integration is overkill.
- **PyPSID** 1.2.6 — Shanechi-lab subspace identification for behavior-relevant neural dynamics.  License is a custom USC academic license; not redistributable.  Reference only.

### Tier-D: NOT recommended

- **slinderman/pyglm, memming/pyglm, slinderman/theano_pyglm** — abandoned 5+ years, Theano backend is dead, no licenses on the public repos.  Superseded entirely by NeMoS + Dynamax.
- **pillowlab/neuroGLM** — MATLAB only; violates the Python-side scope of this survey.
- **pillowlab/GLMspiketraintutorial_python** — pedagogical only, 21 commits, no license declared, the README itself recommends users go to neuroGLM / GLMspiketools for production.
- **spectral_connectivity** v2.0.1 (Eden-Kramer Lab) — **GPL-3.0**.  nstat is GPL-2.0; linking GPL-3.0 from a GPL-2.0 codebase is a license downgrade (GPL-2-only cannot consume GPL-3).  Use mne-connectivity (BSD-3) instead.
- **ssm (lindermanlab/ssm)** — superseded by ssm-jax → dynamax migration path.  Use dynamax.
- **ssm-jax** — explicitly marked superseded by dynamax in its own README.
- **spynal** — capped at Python ≤3.11 (incompatible with nstat's CI matrix of 3.11/3.12).  Skip until upstream fixes Python 3.12 support.

---

## Independence considerations

Every Tier-A/B/C recommendation above is a pure-Python library with no MATLAB dependency.  None of them violate the CLAUDE.md rule that nstat-python and `cajigaslab/nSTAT` (MATLAB) must remain decoupled.

The MATLAB-only neuroGLM and GLMspiketools entries are explicitly flagged as **NOT recommended for runtime integration** for this reason.  They remain useful only as conceptual references.

License-wise, nstat-python is **GPL-2.0**.  Compatibility status of recommended deps:

| License | Direction with GPL-2.0 | Safe to depend on? |
|---|---|---|
| MIT, BSD-2/3, Apache-2.0 | one-way: permissive → GPL-2 OK | yes |
| GPL-2.0 | same | yes (CaImAn) |
| GPL-3.0 | GPL-2-only **cannot** consume GPL-3 | **NO** (spectral_connectivity excluded for this reason) |
| LGPL-2.1+ / LGPL-3 | dynamic linking OK | yes if dyn-linked |
| custom academic (PyPSID) | non-redistributable | reference only |

If nstat-python ever relicenses to **GPL-2.0-or-later** or **GPL-3.0**, spectral_connectivity becomes available.  Worth flagging as a future option.

---

## Recommended actions (prioritized)

### Tier 1 — High value, low friction (this quarter)

1. **Add `[test-parity]` extra** to `pyproject.toml` containing `nemos>=0.2.7`, `pykalman>=0.11.2`, `statsmodels>=0.14`, `nitime>=0.12`.
2. **Write `tests/test_nemos_glmfit_parity.py`** — fit identical Poisson GLM on a synthetic spike train via both `nstat.Analysis.GLMFit` and `nemos.glm.GLM`; assert coefficient L∞ < 1e-3.  This is the strongest single addition to the test suite.
3. **Write `tests/test_kalman_parity.py`** — cross-validate `DecodingAlgorithms.kalman_filter` against `pykalman.KalmanFilter` on a 2-state linear-Gaussian process; check filtered means agree to 1e-8, smoothed means to 1e-6.  This directly addresses AUDIT D3 (`kalman_fixedIntervalSmoother` smoother-index approximation gap).
4. **Vendor `time_rescale` v0.2.1** into `tests/parity/_third_party/` (with MIT license preserved); use as second-opinion oracle for `computeKSStats`.
5. **Add a "Related Projects" section to `README.md`** listing SpikeInterface, Elephant, NeMoS, pynapple, Neo as ecosystem peers.  Costs nothing; positions nstat correctly.

### Tier 2 — Optional dependencies (this year)

6. **`nstat.interop.neo`** — `to_neo_spiketrain` / `from_neo_spiketrain` converters; optional `[neo]` extra.  Unlocks the SpikeInterface / Elephant / Brian2 ecosystem.
7. **`nstat.interop.pynapple`** — `to_pynapple_ts` / `to_pynapple_tsd` / `nspikeTrain.from_pynapple`; optional `[pynapple]` extra.  Lets users keep epoch math in pynapple.
8. **`nstat.interop.nwb`** — NWB reader that returns a populated `Trial`; optional `[nwb]` extra.  Critical for adoption — NWB is the BRAIN Initiative standard.
9. **`nstat.interop.dynamax`** — thin adapters that satisfy the `KF_EM` / `PP_EM` / `mPPCO_EM` API contracts by delegating to `LinearGaussianSSM`, `PoissonHMM`, `GeneralizedGaussianSSM`.  This is the alternative to a multi-thousand-line MATLAB port (AUDIT_REPORT.md §3.1–3.2).  Optional `[em]` extra.

### Tier 3 — Inspiration / borrowing

10. **Study `nemos.basis`** before any v0.4 refactor of `nstat/history.py` — particularly `RaisedCosineLogConv` and the `AdditiveBasis`/`MultiplicativeBasis` composition pattern.  NeMoS's composable-basis design is cleaner than nstat's current basis dispatching.
11. **Study `dynamax.linear_gaussian_ssm.inference`** for the canonical RTS smoother implementation; compare against `DecodingAlgorithms.kalman_fixedIntervalSmoother` to either confirm or fix the AUDIT D3 approximation.
12. **Study `ssqueezepy`'s synchrosqueeze API** before designing `SignalObj.synchrosqueeze`; do not invent a competing API.

### Tier 4 — Explicitly NOT recommended

- Do not add any pyglm fork as a dep.
- Do not add spectral_connectivity (GPL-3 incompatibility).
- Do not add spynal (Python 3.11 cap).
- Do not add ssm (use dynamax).

---

## Follow-up plan — concrete next steps

| # | Action | Effort | Owner |
|---|---|---|---|
| 1 | Open issue: "Add NeMoS as `[test-parity]` extra and write GLMFit parity harness" | 1d | maintainer |
| 2 | Open issue: "Cross-validate Kalman smoother against pykalman and dynamax to settle AUDIT D3" | 1d | maintainer |
| 3 | Open issue: "Vendor `time_rescale` v0.2.1 as KS-test oracle" | 0.5d | maintainer |
| 4 | Open issue: "Design `nstat.interop` subpackage (Neo / pynapple / NWB converters)" | 3–5d | new contributor friendly |
| 5 | Open issue: "Evaluate dynamax as supplier for KF_EM / PP_EM / mPPCO_EM" — explicitly include a decision: port vs adapt | 1d to write up, weeks to execute | maintainer + reviewer |
| 6 | Add "Related Projects" section to `README.md` (SpikeInterface, Elephant, NeMoS, pynapple, Neo) | 30 min | maintainer |
| 7 | Update `AGENT_GUIDE.md` §5.6 ("What the package is NOT") to point users at the recommended tools for each gap | 30 min | maintainer |
| 8 | Document in `parity/report.md` that NeMoS + statsmodels form the Python-side validation pair, alongside the MATLAB gold fixtures | 1h | regenerate via `tools/parity/build_report.py` |

The natural ordering: Tier-1 items first (they pay back in test coverage immediately), then the `nstat.interop` work in Tier-2 (this is where the user-visible win is), then revisit Tier-2 #9 (dynamax for EM) as a v0.4 milestone after the cross-validation harnesses are in place — because once we trust `DecodingAlgorithms.kalman_*` via the parity tests, the case for *replacing* missing methods with dynamax adapters (versus reimplementing them) becomes empirically decidable rather than philosophical.

---

*Generated 2026-05-24 from web-verified upstream metadata.  Re-run this
audit annually or whenever a Tier-1 recommendation changes its
maintenance status.  No MATLAB-side coupling introduced.*
