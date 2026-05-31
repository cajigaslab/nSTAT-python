# MATLAB nSTAT v1.4.0 audit — Python cross-reference

**Date:** 2026-05-21
**MATLAB audit source:** `/Users/iahncajigas/projects/nstat/AUDIT_REPORT.md`
  (2026-03-10 audit; 67 `% FIX:` tags across 8 files; refreshed through
  the May 2026 modernization waves)
**Python audit source:** `AUDIT_REPORT.md` in this repo (2026-03-10
  audit of Python v0.2.0)

This document cross-references the 67 MATLAB-side fixes against the
Python port and records the current status.  Each MATLAB finding falls
into one of five categories:

- **PORTED** — bug existed in MATLAB and the Python port; both sides fixed.
- **PYTHON-CLEAN** — bug existed in MATLAB but the Python port never had it
  (different idiom, different code path, etc.).
- **PYTHON-MIRRORED-MATLAB-BUG (now both fixed)** — Python faithfully
  replicated the MATLAB bug; both sides now corrected.
- **PYTHON-INHERITED-BUG (NEW FINDING)** — Python still carries the bug
  even though MATLAB fixed it.  Phase 4.4 surfaces these explicitly.
- **NOT-APPLICABLE** — MATLAB-language-specific (e.g., `roundn` deprecation,
  handle aliasing, `eval()` patterns).  Python has different semantics
  and isn't subject to the issue.

---

## Section 1 — Critical bugs (MATLAB AUDIT §1)

| MATLAB # | File:Line | Description | Python status |
|---|---|---|---|
| 1.1 | FitResult.m:371 | `delta = sampleRate` inverted in logLL | **PYTHON-CLEAN** — Python's `Analysis.computeKSStats` uses `1/sampleRate`; `Analysis.GLMFit` introduces `delta = 1/sample_rate` at `nstat/analysis.py:255`.  Python's `logLL` formula was inherited as the MATLAB-hybrid form (now both sides fixed). |
| 1.2 | DecodingAlgorithms.m:765,925,932 | `isa(condNum,'nan')` always false | **PYTHON-CLEAN** — Python doesn't have an `isa(x, 'nan')` idiom; numerical-condition checks use `np.isnan(...)` or `np.linalg.cond(...) > threshold`.  Grep confirms no `condNum` regression in `nstat/decoding_algorithms.py`. |
| 1.3 | DecodingAlgorithms.m:5483/5537/8071/8125 | `ExplambdaDeltaCubed` computed as `ld.^2` instead of `ld.^3` | **PYTHON-INHERITED-BUG → FIXED in this PR** — `nstat/decoding_algorithms.py:5092` had ``ExplambdaDeltaCubed = ... np.sum(ld ** 2)`` with a comment ``# Matlab uses ld.^2 here``.  Mirrored the MATLAB bug; MATLAB v1.4.0 fixed it; Python now fixed too.  Lines 5063-5065 already had the correct `ld ** 3` form. |
| 1.4 | CIF.m:240+ (16 sites) | `symvar()` reorders alphabetically | **PYTHON-CLEAN** — `nstat/cif.py` uses sympy's `Symbol` ordering directly via `varIn` storage; no `symvar`-equivalent re-extraction.  Variable order is preserved across `evalLambdaDelta`, gradient, Jacobian calls. |
| 1.5 | SignalObj.m:1574 | `findGlobalPeak('minima')` crashes (typo `sOBj`) | **PORTED** — Python audit's M2 row catches the same `sortedData` vs `sortedPeaks` typo, filed at [nSTAT#13](https://github.com/cajigaslab/nSTAT/issues/13).  Python's `SignalObj.findGlobalPeak` was implemented correctly from the start. |
| 1.6 | SignalObj.m:1596-1598 | `findPeaks('minima')` returns maxima | **PORTED** — Python audit's M1 row (filed at [nSTAT#12](https://github.com/cajigaslab/nSTAT/issues/12)).  Python's implementation negates data correctly. |
| 1.7 | SignalObj.m:1066 | `crosscor` typo (missing trailing `r`) | **PORTED** — Python audit's M10 row.  Python uses `crosscorr` (correct). |
| 1.8 | SignalObj.m:664,713,733 | Handle aliasing in `times`/`rdivide`/`ldivide` | **NOT-APPLICABLE** — Python uses value semantics; `s3 = s1c` doesn't create a handle alias.  Python audit's M11 row was filed against MATLAB, not Python. |
| 1.9 | SignalObj.m:2174 | `length(ciUpper)` should be `length(ciLower)` in `plotAllVariability` ciBottom branch | **UNVERIFIED** — Python's `plotAllVariability` is in `nstat/core.py`; needs inspection.  Filed as Phase 4.4 follow-up. |
| 1.10 | nspikeTrain.m:287 | Burst detection off-by-one in `find(y(burstStart(end):end)==1, 1, 'last')` (missing `+ burstStart(end) - 1` offset; prepend vs append). | **UNVERIFIED — PROBABLY FIXED** — Python's `nstat/_spike_train_impl.py:181-184` uses `flatnonzero(y[burst_start[-1]:] == 1)` then `concatenate([burst_end, [int(last[-1])]])` (i.e., appends).  The Python implementation appears to do it correctly.  Worth a regression test against the MATLAB gold fixture for the prepend/append boundary case. |

---

## Section 2 — Numerical safety fixes (MATLAB AUDIT §2)

| MATLAB # | File:Line | Description | Python status |
|---|---|---|---|
| 2.1 | FitResult.m:353,373,415 | `log(0)` guards (added `max(x, eps)`) | **PORTED** — `nstat/fit.py` uses `np.maximum(lam, 1e-30)` and similar floor patterns; `nstat/analysis.py:300-301` uses `matlab_bin_mass = np.maximum(rate_hz / sample_rate, 1e-12)`. |
| 2.2 | DecodingAlgorithms.m:9299 | Q matrix indexing `min(size(Q,3))` → `min(size(Q,3), k)` | **UNVERIFIED** — Python's equivalent SSGLM Q-matrix indexing needs spot-check.  Filed as Phase 4.4 follow-up. |
| 2.3 | nspikeTrain.m:219 | Division by zero in `avgFiringRate` | **PORTED** — `nstat/_spike_train_impl.py:121-125` `firing_rate_hz` property explicitly guards `if self.duration <= 0: return 0.0`. |

---

## Section 3 — Code quality fixes (MATLAB AUDIT §3)

All Python-irrelevant due to language differences:

| MATLAB # | Description | Why N/A in Python |
|---|---|---|
| 3.1 | 22 `eval()` → `feval()` conversions | Python doesn't use `eval()` in this code path. |
| 3.2 | Silent `catch` → named exception capture (11 sites) | Phase 2 PR #85 already tightened the most-load-bearing one (`Analysis.run_analysis_for_neuron`); 7 other broad `except Exception:` sites in `decoding_algorithms.py` flagged in the original Python audit; partial coverage. |
| 3.3 | `roundn` (Mapping Toolbox) → `round` (7 sites) | Python's `np.round` is built-in. |
| 3.4 | Magic-number `1.96` annotated as `norminv(0.975)` | Python uses `scipy.stats.norm.ppf(0.975)` or has the literal commented inline. |
| 3.5 | Floating-point index `round()` guards | Python uses `int(...)` casts. |
| 3.6 | `%#ok<AGROW>` annotation | Python doesn't have a comparable list-growth lint warning. |
| 3.7 | Handle aliasing in arithmetic | Python uses value semantics. |
| 3.8 | Typo fixes (`sOBj`, `crosscor`, `ciUpper`/`ciLower`) | Section 1 covers each. |
| 3.9 | `findPeaks('minima')` logic fix | Section 1.6. |
| 3.10 | Defensive `fitType` validation in CIF | Python's `nstat/cif.py` and `nstat/linear_cif.py` both validate `fitType ∈ {'poisson', 'binomial'}`. |

---

## Section 4 — Architectural observations (MATLAB AUDIT §6, NOT fixed upstream)

| MATLAB # | Description | Python relevance |
|---|---|---|
| 6.1 | `SignalObj.plot` uses `eval()` with `cell2str` | **N/A** — Python's `nstat/core.py::SignalObj.plot` uses matplotlib's `**kwargs` pattern directly. |
| 6.2 | `CIF.m` uses `assignin('base', ...)` for Simulink | **N/A** — Python's CIF Simulink integration (when present) uses `matlab.engine.eval(...)` or direct function calls.  Documented in `parity/simulink_fidelity.yml`. |
| 6.3 | `simget`/`simset` deprecated API | **N/A** — Python doesn't invoke MATLAB's deprecated simulation accessors. |
| 6.4 | `histc` + `bar(...,'histc')` deprecated | **N/A** — Python uses `np.histogram` and matplotlib `bar(..., align='edge')`. |
| 6.5 | `spectrum.periodogram` and `dspdata.psd` removed from MATLAB | **N/A** — Python uses `scipy.signal.periodogram` and a hand-rolled multitaper; not subject to MATLAB toolkit removal. |
| 6.6 | `nspikeTrain.getSigRep` interval convention | **DOCUMENTED** — Python's `_build_sigrep` (lines 2304-2315 of the pre-refactor `core.py`; now in `_spike_train_impl.py`) explicitly handles the open-right / closed-right switchover at the midpoint, matching MATLAB's `histc` semantics for parity. |

---

## Section 5 — May 2026 audit waves (Phase 0–4 modernization)

The MATLAB AUDIT_REPORT.md preamble references two newer review plans:

1. `docs/superpowers/plans/2026-05-19-nstat-review-action-plan.md`
   — Phase 0–4 fixes: Bernoulli LL, KS U-clamp, DT-correction branch,
   PPAF/PPHF time-indexing, multi-result λ indexing.
2. `docs/superpowers/plans/2026-05-20-comprehensive-codebase-audit.md`
   — May 20 four-phase audit: help-system integrity, cross-document
   drift, sibling-bug hunt, one-command deploy gate.

Both are MATLAB-side plans.  Their MATLAB-relevant content has been
folded into MATLAB v1.4.0.  This Python cross-reference does NOT
recursively walk those plans — to do so would require importing them
into this repo, which violates the independence rule (see
`CLAUDE.md` and `AGENT_GUIDE.md`).  Instead, this xref relies on the
MATLAB `AUDIT_REPORT.md` as the authoritative current-state document.

The relevant items from those review plans (Bernoulli LL, KS U-clamp,
DT-correction) are already addressed in this Python repo through:
- Phase 2 PR #85: `Analysis.GLMFit` adds `stats["loglik"]` as the true
  Bernoulli/Poisson log-likelihood alongside the MATLAB-hybrid `logLL`.

---

## Section 6 — Phase 4.4 follow-up items

These need code-level verification before claiming Python parity with
MATLAB v1.4.0:

1. **§1.9 plotAllVariability ciUpper/ciLower** — Inspect `nstat/core.py`
   `plotAllVariability`.  Confirm the ciBottom branch uses the correct
   bound variable.
2. **§2.2 Q matrix indexing** — Inspect Python's SSGLM Q selector for
   the same off-by-one.
3. **§1.10 burst detection prepend/append boundary** — Add a regression
   test against the MATLAB gold fixture at
   `tests/parity/fixtures/matlab_gold/nspiketrain_*.mat` exercising
   the multi-burst boundary case.
4. **Phase 2 carryover** — 7 broad `except Exception:` blocks remaining
   in `nstat/decoding_algorithms.py` (lines 3485-3493, 6090, 7804) and
   `nstat/trial.py` (1872, 1966, 2815, 2821) should be narrowed
   following the Phase 2 pattern.

Filed as a TODO in `RELEASE_NOTES.md` under v0.3.2.

---

## How this document gets refreshed

Run `tools/parity/diff_against_matlab.py` (PR #83) for the public-API
inventory diff.  This xref is hand-maintained for the bug-level analysis
— it requires reading MATLAB's `% FIX:` comments and reasoning about
Python equivalents, which is not yet automated.

When MATLAB releases a new version with additional `% FIX:` tags,
update this file:

1. Read MATLAB's updated `AUDIT_REPORT.md`.
2. Cross-reference each new fix against the Python source.
3. Update the tables in Sections 1-4 with the new findings.
4. File any PYTHON-INHERITED-BUG items as urgent fixes.

The independence rule (CLAUDE.md) applies: do not import MATLAB code
or paths at runtime; this document is informational only.
