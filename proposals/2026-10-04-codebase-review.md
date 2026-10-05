# nstat-python codebase review — 2026-10-04

Multi-agent review of clarity, speed, help/docs freshness, tests and release hygiene,
run against `main @ dcf0de6` on branch `chore/codebase-review-2026-10`.
Full verified finding records (evidence, exact proposed changes, verifier reasoning):
[`2026-10-04-codebase-review-findings.json`](2026-10-04-codebase-review-findings.json).
IDs below (`docs-1`, `perf/F1`, …) index into that file.

## Summary

1. **The published API reference is unclickable, and CI hides it.** On the live site
   every autosummary entry in `api.html` and `extras.html` renders as plain text with no
   link to its docstring page. The cause is `docs/conf.py` `exclude_patterns`, which
   contains `_autosummary`. A fresh strict build (`sphinx -E -W`) emits **153 warnings**:
   72 "stub file not found" and 77 dead glossary links. `deploy-docs` still passes because its `-W` pass runs
   *incrementally* into the warm-up pass's output directory and re-reads nothing, so the
   strict gate is a no-op.
2. **`AGENT_GUIDE.md` Recipes A, B and C crash when run as written** (3 of 3 tested): wrong
   keyword names, a non-existent method name, and wrong result indexing. Recipes D–H have
   not been executed yet.
3. **Every glossary link in the concept pages is dead** (77 links, 36 terms).
   `glossary.md` defines terms as bold paragraphs, but MyST only generates anchors for headings.
4. **Core `nstat` is healthy and mostly faster than MATLAB.** 8 of 10 benchmarked hot paths
   beat MATLAB, and core passes fully on NumPy 2.5.2. The 7 failing tests are all in the
   optional NWB/pynapple interop and are caused by the local environment (see C8).
5. **The optional-dependency handling cannot tell "not installed" from "installed but
   broken".** The package helper (`require_optional`) and the test gates only catch
   `ImportError`, so ABI-broken dependencies surface as raw crashes or test failures
   instead of actionable messages or skips.

Verdicts: **54 findings** from 5 reviewers. Each was re-checked by an independent
adversarial verifier: **44 confirmed, 8 partially confirmed (with corrections), 2 refuted
and dropped**, plus **15 additional issues** the verifiers found. Final tiers: **A = 26 ·
B = 17 · C = 9**.

## Method

- **Measurement first (step 0).** Before any agent ran I measured: `make perf-check` alone
  on the machine, a full suite run with skip and duration census, help-gate runs, version and
  tag facts, and the live `api.html` snapshot. The results went into a shared evidence pack, so
  reviewers audited measured facts instead of guessing.
- **Review → verify pipeline.** Five read-only reviewers (docs, perf, core clarity,
  extras/architecture, tests/release). Each fed an adversarial verifier told to *refute*
  every finding, rerun the cheapest disproving command, and judge parity safety
  against the `matlab_gold` fixtures.
- **Controller reproductions.** I rebuilt the docs in-tree exactly as CI does. That refuted a
  verifier's claim that the committed tree already fails CI, and uncovered the no-op strict
  gate and the 153 hidden warnings. I also overrode four tiers (noted in the JSON).

### Where verification changed the answer

| Claim as first reported | What verification found |
|---|---|
| API-reference fix = one line in `exclude_patterns` (docs-2) | Necessary but not sufficient: it adds 119 "duplicate object description" warnings. It also needs `napoleon_include_init_with_doc = False` (or a template override). |
| Replace dead anchors with `../_autosummary/...` links (docs-4) | MyST mangles that form into `href="#../_autosummary/…"`. Use raw HTML anchors. |
| Hoist the per-pair weight in 3 `spatial_gof` branches (perf/F4) | Bit-identical in all three, but **14% slower** for `pair_correlation` / `cross_pair_correlation` (compact-support kernel). Hoist in `cross_k_inhom` only. |
| `PPSS_EStep` gold fixture is orphaned (perf/F2) | It is covered by `tools/parity/numerical_drift.py` (53/53 entries pass); that tool just isn't wired into any make target or test. |
| Revert `CITATION.cff` to 0.5.7 (tests/F4) | That would break the documented version-sync convention (`RELEASE_NOTES.md:171`). The real gap is that v0.6.0 was never tagged or released. |
| No CHANGELOG exists (tests/F10) | Refuted: `RELEASE_NOTES.md` is the changelog and README links it. |
| `publish.yml` publish-on-tag is undocumented (tests/F2) | It is documented in `RELEASE_READINESS.md`; only discoverability is weak. |
| The latents-GPFA test flake is a file collision (tests/F7) | Root cause is the hard-coded 60 s subprocess timeout under concurrent load. |
| The committed docs tree already fails CI `-W` (verifier) | Refuted for CI's actual procedure (warm-up, then `-W` into the same directory). True for a fresh `-E -W` build, which is the real problem (summary item 1). |

## Measured state

| Area | Measurement |
|---|---|
| Size | `nstat/` 104 modules, ~50.3 kLOC; 104 test files (docs claim ~50 modules / ~24 kLOC) |
| Tests | 972 passed, 7 failed (all environment), 45 skipped, **7 min 22 s**. Five tests take ~310 s; the docs say `make test` takes ~25 s. |
| Perf vs MATLAB | 8/10 paths faster. `pp_decode_filter_linear` 2.77× and `kalman_filter` 1.73× **only because numba was broken in the base env**: in a venv with numba 0.68 (supports NumPy 2.5) they run in 0.0063 s / 0.0004 s, which is **12–17× faster than MATLAB**. Forcing the fallback in that venv reproduces the base-env times (35× / 27.5× numba speedup). So the gap is environmental, not code. |
| Help gates | `helpfile-check` ✅, `readme-check` ✅ (presence and links only). A fresh `sphinx -E -W` build ❌ (153 warnings). |
| Release | `pyproject` 0.6.0 is unreleased; the latest GitHub release is v0.5.7; legacy `v1.0.0-rc1…rc6` tags (March 2026) sort above both |

## Tier A — apply now (no numerics, no public-API change)

**Help / docs**
- **A1 · docs-1**: Fix `AGENT_GUIDE.md` Recipes A, B and C (verified replacement code is in
  the JSON). Execute Recipes D–H and fix any that fail.
- **A2 · glossary (controller finding)**: Turn the 40 bold glossary terms into `###`
  headings so `myst_heading_anchors = 3` generates the 36 missing anchors. Verify there are
  zero `myst.xref_missing` warnings on a fresh `-E` build.
- **A3 · docs-3/5/6/9**: Add docstrings to the 7 undocumented public symbols. Fix the drift
  in `nspikeTrain` (`sampleRate`), `simulate_two_neuron_network` (1 of 10 parameters
  documented) and `nstat_install`.
- **A4 · docs-7, F6, C9, F7, F8, tests/F2, tests/F4**: Update stale prose:
  - `AGENT_GUIDE.md` size and version facts;
  - `nstat/extras/__init__.py` says "five subpackages" (there are 7) and wrongly claims it raises an `ImportError` at import time;
  - the `decoding_algorithms.py` module docstring names three functions that don't exist;
  - the `pyproject.toml` all-extras and numba comments;
  - add a paragraph documenting the extras layout convention;
  - add a pointer from README/CONTRIBUTING to `RELEASE_READINESS.md`;
  - add a note in `RELEASE_READINESS.md` that v0.6.0 is untagged.
- **A5 · docs-8 (LOCAL, git-ignored)**: Update the version, size, latest-release and `make test` runtime facts in `CLAUDE.md`.

**Dead code** (all behavior-preserving)
- **A6 · C7, C8, F4/#256**: Remove the `if False` branch in `fit.py:addParamsToFit` (keep the
  method; it is frozen API). Remove the dead `_NUMBA_AVAILABLE` alias. Close out issue #256
  (unused helpers and variables in the extras demos; the unreachable `_TRANSITIVE_DEPS` entry).

**Tests / tooling**
- **A7 · tests/F1, tests/F8, verifier**: Add one shared test-side probe that catches
  ImportError, ValueError and RuntimeError and skips with an accurate reason, such as
  "numba installed but incompatible with NumPy 2.5". Add an ABI-failure case to
  `test_lazy_import.py`.
- **A8 · tests/F5, perf/F7**: Register pytest markers (`slow`, `matlab`). This removes the
  `PytestUnknownMarkWarning` and marks the ~310 s of long tests. Whether `make test` should
  skip them by default is a separate policy decision (C9).
- **A9 · tests/F6**: Move `test_matlab_rng`'s reference file from the hard-coded `/tmp/randn_ref.mat` to the
  `matlab_gold` fixtures directory and add a capture script (or a tracked TODO).
- **A10 · tests/F7**: Make the extras demo subprocess timeout configurable (env var, generous default).
- **A11 · perf/F3**: Add a warm-up call in `perf_profile.py`. Its profiles are currently 65–97% import noise.
- **A12 · perf/F2**: Add `make numerical-drift-check` wrapping
  `numerical_drift.py --fail-on-drift`, so the existing 53 drift specs become a real gate.
- **A13 · F11, F12**: Make optional-dependency install hints name the right package and extra.

## Tier B — apply with verification

Every change must keep `matlab_gold` fixtures, `make numerical-drift-check` (A12) and the
targeted tests green, and must be bit-identical unless noted. Perf changes are measured
before and after with `make perf-check`.

**Docs infrastructure** (one change set; done only when a fresh build passes)
- **B1 · docs-2, docs-4, controller**: Remove `_autosummary` from `exclude_patterns` and add
  the `napoleon` companion setting. Use raw-HTML anchors for the 2 dead links. **Make the
  strict gate real**: `deploy-docs.yml` and `make docs-strict` should run the strict pass
  with `-E`, or into a fresh output directory. Done when a fresh `sphinx -E -W` build exits
  0. Consider pinning Sphinx and MyST, which are currently unpinned (`>=8` / `>=4`).
- **B2 · docs-11**: Replace `check_readme_links.py`'s regex import scan with an `ast`-based
  one that ignores code spans. Extend the gate to `AGENT_GUIDE.md` and `docs/extras/*.md`.
  Ideally, also *execute* the recipe code blocks.

**Package code** (behavior-preserving)
- **B3 · extras/F1, F2**: Make `require_optional()` report "installed but failed to import
  (likely ABI-incompatible)" separately from "not installed". Migrate the two hand-rolled
  guards (`dynamax_bridge`, `statsmodels_bridge`) to it.
- **B4 · C3**: Route the 29 `print()` calls in `KF_EM`/`PP_EM`/`mPPCO_EM` through `logging`.
  Signatures are unchanged; console output is silent by default.
- **B5 · F10, F5, perf/F6**: Make imports lazier:
  - core `decoding_algorithms` imports `nstat.extras._numba_kernels` at load time (a core→extras dependency) — defer the numba probe to first use;
  - `nstat/__init__` eagerly imports `paper_examples_full` and, through it, `scipy.signal` — make it lazy;
  - move top-level `scipy.stats` imports into the functions that use them.
  Measure each with `python -X importtime`.
- **B6 · F9, C4, C5**: Share private helpers:
  - move `_sigmoid` (3 call sites) into a small internal numerics module;
  - share the time-rescaling loop between `fit.py` and `analysis.py`, passing the early-return guard as an explicit per-call parameter;
  - share the EM tolerance constants, including the `tolRel` that `PPSS_EM` and `PPSS_EMFB` both define.
- **B7 · perf/F1, F4, F5**: Hoist loop-invariant work:
  - `PPSS_EStep` per-k matrix (O(K²) → O(K) inversions);
  - `cross_k_inhom` **only**;
  - the `dynamax_bridge` Newton outer product.
  Gate on numerical drift, gold fixtures and before/after timings.
- **B8 · verifier**: Add **characterization tests** for currently untested production code:
  `_nearestSPD`, `_ztest_pvalue` and the three `*_ComputeParamStandardErrors` have zero test
  references. These tests pin current behavior before anyone touches C2 or C4.
- **B9 · tests/F9**: Add Python 3.10 (the declared minimum) to the CI matrix, verified with
  one manual dispatch run.

## Tier C — recommendations (maintainer decision; numerics, public API, release, environment)

- **C1 · `_glm_deviance` 1e-12 floors** (`analysis.py:169-192`, `fit.py:1212`,
  `cif.py:735`). CLAUDE.md forbids these. Audit them against gold fixtures with extreme
  rates before switching to `np.finfo(float).eps`, because this changes the numbers.
- **C2 · two `_nearestSPD` / `_ztest_pvalue` implementations with different numerics**
  across the decoding families. Decide which behavior is correct per family (after B8) before unifying them.
- **C3 · split `decoding_algorithms.py` (8,268 lines)** into `nstat/decoding/<Family>.py`
  with re-exports, following the existing `nstat/decoding/PPLFP.py` precedent. Every
  `nstat.decoding_algorithms.X` path and `__all__` entry must keep working.
- **C4 · deduplicate the three `*_ComputeParamStandardErrors`** functions (~450 lines each)
  and split `_ppdecode_filter_linear` / `PPHybridFilterLinear` into named predict/update helpers (after B8).
- **C5 · `k_inhom` border-correction pair-asymmetry bug**, plus a second instance at
  `spatial_gof.py:261-286`. Port the fixed space-time pattern; this changes the numbers.
- **C6 · per-neuron analysis restores the entire trial state** (`analysis.py:131-135`)
  on every neuron. This is a real repeated cost, but fixing it means redesigning cache invalidation.
- **C7 · release**: cut v0.6.0, or record why it is held back. Retire or annotate the
  legacy `v1.0.0-rc*` tags; deleting public tags is destructive and needs your approval.
- **C8 · environment**: the anaconda base env has NumPy 2.5.2 with `pandas`, `h5py` and
  `pyarrow` built for NumPy 1.x, and a `numba` that rejects NumPy above 2.4. Upgrading numba
  to ≥ 0.68 alone restores the JIT kernels: a **27–35× speedup** on the two decoding hot paths
  (measured). Consider making the `[numba]` extra more prominent in the install docs. Rebuild those
  packages, or use a dedicated dev environment or lockfile. This is the cause of the
  7 test failures.
- **C9 · test-speed policy**: decide whether `make test` should skip `slow` / `matlab` tests
  by default (saving ~310 s) in favor of an explicit `make test-full`. This trades local
  coverage for speed.
- **C10 · automatic CI triggers** (tests/F3: weekly cron for `extras-functional`, the only
  CI job built to catch optional-dependency ABI breakage). 7 of 9 workflows are manual-only;
  the exceptions are `parity-upstream-watch.yml` (weekly cron) and `publish.yml` (tag push).
  A cron here would follow the upstream-watch precedent but costs CI minutes, so it's your call.

**Dropped:** docs-10 (refuted: the `Trial` aliases are documented inline), tests/F10
(refuted: `RELEASE_NOTES.md` is the changelog).

## Execution notes (from a second review of this plan)

**The environment sets the scope of Tier B.** On the current env (C8) these items cannot be
verified:
- B7's `dynamax` Newton hoist: `dynamax` isn't installed, and no test calls `_ppem_newton_C`.
- The numba fast path behind B5: numba is ABI-broken here.
- B9: needs a CI dispatch, since there is no 3.10 interpreter locally.

The non-invasive option is a **dedicated venv** for this work, leaving the base anaconda env
untouched. Before/after `perf-check` must then run in that same venv.

**Ordering within PR 4.**
- **B8 before perf/F6.** Moving `scipy.stats` imports into functions can turn a use site in
  untested code into a `NameError`. Write the characterization tests first, then check every
  `norm`, `chi2` and `pearsonr` use site.
- **A6 alias removal and B5's numba probe go in one edit** (`decoding_algorithms.py:34-41`).
  Verified: `tests/test_decoding_algorithms_fidelity.py` patches
  `nstat.extras._numba_kernels._NUMBA_AVAILABLE` (not the dead alias) to force the
  fallback path. So **a lazy probe must keep that module attribute patchable**; an
  `lru_cache`'d probe that ignores it would silently disable those tests.
- **The C4 shared time-rescaling helper keeps the existing `1e-12` floor exactly.** Changing
  the floor is C1 (numerics) and must not happen inside a refactor.
- **A3 (`nspikeTrain` docstring) and B1 (`napoleon_include_init_with_doc`)** affect the same
  rendered page. Check the built `nspikeTrain` page shows `sampleRate` after both land.

**Traps in otherwise-correct items.**
- **#256 `_TRANSITIVE_DEPS`: remove the entry; don't re-key it.** Re-keying it correctly
  would make `em_dynamax_demo` require jax and dynamax, which aren't installed, so a
  demo that runs today would be skipped.
- **A7's broad `except` wraps third-party probes only.** If it wrapped `import nstat.extras.X`,
  a real bug in nstat's own code would turn into a skip.
- **PPSS_EStep hoist (B7): prove it with a direct `np.array_equal` before/after on fixed
  inputs.** The `v9_PPSS_EStep` drift spec has deliberately relaxed tolerances (rtol 1e2,
  atol 1e-1) and would pass a real numeric change.
- **`cross_k_inhom` hoist: measure the speedup before claiming it.** Only `pair_correlation`
  was timed, and it got slower.
- **B4 checked:** no committed notebook output captures the EM iteration prints, so moving to
  `logging` causes no notebook-fidelity drift.
- **A13:** run `tests/extras/test_lazy_import.py` (it pins a canonical message) and the
  dpp and hawkes bridge tests.
- **A2:** expect heading-slug mismatches (en-dashes, slashes, e.g. "Kolmogorov–Smirnov test /
  KS plot"). Use explicit `(label)=` targets for the ones that don't match. The `-E` rebuild
  confirms.

**Every one of the 153 warnings now has an owner:** 72 stub warnings → B1; 77 glossary → A2;
2 dead `api.html` anchors (`spatial_point_processes.md:97,179`) → B1. The 2 previously
unassigned ones also go to B1:
- `spatial_point_processes.md:14` links to `parity/methods_roadmap`, which is excluded from
  the build. Point it at the file on GitHub, or drop the link.
- `docs/proposals/2026-06-11-zero-based-indexing.md` isn't in any toctree. Add it to one,
  or mark the page `orphan`.

## Suggested execution

| PR | Contents | Gate |
|---|---|---|
| 1 | Tier A docs and help (A1–A5) | Recipes execute; `helpfile-check`, `readme-check`; fresh `-E` build has no new warnings |
| 2 | Tier A dead code, tests and tooling (A6–A13) | Full suite (same 7 env failures, nothing new); `numerical-drift-check` |
| 3 | B1–B2 docs infrastructure | Fresh `sphinx -E -W` exits 0; then the real strict gate goes into CI |
| 4 | B3–B9 package code | Gold fixtures, numerical drift, targeted tests, perf-check before and after |

Tier C items stay as recommendations until you approve them individually.

## Cost

Review workflow: 10 agents, 1,525,372 subagent tokens, 551 tool calls, ~28 min.
Measurement and synthesis were done in the main session.
