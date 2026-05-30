# Release Readiness — v0.4.0 plan

> **Status:** working / pre-tag.  This file captures the pre-release
> audit and the v0.4.0 release plan.  When the maintainer is ready to
> cut the tag, copy the relevant sections of "What's in scope" into
> `RELEASE_NOTES.md` as the published `## v0.4.0` entry, bump the
> versions, and run `make release-check`.
>
> _Audit date: 2026-05-30._

## Current state

- **Latest tag / pyproject version:** `v0.3.2` (released 2026-05-26 —
  inaugural `nstat.extras` namespace).
- **Latest published GitHub Release:** `v0.3.0` (2026-03-11).  Note the
  gap: `v0.3.2` is tagged in git but no GitHub Release has been
  published for it.  v0.4.0 should publish a GitHub Release too.
- **Commits since `v0.3.2` on `main`:** 19 (notably PR #109 — PP_EM /
  mPPCO_EM full ports; #110 — EM integration docs; #111 — EM
  stabilization; #112 — methods incorporation roadmap).
- **Open PRs (queued for v0.4.0):**

  | PR | What | Base |
  |---|---|---|
  | #113 | Tier 0.1 — full PLDS canonical-gauge identifiability | (EM stack) |
  | #114 | Tier 0.2 — true held-out predictive log-likelihood diagnostic | stacked on #113 |
  | #115 | Tier 1.1 — multivariate (Tao 2018) population time-rescaling GOF | `main` |
  | #116 | CI hygiene — drop wall-clock `generated_on` (kills recurring local failure) | `main` |

## Audit findings — fixed in this iteration

| Issue | Before | After |
|---|---|---|
| `docs/conf.py` Sphinx `release` drifted from `pyproject.toml` | `0.3.1` (one version behind) | `0.3.2` (matches) |
| `AGENT_GUIDE.md` `Package version` drifted | `0.3.1` | `0.3.2` |
| `tests/test_version_sync.py` only checked 2 of 4 version-bearing files | pyproject + CITATION only | + `docs/conf.py` `release` + `AGENT_GUIDE.md` `Package version` (drift fails CI) |

## Audit findings — known, not blocking

- **5 local `make test` failures** in `tests/extras/test_statsmodels_bridge.py`,
  `tests/extras/test_extras_namespace.py::test_nemos_*`, and
  `tests/test_extras_examples.py::*_demo` are `ImportError: cannot import name
  '_lazywhere' from 'scipy._lib._util'` — a scipy ↔ statsmodels/nemos version
  mismatch in the anaconda dev env.  **CI is clean** (it uses pinned envs in
  the `extras-functional` and `extras-dynamax` jobs).  Not release-blocking.
- **Notebook-fidelity date volatility** (also a local-only failure) is fixed
  in PR #116 — once that lands the local suite drops from 6 → 5 unrelated
  failures.

## v0.4.0 — what's in scope

A minor bump (not patch) is warranted: the public API gains the
**EM trainer family** (`fit_point_process_em`, `fit_hybrid_em`,
`fit_linear_gaussian_em`, plus inference + diagnostics), and the
**core GOF** gains `population_time_rescale` /
`PopulationTimeRescaleResult`.

### `nstat.*` (core) additions

- `population_time_rescale(counts_list, lam_per_bin_list, *, n_tau_bins=1)`
  → `PopulationTimeRescaleResult` — multivariate (Tao, Weber, Arai &
  Eden 2018) marked point-process time-rescaling GOF.  Catches
  inter-neuron coupling misfit that the per-neuron univariate KS test
  misses (validated: a synchronous-pair model whose per-neuron
  univariate KS both pass is rejected by the population ground-process
  KS at p≈3e-49).  Pure NumPy/SciPy, additive — `FitResult.computeKSStats`
  and its MATLAB-gold fixtures untouched.  **(PR #115)**

### `nstat.extras.em.dynamax_bridge` (state-space EM) additions

- **EM trainers** (KF_EM / PP_EM / mPPCO_EM equivalents): `fit_linear_gaussian_em`,
  `fit_point_process_em`, `fit_hybrid_em`, plus result dataclasses.
  *(Already merged via #109; first user-facing release is v0.4.0.)*
- **Inference**: `cmgf_poisson_filter` / `cmgf_poisson_smoother`
  (PPDecodeFilter / PP_fixedIntervalSmoother equivalents). *(Also via #109.)*
- **Tier 0.1 — full PLDS identifiability**: `_canonicalize_gauge` pins the
  full `GL(d)` gauge to a canonical form (whiten + SVD-rotate + sign-fix)
  once after EM convergence.  Across-seed `|ΔC|` drops from ~460 (with
  NaN) → ~0.75 (PP) / ~0.15 (hybrid); returned `CᵀC = diag(S²)` to
  machine precision.  **(PR #113)**
- **Tier 0.2 — predictive log-likelihood diagnostic**: `point_process_predictive_ll`
  / `hybrid_predictive_ll` / `PredictiveLogLik`.  True one-step-ahead
  predictive log-likelihood under the fitted parameters — the honest
  convergence / model-comparison metric replacing the surrogate
  smoother trace.  Pure NumPy (no dynamax), runs in base CI.
  **(PR #114)**

### Hygiene / infrastructure

- **`tools/parity/build_notebook_fidelity_audit.py`**: drop wall-clock
  `generated_on` field that caused daily drift on
  `parity/notebook_fidelity.yml`.  **(PR #116)**
- **`docs/changes/whats_new.html`**: published per-iteration change-summary
  tree (added in #114, extended in #115/#116/this PR).
- **`tests/test_version_sync.py`**: now also checks `docs/conf.py` and
  `AGENT_GUIDE.md` — catches version drift in places not previously
  guarded.  **(this PR)**
- **`parity/methods_roadmap.md`**: 2026 review of unported / Python-only
  methods, with explicit tier sequencing.  *(Merged via #112.)*

## Release checklist (when ready to tag)

1. Merge the queued PRs to `main` in dependency order:
   `#113 → #114` (stacked), then `#115` (off main), then `#116` (off main).
2. Resolve merge conflicts in `whats_new.html` (each PR added its own
   entries; superset wins) and `conf.py` `html_extra_path` (identical
   one-line addition — auto-merges).
3. Bump versions to `0.4.0` synchronously in **all four files**:
   - `pyproject.toml` → `version = "0.4.0"`
   - `CITATION.cff` → `version: "0.4.0"`
   - `docs/conf.py` → `release = "0.4.0"`
   - `AGENT_GUIDE.md` → `Package version: 0.4.0.`
4. Copy the "v0.4.0 — what's in scope" section above into
   `RELEASE_NOTES.md` as the published `## v0.4.0 — <date>` entry.
   Delete this `RELEASE_READINESS.md` (or reset to "next planning").
5. Run `make release-check` (= `version-check + freshness-check + test +
   docs-strict + regen`).  In CI everything passes; locally the 5
   unrelated scipy-mismatch failures will appear — fine to tag based on
   green CI.
6. `git tag v0.4.0` + push, then create a GitHub Release (also create a
   retroactive `v0.3.2` GitHub Release to fill the gap noted above).
7. PyPI publish via the existing release workflow (or `python -m build
   && twine upload`).

## Out-of-scope for v0.4.0 (tracked separately)

- **Tier 0.3** — harden PP_EM convergence under weak observability
  (multi-restart selection + init / dynamics regularization).  The
  diagnostic from Tier 0.2 surfaced the `A → 0` collapse; the fix is a
  follow-up iteration that depends on the EM PRs landing first.
- **Tier 2.1** — clusterless decoding bridge
  (`replay_trajectory_classification`).  Flagship extras addition,
  needs LICENSE verification before adding the dependency.
- **Tier 3** — CLDS / NPNR / GP-GLM bridges, once upstream code is
  confirmed available + license-compatible.
