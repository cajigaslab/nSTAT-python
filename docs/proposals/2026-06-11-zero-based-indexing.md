# Proposal — adopt 0-based indexing throughout (breaking, → v0.5.0)

> **Status:** refined 2026-06-11 after Phase 4 closeout.  Awaiting `#177`
> merge + deploy-docs before execution.
> **Owner:** TBD.
> **Type:** Architecture / public-API contract change.
> **Lands after:** PR #175 (CIF), #176 (Fitting/decoding), #177 (M20
> docs) — all merged or queued.
> **Effort:** ~6-7 focused hours; single working day.
> **Risk:** HIGH — touches every public method that accepts a selector +
> every notebook + every gold-fixture comparison.
> **Strategy (decided 2026-06-11):**
>   - **Completely silent flip** — no `RELEASE_NOTES.md` entry, no
>     `AGENT_GUIDE.md` user-facing paragraph.  Version bump from 0.4.x →
>     0.5.0 happens but the indexing shift itself is not surfaced.
>   - **Single PR** containing all six subsystems (test guard, public-
>     API flip, internal sweep, gold-fixture adapter, notebook sweep,
>     version bump).
>   - **No deprecation cycle** — drop `warn_matlab_one_based` from the
>     original draft.  Hard-flip.  Parity tests + grep CI guard catch
>     misses.
>
> ## Current survey (2026-06-11, post Phase 4 merges)
>
> - `158` `[... - 1]` index-arithmetic sites in `nstat/`
> - `84` `range(1, ...)` loops in `nstat/`
> - `68` explicit 1-based selector translations in
>   `fit.py`/`trial.py`/`analysis.py`/`_spike_train_impl.py`
> - Densest cluster: Granger causality block in
>   `nstat/analysis.py:1235-1322` (~25 sites)
>
> Net change estimate: **+145 LOC, −200 LOC** (smaller than original
> draft's ±200/250 because the deprecation shim was dropped).

---

## Summary

The Python port currently uses MATLAB-style **1-based** selectors at the
public API (`coll.getNST(1)`, `fit.getCoeffsWithLabels(3)`) while
storing arrays 0-based internally.  Every boundary between conventions
is a place where bugs hide; this proposal eliminates the boundary by
adopting **Python-native 0-based throughout**.

**Trade-off:** MATLAB users porting code now subtract 1 from selectors.
Documented in one paragraph in `AGENT_GUIDE.md`; the rest of the API
mirrors MATLAB names verbatim.  This is a breaking change to the
`nstat.*` parity contract → version bump to **v0.5.0**.

## Core principle

The Python port uses Python's native 0-based indexing throughout —
internal storage, loops, and the public API.  MATLAB users porting code
subtract 1 when calling Python equivalents; that's the only place the
difference matters, and it's documented once.

## Hard rules

| # | Rule | Enforcement |
|---|---|---|
| HR1 | All public-API selectors are 0-based. | API docs + grep test |
| HR2 | All loops use Python idioms: `range(n)` or `enumerate(...)`. **No** `range(1, n+1)`. | Grep test fails CI |
| HR3 | No `arr[idx - 1]`, no `idx - 1` anywhere outside the compat shim. | Grep test fails CI |
| HR4 | Return values are 0-based indices into Python collections. **No** `+ 1` "to match MATLAB". | Grep test fails CI |
| HR5 | `_matlab_colon(start, step, stop)` is preserved — that's float-stable range generation, a *different* problem from indexing. | Allowlisted by name |

## Scope

### Public-API methods that change behavior (~30 sites)

| Module | Methods | New behavior |
|---|---|---|
| `nstat/trial.py` | `getNST`, `getCov`, `getEnsCovMatrix`, `getHistMatrices`, `getNeighbors`, `getNeuronIndFromName`, `setNeighbors`, `getDesignMatrix`, `getCovIndFromName` | All selectors 0-based |
| `nstat/fit.py` | `getCoeffsWithLabels`, `getHistCoeffsWithLabels`, `plotCoeffs`, `plotResults`, `evalLambda`, `getSubsetFitResult` | All `fit_num` / config-index args 0-based |
| `nstat/analysis.py` | `GLMFit`, `runAnalysisForNeuron`, `RunAnalysisForAllNeurons`, `computeKSStats` | `neuronNumber` is 0-based |
| `nstat/core.py` | `getCov`, `getValueAt`, `findNearestTimeIndex`, `findNearestTimeIndices` | Returns 0-based indices |
| `nstat/_spike_train_impl.py` | `getEnsembleNeuronCovariates` | `neighbor` args 0-based |
| `nstat/decoding_algorithms.py` | All `neuronIndex` / `selectorIndex` args | 0-based |

### Internal sweep (~120 sites)

`grep -rn "idx - 1\|index - 1\|range(1," nstat/` returns ~120 hits.  Each
becomes idiomatic Python: `arr[idx]`, `range(n)`, `enumerate(...)`.

### Notebooks (~50 sites across 30 notebooks)

`getNST(1)` → `getNST(0)`, `getCoeffsWithLabels(3)` →
`getCoeffsWithLabels(2)`, etc.  Mechanical edit.

### Gold fixtures

`tests/parity/fixtures/matlab_gold/*.mat` files store MATLAB's 1-based
outputs (e.g. `neuronNumbers` arrays, `selectorArray` payloads).
Add one helper in `tests/conftest.py`:

```python
def matlab_to_python_indices(arr):
    """Translate MATLAB-stored 1-based index arrays to Python 0-based at the test boundary."""
    return np.asarray(arr, dtype=int) - 1
```

Tests calling `loadmat(...).get("neuronNumbers")` route through this
helper.  **Production code never sees the translation** — it's a
one-line test-boundary adapter.

## Execution sequence (single PR, six subsystems)

| # | Subsystem | Files | Net LOC | Risk |
|---|---|---|---|---|
| 1 | **Test guard** | `tests/test_indexing_convention.py` (new), `tests/conftest.py::matlab_to_python_indices` helper | +75 | LOW — red-bar; no production code |
| 2 | **Public-API signatures flip** | `nstat/fit.py`, `nstat/trial.py`, `nstat/analysis.py`, `nstat/_spike_train_impl.py`, `nstat/decoding_algorithms.py` (default args `= 1` → `= 0`; strip first-line `- 1`) | ~−80 | HIGH — callsites break until subsystem 4 lands |
| 3 | **Internal body sweep** | same modules + `nstat/core.py`, `nstat/cif.py` (`[idx - 1]` → `[idx]`; `range(1, n+1)` → `range(n)`) | ~−120 | MEDIUM — parity tests catch most regressions |
| 4 | **Gold-fixture boundary** | `tests/conftest.py` adapter goes live; every `loadmat(...)["neuronNumbers"]`-style call routes through it | ~+30 | MEDIUM — fixture mismatches loud and obvious |
| 5 | **Notebook sweep** | ~30 notebooks: `getNST(1)` → `getNST(0)`, `getCoeffsWithLabels(2)` → `getCoeffsWithLabels(1)`, etc. | mechanical | MEDIUM — `notebook-parity-core` re-execution catches misses |
| 6 | **Version bump** | `pyproject.toml` + `nstat/__init__.py` (0.4.x → 0.5.0). **No `RELEASE_NOTES.md` entry, no `AGENT_GUIDE.md` paragraph** — silent flip per maintainer direction. | +2 | LOW |

### Step-by-step

1. Land subsystem 1 first (red bar — tests fail on every existing
   `- 1`).  Allowlist starts populated; sweep drains entries.
2. Subsystem 2 — signature flip across ~30 public methods.
3. Subsystem 3 — body sweep, draining allowlist site by site.
4. Subsystem 4 — gold-fixture adapter goes live; expected outputs flip
   from MATLAB 1-based to Python 0-based at the test boundary.
5. Run full `make test`.  Any parity-test cascade indicates a missed
   internal site — fix and re-run until clean.
6. Subsystem 5 — notebooks.  `make regen` for gallery + fidelity.
7. Subsystem 6 — version bump.  No release notes.
8. Open PR, fire `ci.yml`, fire `notebook-full-fidelity.yml` once.

### Triage decisions (locked)

| Pattern | Treatment |
|---|---|
| `_matlab_colon(start, step, stop)` | **Keep** — range-generation, not indexing.  Allowlisted in grep test. |
| `+ 1` in return values (e.g. `analysis.py:1235`) | **Strip** — no-op after `np.flatnonzero` etc. |
| `fit_num: int = 1` defaults | **Flip** to `= 0`.  Notebook and example callsites flip in lockstep. |
| `b_{idx + 1}` label format (`fit.py:1098`) | **Keep** — user-facing label string for plots; MATLAB-faithful.  Allowlist with a comment explaining the `+ 1` is for display, not indexing. |
| Granger causality block (`analysis.py:1235-1322`) | **Single-function rewrite** — ~25 sites, all `neighbor - 1` / `neuron_index - 1`.  M4/M5 cross-port; rewrite as a clean 0-based block. |

## Files changed (preview)

```
 nstat/_compat.py                                | +28 (new file)
 nstat/trial.py                                  | ~60 (sweep + wrappers)
 nstat/fit.py                                    | ~40
 nstat/analysis.py                               | ~25
 nstat/core.py                                   | ~15
 nstat/_spike_train_impl.py                      | ~12
 nstat/decoding_algorithms.py                    | ~80 (large file, many sites)
 nstat/cif.py                                    | ~8
 tests/conftest.py                               | +15 (matlab_to_python_indices)
 tests/test_indexing_convention.py               | +60 (new grep scan)
 tests/test_api_surface.py                       | ~30 (add 0-based behavior assertions)
 tests/test_workflow_fidelity.py                 | ~20
 notebooks/*.ipynb (30 files)                    | ~50 mechanical edits
 AGENT_GUIDE.md                                  | +1 paragraph
 RELEASE_NOTES.md                                | +1 release section
 pyproject.toml + nstat/__init__.py              | version 0.4.x → 0.5.0
 ────────────────────────────
 ~ +200 LOC, ~ -250 LOC (net negative — boundary code goes away)
```

## Test plan

- [ ] `make test` — all 537 tests still pass under the compatibility shim
- [ ] `tests/test_indexing_convention.py` — passes with empty allowlist
- [ ] Every notebook regenerated under 0-based; visual parity baselines
      should be unchanged
- [ ] `make docs-strict` — clean
- [ ] Manual smoke: run `examples/paper/example0[1-5]_*.py` with the new API
- [ ] `DeprecationWarning` emitted exactly once per `coll.getNST(1)`-style
      call (test via `pytest.warns`)

## Why this lands AFTER Phase 4

Phase 4 (API-trap PR, Collections semantics, CIF unification,
Fitting/decoding fixes, `AUDIT_REPORT` refresh) is **additive** within
the existing 1-based contract.  Doing it first means:

1. Phase 4 ships under the stable 0.4.x line — no premature breaking
   changes.
2. Users who depend on the current API get Phase 4 fixes without
   disruption.
3. This PR has fewer conflicts: every Phase 4 callsite is already
   settled when the indexing sweep hits.
4. The 0.5.0 release notes can list **all** of Phase 1-4 fixes *plus*
   this convention shift in one bumpable release.

## Risk + mitigations

| Risk | Mitigation |
|---|---|
| User code breaks on upgrade | `DeprecationWarning` + clipping shim for one minor cycle |
| Internal sweep misses a site | `tests/test_indexing_convention.py` grep scan catches every `idx - 1` + `range(1, n+1)` |
| Gold fixtures get translated wrong | Single helper in `conftest.py`; tests fail loudly on mismatch |
| Notebook sweep introduces errors | Each notebook re-executes in CI under `notebook-parity-core` |
| Documentation goes stale | One paragraph in `AGENT_GUIDE.md`; the change is "stop using `idx - 1`" |

## Decision log

- **Why not keep 1-based?** Boundary-mixing within function bodies is
  the actual bug source.  Removing the boundary eliminates the
  *category* of bugs, not individual instances.
- **Why not type-level encoding (`OneBasedIndex`/`ZeroBasedIndex`)?**
  Runtime no-op + tooltip docs aren't worth the complexity if there's
  only one convention left.
- **Why a deprecation cycle instead of hard break at 0.5.0?** Lowers
  the migration cliff.  Users see the warning, fix their code, upgrade
  smoothly.
- **Why not preserve MATLAB-port ergonomics?** Per maintainer
  direction: "we don't need to let the users know about the zero versus
  one indexing."  Pick the host language's convention and document the
  one-time difference.
