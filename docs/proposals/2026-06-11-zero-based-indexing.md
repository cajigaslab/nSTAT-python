# Proposal — adopt 0-based indexing throughout (breaking, → v0.5.0)

> **Status:** drafted 2026-06-11; pending Phase 4 completion.
> **Owner:** TBD.
> **Type:** Architecture / public-API contract change.
> **Lands after:** the Phase 4 PR queue (API-trap, Collections semantics,
> CIF unification, Fitting/decoding, `AUDIT_REPORT` refresh).
> **Effort:** ~2 days focused + ~1 day notebook sweep.
> **Risk:** HIGH — touches every public method that accepts a selector +
> every notebook + every gold-fixture comparison.

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

## Migration path (deprecation cycle)

### Phase A — this PR (v0.5.0)

1. **Add `nstat/_compat.py::warn_matlab_one_based`** — a soft-deprecation
   helper:

   ```python
   def warn_matlab_one_based(idx: int, n: int, *, method_name: str) -> int:
       """Detect a likely MATLAB-style 1-based call and emit a warning.

       Heuristic: ``idx == n`` is now the historical "last element via
       1-based" pattern.  Emit a DeprecationWarning with migration hint,
       and ALSO clip the index back to 0-based for one minor cycle so
       existing code keeps working.  In v0.6.0 this helper goes away
       entirely and the call sites silently use idx as Python 0-based.
       """
       if 1 <= idx == n:
           warnings.warn(
               f"{method_name}({idx}) called with what looks like a MATLAB-"
               f"style 1-based selector.  Python uses 0-based indexing; "
               f"pass {idx-1} instead.  This compatibility shim will be "
               f"removed in v0.6.0.",
               DeprecationWarning,
               stacklevel=3,
           )
           return idx - 1
       return idx
   ```

2. **Wire `warn_matlab_one_based(...)` into each affected public method**
   as the FIRST line:

   ```python
   def getNST(self, idx: int) -> nspikeTrain:
       idx = warn_matlab_one_based(idx, self.numSpikeTrains, method_name="getNST")
       return self.nstrain[idx]
   ```

3. **Sweep internal `idx - 1` patterns** — eliminate entirely.  Internal
   code is uniform 0-based after this.

4. **Update gold-fixture comparisons** via the `conftest.py` adapter.

5. **Update notebooks** — 50 mechanical edits.

6. **Update `AGENT_GUIDE.md`** — add the "Indexing" paragraph:

   > **Indexing.** Python is 0-based everywhere — internal storage,
   > loops, and the public API.  MATLAB users porting code subtract 1
   > from selectors (e.g. `coll.getNST(0)` is the first train,
   > equivalent to MATLAB `coll.getNST(1)`).  This is the only
   > convention difference between the two ports; everything else
   > mirrors MATLAB names verbatim.

7. **Add `tests/test_indexing_convention.py`** — grep-based scan that
   fails CI on any new `idx - 1` or `range(1, n+1)` pattern in `nstat/`.

8. **`RELEASE_NOTES.md` + version bump** — entry under "Breaking
   changes" with a one-page migration guide.

### Phase B — v0.6.0 (later, no code work)

1. **Remove `nstat/_compat.py::warn_matlab_one_based`** entirely.
2. **Remove the wrappers** from the ~30 public methods.  Each becomes
   a clean 0-based function.
3. **Remove the deprecation paragraph** from `RELEASE_NOTES.md`
   (covered by v0.5.0).

After v0.6.0, nothing in the codebase mentions 1-based vs 0-based ever
again.

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
