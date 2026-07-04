# nstat-python — Conventions

## Naming
- Use `nSTAT` (CamelCase) for the toolbox/paper/repo name; `nstat` (lowercase)
  only for Python code/imports/PyPI name/CLI commands.
- MATLAB-style class names (`nspikeTrain`, `nstColl`, `CovColl`, `SignalObj`,
  `TrialConfig`) and method names (`GLMFit`, `computeKSStats`, `setMinTime`)
  are the public API — never Pythonize. Private internals get a leading
  underscore (`_spike_train_impl.py`) and are re-exported from
  `core.py`/`trial.py`.

## Code style
- `from __future__ import annotations` everywhere; type hints use `X | None`.
- NumPy-style docstrings with a brief MATLAB cross-reference for ported methods.
- Lazy-import SciPy/scikit-image/heavy deps inside the functions that need them.
- Never add `matplotlib.use("Agg")` at module top in `nstat/` — must not
  clobber the user's backend on `import nstat`.
- New randomness: `np.random.default_rng(seed)`, never legacy `np.random.rand`.
- No `1e-12` clipping floors on log-likelihood/numerics compared to MATLAB —
  use `np.finfo(float).eps` (MATLAB's `eps` is `2.22e-16`; the wrong floor
  shifts logL by tens of units). Discovered v10 iter 47 after a 2-builder hunt.
- Time vectors in seconds, sample rates in Hz, spike times sorted ascending
  on construction. For MATLAB `start:step:stop` semantics use the
  `_matlab_colon` helper, not `np.arange` (avoids float-length drift).

## Design pattern: core vs extras
- `nstat.*` = MATLAB-parity contract, stable, removals/renames need a major bump.
- `nstat.extras.*` = Python-only, opt-in, free to evolve/break across minors.
- Decision rule: exists in MATLAB or has a `parity/manifest.yml` entry → core;
  Python-only, or depends on non-core libs (PyTorch/SpikeInterface/MNE/Neo/JAX),
  or experimental → extras. Each extras module declares its own optional-dep
  group in `pyproject.toml`. MATLAB-independence rule applies to extras too.

## Recurring MATLAB-API gotchas (footguns to know before touching related code)
| Footgun | Fix |
|---|---|
| `CIF(beta, {'1', 'x1'}, ...)` constructor silently fails | use `{'one', 'x1'}` — `sym('1')` is numeric, breaks symbolic eval (`cajigaslab/nSTAT#92`) |
| `Analysis.RunAnalysisForAllNeurons` returns cell-of-`FitResult` | not a `FitResSummary`; iterate the cell array |
| `SignalObj.periodogram` returns `cell{struct(Pxx,f),...}` per channel | not a `SignalObj` |
| `SignalObj.autocorrelation` broken on newer MATLAB | use raw `xcorr` (`cajigaslab/nSTAT#93`) |
| `nstColl.plot` injects a "Spike Train Raster" axis title | `ax.set_title("")` after the call |
| `PPHybridFilter` declares 7 outputs, assigns 3 | call as `[S_est, X, W] = …` (`cajigaslab/nSTAT#91`) |
| MATLAB `eps` ≠ Python `1e-12` | see code-style note above |

## Hot-path performance conventions (v12+, `parity/performance_baseline.yml`)
- Drop defensive per-step `np.linalg.cond`/SVD guards with no MATLAB equivalent.
- Vectorize per-bin `np.sum` loops with paired `np.searchsorted`.
- Inline trivial wrapper calls (e.g. `_symmetrize`) inside hot loops.
- Per-step `np.linalg.solve` on tiny (2x2/4x4) matrices is dispatch-bound —
  closing that gap needs Numba/Cython, tracked as a future-version candidate.
- Cache CIF `lambdify` symbolic compilation on construction, not per-call.
- `CIF(beta, Xnames, ...)` requires `len(beta) == len(Xnames)`; for a constant
  term use a `'one'` intercept variable (`cajigaslab/nSTAT#92`).

## Image-pair parity audit heuristic
`tools/parity/image_content_audit.py` — a figure panel is "degenerate" only if
ALL three hold: `non_white_pct < 1.5%`, `distinct_intensity_count < 20`,
`dark_pixel_count < 500`. Do not flag sparse-but-real plots (trajectory lines,
scatter, axis-off schematics) using non-white-pixel % alone — text/AA edges
push distinct-intensity count past 50 for any panel with real glyphs.

## Agent factory workflow (local, git-ignored)
Non-trivial changes go through `.claude/agents/factory/`: researcher (read-only)
→ planner → **checkpoint** → architect → **checkpoint** → builder (one
subsystem per invocation: `package`=`nstat/**`+`tests/**`, `examples`, `docs`,
`notebooks`, `tooling`=`tools/**`+`Makefile`+workflows+`parity/*.yml`) →
verifier → validator → **checkpoint before commit**. Trivial fixes (typo,
docstring) skip the chain entirely. Started via `/feature <description>`.
