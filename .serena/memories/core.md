# nstat-python — Core

Python port of the MATLAB **nSTAT** neural spike-train analysis toolbox
(Cajigas, Malik & Brown 2012): point-process GLMs, conditional intensity
functions, adaptive decoding (PPAF/PPHF/Kalman/EM), Gaussian-signal (LFP)
analysis, and the 5 canonical paper examples. PyPI name `nstat-toolbox`,
import name `nstat`. Repo `cajigaslab/nstat-python` (GitHub UI shows
`nSTAT-python`), GPL-2.0, default branch `main`. ~50 modules / ~24 kLOC.

## Source map

- `nstat/` — the package.
  - `__init__.py` — public API surface, `__all__` is the contract.
  - `core.py` — `SignalObj` + `Covariate`; `_spike_train_impl.py` — `nspikeTrain`.
  - `trial.py` — `CovariateCollection`/`SpikeTrainCollection`/`Trial`;
    `_trial_config_impl.py` — `TrialConfig`+`ConfigCollection`.
  - `analysis.py` — `Analysis`, `GLMFit`, `psth`. `cif.py` — sympy-backed CIFs.
  - `fit.py` — `FitResult`/`FitResSummary`. `decoding_algorithms.py` — PPAF/PPHF/Kalman/EM.
  - `data_manager.py` — figshare dataset fetch; `NSTAT_OFFLINE=1` forces offline.
  - `compat/matlab/` — MATLAB-style import aliases (no runtime coupling to MATLAB repo).
  - `extras/` — Python-only, opt-in extensions (see below).
  - MATLAB-style shim files at top level (`nspikeTrain.py`, `SignalObj.py`,
    `CovColl.py`, `TrialConfig.py`, `DecodingAlgorithms.py`, `ConfidenceInterval.py`,
    `FitResult.py`, `FitResSummary.py`, `Covariate.py`, `ConfigColl.py`) — thin
    re-export shims kept for MATLAB-user orientation; **never delete/rename**,
    `tests/test_api_surface.py` imports them directly.
- `examples/paper/` — 5 canonical paper-example scripts + `manifest.yml`
  (this is the *only* place new paper examples go — not `examples/nSTATPaperExamples/`).
- `docs/` — Sphinx + MyST; `docs/figures/exampleNN/` is the canonical figure tree.
- `notebooks/` — 30+ Jupyter notebooks (many are MATLAB-help ports).
- `parity/` — MATLAB↔Python audit manifests + generated report; `tests/parity/fixtures/matlab_gold/*.mat`
  is the canonical ground truth for numerical parity.
- `tools/` — artifact regenerators (gallery, parity report, notebook fidelity) + release-gate scripts.
- `.github/workflows/` — `ci.yml`, `deploy-docs.yml`, `helpfile-check.yml`,
  `readme-check.yml`, `notebook-full-fidelity.yml`, `performance-parity.yml`,
  `parity-check.yml`, `parity-upstream-watch.yml`, `publish.yml`.
- `.claude/agents/factory/` — local (git-ignored) subagent chain (researcher →
  planner → architect → builder → verifier → validator) used for non-trivial changes.

## Project-wide invariants

- **Uncoupled from the MATLAB repo (hard rule):** no cross-repo runtime imports,
  no MATLAB-repo URLs/`matlab_source` paths in manifests/YAML. `parity/` and
  `compat/matlab/` record audit results / import aliases *within* this package
  only. A local MATLAB checkout (set via `NSTAT_MATLAB_PATH`, default: a
  sibling `nstat` checkout) may be consulted as reference but never cited in
  committed code/docs. `tests/parity/fixtures/matlab_gold/*.mat` is the sole
  canonical parity authority.
- **Exact-mirror parity contract** (binding on every change): every MATLAB
  function/class/method has an exact Python mirror — same name, same signature,
  same numerical output (gold-fixture tolerance), same figure content and
  appearance (colormap, line styles, overlay conventions). Python MAY add
  extra functions/notebooks/examples with no MATLAB counterpart (encouraged).
  Folder layout/build tooling is free to be Python-native.
- **Tests serve parity, not the reverse:** if a test fails because a change
  made Python *more* faithful to MATLAB, redesign the test, don't revert the
  parity work (except: `matlab_gold/*.mat` fixture failures always mean the
  code is wrong, never the fixture).
- MATLAB-style class/method names are the public API surface — never
  Pythonize `nspikeTrain`, `GLMFit`, `computeKSStats`, etc.
- Ruff/black/mypy are **not enforced** — `make format`/`lint`/`typecheck` are
  no-ops unless the tool happens to be installed locally; CI doesn't gate on them.
- Drift-checked generated artifacts must never be hand-edited (`parity/report.md`,
  `parity/notebook_fidelity.yml`, `docs/paper_examples.md`,
  `docs/figures/manifest.json`, `docs/_build/`, `docs/_autosummary/`) — run
  `make regen` and commit the diff instead.

## Focused memories

- `mem:tech_stack` — Python version, dependencies, optional-extras groups,
  version pins that matter (read when adding a dependency or an `extras` bridge).
- `mem:suggested_commands` — actual install/test/lint/docs/release commands
  from the Makefile (read before running any command, don't guess flags).
- `mem:conventions` — code style, naming, gotchas specific to this codebase,
  including the MATLAB API footgun table and hot-path perf conventions
  (read before writing/editing `nstat/` code).
- `mem:task_completion` — the exact gate to run before calling a change done
  (read at the end of any code change, before committing).
- `mem:serena_usage` — Serena MCP usage rules + gotchas (name-path syntax,
  edit vs read tool map, worktree-edit hazard, rename_symbol misses string
  refs; read before using Serena's semantic tools on this repo).
