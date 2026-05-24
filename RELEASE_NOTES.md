# Release Notes

## v0.3.2 — unreleased

Inaugural release of the **`nstat.extras` namespace** — a monorepo addon
space (modeled after `scikit-learn-contrib`) for Python-only features
that have no MATLAB nSTAT counterpart.  The core `nstat.*` namespace
remains under the strict MATLAB-parity contract; `nstat.extras.*` is
free to evolve at minor-release speed and depends on opt-in third-party
libraries declared in `pyproject.toml`'s `[project.optional-dependencies]`.

### New: `nstat.extras` subpackages

- **`nstat.extras.interop`** — `Trial` / `SpikeTrainCollection` / `nspikeTrain`
  converters for the wider Python neuro-data stack.
  - `interop.neo` — `to_neo_spiketrain`, `from_neo_spiketrain`, `to_neo_segment`
    (install `[neo]`).
  - `interop.pynapple` — `to_pynapple_ts`, `to_pynapple_with_support`,
    `from_pynapple_ts`, `to_pynapple_tsgroup` (install `[pynapple]`).
  - `interop.nwb` — `nwb_units_to_collection`, `read_nwb_path` with
    `obs_intervals` / `time_window=` support and explicit fallback
    warning when neither is available (install `[nwb]`).
- **`nstat.extras.validation`** — Python-side cross-validation bridges
  that triangulate nstat's MATLAB-faithful estimates against
  independent reference implementations.
  - `validation.nemos_bridge.cross_validate_poisson_glm` against
    Flatiron's NeMoS Poisson GLM (install `[nemos]` or `[test-parity]`).
  - `validation.pykalman_bridge.cross_validate_kalman` against
    pykalman.  Includes the documented AUDIT D3 smoother-gap baseline
    (install `[test-parity]`).
- **`nstat.extras.metrics`** — modern spike-train distance metrics.
  - `metrics.spike_distances.{isi,spike,spike_synchronization,pairwise_spike_distance_matrix}`
    via PySpike (install `[metrics]`).

### New: optional-dependency groups in `pyproject.toml`

- `[neo]` (neo + quantities)
- `[pynapple]` (pynapple ≥ 0.7)
- `[nwb]` (pynwb ≥ 2.8)
- `[metrics]` (pyspike ≥ 0.8)
- `[nemos]` (nemos ≥ 0.2)
- `[test-parity]` (nemos + pykalman + statsmodels + nitime)
- `[all-extras]` — install every functional extras module at once

### Docs

- `README.md` adds a "Related Python projects" section with the install-
  command matrix and pointers to ecosystem peers (SpikeInterface,
  Elephant, Dynamax, ssqueezepy).
- `AGENT_GUIDE.md §5.6` updated to name a concrete library / extras
  module for every "what nstat is NOT" scope gap.
- `parity/integration_opportunities.md` (added prior PR) provides the
  upstream-metadata audit driving these decisions.

### Independence

Every bridge is **Python-side only**.  No new coupling to the MATLAB
`cajigaslab/nSTAT` repository is introduced — confirmed by
`tests/test_cleanroom_boundary.py` and a regression test in
`tests/extras/test_extras_namespace.py::test_extras_independence_no_matlab_runtime_imports`.

---

## v0.3.1 — 2026-05-21

Post-audit cleanup release.  No public API breakage; deprecation warnings
introduced for legacy ``FitResult.lambda_*`` aliases (see v0.3.0 notes
below).

### Bug fixes

- **`Analysis.GLMFit`** now returns a ``GLMFitResult`` dataclass with both
  named-field access (``result.AIC``, ``result.loglik``) and tuple-unpack
  back-compat (``lambda_sig, b, dev, stats, AIC, BIC, logLL, dist = result``).
- **`SpikeTrainCollection.toSpikeTrain`** maxTime now sums per-train
  durations.  Homogeneous-collection behaviour unchanged.
- **`SpikeTrainCollection.getNST`** is non-destructive (copy-then-resample).
- **`SpikeTrainCollection.addSingleSpikeToColl`** is amortised O(1) per
  add (was O(n²)).
- **`Analysis.run_analysis_for_neuron`** narrows broad ``except Exception``
  to ``(LinAlgError, ValueError, ZeroDivisionError, FloatingPointError)``
  with a ``RuntimeWarning`` instead of silent failure.
- **`simulate_cif_from_stimulus`** added to the public API
  (``nstat.simulate_cif_from_stimulus``).  Closes three previously-broken
  example scripts (``examples/basic_data_workflow.py``,
  ``examples/fit_poisson_glm.py``, ``examples/simulate_population_psth.py``).
- **`matplotlib.use("Agg")`** no longer called at module import time in
  any ``nstat/`` module.  User-chosen backends survive ``import nstat``.
- **`nstat.fit._matplotlib_version_tuple`** parses release-candidate /
  development version strings (e.g. ``3.9.0rc1``) without crashing.
- **`nspikeTrain.computeStatistics`** — burst-statistics double-``+1.0``
  is documented inline as intentional MATLAB-parity, not a bug.
- **`SpikeTrainCollection.ssglm`** and **`ssglmFB`** now accept an
  optional ``rng: np.random.Generator`` parameter for reproducible
  Q0-jitter.  Legacy ``np.random.rand`` calls removed.

### New features

- **`Analysis.GLMFit`** exposes a true Bernoulli/Poisson log-likelihood at
  ``result.loglik`` (and ``stats["loglik"]``).  AIC = -2·loglik + 2·k
  now holds.  Legacy MATLAB-style hybrid retained at ``result.logLL``.
- **`nstat.apply_plot_style`**, **`nstat.set_plot_style`**, and
  **`nstat.get_plot_style`** exported from the package root.  Paper-example
  scripts accept ``--plot-style {modern,legacy}`` (default ``legacy``).
- **`ensure_example_data`** honours ``NSTAT_OFFLINE=1`` and emits an
  actionable error pointing at ``nstat-install --download-example-data``.
- **`tools/parity/diff_against_matlab.py`** (Phase 4.1) — new tool to
  diff against the upstream MATLAB toolbox.

### Refactors (non-breaking)

- **`nstat.core`** split: ``nspikeTrain`` extracted to
  ``nstat/_spike_train_impl.py``.  ``core.py`` re-exports.
- **`nstat.trial`** split: ``TrialConfig`` + ``ConfigCollection``
  extracted to ``nstat/_trial_config_impl.py``.  ``trial.py`` re-exports.
- **`FitResult`** lambda aliases (``lambda_obj``, ``lambda_model``,
  ``lambdaCov``, etc., 9 historical aliases) emit ``DeprecationWarning``.
  ``lambda_signal`` (canonical) and ``lambdaSignal`` (MATLAB-style) are
  retained without warning.
- **`scipy.signal.filtfilt`** lazy-imported in the one method that
  uses it.
- Type-hint hygiene: ``float | None`` over ``float = None`` in public
  signatures.

### Documentation

- **`AGENT_GUIDE.md`** (toolbox-usage guide for AI agents) added at
  repo root.
- **`CLAUDE.md`** (repo-maintenance playbook) added as a local-only
  config (gitignored); see ``AGENT_GUIDE.md`` for the shared agent
  guidance.
- **Notebook tracker-stub banners** — 11 helpfile-derived notebooks
  prefixed with a markdown banner explaining they are MATLAB-parity
  scaffolding (not executable Python tutorials).
- **Notebook link paths** in ``docs/Examples.md`` and
  ``docs/ClassDefinitions.md`` corrected (``../notebooks/...``).
- **PubMed citation** hyperlinked in ``README.md``.
- **CITATION.cff** gained ``version`` and ``date-released`` fields.

### Repository structure / reproducibility

- **Branch protection on ``main``** now requires
  ``paper-gallery-artifacts``, ``parity-report-artifacts``, ``docs-build``
  status checks (closes the largest pre-v0.3.1 reproducibility gap).
- **CI cache key** in ``notebook-full-fidelity.yml`` bumped to include
  ``hashFiles('nstat/data/manifest.json')`` so figshare-dataset upgrades
  invalidate stale caches.
- **Python version sync** — ``deploy-docs.yml`` and ``docs-build``
  both pin to 3.12.
- **`MPLBACKEND=Agg`** set at workflow level in ``ci.yml``.

### Removed / Deprecated

- ``nstat/_compat.py`` deleted (was a dead helper with zero callers).
- ``examples/paper/figures/`` directory marked deprecated (canonical
  figure tree is ``docs/figures/``).

---

## v0.3.0 — 2026-03-11 ("Full MATLAB Parity")

Initial public release after the 2026-03 cross-toolbox audit.  See
``AUDIT_REPORT.md`` for the audit catalogue (466 of 484 MATLAB methods
ported, 13 MATLAB-side bugs discovered and filed upstream).
