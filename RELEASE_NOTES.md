# Release Notes

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
