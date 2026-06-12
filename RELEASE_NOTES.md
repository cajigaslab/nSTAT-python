# Release Notes

## v0.5.0 — 2026-06-12

Internal consistency improvements and audit fixes.

## v0.4.6 — 2026-06-10

Onboarding deepening + post-v0.4.5 hygiene. **No public API changes, no
behavior changes** &mdash; every importable symbol behaves identically to
v0.4.5.

### Added

- **`notebooks/00_getting_started.ipynb`** &mdash; an executable
  end-to-end mirror of `intro.html`. Thirteen cells: build a spike
  train, group into a population, fit a Poisson GLM, check it with the
  time-rescaling KS test, then decode a hidden stimulus from a
  population with the PPAF. Self-contained (simulated data, no figshare
  download, no JAX-heavy extras). Now part of the CI `ci_smoke`
  notebook group so a regression in the public API surfaces on every
  PR.
- **`intro.html`** &mdash; two new entries at the top of the "Where to
  next" section: a [Concepts &amp; Background](https://cajigaslab.github.io/nSTAT-python/concepts/index.html)
  card (the 14-page learning track) and a Hands-on tutorials card
  pointing at `examples/tutorials/`. Both surfaces existed since
  v0.4.4 but were invisible from the 5-minute tour.
- **"By concept" crosswalk** in `docs/Examples.md` &mdash; a table
  mapping every concepts page to its tutorial scripts, MATLAB-help-port
  notebooks, and paper examples. Bridges "I just read this concept" to
  "what code should I run next?"
- **Glossary anchors + page jump boxes.** 40 HTML
  `<a id="slug"></a>` anchors injected before each `**Term.**`
  definition in `docs/concepts/glossary.md` (slugs derived
  deterministically from term text). **Every concepts learning-track
  page** now opens with a "Glossary jumps" box of 5&ndash;11 inline
  links into the glossary, or a shorter "See also" pointer on the
  three navigational pages (`further_study.md`, `self_check.md`,
  `index.md`).
- **Reproducibility &amp; random-seeds Q&amp;A** in
  `docs/concepts/pitfalls_and_faq.md` (carried in from v0.4.5).

### CI / build hygiene

- **GitHub Actions Node 20 &rarr; Node 24.** All eight official actions
  bumped to their current major float tag: `actions/checkout@v6`,
  `setup-python@v6`, `cache@v5`, `upload-artifact@v7`,
  `download-artifact@v8`, `configure-pages@v6`,
  `upload-pages-artifact@v5`, `deploy-pages@v5`. Removes the
  Node-20-deprecation warning that surfaced on the v0.4.5
  `deploy-docs` run.
- **`AGENT_GUIDE.md` source layout** now lists `examples/tutorials/`,
  `examples/extras/`, and `docs/concepts/` &mdash; surfaces present
  since v0.4.4 but missing from the agent-orientation paragraph.

### Documentation hygiene

- **`RELEASE_READINESS.md`** refreshed: PyPI Trusted-Publisher setup
  marked operational (it's been running every release since v0.4.1);
  Tier 0.3 PP_EM follow-ups moved from "candidates" to "shipped"
  (landed in v0.4.2); v0.4.6 candidate list added; standing release
  checklist expanded with the per-iteration
  `docs/changes/*.html` pattern and the post-v0.4.5 manual
  `deploy-docs` trigger step.
- **`docs/extras_summary.html`** "What's planned" section: removed the
  "EM accuracy hardening" bullet (shipped in v0.4.x &mdash; exact
  lag-one cross-covariance + Laplace correction); section header
  retitled "Beyond v0.4.x"; added a "Already shipped through v0.4.x"
  recap.
- **`parity/methods_roadmap.md`** "Suggested sequencing": rewritten
  to acknowledge that Tiers 0.1/0.2/0.3, 1.1, and 2.1 are now shipped
  &mdash; the remaining sequence is Tier 3.1 (CLDS) &rarr; 3.2 (NPNR)
  &rarr; 3.3 (GP-based GLM filters). Top-of-file lede updated
  accordingly; broken `../CLAUDE.md` link generalised.

### Breaking changes

None.

## v0.4.5 — 2026-06-10

Documentation hygiene, theme + gallery polish, and CI billing conservation.
**No public API changes, no behavior changes** &mdash; every importable symbol
behaves identically to v0.4.4.

### Documentation

- **`docs/concepts/further_study.md`** &mdash; a focused page that lists
  topics nSTAT does not implement (population geometry &amp; dimensionality
  reduction, deep-learning encoders/decoders, spike sorting, vendor-format
  I/O) with primary references for each. Replaces a longer prior version;
  all concepts-page *content* (microelectrode recordings, point-process GLMs,
  time-rescaling KS, decoding, state-space EM, rhythmic firing, etc.) is
  unchanged.
- **`docs/concepts/from_filters_to_deep_learning.md`** now ends at the
  PPAF &rarr; modern-sequence-decoder bridge that the rest of the toolbox
  supports; a trailing section that pointed to research-frontier
  pretrained models is removed.
- **Visual gallery rebuild.** `docs/Examples.md` is now a card-style visual
  gallery mirroring the MATLAB Examples help index. Each paper example
  links to its figure README. (PR #143)
- **Theme polish.** The Sphinx site and the standalone HTML landing pages
  (`intro.html`, `extras_summary.html`, `whats_new.html`, every per-release
  `docs/changes/*.html`) now share a single clean light palette and
  consistent reference tables. (PRs #141, #142, #145)
- **Cross-references updated** in `docs/concepts/index.md`,
  `population_geometry.md`, `self_check.md` to match the new further-study
  page; related word-level cleanup in those files. The v0.4.3 entries in
  `RELEASE_NOTES.md` and `docs/changes/2026-05-31-v0.4.3-docs-cleanup.html`
  generalize a historical reference to an internal-only test filename.
- **Stray-artifact fix.** A `&lt;/content&gt;&lt;/invoke&gt;` tail artifact at the end of
  `rhythmic_firing_and_clinical_microelectrode.md` is removed.

### Added

- **README onramp callouts.** A third "Other ways in" callout points
  readers at the six tutorial scripts (`examples/tutorials/`), the five
  paper examples (`examples/paper/`), and the reference notebooks
  (`notebooks/`) — alongside the existing 5-minute intro and Concepts
  callouts.
- **Reproducibility &amp; random-seeds Q&amp;A** in
  `docs/concepts/pitfalls_and_faq.md` &mdash; explains the
  `np.random.default_rng(seed)` pattern, why simulation results vary
  without a seed, and why GLM fitting is deterministic but EM trainers
  benefit from multi-restart selection.
- **AGENT_GUIDE source layout** now lists `examples/tutorials/`,
  `examples/extras/`, and `docs/concepts/` &mdash; surfaces that have
  existed since v0.4.4 but were not in the agent-orientation paragraph.

### CI

- **Manual-only by default.** Five GitHub Actions workflows
  (`performance-parity`, `notebook-full-fidelity`, `helpfile-check`,
  `readme-check`, `nstat-pip-install`) are now `workflow_dispatch`-only
  on this repository, with a new `make ci-local` target for the same
  checks. The unit test, docs build, data integrity, notebook smoke, and
  cleanroom compliance gates still run automatically on every PR. (PR #144)

### Breaking changes

None.

## v0.4.4 — 2026-06-09

Documentation and packaging polish.  **No public API changes, no behavior
changes** &mdash; every importable symbol behaves identically to v0.4.3.

### Added

- **Concepts page** `rhythmic_firing_and_clinical_microelectrode.md` &mdash;
  an applied teaching arc covering rhythmic/oscillatory (tremor) cells as a
  point-process GLM with a periodic covariate, and the intraoperative
  deep-brain-stimulation microelectrode setting: firing-rate localization,
  the beta-band field-potential biomarker via `SignalObj.MTMspectrum`, and
  reading the rhythm back out with the point-process adaptive filter.  All
  synthetic data; six PubMed-verified clinical references.
- **Tutorial** `examples/tutorials/clinical_microelectrode_walkthrough.py`
  &mdash; a runnable capstone (encode → time-rescaling KS → beta spectrum →
  PPAF decode) on a simulated tremor cell.
- Five cross-page "Applying nSTAT" bridges and the supporting glossary,
  bibliography, self-check, and index entries.

### Packaging

- `pyproject.toml` now declares trove **classifiers** (Python 3.10 / 3.11 /
  3.12, plus audience / topic / license / OS / status).  Fixes the
  `python | missing` badge once this release is published to PyPI &mdash; the
  shields.io `pypi/pyversions` badge reads classifiers, which were absent
  from prior releases.

## v0.4.3 — 2026-05-31

Documentation and packaging cleanup.  Removes cross-references to
internal-only material from public-facing surfaces (README, helpfiles,
module docstrings published via Sphinx autodoc, parity notes) and
genericizes a couple of forward-looking placeholders.  **No public API
changes, no behavior changes** &mdash; every importable symbol behaves
identically to v0.4.2.

### Removed

- An internal cross-validation test that depended on tooling outside this
  distribution. Its parity targets are still exercised by the MATLAB gold
  fixtures and the existing dynamax / population-time-rescale tests.

### Edited

- `AGENT_GUIDE.md`, `docs/ClassDefinitions.md`,
  `nstat/decoding_algorithms.py`, `nstat/linear_cif.py`,
  `nstat/simulation.py`, `parity/integration_opportunities.md`,
  `parity/matlab_audit_xref.md`, `docs/extras_summary.html`,
  `pyproject.toml` &mdash; internal cross-references removed; public
  math citations (Smith &amp; Brown 2003, Truccolo 2005, Lewis &amp;
  Shedler 1979, Ogata 1981, the 2012 nSTAT paper) and the existing
  Sphinx cross-refs are preserved.

### Breaking changes

None.

## v0.4.2 — 2026-05-31

Tier 0.3 follow-ups + a docs fix.  Two new **opt-in** keyword
arguments on the PP_EM / mPPCO_EM trainers and their `_best_of`
variants — defaults unchanged, so v0.4.1 fits remain bit-exactly
reproducible.

### EM hardening — opt-in (`nstat.extras.em.dynamax_bridge`)

- **`init="log_empirical_rate"`** on `fit_point_process_em` /
  `fit_point_process_em_best_of` — seeds `x0` from
  `pinv(C) @ log(empirical_mean_rate)` so the implied initial firing
  rate matches the data instead of `exp(0) = 1`.  Removes one bad-init
  basin where EM starts with systematically wrong rates.
- **`ridge_lambda=λ`** on `fit_point_process_em` and `fit_hybrid_em`
  (and the `_best_of` variants) — biases the A M-step toward the
  identity via a Gaussian prior at `A = I`:
  `A = (S10 + λI)(S11 + λI)⁻¹`.  When `S10`, `S11 → 0` (the
  weak-observability collapse mode) the limit becomes `I` rather than
  `0`.  Try `0.1`–`1.0` if you see the `A → 0` collapse in
  `*_best_of` traces.

```python
result = fit_point_process_em_best_of(
    spike_counts, state_dim=3, n_restarts=8,
    init="log_empirical_rate",   # data-driven x0
    ridge_lambda=0.5,            # bias A toward identity
)
```

Closes the Tier 0.3 follow-up items deferred from v0.4.0 (PR #119) and
documented in `parity/methods_roadmap.md`.

### Docs fix

- **`RELEASE_READINESS.md`** PyPI setup instructions now distinguish
  the **project-level URL** (used when the project already exists on
  PyPI) from the **account-level Pending Publisher URL**
  (`https://pypi.org/manage/account/publishing/`, required for a
  project's first publish).  The v0.4.1 PyPI publish ran into this:
  the project-level URL 404s when the project doesn't exist yet, and
  the OIDC token gets rejected with `invalid-publisher`.

### Breaking changes

None.  All new parameters default to v0.4.1 behavior.

## v0.4.1 — 2026-05-31

A post-v0.4.0 polish release.  No public API changes; closes the
verification gaps surfaced by the post-release audit and ships the
infrastructure for PyPI publication.

**This is the first release configured to publish to PyPI.**  Once the
maintainer completes the one-time PyPI Trusted Publisher setup
documented in `RELEASE_READINESS.md`, the next pushed `v*` tag
auto-uploads the sdist + wheel via OIDC (no API tokens).
Pre-v0.4.1 versions were never on PyPI despite the README's
`pip install nstat-toolbox` claim; v0.4.1 makes that claim true.

### Infrastructure

- **PyPI publishing** — new `.github/workflows/publish.yml` builds and
  uploads sdist + wheel via PyPI Trusted Publisher (OIDC) on each
  `v*` tag push.  The workflow asserts the tag matches
  `pyproject.toml` at build time so a missed version bump cannot ship.
- **`extras-clusterless` CI job** — parallel to `extras-dynamax`.
  Installs `nstat-toolbox[clusterless]` (pulls JAX + `replay_trajectory_classification`)
  and runs `tests/extras/test_clusterless_bridge.py` end-to-end.
  Closes the Tier 2.1 CI story: the clusterless smoke tests no longer
  skip silently in every CI run.
- **`statsmodels>=0.15`** pinned in `[test-parity]` — the older
  `statsmodels` versions imported the private `scipy._lib._util._lazywhere`
  symbol that scipy removed, surfacing as a recurring local-only test
  failure under common conda environments.  Pinning forward resolves
  it without touching the toolbox.

### Docs / hygiene

- **`docs/extras_summary.html`** — replaced the stale "Generated as
  part of the nSTAT-python v0.3.2 release" footer line with a
  version-agnostic note (the extras namespace was introduced in v0.3.2
  and has been substantially expanded since).
- **`RELEASE_READINESS.md`** — reset from the v0.4.0 plan to the v0.5
  planning + PyPI-setup instructions.  The v0.4.0 content lives
  permanently in `RELEASE_NOTES.md` now.

### Breaking changes

None.

## v0.4.0 — 2026-05-31

A substantive minor release: the public API gains the **EM
state-space trainer family** (`fit_linear_gaussian_em`,
`fit_point_process_em`, `fit_hybrid_em` + diagnostics) in
`nstat.extras.em.dynamax_bridge`, the **multivariate (Tao 2018)
population goodness-of-fit** (`population_time_rescale`) in core, and a
flagship **clusterless decoding** bridge in
`nstat.extras.decoding.clusterless_bridge`.  No breaking changes; every
existing public symbol is preserved.  Highlights below.

### `nstat.*` (core) additions

- **`population_time_rescale(counts_list, lam_per_bin_list, *, n_tau_bins=1)`**
  → **`PopulationTimeRescaleResult`** — the Tao, Weber, Arai & Eden
  (2018) marked point-process time-rescaling goodness-of-fit.  Scores a
  *population* jointly: a ground-process KS plus a marked χ².  Catches
  inter-neuron coupling misfit that the per-neuron univariate KS
  (`FitResult.computeKSStats`) misses — e.g., a synchronous-pair model
  whose per-neuron univariate KS both pass is rejected by the population
  ground-process KS at p≈3e-49.  Pure NumPy/SciPy, additive — the
  existing univariate `computeKSStats` and its MATLAB-gold fixtures are
  untouched.  Python-only extension (the 2018 method postdates the 2012
  MATLAB toolbox).

### `nstat.extras.em.dynamax_bridge` (state-space EM — first user-facing release)

The EM trainer family lands publicly for the first time in this
release.  The trainers shipped via `[dynamax]` (JAX, ~200 MB) and are
deliberately excluded from `[all-extras]`:

- **EM trainers** (`KF_EM` / `PP_EM` / `mPPCO_EM` equivalents):
  `fit_linear_gaussian_em`, `fit_point_process_em`, `fit_hybrid_em` +
  result dataclasses.
- **Inference** (PPDecodeFilter / PP_fixedIntervalSmoother equivalents):
  `cmgf_poisson_filter`, `cmgf_poisson_smoother`.
- **Tier 0.1 — full PLDS identifiability gauge** (`_canonicalize_gauge`):
  pins the `GL(d)` gauge freedom to a canonical form (whiten + SVD-rotate
  + sign-fix) once after EM convergence.  Across-seed `|ΔC|` drops from
  ~460 (with NaN) → ~0.75 (PP) / ~0.15 (hybrid); the returned `C`
  satisfies `CᵀC = diag(S²)` to machine precision.
- **Tier 0.2 — true held-out predictive log-likelihood**:
  `point_process_predictive_ll` / `hybrid_predictive_ll` + `PredictiveLogLik`.
  The honest convergence / model-comparison metric (Gauss-Hermite
  quadrature over the latent Gaussian predictive + exact MVN for the
  Gaussian channel), replacing the surrogate Gaussian-smoother trace.
  Pure NumPy (no dynamax/JAX), runs in the base test suite.
- **Tier 0.3 — multi-restart selection** (the recommended workflow on
  real data): `fit_point_process_em_best_of` / `fit_hybrid_em_best_of`
  → `MultiRestartResult`.  Compose Tier 0.1 + 0.2: split by time, fit
  N seeds on train, score each on the held-out tail, return the best.
  Automatically discards the weak-observability `A → 0` collapses that
  Tier 0.2 surfaced.

### `nstat.extras.decoding.clusterless_bridge` (new — flagship)

- **`fit_clusterless_decoder` / `fit_clusterless_classifier`** +
  `ClusterlessDecoderResult` / `ClusterlessClassifierResult`.  Thin
  bridge to MIT-licensed `replay_trajectory_classification`
  (Denovellis et al. 2021, eLife) for marked point-process decoding (no
  spike sorting required) and trajectory-type classification (replay
  vs. local).  The modern descendant of nSTAT's PPAF / PPHF filters.
  New `[clusterless]` opt-dep group (JAX-heavy; not in `[all-extras]`).

### Hygiene / infrastructure

- **Notebook-fidelity date-volatility fixed**: dropped the wall-clock
  `generated_on` field from `parity/notebook_fidelity.yml`'s generator.
  Eliminates the recurring local-only `make test` failure on dev
  machines with the MATLAB checkout.
- **Version-drift guards strengthened**: `tests/test_version_sync.py`
  now also fails loudly on drift between `pyproject.toml` and
  `docs/conf.py` `release` / `AGENT_GUIDE.md` `Package version` — both
  drifted silently to 0.3.1 before v0.4.0 prep caught them.
- **`docs/changes/`** — a published `What's New` tree of per-iteration
  HTML change summaries at `https://cajigaslab.github.io/nSTAT-python/whats_new.html`.

### Documentation

- `docs/extras/em_dynamax.md` — full caveats for PP_EM/mPPCO_EM,
  observability discussion, and the recommended `*_best_of` workflow.
- `docs/extras/decoding_clusterless.md` — new help file for the
  clusterless bridge.
- `parity/methods_roadmap.md` — Tier 0.1 / 0.2 / 0.3 / 1.1 / 2.1 all
  marked SHIPPED.  Remaining: Tier 3 (CLDS / NPNR / GP-GLM) +
  data-driven-init / A-Q-ridge follow-ups for PP_EM.
- `RELEASE_READINESS.md` — pre-release audit + tag checklist (this
  file's working companion; can be deleted post-tag or reset to next
  release's planning).

### Breaking changes

None.  Every pre-v0.4.0 public symbol is preserved.

## v0.3.2 — 2026-05-26

Inaugural release of the **`nstat.extras` namespace** — a monorepo addon
space (modeled after `scikit-learn-contrib`) for Python-only features
that have no MATLAB nSTAT counterpart.  The core `nstat.*` namespace
remains under the strict MATLAB-parity contract; `nstat.extras.*` is
free to evolve at minor-release speed and depends on opt-in third-party
libraries declared in `pyproject.toml`'s `[project.optional-dependencies]`.

### New: `nstat.extras` subpackages

- **`nstat.extras.interop`** — `Trial` / `SpikeTrainCollection` / `nspikeTrain`
  converters for the wider Python neuro-data stack (#93).
  - `interop.neo` — `to_neo_spiketrain`, `from_neo_spiketrain`, `to_neo_segment`
    (install `[neo]`).
  - `interop.pynapple` — `to_pynapple_ts`, `to_pynapple_with_support`,
    `from_pynapple_ts`, `to_pynapple_tsgroup` (install `[pynapple]`).
  - `interop.nwb` — `nwb_units_to_collection`, `read_nwb_path` with
    `obs_intervals` / `time_window=` support and explicit fallback
    warning when neither is available (install `[nwb]`).
- **`nstat.extras.validation`** — Python-side cross-validation bridges
  that triangulate nstat's MATLAB-faithful estimates against
  independent reference implementations.  Three GLM oracles + one
  Kalman oracle now form a complete cross-validation triangle:
  - `validation.nemos_bridge.cross_validate_poisson_glm` against
    Flatiron's NeMoS Poisson GLM — ~5e-3 agreement (#93).
  - `validation.pykalman_bridge.cross_validate_kalman` against
    pykalman.  After the AUDIT D3 correction (see below), filter
    agrees ~2.6e-3, smoother ~1.6e-4 (#93, #102).
  - `validation.statsmodels_bridge.cross_validate_poisson_glm` against
    statsmodels — both use IRLS, agreement ~1e-9 (machine precision),
    making this the **tightest** cross-validation oracle in the
    extras namespace (#100).
- **`nstat.extras.metrics`** — modern spike-train distance metrics
  (#93).
  - `metrics.spike_distances.{isi,spike,spike_synchronization,pairwise_spike_distance_matrix}`
    via PySpike (install `[metrics]`).
- **`nstat.extras.em`** — EM-trained state-space models via Dynamax
  (#104).
  - `em.dynamax_bridge.fit_linear_gaussian_em` — KF_EM equivalent.
    Foundation for closing the AUDIT_REPORT §3.2 gap (19 unported
    MATLAB EM methods, ~7,500 LOC) without re-implementing.  PP_EM
    and mPPCO_EM follow in subsequent releases.

### New: shared infrastructure

- `nstat/extras/_lazy.py` — `require_optional` helper centralizes the
  "lazy-import or actionable ImportError" pattern across all six
  bridges, with a uniform error-message format embedding the exact
  `pip install nstat-toolbox[…]` line (#96).

### Fixed: AUDIT D3 misdiagnosis

The originally-cataloged D3 entry in `AUDIT_REPORT.md` claimed
`DecodingAlgorithms.kalman_fixedIntervalSmoother` was a Python-side
approximation of MATLAB's exact augmented-state smoother.  Investigation
(driven by the new pykalman bridge) showed this was wrong: the Python
implementation IS a faithful port of MATLAB's algorithm.  The
previously-reported ~0.4-unit smoother disagreement was a
fixed-lag-vs-RTS comparison — apples to oranges.  The pykalman bridge
now correctly calls `kalman_smoother` (proper RTS) instead, and
disagreement collapses to ~1.6e-4 (#102).

### New: optional-dependency groups in `pyproject.toml`

- `[neo]` (neo + quantities)
- `[pynapple]` (pynapple ≥ 0.7)
- `[nwb]` (pynwb ≥ 2.8)
- `[metrics]` (pyspike ≥ 0.8)
- `[nemos]` (nemos ≥ 0.2)
- `[test-parity]` (nemos + pykalman + statsmodels + nitime)
- `[dynamax]` (dynamax + JAX) — heavyweight; deliberately **not** in
  `[all-extras]` (the explicit `HEAVY_OPT_OUT_OF_ALL_EXTRAS` pattern).
- `[all-extras]` — install every functional extras module except
  `[dynamax]`.

### New: CI gates

- `tests/test_pyproject_consistency.py` (#95) — four structural
  contracts on `[project.optional-dependencies]`: every non-placeholder
  group has real deps, `[all-extras]` is a true union, README install
  lines reference real groups, every extras subpackage has a backing
  dep group.  Caught a real `[dynamax]`/`[all-extras]` drift on first
  run.
- `extras-functional` CI job in `.github/workflows/ci.yml` (#101)
  installs `[dev,all-extras]` and runs every bridge against its real
  backing library — closes the false-safety gap where
  `pytest.importorskip` made `[dev]`-only CI silently skip the
  functional tests.  Caught a real `model.intercept_` regression in
  the NeMoS bridge on first run.

### New: oracle for time-rescaling KS test

`tests/parity/_third_party/time_rescale_oracle.py` (#103) — clean-room
reference implementation of the Brown / Barbieri / Ventura / Kass /
Frank time-rescaling theorem KS test (Neural Computation 2002,
14:325–346).  Will be wired into a full `FitResult.computeKSStats`
comparison harness in a subsequent release.

### Docs

- `README.md` adds a "Related Python projects" section (#93, #100,
  #105) with the install-command matrix and pointers to ecosystem
  peers (SpikeInterface, Elephant, Dynamax, ssqueezepy).
- `AGENT_GUIDE.md §5.6` updated to name a concrete library / extras
  module for every "what nstat is NOT" scope gap (#93).
- `parity/integration_opportunities.md` provides the upstream-metadata
  audit driving these decisions.
- `docs/extras.rst` — per-tier autosummary index (#94).
- `docs/extras/` — six per-bridge narrative usage guides (#98).
- `docs/extras_summary.html` — self-contained landing page at
  https://cajigaslab.github.io/nSTAT-python/extras_summary.html
  (#99, #105).
- `examples/extras/` — six runnable demo scripts, one per bridge,
  exercised end-to-end in CI (#97).

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
