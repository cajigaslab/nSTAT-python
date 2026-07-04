# Extras Neuroscience-Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-ground the 17 `examples/extras/*_demo.py` demos in specific, literature-backed BCI/clinical neuroscience questions by redesigning each demo's synthetic generative model, docstring, and ground-truth-vs-estimate figure — without changing any public API, dependency, or gallery contract.

**Architecture:** Each demo keeps its method, `run_demo()` signature, `demo_id`, and gallery integration. Per demo we rewrite (a) the generative model to model a named clinical phenomenon, (b) the module docstring header (question + real citations), (c) the figure(s) to show ground-truth-vs-estimate, and (d) the `manifest.yml` + `extras_descriptions.yml` prose. A final task regenerates the gallery once and runs the full gate suite. The authoritative per-demo generative-model spec is the committed design doc `proposals/2026-07-04-extras-neuroscience-grounding-design.md` (referenced per task by section).

**Tech Stack:** Python 3.10+, NumPy/SciPy (core deps only), Matplotlib (Agg for headless), the existing `nstat.extras` APIs, the repo's gallery tooling (`tools/extras_build/build_extras_gallery.py`), and pytest.

## Global Constraints

Every task's requirements implicitly include this section.

- **Synthetic-only.** No data downloads, no new optional dependencies, no new files. Modify existing demo files in place.
- **API frozen.** Do not change any `run_demo(...)` signature, public function, `demo_id`, `name`, or `script` path. Only the generative model, docstrings, figures, and prose change.
- **Citations must be real.** Use only citations from the design doc §9 (verified 2026-07-04). Never invent a citation. The two "novelty" demos (`spatial_hawkes_ecog` Cox-Hawkes, `spatial_stlgcp_microelectrode` LGCP-CSD) state "a method not yet applied to this clinical problem" — no fabricated application cite.
- **Determinism.** All randomness via `np.random.default_rng(seed)` with the demo's existing seed plumbing. No legacy `np.random.rand`.
- **Figures.** Keep the exact figure filenames each demo already emits (do not renumber). Keep the `# === FIGURE: figNN_*.png ===` / `# === END FIGURE ===` markers (the gallery code-extractor depends on them). Use `MPLBACKEND=Agg`.
- **Per-demo procedure (the repeatable cycle every demo task follows):**
  1. Read the current demo file and its design-doc section to see the current generative model and the target one.
  2. Rewrite the generative model to the design-doc scenario (named ground-truth parameters).
  3. Update the module docstring header: the clinical question + 2–4 real citations + a one-line cross-link to sibling demos in the same thread.
  4. Rebuild the figure(s) to show ground-truth-vs-estimate (true field/kernel/phenotype overlaid or side-by-side with the recovered one), keeping filenames + markers.
  5. Update the demo's `manifest.yml` entry (`title`, `question`, `description`) and its `extras_descriptions.yml` entry (`overview`, `goal`, per-figure `caption`/`analysis`) to the new clinical framing.
  6. Verify (see below).
  7. Commit.
- **Per-demo verification (Step "verify"):**
  - Run headless: `cd /Users/iahncajigas/projects/nstat-python && MPLBACKEND=Agg python examples/extras/<demo>.py --export-figures --no-display` → exits 0, prints its recovery summary.
  - The demo's own recovery assertion holds (the printed "true vs fitted/recovered" numbers agree within the demo's stated tolerance).
  - Each emitted figure clears the content audit: `python tools/parity/image_content_audit.py docs/figures/extras/<demo_id>/<fig>.png` → not degenerate.
  - Doc-contract stays green: `python -m pytest tests/test_extras_docs.py tests/test_extras_examples.py -q`.
- **Commit convention:** branch `feat/extras-neuroscience-grounding` (already created). One commit per demo:
  `git commit -m "feat(extras-examples): reground <demo_id> in <clinical phenomenon>"` ending with the `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer.
- **Do NOT** hand-edit `docs/extras_gallery.html` / `docs/galleries.html` — those are regenerated once in the final task.

---

## Phase 1 — Thread 1: Human epilepsy / iEEG (Tasks 1–5)

### Task 1: `spatial_gof_ecog` — traveling wave vs. inhomogeneous-Poisson null

**Files:**
- Modify: `examples/extras/spatial_gof_ecog_demo.py`
- Modify: `examples/extras/manifest.yml` (entry `spatial_gof_ecog`)
- Modify: `tools/extras_build/extras_descriptions.yml` (entry `spatial_gof_ecog`)
- Spec: design doc §3.1

**Interfaces:**
- Consumes: `nstat.extras.spatial.{k_st_inhom, pair_correlation_st, global_envelope_st, intensity_st_kde, simulate_cox_hawkes}` (already imported).
- Produces: nothing consumed by later tasks (each demo is independent).

**Deltas from current demo** (the current demo already contrasts a baseline Poisson epoch vs. a clustered Cox-Hawkes epoch — reframe the *clustered* epoch as a **migrating Gaussian wavefront** and the *baseline* as a **static hot-spot Poisson null**, matching the clinical "is propagation a real traveling wave?" question):
- Baseline generator: inhomogeneous Poisson with a spatially-varying but **temporally static** rate (the existing `_background_rate` hot-spot is fine — keep it as the "static hot-spot null").
- Wave generator: replace the Hawkes cascade with a Gaussian intensity bump whose center translates across the 8×8 grid at fixed velocity over the test window; draw events by thinning.
- Keep the held-out `intensity_st_kde` calibration + `global_envelope_st` (statistic="kst"). Assertion unchanged in spirit: baseline `inside=True`, wave `inside=False`.

- [ ] **Step 1: Add a migrating-wavefront generator + reframe epochs** — in the demo, add a `_traveling_wave_events(rng)` helper (Gaussian bump center moving linearly across the grid over `[0, T_TEST]`, thinning against its own max), and use it for the "wave" epoch; keep `_background_rate` for the "baseline (static hot-spot)" epoch. Update variable/label names baseline→"static hot-spot (Poisson null)", clustered→"traveling wave".

- [ ] **Step 2: Update docstring header** — clinical question (design §3.1), citations Smith 2016 / Diamond 2021 / Martinet 2017 / Smith 2022, cross-link to `spatial_stlgcp_microelectrode`.

- [ ] **Step 3: Rebuild `fig01`** — keep the 2×2 layout from the prior pass: top row = the two ground-truth patterns (static hot-spot vs. traveling wave, over the true rate field, events colored by time so the wave's temporal sweep is visible); bottom row = the K-envelope diagnostic per epoch. Update `suptitle`.

- [ ] **Step 4: Update prose** — `manifest.yml` `title`/`question`/`description` and `extras_descriptions.yml` `overview`/`goal`/`fig01` caption+analysis to the traveling-wave framing.

- [ ] **Step 5: Verify** — run headless; assert baseline `inside=True` and wave `inside=False` in the printed verdict; content-audit the 3 figures; run the doc-contract pytest.

Run: `MPLBACKEND=Agg python examples/extras/spatial_gof_ecog_demo.py --export-figures --no-display`
Expected: prints "static hot-spot ... INSIDE" and "traveling wave ... OUTSIDE"; 3 figures saved.

- [ ] **Step 6: Commit** — `git add` the demo + manifest + descriptions + `docs/figures/extras/spatial_gof_ecog/`; commit `feat(extras-examples): reground spatial_gof_ecog in ictal traveling-wave detection`.

### Task 2: `spatial_hawkes_ecog` — seizure core vs. penumbra (background/excitation split)

**Files:** Modify `examples/extras/spatial_hawkes_ecog_demo.py`, `manifest.yml`, `extras_descriptions.yml`. Spec §3.2.

**Deltas** (the demo already fits branching-EM + Cox-Hawkes and shows true-vs-recovered kernels + true-vs-fitted background — reframe the ground-truth as a **hyperexcitable seizure-onset zone (LGCP background) + a self-exciting recruitment cascade (core)**, so the background/triggered split *is* the core/penumbra decomposition):
- Rename the two catalogues to "penumbra (background-dominated)" and "core (self-excited cascade)"; keep the existing `em_spatial_hawkes` (catalogue A) and `fit_cox_hawkes` (catalogue B) machinery and the true σ_space / c / K ground-truth parameters.
- No numeric-model change is required beyond labeling + choosing background/triggering parameters that make the core/penumbra contrast visible; keep the recovery assertions (recovered σ_space, c, K within tolerance; `background_fraction` sensible).

- [ ] **Step 1: Reframe catalogues + parameters** — relabel to core/penumbra; if needed nudge `K_branch`/σ so the cascade visibly clusters (core) vs. sparse background (penumbra).
- [ ] **Step 2: Docstring header** — question §3.2; citations Schevon 2012 / Truccolo 2014 / Martinet 2017; **state the Cox-Hawkes novelty** ("not yet applied to epilepsy core/penumbra"); cross-link `spatial_gof_ecog`, `spatial_gibbs`.
- [ ] **Step 3: Figures** — keep fig02 (true vs. recovered triggering kernels) and fig03 (declustering P(background) + true vs. fitted LGCP background); relabel to core/penumbra.
- [ ] **Step 4: Prose** — manifest + descriptions updated to the core/penumbra framing.
- [ ] **Step 5: Verify** — run headless; recovered σ_space/c/K within tolerance; 3 figures pass audit; doc-contract pytest green.
- [ ] **Step 6: Commit** — `feat(extras-examples): reground spatial_hawkes_ecog in seizure core/penumbra decomposition`.

### Task 3: `spatial_cluster_cox` — microdischarges around hidden epileptogenic hubs

**Files:** Modify `examples/extras/spatial_cluster_cox_demo.py`, `manifest.yml`, `extras_descriptions.yml`. Spec §3.3.

**Deltas** (demo already simulates Thomas/Matérn + minimum-contrast — reframe the Thomas process as **K latent epileptogenic hub "parents" scattering microdischarge "offspring"**, sub-mm dispersion per Stead 2010):
- Keep the Thomas simulator + minimum-contrast estimator; set parent intensity κ / offspring μ / dispersion σ to a "few hidden hubs, tight clusters" regime; recovery assertion: recovered (σ, κ) within sampling error.

- [ ] **Step 1: Retune to the hub regime** — set κ (few parents), μ (several offspring each), σ (tight) to match the microseizure story; keep both Thomas + Matérn if present, framing Matérn as an alternative offspring geometry.
- [ ] **Step 2: Docstring** — question §3.3; Stead 2010 / Tobochnik 2021; cross-link `spatial_gibbs`.
- [ ] **Step 3: Figures** — true parent locations + offspring scatter; empirical vs. fitted K with CSR reference; recovered-vs-true cluster count/radius (add the true-parent overlay if not already present).
- [ ] **Step 4: Prose** — manifest + descriptions to the hidden-hub framing.
- [ ] **Step 5: Verify** — recovered (σ, κ) within tolerance; figures pass audit; doc-contract green.
- [ ] **Step 6: Commit** — `feat(extras-examples): reground spatial_cluster_cox in hidden epileptogenic hub recovery`.

### Task 4: `spatial_gibbs` — exclusion-zone spacing of co-active sites

**Files:** Modify `examples/extras/spatial_gibbs_demo.py`, `manifest.yml`, `extras_descriptions.yml`. Spec §3.4.

**Deltas** (demo already fits Strauss/hard-core/area-interaction via pseudo-likelihood — reframe hard-core as a **minimum-distance exclusion constraint on co-active/implanted sites**, contrasted with CSR):
- Foreground the CSR-vs-hard-core contrast; set the hard-core distance to a physiologically-motivated value (electrode spacing / SEEG safety margin); assertion: recovered interaction radius ≈ true for hard-core, γ≈1 for CSR.

- [ ] **Step 1: Foreground CSR-vs-hardcore** — ensure the demo simulates both CSR and a hard-core pattern and recovers the exclusion distance; keep Strauss/area-interaction as secondary panels.
- [ ] **Step 2: Docstring** — question §3.4; Meszéna 2026 / Rockhill 2000 / SEEG planning / Schevon 2012 (exclusion zone); cross-link `spatial_cluster_cox` (clustering vs. repulsion).
- [ ] **Step 3: Figures** — CSR vs. hard-core patterns side by side; empirical vs. fitted PCF (dip below hard-core distance only in the repulsive case); recovered-vs-true minimum distance.
- [ ] **Step 4: Prose** — manifest + descriptions to the exclusion-zone framing.
- [ ] **Step 5: Verify** — recovered hard-core distance within tolerance, γ≈1 for CSR; figures pass audit; doc-contract green.
- [ ] **Step 6: Commit** — `feat(extras-examples): reground spatial_gibbs in co-active-site exclusion-zone spacing`.

### Task 5: `spatial_stlgcp_microelectrode` — tracking a moving wavefront (CSD/seizure front)

**Files:** Modify `examples/extras/spatial_stlgcp_microelectrode_demo.py`, `manifest.yml`, `extras_descriptions.yml`. Spec §3.5.

**Deltas** (demo already tracks a moving Gaussian bump with KDE + ST-LGCP and shows true-field-vs-estimate — reframe the moving bump as a **spreading-depolarisation / seizure wavefront**, recover its trajectory + propagation speed):
- Keep the moving-bump generator + `intensity_st_kde` + `lgcp_st_fit`; set the bump speed/geometry to a realistic wavefront; add a recovered-vs-true **propagation-speed** readout (fit a line to the recovered peak trajectory).
- Assertion: recovered peak trajectory tracks the true one (centroid error within tolerance — already computed); recovered speed ≈ true speed.

- [ ] **Step 1: Add propagation-speed recovery** — compute the true wavefront speed from the bump trajectory and the estimated speed from the recovered peak locations; print both.
- [ ] **Step 2: Docstring** — question §3.5; Muller 2018 / Lauritzen 2011 / Hartings 2011 / Diamond 2021; **state the LGCP-CSD novelty**; cross-link `spatial_gof_ecog`.
- [ ] **Step 3: Figures** — keep fig02 (true field vs. KDE) + fig03 (LGCP posterior + band); add/annotate the true-vs-recovered peak-trajectory + speed on fig03 or fig01.
- [ ] **Step 4: Prose** — manifest + descriptions to the wavefront-tracking framing.
- [ ] **Step 5: Verify** — recovered speed within tolerance of true; figures pass audit; doc-contract green.
- [ ] **Step 6: Commit** — `feat(extras-examples): reground spatial_stlgcp_microelectrode in spreading-depolarisation wavefront tracking`.

---

## Phase 2 — Thread 2: Intracortical motor BCI (Tasks 6–10)

### Task 6: `decoding_clusterless` — sorted-vs-clusterless as sorting degrades

**Files:** Modify `examples/extras/decoding_clusterless_demo.py`, `manifest.yml`, `extras_descriptions.yml`. Spec §4.1.

**Deltas:** frame as motor/hippocampal decoding from **unsorted marked events**; add a sorted-decoder comparator whose sorting is corrupted at a tunable merge/split error rate; show clusterless degrades gracefully while sorted collapses.
- [ ] **Step 1: Add a corrupted-sorting comparator** — decode with the clusterless bridge and with a "sorted" decoder at increasing simulated sort-error rates; record decode error vs. error rate.
- [ ] **Step 2: Docstring** — question §4.1; Eden 2004 / Deng 2015 / Kloosterman 2014 / Denovellis 2021; cross-link `decoding_place_field`, `validation_pykalman`.
- [ ] **Step 3: Figure** — true trajectory + clusterless posterior band vs. sorted-decoder posterior at increasing sort-error rates (decode error vs. error-rate curve).
- [ ] **Step 4: Prose** — manifest + descriptions to the clusterless-vs-sorted framing.
- [ ] **Step 5: Verify** — clusterless error < sorted error at high sort-error rate; figures pass audit; doc-contract green.
- [ ] **Step 6: Commit** — `feat(extras-examples): reground decoding_clusterless in sort-error-robust motor decoding`.

### Task 7: `decoding_place_field` — population-vector bias vs. ML decode

**Files:** Modify `examples/extras/decoding_place_field_demo.py`, `manifest.yml`, `extras_descriptions.yml`. Spec §4.2.

**Deltas:** frame as M1 cosine-tuned population with **non-uniform preferred directions**; decode with population vector (biased) vs. ML (unbiased) on a center-out task.
- [ ] **Step 1: Add cosine-tuned M1 population + non-uniform PDs** — simulate ~50–100 units, preferred directions clustered; 8-direction center-out; decode by population vector and by ML.
- [ ] **Step 2: Docstring** — question §4.2; Georgopoulos 1982/1986 / Kettner 1988 / Sanger 1996; cross-link `decoding_clusterless`, `latents_gpfa`.
- [ ] **Step 3: Figure** — per-neuron tuning curves; polar plot true vs. PV vs. ML direction (PV biased toward the dense PD region, ML unbiased).
- [ ] **Step 4: Prose** — manifest + descriptions to the population-vector-bias framing.
- [ ] **Step 5: Verify** — ML angular error < PV angular error under non-uniform PDs; figures pass audit; doc-contract green.
- [ ] **Step 6: Commit** — `feat(extras-examples): reground decoding_place_field in population-vector bias vs ML decode`.

### Task 8: `em_dynamax` — decoder calibration + ReFIT recalibration

**Files:** Modify `examples/extras/em_dynamax_demo.py`, `manifest.yml`, `extras_descriptions.yml`. Spec §4.3.

**Deltas:** frame the EM state-space fit as **BCI decoder calibration**; contrast a naive KF (open-loop-calibrated, overshoots) with a ReFIT pass (intention-relabeled, straightened) on a cursor task.
- [ ] **Step 1: Add naive-KF vs. ReFIT contrast** — simulate the Wu-2006 arm+population; decode closed-loop trials with naive-KF and a ReFIT recalibration; record path curvature/acquisition metric.
- [ ] **Step 2: Docstring** — question §4.3; Wu 2006 / Gilja 2012 / Eden 2004 / Kim 2011; cross-link `validation_pykalman`.
- [ ] **Step 3: Figure** — true vs. decoded target-acquisition trajectories, naive-KF (curved/overshoot) vs. ReFIT (straight/fast).
- [ ] **Step 4: Prose** — manifest + descriptions to the ReFIT-calibration framing.
- [ ] **Step 5: Verify** — ReFIT acquisition metric better than naive-KF; figures pass audit; doc-contract green.
- [ ] **Step 6: Commit** — `feat(extras-examples): reground em_dynamax in closed-loop BCI decoder recalibration`.

### Task 9: `latents_gpfa` — single-trial rotational manifold recovery

**Files:** Modify `examples/extras/latents_gpfa_demo.py`, `manifest.yml`, `extras_descriptions.yml`. Spec §4.4.

**Deltas:** frame the shared latent as a **rotational motor-cortical manifold**; recover single-trial rotational trajectories vs. ground truth.
- [ ] **Step 1: Use a rotational latent** — replace the generic smooth latent with a 2–3-D skew-symmetric rotational system (one trajectory per reach condition) projected to ~50–100 Poisson units.
- [ ] **Step 2: Docstring** — question §4.4; Yu 2009 / Churchland 2012 / Gallego 2018/2020; cross-link `decoding_place_field`.
- [ ] **Step 3: Figure** — ground-truth rotational phase portrait (color per condition) overlaid with GPFA-recovered single-trial trajectories; fidelity vs. population size.
- [ ] **Step 4: Prose** — manifest + descriptions to the rotational-manifold framing.
- [ ] **Step 5: Verify** — best |corr| of recovered vs. true latent above the demo's threshold; figures pass audit; doc-contract green.
- [ ] **Step 6: Commit** — `feat(extras-examples): reground latents_gpfa in rotational motor-cortex manifold recovery`.

### Task 10: `validation_pykalman` — workhorse velocity KF + chronic drift

**Files:** Modify `examples/extras/validation_pykalman_demo.py`, `manifest.yml`, `extras_descriptions.yml`. Spec §4.5.

**Deltas:** frame as the **clinical velocity Kalman decoder**; keep the nstat-vs-pykalman agreement check, and add a chronic-drift panel (per-day unit gain/baseline drift + turnover) showing accuracy decays without recalibration.
- [ ] **Step 1: Add a chronic-drift panel** — over simulated days, perturb unit gains/baselines (Perge) and retire/replace units (Downey); track decode accuracy with/without periodic recalibration. Keep the existing nstat↔pykalman filtered/smoothed agreement assertion.
- [ ] **Step 2: Docstring** — question §4.5; Wu 2006 / Malik 2011 / Perge 2013 / Downey 2018; cross-link `em_dynamax`.
- [ ] **Step 3: Figure** — nstat vs. pykalman decoded trajectories (agreement); accuracy vs. simulated day, with/without recalibration.
- [ ] **Step 4: Prose** — manifest + descriptions to the workhorse-KF/chronic-drift framing.
- [ ] **Step 5: Verify** — nstat↔pykalman agree within existing tolerance; drift panel shows recalibration helps; figures pass audit; doc-contract green.
- [ ] **Step 6: Commit** — `feat(extras-examples): reground validation_pykalman in clinical velocity-KF drift & cross-check`.

---

## Phase 3 — Thread 3: DBS / movement-disorders MER (Tasks 11–12)

### Task 11: `modulated_renewal_microelectrode` — STN firing phenotypes for DBS targeting

**Files:** Modify `examples/extras/modulated_renewal_microelectrode_demo.py`, `manifest.yml`, `extras_descriptions.yml`. Spec §5.1.

**Deltas:** simulate **three ground-truth phenotypes along one MER pass** — (1) tonic-regular (high-shape gamma), (2) tremor-locked bursting (4–6 Hz rate-modulated), (3) irregular/high-entropy (shape≈1) — fit each with the modulated-renewal CIF; recover rate + regularity + KS GoF.
- [ ] **Step 1: Simulate the three phenotypes** — generalize the current single-unit demo to three units with the phenotype parameters above; fit each with `fit_modulated_renewal`.
- [ ] **Step 2: Docstring** — question §5.1; Bergman 1994 / Levy 2000 / Sterio 2002 / Sedov 2025 / Barbieri-Brown lineage; cross-link `metrics_spike_distances`.
- [ ] **Step 3: Figures** — rasters + ISI histograms with true density overlaid (per phenotype); recovered rate/tremor-drive vs. truth; recovered regularity (CV/burst index) per phenotype; KS diagnostic.
- [ ] **Step 4: Prose** — manifest + descriptions to the STN-phenotype/DBS-targeting framing.
- [ ] **Step 5: Verify** — recovered shape/CV separates the three phenotypes; KS passes for the true-family fit; figures pass audit; doc-contract green.
- [ ] **Step 6: Commit** — `feat(extras-examples): reground modulated_renewal_microelectrode in STN firing-phenotype MER targeting`.

### Task 12: `metrics_spike_distances` — beta-burst synchrony (OFF vs. adaptive-DBS)

**Files:** Modify `examples/extras/metrics_spike_distances_demo.py`, `manifest.yml`, `extras_descriptions.yml`. Spec §5.2.

**Deltas:** simulate 6–10 units with a **time-varying shared beta-burst drive** toggling between synchronized (OFF, long bursts) and desynchronized (adaptive-DBS, truncated bursts); track with SPIKE-distance/-synchronization in sliding windows.
- [ ] **Step 1: Add a time-varying synchrony ground truth** — inject a shared 15–30 Hz burst drive with an OFF epoch (long bursts) and a treated epoch (short bursts); compute windowed SPIKE-distance/-synchronization + ISI-distance.
- [ ] **Step 2: Docstring** — question §5.2; Kreuz 2011/2012 / Satuvuori 2017 / Tinkhauser 2017 / Kühn 2009 / Levy 2000; cross-link `modulated_renewal_microelectrode`.
- [ ] **Step 3: Figures** — rasters with the true synchrony-drive envelope shaded; time-resolved metric trace over the envelope; burst-duration distributions per regime.
- [ ] **Step 4: Prose** — manifest + descriptions to the beta-burst-synchrony framing.
- [ ] **Step 5: Verify** — the metric trace tracks the injected envelope (higher synchrony in OFF epoch); figures pass audit; doc-contract green.
- [ ] **Step 6: Commit** — `feat(extras-examples): reground metrics_spike_distances in Parkinsonian beta-burst synchrony`.

---

## Phase 4 — Thread 4: Clinical data rigor (Tasks 13–17)

*These are prose + docstring reframings (infrastructure demos); the generative content changes little. Keep them light.*

### Task 13: `interop_neo` — vendor-agnostic multi-site iBCI ingestion

**Files:** Modify `examples/extras/interop_neo_demo.py`, `manifest.yml`, `extras_descriptions.yml`. Spec §6.1.
- [ ] **Step 1: Docstring + prose** — frame as one ingestion path for a multi-site iBCI trial (Blackrock Utah-array → Neo → nstat); citation Garcia 2014. No generative change; keep the round-trip demo.
- [ ] **Step 2: Verify** — run headless; doc-contract green.
- [ ] **Step 3: Commit** — `feat(extras-examples): reframe interop_neo as multi-site iBCI ingestion`.

### Task 14: `interop_nwb` — reproduce a step from a real DANDI iEEG dataset (prose)

**Files:** Modify `examples/extras/interop_nwb_demo.py`, `manifest.yml`, `extras_descriptions.yml`. Spec §6.2.
- [ ] **Step 1: Docstring + prose** — frame around the AJILE12 dandiset (naturalistic ECoG + pose during epilepsy monitoring) as the motivating real dataset; keep the synthetic in-memory NWB (no fetch); citations Teeters 2015 / Rübel 2022 / Peterson 2022.
- [ ] **Step 2: Verify** — run headless; doc-contract green.
- [ ] **Step 3: Commit** — `feat(extras-examples): reframe interop_nwb around DANDI human-iEEG (AJILE12) workflow`.

### Task 15: `interop_pynapple` — interictal/trial epoch bookkeeping

**Files:** Modify `examples/extras/interop_pynapple_demo.py`, `manifest.yml`, `extras_descriptions.yml`. Spec §6.3.
- [ ] **Step 1: Docstring + prose** — frame `IntervalSet` restriction as interictal-only / BCI-trial epoch bookkeeping; citation Viejo 2023. Keep the round-trip.
- [ ] **Step 2: Verify** — run headless; doc-contract green.
- [ ] **Step 3: Commit** — `feat(extras-examples): reframe interop_pynapple as interictal/trial epoch bookkeeping`.

### Task 16: `validation_nemos` — reach-tuned encoding-GLM cross-check

**Files:** Modify `examples/extras/validation_nemos_demo.py`, `manifest.yml`, `extras_descriptions.yml`. Spec §6.4.
- [ ] **Step 1: Docstring + prose** — frame the fixture as a reach-tuned motor encoding GLM; the nstat↔NeMoS agreement is the reviewer-grade rigor check; citations Seabold 2010 / NeMoS / Weber-Pillow 2017. Keep the fixture + agreement assertion.
- [ ] **Step 2: Verify** — run headless (skips gracefully if NeMoS absent); doc-contract green.
- [ ] **Step 3: Commit** — `feat(extras-examples): reframe validation_nemos as reach-tuned encoding-GLM cross-check`.

### Task 17: `validation_statsmodels` — IRLS Poisson-GLM machine-precision agreement

**Files:** Modify `examples/extras/validation_statsmodels_demo.py`, `manifest.yml`, `extras_descriptions.yml`. Spec §6.5.
- [ ] **Step 1: Docstring + prose** — frame as the IRLS-vs-IRLS Poisson-GLM regression guard for a clinical encoding model; citation Seabold 2010. Keep the fixture + agreement assertion.
- [ ] **Step 2: Verify** — run headless (skips gracefully if statsmodels absent); doc-contract green.
- [ ] **Step 3: Commit** — `feat(extras-examples): reframe validation_statsmodels as clinical encoding-GLM regression guard`.

---

## Phase 5 — Integration (Task 18)

### Task 18: Regenerate gallery, run full gates, open PR

**Files:** `docs/extras_gallery.html`, `docs/galleries.html` (regenerated), `RELEASE_NOTES.md` (optional entry).

- [ ] **Step 1: Regenerate the extras gallery** — `make regen-extras-gallery`; confirm all 17 `demo_id` anchors present and captions updated.
- [ ] **Step 2: Content-audit all regenerated figures** — loop `tools/parity/image_content_audit.py` over `docs/figures/extras/**/*.png`; none degenerate.
- [ ] **Step 3: Full gate suite** — `make test` (955+ passing), `make freshness-check`, `python -m pytest tests/test_extras_docs.py tests/test_extras_examples.py -q`.

Run: `make test`
Expected: all pass, no new failures.

- [ ] **Step 4: Commit + open PR** — commit the regenerated gallery; `git push -u origin feat/extras-neuroscience-grounding`; open a PR to `main` summarizing the four clinical threads + the verified citation basis; do not self-merge.

---

## Self-review

- **Spec coverage:** every design-doc section §3.1–§6.5 maps to a task (Tasks 1–17); §7 guardrails → Global Constraints; §8 verification → per-task verify + Task 18. ✔
- **Placeholders:** per-demo generative details are specified as deltas + a spec-section pointer (the spec is committed and complete); commands and assertions are concrete. Where full generative code isn't inlined, it is because the scenario is specified in the committed spec the implementer reads — not a TBD. ✔
- **Consistency:** every task uses the same file set (`<demo>.py` + `manifest.yml` + `extras_descriptions.yml`), the same figure-marker rule, and the same verify harness. Demo IDs unchanged throughout. ✔
