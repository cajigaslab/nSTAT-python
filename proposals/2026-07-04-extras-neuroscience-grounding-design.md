# Design spec: BCI/clinical neuroscience grounding for the `nstat.extras` demos

- **Date:** 2026-07-04
- **Status:** approved (design); pending implementation planning
- **Scope:** the 17 `examples/extras/*_demo.py` demos only. The 8 paper examples and the
  35 notebooks are out of scope (they mirror the MATLAB toolbox and are parity-locked).
- **Author:** design produced via brainstorming + a 4-cluster literature review (2026-07-04).

## 1. Motivation

The `nstat.extras` demos currently answer *methods* questions ("does estimator X recover
parameter Y on a synthetic pattern?"). Each is a sound *simulate-ground-truth → recover →
verify* exercise, but the synthetic scenarios are generic. This spec re-grounds each demo in
a specific, translationally-valuable **BCI / clinical-neuroscience** question, so a student
sees the method solving a real problem a neurosurgeon / BCI researcher would recognise, backed
by real literature.

**Decisions locked during brainstorming:**
1. **Depth = redesign scenarios** — change the synthetic data-generating process to model a
   literature-grounded phenomenon (not merely reword the motivation).
2. **Domain = BCI / clinical-leaning** — intracortical motor BCI, epilepsy/iEEG, DBS/MER,
   clinical data rigor.
3. **Scope = the 17 extras only** — paper examples stay parity-locked.

## 2. Approach

**Per-example scenario redesign, woven into four clinical narrative threads via docstring
cross-links.** Each demo keeps its method, its public API, its self-contained fully-synthetic
structure (no data downloads, no cross-file coupling) and its gallery integration. What
changes per demo: the generative model, the docstring header (clinical question + citations),
the ground-truth-vs-estimate figure, and cross-links to sibling demos in the same thread.

Rejected alternatives:
- *Shared-simulation "one dataset, many analyses"*: couples the demo files and fights the
  opt-in module layout.
- *Prose-only reframing*: explicitly ruled out by the depth decision.

**Design template — every redesigned demo delivers:**
1. Docstring header: the clinical question + 2–4 real citations (author, year, venue, finding).
2. A generative model with named, physiologically-motivated ground-truth parameters.
3. A ground-truth-vs-estimate figure (true field/kernel/phenotype vs. recovered), in the style
   added in the 2026-07-03 STPP-figures pass.
4. Docstring cross-links to sibling demos in the same thread.
5. Unchanged public API, dependencies, and gallery/manifest integration.

**Two honest novelty framings** (no prior neuro-application paper exists; presented as
"a method the field has not yet applied here", never as a fabricated citation):
- Cox-Hawkes applied to seizure core/penumbra separation.
- LGCP applied to spreading-depolarisation / wavefront tracking.

## 3. Thread 1 — Human epilepsy / iEEG (5 demos)

Unifying substrate: a micro-ECoG / Utah-array grid over epileptic cortex. Anchors: the seizure
**core vs. penumbra** (Schevon 2012) and the **traveling-wave** debate.

### 3.1 `spatial_gof_ecog`
- **Question:** Is ictal propagation a genuine traveling wave, or a static hot-spot consistent
  with an inhomogeneous-Poisson (independent-site) null?
- **Ground truth:** 10×10 MEA (400 µm pitch). Two generators the student must distinguish:
  (a) static-hotspot inhomogeneous-Poisson null (spatially varying, temporally static rate);
  (b) migrating Gaussian wavefront at fixed velocity. Events drawn conditionally.
- **Recovery:** space-time inhomogeneous K-function / pair-correlation + global-rank envelope
  (held-out λ̂ to avoid plug-in bias). Envelope rejects the null **only** for the wave; a naive
  per-electrode rate map cannot separate the two.
- **Figure:** true intensity contours at 3 time slices (top) → observed K vs. Poisson envelope
  (middle) → verdict/statistic time series crossing significance only in the wave epoch.
- **Citations:** Smith 2016; Diamond 2021; Martinet 2017; Smith 2022.
- **Cross-links:** `spatial_stlgcp_microelectrode` (same wave, tracked as a moving surface).

### 3.2 `spatial_hawkes_ecog`
- **Question:** How much seizure activity is tonic **background** (hyperexcitable zone) vs.
  **self-excited recruitment** (the actively-recruited core)?
- **Ground truth:** LGCP background hot near a "seizure-onset zone" + a spatiotemporal Hawkes
  triggering kernel that recruits nearby-in-space-and-time discharges into cascades (core),
  while background events populate the penumbra sparsely.
- **Recovery:** branching EM (homogeneous background) and Cox-Hawkes (inhomogeneous LGCP
  background); recover the background surface *and* the true triggering kernel, separating
  "hyperexcitable tissue" from "recruited-by-cascade".
- **Figure:** true vs. estimated background heatmap; true vs. recovered triggering kernel;
  declustering (P(background)) map.
- **Citations:** Schevon 2012 (core/penumbra); Truccolo 2014; Martinet 2017. **Novelty:**
  Cox-Hawkes not yet applied to epilepsy — framed as an opportunity.
- **Cross-links:** `spatial_gof_ecog`, `spatial_gibbs` (exclusion-zone story).

### 3.3 `spatial_cluster_cox`
- **Question:** Are clinically "silent" microdischarges organised around a few hidden
  epileptogenic **hub** sites, or randomly scattered?
- **Ground truth:** Thomas process — K latent parent hubs (unobserved epileptogenic
  microdomains); around each, Poisson offspring microdischarges with isotropic Gaussian
  dispersion (sub-mm, sparse, per Stead 2010).
- **Recovery:** minimum-contrast estimation recovers parent intensity κ, mean offspring μ,
  dispersion σ. Naive "count active electrodes" cannot distinguish 5 hubs from 1 blob.
- **Figure:** true parents + offspring scatter; empirical vs. fitted Thomas K-function with a
  CSR reference; recovered-vs-true cluster count/radius.
- **Citations:** Stead 2010 (microseizures); Tobochnik 2021 (noncontiguous seizure hubs).
- **Cross-links:** `spatial_gibbs`.

### 3.4 `spatial_gibbs`
- **Question:** When is co-active-site / implanted-contact spacing governed by a
  minimum-distance **repulsion** constraint rather than independent placement?
- **Ground truth:** two placements on a grid — (a) CSR (independent) and (b) Strauss/hard-core
  enforcing a minimum inter-site distance (calibrated to real electrode-spacing / SEEG
  safety-margin literature).
- **Recovery:** Berman-Turner pseudo-likelihood recovers the interaction radius/strength for
  (b) and reports γ≈1 (no interaction) for (a). PCF dips below the hard-core distance only in
  the repulsive case.
- **Figure:** CSR vs. hard-core point patterns side by side; empirical vs. fitted PCF;
  recovered-vs-true minimum-distance.
- **Citations:** Meszéna 2026 (optimal electrode spacing); Rockhill 2000 (retinal mosaics);
  SEEG trajectory-planning min-distance (Sparks/Zombori et al.); Schevon 2012 (exclusion zone).
- **Cross-links:** `spatial_cluster_cox` (clustering vs. repulsion as opposite second-order
  signatures).

### 3.5 `spatial_stlgcp_microelectrode`
- **Question:** Can we track a moving wavefront (spreading depolarisation / seizure front)
  from sparse electrode events *without* a parametric wave-equation model?
- **Ground truth:** 8×8 grid; a latent GP intensity surface whose Gaussian peak moves along a
  smooth trajectory at a realistic speed (CSD ~2–5 mm/min per Lauritzen/Dreier, or a faster
  ictal front). Events via thinning.
- **Recovery:** spatiotemporal LGCP (Kronecker-Laplace) reconstructs the intensity surface
  frame-by-frame; recover peak trajectory + propagation speed vs. ground truth.
- **Figure:** true vs. estimated intensity heatmaps at 3–4 snapshots; overlaid true-vs-estimated
  peak-location trajectory; recovered-vs-true propagation speed.
- **Citations:** Muller 2018; Lauritzen 2011; Hartings 2011 (CSD in TBI ICU); Diamond 2021.
  **Novelty:** LGCP not yet applied to CSD tracking — framed as an opportunity.
- **Cross-links:** `spatial_gof_ecog`.

## 4. Thread 2 — Intracortical motor BCI (5 demos)

Unifying substrate: an M1/PMd population during reaching / cursor control. Anchor: the
toolbox's own PPAF lineage (Eden-Frank-Barbieri-Solo-Brown 2004) → BrainGate clinical decoding.

### 4.1 `decoding_clusterless`
- **Question:** Can we decode kinematics from **unsorted** multiunit events + waveform marks,
  and does clusterless decoding match/beat sorted decoding as sorting degrades?
- **Ground truth:** place-/kinematic-tuned units emit unsorted events carrying a 4-channel
  tetrode-amplitude mark from overlapping cluster distributions; a 1-D/2-D trajectory.
- **Recovery:** clusterless marked-point-process filter vs. a sorted decoder corrupted by a
  tunable merge/split error rate → clusterless degrades gracefully, sorted collapses.
- **Figure:** true trajectory + clusterless posterior band vs. sorted-decoder posterior at
  increasing sorting-error rates.
- **Citations:** Eden 2004 (PPAF); Deng 2015 (clusterless MPP filter); Kloosterman 2014;
  Denovellis 2021.
- **Cross-links:** `decoding_place_field`, `validation_pykalman`.

### 4.2 `decoding_place_field`
- **Question:** Is the **population vector** biased when preferred directions are non-uniform,
  and does a maximum-likelihood/Bayesian decode fix it?
- **Ground truth:** ~50–100 cosine/von-Mises-tuned M1 units, preferred directions deliberately
  clustered (per Sanger's critique), plus Poisson spiking; 8-direction center-out task.
- **Recovery:** fit tuning curves; decode held-out directions with (a) population vector and
  (b) ML → PV biased toward the dense region, ML unbiased.
- **Figure:** per-neuron tuning-curve panel; polar plot of true vs. PV vs. ML direction.
- **Citations:** Georgopoulos 1982/1986; Kettner 1988; Sanger 1996.
- **Cross-links:** `decoding_clusterless`, `latents_gpfa`.

### 4.3 `em_dynamax`
- **Question:** How should a decoder be **calibrated** from neural + behavioral data, and does
  closed-loop-aware (**ReFIT**) recalibration straighten trajectories?
- **Ground truth:** Wu-2006 linear-Gaussian arm state (position/velocity) coupled to a
  population with linear velocity encoding + Gaussian noise.
- **Recovery:** estimate transition/observation matrices by EM; compare naive KF (overshoots on
  closed-loop trials) vs. a ReFIT pass that relabels training kinematics with the intended
  target and recalibrates.
- **Figure:** true vs. decoded target-acquisition trajectories, standard-KF vs. ReFIT-KF.
- **Citations:** Wu 2006; Gilja 2012 (ReFIT-KF); Eden 2004; Kim 2011.
- **Cross-links:** `validation_pykalman`.

### 4.4 `latents_gpfa`
- **Question:** Do motor populations occupy a low-D **rotational manifold** recoverable on
  single trials?
- **Ground truth:** a 2–3-D skew-symmetric rotational latent (one trajectory per reach
  condition) projected through a fixed random loading matrix into ~50–100 "neurons" with
  condition-locked variability + Poisson noise.
- **Recovery:** GPFA recovers single-trial low-D trajectories vs. the known latent; fidelity
  improves with population size/trial count.
- **Figure:** ground-truth rotational phase portrait (one colour per condition) overlaid with
  GPFA-recovered single-trial trajectories; reconstruction fidelity vs. population size.
- **Citations:** Yu 2009 (GPFA); Churchland 2012 (rotational dynamics); Gallego 2018/2020
  (manifold stability).
- **Cross-links:** `decoding_place_field`.

### 4.5 `validation_pykalman`
- **Question:** Does the clinical workhorse **velocity Kalman decoder** replicate across
  implementations, and how does it degrade under **chronic drift**?
- **Ground truth:** Wu-style M1 velocity-encoding sim + Perge-style per-day unit gain/baseline
  drift + Downey-style unit turnover.
- **Recovery:** nstat's Kalman decoder vs. `pykalman` agree on the same binned counts;
  decode accuracy decays across simulated days without recalibration.
- **Figure:** nstat vs. pykalman decoded trajectories (agreement); accuracy vs. simulated
  day with/without recalibration.
- **Citations:** Wu 2006; Malik-Truccolo-Brown-Hochberg 2011; Perge 2013; Downey 2018.
- **Cross-links:** `em_dynamax`.

## 5. Thread 3 — DBS / movement-disorders MER (2 demos)

Unifying substrate: intraoperative microelectrode recording along an STN/GPi trajectory.
Anchor: ISI regularity as the targeting signature; beta-burst synchrony as the adaptive-DBS
biomarker.

### 5.1 `modulated_renewal_microelectrode`
- **Question:** Can a conditional-ISI model recover the **firing phenotypes** (tonic-regular /
  tremor-locked bursting / irregular-high-entropy) neurosurgeons use to confirm they are in
  the target nucleus?
- **Ground truth:** three units along one MER pass — (1) high-shape gamma (regular, low CV);
  (2) rate modulated at 4–6 Hz → tremor-locked bursts; (3) shape≈1 (irregular / high-entropy).
- **Recovery:** modulated-renewal CIF (gamma/IG density × covariate-driven rate) recovers rate
  + regularity (CV / burst index) + KS time-rescaling GoF, reproducing the three-way separation.
- **Figure:** rasters + ISI histograms with true generating density overlaid; recovered rate /
  tremor-drive vs. truth; recovered regularity parameter per phenotype; KS diagnostic.
- **Citations:** Bergman 1994; Levy 2000; Sterio 2002; Sedov 2025; Barbieri-Brown lineage
  (Barbieri 2004; Iyengar-Liao 1997).
- **Cross-links:** `metrics_spike_distances`.

### 5.2 `metrics_spike_distances`
- **Question:** Can time-resolved synchrony metrics track the emergence and collapse of
  pathological **beta-burst synchrony** (Parkinsonian OFF vs. adaptive-DBS)?
- **Ground truth:** 6–10 units with a time-varying shared beta-burst (15–30 Hz) drive toggling
  between synchronized (long bursts, OFF) and desynchronized (truncated bursts, adaptive-DBS)
  regimes.
- **Recovery:** SPIKE-distance / SPIKE-synchronization / ISI-distance in sliding windows track
  the injected synchrony envelope; summary compares "synchrony-burst duration" distributions
  between regimes.
- **Figure:** rasters with true synchrony-drive envelope shaded; time-resolved metric trace
  over the envelope; burst-duration distributions per regime.
- **Citations:** Kreuz 2011/2012; Satuvuori 2017; Tinkhauser 2017; Kühn 2009; Levy 2000.
- **Cross-links:** `modulated_renewal_microelectrode`.

## 6. Thread 4 — Clinical data rigor (5 demos)

Lighter clinical framing; infrastructure demos reframed around real clinical/BCI data
workflows. Methods and dependencies unchanged.

### 6.1 `interop_neo`
- **Framing:** one vendor-agnostic ingestion path for a multi-site iBCI trial (Blackrock
  Utah-array `.ns5/.nev` → Neo → nstat). **Citation:** Garcia 2014.

### 6.2 `interop_nwb`
- **Framing:** reproduce a preprocessing step from a real human iEEG dataset on DANDI — the
  **AJILE12** dandiset (naturalistic chronic ECoG + upper-body pose during epilepsy
  monitoring). Kept illustrative in prose; an actual (optional-dep) DANDI fetch is a
  possible follow-up, not part of this spec. **Citations:** Teeters 2015; Rübel 2022;
  Peterson 2022 (AJILE12, dandiset 000055).

### 6.3 `interop_pynapple`
- **Framing:** bug-resistant epoch bookkeeping — restrict analysis to interictal-only / BCI-
  trial windows via `IntervalSet`. **Citation:** Viejo 2023.

### 6.4 `validation_nemos`
- **Framing:** numerical cross-check of a reach-tuned encoding GLM — the rigor a clinical
  encoding-model paper's reviewers expect. **Citations:** Seabold 2010; NeMoS; Weber-Pillow 2017.

### 6.5 `validation_statsmodels`
- **Framing:** IRLS-vs-IRLS Poisson-GLM agreement to machine precision — catches any regression
  in the encoding path. **Citation:** Seabold 2010.

## 7. Scope guardrails / non-goals

- Public API, function signatures, dependencies: **unchanged**.
- Everything stays **fully synthetic** — no download/caching infrastructure. AJILE12/DANDI
  stays prose-illustrative unless a separate opt-in-dep follow-up is approved.
- Paper examples and notebooks: **untouched** (parity-locked).
- No new optional dependencies.
- Gallery integration preserved: each redesigned demo keeps its `demo_id`, its `manifest.yml`
  entry, and its `extras_descriptions.yml` entry (captions updated to the new scenario).

## 8. Verification plan

Per demo:
- Runs headless (`--export-figures --no-display`) with no errors.
- Figures clear the multi-signal image-content audit (`tools/parity/image_content_audit.py`).
- Recovery assertion still holds (the demo's numerical "the estimate recovers the truth" check).
- `demo_id` unchanged → `test_extras_docs`, `test_extras_examples` stay green; regenerate
  `docs/extras_gallery.html` after caption updates.

Global gates before merge: `make test`, `make freshness-check`, the extras doc/gallery tests.

## 9. Consolidated references (verified during the 2026-07-04 literature review)

*Epilepsy / iEEG:* Smith EH et al. 2016 Nat Commun 7:11098 · Martinet LE et al. 2017 Nat Commun
8:14896 · Diamond JM et al. 2021 Brain 144:1751 · Smith EH et al. 2022 eLife 11:e73541 ·
Schevon CA et al. 2012 Nat Commun 3:1060 · Truccolo W et al. 2014 J Neurosci 34:9927 · Stead M
et al. 2010 Brain 133:2789 · Tobochnik S et al. 2021 J Clin Neurophysiol 39:592 · Meszéna D et
al. 2026 Microsyst Nanoeng 12:41 · Rockhill RL et al. 2000 PNAS 97:2303 · Muller L et al. 2018
Nat Rev Neurosci 19:255 · Lauritzen M et al. 2011 J Cereb Blood Flow Metab 31:17 · Hartings JA
et al. 2011 Lancet Neurol 10:1058 · Krumin M, Shoham S 2010 Front Comput Neurosci 4:147.

*Motor BCI:* Eden UT et al. 2004 Neural Comput 16:971 · Deng X et al. 2015 Neural Comput 27:1438
· Kloosterman F et al. 2014 J Neurophysiol 111:217 · Denovellis EL et al. 2021 eLife 10:e64505 ·
Georgopoulos AP et al. 1982 J Neurosci 2:1527; 1986 Science 233:1416 · Kettner RE et al. 1988 J
Neurosci 8:2938 · Sanger TD 1996 J Neurophysiol 76:2790 · Wu W et al. 2006 Neural Comput 18:80 ·
Gilja V et al. 2012 Nat Neurosci 15:1752 · Kim SP et al. 2011 IEEE TNSRE 19:193 · Yu BM et al.
2009 J Neurophysiol 102:614 · Churchland MM et al. 2012 Nature 487:51 · Gallego JA et al. 2018
Nat Commun 9:4233; 2020 Nat Neurosci 23:260 · Hochberg LR et al. 2006 Nature 442:164; 2012
Nature 485:372 · Collinger JL et al. 2013 Lancet 381:557 · Malik WQ et al. 2011 IEEE TNSRE 19:25
· Perge JA et al. 2013 J Neural Eng 10:036004 · Downey JE et al. 2018 J Neural Eng 15:046016 ·
Pandarinath C et al. 2017 eLife 6:e18554; 2018 Nat Methods 15:805.

*DBS / MER:* Bergman H et al. 1994 J Neurophysiol 72:507 · Levy R et al. 2000 J Neurosci 20:7766
· Kaneoke Y, Vitek JL 1996 J Neurosci Methods 68:211 · Sterio D et al. 2002 Neurosurgery 50:58 ·
Moran A et al. 2008 Brain 131:3395 · Sedov A et al. 2025 Eur J Neurosci 61:e16653 · Iyengar S,
Liao Q 1997 Biol Cybern 77:289 · Barbieri R et al. 2004 Am J Physiol 288:H424 · Kreuz T et al.
2011 J Neurosci Methods 195:92; 2012 J Neurophysiol 109:1457 · Satuvuori E et al. 2017 J Neurosci
Methods 287:25 · Victor JD, Purpura KP 1996 J Neurophysiol 76:1310 · Houghton C, Sen K 2008 Neural
Comput 20:1495 · Kühn AA et al. 2009 Exp Neurol 215:380 · Tinkhauser G et al. 2017 Brain 140:1053
· Weinberger M et al. 2009 Exp Neurol 219:58.

*Data standards / rigor:* Garcia S et al. 2014 Front Neuroinform 8:10 · Teeters JL et al. 2015
Neuron 88:629 · Rübel O et al. 2022 eLife 11:e78362 · Peterson SM et al. 2022 Sci Data 9:184
(DANDI 000055) · Viejo G et al. 2023 eLife 12:e85786 · Seabold S, Perktold J 2010 SciPy Proc ·
Weber AI, Pillow JW 2017 Neural Comput (arXiv:1602.07389) · NeMoS (Flatiron Institute).

*Novelty (no prior neuro-application; framed as opportunity):* Cox-Hawkes doubly-stochastic
spatiotemporal model (statistics literature) applied to seizure core/penumbra; LGCP applied to
CSD/wavefront tracking.

## 10. Decisions confirmed at approval (2026-07-04)

The proposal was approved ("proceed with these"), which settled the three review questions to
their proposed defaults. They remain adjustable during spec review:

1. **Clinical framings accepted as proposed** — incl. the tremor/tonic/irregular triad for
   `modulated_renewal` (dystonia pause-cells could be added as a fourth phenotype later if
   desired).
2. **Both novelty framings retained** — Cox-Hawkes for core/penumbra and LGCP for CSD tracking
   are presented as "a method not yet applied to this clinical problem", never as a fabricated
   citation.
3. **`interop_nwb` AJILE12/DANDI stays prose-illustrative** — no data-fetch dependency in this
   spec; an optional-dep DANDI fetch is a possible future follow-up.
