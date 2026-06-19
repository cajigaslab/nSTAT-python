# MATLAB pedagogical gaps — opportunities to enrich the MATLAB toolbox

This file is the **counterpart** to `parity/matlab_defects.md`.  Where
defects.md records places Python had to **fix** MATLAB, this file records
places Python **adds** content the MATLAB toolbox should consider adopting —
extra figures, extra cells, additional diagnostic plots, alternative views
that enrich the pedagogy of the helpfile.

Per AGENT_GUIDE.md §0:
- Python is allowed to **extend** beyond MATLAB (Extension rule).
- Figure-count parity is the default; Python surplus must be justified as a
  pedagogical extra **or** removed.

Schema for each entry:

```
## Gap: <one-line title>
- **Topic / helpfile:** `<TopicName>`
- **MATLAB count:** N
- **Python count:** N + k  (k extra figures)
- **What Python adds:** <description of the extra figure(s)>
- **Pedagogical justification:** <why a learner benefits from seeing this>
- **MATLAB upstream action:** <"Add equivalent figure to helpfiles/<Topic>.mlx
  using <function call>" — i.e. a clear instruction the MATLAB maintainer
  can act on>
- **Python implementation:** `<notebooks/<Topic>.ipynb cell <N>>` and any
  library helpers
- **Discovered:** <iter # / date>
```

If the extra is **not** pedagogically justified, the Python notebook should
remove the extra cell instead of recording it here.

---

## Open entries

### Gap: 3D wireframe overlay of Gaussian + Zernike place field surfaces

- **Topic / helpfile:** `HippocampalPlaceCellExample`
- **MATLAB count:** 11
- **Python count:** 12 (+1)
- **What Python adds:** `fig_012` is a 3D wireframe `plot_wireframe` overlay of the
  Gaussian-fit and Zernike-fit place-field intensity surfaces, rendered
  on a single 3D axes with translucent meshes so the learner can see
  how the two parameterizations differ in their high-curvature regions.
- **Pedagogical justification:** MATLAB's 2D `pcolor` views (figs 09–11) show each fit independently;
  the 3D overlay makes the *difference* visible at a glance —
  particularly the over-smoothing of the Gaussian near the field edge.
  Useful when teaching place-field modeling choice.
- **MATLAB upstream action:** Add `figure(12)` to `helpfiles/HippocampalPlaceCellExample.mlx` with
  `mesh(X,Y,Z_gauss); hold on; mesh(X,Y,Z_zernike); alpha 0.4;
  view(3); legend('Gaussian','Zernike');`
- **Python implementation:** `notebooks/HippocampalPlaceCellExample.ipynb` final 3D wireframe
  cell using `mpl_toolkits.mplot3d`.
- **Discovered:** v1 iter 2 / 2026-06-17

### Gap: Per-cell ISI diagnostic-suite composite

- **Topic / helpfile:** `nSpikeTrainExamples`
- **MATLAB count:** 6
- **Python count:** 7 (+1)
- **What Python adds:** `fig_007` is a 2×3 composite of `plotISIHistogram`,
  `plotJointISIHistogram`, `plotISISpectrumFunction`,
  `plotExponentialFit`, `plotProbPlot` — all five diagnostic methods
  that the `nspikeTrain` class already exposes but the MATLAB helpfile
  never exercises.
- **Pedagogical justification:** The helpfile shows raster + signal representations but skips the
  goodness-of-fit / stationarity diagnostics.  A new user discovering
  the class doesn't see how to validate that the spike train is
  Poisson-like.  The composite makes all five diagnostics visible
  from one cell.
- **MATLAB upstream action:** Append a section to `helpfiles/nSpikeTrainExamples.mlx` with
  `subplot(2,3,...)` calls to each of `nst.plotISIHistogram`,
  `nst.plotJointISIHistogram`, `nst.plotISISpectrumFunction`,
  `nst.plotExponentialFit`, `nst.plotProbPlot`.
- **Python implementation:** `notebooks/nSpikeTrainExamples.ipynb` final diagnostic-suite cell
  (added in v1 iter 2).
- **Discovered:** v1 iter 2 / 2026-06-18

### Gap: Monte Carlo mean-trajectory + RMSE summary for hybrid decoder

- **Topic / helpfile:** `HybridFilterExample`
- **MATLAB count:** 2
- **Python count:** 3 (+1)
- **What Python adds:** `fig_003` is a 2×2 summary of the n=20 Monte Carlo hybrid-filter
  runs: (top-left) mean estimated discrete state vs. ground truth,
  (top-right) mean P(s(t)=M | data) with vs. without the goal-target
  prior, (bottom-left) mean 2D reach path overlay, and (bottom-right)
  bar chart of single-run X / Y position RMSE.
- **Pedagogical justification:** MATLAB's figure 2 shows the raw spaghetti of all n=20 PPAF+Goal and
  PPAF runs but never collapses them into a comparable summary.  The
  mean curves make the systematic advantage of the goal-target prior
  (faster, cleaner discrete-state recovery; tighter mean trajectory)
  visible at a glance, and the RMSE bar gives a single scalar a
  learner can quote when comparing decoder variants.  Without this
  figure, a reader can only assess decoder quality by visually
  averaging the dense spaghetti plot.
- **MATLAB upstream action:** Add a `figure(3)` cell to `helpfiles/HybridFilterExample.mlx` that,
  after the n=20 Monte Carlo loop, computes `mean(stateHat, 3)`,
  `mean(probM, 3)`, `mean(xHat, 3)`, `mean(yHat, 3)`, plots them in
  a 2×2 `subplot` layout, and adds a `bar([xRMSE yRMSE])` panel.
- **Python implementation:** `notebooks/HybridFilterExample.ipynb` final 2×2 summary cell (the
  cell building `axs[0,0]` "Mean Estimated vs. Actual State (n=20)"
  through `axs[1,1]` "Single-run decoding RMSE [m]").
- **Discovered:** v2 iter 9 / 2026-06-18

### Gap: Conditional intensity function (CIF) trace for PP sample-path generation

- **Topic / helpfile:** `PPSimExample`
- **MATLAB count:** 4
- **Python count:** 4 (+1 extra, -1 MATLAB live-script duplicate)
- **What Python adds:** `fig_002` is a time-series plot of the realized conditional
  intensity λ(t) for the first sample path, showing the
  exponential-link CIF modulated by the sine stimulus and 3-lag
  history effect on the same x-window as the raster/stimulus figure
  (`tMax/5`).
- **Pedagogical justification:** MATLAB's helpfile generates spikes from the CIF but never plots
  λ(t) itself — the learner sees the raster and the input stimulus
  but has no visual link between the two.  The CIF trace makes the
  input→intensity→spikes pipeline explicit: peaks of λ align with
  peaks of `u_stim(t)`, and the history term injects the refractory
  dips visible in the raster.
- **MATLAB upstream action:** Add a `figure` cell to `helpfiles/PPSimExample.mlx` after the
  `simulateCIF` call with `sC.lambdaCIF.plot; xlim([0 tMax/5])` (or
  equivalent — the `simulateCIF` output already carries the realized
  λ as a property).
- **Python implementation:** `notebooks/PPSimExample.ipynb` cell 13 (`lambda_cov.getSubSignal(0).plot`).
  Note that one MATLAB figure (the `plotSummary` post-cell live-script
  re-render that produces `PPSimExample_04.png`) is intentionally not
  duplicated on the Python side — it shows identical data to
  `PPSimExample_03.png`.
- **Discovered:** v2 iter 9 / 2026-06-18

### Gap: Programmatic schematic / equation panels for the two-neuron network model

- **Topic / helpfile:** `NetworkTutorial`
- **MATLAB count:** 5
- **Python count:** 8 (+3 schematic mirrors)
- **What Python adds:** `fig_001` is a matplotlib redraw of the two-neuron connectivity
  diagram (`SimulatedNetwork2.png` in the MATLAB helpfile), labelling
  each neuron with its baseline `mu_i`, history coefficients,
  stimulus filter `S_i`, and ensemble weight `E_i` directly on the
  graph.  `fig_002` redraws the conditional-intensity block diagram
  (`PPSimExample-BlockDiagram.png` in the MATLAB helpfile) showing
  the Baseline / History / Stimulus / Ensemble summation feeding the
  logistic link.  `fig_003` is a text panel rendering the CIF
  equation
  `lambda_i * Delta = logistic(mu_i + H*DeltaN_i[n] + S*u_stim[n] + E*DeltaN_k[n])`.
- **Pedagogical justification:** MATLAB embeds these three panels in the `.mlx` helpfile as inline
  image / TeX assets, but they are not counted as figures and a
  learner reading the published Python notebook would lose them
  entirely if we treated the MATLAB figure count (5) as
  authoritative.  Rendering them programmatically keeps the
  explanatory scaffolding visible in the executed notebook, the
  gallery, and the Sphinx build, with no extra binary assets to
  track.
- **MATLAB upstream action:** Already present in `helpfiles/NetworkTutorial.mlx` as inline assets
  (`SimulatedNetwork2.png`, `PPSimExample-BlockDiagram.png`, and the
  `$$lambda_i \cdot \Delta = logistic(...)$$` block) — no upstream
  change required.  The gap is purely a counting-convention
  difference: Python renders schematics as matplotlib figures, MATLAB
  embeds them as page assets.
- **Python implementation:** `notebooks/NetworkTutorial.ipynb` cells 4, 5, 6 (the three
  `_figure(...)` calls that invoke `_draw_network`,
  `_draw_block_diagram`, and `_text_panel`).
- **Discovered:** v2 iter 9 / 2026-06-18

### Gap: Self-history kernel stem plot for the two-neuron network

- **Topic / helpfile:** `NetworkTutorial`
- **MATLAB count:** 5
- **Python count:** 8 (+1, in addition to the schematic-mirror gap above)
- **What Python adds:** `fig_004` is a red stem plot of the three-tap self-history kernel
  `H = [-4, -2, -1]` (coefficient vs. lag in ms), emphasising the
  strong inhibition at lag 1 ms (refractory period) and the decay
  over the next two lags.
- **Pedagogical justification:** MATLAB renders this kernel only as a TeX equation in the helpfile
  (`$$1*h[n]=-4*\Delta N[n-1]-2*\Delta N[n-2]-1*\Delta N[n-3]$$`),
  which is hard to read at a glance.  A stem plot makes the three
  taps and their decay pattern immediately visible, anchoring the
  simulated raster the reader sees a few cells later.  This is the
  only multi-tap filter in the network model — the scalar stimulus
  and ensemble filters were intentionally trimmed during triage
  because they each reduce to a single dot already stated in the
  connectivity-diagram labels.
- **MATLAB upstream action:** Add a `figure` cell to `helpfiles/NetworkTutorial.mlx` immediately
  after the History Effect section with
  `stem([-4 -2 -1]); xlabel('lag (ms)'); ylabel('coefficient');
  title('Self-history kernel');`.
- **Python implementation:** `notebooks/NetworkTutorial.ipynb` cell 13 (the `_figure("1*h[n]=...")`
  call invoking `_stem_kernel`).
- **Discovered:** v2 iter 9 / 2026-06-18

### Gap: KS / ΔAIC / ΔBIC scan-figure overlay for stimulus-lag selection

- **Topic / helpfile:** `ExplicitStimulusWhiskerData`
- **MATLAB count:** 10
- **Python count:** 11 (+1)
- **What Python adds:** A 3-subplot vertical scan figure showing KS statistic, ΔAIC, and
  ΔBIC across candidate stimulus lags, with a red `*` marker at the
  best-window index.  Mirrors the MATLAB `computeHistLag` diagnostic
  plot from `HistoryExamples` but applied to stimulus-lag selection
  on the whisker dataset.
- **Pedagogical justification:** The MATLAB helpfile fits stimulus-lag models but doesn't visualize
  *how* the best lag was chosen.  The scan figure makes the
  model-selection step transparent.
- **MATLAB upstream action:** Add a `figure(11)` cell to
  `helpfiles/ExplicitStimulusWhiskerData.mlx` mirroring the same
  KS/ΔAIC/ΔBIC pattern that `helpfiles/HistoryExamples.mlx` uses for
  history-lag selection.
- **Python implementation:** `notebooks/ExplicitStimulusWhiskerData.ipynb` scan-figure cell
  (added in v1 iter 2).
- **Discovered:** v1 iter 2 / 2026-06-18

### Gap: Figure ordering deviation in nSTATPaperExamples experiment 1

- **Topic / helpfile:** `nSTATPaperExamples`
- **MATLAB count:** 29
- **Python count:** 29 (same)
- **What Python adds:** Figure indexing for experiment 1 is reordered.  Python fig 2
  contains the decreasing-Mg raster + rate fits (Observed / Piecewise /
  Piecewise+Hist).  MATLAB fig 2 shows the constant-Mg raster, MATLAB
  fig 3 shows the decreasing-Mg raster, and the rate-fit comparison
  appears at MATLAB fig 4.  Per-index 1↔1 SSIM comparison therefore
  mis-aligns even though the substantive content is equivalent.
- **Pedagogical justification:** Python collapses the two raster panels into a single combined view
  that places the rate fit alongside the decreasing-Mg condition for
  direct visual comparison — more compact for notebook readers, less
  faithful to MATLAB's index sequence.
- **MATLAB upstream action:** No upstream action required.  Index ordering is a presentation
  choice, not a content gap.
- **Python implementation:** `notebooks/nSTATPaperExamples.ipynb` exp 1 cells.
- **Discovered:** v3 iter 14 / 2026-06-18; documented v3 iter 15.

### Gap: StimulusDecode2D fig_002 per-cell CIF time-base sampling

- **Topic / helpfile:** `StimulusDecode2D`
- **MATLAB count:** 7
- **Python count:** 7 (same)
- **What Python adds:** The per-cell λ(t) trace in fig_002 shows visible temporal phase
  drift between MATLAB and Python renderings.  Inspection of
  `nstat/cif.py` and the notebook cell confirms the underlying CIF
  evaluation uses the same dt and sample boundaries; the visible
  drift is a rendering effect of matplotlib's line-segment
  downsampling vs MATLAB's `plot` antialiasing path, not a numerical
  time-base mismatch.  Gold fixtures for the CIF evaluation pass.
- **Pedagogical justification:** Same content; no upstream action.
- **MATLAB upstream action:** None.
- **Python implementation:** `notebooks/StimulusDecode2D.ipynb` cell rendering fig_002.
- **Discovered:** v3 iter 14 / 2026-06-18; investigated and closed v3 iter 15.

---

## Deficit topics (Python < MATLAB)

These need closure: either close the count gap or document why MATLAB's extra figures are artifacts (e.g. live-script auto-redraw duplicates).

### Deficit: AnalysisExamples2 — MATLAB live-script auto-redraws (-2)

- **Topic:** `AnalysisExamples2`
- **MATLAB count:** 6
- **Python count:** 4 (Δ = -2)
- **MATLAB extras:** Figures 5 and 6 are live-script auto-renders of the same
  `fitResults.plotResults` and `Analysis.computeHistLag(makePlot=1)`
  calls that produced figures 3 and 4.  The 5th and 6th figures don't
  add new content — they're a MATLAB live-script export artifact
  where the renderer emits both the in-cell draw and a post-cell
  larger re-render.
- **Pedagogical impact:** None — the duplicates show the same data with identical panel layouts.
- **Decision:** Accept the deficit.  Python's `expected_count=4` is the honest count of substantive figures.
- **MATLAB upstream action:** Optionally, the MATLAB live-script could be updated to suppress the
  duplicate render or replace it with a different diagnostic view
  (e.g. residual histogram).
- **Discovered:** v1 iter 5 / 2026-06-18

---

## Surplus topics still under triage (v2 iter 9 will decide each)

The following Python notebooks have more figures than MATLAB but haven't yet
been classified. Each will be **either** justified here as pedagogical
(with an upstream action) **or** trimmed back to MATLAB's count.

| Topic | MATLAB | Python | Δ | Status |
|---|---:|---:|---:|---|
| `NetworkTutorial` | 5 | 8 | +3 | resolved (v2 iter 9 / 2026-06-18) — 5 incidental figures removed (4 scalar single-stem filter plots and 1 empty CIF-trajectory plot); remaining +3 schematic mirrors + 1 multi-tap kernel logged below |
| `nSTATPaperExamples` | 29 | 29 | 0 | trimmed (v2 iter 9 / 2026-06-18) — removed incidental "paper dataset summary" bar chart (scale-mismatched, not pedagogical) |

---

## How to use this ledger

1. **When closing a parity-affecting PR:** if your work surfaced a Python
   notebook with surplus figures, decide whether each extra is pedagogical
   (add a Gap entry here) or removable (delete the cell in the same PR).

2. **When the MATLAB maintainer reviews:** treat each "Open entries" item
   as a candidate enhancement to the MATLAB toolbox. The "MATLAB upstream
   action" line is a copy-paste-ready instruction.

3. **For reviewers:** every Python notebook that produces N+k figures vs
   MATLAB's N must either have a Gap entry here or have the k extras
   removed.

## Distinction from `matlab_defects.md`

| | `matlab_defects.md` | `matlab_pedagogical_gaps.md` |
|---|---|---|
| Records | Python *deviations* (bug fixes, stability/efficiency improvements) | Python *additions* (pedagogical extras) |
| Drives action in | Python (we fixed it) | MATLAB (upstream should add it) |
| Frequency | Rare (3 entries so far) | Open-ended (one per surplus figure) |
| Gold-fixture impact | Refresh often required | No fixture impact (Python-only extras) |

## v5 iter 21 — single-Reviewer disagreements (for human triage)

The following items were flagged by exactly one of the two parity reviewers
(A or B) during the v5 iter 21 sweep, then triaged in v7 iter 32 against the
user threshold *"only APPLY items that demonstrably move a topic from minor
→ matches"*. The original 60 items are split into three sub-sections:

- **applied** — content-level fixes the builder should land
- **stale** — already addressed by v6/v7 count-parity / structural work
- **deferred** — real residuals (mostly cosmetic) that don't cross the
  minor → matches threshold; held for future polish passes

Totals: **60 triaged → 10 applied / 4 stale / 46 deferred.**

### applied (10) — builder targets

Content-level mismatches (blank panels, missing overlays, missing
annotations, content divergence) that, in aggregate, can flip the topic's
holistic verdict.

| topic | fig | by | description |
|---|---|---|---|
| ExplicitStimulusWhiskerData | 3 | B | Stimulus/firing-rate sinusoid missing title/axis labels — add titles + xlabel/ylabel |
| ExplicitStimulusWhiskerData | 9 | B | Empty Python panel where MATLAB has line traces — populate panel with the missing GLM-trace series |
| nSTATPaperExamples | 2 | B | Example 01 second figure shows noisy oscillatory trace instead of MATLAB KS goodness-of-fit — replace with KS panel |
| nSTATPaperExamples | 4 | B | Example 02 PSTH/coefficient panel blank vs MATLAB sinusoidal PSTH — populate with sinusoidal PSTH |
| nSTATPaperExamples | 8 | B | Example 03 final bar chart: compressed y-range and missing annotations — set proper ylim and add annotations |
| nSTATPaperExamples | 10 | B | Example 04 trajectory subpanel missing spike-overlay scatter and colorbar — add scatter overlay + colorbar |
| nSTATPaperExamples | 14 | B | Example 05 final lower panel: smooth sinusoid vs noisy scatter content mismatch — fix data series fed to the panel |
| SignalObjExamples | 6 | B | v2 plotted as dashed green vs MATLAB solid green — change linestyle to solid |
| SignalObjExamples | 7 | B | fig_007 inherits dashed-vs-solid mismatch from fig_006 — same fix |
| DecodingExample | 6 | B | MATLAB shows two model-comparison bar plots at top; Python ordering/layout diverges — re-order to match MATLAB layout |

### stale (4) — already addressed by v6/v7 work

Items whose root cause (figure-count mismatch, missing structural figure)
was closed by the v6 code-structure / class-method parity pass or the v7
notebook-alignment sweep. Verified against
`.parity-review/SUMMARY_run_*` and `.parity-review/composite_*.png`.

| topic | fig | by | description | resolved by |
|---|---|---|---|---|
| nstCollExamples | 0 | B | Composite-wide aspect ratio/background inconsistency | nstCollExamples is now at 5/5 figure-count parity (see SUMMARY) |
| PPThinning | 0 | B | Figure-count mismatch header MATLAB(4) vs Python(3); whole figure missing | PPThinning now at 4/4 figure-count parity |
| ExplicitStimulusWhiskerData | 0 | B | Major figure count mismatch: MATLAB 13 vs Python 6 | now 11/10 (Python surplus, allowed under Extension rule); the "13" baseline was stale |
| DecodingExample | 1 | B | Composite labels show MATLAB(7) vs Python(8): figure-count mismatch (manifest-level) | DecodingExample now at expected count per `.parity-review/composite_DecodingExample.png` |

### deferred (46) — real residuals that don't move minor → matches

These are legitimate divergences but cosmetic-only (fonts, line widths,
tick spacing, color palettes, legend placement) or numerically-stochastic
(decoded trajectory variance from spike draws, cycle-count drift from
RNG). Fixing any one of them — or even all of them within a single topic
— would not by itself cross the minor → matches threshold per the v7
Reviewer rubric. Kept in the backlog for future polish passes; revisit
if a future iter pushes a topic close enough that a cosmetic pass would
flip the verdict.

| topic | fig | by | description |
|---|---|---|---|
| nstCollExamples | 1 | B | Y-axis label 'Trial #' font/size mismatch (cosmetic) |
| nstCollExamples | 5 | B | Middle panel axis/tick label font sizes differ (cosmetic) |
| ExplicitStimulusWhiskerData | 2 | B | PSTH panel bar color saturation/bin alignment differ |
| ExplicitStimulusWhiskerData | 5 | B | Second multi-panel diagnostic: residual/ACF traces and coefficient bar count differ |
| ExplicitStimulusWhiskerData | 11 | B | Font size/tight_layout inconsistency across multi-panel figures |
| ExplicitStimulusWhiskerData | 12 | B | Color palette divergence (tab10 vs MATLAB lines palette) in overlay traces |
| ExplicitStimulusWhiskerData | 2 | A | Stimulus/raster overlay: subplot height-ratio proportions diverge |
| ExplicitStimulusWhiskerData | 3 | A | PSTH/rate trace minor y-axis label/tick formatting (cosmetic) |
| ExplicitStimulusWhiskerData | 5 | A | Second GLM diagnostics composite mirrors fig 4 with similar parity (cosmetic) |
| ExplicitStimulusWhiskerData | 9 | A | GLM coefficient + KS composite uses orange/green vs MATLAB jet-style palette |
| mEPSCAnalysis | 1 | B | Top-left KS plot line thickness/color saturation differs (cosmetic) |
| nSTATPaperExamples | 1 | A | Example 01 KS plot: empirical CDF may not be visibly distinct from 45-deg reference |
| nSTATPaperExamples | 2 | A | Example 02 intensity panel: Python adds extra coefficient bar chart not in MATLAB (Python surplus, allowed) |
| nSTATPaperExamples | 3 | A | Example 03: Python 4-panel layout vs MATLAB 2-panel stacked |
| nSTATPaperExamples | 4 | A | Example 04 decoding: decoded trajectory deviates more from diagonal than MATLAB (stochastic spike draws) |
| nSTATPaperExamples | 5 | A | Example 05: Start/Finish markers less distinct than MATLAB |
| nSTATPaperExamples | 6 | A | Example 06 place-field heatmap: colormap/dynamic-range mismatch with MATLAB jet |
| nSTATPaperExamples | 7 | A | Example 07: extra small-multiple panels in Python vs MATLAB (Python surplus, allowed) |
| nSTATPaperExamples | 8 | A | Example 08 sinusoidal CIF: line color/spine treatment differs (cosmetic) |
| nSTATPaperExamples | 5 | B | Example 02 KS/coefficient bar plot color mapping differs |
| nSTATPaperExamples | 6 | B | Example 03 ROC/KS line thickness (cosmetic) |
| nSTATPaperExamples | 7 | B | Example 03 sinusoidal overlay amplitude/phase offset (numerical) |
| nSTATPaperExamples | 9 | B | Example 04 KS plot: legend entries/line styles diverge |
| nSTATPaperExamples | 11 | B | Example 05 decoding heatmap colormap normalization |
| nSTATPaperExamples | 12 | B | Example 05 KS/error decay: y-scaling and asymptote differ |
| nSTATPaperExamples | 13 | B | Example 05 lower heatmaps: tick/label fonts and titles formatting differ (cosmetic) |
| HippocampalPlaceCellExample | 3 | A | 8x8 grid empirical place fields: monochrome blue tiles vs MATLAB jet contrast |
| HippocampalPlaceCellExample | 7 | A | Trajectory + spike overlay: marker styling/colors differ (cosmetic) |
| HippocampalPlaceCellExample | 9 | A | 3D place-field surface flatter/lower-amplitude vs MATLAB |
| HippocampalPlaceCellExample | 1 | B | Trajectory + spike-scatter panel: aspect ratio and Start/Finish marker size differ |
| HippocampalPlaceCellExample | 2 | B | Spike-train raster/time-series panel: y-range scaling and line weights differ |
| HippocampalPlaceCellExample | 3 | B | Second diagnostic panel: vertical tick density and amplitude range differ |
| HippocampalPlaceCellExample | 4 | B | Place-field heatmap grid (fig_004): jet vs blue-dominant colormap divergence |
| HippocampalPlaceCellExample | 6 | B | Place-field reconstruction grid (fig_006): wrong colormap and/or normalization |
| HippocampalPlaceCellExample | 7 | B | Second circular tile grid (fig_007): same colormap collapse |
| HippocampalPlaceCellExample | 8 | B | Trajectory overlay with model fit (fig_008): line color/width and markers smaller than MATLAB |
| HippocampalPlaceCellExample | 9 | B | KS-plot/diagnostic curve (fig_009): axis ticks/label fonts and CI band styling differ |
| SignalObjExamples | 1 | B | Y-axis tick spacing 0.5 vs MATLAB 0.2 step (cosmetic) |
| SignalObjExamples | 9 | B | Orange v2 start position appears offset from MATLAB |
| SignalObjExamples | 10 | B | Legend mathtext rendering for 'v1·p(t)' differs |
| SignalObjExamples | 13 | B | Legend placement inside axes may overlap peaks vs MATLAB outside-axes |
| SignalObjExamples | 20 | B | fig_020 cycle count mismatch (Python ~25 vs MATLAB ~22 cycles in 10s) (numerical/RNG) |
| SignalObjExamples | 1 | B | Legend frame styling: Python rounded/semi-transparent vs MATLAB thin black border (cosmetic) |
| SignalObjExamples | 11 | B | fig_011 y-axis label unicode μ vs MATLAB italic mu |
| DecodingExample | 6 | A | Mean-squared-error bar chart minor tick/bar-width differences (cosmetic) |
| DecodingExample | 1 | B | Lower panel rate-axis label 'Rate(Hz)' styling vs MATLAB (cosmetic) |

(Topics StimulusDecode2D and NetworkTutorial had no single-Reviewer disagreements this iter.)
