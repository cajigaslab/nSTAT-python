# MATLAB pedagogical gaps — opportunities to enrich the MATLAB toolbox

This file is the **counterpart** to `parity/matlab_defects.md`. Where defects.md
records places Python had to **fix** MATLAB, this file records places Python
**adds** content the MATLAB toolbox should consider adopting — extra figures,
extra cells, additional diagnostic plots, alternative views that enrich the
pedagogy of the helpfile.

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
- **What Python adds:** `fig_012` is a 3D wireframe `plot_wireframe` overlay
  of the Gaussian-fit and Zernike-fit place-field intensity surfaces,
  rendered on a single 3D axes with translucent meshes so the learner can
  see how the two parameterizations differ in their high-curvature regions.
- **Pedagogical justification:** MATLAB's 2D `pcolor` views (figs 09–11)
  show each fit independently; the 3D overlay makes the *difference*
  visible at a glance — particularly the over-smoothing of the Gaussian
  near the field edge. Useful when teaching place-field modeling choice.
- **MATLAB upstream action:** Add `figure(12)` to
  `helpfiles/HippocampalPlaceCellExample.mlx` with `mesh(X,Y,Z_gauss);
  hold on; mesh(X,Y,Z_zernike); alpha 0.4; view(3); legend('Gaussian','Zernike');`
- **Python implementation:** `notebooks/HippocampalPlaceCellExample.ipynb`
  final 3D wireframe cell using `mpl_toolkits.mplot3d`.
- **Discovered:** v1 iter 2 / 2026-06-17

### Gap: Per-cell ISI diagnostic-suite composite

- **Topic / helpfile:** `nSpikeTrainExamples`
- **MATLAB count:** 6
- **Python count:** 7 (+1)
- **What Python adds:** `fig_007` is a 2×3 composite of
  `plotISIHistogram`, `plotJointISIHistogram`, `plotISISpectrumFunction`,
  `plotExponentialFit`, `plotProbPlot` — all five diagnostic methods that
  the `nspikeTrain` class already exposes but the MATLAB helpfile never
  exercises.
- **Pedagogical justification:** The helpfile shows raster + signal
  representations but skips the goodness-of-fit / stationarity
  diagnostics. A new user discovering the class doesn't see how to
  validate that the spike train is Poisson-like. The composite makes
  all five diagnostics visible from one cell.
- **MATLAB upstream action:** Append a section to
  `helpfiles/nSpikeTrainExamples.mlx` with `subplot(2,3,...)` calls to
  each of `nst.plotISIHistogram`, `nst.plotJointISIHistogram`,
  `nst.plotISISpectrumFunction`, `nst.plotExponentialFit`,
  `nst.plotProbPlot`.
- **Python implementation:** `notebooks/nSpikeTrainExamples.ipynb` final
  diagnostic-suite cell (added in v1 iter 2).
- **Discovered:** v1 iter 2 / 2026-06-18

### Gap: KS / ΔAIC / ΔBIC scan-figure overlay for stimulus-lag selection

- **Topic / helpfile:** `ExplicitStimulusWhiskerData`
- **MATLAB count:** 10
- **Python count:** 11 (+1)
- **What Python adds:** A 3-subplot vertical scan figure showing KS
  statistic, ΔAIC, and ΔBIC across candidate stimulus lags, with a red
  `*` marker at the best-window index. Mirrors the MATLAB `computeHistLag`
  diagnostic plot from `HistoryExamples` but applied to stimulus-lag
  selection on the whisker dataset.
- **Pedagogical justification:** The MATLAB helpfile fits stimulus-lag
  models but doesn't visualize *how* the best lag was chosen.
  The scan figure makes the model-selection step transparent.
- **MATLAB upstream action:** Add a `figure(11)` cell to
  `helpfiles/ExplicitStimulusWhiskerData.mlx` mirroring the same
  KS/ΔAIC/ΔBIC pattern that `helpfiles/HistoryExamples.mlx` uses for
  history-lag selection.
- **Python implementation:** `notebooks/ExplicitStimulusWhiskerData.ipynb`
  scan-figure cell (added in v1 iter 2).
- **Discovered:** v1 iter 2 / 2026-06-18

---

## Surplus topics still under triage (v2 iter 9 will decide each)

The following Python notebooks have more figures than MATLAB but haven't yet
been classified. Each will be **either** justified here as pedagogical
(with an upstream action) **or** trimmed back to MATLAB's count.

| Topic | MATLAB | Python | Δ | Status |
|---|---:|---:|---:|---|
| `HybridFilterExample` | 2 | 3 | +1 | pending triage |
| `NetworkTutorial` | 5 | 13 | +8 | pending triage |
| `PPSimExample` | 4 | 9 | +5 | pending triage |
| `nSTATPaperExamples` | 29 | 30 | +1 | pending triage |

---

## Deficit topics (Python < MATLAB)

These need closure: either close the count gap or document why MATLAB's
extra figures are artifacts (e.g. live-script auto-redraw duplicates).

### Deficit: AnalysisExamples2 — MATLAB live-script auto-redraws (-2)

- **Topic:** `AnalysisExamples2`
- **MATLAB count:** 6
- **Python count:** 4 (Δ = -2)
- **MATLAB extras:** Figures 5 and 6 are live-script auto-renders of the
  same `fitResults.plotResults` and `Analysis.computeHistLag(makePlot=1)`
  calls that produced figures 3 and 4. The 5th and 6th figures don't
  add new content — they're a MATLAB live-script export artifact where
  the renderer emits both the in-cell draw and a post-cell larger
  re-render.
- **Pedagogical impact:** None — the duplicates show the same data with
  identical panel layouts.
- **Decision:** Accept the deficit. Python's `expected_count=4` is the
  honest count of substantive figures.
- **MATLAB upstream action:** Optionally, the MATLAB live-script could
  be updated to suppress the duplicate render or replace it with a
  different diagnostic view (e.g. residual histogram).
- **Discovered:** v1 iter 5 / 2026-06-18

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
