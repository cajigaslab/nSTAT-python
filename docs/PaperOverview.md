# Paper-Aligned Toolbox Map

This page aligns the Python port with the original nSTAT toolbox paper:

- Cajigas I, Malik WQ, Brown EN. *nSTAT: Open-source neural spike train
  analysis toolbox for Matlab*. Journal of Neuroscience Methods 211:245-264
  (2012)
- DOI: [10.1016/j.jneumeth.2012.08.009](https://doi.org/10.1016/j.jneumeth.2012.08.009)
- PubMed: [22981419](https://pubmed.ncbi.nlm.nih.gov/22981419/)
- PMC full text: [PMC3491120](https://pmc.ncbi.nlm.nih.gov/articles/PMC3491120/)

## Class Hierarchy and Object Model

The Python port preserves the MATLAB toolbox's core object groupings:

- Signal and covariate primitives: `SignalObj`, `Covariate`,
  `ConfidenceInterval`, `CovColl`
- Spiking data structures: `nspikeTrain`, `nstColl`, `History`, `Events`
- Experiment and configuration objects: `Trial`, `TrialConfig`, `ConfigColl`
- Modeling and inference objects: `CIF`, `Analysis`, `FitResult`,
  `FitResSummary`, `DecodingAlgorithms`

Related navigation pages:

- [Class Definitions](ClassDefinitions.md)
- [Example Index](Examples.md)

## Fitting and Assessment Workflow

The paper's core GLM workflow (Section 2.3) maps to Python as:

1. Build trial data with `Trial`, `CovColl`, and `nstColl`.
2. Define model configurations with `TrialConfig` and `ConfigColl`.
3. Fit and evaluate analyses with `Analysis` and `FitResult`.
4. Summarize across fits with `FitResSummary`.

Key `Analysis` methods for model selection:

- `computeHistLag` / `computeHistLagForAll` — sweep spike-history window
  configurations and select optimal lag via AIC/BIC/KS.
- `computeGrangerCausalityMatrix` — ensemble Granger causality analysis.
- `compHistEnsCoeff` / `compHistEnsCoeffForAll` — ensemble history coefficients.
- `computeNeighbors` — identify functionally connected neighbors.

Key `FitResSummary` plotting methods:

- `plotIC`, `plotAIC`, `plotBIC`, `plotlogLL` — information criterion plots.
- `plotSummary`, `plotResidualSummary` — aggregate diagnostics.
- `boxPlot`, `binCoeffs` — coefficient distribution analysis.

Representative notebooks are indexed in [Examples](Examples.md), especially
`AnalysisExamples`, `FitResultExamples`, and `FitResSummaryExamples`.

## Simulation Workflow

The simulation workflow remains centered on conditional intensity functions
and thinning-based point-process simulation (Section 2.2):

- `CIF` — conditional intensity function primitives, including `simulateCIF`
- `PPThinning` — point-process thinning simulation
- `PPSimExample` — complete simulation workflow

These are covered by the corresponding notebooks listed in
[Examples](Examples.md).

## State-Space GLM (SSGLM) Workflow

The state-space GLM estimation (Section 2.4) is now fully implemented:

- `DecodingAlgorithms.PPSS_EMFB` — full EM algorithm with forward-backward
  Kalman smoothing. Estimates across-trial coefficient dynamics with
  confidence intervals.
- `DecodingAlgorithms.PPSS_EStep` — E-step: forward Kalman filter + backward
  RTS smoother + cross-covariance computation.
- `DecodingAlgorithms.PPSS_MStep` — M-step: Newton-Raphson update for
  observation model parameters.
- `DecodingAlgorithms.PPSS_EM` — single-neuron EM driver.
- `nstColl.ssglm` / `nstColl.ssglmFB` — collection-level convenience wrappers
  that run SSGLM across all neurons in a spike train collection.

This workflow is demonstrated in Example 03 (PSTH and SSGLM dynamics).

## Decoding Workflow

The adaptive filtering and decoding portions of the paper (Sections 2.5-2.6)
map to:

**Point-process adaptive filters (Section 2.5)**:

- `DecodingAlgorithms.PPDecodeFilterLinear` — linear-CIF point-process
  adaptive filter for continuous stimulus decoding.
- `DecodingAlgorithms.PPDecodeFilter` — general CIF version using symbolic
  gradients/Jacobians.
- `DecodingAlgorithms.ComputeStimulusCIs` — stimulus confidence intervals.
- `DecodingAlgorithms.PP_fixedIntervalSmoother` — fixed-interval smoother
  for off-line smoothing of decode estimates.

**Hybrid filter (Section 2.6)**:

- `DecodingAlgorithms.PPHybridFilterLinear` — joint discrete/continuous state
  estimation combining point-process observations with a hidden Markov model
  over discrete states and Kalman filtering over continuous kinematics.

**Kalman and UKF filters**:

- `DecodingAlgorithms.kalman_filter` — standard linear Kalman filter.
- `DecodingAlgorithms.ukf` — unscented Kalman filter for nonlinear state
  estimation.

Related notebooks:

- `DecodingExample`, `DecodingExampleWithHist`, `StimulusDecode2D`,
  `HybridFilterExample`

## Signal Processing Methods

`SignalObj` now includes a full suite of signal processing methods mirroring
the MATLAB toolbox:

**Spectral analysis**:

- `MTMspectrum` — multi-taper spectral estimation using DPSS tapers.
- `spectrogram` — time-frequency decomposition.
- `periodogram` — standard periodogram PSD estimate.

**Correlation and cross-covariance**:

- `xcorr` — cross-correlation (or auto-correlation).
- `xcov` — cross-covariance with lag support.
- `autocorrelation`, `crosscorrelation` — normalized correlation functions.

**Time manipulation**:

- `shift` / `shiftMe` — shift signal in time (copy vs in-place).
- `alignTime` — re-reference time axis to a marker.
- `power`, `sqrt` — element-wise transforms.

**Peak-finding**:

- `findPeaks`, `findMaxima`, `findMinima` — local extrema via
  `scipy.signal.find_peaks`.
- `findGlobalPeak` — global maximum or minimum per channel.

## Example-to-Paper Section Mapping

The paper's representative workflows align to the following Python surfaces:

| Paper section | Example | Python surface |
|---|---|---|
| 2.3.1 — mEPSC Poisson | Example 01 | `examples/paper/example01_mepsc_poisson.py` |
| 2.3.2 — Whisker stimulus | Example 02 | `examples/paper/example02_whisker_stimulus_thalamus.py` |
| 2.3.3 — PSTH | Example 03, Part A | `examples/paper/example03_psth_and_ssglm.py` |
| 2.4 — SSGLM | Example 03, Part B | `examples/paper/example03_psth_and_ssglm.py` |
| 2.3.4 — Place cells | Example 04 | `examples/paper/example04_place_cells_continuous_stimulus.py` |
| 2.5 — PPAF decoding | Example 05, Parts A-B | `examples/paper/example05_decoding_ppaf_pphf.py` |
| 2.6 — Hybrid filter | Example 05, Part C | `examples/paper/example05_decoding_ppaf_pphf.py` |

See also:

- `nSTATPaperExamples` notebook and the canonical gallery in [Paper Examples](paper_examples.md)
- All five paper examples are now **self-contained scripts** with full analysis
  workflows, matching their MATLAB counterparts.
