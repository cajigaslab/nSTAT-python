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

The paper's core GLM workflow maps to Python as:

1. Build trial data with `Trial`, `CovColl`, and `nstColl`.
2. Define model configurations with `TrialConfig` and `ConfigColl`.
3. Fit and evaluate analyses with `Analysis` and `FitResult`.
4. Summarize across fits with `FitResSummary`.

Representative notebooks are indexed in [Examples](Examples.md), especially
`AnalysisExamples`, `FitResultExamples`, and `FitResSummaryExamples`.

## Simulation Workflow

The simulation workflow remains centered on conditional intensity functions
and thinning-based point-process simulation:

- `CIF`
- `PPThinning`
- `PPSimExample`

These are covered by the corresponding notebooks listed in
[Examples](Examples.md).

## Decoding Workflow

The adaptive filtering and decoding portions of the paper map to:

- `DecodingAlgorithms`
- `DecodingExample`
- `DecodingExampleWithHist`
- `StimulusDecode2D`
- `HybridFilterExample`

## Example-to-Paper Section Mapping

The paper's representative workflows align to the following Python surfaces:

- `mEPSCAnalysis` and `PSTHEstimation`: event-process and PSTH analysis
- `ExplicitStimulusWhiskerData` and `HippocampalPlaceCellExample`:
  stimulus-response and receptive-field modeling
- `DecodingExample`, `DecodingExampleWithHist`, and `StimulusDecode2D`:
  decoding and state estimation
- `nSTATPaperExamples` and the canonical gallery in [Paper Examples](paper_examples.md):
  consolidated reproduction workflow for the toolbox paper
