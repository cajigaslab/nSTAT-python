# Example Index

This page mirrors the MATLAB `Examples` help index and points to the canonical
Python notebook or script equivalent for each workflow.

Paper cross-reference:

- [Paper-Aligned Toolbox Map](PaperOverview.md)

## Class and Object Workflows

- `SignalObjExamples`: `notebooks/SignalObjExamples.ipynb`
- `CovariateExamples`: `notebooks/CovariateExamples.ipynb`
- `CovCollExamples`: `notebooks/CovCollExamples.ipynb`
- `nSpikeTrainExamples`: `notebooks/nSpikeTrainExamples.ipynb`
- `nstCollExamples`: `notebooks/nstCollExamples.ipynb`
- `EventsExamples`: `notebooks/EventsExamples.ipynb`
- `HistoryExamples`: `notebooks/HistoryExamples.ipynb`
- `TrialExamples`: `notebooks/TrialExamples.ipynb`
- `TrialConfigExamples`: `notebooks/TrialConfigExamples.ipynb`
- `ConfigCollExamples`: `notebooks/ConfigCollExamples.ipynb`
- `ConfidenceIntervalOverview`: `notebooks/ConfidenceIntervalOverview.ipynb`

## Fitting, Assessment, and Analysis

- `AnalysisExamples`: `notebooks/AnalysisExamples.ipynb`
- `AnalysisExamples2`: `notebooks/AnalysisExamples2.ipynb`
- `FitResultExamples`: `notebooks/FitResultExamples.ipynb`
- `FitResultReference`: `notebooks/FitResultReference.ipynb`
- `FitResSummaryExamples`: `notebooks/FitResSummaryExamples.ipynb`

## Simulation and Example-Data Analyses

- `PPThinning`: `notebooks/PPThinning.ipynb`
- `PPSimExample`: `notebooks/PPSimExample.ipynb`
- `PSTHEstimation`: `notebooks/PSTHEstimation.ipynb`
- `ValidationDataSet`: `notebooks/ValidationDataSet.ipynb`
- `mEPSCAnalysis`: `notebooks/mEPSCAnalysis.ipynb`
- `ExplicitStimulusWhiskerData`: `notebooks/ExplicitStimulusWhiskerData.ipynb`
- `HippocampalPlaceCellExample`: `notebooks/HippocampalPlaceCellExample.ipynb`

## Decoding and Network Workflows

- `DecodingExample`: `notebooks/DecodingExample.ipynb`
- `DecodingExampleWithHist`: `notebooks/DecodingExampleWithHist.ipynb`
- `StimulusDecode2D`: `notebooks/StimulusDecode2D.ipynb`
- `HybridFilterExample`: `notebooks/HybridFilterExample.ipynb`
- `NetworkTutorial`: `notebooks/NetworkTutorial.ipynb`

## Consolidated Paper Workflow

All five paper examples are self-contained scripts mirroring their MATLAB
counterparts:

| Example | Script | What it demonstrates |
|---|---|---|
| 01 — mEPSC Poisson | [example01_mepsc_poisson.py](../examples/paper/example01_mepsc_poisson.py) | Constant vs piecewise Poisson under Mg2+ washout |
| 02 — Whisker Stimulus | [example02_whisker_stimulus_thalamus.py](../examples/paper/example02_whisker_stimulus_thalamus.py) | Explicit-stimulus GLM with lag and history selection |
| 03 — PSTH and SSGLM | [example03_psth_and_ssglm.py](../examples/paper/example03_psth_and_ssglm.py) | PSTH comparison and state-space GLM dynamics |
| 04 — Place Cells | [example04_place_cells_continuous_stimulus.py](../examples/paper/example04_place_cells_continuous_stimulus.py) | Gaussian vs Zernike receptive-field models |
| 05 — PPAF and PPHF | [example05_decoding_ppaf_pphf.py](../examples/paper/example05_decoding_ppaf_pphf.py) | Adaptive and hybrid point-process decoding |

- Generated gallery and figure index: [Paper Examples](paper_examples.md)
- Master notebook: `notebooks/nSTATPaperExamples.ipynb`

## Supplementary (README) Examples

These smaller demos serve as quick install and plotting checks:

| Example | Run command |
|---|---|
| Multitaper spectrum + spectrogram | `python examples/readme_examples/example1_multitaper_and_spectrogram.py` |
| Simulated CIF spike train | `python examples/readme_examples/example2_simulate_cif_spiketrain_10s.py` |
| Spike-train raster | `python examples/readme_examples/example3_nstcoll_raster_from_example2.py` |
