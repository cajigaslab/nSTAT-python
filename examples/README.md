# nSTAT Python Examples

## Tutorials (Learn-by-reading)

Heavily annotated, self-contained scripts written as lessons — run anywhere
(simulated data, no download). Pair them with the
[Concepts &amp; Background](../docs/concepts/index.md) documentation.

```bash
# Encoding -> GLM fit -> goodness-of-fit, with a correct-vs-wrong-model contrast
python examples/tutorials/encoding_to_goodness_of_fit.py
python examples/tutorials/encoding_to_goodness_of_fit.py --save-fig out.png
```

- **`examples/tutorials/Tutorial_MicroelectrodeToDecoding.ipynb`** — a notebook
  tour from microelectrode recordings (spikes vs. LFP) through spike trains,
  multitaper spectra, GLM fitting, goodness-of-fit, and decoding. Runs on
  simulated data; ships with outputs and figures embedded.

## Paper Examples (Self-Contained)

Five self-contained scripts mirroring the MATLAB paper examples. Each
generates publication-quality figures and supports `--export-figures`.

```bash
python examples/paper/example01_mepsc_poisson.py --export-figures
python examples/paper/example02_whisker_stimulus_thalamus.py --export-figures
python examples/paper/example03_psth_and_ssglm.py --export-figures
python examples/paper/example04_place_cells_continuous_stimulus.py --export-figures
python examples/paper/example05_decoding_ppaf_pphf.py --export-figures
```

| Example | Focus | Paper Section |
|---|---|---|
| 01 | mEPSC Poisson models (constant vs piecewise baseline) | 2.3.1 |
| 02 | Whisker stimulus GLM with lag and history selection | 2.3.2 |
| 03 | PSTH and SSGLM across-trial dynamics | 2.3.3-2.3.4 |
| 04 | Place-cell receptive fields (Gaussian vs Zernike) | 2.3.5 |
| 05 | PPAF and hybrid filter decoding | 2.5-2.6 |

## Basic Examples

```bash
python examples/basic_data_workflow.py
python examples/fit_poisson_glm.py
python examples/simulate_population_psth.py
```

## README Examples (Quick Checks)

```bash
python examples/readme_examples/example1_multitaper_and_spectrogram.py
python examples/readme_examples/example2_simulate_cif_spiketrain_10s.py
python examples/readme_examples/example3_nstcoll_raster_from_example2.py
```

## Jupyter Notebooks

All 29 class-tutorial and data-analysis notebooks are in `notebooks/`.
They mirror the MATLAB helpfile examples one-to-one. See
[docs/Examples.md](../docs/Examples.md) for the full index.
