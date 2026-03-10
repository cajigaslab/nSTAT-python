# Neural Spike Train Analysis Toolbox (nSTAT)

`nSTAT-python` is the standalone Python port of the neural spike-train analysis
toolbox. It implements a range of models and algorithms for neural spike train
data analysis, with a focus on point-process generalized linear models (GLMs),
model fitting, model-order analysis, and adaptive decoding. In addition to
point-process algorithms, nSTAT also provides tools for Gaussian signals — from
correlation analysis to the Kalman filter — applicable to continuous neural
signals such as LFP, EEG, and ECoG.

The port preserves the MATLAB toolbox's public API naming, paper-example
structure, and help/example coverage wherever a Python equivalent is
reasonable.

## Documentation Navigation

- [Paper-Aligned Toolbox Map](PaperOverview.md) — maps classes and methods to
  the 2012 paper's sections
- [Class Definitions](ClassDefinitions.md) — all classes with grouped method
  listings
- [Example Index](Examples.md) — full help-style index of notebooks and scripts
- [nSTAT Paper Examples](paper_examples.md) — generated gallery with figures
  from all 5 paper examples
- [Documentation Setup](DocumentationSetup.md) — installation, build, and
  troubleshooting
- [Data Installation](data_installation.rst) — example dataset download
- [API Reference](api.rst) — module layout

## Key Capabilities

- **GLM fitting and assessment**: Point-process GLMs with stimulus, history,
  and ensemble covariates. AIC/BIC model selection, KS goodness-of-fit,
  residual analysis.
- **SSGLM (state-space GLM)**: Full EM algorithm (`PPSS_EMFB`) for estimating
  across-trial coefficient dynamics with forward-backward Kalman smoothing.
- **Adaptive decoding**: Point-process adaptive filter (PPAF) for real-time
  stimulus and state decoding from neural spike trains.
- **Hybrid filter**: Joint discrete/continuous state estimation combining
  point-process observations with hidden Markov models.
- **UKF support**: Unscented Kalman filter for nonlinear state estimation.
- **Signal processing**: Multi-taper spectral estimation, spectrograms,
  cross-covariance, peak-finding, and time-domain signal manipulation.
- **Granger causality**: Ensemble Granger causality analysis for network
  connectivity inference.

## Citation

Cajigas I, Malik WQ, Brown EN. *nSTAT: Open-source neural spike train analysis
toolbox for Matlab*. Journal of Neuroscience Methods 211:245-264 (2012).
DOI: [10.1016/j.jneumeth.2012.08.009](https://doi.org/10.1016/j.jneumeth.2012.08.009)
PMID: [22981419](https://pubmed.ncbi.nlm.nih.gov/22981419/)

## Lab Websites

- Neuroscience Statistics Research Laboratory: [neurostat.mit.edu](https://www.neurostat.mit.edu)
- RESToRe Lab: [cajigaslab](https://www.med.upenn.edu/cajigaslab/)
