# Analysis

Python implementation: `nstat.analysis.Analysis`

## Purpose
This class preserves MATLAB-facing structure while using a Python-native,
fully independent implementation in `nSTAT-python`.

## Core methods
- `fit_glm(X, y, fit_type='poisson', dt=1.0, l2_penalty=0.0)`:
  analytical-gradient GLM fit for Poisson or binomial observations.
- `fit_trial(trial, config, unit_index=0)`:
  convenience wrapper that bins observations from `Trial` and calls `fit_glm`.

## Estimation details
- Poisson fit minimizes negative log-likelihood for counts per bin:
  \[
  \sum_t \lambda_t\Delta t - y_t\log(\lambda_t\Delta t) + \log(y_t!)
  \]
- Binomial fit minimizes Bernoulli cross-entropy.
- Optimization uses `L-BFGS-B` with closed-form gradients.

## References
- [API reference](../../api.md)
- [Paper overview](../paper_overview.md)
- DOI: `10.1016/j.jneumeth.2012.08.009`
- PMID: `22981419`

## Related learning resources
- [AnalysisExamples notebook](https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/AnalysisExamples.ipynb)
- [AnalysisExamples help page](../examples/AnalysisExamples.md)
- [nSTATPaperExamples notebook](https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/nSTATPaperExamples.ipynb)
- [nSTATPaperExamples help page](../examples/nSTATPaperExamples.md)
