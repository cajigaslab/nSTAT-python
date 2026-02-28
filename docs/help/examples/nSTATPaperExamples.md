# nSTATPaperExamples

This page documents the end-to-end clean-room workflow implemented in
`notebooks/nSTATPaperExamples.ipynb`.

## Notebook
- [nSTATPaperExamples.ipynb](https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/nSTATPaperExamples.ipynb)
- Execution group: `smoke`

## What the notebook covers
1. Simulate a time-varying stimulus and generate spikes from a Poisson CIF.
2. Fit a Poisson GLM (`nstat.analysis.Analysis.fit_trial`) to recover parameters.
3. Convert the fit to a model object (`nstat.fit.FitResult.as_cif_model`) and
   compare estimated vs. true rates.
4. Decode a latent discrete state sequence from population spike counts using
   point-process Bayes filtering (`nstat.decoding.DecodingAlgorithms.decode_state_posterior`).

## Mathematical references used in implementation
- Poisson conditional-intensity log-likelihood:
  \[
  \log p(y_t \mid \lambda_t) = y_t \log(\lambda_t \Delta t) - \lambda_t \Delta t - \log(y_t!)
  \]
- Binomial/logistic observation model:
  \[
  p(y_t=1\mid x_t)=\sigma(\beta_0 + x_t^\top\beta)
  \]
- Discrete-state posterior filtering:
  \[
  p(z_t \mid y_{1:t}) \propto p(y_t \mid z_t)\sum_{z_{t-1}} p(z_t \mid z_{t-1})p(z_{t-1}\mid y_{1:t-1})
  \]

These correspond to the formulations discussed in:
- Cajigas I, Malik WQ, Brown EN. *J Neurosci Methods* (2012).
- DOI: `10.1016/j.jneumeth.2012.08.009`
- PMID: `22981419`

## Expected outputs
- Recovered Poisson coefficient error below `tests/parity/tolerances.yml`:
  `poisson_glm.coefficient_abs_tol`
- Decoding normalized RMSE below:
  `decoding.normalized_rmse_tol`
- Reproducible results using fixed random seeds.

## Linked references
- [Examples index](../examples_index.md)
- [Class definitions](../class_definitions.md)
- [CIF class help](../classes/CIF.md)
- [Analysis class help](../classes/Analysis.md)
- [DecodingAlgorithms class help](../classes/DecodingAlgorithms.md)
- [Paper overview](../paper_overview.md)
