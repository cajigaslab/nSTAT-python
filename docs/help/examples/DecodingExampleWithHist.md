# DecodingExampleWithHist

State decoding workflow with explicit history modulation and correction.

## Notebook
- [DecodingExampleWithHist.ipynb](https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/DecodingExampleWithHist.ipynb)
- Execution group: `full`

## What this example demonstrates
1. Simulate a latent discrete-state random walk with state-dependent firing.
2. Add global spike-history gain that distorts raw observations.
3. Decode latent states via
   `nstat.decoding.DecodingAlgorithms.decode_state_posterior`.
4. Apply history correction and confirm improved normalized RMSE.
5. Construct and inspect a history design matrix using
   `nstat.history.HistoryBasis`.

## Expected validation outputs
- `NRMSE(history-corrected) <= NRMSE(raw)`
- `NRMSE(history-corrected) < 0.20`
- Posterior normalization error `< 1e-6`

## Linked references
- [Examples index](../examples_index.md)
- [Class definitions](../class_definitions.md)
- [DecodingAlgorithms class help](../classes/DecodingAlgorithms.md)
- [History class help](../classes/History.md)
- [Paper overview](../paper_overview.md)
- DOI: `10.1016/j.jneumeth.2012.08.009`
- PMID: `22981419`
