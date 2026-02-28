# DecodingAlgorithms

Python implementation: `nstat.decoding.DecodingAlgorithms`

## Purpose
This class preserves MATLAB-facing structure while using a Python-native,
fully independent implementation in `nSTAT-python`.

## Core methods
- `compute_spike_rate_cis(spike_matrix, alpha=0.05)`:
  pairwise trial-rate comparisons using two-proportion tests and
  Benjamini-Hochberg FDR control.
- `decode_weighted_center(spike_counts, tuning_curves)`:
  center-of-mass state estimator.
- `decode_state_posterior(spike_counts, tuning_rates, transition=None, prior=None)`:
  discrete-state point-process Bayes filter returning decoded state sequence
  and posterior over states at each time.

## Decoding formulation
The state posterior update is
\[
p(z_t \mid y_{1:t}) \propto p(y_t \mid z_t)
\sum_{z_{t-1}} p(z_t \mid z_{t-1}) p(z_{t-1}\mid y_{1:t-1}),
\]
with Poisson emission likelihood \(p(y_t \mid z_t)\) evaluated in log-space
for numerical stability.

## References
- [API reference](../../api.md)
- [Paper overview](../paper_overview.md)
- DOI: `10.1016/j.jneumeth.2012.08.009`
- PMID: `22981419`

## Related learning resources
- [DecodingExample notebook](https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/DecodingExample.ipynb)
- [DecodingExample help page](../examples/DecodingExample.md)
- [StimulusDecode2D notebook](https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/StimulusDecode2D.ipynb)
- [StimulusDecode2D help page](../examples/StimulusDecode2D.md)
