# CIF

Python implementation: `nstat.cif.CIFModel`

## Purpose
This class preserves MATLAB-facing structure while using a Python-native,
fully independent implementation in `nSTAT-python`.

## Core methods
- `linear_predictor(X)`: compute \(\eta = \beta_0 + X\beta\).
- `evaluate(X)`: return \(\lambda = \exp(\eta)\) for Poisson or
  \(p = \sigma(\eta)\) for binomial.
- `log_likelihood(y, X, dt=1.0)`: evaluate independent-bin model likelihood.
- `simulate_by_thinning(time, X, rng=None)`: generate spike times on a fixed
  grid using Poisson/binomial observation assumptions.

## Notes
- Poisson simulation is exact for a piecewise-constant intensity on the given
  grid bins.
- The implementation is numerically stabilized for high-magnitude linear
  predictors.

## References
- [API reference](../../api.md)
- [Paper overview](../paper_overview.md)
- DOI: `10.1016/j.jneumeth.2012.08.009`
- PMID: `22981419`

## Related learning resources
- [PPSimExample notebook](https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/PPSimExample.ipynb)
- [PPSimExample help page](../examples/PPSimExample.md)
- [PPThinning notebook](https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/PPThinning.ipynb)
- [PPThinning help page](../examples/PPThinning.md)
