# FitResult

Python implementation: `nstat.fit.FitResult`

## Purpose
This class preserves MATLAB-facing structure while using a Python-native,
fully independent implementation in `nSTAT-python`.

## Core methods
- `as_cif_model()`: convert fitted parameters to `nstat.cif.CIFModel`.
- `predict(X)`: compute model-predicted mean response for design matrix `X`.
- `aic()` / `bic()`: information-criterion model comparison statistics.

## References
- [API reference](../../api.md)
- [Paper overview](../paper_overview.md)
- DOI: `10.1016/j.jneumeth.2012.08.009`
- PMID: `22981419`

## Related learning resources
- [FitResultExamples notebook](https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/FitResultExamples.ipynb)
- [FitResultExamples help page](../examples/FitResultExamples.md)
- [nSTATPaperExamples notebook](https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/nSTATPaperExamples.ipynb)
- [nSTATPaperExamples help page](../examples/nSTATPaperExamples.md)
