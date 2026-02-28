# PPSimExample

Point-process simulation and recovery workflow for Poisson CIF models.

## Notebook
- [PPSimExample.ipynb](https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/PPSimExample.ipynb)
- Execution group: `full`

## What this example demonstrates
1. Build a sinusoidal covariate and Poisson conditional-intensity model.
2. Simulate multiple spike-train realizations with
   `nstat.cif.CIFModel.simulate_by_thinning`.
3. Fit Poisson GLM parameters with `nstat.analysis.Analysis.fit_glm`.
4. Validate parameter and rate recovery against deterministic thresholds.

## Expected validation outputs
- Coefficient recovery error `< 0.25`
- Relative rate error `< 0.20`
- AIC/BIC reported from the fitted model (`nstat.fit.FitResult`)

## Linked references
- [Examples index](../examples_index.md)
- [Class definitions](../class_definitions.md)
- [CIF class help](../classes/CIF.md)
- [Analysis class help](../classes/Analysis.md)
- [Paper overview](../paper_overview.md)
- DOI: `10.1016/j.jneumeth.2012.08.009`
- PMID: `22981419`
