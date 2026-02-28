# HippocampalPlaceCellExample

2D place-field simulation and trajectory decoding workflow.

## Notebook
- [HippocampalPlaceCellExample.ipynb](https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/HippocampalPlaceCellExample.ipynb)
- Execution group: `full`

## What this example demonstrates
1. Simulate a smooth 2D trajectory in a unit-square arena.
2. Generate Gaussian place fields for a population of units.
3. Simulate Poisson spike counts from place-field rates.
4. Decode positions using weighted-center and Poisson maximum-likelihood
   state estimators.
5. Compare decoder error with normalized RMSE metrics.

## Expected validation outputs
- ML decoder RMSE no worse than weighted-center decoder.
- Normalized ML decoder RMSE `< 0.30`.
- Example place-field map and decoded trajectory plots.

## Linked references
- [Examples index](../examples_index.md)
- [Class definitions](../class_definitions.md)
- [DecodingAlgorithms class help](../classes/DecodingAlgorithms.md)
- [Paper overview](../paper_overview.md)
- DOI: `10.1016/j.jneumeth.2012.08.009`
- PMID: `22981419`
