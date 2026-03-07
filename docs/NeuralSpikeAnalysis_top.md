# Neural Spike Train Analysis Toolbox (nSTAT)

`nSTAT-python` is the standalone Python port of the neural spike-train analysis
toolbox. It preserves the MATLAB toolbox's public API naming, paper-example
structure, and help/example coverage wherever a Python equivalent is
reasonable.

## Documentation Navigation

- [Paper-Aligned Toolbox Map](PaperOverview.md)
- [Class Definitions](ClassDefinitions.md)
- [Example Index](Examples.md)
- [nSTAT Paper Examples](paper_examples.md)
- [Documentation Setup](DocumentationSetup.md)
- [API Reference](api.rst)

## Purpose

The toolbox consolidates point-process and GLM-based neural data analysis into
a coherent Python package with:

- MATLAB-compatible public entry points such as `Analysis`, `TrialConfig`,
  `FitResult`, `DecodingAlgorithms`, `nSTAT_Install`, and `getPaperDataDirs`
- Canonical paper examples exported as `examples/paper/example01` through
  `example05`
- Notebook-backed help workflows mirroring the MATLAB helpfiles
- A generated figure gallery under `docs/figures/`

## Citation

Cajigas I, Malik WQ, Brown EN. *nSTAT: Open-source neural spike train analysis
toolbox for Matlab*. Journal of Neuroscience Methods 211:245-264 (2012).
DOI: [10.1016/j.jneumeth.2012.08.009](https://doi.org/10.1016/j.jneumeth.2012.08.009)
