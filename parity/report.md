# nSTAT Python Parity Report

Generated from `parity/manifest.yml`, `parity/class_fidelity.yml`, and `tools/notebooks/parity_notes.yml`.

- MATLAB reference: https://github.com/cajigaslab/nSTAT
- Python target: https://github.com/cajigaslab/nSTAT-python
- Inventory version: 1
- Generated on: 2026-03-07

## Summary

| Section | Mapped | Partial | Missing | Not Applicable |
|---|---:|---:|---:|---:|
| `public api` | 18 | 0 | 0 | 1 |
| `help workflows` | 34 | 0 | 0 | 0 |
| `paper examples` | 8 | 0 | 0 | 0 |
| `docs gallery` | 8 | 0 | 0 | 0 |
| `installer setup` | 4 | 0 | 0 | 3 |
| `repo structure` | 1 | 0 | 0 | 0 |

## Class Fidelity Summary

| Status | Count |
|---|---:|
| `exact` | 0 |
| `high_fidelity` | 13 |
| `partial` | 5 |
| `wrapper_only` | 0 |
| `missing` | 0 |
| `not_applicable` | 1 |

## Notebook Fidelity Summary

| Status | Count |
|---|---:|
| `exact` | 0 |
| `high_fidelity` | 5 |
| `partial` | 6 |

## Simulink Fidelity Summary

| Strategy | Count |
|---|---:|
| `native_python` | 1 |
| `generated_code_wrapped` | 0 |
| `packaged_runtime` | 0 |
| `matlab_engine_fallback` | 1 |
| `unsupported` | 0 |
| `reference_only` | 4 |

## Coverage Notes

- Public API: no missing MATLAB public APIs remain; only the MATLAB help-browser utility is explicitly non-applicable.
- Help/notebook parity: all inventoried MATLAB help workflows are mapped to Python notebooks or equivalents.
- Notebook fidelity: workflow coverage is complete, but 6 MATLAB-helpfile notebook ports are still marked partial in `tools/notebooks/parity_notes.yml`.
- Notebook fidelity audit: structural section/figure comparisons plus placeholder/tracker-only cell detection are recorded in `parity/notebook_fidelity.yml`.
- Paper examples and docs gallery: all canonical paper examples and committed gallery directories are mapped.
- Class fidelity: mapping parity is ahead of semantic parity; the audit still reports partial fidelity for several MATLAB-facing classes and workflows.
- Simulink fidelity: 2 Simulink-backed assets still rely on partial, fallback, or unsupported Python execution paths.

## Remaining Mapping Deltas

No partial or missing items remain in the mapping inventory.

## Remaining Notebook-Fidelity Deltas

- `nSTATPaperExamples` -> `notebooks/nSTATPaperExamples.ipynb` [partial]: Python uses standalone figshare-backed data access and generated gallery assets rather than MATLAB path-based setup, and several sections still rely on placeholder or tracker-only cells instead of full MATLAB-equivalent computations.
- `TrialExamples` -> `notebooks/TrialExamples.ipynb` [partial]: The notebook still contains placeholder and tracker-only cells, so the current port demonstrates the Trial workflow structure but not yet the full MATLAB helpfile computation and figure path.
- `AnalysisExamples` -> `notebooks/AnalysisExamples.ipynb` [partial]: Advanced MATLAB algorithm-selection branches and report plots remain lighter in Python, and the notebook still contains tracker-only visualization sections rather than a fully executable MATLAB-equivalent workflow.
- `HybridFilterExample` -> `notebooks/HybridFilterExample.ipynb` [partial]: The notebook reproduces the hybrid-filter workflow structure and figure contract, but it still contains placeholder cells and does not yet provide a fully executable MATLAB-equivalent report path.
- `PPSimExample` -> `notebooks/PPSimExample.ipynb` [partial]: The core point-process simulation path exists in Python, but this notebook still contains placeholder and tracker-only sections instead of a full MATLAB-equivalent simulation and report workflow.
- `ValidationDataSet` -> `notebooks/ValidationDataSet.ipynb` [partial]: The notebook follows the MATLAB validation workflow structure, but it still contains placeholder sections and uses shorter deterministic simulations rather than a full MATLAB-equivalent execution path.

## Remaining Class-Fidelity Deltas

- `Analysis` -> `nstat.Analysis` [partial]: Add dataset-backed numerical parity fixtures for canonical analysis workflows.
- `FitResult` -> `nstat.FitResult` [partial]: Add MATLAB-derived golden fixtures for coefficient metadata and validation/report payloads.
- `FitResSummary` -> `nstat.FitResSummary` [partial]: Add golden fixtures for multi-neuron summary aggregation and remaining report outputs.
- `CIF` -> `nstat.CIF` [partial]: Add MATLAB-derived fixtures for CIF evaluation and thinning outputs.
- `DecodingAlgorithms` -> `nstat.DecodingAlgorithms` [partial]: Add MATLAB-derived numerical fixtures for DecodingExample, DecodingExampleWithHist, StimulusDecode2D, and HybridFilterExample.

## Simulink Fidelity Deltas

- `PointProcessSimulation` -> `PointProcessSimulation.slx` [native_python/partial]: Native Python simulation through `nstat.cif` and `nstat.simulation`, with MATLAB/Simulink fixture comparison still pending.
- `SimulatedNetwork2` -> `helpfiles/SimulatedNetwork2.mdl` [matlab_engine_fallback/partial]: Prefer a future native Python reimplementation, but document MATLAB Engine fallback first because no faithful Python executable path exists yet.

## Justified Non-Applicable Items

- `public_api`: `nstatOpenHelpPage`. MATLAB help-browser integration does not have a direct Python equivalent.
- `installer_setup`: `CleanUserPathPrefs option`. Accepted as a compatibility no-op because Python does not use MATLAB-style saved user path preferences.
- `installer_setup`: `MATLAB runtime path pruning`. Python packaging/import resolution replaces MATLAB path management.
- `installer_setup`: `MATLAB toolbox cache refresh and savepath`. There is no Python equivalent to MATLAB toolbox cache refresh or savepath persistence.
- `class_fidelity`: `nstatOpenHelpPage`. Python uses Sphinx docs pages instead of the MATLAB help browser.
