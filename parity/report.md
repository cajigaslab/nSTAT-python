# nSTAT Python Parity Report

Generated from `parity/manifest.yml`, `parity/class_fidelity.yml`, `tools/notebooks/parity_notes.yml`, and live runtime inspection of the audited Python public surface.

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
| `high_fidelity` | 18 |
| `partial` | 0 |
| `wrapper_only` | 0 |
| `missing` | 0 |
| `not_applicable` | 1 |

## Runtime Symbol Verification

| Status | Count |
|---|---:|
| `verified` | 18 |
| `unverified` | 0 |
| `not_applicable` | 1 |

## Notebook Fidelity Summary

| Status | Count |
|---|---:|
| `exact` | 0 |
| `high_fidelity` | 12 |
| `partial` | 1 |

## Simulink Fidelity Summary

| Strategy | Count |
|---|---:|
| `native_python` | 2 |
| `generated_code_wrapped` | 0 |
| `packaged_runtime` | 0 |
| `matlab_engine_fallback` | 0 |
| `unsupported` | 0 |
| `reference_only` | 10 |

## Coverage Notes

- Public API: no missing MATLAB public APIs remain; only the MATLAB help-browser utility is explicitly non-applicable.
- Help/notebook parity: all inventoried MATLAB help workflows are mapped to Python notebooks or equivalents.
- Notebook fidelity: workflow coverage is complete, but 1 MATLAB-helpfile notebook ports are still marked partial in `tools/notebooks/parity_notes.yml`.
- Notebook fidelity audit: structural section/figure comparisons plus placeholder/tracker-only cell detection are recorded in `parity/notebook_fidelity.yml`.
- Paper examples and docs gallery: all canonical paper examples and committed gallery directories are mapped.
- Class fidelity: the class audit reports no partial, wrapper-only, or missing items.
- Runtime symbol verification: every audited MATLAB-facing Python symbol marked present in `parity/class_fidelity.yml` resolves on the live public surface.
- Simulink fidelity: native Python coverage exists for the required published workflows, and 10 inventoried MATLAB assets remain reference-only.

## Remaining Mapping Deltas

No partial or missing items remain in the mapping inventory.

## Remaining Notebook-Fidelity Deltas

- `StimulusDecode2D` -> `notebooks/StimulusDecode2D.ipynb` [partial]: The notebook reproduces the MATLAB section order, figure inventory, simulated receptive fields, and decoded-trajectory presentation, but the current Python decoder still uses regression-based state recovery instead of MATLAB's symbolic-CIF nonlinear filter.

## Remaining Class-Fidelity Deltas

No partial, wrapper-only, or missing class-fidelity items remain.

## Runtime Symbol Drift

No audit/runtime symbol mismatches were detected.

## Simulink Fidelity Deltas

- `PointProcessSimulationCont` -> `PointProcessSimulationCont.slx` [reference_only/reference_only]: Keep as reference while the Python port uses the native discrete simulation path.
- `PointProcessSimulationLegacy2010b` -> `PointProcessSimulation.mdl.r2010b` [reference_only/reference_only]: Treat as a compatibility/reference asset because the native Python port targets the current `PointProcessSimulation.slx` behavior rather than every historic MATLAB model format.
- `PointProcessSimulationLegacy2011a` -> `PointProcessSimulation.mdl.r2011a` [reference_only/reference_only]: Treat as a compatibility/reference asset alongside the current `.slx` model.
- `PointProcessSimulationLegacy2011b` -> `PointProcessSimulation.mdl.r2011b` [reference_only/reference_only]: Treat as a compatibility/reference asset alongside the current `.slx` model.
- `PointProcessSimulationLegacy2013a` -> `PointProcessSimulation.mdl.r2013a` [reference_only/reference_only]: Treat as a compatibility/reference asset alongside the current `.slx` model.
- `PointProcessSimulationLegacySLX2013a` -> `PointProcessSimulation.slx.r2013a` [reference_only/reference_only]: Treat as a versioned MATLAB reference asset rather than a distinct Python execution target.
- `PointProcessSimulationThinningLegacy2011a` -> `PointProcessSimulationThinning.mdl.r2011a` [reference_only/reference_only]: Keep as a MATLAB reference asset while the Python port validates thinning behavior through `CIF.simulateCIFByThinning` and MATLAB Engine comparisons.
- `PointProcessSimulationCache` -> `PointProcessSimulation.slxc` [reference_only/reference_only]: Treat as a MATLAB build artifact, not as a Python execution target.
- `HelpPointProcessSimulationCache` -> `helpfiles/PointProcessSimulation.slxc` [reference_only/reference_only]: Treat as a published-help artifact only.
- `SimulatedNetwork2Cache` -> `helpfiles/SimulatedNetwork2.slxc` [reference_only/reference_only]: Treat as a MATLAB build artifact, not a Python target.

## Justified Non-Applicable Items

- `public_api`: `nstatOpenHelpPage`. MATLAB help-browser integration does not have a direct Python equivalent.
- `installer_setup`: `CleanUserPathPrefs option`. Accepted as a compatibility no-op because Python does not use MATLAB-style saved user path preferences.
- `installer_setup`: `MATLAB runtime path pruning`. Python packaging/import resolution replaces MATLAB path management.
- `installer_setup`: `MATLAB toolbox cache refresh and savepath`. There is no Python equivalent to MATLAB toolbox cache refresh or savepath persistence.
- `class_fidelity`: `nstatOpenHelpPage`. Python uses Sphinx docs pages instead of the MATLAB help browser.
