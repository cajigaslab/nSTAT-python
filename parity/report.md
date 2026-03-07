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
| `high_fidelity` | 18 |
| `partial` | 0 |
| `shim_only` | 0 |
| `missing` | 0 |
| `not_applicable` | 1 |

## Notebook Fidelity Summary

| Status | Count |
|---|---:|
| `exact` | 0 |
| `high_fidelity` | 8 |
| `partial` | 3 |

## Coverage Notes

- Public API: no missing MATLAB public APIs remain; only the MATLAB help-browser utility is explicitly non-applicable.
- Help/notebook parity: all inventoried MATLAB help workflows are mapped to Python notebooks or equivalents.
- Notebook fidelity: workflow coverage is complete, but 3 MATLAB-helpfile notebook ports are still marked partial in `tools/notebooks/parity_notes.yml`.
- Notebook fidelity audit: structural section/figure comparisons are recorded in `parity/notebook_fidelity.yml`.
- Paper examples and docs gallery: all canonical paper examples and committed gallery directories are mapped.
- Class fidelity: the class audit reports no partial, shim-only, or missing items.

## Remaining Mapping Deltas

No partial or missing items remain in the mapping inventory.

## Remaining Notebook-Fidelity Deltas

- `HippocampalPlaceCellExample` -> `notebooks/HippocampalPlaceCellExample.ipynb` [partial]: Core place-cell workflow is ported, but MATLAB figure sequencing and summary outputs are not yet exact.
- `HybridFilterExample` -> `notebooks/HybridFilterExample.ipynb` [partial]: Hybrid filtering workflow executes, but MATLAB-specific output details and downstream validation remain incomplete.
- `StimulusDecode2D` -> `notebooks/StimulusDecode2D.ipynb` [partial]: The 2D stimulus decoding workflow runs, but MATLAB-equivalent outputs and tolerance-backed parity checks still need expansion.

## Remaining Class-Fidelity Deltas

No partial, shim-only, or missing class-fidelity items remain.

## Justified Non-Applicable Items

- `public_api`: `nstatOpenHelpPage`. MATLAB help-browser integration does not have a direct Python equivalent.
- `installer_setup`: `CleanUserPathPrefs option`. Accepted as a compatibility no-op because Python does not use MATLAB-style saved user path preferences.
- `installer_setup`: `MATLAB runtime path pruning`. Python packaging/import resolution replaces MATLAB path management.
- `installer_setup`: `MATLAB toolbox cache refresh and savepath`. There is no Python equivalent to MATLAB toolbox cache refresh or savepath persistence.
- `class_fidelity`: `nstatOpenHelpPage`. Python uses Sphinx docs pages instead of the MATLAB help browser.
