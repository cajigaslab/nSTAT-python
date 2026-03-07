# nSTAT Python Parity Report

Generated from `parity/manifest.yml` and `parity/class_fidelity.yml`.

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
| `high_fidelity` | 12 |
| `partial` | 6 |
| `shim_only` | 0 |
| `missing` | 0 |
| `not_applicable` | 1 |

## Coverage Notes

- Public API: no missing MATLAB public APIs remain; only the MATLAB help-browser utility is explicitly non-applicable.
- Help/notebook parity: all inventoried MATLAB help workflows are mapped to Python notebooks or equivalents.
- Paper examples and docs gallery: all canonical paper examples and committed gallery directories are mapped.
- Class fidelity: mapping parity is ahead of semantic parity; the audit still reports partial fidelity for several MATLAB-facing classes and workflows.

## Remaining Mapping Deltas

No partial or missing items remain in the mapping inventory.

## Remaining Class-Fidelity Deltas

- `SignalObj` -> `nstat.SignalObj` [partial]: Port arithmetic, filtering, plotting, and structure round-trip methods from MATLAB.
- `Covariate` -> `nstat.Covariate` [partial]: Port arithmetic overloads and CI plotting semantics from MATLAB.
- `nspikeTrain` -> `nstat.nspikeTrain` [partial]: Port burst/statistics helpers and plotting routines.
- `DecodingAlgorithms` -> `nstat.DecodingAlgorithms` [partial]: Port canonical decoding workflows and validate them against MATLAB-derived outputs.
- `ConfidenceInterval` -> `nstat.ConfidenceInterval` [partial]: Port MATLAB plotting and serialization semantics.
- `nSTAT_Install` -> `nstat.nSTAT_Install` [partial]: Keep documenting the no-op compatibility behavior and test installer status outputs.

## Justified Non-Applicable Items

- `public_api`: `nstatOpenHelpPage`. MATLAB help-browser integration does not have a direct Python equivalent.
- `installer_setup`: `CleanUserPathPrefs option`. Accepted as a compatibility no-op because Python does not use MATLAB-style saved user path preferences.
- `installer_setup`: `MATLAB runtime path pruning`. Python packaging/import resolution replaces MATLAB path management.
- `installer_setup`: `MATLAB toolbox cache refresh and savepath`. There is no Python equivalent to MATLAB toolbox cache refresh or savepath persistence.
- `class_fidelity`: `nstatOpenHelpPage`. Python uses Sphinx docs pages instead of the MATLAB help browser.
