# nSTAT Python Parity Report

Generated from `parity/manifest.yml`.

- MATLAB reference: https://github.com/cajigaslab/nSTAT
- Python target: https://github.com/cajigaslab/nSTAT-python
- Inventory version: 1
- Generated on: 2026-03-06

## Summary

| Section | Mapped | Partial | Missing | Not Applicable |
|---|---:|---:|---:|---:|
| `public api` | 18 | 0 | 0 | 1 |
| `help workflows` | 29 | 0 | 0 | 0 |
| `paper examples` | 5 | 3 | 0 | 0 |
| `docs gallery` | 7 | 1 | 0 | 0 |
| `installer setup` | 3 | 2 | 0 | 2 |
| `repo structure` | 1 | 0 | 0 | 0 |

## Coverage Notes

- Public API: no missing MATLAB public APIs remain; only the MATLAB help-browser utility is explicitly non-applicable.
- Help/notebook parity: all inventoried MATLAB help workflows are mapped to Python notebooks or equivalents.
- Paper examples and docs gallery: canonical structure is present, but dataset-backed outputs and figure files are still partial.

## Remaining Deltas

### `paper_examples`

- `example05_decoding_ppaf_pphf` -> `nstat.paper_examples_full.run_experiment5`: Canonical Python script now exists and bundles experiment5/5b/6, but the MATLAB-matched figure gallery is not yet exported.
- `nSTATPaperExamples section 5b` -> `nstat.paper_examples_full.run_experiment5b`: Implemented as a helper function only.
- `nSTATPaperExamples section 6` -> `nstat.paper_examples_full.run_experiment6`: Implemented as a helper function only.

### `docs_gallery`

- `docs/figures/example05/` -> `docs/figures/example05/`: Gallery directory exists, but the MATLAB-matched PNG set is not yet generated.

### `installer_setup`

- `RebuildDocSearch option` -> `nstat.install.nstat_install(rebuild_doc_search=...)`: The option exists, but no Python docs-search rebuild is implemented.
- `CleanUserPathPrefs option` -> `nstat.install.nstat_install(clean_user_path_prefs=...)`: The option exists, but no Python-side cleanup routine is implemented.

## Justified Non-Applicable Items

- `public_api`: `nstatOpenHelpPage`. MATLAB help-browser integration does not have a direct Python equivalent.
- `installer_setup`: `MATLAB runtime path pruning`. Python packaging/import resolution replaces MATLAB path management.
- `installer_setup`: `MATLAB toolbox cache refresh and savepath`. There is no Python equivalent to MATLAB toolbox cache refresh or savepath persistence.
