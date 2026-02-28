# nSTAT-to-nSTAT-python Parity Specification

This document defines how `nSTAT-python` is measured against MATLAB `nSTAT`.

## Gold Standard Baseline
- MATLAB reference repository: `/Users/iahncajigas/Library/CloudStorage/Dropbox/Research/Matlab/nSTAT_currentRelease_Local`
- Baseline lock file: `baseline/baseline_lock.yml`
- MATLAB commit hash is pinned in the lock file and must be updated intentionally.

## Scope
- In scope:
  - Core computational classes and workflows (Signal, spike trains, trial assembly, CIF/GLM fitting, decoding)
  - Help pages and class/topic discoverability
  - Executable notebooks that mirror MATLAB example workflows
  - CI parity discovery and reporting
- Out of scope:
  - MATLAB desktop help-browser internals
  - Simulink-only integrations

## Parity Contract
1. Every in-scope MATLAB class has a Python implementation route through:
   - Primary Python API (`nstat.*`), and/or
   - Compatibility API (`nstat.compat.matlab.*`)
2. Example workflows in `parity/example_mapping.yaml` must have:
   - A notebook in `notebooks/`
   - A help page in `docs/help/examples/`
3. Class help pages must exist for all mapped MATLAB classes.
4. Numerical behavior is validated by tolerance-driven tests under `tests/parity/`.
5. Parity discovery produces machine-readable artifacts:
   - `parity/matlab_api_inventory.json`
   - `parity/python_api_inventory.json`
   - `parity/parity_gap_report.json`

## Severity Model
- `high`: missing class implementation route, missing notebook/help artifact, missing mapped class help page.
- `medium`: missing mapped method/alias, metadata/TOC mismatch.
- `low`: informational or optional parity signal.

CI enforces `--fail-on high` for parity discovery so missing critical artifacts block merges.

## Source Files
- Method mapping: `parity/method_mapping.yaml`
- Example mapping: `parity/example_mapping.yaml`
- Discovery scripts: `tools/parity/`

## Notes
- This repository is a clean-room implementation. MATLAB code is a behavioral reference only.
- Runtime MATLAB dependency is prohibited for normal package use.
