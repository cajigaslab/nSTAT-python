# nSTAT-to-nSTAT-python Parity Specification

This document defines how `nSTAT-python` is measured against MATLAB `nSTAT`.

## Gold Standard Baseline
- MATLAB reference repository: `/Users/iahncajigas/Library/CloudStorage/Dropbox/Research/Matlab/nSTAT_currentRelease_Local`
- Baseline lock file: `baseline/baseline_lock.yml`
- Frozen MATLAB example-data snapshot: `parity/matlab_gold_snapshot_20260302.yml`
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
  - MATLAB documentation-only setup/reference examples that have no executable
    computational workflow parity target in Python:
    - `DocumentationSetup2025b`
    - `FitResSummaryExamples`
    - `FitResultExamples`
    - `FitResultReference`

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
   - `parity/method_probe_report.json`
   - `parity/function_example_alignment_report.json`

## Severity Model
- `high`: missing class implementation route, missing notebook/help artifact, missing mapped class help page.
- `medium`: missing mapped method/alias, metadata/TOC mismatch.
- `low`: informational or optional parity signal.

CI enforces `--fail-on high` for parity discovery so missing critical artifacts block merges.

## Source Files
- Method mapping: `parity/method_mapping.yaml`
- Example mapping: `parity/example_mapping.yaml`
- Discovery scripts: `tools/parity/`

## Current Status (2026-03-02)
- Baseline lock refreshed:
  - MATLAB commit: `1b5237b3176f6fc8aa3199d471e4bb7845a3ad5a`
  - Python commit: `8b69adf11dc0ff340e416ce97ffc90eebc011c41`
- Latest structural parity snapshot (`parity/parity_gap_report.json`):
  - `summary.high = 0`
  - `summary.medium = 0`
  - `summary.low = 0`
- Latest functional equivalence audit (`parity/function_example_alignment_report.json`):
  - Method-level audit:
    - `total_methods = 501`
    - `contract_verified_methods = 480`
    - `contract_explicit_verified_methods = 277`
    - `probe_verified_methods = 203`
    - `unverified_behavior_methods = 0`
    - `missing_symbol_methods = 0`
  - Example-level audit:
    - `total_topics = 30`
    - `pending_manual_review_topics = 0`
    - `missing_artifact_topics = 0`
    - `missing_executable_topics = 0`
    - `matlab_doc_only_topics = 4`
    - `validated_topics = 26`
- Updated visual validation report:
  - `output/pdf/nstat_python_validation_report_20260302_145510.pdf` (all notebooks, gate mode)

## Acceptance Checklist
- [x] Class and example inventory artifacts regenerate successfully.
- [x] High-severity parity issues remain at zero.
- [x] Full notebook suite is passing on the validated commit.
- [x] Visual validation PDF has been regenerated after parity changes.
- [x] Structural method-mapping gaps are closed (`parity/parity_gap_report.json`).
- [ ] Functional parity contracts cover all mapped methods (`parity/function_example_alignment_report.json` currently 480/501; 21 methods explicitly excluded by policy).
- [x] Example workflows complete line-by-line manual review and output-lock verification for in-scope topics (0 pending manual review).
- [x] Method-closure sprint backlog generated (`parity/method_closure_sprint.md`).

## Notes
- This repository is a clean-room implementation. MATLAB code is a behavioral reference only.
- Runtime MATLAB dependency is prohibited for normal package use.
