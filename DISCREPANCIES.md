# nSTAT-python Discrepancy Log

This log tracks MATLAB-vs-Python parity issues with minimal repro details.

| ID | Scope | Symptom | Minimal Repro | Suspected Cause | Status | Fix / PR |
|---|---|---|---|---|---|---|
| DSP-001 | `ExplicitStimulusWhiskerData` notebook | Strict line-port remained partial and notebook used synthetic stimulus instead of MATLAB gold fixture arrays | `python tools/parity/sync_parity_artifacts.py --matlab-root <nSTAT>` then inspect `parity/function_example_alignment_report.json` topic row | Notebook template had extra synthetic workflow lines and lacked fixture-backed assertion | Resolved | `codex/robust-parity-sprint-20260303` |
| DSP-002 | `HybridFilterExample` notebook | Strict line-port partial | same as above | Python notebook contained extra simulation scaffolding and lacked MATLAB-fixture numeric assertions | Resolved | `codex/robust-parity-sprint-20260303` |
| DSP-003 | `ValidationDataSet` notebook | Strict line-port partial | same as above | Python workflow was synthetic-only and lacked MATLAB-gold fixture parity assertions | Resolved | `codex/robust-parity-sprint-20260303` |
| DSP-004 | `PPSimExample` notebook | Strict line-port partial | same as above | Python execution cell had synthetic scaffolding and no direct MATLAB fixture comparison | Resolved | `codex/robust-parity-sprint-20260303` |
| DSP-005 | `StimulusDecode2D` notebook | Strict line-port partial | same as above | Python workflow lacked MATLAB-gold 2D decode fixture metrics | Resolved | `codex/robust-parity-sprint-20260303` |

## Rules
- Every parity bug fix must include a regression test that would fail before the fix.
- Close an item only when:
  - parity test(s) pass locally and in CI
  - corresponding row in `parity/function_example_alignment_report.json` is updated
  - PR/commit link is recorded.
