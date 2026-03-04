# Cycle Validation Checklist (2026-03-04)

Commands used each cycle:
- `pytest -q`
- `python tools/parity/build_numeric_drift_report.py --fixtures-manifest tests/parity/fixtures/matlab_gold/manifest.yml --thresholds parity/numeric_drift_thresholds.yml --report-out parity/numeric_drift_report.json --fail-on-violation`
- `python tools/parity/check_functional_parity_progress.py --report parity/function_example_alignment_report.json --policy parity/functional_gate_policy.yml`
- `python tools/parity/check_example_output_spec.py --report parity/function_example_alignment_report.json --spec parity/example_output_spec.yml`
- `python tools/reports/generate_validation_pdf.py --repo-root "$PWD" --matlab-help-root /Users/iahncajigas/Library/CloudStorage/Dropbox/Research/Matlab/nSTAT_currentRelease_Local/helpfiles --notebook-group all --timeout 900 --skip-command-tests --parity-mode gate --enforce-unique-images --min-unique-images-per-topic 1 --max-cross-topic-reuse-ratio 1.0`
- `python tools/reports/generate_validation_pdf.py --repo-root "$PWD" --matlab-help-root /Users/iahncajigas/Library/CloudStorage/Dropbox/Research/Matlab/nSTAT_currentRelease_Local/helpfiles --notebook-group all --timeout 900 --skip-command-tests --parity-mode image --skip-parity-check`
- `python tools/reports/build_image_parity_pdfs.py --report-json <latest-json> --python-out output/pdf/image_mode_parity/python_pages.pdf --matlab-out output/pdf/image_mode_parity/matlab_pages.pdf --pairs-json output/pdf/image_mode_parity/pairs.json`
- `python tools/reports/check_pdf_image_parity.py --python-pdf output/pdf/image_mode_parity/python_pages.pdf --matlab-pdf output/pdf/image_mode_parity/matlab_pages.pdf --out-dir output/pdf/image_mode_parity --dpi 150 --ssim-threshold 0.70 --max-failing-pages 0`
- `python tools/performance/run_python_benchmarks.py --tiers S --repeats 5 --warmup 1 --out-json output/performance/python_performance_report.json --out-csv output/performance/python_performance_report.csv`
- `python tools/performance/compare_matlab_python_performance.py --python-report output/performance/python_performance_report.json --matlab-report tests/performance/fixtures/matlab/performance_baseline_470fde8.json --policy parity/performance_gate_policy.yml --previous-python-report tests/performance/fixtures/python/performance_baseline_linux_20260304.json --report-out output/performance/performance_parity_report.json --csv-out output/performance/performance_parity_report.csv --fail-on-regression --require-regression-env-match`
- Local macOS reruns use `tests/performance/fixtures/python/performance_baseline_20260303.json` with the same command to satisfy strict env matching.

## Cycle 1
- Log: `output/cycle/cycle1.log`
- `pytest`: PASS
- numeric drift (0 failed topics): PASS
- functional parity (no gaps/partials): PASS
- example output spec: PASS
- gate-mode validation PDF (0 parity failures, 0 uniqueness violations): PASS
- image-mode parity (0 failing pages): PASS
- performance-parity (0 regression failures): PASS
- Fixes applied in cycle: comparator option to require regression env match + regression test coverage.

## Cycle 2
- Log: `output/cycle/cycle2.log`
- `pytest`: PASS
- numeric drift (0 failed topics): PASS
- functional parity (no gaps/partials): PASS
- example output spec: PASS
- gate-mode validation PDF (0 parity failures, 0 uniqueness violations): PASS
- image-mode parity (0 failing pages): PASS
- performance-parity (0 regression failures): PASS
- Fixes applied in cycle: Linux baseline + strict regression env matching in workflow/tests, decoding `computeSpikeRateCIs` vectorization, and added deterministic performance workloads for `nspikeTrain.getSigRep` and `Analysis.fitGLM`.
