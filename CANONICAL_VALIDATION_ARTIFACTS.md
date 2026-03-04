# Canonical Validation Artifacts

This document records the canonical gate-mode validation artifact set and
reproduction command for `nSTAT-python` parity checks.

## Pinned MATLAB reference
- Repository: `https://github.com/cajigaslab/nSTAT.git`
- Commit SHA: `470fde8f9f6b60fe8f9ec51155e34478b6d541f6`
- Config source: [`parity/matlab_reference.yml`](./parity/matlab_reference.yml)

## Canonical local artifact set (latest)
Generated on: `2026-03-03` (America/New_York)

- PDF: [`output/pdf/nstat_python_validation_report_20260303_232103.pdf`](./output/pdf/nstat_python_validation_report_20260303_232103.pdf)
- JSON: [`output/pdf/nstat_python_validation_report_20260303_232103.json`](./output/pdf/nstat_python_validation_report_20260303_232103.json)
- CSV: [`output/pdf/nstat_python_validation_report_20260303_232103.csv`](./output/pdf/nstat_python_validation_report_20260303_232103.csv)

## Reproduction command
```bash
python tools/reports/generate_validation_pdf.py \
  --repo-root "$PWD" \
  --matlab-help-root /tmp/upstream-nstat/helpfiles \
  --notebook-group all \
  --timeout 900 \
  --skip-command-tests \
  --parity-mode gate \
  --enforce-unique-images \
  --min-unique-images-per-topic 1 \
  --max-cross-topic-reuse-ratio 1.0
```

## CI canonical names
CI workflows normalize the latest gate-mode report to stable artifact names:

- `output/pdf/validation_gate_mode_latest.pdf`
- `output/pdf/validation_gate_mode_latest.json`
- `output/pdf/validation_gate_mode_latest.csv`

Image-mode parity artifacts are emitted under:

- `output/pdf/image_mode_parity/summary.json`
- `output/pdf/image_mode_parity/pairs.json`
- `output/pdf/image_mode_parity/diff/`
