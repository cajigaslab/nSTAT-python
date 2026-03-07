# Parity Inventory

This directory tracks MATLAB-to-Python parity for the standalone port.

Current inventory source:
- [`manifest.yml`](./manifest.yml)
- [`report.md`](./report.md)

Generated report:

```bash
python tools/parity/build_report.py
```

Current headline status:
- Public API coverage matches the MATLAB inventory except for the explicitly non-applicable `nstatOpenHelpPage`.
- Help/notebook parity covers the inventoried MATLAB help workflow surface.
- Canonical paper examples, gallery structure, and README/docs presentation now exist in Python, but the example outputs remain partial until the canonical figures are generated.
- CI now validates API surface, dataset integrity, notebook smoke execution, paper gallery drift, and docs builds.
