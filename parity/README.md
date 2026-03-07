# Parity Inventory

This directory tracks MATLAB-to-Python parity for the standalone port.

Current inventory source:
- [`manifest.yml`](./manifest.yml)
- [`class_fidelity.yml`](./class_fidelity.yml)
- [`report.md`](./report.md)

Generated report:

```bash
python tools/parity/build_report.py
```

Current headline status:
- Public API coverage matches the MATLAB inventory except for the explicitly non-applicable `nstatOpenHelpPage`.
- Class-fidelity auditing is tracked separately from name-mapping parity in `class_fidelity.yml`, and it remains intentionally stricter and more conservative than the mapping manifest.
- Help/notebook parity covers the inventoried MATLAB help workflow surface, including the top-level `NeuralSpikeAnalysis_top`, `PaperOverview`, `Examples`, and `ClassDefinitions` navigation pages.
- Canonical paper examples, gallery structure, and README/docs presentation are committed and mapped in Python.
- CI now validates API surface, dataset integrity, notebook smoke execution, paper gallery drift, and docs builds.
