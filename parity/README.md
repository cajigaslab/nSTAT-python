# Parity Inventory

This directory tracks MATLAB-to-Python parity for the standalone port.

Current inventory source:
- [`manifest.yml`](./manifest.yml)
- [`class_fidelity.yml`](./class_fidelity.yml)
- [`simulink_fidelity.yml`](./simulink_fidelity.yml)
- [`report.md`](./report.md)

Generated report:

```bash
python tools/parity/build_report.py
```

Refresh MATLAB-derived fixtures from the sibling MATLAB repo:

```bash
matlab -batch "cd('../nSTAT'); addpath(fullfile(pwd,'tools','python')); export_python_port_fixtures; exit"
```

Run the pure-Python release gate:

```bash
python tools/release/run_fidelity_gate.py
```

Run the MATLAB-side `pyenv` fidelity suite from the sibling MATLAB repo:

```bash
matlab -batch "cd('../nSTAT'); addpath(fullfile(pwd,'tools','python')); results = runtests('tests/python_port_fidelity'); assertSuccess(results); exit"
```

Current headline status:
- Public API coverage matches the MATLAB inventory except for the explicitly non-applicable `nstatOpenHelpPage`.
- Class-fidelity auditing is tracked separately from name-mapping parity in `class_fidelity.yml`, and it now records `symbol_presence_verified` so the audit can distinguish prose parity from live runtime symbol resolution.
- Simulink-backed workflows are inventoried separately in `simulink_fidelity.yml` so model-dependent execution paths are not conflated with native Python parity.
- Help/notebook parity covers the inventoried MATLAB help workflow surface, including the top-level `NeuralSpikeAnalysis_top`, `PaperOverview`, `Examples`, and `ClassDefinitions` navigation pages.
- Canonical paper examples, gallery structure, and README/docs presentation are committed and mapped in Python.
- CI now validates API surface, dataset integrity, notebook smoke execution, notebook-helpfile full runs, paper gallery drift, and docs builds.
