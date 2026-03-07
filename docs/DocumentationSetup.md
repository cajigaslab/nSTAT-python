# Documentation Setup

This page is the Python-native equivalent of the MATLAB help-integration
guide. `nSTAT-python` does not register pages inside the MATLAB help browser;
instead it ships a static Sphinx documentation site and local rebuild hooks.

## Install and Configure

Install from PyPI:

```bash
python -m pip install nstat-toolbox
```

Install from source:

```bash
git clone https://github.com/cajigaslab/nSTAT-python
cd nSTAT-python
python -m pip install -e .[dev]
```

Run the installer helper:

```bash
nstat-install --download-example-data prompt
```

Equivalent module form:

```bash
python -m nstat.install --download-example-data prompt
```

## Build and Refresh the Search Database

The Python installer can rebuild the local Sphinx HTML search index:

```bash
nstat-install
```

or directly:

```bash
python -m sphinx -W -b html docs docs/_build/html
```

The resulting search index is written to `docs/_build/html/searchindex.js`.

## Documentation Entry Points

Use these pages as the Python documentation entry points:

- [nSTAT Home](NeuralSpikeAnalysis_top.md)
- [Class Definitions](ClassDefinitions.md)
- [Example Index](Examples.md)
- [Paper Examples](paper_examples.md)

## Troubleshooting

- If example data is missing, rerun `nstat-install --download-example-data always`.
- If local docs are stale, rebuild with `python -m sphinx -W -b html docs docs/_build/html`.
- If you need a different example-data cache, set `NSTAT_DATA_DIR` before
  running the installer or examples.
