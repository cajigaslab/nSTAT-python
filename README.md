# nSTAT-python

`nSTAT-python` is a Python toolbox for neural spike-train analysis, modeling, and decoding.

[![test-and-build](https://github.com/cajigaslab/nSTAT-python/actions/workflows/ci.yml/badge.svg)](https://github.com/cajigaslab/nSTAT-python/actions/workflows/ci.yml)
[![pages](https://github.com/cajigaslab/nSTAT-python/actions/workflows/pages.yml/badge.svg)](https://github.com/cajigaslab/nSTAT-python/actions/workflows/pages.yml)

## Installation

```bash
python -m pip install nstat
```

From source:

```bash
git clone git@github.com:cajigaslab/nSTAT-python.git
cd nSTAT-python
python -m pip install -e .[dev,docs,notebooks]
```

## How to install nSTAT (post-install setup)

Run the setup helper:

```bash
nstat-install
```

Equivalent Python API:

```python
from nstat.install import nstat_install

report = nstat_install()
print(report.cache_dir)
```

## Quick start

```python
import numpy as np
from nstat.signal import Covariate
from nstat.spikes import SpikeTrain

t = np.linspace(0.0, 1.0, 1001)
x = np.sin(2 * np.pi * 5 * t)
cov = Covariate(time=t, data=x, name="stimulus", labels=["stim"])
spikes = SpikeTrain(spike_times=np.array([0.11, 0.42, 0.77]), t_start=0.0, t_end=1.0)
print(cov.sample_rate_hz, spikes.firing_rate_hz())
```

## Examples

- Python scripts and notebooks: `notebooks/`
- Learning notebooks are executable and suitable for local exploration or CI smoke runs.

## Documentation

- Docs home: [cajigaslab.github.io/nSTAT-python](https://cajigaslab.github.io/nSTAT-python/)
- Help index: [cajigaslab.github.io/nSTAT-python/help](https://cajigaslab.github.io/nSTAT-python/help/)

## Developer notes

- Run tests:

```bash
pytest -q
```

- Build docs:

```bash
sphinx-build -b html docs docs/_build
```

## Reference

Cajigas I, Malik WQ, Brown EN. nSTAT: Open-source neural spike train analysis toolbox for Matlab. *J Neurosci Methods* (2012), DOI: `10.1016/j.jneumeth.2012.08.009`, PMID: `22981419`.
