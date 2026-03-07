nSTAT-python
============

Neural Spike Train Analysis Toolbox for Python

[![test-and-build](https://github.com/cajigaslab/nSTAT-python/actions/workflows/ci.yml/badge.svg)](https://github.com/cajigaslab/nSTAT-python/actions/workflows/ci.yml)
[![pages](https://github.com/cajigaslab/nSTAT-python/actions/workflows/pages.yml/badge.svg)](https://github.com/cajigaslab/nSTAT-python/actions/workflows/pages.yml)

nSTAT-python is an open-source, object-oriented Python toolbox that implements a range of models and algorithms for neural spike-train analysis, modeling, and decoding. The toolbox is designed for quick, consistent neural data analysis in native Python while keeping the paper-example dataset outside the Git repository.

Like the MATLAB toolbox paper, the Python port centers point-process generalized linear models for spike trains, while also supporting Gaussian-signal workflows, simulation, fitting diagnostics, and decoding. Although created with neural signal processing in mind, nSTAT-python can be used more generally for discrete and continuous time-series analysis.

Like all open-source projects, nSTAT-python benefits from issues, suggestions, and code contributions. The current source repository is:

- https://github.com/cajigaslab/nSTAT-python

Lab websites:

- Neuroscience Statistics Research Laboratory: https://www.neurostat.mit.edu
- RESToRe Lab: https://www.med.upenn.edu/cajigaslab/

How to install nSTAT-python
---------------------------

1. Clone this repository and create or activate a Python 3.10+ environment.
2. Install the package from source:

```bash
git clone https://github.com/cajigaslab/nSTAT-python.git
cd nSTAT-python
python -m pip install -e .[dev]
```

3. Optional post-install helper:

```bash
nstat-install
```

When a paper example or dataset helper needs the canonical example dataset, nSTAT-python downloads the figshare dataset automatically into a local cache. The raw dataset is not stored in this Git repository.

To prefetch the dataset ahead of time:

```bash
nstat-install --download-example-data always
```

Equivalent Python API:

```python
from nstat.data_manager import ensure_example_data

data_dir = ensure_example_data(download=True)
print(data_dir)
```

Quickstart (Python 3.10+)
-------------------------

```bash
git clone https://github.com/cajigaslab/nSTAT-python.git
cd nSTAT-python
python -m pip install -e .[dev]
python examples/nSTATPaperExamples.py --repo-root .
```

The first paper-example or dataset call downloads the figshare dataset automatically. Repository checkouts cache it under `data_cache/nstat_data/` by default. Set `NSTAT_DATA_DIR` to use another cache location.

Paper Examples (Self-Contained)
-------------------------------

Canonical source files:

- `examples/nSTATPaperExamples.py` (full command-line runner)
- `nstat/paper_examples_full.py` (paper-aligned experiment implementations)
- `examples/nstat_paper_examples.py` and `nstat/paper_examples.py` (lighter-weight summary runner)
- `notebooks/nSTATPaperExamples.ipynb` (notebook narrative)

Single command to run the full paper-aligned example suite:

```bash
python examples/nSTATPaperExamples.py --repo-root .
```

This command downloads the figshare dataset automatically when needed and prints JSON summaries for the experiment blocks. The Python package does not require a MATLAB checkout.

| Example | What question it answers | Python entrypoint |
|---|---|---|
| Example 01 | Do mEPSCs follow constant vs piecewise Poisson firing under Mg2+ washout? | `nstat.paper_examples_full.run_experiment1` |
| Example 02 | How do explicit whisker stimulus and spike history improve thalamic GLM fits? | `nstat.paper_examples_full.run_experiment2` |
| Example 03 | How do PSTH and SSGLM capture within-trial and across-trial dynamics? | `nstat.paper_examples_full.run_experiment3` and `run_experiment3b` |
| Example 04 | Which receptive-field basis (Gaussian vs Zernike-like) better fits place cells? | `nstat.paper_examples_full.run_experiment4` |
| Example 05 | How well do point-process-inspired decoders recover latent stimulus and state? | `nstat.paper_examples_full.run_experiment5`, `run_experiment5b`, and `run_experiment6` |

For a lighter-weight paper overview with plot payloads:

```python
from pathlib import Path

from nstat.paper_examples import run_paper_examples

results = run_paper_examples(Path.cwd())
print(results["experiment2"])
print(results["experiment3"])
print(results["experiment4"])
print(results["experiment5"])
```

Documentation
-------------

Minimal package docs live under [`docs/`](docs/).

- API reference: [`docs/api.rst`](docs/api.rst)
- Data installation: [`docs/data_installation.rst`](docs/data_installation.rst)

For mathematical and programmatic details of the toolbox, see:

Cajigas I, Malik WQ, Brown EN. nSTAT: Open-source neural spike train analysis toolbox for Matlab. Journal of Neuroscience Methods 211: 245-264, Nov. 2012
http://doi.org/10.1016/j.jneumeth.2012.08.009
PMID: 22981419

Paper-aligned toolbox map
-------------------------

To keep terminology and workflows aligned with the 2012 toolbox paper, the Python package groups core functionality along the same analysis paths:

- Class hierarchy and object model (`SignalObj`, `Covariate`, `Trial`, `Analysis`, `FitResult`, `DecodingAlgorithms`)
- Fitting and assessment workflow (GLM fitting, diagnostics, summaries)
- Simulation workflow (conditional intensity and thinning examples)
- Decoding workflow (stimulus and state reconstruction)
- Example-to-paper section mapping via `nstat.paper_examples_full`

If you use nSTAT-python in your work, please cite the paper above.
nSTAT is protected by the GPL v2 Open Source License.

The code repository for nSTAT-python is hosted on GitHub at https://github.com/cajigaslab/nSTAT-python .
The paper-example dataset is distributed separately from the Git repository:

- Figshare dataset DOI: https://doi.org/10.6084/m9.figshare.4834640.v3
- Paper DOI: https://doi.org/10.1016/j.jneumeth.2012.08.009

Standalone Python repository
----------------------------

`nSTAT-python` is maintained as a separate repository from the MATLAB toolbox and does not require files from `cajigaslab/nSTAT`.

This repository provides:

- Native Python implementations of core spike-train analysis and decoding workflows
- On-demand dataset download directly from figshare
- Notebook and script examples that run without a MATLAB install
- A `nstat.compat.matlab` namespace for familiar class names where API continuity is useful
