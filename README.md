# nSTAT-python

**Neural Spike Train Analysis Toolbox for Python**

[![test-and-build](https://github.com/cajigaslab/nSTAT-python/actions/workflows/ci.yml/badge.svg)](https://github.com/cajigaslab/nSTAT-python/actions/workflows/ci.yml)

`nSTAT-python` is a Python port of the [nSTAT](https://github.com/cajigaslab/nSTAT)
open-source neural spike train analysis toolbox. It implements a range of models and
algorithms for neural spike train data analysis, with a focus on point-process
generalized linear models (GLMs), model fitting, model-order analysis, and adaptive
decoding. In addition to point-process algorithms, nSTAT also provides tools for
Gaussian signals — from correlation analysis to the Kalman filter — applicable to
continuous neural signals such as LFP, EEG, and ECoG.

One of nSTAT's key strengths is point-process generalized linear models for spike
train signals that provide a formal statistical framework for processing signals
recorded from ensembles of single neurons. It also has extensive support for model
fitting, model-order analysis, and adaptive decoding.

Although created with neural signal processing in mind, nSTAT can be used as a
generic tool for analyzing any types of discrete and continuous signals, and thus
has wide applicability.

Like all open-source projects, nSTAT will benefit from your involvement,
suggestions and contributions. This platform is intended as a repository for
extensions to the toolbox based on your code contributions as well as for flagging
and tracking open issues.

The current release can be installed from PyPI: `pip install nstat-toolbox`

Lab websites:
- Neuroscience Statistics Research Laboratory: https://www.neurostat.mit.edu
- RESToRe Lab: https://www.med.upenn.edu/cajigaslab/

## How to install nSTAT

```bash
python -m pip install nstat-toolbox
```

From source:

```bash
git clone git@github.com:cajigaslab/nSTAT-python.git
cd nSTAT-python
python -m pip install -e .[dev]
```

Install the example dataset:

```bash
nstat-install --download-example-data always
```

Equivalent Python API:

```python
from nstat.data_manager import ensure_example_data
data_dir = ensure_example_data(download=True)
```

Quickstart:

```bash
cd /path/to/nSTAT-python
pip install -e .[dev]
nstat-install --download-example-data always
pytest -q && python tools/paper_examples/build_gallery.py
```

## Paper Examples (Self-Contained)

Canonical source files:
- `examples/paper/*.py`
- `nstat/paper_examples_full.py`

Single command to regenerate the paper-example gallery metadata:

```bash
python tools/paper_examples/build_gallery.py
```

This writes `docs/paper_examples.md`, `docs/figures/manifest.json`, and
refreshes the canonical README paper-example table from
`examples/paper/manifest.yml`.

| Example | Thumbnail | What question it answers | Run command | Links |
|---|---|---|---|---|
| Example 01 | ![Example 01](docs/figures/example01/fig01_constant_mg_summary.png) | Do mEPSCs follow constant vs piecewise Poisson firing under Mg2+ washout? | `python examples/paper/example01_mepsc_poisson.py` | [Script](examples/paper/example01_mepsc_poisson.py) · [Figures](docs/figures/example01/) |
| Example 02 | ![Example 02](docs/figures/example02/fig01_data_overview.png) | How do explicit whisker stimulus and spike history improve thalamic GLM fits? | `python examples/paper/example02_whisker_stimulus_thalamus.py` | [Script](examples/paper/example02_whisker_stimulus_thalamus.py) · [Figures](docs/figures/example02/) |
| Example 03 | ![Example 03](docs/figures/example03/fig01_simulated_and_real_rasters.png) | How do PSTH and SSGLM capture within-trial and across-trial dynamics? | `python examples/paper/example03_psth_and_ssglm.py` | [Script](examples/paper/example03_psth_and_ssglm.py) · [Figures](docs/figures/example03/) |
| Example 04 | ![Example 04](docs/figures/example04/fig01_example_cells_path_overlay.png) | Which receptive-field basis (Gaussian vs Zernike) better fits place cells? | `python examples/paper/example04_place_cells_continuous_stimulus.py` | [Script](examples/paper/example04_place_cells_continuous_stimulus.py) · [Figures](docs/figures/example04/) |
| Example 05 | ![Example 05](docs/figures/example05/fig01_univariate_setup.png) | How well do adaptive/hybrid point-process filters decode stimulus and reach state? | `python examples/paper/example05_decoding_ppaf_pphf.py` | [Script](examples/paper/example05_decoding_ppaf_pphf.py) · [Figures](docs/figures/example05/) |

Expanded paper-example index and figure gallery:
- [docs/paper_examples.md](docs/paper_examples.md)

Plot style policy:

```python
from nstat.plot_style import set_plot_style

# Modern readability-focused plots (default)
set_plot_style('modern')

# Legacy visual style for strict reproduction
set_plot_style('legacy')
```

Rendered help documentation (GitHub Pages):
- https://cajigaslab.github.io/nSTAT-python/

For mathematical and programmatic details of the toolbox, see:

Cajigas I, Malik WQ, Brown EN. nSTAT: Open-source neural spike train analysis
toolbox for Matlab. Journal of Neuroscience Methods 211: 245–264, Nov. 2012.
https://doi.org/10.1016/j.jneumeth.2012.08.009
PMID: 22981419

## Paper-Aligned Toolbox Map

To keep terminology and workflows consistent with the 2012 toolbox paper,
the documentation includes a dedicated mapping page:
[docs/PaperOverview.md](docs/PaperOverview.md).

This page ties the Python toolbox to the paper's workflow categories:

- Class hierarchy and object model (`SignalObj`, `Covariate`, `Trial`,
  `Analysis`, `FitResult`, `DecodingAlgorithms`)
- Fitting and assessment workflow (GLM fitting, diagnostics, summaries)
- Simulation workflow (conditional intensity and thinning examples)
- Decoding workflow (univariate/bivariate and history-aware decoding)
- Example-to-paper section mapping via `nSTATPaperExamples`

If you use nSTAT in your work, please remember to cite the above paper in any publications.
nSTAT is protected by the GPL v2 Open Source License.

The code repository for the Python port of nSTAT is hosted on GitHub at
https://github.com/cajigaslab/nSTAT-python.
The paper-example dataset is distributed separately from the Git repository:
- Figshare dataset DOI: https://doi.org/10.6084/m9.figshare.4834640.v3
- Paper DOI: https://doi.org/10.1016/j.jneumeth.2012.08.009

## Code audit (2026-03-11)

The Python port was verified against the MATLAB reference through a comprehensive
5-phase audit covering all 16 classes and 484 methods. **466 methods found in
Python, 6 nominal (MATLAB-infrastructure) gaps.** Full class-level and behavioral
parity verified.

**Python bugs fixed during the port:**

- `SignalObj.std()` used `ddof=0`; MATLAB uses `ddof=1` (N-1 normalization)
- `CovariateCollection.isCovPresent()` off-by-one in boundary check
- `SpikeTrainCollection.psthGLM()` was a stub; now wired to the full GLM path
- `SpikeTrainCollection.getNSTnames()` / `getUniqueNSTnames()` ignored the
  `selectorArray` filter parameter
- `nspikeTrain.getNST()` missing resample check on retrieval

**MATLAB bugs discovered (13 total, filed as GitHub issues):**

- `FitResult.m` — KS test used `sampleRate` as bin width instead of
  `1/sampleRate`, invalidating goodness-of-fit for any sampleRate != 1
- `CIF.m` — `symvar()` reordered variables alphabetically, causing silent
  argument mismatch for non-alphabetical variable names
- `SignalObj.m` — `findPeaks('minima')` returned maxima; `findGlobalPeak('minima')`
  crashed; handle aliasing mutated input signals in arithmetic
- `DecodingAlgorithms.m` — `isa(condNum,'nan')` always false; `ExplambdaDeltaCubed`
  used `.^2` instead of `.^3`
- `Analysis.m` — Granger causality mask zeroed all columns instead of column `i`

See [parity/report.md](parity/report.md) for the full audit.

## MATLAB Toolbox

The original MATLAB nSTAT toolbox lives in a separate repository:

- https://github.com/cajigaslab/nSTAT

That repository is MATLAB-focused and retains:

- Original MATLAB class/source files
- MATLAB helpfiles and help index (`helpfiles/helptoc.xml`)
- MATLAB example workflows, including `.mlx` examples
