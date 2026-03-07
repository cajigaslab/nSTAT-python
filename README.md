# nSTAT-python

`nSTAT-python` is a Python toolbox for neural spike-train analysis, modeling, and decoding.

[![test-and-build](https://github.com/cajigaslab/nSTAT-python/actions/workflows/ci.yml/badge.svg)](https://github.com/cajigaslab/nSTAT-python/actions/workflows/ci.yml)
[![pages](https://github.com/cajigaslab/nSTAT-python/actions/workflows/pages.yml/badge.svg)](https://github.com/cajigaslab/nSTAT-python/actions/workflows/pages.yml)

## Installation

```bash
python -m pip install nstat-toolbox
```

From source:

```bash
git clone git@github.com:cajigaslab/nSTAT-python.git
cd nSTAT-python
python -m pip install -e .[dev]
```

## Example data

`nSTAT-python` does not commit raw example data to the repository.

Install the example dataset with:

```bash
nstat-install --download-example-data always
```

Equivalent Python API:

```python
from nstat.data_manager import ensure_example_data

data_dir = ensure_example_data(download=True)
print(data_dir)
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
```

## Examples

### Paper Examples (Self-Contained)

Canonical source files:
- `examples/paper/*.py`
- `nstat/paper_examples_full.py`

Single command to regenerate the paper-example gallery metadata:

```bash
python tools/paper_examples/build_gallery.py
```

This writes `docs/paper_examples.md` and `docs/figures/manifest.json`, and
ensures canonical figure-gallery directories exist under
`docs/figures/example01/` through `docs/figures/example05/`.

| Example | What question it answers | Run command | Links |
|---|---|---|---|
| Example 01 | Does Mg2+ washout produce firing-rate dynamics beyond a constant Poisson baseline? | `python examples/paper/example01_mepsc_poisson.py` | [Script](examples/paper/example01_mepsc_poisson.py) · [Figures](docs/figures/example01/) |
| Example 02 | What stimulus lag and history order best explain whisker-evoked spike trains? | `python examples/paper/example02_whisker_stimulus_thalamus.py` | [Script](examples/paper/example02_whisker_stimulus_thalamus.py) · [Figures](docs/figures/example02/) |
| Example 03 | How do PSTH and SSGLM differ in capturing trial learning dynamics? | `python examples/paper/example03_psth_and_ssglm.py` | [Script](examples/paper/example03_psth_and_ssglm.py) · [Figures](docs/figures/example03/) |
| Example 04 | How do Gaussian and Zernike basis models compare for place-field mapping? | `python examples/paper/example04_place_cells_continuous_stimulus.py` | [Script](examples/paper/example04_place_cells_continuous_stimulus.py) · [Figures](docs/figures/example04/) |
| Example 05 | How accurately can neural populations decode latent stimulus and reach state? | `python examples/paper/example05_decoding_ppaf_pphf.py` | [Script](examples/paper/example05_decoding_ppaf_pphf.py) · [Figures](docs/figures/example05/) |

Expanded paper-example index and figure gallery:
- [docs/paper_examples.md](docs/paper_examples.md)

### Supplementary Examples

These smaller demos remain useful as quick install and plotting checks.

| Example | Run command | Output |
|---|---|---|
| Multitaper spectrum + spectrogram | `python examples/readme_examples/example1_multitaper_and_spectrogram.py` | [PNG](examples/readme_examples/images/readme_example1_multitaper_and_spectrogram.png) |
| Simulated CIF spike train | `python examples/readme_examples/example2_simulate_cif_spiketrain_10s.py` | [PNG](examples/readme_examples/images/readme_example2_simulate_cif_spiketrain_10s.png) |
| Spike-train raster | `python examples/readme_examples/example3_nstcoll_raster_from_example2.py` | [PNG](examples/readme_examples/images/readme_example3_nstcoll_raster.png) |

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

## Cite

Cajigas, I., Malika, W. Q., & Brown, E. N. (2012).  
nSTAT: Open-source neural spike train analysis toolbox for Matlab.  
Journal of Neuroscience Methods, 211, 245–264.  
https://doi.org/10.1016/j.jneumeth.2012.08.009
