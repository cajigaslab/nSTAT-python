<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/nstat-logo-light.png">
  <img alt="nSTAT — neural Spike Train Analysis Toolbox" src="docs/_static/nstat-logo.png" width="440">
</picture>

### Neural Spike Train Analysis Toolbox for Python

[![PyPI version](https://img.shields.io/pypi/v/nstat-toolbox.svg?color=2c5282)](https://pypi.org/project/nstat-toolbox/)
[![Python versions](https://img.shields.io/pypi/pyversions/nstat-toolbox.svg)](https://pypi.org/project/nstat-toolbox/)
[![test-and-build](https://github.com/cajigaslab/nSTAT-python/actions/workflows/ci.yml/badge.svg)](https://github.com/cajigaslab/nSTAT-python/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-2c5282.svg)](https://cajigaslab.github.io/nSTAT-python/)
[![License: GPL v2](https://img.shields.io/badge/license-GPL--2.0-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.jneumeth.2012.08.009-blue.svg)](https://doi.org/10.1016/j.jneumeth.2012.08.009)

</div>

**Jump to:** [Quickstart](#quickstart) · [Concepts learning track](#concepts-learning-track) · [Tutorials and notebooks](#tutorials-and-notebooks) · [Paper examples](#paper-examples) · [`nstat.extras` helpfiles](#nstatextras-helpfiles) · [Reference](#reference) · [Ecosystem](#ecosystem) · [Citation](#citation)

> 📖 **New here?** Start with the
> [**5-minute intro**](https://cajigaslab.github.io/nSTAT-python/intro.html) —
> a friendly tour of the toolbox with runnable snippets, the
> `nstat.extras` opt-in bridges, and the paper-example gallery.
>
> 🎓 **Learning the science?** The
> [**Concepts & Background**](https://cajigaslab.github.io/nSTAT-python/concepts/index.html)
> guide teaches microelectrode recordings, the LFP, point-process GLMs,
> goodness-of-fit, and decoding — with figures, runnable examples, and cited
> literature.
>
> 🛠 **Other ways in:** the runnable [`00_getting_started.ipynb`](notebooks/00_getting_started.ipynb)
> notebook; six end-to-end tutorial scripts in [`examples/tutorials/`](examples/tutorials/);
> the five paper examples in [`examples/paper/`](examples/paper/); and the
> reference notebooks in [`notebooks/`](notebooks/).

`nSTAT-python` is the Python port of the
[nSTAT toolbox](https://github.com/cajigaslab/nSTAT)
([Cajigas, Malik & Brown 2012](https://doi.org/10.1016/j.jneumeth.2012.08.009)).
It implements point-process generalized linear models (GLMs) with stimulus,
history, and ensemble terms; the
[time-rescaling KS test](https://cajigaslab.github.io/nSTAT-python/concepts/goodness_of_fit_and_decoding.html)
for goodness-of-fit (per-neuron and population-level via
`population_time_rescale`, [Tao et al. 2018](https://pubmed.ncbi.nlm.nih.gov/30298220/));
adaptive decoding (PPAF / PPHF / Kalman / EM); and continuous-signal analysis
for LFP / EEG / ECoG (multitaper spectra, spectrograms, Kalman filtering). An
opt-in [`nstat.extras`](#nstatextras-helpfiles) namespace adds bridges to
Dynamax (EM-trained state-space models), `replay_trajectory_classification`
(clusterless decoding), Neo / pynapple / pynwb (data I/O), and
validation oracles.

Although designed for neural signal processing, nSTAT works as a generic
toolkit for analyzing any discrete or continuous time series. From the lab:
[Neuroscience Statistics Research Laboratory](https://www.neurostat.mit.edu)
and [RESToRe Lab](https://www.med.upenn.edu/cajigaslab/).

## Quickstart

```bash
python -m pip install nstat-toolbox
nstat-install --download-example-data always   # ~150 MB figshare dataset
```

```python
import numpy as np
from nstat import nspikeTrain

times = np.sort(np.random.default_rng(0).uniform(0, 1, 100))
st = nspikeTrain(times, name="neuron1", sampleRate=1000,
                 minTime=0.0, maxTime=1.0)
print(f"{st.n_spikes} spikes")
```

From source (development install):

```bash
git clone git@github.com:cajigaslab/nSTAT-python.git
cd nSTAT-python
python -m pip install -e .[dev]
nstat-install --download-example-data always
pytest -q && python tools/paper_examples/build_gallery.py
```

The example dataset is also reachable from Python:

```python
from nstat.data_manager import ensure_example_data
data_dir = ensure_example_data(download=True)
```

## Concepts learning track

A 15-page didactic track that teaches the neuroscience and statistics behind
the API. Each page builds intuition first, then shows the matching nSTAT
objects and a runnable snippet, and cites the primary literature. The
[concepts index](https://cajigaslab.github.io/nSTAT-python/concepts/index.html)
hosts the full path; the table below is the direct map.

| # | Page | What you'll learn |
|---|---|---|
| 1 | [Microelectrode recordings: spikes & the LFP](https://cajigaslab.github.io/nSTAT-python/concepts/microelectrode_recordings.html) | What an electrode measures; the broadband → spikes + LFP split; single vs multi-unit; spike sorting. |
| 2 | [Spike trains & point-process GLMs](https://cajigaslab.github.io/nSTAT-python/concepts/spike_trains_and_glms.html) | Spike trains as point processes, the conditional intensity function, log-link GLMs with stimulus + history + ensemble terms. |
| 3 | [The LFP & spectral analysis](https://cajigaslab.github.io/nSTAT-python/concepts/lfp_and_spectral.html) | Multitaper power spectra, spectrograms, the time–bandwidth trade-off, Kalman filtering. |
| 4 | [Goodness-of-fit & decoding](https://cajigaslab.github.io/nSTAT-python/concepts/goodness_of_fit_and_decoding.html) | Time-rescaling KS, population GOF, PPAF / PPHF decoding, clusterless decoding from waveform features. |
| 5 | [State-space models & EM](https://cajigaslab.github.io/nSTAT-python/concepts/state_space_and_em.html) | The across-trial SSGLM (forward–backward smoother); EM-trained latent state-space models with multi-restart selection. |
| 6 | [Network connectivity](https://cajigaslab.github.io/nSTAT-python/concepts/network_connectivity.html) | Ensemble GLM coupling, cross-correlograms, Granger; why correlation is not connection (the common-input trap). |
| 7 | [Uncertainty & confidence intervals](https://cajigaslab.github.io/nSTAT-python/concepts/uncertainty_and_confidence.html) | Fisher information, CIs on coefficients and firing rates, credible bands on a decode. |
| 8 | [Rhythmic firing & the clinical microelectrode](https://cajigaslab.github.io/nSTAT-python/concepts/rhythmic_firing_and_clinical_microelectrode.html) | Tremor cells modelled as a periodic-covariate GLM; the beta-band biomarker that guides adaptive DBS. |
| 9 | [Population geometry](https://cajigaslab.github.io/nSTAT-python/concepts/population_geometry.html) | A PCA sketch of population activity; neural manifolds; pointers to GPFA and dimensionality reduction. |
| 10 | [From filters to deep learning](https://cajigaslab.github.io/nSTAT-python/concepts/from_filters_to_deep_learning.html) | What carries over from the PPAF to RNN / LSTM / transformer decoders, and what changes. |
| 11 | [Further study](https://cajigaslab.github.io/nSTAT-python/concepts/further_study.html) | Topics nSTAT does not implement, with primary references for each. |
| 12 | [Pitfalls & FAQ](https://cajigaslab.github.io/nSTAT-python/concepts/pitfalls_and_faq.html) | Common mistakes that quietly invalidate an analysis — bin width, multiple comparisons, missing history, multitaper choices, reproducibility & seeds. |
| 13 | [Self-check](https://cajigaslab.github.io/nSTAT-python/concepts/self_check.html) | Per-topic quizzes plus cross-cutting synthesis questions; answers collapsible. |
| 14 | [Glossary](https://cajigaslab.github.io/nSTAT-python/concepts/glossary.html) | Plain-language definitions of every term, with HTML anchors for deep-linking from any page. |
| 15 | [Annotated bibliography](https://cajigaslab.github.io/nSTAT-python/concepts/bibliography.html) | Every cited reference with a one-line note on why it matters for nSTAT users. |

Every topical page opens with a **"Glossary jumps"** box that deep-links into
the glossary entries it uses.

## Tutorials and notebooks

Runnable, end-to-end lessons organized by depth:

| File | What you'll do |
|---|---|
| [`notebooks/00_getting_started.ipynb`](notebooks/00_getting_started.ipynb) | The executable mirror of the 5-minute intro: build a spike train, fit a GLM, run the KS test, decode a stimulus with the PPAF — all in one notebook. |
| [`examples/tutorials/encoding_to_goodness_of_fit.py`](examples/tutorials/encoding_to_goodness_of_fit.py) | Encoding → GLM → time-rescaling KS, with a correct-vs-wrong model contrast. |
| [`examples/tutorials/model_comparison.py`](examples/tutorials/model_comparison.py) | AIC / BIC + KS contrasts on nested GLMs, with Fisher-information confidence intervals on every coefficient. |
| [`examples/tutorials/decoding_ppaf.py`](examples/tutorials/decoding_ppaf.py) | Decode a hidden stimulus from a population with the point-process adaptive filter; RMSE drops with population size. |
| [`examples/tutorials/network_coupling.py`](examples/tutorials/network_coupling.py) | Recover known asymmetric excite/inhibit wiring from two simulated neurons via the cross-correlogram and a coupling GLM. |
| [`examples/tutorials/clinical_microelectrode_walkthrough.py`](examples/tutorials/clinical_microelectrode_walkthrough.py) | A simulated tremor cell: encode → KS check → beta-band spectrum → PPAF phase decode. |
| [`examples/tutorials/place_cell_walkthrough.py`](examples/tutorials/place_cell_walkthrough.py) | Capstone on **real** hippocampal place-cell data: encode → check → decode, with the honest lesson that a model can decode well yet still fail goodness-of-fit. |
| [`examples/tutorials/Tutorial_MicroelectrodeToDecoding.ipynb`](examples/tutorials/Tutorial_MicroelectrodeToDecoding.ipynb) | Notebook-format guided tour spanning microelectrode signals → spikes → multitaper spectra → GLM → GOF → decoding. |

Reference notebooks (MATLAB-help ports) live under
[`notebooks/`](notebooks/) and the
["By concept" crosswalk](https://cajigaslab.github.io/nSTAT-python/Examples.html#by-concept-where-to-start)
in the Example Index maps each concepts page to the notebooks that
demonstrate it.

> 🖼 **Browse rendered notebook figures on GitHub:**
> [`docs/notebook_galleries/`](docs/notebook_galleries/) — every notebook's `FigureTracker`
> output, regenerated by `make regen-notebook-galleries` and drift-checked
> in CI. Each gallery links back to its source `.ipynb`.

## Paper examples

The five canonical examples from Cajigas, Malik & Brown (2012), each
reproduced as a self-contained Python script with a generated figure gallery.

Regenerate the gallery metadata after editing any paper-example script:

```bash
python tools/paper_examples/build_gallery.py
```

| Example | Thumbnail | What question it answers | Run command | Links |
|---|---|---|---|---|
| Example 01 | ![Example 01](docs/figures/example01/fig01_constant_mg_summary.png) | Do mEPSCs follow constant vs piecewise Poisson firing under Mg2+ washout? | `python examples/paper/example01_mepsc_poisson.py` | [Script](examples/paper/example01_mepsc_poisson.py) · [Figures](docs/figures/example01/) |
| Example 02 | ![Example 02](docs/figures/example02/fig01_data_overview.png) | How do explicit whisker stimulus and spike history improve thalamic GLM fits? | `python examples/paper/example02_whisker_stimulus_thalamus.py` | [Script](examples/paper/example02_whisker_stimulus_thalamus.py) · [Figures](docs/figures/example02/) |
| Example 03 | ![Example 03](docs/figures/example03/fig01_simulated_and_real_rasters.png) | How do PSTH and SSGLM capture within-trial and across-trial dynamics? | `python examples/paper/example03_psth_and_ssglm.py` | [Script](examples/paper/example03_psth_and_ssglm.py) · [Figures](docs/figures/example03/) |
| Example 04 | ![Example 04](docs/figures/example04/fig01_example_cells_path_overlay.png) | Which receptive-field basis (Gaussian vs Zernike) better fits place cells? | `python examples/paper/example04_place_cells_continuous_stimulus.py` | [Script](examples/paper/example04_place_cells_continuous_stimulus.py) · [Figures](docs/figures/example04/) |
| Example 05 | ![Example 05](docs/figures/example05/fig01_univariate_setup.png) | How well do adaptive/hybrid point-process filters decode stimulus and reach state? | `python examples/paper/example05_decoding_ppaf_pphf.py` | [Script](examples/paper/example05_decoding_ppaf_pphf.py) · [Figures](docs/figures/example05/) |
| Example 06 | ![Example 06](docs/figures/example06/fig01_true_vs_basis_recovered_rate.png) | How does a tensor-product B-spline Poisson GLM recover a known 2-D place field, and how do its rate and second-order diagnostics compare to an LGCP? | `python examples/paper/example06_place_fields_glm_basis.py` | [Script](examples/paper/example06_place_fields_glm_basis.py) · [Figures](docs/figures/example06/) |
| Example 07 | ![Example 07](docs/figures/example07/fig01_adjacency_and_positions.png) | Can the Bartlett spectrum and wave-peak detector recover the speed and direction of a known planar wave embedded in a multivariate Hawkes triggering matrix? | `python examples/paper/example07_spatiotemporal_hawkes_waves.py` | [Script](examples/paper/example07_spatiotemporal_hawkes_waves.py) · [Figures](docs/figures/example07/) |

Expanded paper-example index and figure gallery:
[docs/paper_examples.md](docs/paper_examples.md).

The figshare paper dataset is distributed separately from the Git repository:
[DOI 10.6084/m9.figshare.4834640.v3](https://doi.org/10.6084/m9.figshare.4834640.v3)
(`nstat-install --download-example-data always` fetches it; `NSTAT_OFFLINE=1`
forces offline mode).

Plot-style policy (modern readability vs strict-reproduction legacy):

```python
from nstat.plot_style import set_plot_style
set_plot_style('modern')   # default
set_plot_style('legacy')   # strict paper reproduction
```

## `nstat.extras` helpfiles

Opt-in bridges (`nstat.extras.*`) to libraries in the modern Python
systems-neuroscience stack. Each has a dedicated narrative helpfile under
[`docs/extras/`](docs/extras/) covering install, intended use, gotchas, and
runnable snippets. Install via the optional-dep group; install everything
non-JAX at once with `pip install nstat-toolbox[all-extras]`.

| Bridge | What it does | Helpfile | Optional dep |
|---|---|---|---|
| `nstat.extras.em.dynamax_bridge` | EM-trained linear-Gaussian / point-process / hybrid state-space models — the `KF_EM` / `PP_EM` / `mPPCO_EM` family. Held-out predictive log-likelihood + multi-restart selection (canonical-gauge identifiability). | [em_dynamax](docs/extras/em_dynamax.md) | `[dynamax]` (pulls JAX ~200 MB) |
| `nstat.extras.decoding.clusterless_bridge` | Clusterless marked point-process decoding (no spike sorting) + trajectory-type classification — the modern descendant of nSTAT's PPAF / PPHF filters ([Denovellis et al. 2021](https://pubmed.ncbi.nlm.nih.gov/34570699/)). | [decoding_clusterless](docs/extras/decoding_clusterless.md) | `[clusterless]` (pulls JAX ~200 MB) |
| `nstat.extras.spatial` | Spatial / spatiotemporal point processes: LGCP rate maps with credible bands (Laplace), inhomogeneous second-order goodness-of-fit (`g`/`K`/`L`, F/G/J, global-rank envelope), and the discrete-time-rescaling KS correction ([Haslinger-Pipa-Brown 2010](https://pubmed.ncbi.nlm.nih.gov/20608868/)). Pure NumPy/SciPy core; optional Hawkes/DPP/GP bridges. | [spatial_point_processes](docs/extras/spatial_point_processes.md) | core: none; `[spatial-gp]` / `[hawkes]` / `[dpp]` (optional) |
| `nstat.extras.interop.neo` | Vendor-format I/O via [Neo](https://github.com/NeuralEnsemble/python-neo) — Spike2 / NEX / Blackrock / Plexon / TDT / NWB. | [interop_neo](docs/extras/interop_neo.md) | `[neo]` |
| `nstat.extras.interop.nwb` | BRAIN-Initiative NWB:N standard reader via [pynwb](https://github.com/NeurodataWithoutBorders/pynwb). | [interop_nwb](docs/extras/interop_nwb.md) | `[nwb]` |
| `nstat.extras.interop.pynapple` | Time-series + epoch math via [pynapple](https://github.com/pynapple-org/pynapple), NWB-native. | [interop_pynapple](docs/extras/interop_pynapple.md) | `[pynapple]` |
| `nstat.extras.validation.nemos_bridge` | JAX Poisson-GLM cross-validation oracle via [NeMoS](https://github.com/flatironinstitute/nemos). | [validation_nemos](docs/extras/validation_nemos.md) | `[nemos]` |
| `nstat.extras.validation.pykalman_bridge` | Pure-NumPy Kalman cross-validation reference via [pykalman](https://github.com/pykalman/pykalman). | [validation_pykalman](docs/extras/validation_pykalman.md) | `[test-parity]` |
| `nstat.extras.validation.statsmodels_bridge` | Poisson GLM IRLS cross-validation oracle (~1e-9 agreement) via [statsmodels](https://www.statsmodels.org). | [validation_statsmodels](docs/extras/validation_statsmodels.md) | `[test-parity]` |
| `nstat.extras.metrics.spike_distances` | ISI / SPIKE-distance spike-train metrics via [PySpike](https://github.com/mariomulansky/PySpike). | [metrics_spike_distances](docs/extras/metrics_spike_distances.md) | `[metrics]` |

The interactive [`extras_summary.html`](https://cajigaslab.github.io/nSTAT-python/extras_summary.html)
landing page has the same content as bigger cards with code snippets.

## Reference

Full rendered documentation is on
[**GitHub Pages**](https://cajigaslab.github.io/nSTAT-python/):

| Page | What you'll find |
|---|---|
| [5-minute intro](https://cajigaslab.github.io/nSTAT-python/intro.html) | The friendly, illustrated tour with runnable snippets |
| [API reference](https://cajigaslab.github.io/nSTAT-python/api.html) | Every public symbol, auto-generated from NumPy-style docstrings |
| [Class definitions](https://cajigaslab.github.io/nSTAT-python/ClassDefinitions.html) | Method catalog for each MATLAB-faithful core class |
| [Paper-aligned toolbox map](https://cajigaslab.github.io/nSTAT-python/PaperOverview.html) | Crosswalk between the 2012 paper's workflow categories (object model, fitting/assessment, simulation, decoding) and the Python API |
| [`nstat.extras` summary](https://cajigaslab.github.io/nSTAT-python/extras_summary.html) | Per-bridge cards with install commands, status, and runnable snippets |
| [Example Index](https://cajigaslab.github.io/nSTAT-python/Examples.html) | Visual gallery of every runnable example, including a "By concept" crosswalk back to the concepts pages |
| [What's New](https://cajigaslab.github.io/nSTAT-python/whats_new.html) | Per-release change summaries |
| [`RELEASE_NOTES.md`](RELEASE_NOTES.md) | Full changelog (start here when upgrading) |
| [Methods roadmap](parity/methods_roadmap.md) | What's queued for upcoming releases |
| [Parity audit](parity/report.md) | MATLAB ↔ Python class & method parity verification |

## Ecosystem

The MATLAB reference toolbox lives in a separate repository:
[github.com/cajigaslab/nSTAT](https://github.com/cajigaslab/nSTAT). It retains
the original MATLAB classes, the `helpfiles/helptoc.xml` index, and the
`.mlx` example workflows.

For ecosystem peers nSTAT does **not** wrap (spike sorting, calcium imaging,
deep-learning decoders), recommended alternatives — with rationale in
[`parity/integration_opportunities.md`](parity/integration_opportunities.md):

- **[SpikeInterface](https://github.com/SpikeInterface/spikeinterface)** — spike sorting (nSTAT consumes pre-sorted data).
- **[Elephant](https://github.com/NeuralEnsemble/elephant)** — overlapping spike-train statistics; Neo-typed.
- **[ssqueezepy](https://github.com/OverLordGoldDragon/ssqueezepy)** — wavelet synchrosqueezing; planned `nstat.extras.spectral`.

nSTAT will benefit from your involvement. Open issues / PRs at
[github.com/cajigaslab/nSTAT-python](https://github.com/cajigaslab/nSTAT-python).

## Citation

If you use nSTAT in your work, please cite the toolbox paper:

> Cajigas I, Malik WQ, Brown EN. **nSTAT: Open-source neural spike train
> analysis toolbox for Matlab.** *Journal of Neuroscience Methods* 211:
> 245–264, Nov. 2012.
> [doi:10.1016/j.jneumeth.2012.08.009](https://doi.org/10.1016/j.jneumeth.2012.08.009) ·
> [PMID 22981419](https://pubmed.ncbi.nlm.nih.gov/22981419/)

Method references for every page in the concepts track are in the
[annotated bibliography](https://cajigaslab.github.io/nSTAT-python/concepts/bibliography.html).
The paper-example dataset has its own
[figshare DOI](https://doi.org/10.6084/m9.figshare.4834640.v3).
nSTAT-python is distributed under the **GPL-2.0** license.
