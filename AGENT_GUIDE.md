# AGENT_GUIDE — nSTAT-python for AI assistants

> **Audience:** AI coding assistants (Claude, GPT, Cursor, Copilot, etc.) and
> autonomous agents that need to use the `nstat-python` toolbox correctly.
> Updated: 2026-06-12. Package version: 0.5.0.
>
> The MATLAB reference toolbox lives in a *separate* repository
> (https://github.com/cajigaslab/nSTAT) and is deliberately kept independent
> from this Python port — there is no runtime coupling.  Only the Simulink
> point-process thinning simulator pre-dates this Python port; everything
> else stands alone.

This is a condensed, machine-readable orientation. For human-facing docs,
see [README.md](README.md), [docs/PaperOverview.md](docs/PaperOverview.md),
and [docs/ClassDefinitions.md](docs/ClassDefinitions.md).

---

## 1. What this package is (one paragraph)

`nstat-python` is a Python port of the MATLAB **nSTAT** (neural spike train
analysis toolbox), tied to Cajigas, Malik & Brown, *J Neurosci Methods*,
211:245-264 (2012). It implements point-process generalized linear models
(GLMs), conditional intensity functions (CIFs), adaptive decoding
(PPAF/PPHF/Kalman/EM), Gaussian-signal analysis (LFP/ECoG/EEG), and the
five canonical paper examples. The Python port preserves MATLAB class
names and method signatures wherever feasible — a deliberate design choice
to keep MATLAB users oriented.

**Install:**

```bash
pip install nstat-toolbox          # PyPI
nstat-install --download-example-data always   # example dataset (figshare, ~150 MB)
```

**Source layout:**

```
nstat/                  ~50 modules, 24.7 kLOC — the package itself
examples/paper/         5 canonical paper-example scripts (Cajigas 2012)
examples/tutorials/     6 runnable end-to-end teaching scripts + 1 notebook
examples/extras/        per-bridge demos for nstat.extras
examples/readme_examples/   4 short snippets
notebooks/              30+ Jupyter notebooks (many MATLAB-help-derived)
docs/                   Sphinx + MyST documentation
docs/concepts/          neuroscience + statistics learning track (14 pages)
parity/                 MATLAB↔Python parity manifests + audit report
tests/                  48 test files, 268 tests
tools/{notebooks,paper_examples,parity,release}/  build/maintenance scripts
```

---

## 2. Public API — canonical entry points

`nstat/__init__.py` exposes everything via `__all__`. The following is the
authoritative, current surface (run `python -c "import nstat; print(sorted(nstat.__all__))"`
to re-verify).

### Core data primitives
- `SignalObj` — continuous signal with time axis (LFP/ECoG/EEG); arithmetic, FFT, filtering.
- `Signal` — Python-flavored alias / subclass of `SignalObj` (use either; same class hierarchy).
- `Covariate` — design-matrix column with named labels (subclass of `SignalObj`).
- `nspikeTrain` — a single neuron's spike train. **Note the lowercase 'n'** — preserved from MATLAB.
- `SpikeTrain` — alias for `nspikeTrain`.
- `Events` — discrete event markers.
- `ConfidenceInterval` — CI value/interval with arithmetic.

### Collections
- `nstColl` — collection of `nspikeTrain` objects (ensemble of neurons).
- `SpikeTrainCollection` — alias for `nstColl`.
- `CovColl` — collection of `Covariate` objects.
- `CovariateCollection` — alias for `CovColl`.
- `ConfigColl` — collection of `TrialConfig` objects.
- `ConfigCollection` — alias for `ConfigColl`.

### Experiment / configuration
- `Trial` — bundle of spike trains, covariates, neighbors for one experiment.
- `TrialConfig` — covariate-by-covariate spec (constant / b-spline / unit-impulse).
- `History`, `HistoryBasis` — spike-history kernel design.

### Modeling and inference
- `Analysis` — top-level fit/analysis orchestrator (`GLMFit`, `computeHistLagForAll`, `computeGrangerCausalityMatrix`, etc.).
- `CIF`, `CIFModel` — conditional intensity functions, symbolic ↔ numeric evaluation.
- `LinearCIF` — closed-form (sympy-free) CIF for the two canonical link cases (Poisson log-link, binomial logit-link).  Drop-in compatible with `CIF` for the 5 eval methods used by `DecodingAlgorithms.PPDecode_update`.
- `FitResult` — fit output (coefficients, lambda signal, KS, AIC/BIC).
- `FitResSummary` / `FitSummary` — aggregator across fits/cells.
- `population_time_rescale` → `PopulationTimeRescaleResult` — multivariate
  (marked) point-process time-rescaling GOF (Tao, Weber, Arai & Eden 2018).
  Tests a *population* jointly: a ground-process KS plus a marked χ². Catches
  inter-neuron coupling misfit that the per-neuron `FitResult.computeKSStats`
  misses (e.g. synchronous neurons modeled as independent). Pure NumPy/SciPy,
  operates on per-neuron binned spike counts + model `lambda` per bin.
- `psth` — peri-stimulus time histogram convenience function.

### Decoding
- `DecodingAlgorithms` — class namespace for PPAF / PPHF / Kalman / EM / smoothers (most are static).
- `DecoderSuite` — thin Pythonic wrapper.
- `PoissonGLMResult`, `fit_poisson_glm` — standalone Poisson GLM utility (separate from `Analysis.GLMFit`).

### Simulation
- `simulate_poisson_from_rate(rate_signal, ...)` — homogeneous/inhomogeneous Poisson simulator.
- `simulate_point_process(...)`, `PointProcessSimulation`.
- `simulate_two_neuron_network(...)`, `NetworkSimulationResult`.
- `run_full_paper_examples(...)` — runs all five paper examples programmatically.

### Plot style
- `set_plot_style("modern" | "legacy")` — persists choice in a sidecar file.
- `get_plot_style()` — reads current style.
- `apply_plot_style(fig_or_axes, *, style="")` — applies the (persisted or
  explicitly passed) style to a figure. The paper examples accept
  `--plot-style {modern,legacy}` (default `legacy` for strict paper reproduction).

### Installation / data
- `nstat_install`, `nSTAT_Install` — CLI/programmatic installer.
- `get_dataset_path`, `list_datasets`, `verify_checksums` — dataset registry.
- `ensure_example_data(download=True)` — **importable but not in `__all__`**: `from nstat.data_manager import ensure_example_data`.
- `getPaperDataDirs`, `get_paper_data_dirs` — paths into the example dataset.

### MATLAB bridge (optional)
- `is_matlab_available()`, `get_matlab_nstat_path()`, `set_matlab_nstat_path(path)`.
- `MatlabFallbackWarning` — warning class when a method delegates to MATLAB.

### Exceptions
- `DataNotFoundError`, `MatlabEngineError`, `ParityValidationError`,
  `UnsupportedWorkflowError`.

---

## 3. How to use the toolbox — recipes

### Recipe A: build a Trial from spike times + a covariate

```python
import numpy as np
from nstat import Trial, TrialConfig, Covariate, nspikeTrain, nstColl

sample_rate = 1000  # Hz

# Spike train for one neuron (spike times in seconds)
spike_times = np.array([0.012, 0.087, 0.123, 0.301])
st = nspikeTrain(spike_times, name="n1", sampleRate=sample_rate,
                 minTime=0.0, maxTime=1.0)

# A continuous covariate (e.g., whisker velocity)
t = np.arange(0, 1.0, 1.0 / sample_rate)
stim = np.sin(2 * np.pi * 5 * t).reshape(-1, 1)
cov = Covariate(t, stim, name="stim", xlabelval="time", xunitval="s",
                ylabelval="vel", yunitval="mm/s",
                dataLabels=["stim"])

# A collection of spike trains (single neuron here)
nstc = nstColl([st])
nstc.setMinTime(0.0)
nstc.setMaxTime(1.0)

# Assemble Trial
trial = Trial(nstc, ev=None, covarColl=None, neighbors=None)
```

### Recipe B: fit a Poisson GLM with stimulus + history

```python
from nstat import TrialConfig, ConfigColl, Analysis, CovColl

cfg = TrialConfig(
    covariate_specs=[("Baseline", "constant"), ("stim", "spline")],
    sampleRate=sample_rate,
    history_window_times=[0.001, 0.002, 0.005, 0.01],  # history kernel knots
    ensCovHist=[],
)
configs = ConfigColl([cfg])

# Run the analysis (model selection across configs)
results = Analysis.runAnalysisForAllNeurons(trial, configs)
fit = results[0][0]   # FitResult for neuron 0 under config 0
print("AIC:", fit.AIC, "BIC:", fit.BIC)
print("KS stat:", fit.computeKSStats())
```

For a single GLM fit (no model-selection sweep), use ``Analysis.GLMFit``
directly.  It returns a ``GLMFitResult`` dataclass with both named-field
and tuple-unpack access:

```python
from nstat.analysis import Analysis  # GLMFitResult is also exported

result = Analysis.GLMFit(trial, neuron_number=1, lambdaIndex=1, Algorithm="GLM")
print("AIC:", result.AIC, "BIC:", result.BIC)
print("true log-lik:", result.loglik)         # AIC = -2*loglik + 2*k holds
print("matlab logLL:", result.logLL)          # hybrid MATLAB-parity quantity

# Legacy unpacking still works:
lambda_sig, b, dev, stats, AIC, BIC, logLL, distribution = result
```

### Recipe C: simulate spikes from a rate signal

```python
from nstat import simulate_poisson_from_rate, Signal

t = np.arange(0, 5.0, 0.001)
rate_hz = 20 + 10 * np.cos(2 * np.pi * 1.5 * t)
rate_signal = Signal(t, rate_hz.reshape(-1, 1), sampleRate=1000.0,
                     dataLabels=["rate"])
spikes = simulate_poisson_from_rate(rate_signal, n_trials=5)
```

### Recipe D: load the figshare paper dataset

```python
from nstat.data_manager import ensure_example_data, get_paper_data_dirs

data_root = ensure_example_data(download=True)  # ~150 MB on first call
# OR: data_root = ensure_example_data(download=False)  # raises if missing
mepsc = data_root / "mEPSCs"
place_cells = data_root / "Place Cells"
```

### Recipe E: run a paper example programmatically

```python
from examples.paper.example01_mepsc_poisson import run_example01

run_example01(export_figures=True,
              export_dir=None,        # defaults to docs/figures/example01/
              visible=False,          # headless
              plot_style="modern")    # or "legacy"
```

### Recipe F: regenerate the full figure gallery

```bash
python examples/paper/regenerate_all_figures.py             # legacy style
python examples/paper/regenerate_all_figures.py --plot-style modern
```

### Recipe G: PPAF/PPHF adaptive decoding

```python
from nstat import DecodingAlgorithms
# Most methods are @staticmethod
x_decoded, W, _ = DecodingAlgorithms.PPDecodeFilterLinear(
    A, Q, C, lambda_cif, dN, x0, W0, delta)
```

---

## 4. Toolbox capabilities — what nSTAT does well

- **Point-process GLM fitting** for spike trains with stimulus + history,
  including model-order selection via AIC/BIC and KS-based goodness-of-fit
  (time-rescaling theorem).
- **Conditional intensity functions** as symbolic objects (`sympy`-backed)
  with batch evaluation and exact-form CIF arithmetic.
- **State-space decoding**: linear/nonlinear point-process adaptive filters
  (PPAF), hybrid filters (PPHF mixing discrete + continuous state), Kalman
  filter/smoother for Gaussian observations.
- **Basis families**: b-spline / unit-impulse / Gaussian / Zernike
  polynomial bases for receptive-field modeling.
- **Ensemble analyses**: Granger causality matrix on neuron pairs,
  history-coefficient summaries across populations.
- **Simulation**: thinning-based Poisson, two-neuron network with
  reciprocal coupling, parametric CIF simulation.
- **MATLAB interop**: optional bridge via `matlab.engine` for direct
  numerical comparison (`is_matlab_available()`); the package gracefully
  degrades to pure-Python when MATLAB is absent.
- **Parity tracking**: every public class is audited against the MATLAB
  reference (see [parity/report.md](parity/report.md)).

---

## 5. Limitations — what NOT to assume

### 5.1 API gaps vs the MATLAB reference

These MATLAB methods exist but are **not yet ported** to Python (tracked in
[AUDIT_REPORT.md](AUDIT_REPORT.md) §3):

- `DecodingAlgorithms.KF_EM` family (Gaussian state-space EM)
  — `KF_EM`, `KF_EStep`, `KF_MStep`, `KF_ComputeParamStandardErrors`,
  `KF_EMCreateConstraints`.
- `DecodingAlgorithms.PP_EM` family (point-process state-space EM, no basis)
  — `PP_EM`, `PP_EStep`, `PP_MStep`, `PP_ComputeParamStandardErrors`,
  `PP_EMCreateConstraints`.
- `DecodingAlgorithms.mPPCO` family (mixed PP + Gaussian)
  — `mPPCODecodeLinear`, `mPPCODecode_predict`, `mPPCODecode_update`,
  `mPPCO_fixedIntervalSmoother`, `mPPCO_EM`, `mPPCO_EStep`, `mPPCO_MStep`,
  `mPPCO_ComputeParamStandardErrors`, `mPPCO_EMCreateConstraints`.
- `DecodingAlgorithms.computeSpikeRateCIs`, `computeSpikeRateDiffCIs` (Monte Carlo CIs).
- `FitResSummary.plotCoeffsWithoutHistory`, `plotHistCoeffs`.
- `Trial.toStructure` / `fromStructure` (serialization round-trip),
  `Trial.getNumHist`, `Trial.findMinSampleRate`.
- Various `SignalObj` plotting/variability methods (`plotVariability`,
  `plotAllVariability`, `alignToMax`, `windowedSignal`, etc.).

> **State-space EM is available via `nstat.extras`.** The KF_EM / PP_EM /
> mPPCO_EM families above are not in **core** `nstat.*` (which holds the
> strict MATLAB-parity contract), but functional Python equivalents ship
> in the opt-in [`nstat.extras.em.dynamax_bridge`](docs/extras/em_dynamax.md)
> module: `fit_linear_gaussian_em` (KF_EM), `fit_point_process_em` (PP_EM),
> `fit_hybrid_em` (mPPCO_EM), plus `cmgf_poisson_filter` /
> `cmgf_poisson_smoother` for point-process inference on a known model.
> To check fit quality, use `point_process_predictive_ll` /
> `hybrid_predictive_ll` — a true held-out predictive log-likelihood
> (pure NumPy, no dynamax needed), **not** the trainers' surrogate
> `marginal_log_likelihoods` trace.
> Install with `pip install nstat-toolbox[dynamax]`. These are
> independent reimplementations (Smith & Brown 2003 PPLDS algorithm +
> Dynamax primitives), not bit-exact MATLAB ports — see the help file
> for the parity and weak-observability caveats.

**If an agent needs one of these, raise the gap explicitly — do not silently
substitute a related Python method.**  For state-space EM specifically,
point users at the `nstat.extras.em` equivalents above.

### 5.2 Behavioral differences vs MATLAB

Documented in [AUDIT_REPORT.md](AUDIT_REPORT.md) §4. The substantive ones:

| Class | Difference |
|---|---|
| `Analysis` | GLM solver is Newton-Raphson with L2=1e-6 regularization; MATLAB uses CG. Coefficients agree to ~1e-6 but are not bit-identical. |
| `DecodingAlgorithms.kalman_fixedIntervalSmoother` | Python uses smoother-index extraction (approximation); MATLAB uses exact state augmentation. Differs at intermediate lags. |
| `DecodingAlgorithms.ComputeStimulusCIs` | Python public path uses Gaussian approximation; MATLAB uses Monte Carlo. For Monte Carlo, call `_ComputeStimulusCIs_MC` explicitly. |
| `DecodingAlgorithms.PPHybridFilter` | Python delegates to the linear version; **does not support nonlinear CIF models** in hybrid decoding. |
| `SignalObj` arithmetic | Python does NOT auto-align time grids via `makeCompatible`; raises `ValueError` if grids differ. Users must `resample()` manually. |
| `nspikeTrain` burst stats | Burst statistics go stale after `setMinTime`/`setMaxTime` in Python (cache invalidation gap). |
| `SpikeTrainCollection.psthBars` | Uses a deterministic smoothing fallback, not the MATLAB BARS package. |

### 5.3 Known Python-side bugs / footguns (status as of v0.3.0+post)

Most of the audit-reported bugs have been **fixed** in this branch; the
list below records the current state.

**Fixed:**

- ✅ `SpikeTrainCollection.toSpikeTrain` maxTime now sums per-train
  durations (was: `maxTime * len(selector)`).  Homogeneous-collection
  behavior unchanged.
- ✅ `SpikeTrainCollection.getNST` now copies-then-resamples (non-destructive).
- ✅ `SpikeTrainCollection.addSingleSpikeToColl` now amortized O(1) per add
  (was: O(n²) — resampled all existing trains on every append).
- ✅ `Analysis.run_analysis_for_neuron` exception handling tightened:
  KS / validation-lambda failures now emit a `RuntimeWarning` instead of
  silently swallowing all `Exception` subclasses.
- ✅ `Analysis.GLMFit` now returns a **`GLMFitResult` dataclass** with
  named fields *and* tuple-unpacking back-compat
  (`lambda_signal, b, dev, stats, AIC, BIC, logLL, distribution = GLMFit(...)`
  still works).  A true Bernoulli/Poisson log-likelihood is exposed at
  `result.loglik` (and `stats["loglik"]`).  The legacy MATLAB-style
  hybrid is retained at `result.logLL` and `stats["matlab_logLL"]`.
- ✅ `FitResult` lambda aliases: the 9 historical aliases
  (`lambda_obj`, `lambda_model`, `lambda_result`, `lambdaObj`,
  `lambdaCov`, `lambda_sig`, `lambda_data`, `lambda_values`, `lambda_time`,
  `lambda_rate`) now emit `DeprecationWarning`.  Use `lambda_signal`
  (canonical) or `lambdaSignal` (MATLAB-style alias, no warning).
- ✅ `matplotlib.use("Agg")` is **no longer** called at import time;
  user-chosen backends survive `import nstat`.
- ✅ The three previously broken examples (`basic_data_workflow.py`,
  `fit_poisson_glm.py`, `simulate_population_psth.py`) now work —
  `simulate_cif_from_stimulus` is implemented in `nstat.simulation`.
- ✅ `ensure_example_data` now respects `NSTAT_OFFLINE=1` and emits a
  helpful error pointing at `nstat-install --download-example-data always`.

**Verified false alarms (no bug):**

- `SignalObj.setMaxTime` off-by-one: verified inclusive of the boundary
  sample, symmetric with `setMinTime`.
- `History.toFilter` `start_sample + 1`: verified as the intentional
  leading-zero filter-coefficient offset.
- `nspikeTrain.computeStatistics` "double +1" on `avgSpikesPerBurst`:
  this is the MATLAB convention (gold fixture confirms `avg = mean(counts) + 1`).
  Documented inline.

**Still present:**

- **`np.random.rand` (legacy API)** is used in `nstat.trial:1865`;
  reproducibility via `np.random.default_rng(seed)` is not yet plumbed.

### 5.4 Module layout (post-refactor)

- `nstat/core.py` (~2,070 lines) hosts `SignalObj` + `Covariate`.  The
  `nspikeTrain` class was extracted to `nstat/_spike_train_impl.py`
  (private module).  `from nstat.core import nspikeTrain` and
  `from nstat.nspikeTrain import nspikeTrain` continue to work.
- `nstat/trial.py` (~2,845 lines) hosts `CovariateCollection`,
  `SpikeTrainCollection`, and `Trial`.  `TrialConfig`/`ConfigCollection`
  were extracted to `nstat/_trial_config_impl.py`.  All legacy import
  paths continue to work.
- `Analysis.GLMFit` now returns `GLMFitResult` (a dataclass with `__iter__`)
  — see Recipe B.

### 5.5 Architectural / structural notes

- **MATLAB-style import shims** exist (e.g., `from nstat.SignalObj import
  SignalObj`); both shim paths and canonical paths work. Prefer canonical
  paths from `nstat/__init__.py` for new code.
- **`core.py` (~2,074 lines) and `trial.py` (~2,849 lines)** still
  host several classes each, but the largest single classes (`nspikeTrain`,
  `TrialConfig`/`ConfigCollection`) were extracted to private impl modules
  in v0.3.1 — see §5.4 above.  Further splits may land in v0.4; do not
  rely on internal file layout.
- **Two parallel install entry points** exist: `nstat.install.main` and
  `nstat.nstat_install.nSTAT_Install`. Prefer the `nstat-install` CLI.
- **Data sourcing**: paper-example data is on figshare (DOI
  10.6084/m9.figshare.4834640.v3, ~150 MB). It is **not** in the git repo.
  `ensure_example_data(download=True)` fetches it; `download=False` raises
  if absent. There is no `--offline` flag yet — agents in offline contexts
  should set `ensure_example_data(download=False)` and handle the
  `DataNotFoundError`.

### 5.6 What the package is NOT (and where to look instead)

For each scope gap, the recommended ecosystem peer is named.  Many are
wired up via opt-in bridges under `nstat.extras` — see the "Related
Python projects" table in `README.md` for install commands.

- **Not** a real-time decoding pipeline.  Decoding methods are
  offline/batch.
- **Not** a deep-learning toolkit. There are no neural-network models;
  fits are GLM/state-space classical statistics.  Bridge to PyTorch
  decoders is planned (`nstat.extras.deep_learning`, v0.4+).
- **Not** a spike-sorting toolkit.  Spike times are assumed to be
  pre-sorted.  Use [SpikeInterface](https://github.com/SpikeInterface/spikeinterface)
  upstream; pipe its output through `nstat.extras.interop.neo`.
- **Not** an LFP/EEG signal-processing library on its own.  `SignalObj`
  has periodogram / spectrogram methods, but for serious spectral
  analysis use [nitime](https://github.com/nipy/nitime) (multitaper),
  [MNE](https://mne.tools) (time-frequency), or
  [ssqueezepy](https://github.com/OverLordGoldDragon/ssqueezepy)
  (synchrosqueeze).  A [mne-connectivity](https://github.com/mne-tools/mne-connectivity)
  bridge is preferable over `spectral_connectivity` due to GPL-3
  incompatibility (`parity/integration_opportunities.md` documents
  the license analysis).
- **Not** an NWB / Neo / pynapple reader.  Use the opt-in bridges:
  `nstat.extras.interop.nwb`, `.neo`, `.pynapple`.
- **Not** a spike-train distance-metric library.  Use the opt-in
  `nstat.extras.metrics.spike_distances` wrapper around PySpike
  (ISI / SPIKE / SPIKE-synchronization).
- **Not** a state-space EM toolbox.  The MATLAB nSTAT `KF_EM` /
  `PP_EM` / `mPPCO_EM` families are unported; use
  [Dynamax](https://github.com/probml/dynamax) instead (planned
  bridge `nstat.extras.em.dynamax`).
- **Not** a spatial / spatiotemporal point-process estimator in core.
  The Python-only `nstat.extras.spatial` module adds LGCP rate maps
  (Laplace), inhomogeneous second-order goodness-of-fit, and the
  discrete-time-rescaling KS correction (pure NumPy/SciPy core; optional
  `tick` / `DPPy` / `gpflow` bridges via `[hawkes]` / `[dpp]` /
  `[spatial-gp]`).  Tensor-product B-spline log-rate bases
  (`bspline_basis_1d`, `bspline_basis_2d`, `BSplineBasis2D`; de Boor 1978,
  Eilers-Marx 1996) produce a design matrix that drops straight into
  `nstat.glm.fit_poisson_glm` as the `x` argument, and the P-spline
  second-difference penalty is available via `BSplineBasis2D.gram()`.
  The basis-projected `lgcp_fit_glm(points, domain, basis, prior)`
  (Diggle-Moraga-Rowlingson-Taylor 2013; Wood 2017 *GAMs*) fits an LGCP
  through penalized Poisson IRLS on the B-spline coefficients — pair it
  with `MaternPrior(nu, length_scale, marginal_var)` (the GP prior
  evaluated at the basis' Greville abscissae) and prefer it over the
  dense per-cell `lgcp_fit` when the grid is large (`G >= 50`), since
  the cubic cost scales with the basis dimension `K` rather than the
  cell count `G*G`.
  The `pair_correlation` / `k_inhom` / `l_function` estimators take an
  `edge_correction` keyword (default `"epanechnikov"`; also `"isotropic"`
  for Ripley 1976/1977, `"translation"` for Ohser 1983, `"border"` for
  Baddeley-Rubak-Turner 2015).  The same module also exposes pure-NumPy
  spatiotemporal-wave diagnostics — `bartlett_spectrum` (the frequency
  × wave-vector spectrum of a Hawkes triggering matrix),
  `reconstruct_kernel` (the parametric exponential-kernel
  reconstruction), and `detect_wave_peaks` returning a
  `WaveAnalysisResult` with `(freq, kx, ky, power, speed, direction)`
  for each accepted peak.  No MATLAB counterpart, so no
  `parity/manifest.yml` entry.  See
  [`docs/extras/spatial_point_processes.md`](docs/extras/spatial_point_processes.md).
  The per-channel discrete-time test `multivariate_time_rescaling` and
  the population coupling test `nstat.population_time_rescale` (Tao
  et al. 2018) compose via `multivariate_gof_with_coupling`, which
  runs both on the same data and returns a `CoupledMarkedGOFResult`.
- **Not** distributed.  All routines are single-process NumPy / SciPy.

---

## 6. Data flow patterns

### Typical workflow

```
spike times (np.ndarray, seconds)
        │
        ▼
   nspikeTrain ────► nstColl  ─┐
                                ├─► Trial ──► Analysis.GLMFit ──► FitResult
   covariate arrays             │                                       │
        │                       │                                       ▼
        ▼                       │                              FitResSummary
   Signal/Covariate ──► CovColl ┘                                  (aggregate)
        │
        ▼
   TrialConfig ──► ConfigColl
```

### Decoding workflow

```
FitResult (per cell)  ─────► DecodingAlgorithms.PPDecodeFilterLinear
spike trains (ensemble) ───►        │
state model (A, Q)        ───►      ▼
                              decoded state x(t), CI W(t)
```

---

## 7. Conventions and gotchas

- **Class names use MATLAB-style camelCase** (`nspikeTrain`, `nstColl`,
  `CovColl`) — the lowercase initials of `nspikeTrain` / `nstColl` are
  intentional and preserved.
- **Method names** also follow MATLAB style (`GLMFit`, `computeKSStats`,
  `setMinTime`). Python-style aliases exist in some cases but not all.
- **Time axis**: all time vectors are in **seconds**. `sampleRate` is in Hz.
- **Spike times** are stored in seconds and **sorted on construction**.
- **`MATLAB-colon equivalence`**: when reproducing MATLAB `start:step:stop`,
  use the `_matlab_colon` pattern (compute element count from
  `floor((stop-start)/step)+1`, then `start + np.arange(n)*step`).
  `np.arange` accumulates float error and can produce off-by-one length.
- **Figures**: `apply_plot_style(fig, style="modern")` mutates the figure
  in place; `apply_plot_style(fig, style="legacy")` is a no-op. The
  paper-example scripts default to `legacy` to preserve strict figure
  parity; pass `--plot-style modern` to opt into readability formatting.
- **The figure tree is consolidated under `docs/figures/`**. The legacy
  `examples/paper/figures/` duplicate tree was removed — paper scripts
  now write only to `docs/figures/exampleNN/`.
- **`__init__.py` does some `__class__` patching** to allow
  `nstat.SignalObj.SignalObj` to coexist with `nstat.SignalObj`. This is
  invisible at runtime but may confuse static analyzers (mypy/pyright).
- **CI is at `.github/workflows/ci.yml`**; the `paper-gallery-artifacts`
  job enforces that the committed PNG set matches `examples/paper/manifest.yml`.

---

## 7.5 Keeping `README.md` current

The README is the user-facing landing page on GitHub and PyPI.  When
you commit anything that touches the surface README references, verify
the README is still accurate **before** you commit.

**Triggers** — if you change any of these, re-read `README.md`:

- ANY file path under `docs/`, `examples/`, `notebooks/`
- Anything in `nstat/__all__` (rename / add / remove)
- `pyproject.toml` (version, dependencies, entry points)
- `CITATION.cff`
- ANY image under `docs/figures/`
- A workflow name in `.github/workflows/` (the CI badge URL embeds it)

**Local check** (one command, ~1 second):

```bash
make readme-check    # or: python tools/check_readme_links.py
```

The checker validates three things:

1. Every Markdown intra-repo link (`[text](path)`) resolves to an
   existing file.
2. Every Markdown image (`![alt](path)`) resolves.
3. Every `from nstat import X` / `import nstat.X` inside a fenced
   ```` ```python ```` block imports cleanly.

**CI enforcement.** Two workflows gate this:

- `.github/workflows/readme-check.yml::readme-intra` — **hard gate**.
  Runs the same script.  Blocks PRs that introduce broken intra-repo
  links, missing images, or stale imports.
- `.github/workflows/readme-check.yml::readme-external` — **advisory**.
  Uses `lychee` to check external URLs (PyPI, GitHub, DOI, lab
  websites).  Failures show as yellow check-runs but do NOT block —
  external links fail for transient reasons (rate limits, CDN hiccups).

**What the checker does NOT catch.** Section anchors (`[foo](file.md#section)`)
— GitHub's heading-slug algorithm is implementation-specific, and the
benefit-to-flakiness ratio of validating them is poor.  If you rename
a section, grep for the old anchor manually:
`grep -rn "#old-anchor-name" *.md docs/`.

---

## 7.6 Keeping helpfiles current

The "helpfiles" of `nstat-python` (the Python analogue of MATLAB's
`helpfiles/*.m`) are the hand-maintained documentation surfaces that
explain how to *use* the package — not just docstrings, which are tied
to symbols.  These can silently lag the public API when contributors
add new symbols.

**Two enforcement layers:**

1. **Sphinx autosummary** (zero-maintenance).  `docs/api.rst` is now
   auto-generated from `nstat.__all__` (and `nstat.extras.__all__` when
   present) by the `sphinx.ext.autosummary` extension.  Adding a name
   to `__all__` automatically makes it appear in the API reference on
   the next docs build.  No manual update needed.

2. **`tools/check_helpfile_freshness.py`** (catches the rest).  For
   every symbol in `nstat.__all__` (and `nstat.extras.__all__`):
   - Asserts the symbol name appears in `AGENT_GUIDE.md` (any prose
     context).
   - If the symbol is a class (per `inspect.isclass`), asserts it also
     appears in `docs/ClassDefinitions.md`.

**Out of scope for the checker:**

- `PORTING_MAP.md` — frozen snapshot; only updated on genuine
  MATLAB↔Python parity refreshes, not on Python-only additions.
- `notebooks/` — tracked separately by `parity/notebook_fidelity.yml`.

**Local check** (~1 second):

```bash
make helpfile-check
# or:
python tools/check_helpfile_freshness.py
```

**CI enforcement.**  `.github/workflows/helpfile-check.yml` runs the
same checker on every PR that touches `nstat/__init__.py`, `nstat/**/*.py`,
`docs/`, or `AGENT_GUIDE.md`.  Hard gate — blocks PRs that introduce
undocumented `__all__` additions.

---

## 7.7 Core vs `nstat.extras` — where new code goes

The package has two namespaces with different stability contracts:

- **`nstat.*`** (core) — MATLAB-parity contract.  Stable.  Every symbol
  here mirrors something in upstream MATLAB nSTAT.  Removals / renames
  require a major-version bump.
- **`nstat.extras.*`** — Python-only extensions.  Opt-in.  Free to evolve.
  Minor-version releases may add / rename / remove extras symbols.

**Decision rule for new code:**

| If the feature... | It goes in... |
|---|---|
| Exists in MATLAB nSTAT (`.m` source file) | `nstat/` (core) |
| Has a `parity/manifest.yml` entry | `nstat/` (core) |
| Is Python-only with no MATLAB counterpart | `nstat/extras/` |
| Depends on libraries outside core deps (PyTorch, SpikeInterface, MNE, Neo, …) | `nstat/extras/` |
| Uses Pythonic snake_case where the MATLAB-style would clash | `nstat/extras/` |
| Is experimental — API may break across minor releases | `nstat/extras/` |
| Wraps an external library to make it work with `nstat.*` | `nstat/extras/` |

**Examples:**

- `LinearCIF` → **core**.  Mirrors MATLAB v1.4.0's `LinearCIF.m`.
- A hypothetical `nstat.extras.spikeinterface_bridge` → **extras**.  No
  MATLAB counterpart; depends on the SpikeInterface package.
- A new Pillow raised-cosine basis (per `parity/integration_opportunities.md`)
  → **extras** initially; could later migrate to core if MATLAB adopts
  the same basis family.
- A PyTorch-based deep-learning decoder bridge → **extras**.  No MATLAB
  counterpart; heavy PyTorch dependency.

**Optional dependencies for extras.**  Each `nstat.extras.X` module
declares its optional dep in `pyproject.toml` under
`[project.optional-dependencies]`.  Install via:

```bash
pip install nstat-toolbox[spikeinterface]
pip install nstat-toolbox[all-extras]   # everything at once
```

Each extras module should raise a clear, actionable `ImportError` at
import time when its optional dependency is missing.

**Independence.**  The "no MATLAB-repo coupling" rule applies equally to
`nstat.extras.*`.  Extras may depend on any *Python* package but must
not introduce runtime dependencies on the MATLAB nSTAT repository
beyond the sanctioned `matlab.engine` bridge in `nstat.matlab_engine`.

---

## 8. Where to look when stuck

| Question | First place to check |
|---|---|
| Does method X exist? | `python -c "import nstat; help(nstat.ClassName)"` |
| Is method X ported from MATLAB? | [PORTING_MAP.md](PORTING_MAP.md) |
| Is there a known bug? | [AUDIT_REPORT.md](AUDIT_REPORT.md) §1–2 |
| Does my output match MATLAB? | [AUDIT_REPORT.md](AUDIT_REPORT.md) §4 + [parity/report.md](parity/report.md) |
| How is class X used? | `examples/paper/example0X_*.py`, [docs/PaperOverview.md](docs/PaperOverview.md), [docs/ClassDefinitions.md](docs/ClassDefinitions.md) |
| Where is the data? | `nstat-install --download-example-data always`, then `ensure_example_data()` |
| What changed lately? | `git log --oneline -20`, or [parity/report.md](parity/report.md) generation date |
| Where do figures go? | `docs/figures/exampleNN/*.png` (canonical) |
| How do I run tests? | `pytest -q`, or `pytest -q -k "test_repo_layout or test_api_surface"` for smoke |

---

## 9. Sanity checks before reporting results

When an agent uses nSTAT to produce a scientific result, verify:

1. **Sample rate consistency** — every `nspikeTrain`, `Covariate`, and
   `Trial` should agree on `sampleRate`. Mixing rates triggers automatic
   resampling that may not be what you want (see §5.3 footgun on `getNST`).
2. **Time bounds** — call `getTime()` / `nspikeTrain.spikeTimes.min()` to
   confirm `minTime`/`maxTime` cover your analysis window. The
   `setMaxTime` off-by-one (§5.3) means you may be dropping the boundary
   sample.
3. **KS goodness-of-fit** — for GLMs, always call
   `fit_result.computeKSStats()` and verify the KS statistic is below the
   95% CI band. A passing AIC/BIC alone is not sufficient.
4. **Lambda signal sanity** — `fit_result.lambda_signal.getData()` should
   be strictly positive. If it isn't, the Newton iteration likely failed;
   check `fit_result.b` for `NaN`.
5. **For decoding**: confirm the state covariance `W` is positive
   semi-definite at each timestep. Negative diagonals indicate numerical
   blow-up in the PPAF updates.
6. **Reproducibility** — runs are NOT seeded by default. For reproducible
   simulations, set `np.random.seed()` BEFORE calling any nSTAT simulator
   (the package uses legacy NumPy RNG, not `default_rng`).

---

## 10. When in doubt

- The README at the repo root is the canonical user-facing introduction.
- The paper at [DOI:10.1016/j.jneumeth.2012.08.009](https://doi.org/10.1016/j.jneumeth.2012.08.009)
  describes the underlying methods.
- The MATLAB reference toolbox is at https://github.com/cajigaslab/nSTAT —
  if behaviour disagrees with this Python port, MATLAB is the gold
  standard except for the cases listed in [AUDIT_REPORT.md](AUDIT_REPORT.md)
  §1 (MATLAB bugs fixed in Python).
- The lab websites:
  - https://www.neurostat.mit.edu (Neuroscience Statistics Research Lab)
  - https://www.med.upenn.edu/cajigaslab/ (RESToRe Lab)

---

*This file is intended to be loaded into the context of an AI agent
before it begins work with `nstat-python`. Keep it under 600 lines so it
fits in a typical context window alongside actual code.*
