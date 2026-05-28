# Class Definitions

The Python port preserves the MATLAB-facing class names as canonical imports or
compatibility adapters. Each class below lists its key public methods grouped by
category.

## Signal and Covariate Primitives

### `SignalObj` (`nstat.SignalObj`)

Primary notebook: `../notebooks/SignalObjExamples.ipynb`

**Construction and metadata**:
`copySignal`, `setName`, `setXlabel`, `setYLabel`, `setUnits`, `setXUnits`,
`setYUnits`, `setSampleRate`, `setDataLabels`, `setPlotProps`, `with_metadata`

**Accessors**:
`getTime`, `getData`, `getOriginalData`, `getOrigDataSig`, `getPlotProps`,
`getValueAt`, `getSubSignalFromInd`, `getSubSignalFromNames`, `getSubSignal`,
`findNearestTimeIndex`, `findNearestTimeIndices`, `dataToMatrix`,
`dataToStructure`, `toStructure`, `signalFromStruct`

**Masking**:
`setDataMask`, `setMaskByInd`, `setMaskByLabels`, `setMask`, `resetMask`,
`restoreToOriginal`, `findIndFromDataMask`, `isMaskSet`

**Time windowing**:
`setMinTime`, `setMaxTime`, `getSigInTimeWindow`, `makeCompatible`

**Math and transforms**:
`abs`, `log`, `power`, `sqrt`, `median`, `mode`, `mean`, `std`, `max`, `min`,
`derivative`, `derivativeAt`, `integral`, `resample`, `resampleMe`,
`filter`, `filtfilt`, `merge`

**Shift and alignment**:
`shift`, `shiftMe`, `alignTime`

**Correlation**:
`autocorrelation`, `crosscorrelation`, `xcorr`, `xcov`

**Spectral analysis**:
`periodogram`, `MTMspectrum`, `spectrogram`

**Peak-finding**:
`findPeaks`, `findMaxima`, `findMinima`, `findGlobalPeak`

**Plotting**:
`plot`

### `Covariate` (`nstat.Covariate`)

Primary notebook: `../notebooks/CovariateExamples.ipynb`

Inherits from `SignalObj`. Adds confidence interval support:
`setConfInterval`, `mu`, `sigma`

`nstat.CovariateCollection` is the Pythonic alias for `CovColl` (below).

### `ConfidenceInterval` (`nstat.ConfidenceInterval`)

Primary notebook: `../notebooks/ConfidenceIntervalOverview.ipynb`

### `CovColl` (`nstat.CovColl`)

Primary notebook: `../notebooks/CovCollExamples.ipynb`

**Collection management**:
`add`, `addCovariate`, `addCovCollection`, `addToColl`, `removeCovariate`,
`copy`, `get`, `getCov`, `getCovIndFromName`, `getCovIndicesFromNames`,
`isCovPresent`

**Time and sample rate**:
`findMinTime`, `findMaxTime`, `findMaxSampleRate`, `setMinTime`, `setMaxTime`,
`restrictToTimeWindow`, `setSampleRate`, `resample`, `enforceSampleRate`

**Masking**:
`resetMask`, `getCovDataMask`, `isCovMaskSet`, `flattenCovMask`,
`getSelectorFromMasks`, `generateSelectorCell`, `setMasksFromSelector`,
`setMask`, `nActCovar`, `maskAwayCov`, `maskAwayOnlyCov`, `maskAwayAllExcept`

**Shift and restore**:
`setCovShift`, `resetCovShift`, `restoreToOriginal`

**Data export**:
`getAllCovLabels`, `getCovLabelsFromMask`, `matrixWithTime`, `dataToMatrix`,
`dataToStructure`, `toStructure`, `fromStructure`

---

## Spiking Data Structures

### `nspikeTrain` (`nstat.nspikeTrain`)

Primary notebook: `../notebooks/nSpikeTrainExamples.ipynb`

`nstat.SpikeTrain` is the Pythonic alias for `nspikeTrain`.

### `nstColl` (`nstat.nstColl`)

Primary notebook: `../notebooks/nstCollExamples.ipynb`

`nstat.SpikeTrainCollection` is the Pythonic alias for `nstColl`.

**Collection management**:
`addSingleSpikeToColl`, `addToColl`, `merge`, `length`, `get_nst`, `getNST`,
`getNSTnames`, `getUniqueNSTnames`, `toSpikeTrain`

**Time operations**:
`shiftTime`, `setMinTime`, `setMaxTime`

**Data export**:
`dataToMatrix`, `resample`

**PSTH**:
`psth`, `psthGLM`, `estimateVarianceAcrossTrials`, `psthBars`

**SSGLM (state-space GLM)**:
`ssglm`, `ssglmFB`

**Basis generation**:
`generateUnitImpulseBasis`

**Plotting**:
`plot`

### `History` (`nstat.History`)

`nstat.HistoryBasis` is a companion dataclass exposing the basis-function
parameters used to build self-history kernels (window times, basis
family, raised-cosine vs unit-impulse choices).



Primary notebook: `../notebooks/HistoryExamples.ipynb`

### `Events` (`nstat.Events`)

Primary notebook: `../notebooks/EventsExamples.ipynb`

---

## Experiment and Configuration Objects

### `Trial` (`nstat.Trial`)

Primary notebook: `../notebooks/TrialExamples.ipynb`

**Partitioning**:
`getTrialPartition`, `setTrialPartition`, `setTrialTimesFor`,
`updateTimePartitions`

**Time and sample rate**:
`setMinTime`, `setMaxTime`, `setSampleRate`, `resample`,
`makeConsistentSampleRate`, `makeConsistentTime`, `isSampleRateConsistent`,
`findMaxSampleRate`, `findMinTime`, `findMaxTime`

**Covariate and neuron masks**:
`setCovMask`, `resetCovMask`, `setNeuronMask`, `resetNeuronMask`,
`setNeighbors`, `setHistory`, `resetHistory`, `setEnsCovHist`, `setEnsCovMask`,
`resetEnsCovMask`, `isNeuronMaskSet`, `isCovMaskSet`, `isMaskSet`, `isHistSet`,
`isEnsCovHistSet`

**Data access**:
`addCov`, `removeCov`, `getSpikeVector`, `get_covariate_matrix`,
`getDesignMatrix`, `getHistForNeurons`, `getHistMatrices`,
`getEnsembleNeuronCovariates`, `getEnsCovMatrix`, `getNeuronIndFromMask`,
`getNumUniqueNeurons`, `getNeuronNames`, `getUniqueNeuronNames`,
`getNeuronIndFromName`, `getAllCovLabels`, `getCovLabelsFromMask`,
`getHistLabels`, `getEnsCovLabels`, `getLabelsFromMask`, `flattenCovMask`,
`flattenMask`

**Utilities**:
`shiftCovariates`, `resampleEnsColl`, `restoreToOriginal`, `getAllLabels`,
`plot`

### `TrialConfig` (`nstat.TrialConfig`)

Primary notebook: `../notebooks/TrialConfigExamples.ipynb`

### `ConfigColl` (`nstat.ConfigColl`)

Primary notebook: `../notebooks/ConfigCollExamples.ipynb`

`nstat.ConfigCollection` is the Pythonic alias for `ConfigColl`.

---

## Modeling and Inference

### `CIF` (`nstat.CIF`)

Primary notebook: `../notebooks/PPSimExample.ipynb`

Conditional intensity function (CIF) primitives and point-process simulation
via thinning.

`nstat.CIFModel` is a lightweight dataclass capturing the parameters of
a fitted CIF — coefficients, link function, history kernel.  Used as a
return type by some Analysis paths.

### `LinearCIF` (`nstat.LinearCIF`)

Closed-form, sympy-free CIF for the two canonical link cases (Poisson
log-link, binomial logit-link).  Ported from MATLAB `LinearCIF.m` (added
upstream in v1.4.0).  Drop-in compatible with `CIF` for the 5 eval
methods used by `DecodingAlgorithms.PPDecode_update`:
`evalLambdaDelta`, `evalGradient`, `evalGradientLog`, `evalJacobian`,
`evalJacobianLog`.  See `Decoding the Brain` §4.B.7.4 for the
derivation.

### `Analysis` (`nstat.Analysis`)

Primary notebook: `../notebooks/AnalysisExamples.ipynb`

**Core fitting**:
`GLMFit`, `run_analysis_for_neuron`, `run_analysis_for_all_neurons`,
`RunAnalysisForNeuron`, `RunAnalysisForAllNeurons`

**PSTH**:
`psth`

**Diagnostics**:
`computeKSStats`, `computeInvGausTrans`, `computeFitResidual`,
`KSPlot`, `plotFitResidual`, `plotInvGausTrans`, `plotSeqCorr`, `plotCoeffs`

**History and model selection**:
`computeHistLag`, `computeHistLagForAll`, `compHistEnsCoeff`,
`compHistEnsCoeffForAll`

**Network and Granger causality**:
`computeGrangerCausalityMatrix`, `computeNeighbors`, `spikeTrigAvg`

### `FitResult` (`nstat.FitResult`)

Primary notebook: `../notebooks/FitResultExamples.ipynb`

**Coefficient access**:
`getCoeffs`, `getHistCoeffs`, `getCoeffIndex`, `getHistIndex`, `getParam`,
`getCoeffsWithLabels`, `computePlotParams`, `getPlotParams`

**Lambda (conditional intensity) access**:
`lambdaSignal`, `lambda_obj`, `lambda_model`, `lambda_result`, `lambdaObj`,
`lambdaCov`, `lambda_sig`, `lambda_data`, `lambda_values`, `lambda_time`,
`lambda_rate`, `evalLambda`

**Diagnostics and statistics**:
`computeKSStats`, `computeInvGausTrans`, `computeFitResidual`

**Plotting**:
`plotResults`, `KSPlot`, `plotResidual`, `plotInvGausTrans`, `plotSeqCorr`,
`plotCoeffs`, `plotCoeffsWithoutHistory`, `plotHistCoeffs`, `plotValidation`

**Serialization**:
`toStructure`, `fromStructure`, `CellArrayToStructure`, `mergeResults`

### `FitResSummary` (`nstat.FitResSummary`)

Primary notebook: `../notebooks/FitResSummaryExamples.ipynb`

Alias of `FitSummary`. Aggregates multiple `FitResult` objects.

**Information criterion**:
`getDiffAIC`, `getDiffBIC`, `getDifflogLL`

**Coefficient extraction**:
`getCoeffs`, `getHistCoeffs`, `getSigCoeffs`, `binCoeffs`, `setCoeffRange`,
`mapCovLabelsToUniqueLabels`

**Plotting**:
`plotIC`, `plotAIC`, `plotBIC`, `plotlogLL`, `plotResidualSummary`,
`plotSummary`, `boxPlot`, `plotAllCoeffs`, `plot3dCoeffSummary`,
`plot2dCoeffSummary`, `plotKSSummary`

### `PopulationTimeRescaleResult` (`nstat.PopulationTimeRescaleResult`)

Frozen dataclass returned by `population_time_rescale(...)` — the
multivariate (marked) point-process time-rescaling goodness-of-fit of
Tao, Weber, Arai & Eden (2018).  Unlike `FitResult.computeKSStats`
(per-neuron, univariate), it scores a population *jointly* and catches
inter-neuron coupling misfit (e.g. synchronous neurons modeled as
independent).  Python-only extension — no MATLAB `matlab_gold` counterpart.

**Fields**:
- `ground_uniforms` — rescaled ground-process inter-event values on `[0, 1]`.
- `ground_ks_stat`, `ground_ks_pvalue` — KS of the pooled (ground) process
  vs. Uniform(0, 1); detects aggregate-temporal / dependency misfit.
- `mark_chi2_stat`, `mark_chi2_dof`, `mark_chi2_pvalue` — Pearson χ² for
  uniform fill of the marked region `R = {(τ, m): 0 ≤ τ ≤ b(m)}`; detects
  relative-allocation (and, with `n_tau_bins > 1`, within-neuron timing) misfit.
- `expected_counts`, `observed_counts` — per-neuron `∫λ` vs. observed spikes.

### `DecodingAlgorithms` (`nstat.DecodingAlgorithms`)

Primary notebook: `../notebooks/DecodingExample.ipynb`

**Point-process decode filters**:
`PPDecodeFilterLinear`, `PPDecodeFilter`, `PPHybridFilterLinear`,
`ComputeStimulusCIs`, `computeSpikeRateCIs`

**Kalman and unscented Kalman filters**:
`kalman_filter`, `PP_fixedIntervalSmoother`, `ukf`

**Helper methods**:
`PPDecode_predict`, `PPDecode_updateLinear`

**State-space GLM (SSGLM) — EM algorithm**:
`PPSS_EStep`, `PPSS_MStep`, `PPSS_EM`, `PPSS_EMFB`

**SSGLM utilities**:
`estimateInfoMat`, `prepareEMResults`

### `DecoderSuite` (`nstat.DecoderSuite`)

Pythonic wrapper that exposes the most common `DecodingAlgorithms`
static methods through a friendlier object-oriented API.  No MATLAB
counterpart — this is a Python convenience layer over the parity-ported
algorithm namespace.

### `PoissonGLMResult` (`nstat.PoissonGLMResult`)

Dataclass returned by :func:`nstat.fit_poisson_glm`.  Captures the
fitted coefficients, intercept, deviance, iteration count, convergence
status, log-likelihood, and standard errors of a standalone Poisson
GLM fit (separate from `Analysis.GLMFit`, which is wired through the
`Trial` machinery).

---

## Simulation Objects

### `PointProcessSimulation` (`nstat.PointProcessSimulation`)

Result dataclass for :func:`nstat.simulate_point_process`.  Holds the
simulated spike train, the underlying conditional intensity trace, and
the seed / RNG state for reproducibility.

### `NetworkSimulationResult` (`nstat.NetworkSimulationResult`)

Result dataclass for :func:`nstat.simulate_two_neuron_network`.  Captures
both neurons' spike trains and the cross-coupling history that drove
the simulation.

---

See [Examples](Examples.md) for the full help-style index and
[API Reference](api.rst) for the module layout.
