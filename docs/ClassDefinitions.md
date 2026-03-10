# Class Definitions

The Python port preserves the MATLAB-facing class names as canonical imports or
compatibility adapters. Each class below lists its key public methods grouped by
category.

## Signal and Covariate Primitives

### `SignalObj` (`nstat.SignalObj`)

Primary notebook: `notebooks/SignalObjExamples.ipynb`

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

Primary notebook: `notebooks/CovariateExamples.ipynb`

Inherits from `SignalObj`. Adds confidence interval support:
`setConfInterval`, `mu`, `sigma`

### `ConfidenceInterval` (`nstat.ConfidenceInterval`)

Primary notebook: `notebooks/ConfidenceIntervalOverview.ipynb`

### `CovColl` (`nstat.CovColl`)

Primary notebook: `notebooks/CovCollExamples.ipynb`

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

Primary notebook: `notebooks/nSpikeTrainExamples.ipynb`

### `nstColl` (`nstat.nstColl`)

Primary notebook: `notebooks/nstCollExamples.ipynb`

**Collection management**:
`addSingleSpikeToColl`, `addToColl`, `merge`, `length`, `get_nst`, `getNST`,
`getNSTnames`, `getUniqueNSTnames`, `toSpikeTrain`

**Time operations**:
`shiftTime`, `setMinTime`, `setMaxTime`

**SSGLM (state-space GLM)**:
`ssglm`, `ssglmFB`

**Basis generation**:
`generateUnitImpulseBasis`

### `History` (`nstat.History`)

Primary notebook: `notebooks/HistoryExamples.ipynb`

### `Events` (`nstat.Events`)

Primary notebook: `notebooks/EventsExamples.ipynb`

---

## Experiment and Configuration Objects

### `Trial` (`nstat.Trial`)

Primary notebook: `notebooks/TrialExamples.ipynb`

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
`shiftCovariates`, `resampleEnsColl`, `restoreToOriginal`, `plot`

### `TrialConfig` (`nstat.TrialConfig`)

Primary notebook: `notebooks/TrialConfigExamples.ipynb`

### `ConfigColl` (`nstat.ConfigColl`)

Primary notebook: `notebooks/ConfigCollExamples.ipynb`

---

## Modeling and Inference

### `CIF` (`nstat.CIF`)

Primary notebook: `notebooks/PPSimExample.ipynb`

Conditional intensity function (CIF) primitives and point-process simulation
via thinning.

### `Analysis` (`nstat.Analysis`)

Primary notebook: `notebooks/AnalysisExamples.ipynb`

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

Primary notebook: `notebooks/FitResultExamples.ipynb`

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

Primary notebook: `notebooks/FitResSummaryExamples.ipynb`

Alias of `FitSummary`. Aggregates multiple `FitResult` objects.

**Information criterion**:
`getDiffAIC`, `getDiffBIC`, `getDifflogLL`

**Coefficient extraction**:
`getCoeffs`, `getHistCoeffs`, `getSigCoeffs`, `binCoeffs`, `setCoeffRange`,
`mapCovLabelsToUniqueLabels`

**Plotting**:
`plotIC`, `plotAIC`, `plotBIC`, `plotlogLL`, `plotResidualSummary`,
`plotSummary`, `boxPlot`

### `DecodingAlgorithms` (`nstat.DecodingAlgorithms`)

Primary notebook: `notebooks/DecodingExample.ipynb`

**Point-process decode filters**:
`PPDecodeFilterLinear`, `PPDecodeFilter`, `PPHybridFilterLinear`,
`ComputeStimulusCIs`

**Kalman and unscented Kalman filters**:
`kalman_filter`, `PP_fixedIntervalSmoother`, `ukf`

**State-space GLM (SSGLM) — EM algorithm**:
`PPSS_EStep`, `PPSS_MStep`, `PPSS_EM`, `PPSS_EMFB`

**SSGLM utilities**:
`estimateInfoMat`, `prepareEMResults`

---

See [Examples](Examples.md) for the full help-style index and
[API Reference](api.rst) for the module layout.
