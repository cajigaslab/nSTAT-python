# Method Closure Sprint Backlog

This sprint backlog targets methods that are probe-verified but not yet explicitly covered by behavior contracts.

## Functional Summary
- Total methods: `501`
- Contract-explicit verified methods: `391`
- Probe-verified methods: `89`
- Eligible verified ratio: `1.000`
- Excluded methods: `21`

## Priority Class Queue
| Class | Probe-verified | Contract-verified | Probe-only methods |
|---|---:|---:|---:|
| CovColl | 27 | 55 | 27 |
| FitResult | 16 | 33 | 16 |
| nspikeTrain | 16 | 29 | 16 |
| FitResSummary | 14 | 30 | 14 |
| Analysis | 7 | 22 | 7 |
| Covariate | 3 | 14 | 3 |
| CIF | 2 | 21 | 2 |
| nstColl | 1 | 53 | 1 |

## Sprint Work Packages

### CovColl
- Goal: Convert probe-only functional verification to explicit behavior contracts.
- Candidate methods:
  - `CovColl`
  - `copy`
  - `dataToStructure`
  - `enforceSampleRate`
  - `getCov`
  - `getCovDataMask`
  - `getCovLabelsFromMask`
  - `getCovMaskFromSelector`
  - `isCovMaskSet`
  - `isaSelectorCell`
  - `maskAwayAllExcept`
  - `maskAwayCov`
  - `maskAwayOnlyCov`
  - `nActCovar`
  - `plot`
  - `removeCovariate`
  - `removeFromColl`
  - `resample`
  - `resetCovShift`
  - `resetMask`
  - `... (+7 additional methods)`

### FitResult
- Goal: Convert probe-only functional verification to explicit behavior contracts.
- Candidate methods:
  - `FitResult`
  - `KSPlot`
  - `addParamsToFit`
  - `getHistCoeffs`
  - `getHistIndex`
  - `mapCovLabelsToUniqueLabels`
  - `mergeResults`
  - `plotCoeffs`
  - `plotCoeffsWithoutHistory`
  - `plotHistCoeffs`
  - `plotInvGausTrans`
  - `plotResidual`
  - `plotResults`
  - `plotSeqCorr`
  - `plotValidation`
  - `setNeuronName`

### nspikeTrain
- Goal: Convert probe-only functional verification to explicit behavior contracts.
- Candidate methods:
  - `clearSigRep`
  - `computeStatistics`
  - `getLStatistic`
  - `partitionNST`
  - `plot`
  - `plotExponentialFit`
  - `plotISIHistogram`
  - `plotISISpectrumFunction`
  - `plotJointISIHistogram`
  - `plotProbPlot`
  - `resample`
  - `restoreToOriginal`
  - `setMER`
  - `setMaxTime`
  - `setMinTime`
  - `setName`

### FitResSummary
- Goal: Convert probe-only functional verification to explicit behavior contracts.
- Candidate methods:
  - `boxPlot`
  - `getHistIndex`
  - `plot2dCoeffSummary`
  - `plot3dCoeffSummary`
  - `plotAIC`
  - `plotAllCoeffs`
  - `plotBIC`
  - `plotCoeffsWithoutHistory`
  - `plotHistCoeffs`
  - `plotIC`
  - `plotKSSummary`
  - `plotResidualSummary`
  - `plotSummary`
  - `plotlogLL`

### Analysis
- Goal: Convert probe-only functional verification to explicit behavior contracts.
- Candidate methods:
  - `KSPlot`
  - `bnlrCG`
  - `compHistEnsCoeff`
  - `computeHistLagForAll`
  - `plotCoeffs`
  - `plotFitResidual`
  - `plotInvGausTrans`

### Covariate
- Goal: Convert probe-only functional verification to explicit behavior contracts.
- Candidate methods:
  - `Covariate`
  - `plot`
  - `toStructure`

### CIF
- Goal: Convert probe-only functional verification to explicit behavior contracts.
- Candidate methods:
  - `simulateCIF`
  - `simulateCIFByThinning`

### nstColl
- Goal: Convert probe-only functional verification to explicit behavior contracts.
- Candidate methods:
  - `nstColl`

## Excluded MATLAB Stub Methods
- `DecodingAlgorithms`
  - `KF_ComputeParamStandardErrors`
  - `KF_EM`
  - `KF_EMCreateConstraints`
  - `KF_EStep`
  - `KF_MStep`
  - `PPSS_EM`
  - `PPSS_EMFB`
  - `PPSS_EStep`
  - `PPSS_MStep`
  - `PP_ComputeParamStandardErrors`
  - `PP_EM`
  - `PP_EMCreateConstraints`
  - `PP_EStep`
  - `PP_MStep`
  - `estimateInfoMat`
  - `mPPCO_ComputeParamStandardErrors`
  - `mPPCO_EM`
  - `mPPCO_EMCreateConstraints`
  - `mPPCO_EStep`
  - `mPPCO_MStep`
  - `prepareEMResults`

## Exit Criteria
- Each listed method has an explicit behavior contract in parity audit generation.
- New/updated contract tests are added and pass in CI.
- Functional parity summary increases `contract_explicit_verified_methods` and preserves gate pass.

