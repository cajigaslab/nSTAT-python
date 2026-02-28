# Tier-1 Port Backlog

Generated from `parity/parity_gap_report.json`.

| Priority | MATLAB Class | Missing Methods | Coverage | Next Implementation Focus |
|---|---:|---:|---:|---|
| P1 | `DecodingAlgorithms` | 38 | 0.156 | KF_ComputeParamStandardErrors, KF_EM, KF_EMCreateConstraints, KF_EStep, KF_MStep, PPDecodeFilter |
| P2 | `Analysis` | 15 | 0.318 | KSPlot, bnlrCG, compHistEnsCoeff, compHistEnsCoeffForAll, computeGrangerCausalityMatrix, computeHistLag |
| P0 | `Trial` | 58 | 0.147 | Trial, addCov, flattenCovMask, flattenMask, fromStructure, getAllLabels |
| P1 | `CovColl` | 41 | 0.255 | CovColl, addCovCellToColl, addCovCollection, addSingleCovToColl, containsChars, covIndFromSelector |
| P1 | `nstColl` | 36 | 0.321 | BinarySigRep, addNeuronNamesToEnsCovColl, areNeighborsSet, enforceSampleRate, ensureConsistancy, estimateVarianceAcrossTrials |
| P1 | `FitResult` | 27 | 0.182 | CellArrayToStructure, FitResult, KSPlot, addParamsToFit, computePlotParams, fromStructure |
| P1 | `FitResSummary` | 26 | 0.133 | FitResSummary, binCoeffs, boxPlot, fromStructure, getCoeffIndex, getCoeffs |

## Implementation Order
1. `DecodingAlgorithms` and `Analysis` (numerical behavior parity)
2. `Trial` / `CovColl` / `nstColl` (data plumbing parity)
3. `FitResult` / `FitResSummary` (model diagnostics parity)

## Acceptance for Each Tier-1 Class
- Implement method aliases in `nstat.compat.matlab`.
- Add numerical fixture checks when outputs are deterministic.
- Add unit tests for method signatures and key behavior.
