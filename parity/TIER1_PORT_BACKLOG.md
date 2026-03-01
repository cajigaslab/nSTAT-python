# Tier-1 Port Backlog

Generated from `parity/parity_gap_report.json`.

| Priority | MATLAB Class | Missing Methods | Coverage | Next Implementation Focus |
|---|---:|---:|---:|---|
| P2 | `DecodingAlgorithms` | 0 | 1.000 | n/a |
| P2 | `Analysis` | 15 | 0.318 | KSPlot, bnlrCG, compHistEnsCoeff, compHistEnsCoeffForAll, computeGrangerCausalityMatrix, computeHistLag |
| P1 | `Trial` | 32 | 0.529 | flattenCovMask, flattenMask, getAllLabels, getCovSelectorFromMask, getEnsCovLabels, getEnsCovLabelsFromMask |
| P2 | `CovColl` | 21 | 0.618 | containsChars, covIndFromSelector, dataToStructure, enforceSampleRate, findMaxTime, findMinTime |
| P1 | `nstColl` | 25 | 0.528 | addNeuronNamesToEnsCovColl, areNeighborsSet, enforceSampleRate, ensureConsistancy, estimateVarianceAcrossTrials, findMaxSampleRate |
| P2 | `FitResult` | 23 | 0.303 | KSPlot, addParamsToFit, computePlotParams, getHistCoeffs, getHistIndex, getPlotParams |
| P2 | `FitResSummary` | 17 | 0.433 | getHistCoeffs, getHistIndex, getSigCoeffs, plot2dCoeffSummary, plot3dCoeffSummary, plotAIC |

## Implementation Order
1. `DecodingAlgorithms` and `Analysis` (numerical behavior parity)
2. `Trial` / `CovColl` / `nstColl` (data plumbing parity)
3. `FitResult` / `FitResSummary` (model diagnostics parity)

## Acceptance for Each Tier-1 Class
- Implement method aliases in `nstat.compat.matlab`.
- Add numerical fixture checks when outputs are deterministic.
- Add unit tests for method signatures and key behavior.
