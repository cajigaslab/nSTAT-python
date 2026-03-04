# Method Closure Sprint Backlog

This sprint backlog targets methods that are probe-verified but not yet explicitly covered by behavior contracts.

## Functional Summary
- Total methods: `502`
- Contract-explicit verified methods: `481`
- Probe-verified methods: `0`
- Eligible verified ratio: `1.000`
- Excluded methods: `21`

## Priority Class Queue
| Class | Probe-verified | Contract-verified | Probe-only methods |
|---|---:|---:|---:|
| SignalObj | 0 | 98 | 0 |
| Trial | 0 | 68 | 0 |
| CovColl | 0 | 55 | 0 |
| nstColl | 0 | 53 | 0 |
| FitResult | 0 | 33 | 0 |
| FitResSummary | 0 | 30 | 0 |
| nspikeTrain | 0 | 29 | 0 |
| DecodingAlgorithms | 0 | 25 | 0 |

## Sprint Work Packages

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

