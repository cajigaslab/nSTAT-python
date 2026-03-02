# Tier-1 Port Backlog

Generated from `parity/parity_gap_report.json`.

| Priority | MATLAB Class | Missing Methods | Coverage | Next Implementation Focus |
|---|---:|---:|---:|---|
| P2 | `DecodingAlgorithms` | 0 | 1.000 | n/a |
| P2 | `Analysis` | 0 | 1.000 | n/a |
| P1 | `Trial` | 32 | 0.529 | flattenCovMask, flattenMask, getAllLabels, getCovSelectorFromMask, getEnsCovLabels, getEnsCovLabelsFromMask |
| P2 | `CovColl` | 0 | 1.000 | n/a |
| P2 | `nstColl` | 0 | 1.000 | n/a |
| P2 | `FitResult` | 0 | 1.000 | n/a |
| P2 | `FitResSummary` | 0 | 1.000 | n/a |

## Implementation Order
1. `DecodingAlgorithms` and `Analysis` (numerical behavior parity)
2. `Trial` / `CovColl` / `nstColl` (data plumbing parity)
3. `FitResult` / `FitResSummary` (model diagnostics parity)

## Acceptance for Each Tier-1 Class
- Implement method aliases in `nstat.compat.matlab`.
- Add numerical fixture checks when outputs are deterministic.
- Add unit tests for method signatures and key behavior.
