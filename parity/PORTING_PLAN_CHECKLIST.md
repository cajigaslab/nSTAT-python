# nSTAT MATLAB -> Python Porting Plan Checklist

Status definitions:
- `Not started`: Python class missing or unusable.
- `Partial`: Class exists but method/behavior parity is not complete.
- `Complete`: Class and mapped methods implemented with contract coverage; per-class MATLAB fixture parity loop not fully closed yet.
- `Verified`: Full loop completed for this class (`implementation + MATLAB fixture generator + fixture artifact + pytest parity test + related demo`).

| MATLAB Class | Python Counterpart | Contract Coverage | Status | Notes |
|---|---|---:|---|---|
| `SignalObj` | `nstat.signal.Signal` + `nstat.compat.matlab.SignalObj` | 98/98 | Verified | Added MATLAB fixtures, parity tests, and demo in this sprint. |
| `Covariate` | `nstat.signal.Covariate` + `nstat.compat.matlab.Covariate` | 14/14 | Verified | Added MATLAB fixtures, parity tests, and demo in this sprint. |
| `ConfidenceInterval` | `nstat.confidence.ConfidenceInterval` + `nstat.compat.matlab.ConfidenceInterval` | 5/5 | Verified | Added MATLAB fixtures, parity tests, and demo in this sprint. |
| `Events` | `nstat.events.Events` + `nstat.compat.matlab.Events` | 5/5 | Verified | Added MATLAB fixtures, parity tests, and demo in this sprint. |
| `History` | `nstat.history.HistoryBasis` + `nstat.compat.matlab.History` | 8/8 | Verified | Added MATLAB fixtures, parity tests, and demo in this sprint. |
| `nspikeTrain` | `nstat.spikes.SpikeTrain` + `nstat.compat.matlab.nspikeTrain` | 29/29 | Complete | Needs dedicated class fixture loop. |
| `nstColl` | `nstat.spikes.SpikeTrainCollection` + `nstat.compat.matlab.nstColl` | 53/53 | Complete | Needs dedicated class fixture loop. |
| `CovColl` | `nstat.trial.CovariateCollection` + `nstat.compat.matlab.CovColl` | 55/55 | Complete | Needs dedicated class fixture loop. |
| `TrialConfig` | `nstat.trial.TrialConfig` + `nstat.compat.matlab.TrialConfig` | 6/6 | Complete | Needs dedicated class fixture loop. |
| `ConfigColl` | `nstat.trial.ConfigCollection` + `nstat.compat.matlab.ConfigColl` | 9/9 | Complete | Needs dedicated class fixture loop. |
| `Trial` | `nstat.trial.Trial` + `nstat.compat.matlab.Trial` | 68/68 | Complete | Needs dedicated class fixture loop. |
| `CIF` | `nstat.cif.CIFModel` + `nstat.compat.matlab.CIF` | 21/21 | Complete | Needs dedicated class fixture loop. |
| `Analysis` | `nstat.analysis.Analysis` + `nstat.compat.matlab.Analysis` | 22/22 | Complete | Needs dedicated class fixture loop. |
| `FitResult` | `nstat.fit.FitResult` + `nstat.compat.matlab.FitResult` | 33/33 | Complete | Needs dedicated class fixture loop. |
| `FitResSummary` | `nstat.fit.FitSummary` + `nstat.compat.matlab.FitResSummary` | 30/30 | Complete | Needs dedicated class fixture loop. |
| `DecodingAlgorithms` | `nstat.decoding.DecodingAlgorithms` + `nstat.compat.matlab.DecodingAlgorithms` | 24/45 (21 excluded) | Partial | Remaining excluded methods require MATLAB-grounded closure. |

## Execution Order (next)
1. `nspikeTrain`
2. `nstColl`
3. `CovColl`
4. `TrialConfig`
5. `ConfigColl`
6. `Trial`
7. `CIF`
8. `Analysis`
9. `FitResult`
10. `FitResSummary`
11. `DecodingAlgorithms`
