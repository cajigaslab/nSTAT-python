# MATLAB -> Python Porting Plan Checklist

This checklist tracks class-level parity status against MATLAB `nSTAT` (source of truth).

Status legend:
- `Not started`: no Python implementation route yet.
- `Partial`: class exists but major API/method parity gaps remain.
- `Complete`: class exists with mapped API and compatibility contracts, pending class-specific MATLAB golden-fixture verification.
- `Verified`: class has the full parity loop (MATLAB fixture generator + fixture artifact + pytest golden-master parity test + demo).

| MATLAB class | Python equivalent | Status | Evidence |
| --- | --- | --- | --- |
| `SignalObj` | `nstat.signal.Signal` + `nstat.compat.matlab.SignalObj` | `Verified` | `tests/test_signalobj_matlab_parity.py`, `matlab/fixture_gen/SignalObj_fixtures.m` |
| `Covariate` | `nstat.signal.Covariate` + `nstat.compat.matlab.Covariate` | `Verified` | `tests/test_covariate_matlab_parity.py`, `matlab/fixture_gen/Covariate_fixtures.m` |
| `ConfidenceInterval` | `nstat.confidence.ConfidenceInterval` + `nstat.compat.matlab.ConfidenceInterval` | `Verified` | `tests/test_confidence_matlab_parity.py`, `matlab/fixture_gen/ConfidenceInterval_fixtures.m` |
| `Events` | `nstat.events.Events` + `nstat.compat.matlab.Events` | `Verified` | `tests/test_events_matlab_parity.py`, `matlab/fixture_gen/Events_fixtures.m` |
| `History` | `nstat.history.HistoryBasis` + `nstat.compat.matlab.History` | `Verified` | `tests/test_history_matlab_parity.py`, `matlab/fixture_gen/History_fixtures.m` |
| `nspikeTrain` | `nstat.spikes.SpikeTrain` + `nstat.compat.matlab.nspikeTrain` | `Verified` | `tests/test_nspiketrain_matlab_parity.py`, `matlab/fixture_gen/nspikeTrain_fixtures.m` |
| `nstColl` | `nstat.spikes.SpikeTrainCollection` + `nstat.compat.matlab.nstColl` | `Verified` | `tests/test_nstcoll_matlab_parity.py`, `matlab/fixture_gen/nstColl_fixtures.m` |
| `CovColl` | `nstat.trial.CovariateCollection` + `nstat.compat.matlab.CovColl` | `Complete` | API mapped in `parity/method_mapping.yaml`; no class-specific MATLAB fixture loop yet |
| `TrialConfig` | `nstat.trial.TrialConfig` + `nstat.compat.matlab.TrialConfig` | `Verified` | `tests/test_trialconfig_matlab_parity.py`, `matlab/fixture_gen/TrialConfig_fixtures.m` |
| `ConfigColl` | `nstat.trial.ConfigCollection` + `nstat.compat.matlab.ConfigColl` | `Verified` | `tests/test_configcoll_matlab_parity.py`, `matlab/fixture_gen/ConfigColl_fixtures.m` |
| `Trial` | `nstat.trial.Trial` + `nstat.compat.matlab.Trial` | `Complete` | API mapped in `parity/method_mapping.yaml`; no class-specific MATLAB fixture loop yet |
| `CIF` | `nstat.cif.CIFModel` + `nstat.compat.matlab.CIF` | `Complete` | API mapped in `parity/method_mapping.yaml`; no class-specific MATLAB fixture loop yet |
| `Analysis` | `nstat.analysis.Analysis` + `nstat.compat.matlab.Analysis` | `Complete` | API mapped in `parity/method_mapping.yaml`; no class-specific MATLAB fixture loop yet |
| `FitResult` | `nstat.fit.FitResult` + `nstat.compat.matlab.FitResult` | `Complete` | API mapped in `parity/method_mapping.yaml`; no class-specific MATLAB fixture loop yet |
| `FitResSummary` | `nstat.fit.FitSummary` + `nstat.compat.matlab.FitResSummary` | `Complete` | API mapped in `parity/method_mapping.yaml`; no class-specific MATLAB fixture loop yet |
| `DecodingAlgorithms` | `nstat.decoding.DecodingAlgorithms` + `nstat.compat.matlab.DecodingAlgorithms` | `Complete` | API mapped in `parity/method_mapping.yaml`; no class-specific MATLAB fixture loop yet |

## Next verification order
1. `CovColl`
2. `Trial`
3. `CIF`
4. `Analysis`
5. `FitResult`
6. `FitResSummary`
7. `DecodingAlgorithms`
