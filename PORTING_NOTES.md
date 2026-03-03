# Porting Notes

This file tracks MATLAB-to-Python parity constraints, known deviations, and fixture regeneration steps.

## Current scope
- Completed full parity loop for `Events`:
  - Python implementation updates (`src/nstat/events.py`)
  - MATLAB compatibility wrapper updates (`src/nstat/compat/matlab/__init__.py`, `Events` section)
  - MATLAB fixture generator (`matlab/fixture_gen/Events_fixtures.m`)
  - Fixture artifact (`tests/fixtures/Events/basic.mat`)
  - Python parity tests (`tests/test_events_matlab_parity.py`)
  - Python demo (`examples/events_demo.py`)
- Completed full parity loop for `History`:
  - Python implementation updates (`src/nstat/history.py`)
  - MATLAB compatibility wrapper updates (`src/nstat/compat/matlab/__init__.py`, `History` section)
  - MATLAB fixture generator (`matlab/fixture_gen/History_fixtures.m`)
  - Fixture artifact (`tests/fixtures/History/basic.mat`)
  - Python parity tests (`tests/test_history_matlab_parity.py`)
  - Python demo (`examples/history_demo.py`)
- Completed full parity loop for `ConfidenceInterval`:
  - Python implementation updates (`src/nstat/confidence.py`)
  - MATLAB compatibility wrapper updates (`src/nstat/compat/matlab/__init__.py`, `ConfidenceInterval` section)
  - MATLAB fixture generator (`matlab/fixture_gen/ConfidenceInterval_fixtures.m`)
  - Fixture artifact (`tests/fixtures/ConfidenceInterval/basic.mat`)
  - Python parity tests (`tests/test_confidence_matlab_parity.py`)
  - Python demo (`examples/confidence_interval_demo.py`)
- Completed full parity loop for `SignalObj`:
  - MATLAB compatibility wrapper updates (`src/nstat/compat/matlab/__init__.py`, `SignalObj` section)
  - MATLAB fixture generator (`matlab/fixture_gen/SignalObj_fixtures.m`)
  - Fixture artifact (`tests/fixtures/SignalObj/basic.mat`)
  - Python parity tests (`tests/test_signalobj_matlab_parity.py`)
  - Python demo (`examples/signalobj_demo.py`)
- Completed full parity loop for `Covariate`:
  - MATLAB compatibility wrapper updates (`src/nstat/compat/matlab/__init__.py`, `Covariate` section)
  - MATLAB fixture generator (`matlab/fixture_gen/Covariate_fixtures.m`)
  - Fixture artifact (`tests/fixtures/Covariate/basic.mat`)
  - Python parity tests (`tests/test_covariate_matlab_parity.py`)
  - Python demo (`examples/covariate_demo.py`)

## Intentional deviations
- MATLAB indexing is 1-based; Python indexing is 0-based. This does not change `Events` numeric output, but affects user-facing index expectations in general.
- `nstat.events.Events` keeps a Pythonic `subset(start_s, end_s)` helper even though MATLAB `Events` does not define `subset`; this is additive and does not alter MATLAB compatibility wrapper behavior.
- `SignalObj.findNearestTimeIndex` and `findNearestTimeIndices` in Python compatibility currently return 0-based indices; parity assertions convert to MATLAB's 1-based convention when comparing fixtures.

## Tolerances
- `Events` parity checks use exact shape matching and `np.testing.assert_allclose(..., rtol=0.0, atol=1e-12)` for floating-point vectors.
- `Covariate.filtfilt` parity checks currently use `atol=2e-3` due MATLAB/Scipy edge-handling differences at short sequence boundaries.

## Regenerate MATLAB fixtures
From repo root (`nSTAT-python`), run:

```bash
matlab -batch "addpath('/Users/iahncajigas/Library/CloudStorage/Dropbox/Research/Matlab/nSTAT_currentRelease_Local'); addpath('matlab/fixture_gen'); Events_fixtures('tests/fixtures/Events/basic.mat');"
```

## Run parity checks

```bash
pytest -q tests/test_events_matlab_parity.py
```
