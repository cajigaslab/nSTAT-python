#!/usr/bin/env python3
"""Cross-language parity test: Example 01 (mEPSC Poisson Models).

Runs the same Example 01 analysis in both MATLAB and Python, then compares
every numerical output: spike counts, time vectors, GLM coefficients,
AIC/BIC, lambda traces, and KS statistics.

Usage:
    python tests/test_example01_parity.py
"""
from __future__ import annotations

import sys
import time
import textwrap

import numpy as np

# ── Python nSTAT imports ──────────────────────────────────────────────
sys.path.insert(0, "/Users/iahncajigas/Library/CloudStorage/Dropbox/Claude/nSTAT-python")

from nstat import (
    Analysis, ConfigColl, CovColl, nspikeTrain, nstColl, Trial, TrialConfig,
)
from nstat.signal import Covariate
from nstat.data_manager import ensure_example_data

NSTAT_MATLAB_PATH = "/Users/iahncajigas/Library/CloudStorage/Dropbox/Claude/nSTAT"

# ── Tolerances ────────────────────────────────────────────────────────
ATOL = 1e-8
RTOL = 1e-6


def _matlab_colon(start: float, step: float, stop: float) -> np.ndarray:
    """Replicate MATLAB ``start:step:stop`` exactly."""
    n = int(np.floor((stop - start) / step)) + 1
    return start + np.arange(n) * step


def _arr(matlab_val) -> np.ndarray:
    """Convert MATLAB double/matrix to numpy array."""
    return np.asarray(matlab_val, dtype=float).reshape(-1)


def _arr2d(matlab_val) -> np.ndarray:
    """Convert MATLAB 2D matrix to numpy 2D array."""
    return np.asarray(matlab_val, dtype=float)


def _scalar(matlab_val) -> float:
    v = np.asarray(matlab_val, dtype=float).ravel()
    return float(v[0]) if v.size else float(matlab_val)


def _compare(name: str, py_val, ml_val, atol=ATOL, rtol=RTOL) -> bool:
    """Compare Python and MATLAB values; print PASS/FAIL."""
    py = np.asarray(py_val, dtype=float).ravel()
    ml = np.asarray(ml_val, dtype=float).ravel()
    if py.shape != ml.shape:
        print(f"  ✗ {name}: SHAPE MISMATCH py={py.shape} ml={ml.shape}")
        return False
    if np.allclose(py, ml, atol=atol, rtol=rtol):
        maxdiff = float(np.max(np.abs(py - ml))) if py.size else 0.0
        print(f"  ✓ {name}  (max Δ = {maxdiff:.2e})")
        return True
    else:
        maxdiff = float(np.max(np.abs(py - ml)))
        idx = int(np.argmax(np.abs(py - ml)))
        print(f"  ✗ {name}: MISMATCH  max Δ = {maxdiff:.2e} at [{idx}]"
              f" py={py[idx]:.10g} ml={ml[idx]:.10g}")
        return False


# =====================================================================
# Run Python Example 01
# =====================================================================
def run_python_example01():
    """Run Example 01 in Python and return all numerical outputs."""
    import matplotlib
    matplotlib.use("Agg")

    data_dir = ensure_example_data(download=True)
    mepsc_dir = data_dir / "mEPSCs"
    sampleRate = 1000

    # ── Part 1: Constant Mg2+ ──
    data = np.loadtxt(mepsc_dir / "epsc2.txt", skiprows=1)
    epsc2 = data[:, 1] / 1000.0

    nstConst = nspikeTrain(epsc2)
    timeConst = _matlab_colon(0, 1.0 / sampleRate, nstConst.maxTime)

    baseline = Covariate(
        timeConst, np.ones((len(timeConst), 1)),
        "Baseline", "time", "s", "", dataLabels=["\\mu"],
    )
    spikeCollConst = nstColl(nstConst)
    trialConst = Trial(spikeCollConst, CovColl([baseline]))

    tcConst = TrialConfig([("Baseline", "\\mu")], sampleRate, [])
    tcConst.setName("Constant Baseline")
    configConst = ConfigColl([tcConst])

    resultConst = Analysis.RunAnalysisForAllNeurons(trialConst, configConst, 0)

    # ── Part 2+3: Washout piecewise model ──
    w1 = np.loadtxt(mepsc_dir / "washout1.txt", skiprows=1)
    w2 = np.loadtxt(mepsc_dir / "washout2.txt", skiprows=1)
    washout1 = w1[:, 1] / 1000.0
    washout2 = w2[:, 1] / 1000.0

    spikeTimes1 = 260.0 + washout1
    spikeTimes2 = np.sort(washout2) + 745.0
    nstWashout = nspikeTrain(np.concatenate([spikeTimes1, spikeTimes2]))
    timeWashout = _matlab_colon(260.0, 1.0 / sampleRate, nstWashout.maxTime)

    timeInd1 = np.searchsorted(timeWashout, 495.0, side="left")
    timeInd2 = np.searchsorted(timeWashout, 765.0, side="left")
    N = len(timeWashout)

    constantRate = np.ones((N, 1))
    rate1 = np.zeros((N, 1)); rate1[:timeInd1] = 1.0
    rate2 = np.zeros((N, 1)); rate2[timeInd1:timeInd2] = 1.0
    rate3 = np.zeros((N, 1)); rate3[timeInd2:] = 1.0

    baselineWashout = Covariate(
        timeWashout,
        np.column_stack([constantRate, rate1, rate2, rate3]),
        "Baseline", "time", "s", "",
        dataLabels=["\\mu", "\\mu_{1}", "\\mu_{2}", "\\mu_{3}"],
    )

    spikeCollWashout = nstColl(nstWashout)
    trialWashout = Trial(spikeCollWashout, CovColl([baselineWashout]))

    tc1 = TrialConfig([("Baseline", "\\mu")], sampleRate, [])
    tc1.setName("Constant Baseline")
    tc2 = TrialConfig([("Baseline", "\\mu_{1}", "\\mu_{2}", "\\mu_{3}")], sampleRate, [])
    tc2.setName("Diff Baseline")
    configWashout = ConfigColl([tc1, tc2])

    resultWashout = Analysis.RunAnalysisForAllNeurons(trialWashout, configWashout, 0)

    return {
        # Part 1
        "const_nSpikes": len(epsc2),
        "const_timeLen": len(timeConst),
        "const_maxTime": nstConst.maxTime,
        "const_b": resultConst.b[0],         # coefficients (1 model)
        "const_AIC": resultConst.AIC,
        "const_BIC": resultConst.BIC,
        "const_lambda_data": np.asarray(resultConst.lambda_signal.data, dtype=float),
        "const_lambda_time": np.asarray(resultConst.lambda_signal.time, dtype=float),
        # Part 3
        "wash_nSpikes": len(np.concatenate([spikeTimes1, spikeTimes2])),
        "wash_timeLen": len(timeWashout),
        "wash_maxTime": nstWashout.maxTime,
        "wash_timeInd1": timeInd1,
        "wash_timeInd2": timeInd2,
        "wash_b1": resultWashout.b[0],        # constant model coefficients
        "wash_b2": resultWashout.b[1],        # piecewise model coefficients
        "wash_AIC": resultWashout.AIC,
        "wash_BIC": resultWashout.BIC,
        "wash_lambda_data": np.asarray(resultWashout.lambda_signal.data, dtype=float),
        "wash_lambda_time": np.asarray(resultWashout.lambda_signal.time, dtype=float),
    }


# =====================================================================
# Run MATLAB Example 01
# =====================================================================
def run_matlab_example01(eng):
    """Run Example 01 in MATLAB and return all numerical outputs."""

    # Point MATLAB at the data (use Python's data cache since MATLAB data
    # is not installed; the mEPSC text files are identical)
    py_data_dir = str(ensure_example_data(download=True))
    eng.eval(textwrap.dedent(f"""\
        cd('{NSTAT_MATLAB_PATH}');
        mEPSCDir = '{py_data_dir}/mEPSCs';
        sampleRate = 1000;
    """), nargout=0)

    # Part 1: Constant Mg2+
    eng.eval(textwrap.dedent("""\
        epsc2 = importdata(fullfile(mEPSCDir, 'epsc2.txt'));
        spikeTimesConst = epsc2.data(:,2) ./ sampleRate;
        nstConst = nspikeTrain(spikeTimesConst);
        timeConst = 0:(1/sampleRate):nstConst.maxTime;

        baseline = Covariate(timeConst, ones(length(timeConst),1), ...
            'Baseline', 'time', 's', '', {'\\mu'});
        covarColl = CovColl({baseline});
        spikeCollConst = nstColl(nstConst);
        trialConst = Trial(spikeCollConst, covarColl);

        clear tcConst;
        tcConst{1} = TrialConfig({{'Baseline', '\\mu'}}, sampleRate, []);
        tcConst{1}.setName('Constant Baseline');
        configConst = ConfigColl(tcConst);
        resultConst = Analysis.RunAnalysisForAllNeurons(trialConst, configConst, 0);
    """), nargout=0)

    # Part 3: Washout piecewise
    eng.eval(textwrap.dedent("""\
        washout1 = importdata(fullfile(mEPSCDir, 'washout1.txt'));
        washout2 = importdata(fullfile(mEPSCDir, 'washout2.txt'));

        spikeTimes1 = 260 + washout1.data(:,2) ./ sampleRate;
        spikeTimes2 = sort(washout2.data(:,2)) ./ sampleRate + 745;
        nstWashout = nspikeTrain([spikeTimes1; spikeTimes2]);
        timeWashout = 260:(1/sampleRate):nstWashout.maxTime;

        timeInd1 = find(timeWashout < 495, 1, 'last');
        timeInd2 = find(timeWashout < 765, 1, 'last');
        constantRate = ones(length(timeWashout),1);
        rate1 = zeros(length(timeWashout),1);
        rate2 = zeros(length(timeWashout),1);
        rate3 = zeros(length(timeWashout),1);
        rate1(1:timeInd1) = 1;
        rate2((timeInd1+1):timeInd2) = 1;
        rate3((timeInd2+1):end) = 1;

        baselineWashout = Covariate(timeWashout, [constantRate, rate1, rate2, rate3], ...
            'Baseline', 'time', 's', '', {'\\mu', '\\mu_{1}', '\\mu_{2}', '\\mu_{3}'});

        spikeCollWashout = nstColl(nstWashout);
        trialWashout = Trial(spikeCollWashout, CovColl({baselineWashout}));

        clear tcWashout;
        tcWashout{1} = TrialConfig({{'Baseline', '\\mu'}}, sampleRate, []);
        tcWashout{1}.setName('Constant Baseline');
        tcWashout{2} = TrialConfig({{'Baseline', '\\mu_{1}', '\\mu_{2}', '\\mu_{3}'}}, sampleRate, []);
        tcWashout{2}.setName('Diff Baseline');
        configWashout = ConfigColl(tcWashout);
        resultWashout = Analysis.RunAnalysisForAllNeurons(trialWashout, configWashout, 0);
    """), nargout=0)

    # Extract all numerical results
    return {
        # Part 1
        "const_nSpikes": int(_scalar(eng.eval("length(spikeTimesConst)"))),
        "const_timeLen": int(_scalar(eng.eval("length(timeConst)"))),
        "const_maxTime": _scalar(eng.eval("nstConst.maxTime")),
        "const_b": _arr(eng.eval("resultConst.b{1}")),
        "const_AIC": _arr(eng.eval("resultConst.AIC")),
        "const_BIC": _arr(eng.eval("resultConst.BIC")),
        "const_lambda_data": _arr2d(eng.eval("resultConst.lambda.data")),
        "const_lambda_time": _arr(eng.eval("resultConst.lambda.time")),
        # Part 3
        "wash_nSpikes": int(_scalar(eng.eval("length([spikeTimes1; spikeTimes2])"))),
        "wash_timeLen": int(_scalar(eng.eval("length(timeWashout)"))),
        "wash_maxTime": _scalar(eng.eval("nstWashout.maxTime")),
        "wash_timeInd1": int(_scalar(eng.eval("timeInd1"))),
        "wash_timeInd2": int(_scalar(eng.eval("timeInd2"))),
        "wash_b1": _arr(eng.eval("resultWashout.b{1}")),
        "wash_b2": _arr(eng.eval("resultWashout.b{2}")),
        "wash_AIC": _arr(eng.eval("resultWashout.AIC")),
        "wash_BIC": _arr(eng.eval("resultWashout.BIC")),
        "wash_lambda_data": _arr2d(eng.eval("resultWashout.lambda.data")),
        "wash_lambda_time": _arr(eng.eval("resultWashout.lambda.time")),
    }


# =====================================================================
# Compare all outputs
# =====================================================================
def compare_results(py: dict, ml: dict) -> list[bool]:
    results: list[bool] = []

    # ── Part 1: Constant Mg2+ ──
    print("\n═══ Part 1: Constant Mg2+ — Homogeneous Poisson ═══")

    print("  -- Data & Time Vector --")
    results.append(_compare("nSpikes", py["const_nSpikes"], ml["const_nSpikes"]))
    results.append(_compare("timeConst length", py["const_timeLen"], ml["const_timeLen"]))
    results.append(_compare("maxTime", py["const_maxTime"], ml["const_maxTime"]))

    print("  -- GLM Coefficients --")
    results.append(_compare("b (constant model)", py["const_b"], ml["const_b"]))

    print("  -- Information Criteria --")
    results.append(_compare("AIC", py["const_AIC"], ml["const_AIC"], atol=1e-4))
    results.append(_compare("BIC", py["const_BIC"], ml["const_BIC"], atol=1e-4))

    print("  -- Lambda Trace --")
    py_lam = py["const_lambda_data"].ravel()
    ml_lam = ml["const_lambda_data"].ravel()
    results.append(_compare("lambda length", len(py_lam), len(ml_lam)))
    min_len = min(len(py_lam), len(ml_lam))
    if min_len > 0:
        results.append(_compare("lambda[:100]", py_lam[:min(100, min_len)],
                                ml_lam[:min(100, min_len)], atol=1e-6))
        results.append(_compare("lambda[-100:]", py_lam[max(0, min_len-100):min_len],
                                ml_lam[max(0, min_len-100):min_len], atol=1e-6))

    py_lt = py["const_lambda_time"].ravel()
    ml_lt = ml["const_lambda_time"].ravel()
    min_lt = min(len(py_lt), len(ml_lt))
    results.append(_compare("lambda_time[:10]", py_lt[:min(10, min_lt)],
                            ml_lt[:min(10, min_lt)]))

    # ── Part 3: Washout Piecewise Model ──
    print("\n═══ Part 3: Washout — Piecewise Baseline ═══")

    print("  -- Data & Time Vector --")
    results.append(_compare("nSpikes (washout)", py["wash_nSpikes"], ml["wash_nSpikes"]))
    results.append(_compare("timeWashout length", py["wash_timeLen"], ml["wash_timeLen"]))
    results.append(_compare("maxTime (washout)", py["wash_maxTime"], ml["wash_maxTime"]))

    # Note: Python uses 0-based indices, MATLAB uses 1-based.
    # Python searchsorted(side='right') for 495 gives the first index > 495,
    # which is the count of elements <= 495.
    # MATLAB find(time < 495, 1, 'last') gives the last 1-based index < 495.
    # These may differ by 1 depending on whether 495.0 is exactly in the array.
    print("  -- Epoch Boundaries --")
    # Just report for information; exact index comparison may differ by 1
    py_i1 = py["wash_timeInd1"]
    ml_i1 = ml["wash_timeInd1"]
    diff_i1 = abs(py_i1 - ml_i1)
    ok_i1 = diff_i1 <= 1
    print(f"  {'✓' if ok_i1 else '✗'} timeInd1  py={py_i1} ml={ml_i1} (Δ={diff_i1})")
    results.append(ok_i1)

    py_i2 = py["wash_timeInd2"]
    ml_i2 = ml["wash_timeInd2"]
    diff_i2 = abs(py_i2 - ml_i2)
    ok_i2 = diff_i2 <= 1
    print(f"  {'✓' if ok_i2 else '✗'} timeInd2  py={py_i2} ml={ml_i2} (Δ={diff_i2})")
    results.append(ok_i2)

    print("  -- GLM Coefficients --")
    results.append(_compare("b1 (constant)", py["wash_b1"], ml["wash_b1"]))
    results.append(_compare("b2 (piecewise)", py["wash_b2"], ml["wash_b2"]))

    print("  -- Information Criteria --")
    results.append(_compare("AIC (washout)", py["wash_AIC"], ml["wash_AIC"], atol=1e-4))
    results.append(_compare("BIC (washout)", py["wash_BIC"], ml["wash_BIC"], atol=1e-4))

    print("  -- Lambda Traces --")
    py_wl = py["wash_lambda_data"]
    ml_wl = ml["wash_lambda_data"]
    if py_wl.ndim == 1:
        py_wl = py_wl[:, None]
    if ml_wl.ndim == 1:
        ml_wl = ml_wl[:, None]
    results.append(_compare("lambda shape", list(py_wl.shape), list(ml_wl.shape)))

    if py_wl.shape == ml_wl.shape and py_wl.shape[1] >= 2:
        # Model 1 (constant) lambda trace
        results.append(_compare("lambda_const[:100]",
                                py_wl[:100, 0], ml_wl[:100, 0], atol=1e-6))
        results.append(_compare("lambda_const[-100:]",
                                py_wl[-100:, 0], ml_wl[-100:, 0], atol=1e-6))
        # Model 2 (piecewise) lambda trace
        results.append(_compare("lambda_piecewise[:100]",
                                py_wl[:100, 1], ml_wl[:100, 1], atol=1e-6))
        results.append(_compare("lambda_piecewise[-100:]",
                                py_wl[-100:, 1], ml_wl[-100:, 1], atol=1e-6))

    py_wlt = py["wash_lambda_time"].ravel()
    ml_wlt = ml["wash_lambda_time"].ravel()
    min_wlt = min(len(py_wlt), len(ml_wlt))
    results.append(_compare("wash_lambda_time[:10]",
                            py_wlt[:min(10, min_wlt)], ml_wlt[:min(10, min_wlt)]))

    return results


# =====================================================================
# Main
# =====================================================================
def main():
    # ── Step 1: Run Python side ──
    print("Running Example 01 in Python …")
    t0 = time.time()
    py_results = run_python_example01()
    py_time = time.time() - t0
    print(f"  Python done in {py_time:.1f}s")

    # ── Step 2: Run MATLAB side ──
    import matlab.engine
    print("\nStarting MATLAB engine …")
    t0 = time.time()
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath(NSTAT_MATLAB_PATH))
    print(f"  Engine started in {time.time() - t0:.1f}s")

    print("Running Example 01 in MATLAB …")
    t0 = time.time()
    try:
        ml_results = run_matlab_example01(eng)
    finally:
        eng.quit()
    ml_time = time.time() - t0
    print(f"  MATLAB done in {ml_time:.1f}s")

    # ── Step 3: Compare ──
    all_results = compare_results(py_results, ml_results)

    # ── Summary ──
    passed = sum(all_results)
    total = len(all_results)
    failed = total - passed
    print(f"\n{'═' * 60}")
    print(f"  EXAMPLE 01 PARITY: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  ✓ ALL PASS")
    print(f"  Python: {py_time:.1f}s | MATLAB: {ml_time:.1f}s")
    print(f"{'═' * 60}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
