#!/usr/bin/env python3
"""Example 02 — Cross-language parity test.

Runs the Example 02 analysis (Whisker Stimulus GLM with Lag and History
Selection) in *both* Python and MATLAB and compares every numerical output.

Requirements:
    - MATLAB Engine API for Python (``pip install matlabengine``)
    - nSTAT MATLAB repo at ``/Users/iahncajigas/Library/CloudStorage/Dropbox/Claude/nSTAT``

Usage::

    python tests/test_example02_parity.py
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nstat.data_manager import ensure_example_data

MATLAB_NSTAT = Path("/Users/iahncajigas/Library/CloudStorage/Dropbox/Claude/nSTAT")
TOL = 1e-4  # tolerance for floating-point comparisons


def _matlab_colon(start: float, step: float, stop: float) -> np.ndarray:
    """Replicate MATLAB ``start:step:stop`` exactly."""
    n = int(np.floor((stop - start) / step)) + 1
    return start + np.arange(n) * step


# ═══════════════════════════════════════════════════════════════════════════
# Python side
# ═══════════════════════════════════════════════════════════════════════════
def run_python():
    """Run Example 02 analysis in Python and return all numerical outputs."""
    import matplotlib
    matplotlib.use("Agg")

    from scipy.io import loadmat
    from nstat import (
        Analysis, ConfigColl, CovColl, nspikeTrain, nstColl, Trial, TrialConfig,
    )
    from nstat.signal import Covariate

    data_dir = ensure_example_data(download=True)
    sampleRate = 1000

    # --- Load data ---
    mat_path = (data_dir / "Explicit Stimulus" / "Dir3" / "Neuron1"
                / "Stim2" / "trngdataBis.mat")
    d = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if hasattr(d.get("data", None), "t"):
        stimData = np.asarray(d["data"].t, dtype=float).reshape(-1)
        yData = np.asarray(d["data"].y, dtype=float).reshape(-1)
    else:
        stimData = np.asarray(d["t"], dtype=float).reshape(-1)
        yData = np.asarray(d["y"], dtype=float).reshape(-1)

    time = np.arange(0, len(stimData)) * (1.0 / sampleRate)
    spikeTimes = time[yData == 1]

    # --- Create nSTAT objects ---
    stim = Covariate(time, stimData / 10.0, "Stimulus", "time", "s", "mm",
                     dataLabels=["stim"])
    baseline = Covariate(time, np.ones((len(time), 1)), "Baseline", "time",
                         "s", "", dataLabels=["constant"])
    nst = nspikeTrain(spikeTimes)
    spikeColl = nstColl(nst)
    trial = Trial(spikeColl, CovColl([stim, baseline]))

    # --- Fit baseline-only model ---
    cfgBase = TrialConfig([("Baseline", "constant")], sampleRate, [], [])
    cfgBase.setName("Baseline")
    baselineResults = Analysis.RunAnalysisForAllNeurons(
        trial, ConfigColl([cfgBase]), 0)
    baselineCoeffs = np.asarray(baselineResults.b[0], dtype=float).flatten()
    baselineAIC = np.asarray(baselineResults.AIC, dtype=float).flatten()

    # --- Residual cross-covariance ---
    residual = baselineResults.computeFitResidual()
    xcovSig = residual.xcov(stim)
    xcovWindowed = xcovSig.windowedSignal([0, 1])
    peakTimes, peakVals = xcovWindowed.findGlobalPeak("maxima")
    shiftTime = float(peakTimes[0])
    peakVal = float(peakVals[0])

    # --- Shift stimulus ---
    stimShifted = Covariate(time, stimData, "Stimulus", "time", "s", "V",
                            dataLabels=["stim"])
    stimShifted = stimShifted.shift(shiftTime)
    baselineMu = Covariate(time, np.ones((len(time), 1)), "Baseline", "time",
                           "s", "", dataLabels=["\\mu"])
    trialShifted = Trial(nstColl(nspikeTrain(spikeTimes)),
                         CovColl([stimShifted, baselineMu]))

    # --- History sweep ---
    delta = 1.0 / sampleRate
    maxWindow = 1.0
    numWindows = 32
    logVals = np.logspace(np.log10(delta), np.log10(maxWindow), numWindows)
    windowTimes = np.concatenate([[0.0], logVals])
    windowTimes = np.unique(np.round(windowTimes * sampleRate) / sampleRate)

    # Use GLM algorithm (matching what Python example does)
    historySweep = Analysis.computeHistLagForAll(
        trialShifted, windowTimes,
        CovLabels=[("Baseline", "\\mu"), ("Stimulus", "stim")],
        Algorithm="GLM",
        batchMode=0, sampleRate=sampleRate, makePlot=0,
    )

    sweep = historySweep[0]
    aicArr = np.asarray(sweep.AIC, dtype=float)
    bicArr = np.asarray(sweep.BIC, dtype=float)
    ksArr = np.asarray(sweep.KSStats, dtype=float).ravel()

    # Window selection
    dAIC = aicArr[1:] - aicArr[0]
    dBIC = bicArr[1:] - bicArr[0]
    aicIdx = int(np.argmin(dAIC)) + 1 if dAIC.size > 0 else None
    bicIdx = int(np.argmin(dBIC)) + 1 if dBIC.size > 0 else None
    ksIdx = int(np.argmin(ksArr)) if ksArr.size > 0 else 0

    candidates = []
    if aicIdx is not None and aicIdx > 0:
        candidates.append(aicIdx)
    if bicIdx is not None and bicIdx > 0:
        candidates.append(bicIdx)
    windowIndex = min(candidates) if candidates else ksIdx

    if windowIndex > len(windowTimes):
        windowIndex = ksIdx

    if windowIndex > 1:
        selectedHistory = list(windowTimes[:windowIndex + 1])
    else:
        selectedHistory = []

    # --- Final 3-model comparison ---
    cfg1 = TrialConfig([("Baseline", "\\mu")], sampleRate, [], [])
    cfg1.setName("Baseline")
    cfg2 = TrialConfig([("Baseline", "\\mu"), ("Stimulus", "stim")],
                       sampleRate, [], [])
    cfg2.setName("Baseline+Stimulus")
    cfg3 = TrialConfig([("Baseline", "\\mu"), ("Stimulus", "stim")],
                       sampleRate, selectedHistory, [])
    cfg3.setName("Baseline+Stimulus+Hist")

    modelCompare = Analysis.RunAnalysisForAllNeurons(
        trialShifted, ConfigColl([cfg1, cfg2, cfg3]), 0)

    modelAIC = np.asarray(modelCompare.AIC, dtype=float).flatten()
    modelBIC = np.asarray(modelCompare.BIC, dtype=float).flatten()
    modelCoeffs = [np.asarray(c, dtype=float).flatten() for c in modelCompare.b]

    # Lambda traces
    lambdaData = np.asarray(modelCompare.lambda_signal.data, dtype=float)

    return {
        "nSpikes": len(spikeTimes),
        "dataLen": len(stimData),
        "timeLen": len(time),
        "maxTime": float(time[-1]),
        "baselineCoeffs": baselineCoeffs,
        "baselineAIC": baselineAIC,
        "shiftTime": shiftTime,
        "peakVal": peakVal,
        "windowTimes": windowTimes,
        "numWindowTimes": len(windowTimes),
        "sweepAIC": aicArr,
        "sweepBIC": bicArr,
        "sweepKS": ksArr,
        "aicIdx": aicIdx,
        "bicIdx": bicIdx,
        "ksIdx": ksIdx,
        "windowIndex": windowIndex,
        "numSelectedHistory": len(selectedHistory),
        "selectedHistory": np.array(selectedHistory),
        "modelAIC": modelAIC,
        "modelBIC": modelBIC,
        "modelCoeffs": modelCoeffs,
        "lambdaShape": lambdaData.shape,
        "lambda_first100": lambdaData[:100],
        "lambda_last100": lambdaData[-100:],
    }


# ═══════════════════════════════════════════════════════════════════════════
# MATLAB side
# ═══════════════════════════════════════════════════════════════════════════
def run_matlab():
    """Run Example 02 analysis in MATLAB and return all numerical outputs."""
    import matlab.engine

    py_data_dir = str(ensure_example_data(download=True))

    eng = matlab.engine.start_matlab()
    eng.addpath(str(MATLAB_NSTAT), nargout=0)

    # Point MATLAB at the Python data cache
    eng.eval(f"explicitStimulusDir = '{py_data_dir}/Explicit Stimulus';", nargout=0)
    eng.eval("sampleRate = 1000;", nargout=0)

    # Load data
    eng.eval("""
        dataPath = fullfile(explicitStimulusDir, 'Dir3', 'Neuron1', 'Stim2');
        data = load(fullfile(dataPath, 'trngdataBis.mat'));
    """, nargout=0)

    eng.eval("""
        time = 0:0.001:(length(data.t)-1)*0.001;
        stimData = data.t;
        spikeTimes = time(data.y == 1);
    """, nargout=0)

    nSpikes = int(eng.eval("length(spikeTimes)"))
    dataLen = int(eng.eval("length(stimData)"))
    timeLen = int(eng.eval("length(time)"))
    maxTime = float(eng.eval("time(end)"))

    # Create nSTAT objects
    eng.eval("""
        stim = Covariate(time, stimData ./ 10, 'Stimulus', 'time', 's', 'mm', {'stim'});
        baseline = Covariate(time, ones(length(time), 1), 'Baseline', 'time', 's', '', {'constant'});
        nst = nspikeTrain(spikeTimes);
        spikeColl = nstColl(nst);
        trial = Trial(spikeColl, CovColl({stim, baseline}));
    """, nargout=0)

    # Fit baseline model
    eng.eval("""
        clear cfg;
        cfg{1} = TrialConfig({{'Baseline', 'constant'}}, sampleRate, [], []);
        cfg{1}.setName('Baseline');
        baselineResults = Analysis.RunAnalysisForAllNeurons(trial, ConfigColl(cfg), 0);
    """, nargout=0)

    baselineCoeffs = np.array(eng.eval("baselineResults.b{1}")).flatten()
    baselineAIC = np.array(eng.eval("baselineResults.AIC")).flatten()

    # Residual cross-covariance and peak finding
    eng.eval("""
        [peakVal, ~, shiftTime] = max(baselineResults.Residual.xcov(stim).windowedSignal([0, 1]));
    """, nargout=0)

    shiftTime = float(eng.eval("shiftTime"))
    peakVal = float(eng.eval("peakVal"))

    # Shift stimulus
    eng.eval("""
        stimShifted = Covariate(time, stimData, 'Stimulus', 'time', 's', 'V', {'stim'});
        stimShifted = stimShifted.shift(shiftTime);
        baselineMu = Covariate(time, ones(length(time), 1), 'Baseline', 'time', 's', '', {'\\mu'});
        trialShifted = Trial(nstColl(nspikeTrain(spikeTimes)), CovColl({stimShifted, baselineMu}));
    """, nargout=0)

    # History sweep — use 'GLM' to match Python
    eng.eval("""
        delta = 1 / sampleRate;
        maxWindow = 1;
        numWindows = 32;
        windowTimes = unique(round([0 logspace(log10(delta), log10(maxWindow), numWindows)] .* sampleRate) ./ sampleRate);
        historySweep = Analysis.computeHistLagForAll(trialShifted, windowTimes, ...
            {{'Baseline', '\\mu'}, {'Stimulus', 'stim'}}, 'GLM', 0, sampleRate, 0);
    """, nargout=0)

    windowTimes_ml = np.array(eng.eval("windowTimes")).flatten()
    numWindowTimes = int(eng.eval("length(windowTimes)"))
    sweepAIC = np.array(eng.eval("historySweep{1}.AIC")).flatten()
    sweepBIC = np.array(eng.eval("historySweep{1}.BIC")).flatten()
    sweepKS = np.array(eng.eval("historySweep{1}.KSStats.ks_stat")).flatten()

    # Window selection (using MATLAB logic, indices converted to 0-based)
    eng.eval("""
        aicIdx_ml = find((historySweep{1}.AIC(2:end) - historySweep{1}.AIC(1)) == ...
                  min(historySweep{1}.AIC(2:end) - historySweep{1}.AIC(1)), 1, 'first') + 1;
        bicIdx_ml = find((historySweep{1}.BIC(2:end) - historySweep{1}.BIC(1)) == ...
                  min(historySweep{1}.BIC(2:end) - historySweep{1}.BIC(1)), 1, 'first') + 1;
        ksIdx_ml = find(historySweep{1}.KSStats.ks_stat == min(historySweep{1}.KSStats.ks_stat), 1, 'first');

        if isempty(aicIdx_ml) || aicIdx_ml == 1
            aicIdx_ml = inf;
        end
        if isempty(bicIdx_ml) || bicIdx_ml == 1
            bicIdx_ml = inf;
        end
        windowIndex_ml = min([aicIdx_ml, bicIdx_ml]);
        if ~isfinite(windowIndex_ml) || windowIndex_ml > numel(windowTimes)
            windowIndex_ml = ksIdx_ml;
        end
    """, nargout=0)

    # Convert MATLAB 1-based indices to Python 0-based for comparison
    aicIdx_ml_raw = eng.eval("aicIdx_ml")
    bicIdx_ml_raw = eng.eval("bicIdx_ml")
    ksIdx_ml = int(eng.eval("ksIdx_ml")) - 1  # 1-based to 0-based
    windowIndex_ml = int(eng.eval("windowIndex_ml")) - 1  # 1-based to 0-based

    # aicIdx/bicIdx in MATLAB are 1-based (or inf)
    import math
    aicIdx_ml = int(aicIdx_ml_raw) - 1 if not math.isinf(float(aicIdx_ml_raw)) else None
    bicIdx_ml = int(bicIdx_ml_raw) - 1 if not math.isinf(float(bicIdx_ml_raw)) else None

    # Selected history
    eng.eval("""
        if windowIndex_ml > 1
            selectedHistory_ml = windowTimes(1:windowIndex_ml);
        else
            selectedHistory_ml = [];
        end
    """, nargout=0)

    numSelectedHistory = int(eng.eval("length(selectedHistory_ml)"))
    selectedHistory = np.array(eng.eval("selectedHistory_ml")).flatten() if numSelectedHistory > 0 else np.array([])

    # Final 3-model comparison
    eng.eval("""
        clear cfg;
        cfg{1} = TrialConfig({{'Baseline', '\\mu'}}, sampleRate, [], []);
        cfg{1}.setName('Baseline');
        cfg{2} = TrialConfig({{'Baseline', '\\mu'}, {'Stimulus', 'stim'}}, sampleRate, [], []);
        cfg{2}.setName('Baseline+Stimulus');
        cfg{3} = TrialConfig({{'Baseline', '\\mu'}, {'Stimulus', 'stim'}}, sampleRate, selectedHistory_ml, []);
        cfg{3}.setName('Baseline+Stimulus+Hist');
        modelCompare = Analysis.RunAnalysisForAllNeurons(trialShifted, ConfigColl(cfg), 0);
    """, nargout=0)

    modelAIC = np.array(eng.eval("modelCompare.AIC")).flatten()
    modelBIC = np.array(eng.eval("modelCompare.BIC")).flatten()

    nModels = int(eng.eval("length(modelCompare.b)"))
    modelCoeffs = []
    for i in range(1, nModels + 1):
        modelCoeffs.append(np.array(eng.eval(f"modelCompare.b{{{i}}}")).flatten())

    lambdaData = np.array(eng.eval("modelCompare.lambda.dataToMatrix()"))
    lambda_first100 = lambdaData[:100]
    lambda_last100 = lambdaData[-100:]

    eng.quit()

    return {
        "nSpikes": nSpikes,
        "dataLen": dataLen,
        "timeLen": timeLen,
        "maxTime": maxTime,
        "baselineCoeffs": baselineCoeffs,
        "baselineAIC": baselineAIC,
        "shiftTime": shiftTime,
        "peakVal": peakVal,
        "windowTimes": windowTimes_ml,
        "numWindowTimes": numWindowTimes,
        "sweepAIC": sweepAIC,
        "sweepBIC": sweepBIC,
        "sweepKS": sweepKS,
        "aicIdx": aicIdx_ml,
        "bicIdx": bicIdx_ml,
        "ksIdx": ksIdx_ml,
        "windowIndex": windowIndex_ml,
        "numSelectedHistory": numSelectedHistory,
        "selectedHistory": selectedHistory,
        "modelAIC": modelAIC,
        "modelBIC": modelBIC,
        "modelCoeffs": modelCoeffs,
        "lambdaShape": lambdaData.shape,
        "lambda_first100": lambda_first100,
        "lambda_last100": lambda_last100,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Compare
# ═══════════════════════════════════════════════════════════════════════════
def compare(py: dict, ml: dict):
    """Compare Python and MATLAB results; return (passed, total)."""
    passed = 0
    total = 0

    def check(name, py_val, ml_val, tol=TOL):
        nonlocal passed, total
        total += 1
        py_a = np.asarray(py_val, dtype=float).flatten()
        ml_a = np.asarray(ml_val, dtype=float).flatten()
        if py_a.shape != ml_a.shape:
            print(f"  ✗ {name}  SHAPE MISMATCH py={py_a.shape} ml={ml_a.shape}")
            return
        if py_a.size == 0 and ml_a.size == 0:
            print(f"  ✓ {name}  (both empty)")
            passed += 1
            return
        delta = np.max(np.abs(py_a - ml_a))
        ok = delta < tol
        sym = "✓" if ok else "✗"
        extra = ""
        if not ok:
            idx = int(np.argmax(np.abs(py_a - ml_a)))
            extra = f" at [{idx}] py={py_a[idx]:.6f} ml={ml_a[idx]:.6f}"
        print(f"  {sym} {name}  (max Δ = {delta:.2e}){extra}")
        if ok:
            passed += 1

    def check_int(name, py_val, ml_val):
        nonlocal passed, total
        total += 1
        ok = py_val == ml_val
        sym = "✓" if ok else "✗"
        print(f"  {sym} {name}  py={py_val} ml={ml_val}" + ("" if ok else f" (Δ={py_val - ml_val if py_val is not None and ml_val is not None else 'N/A'})"))
        if ok:
            passed += 1

    def info_int(name, py_val, ml_val):
        """Report comparison but don't count towards pass/fail."""
        ok = py_val == ml_val
        sym = "≈" if ok else "~"
        print(f"  {sym} {name}  py={py_val} ml={ml_val} [informational — RNG-dependent]")

    print("\n═══ Data & Time Vector ═══")
    check_int("nSpikes", py["nSpikes"], ml["nSpikes"])
    check_int("dataLen", py["dataLen"], ml["dataLen"])
    check_int("timeLen", py["timeLen"], ml["timeLen"])
    check("maxTime", py["maxTime"], ml["maxTime"])

    print("\n═══ Baseline Model ═══")
    check("baseline coefficients", py["baselineCoeffs"], ml["baselineCoeffs"])
    check("baseline AIC", py["baselineAIC"], ml["baselineAIC"])

    print("\n═══ Cross-Covariance Peak ═══")
    check("shiftTime (lag)", py["shiftTime"], ml["shiftTime"])
    check("peakVal", py["peakVal"], ml["peakVal"])

    print("\n═══ History Window Times ═══")
    check_int("numWindowTimes", py["numWindowTimes"], ml["numWindowTimes"])
    check("windowTimes", py["windowTimes"], ml["windowTimes"])

    print("\n═══ History Sweep ═══")
    check("sweep AIC", py["sweepAIC"], ml["sweepAIC"], tol=0.1)
    check("sweep BIC", py["sweepBIC"], ml["sweepBIC"], tol=0.1)
    # KS stats use DT correction (Haslinger, Pipa, Brown 2010) with random
    # draws, so Python and MATLAB produce different values due to different
    # RNGs.  Use 0.05 tolerance (enough to catch formula bugs).
    check("sweep KS", py["sweepKS"], ml["sweepKS"], tol=0.05)

    print("\n═══ Window Selection ═══")
    check_int("aicIdx", py["aicIdx"], ml["aicIdx"])
    check_int("bicIdx", py["bicIdx"], ml["bicIdx"])
    # ksIdx depends on random DT-correction draws — skip exact comparison,
    # but windowIndex (determined by AIC/BIC) must match.
    info_int("ksIdx", py["ksIdx"], ml["ksIdx"])
    check_int("windowIndex", py["windowIndex"], ml["windowIndex"])

    print("\n═══ Selected History ═══")
    check_int("numSelectedHistory", py["numSelectedHistory"], ml["numSelectedHistory"])
    if py["numSelectedHistory"] > 0 and ml["numSelectedHistory"] > 0:
        min_len = min(len(py["selectedHistory"]), len(ml["selectedHistory"]))
        check("selectedHistory", py["selectedHistory"][:min_len],
              ml["selectedHistory"][:min_len])

    print("\n═══ Final 3-Model Comparison ═══")
    check("model AIC", py["modelAIC"], ml["modelAIC"], tol=0.1)
    check("model BIC", py["modelBIC"], ml["modelBIC"], tol=0.1)
    for i, name in enumerate(["Baseline", "Baseline+Stim", "Baseline+Stim+Hist"]):
        if i < len(py["modelCoeffs"]) and i < len(ml["modelCoeffs"]):
            check(f"coeffs[{name}]", py["modelCoeffs"][i], ml["modelCoeffs"][i])

    print("\n═══ Lambda Traces ═══")
    check_int("lambda shape[0]", py["lambdaShape"][0], ml["lambdaShape"][0])
    if py["lambdaShape"][0] == ml["lambdaShape"][0]:
        ncols = min(py["lambda_first100"].shape[1] if py["lambda_first100"].ndim > 1 else 1,
                    ml["lambda_first100"].shape[1] if ml["lambda_first100"].ndim > 1 else 1)
        for col in range(ncols):
            py_f = py["lambda_first100"][:, col] if py["lambda_first100"].ndim > 1 else py["lambda_first100"]
            ml_f = ml["lambda_first100"][:, col] if ml["lambda_first100"].ndim > 1 else ml["lambda_first100"]
            py_l = py["lambda_last100"][:, col] if py["lambda_last100"].ndim > 1 else py["lambda_last100"]
            ml_l = ml["lambda_last100"][:, col] if ml["lambda_last100"].ndim > 1 else ml["lambda_last100"]
            check(f"lambda_first100[col={col}]", py_f, ml_f, tol=0.01)
            check(f"lambda_last100[col={col}]", py_l, ml_l, tol=0.01)

    return passed, total


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = _time.time()
    print("Running Example 02 in Python …")
    py = run_python()
    t_py = _time.time() - t0
    print(f"  Python done in {t_py:.1f}s")

    t1 = _time.time()
    print("\nStarting MATLAB engine …")
    ml = run_matlab()
    t_ml = _time.time() - t1
    print(f"  MATLAB done in {t_ml:.1f}s")

    passed, total = compare(py, ml)

    print(f"\n{'═' * 60}")
    status = "✓ ALL PASS" if passed == total else f"✗ {total - passed} FAILED"
    print(f"  EXAMPLE 02 PARITY: {passed}/{total} passed  {status}")
    print(f"  Python: {t_py:.1f}s | MATLAB: {t_ml:.1f}s")
    print(f"{'═' * 60}")

    sys.exit(0 if passed == total else 1)
