from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from nstat import Analysis, CIF, ConfigColl, CovColl, Covariate, SignalObj, Trial, TrialConfig, nspikeTrain, nstColl


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "parity" / "fixtures" / "matlab_gold"


def _load_fixture(name: str) -> dict[str, np.ndarray]:
    return loadmat(FIXTURE_ROOT / name, squeeze_me=True, struct_as_record=False)


def _scalar(payload: dict[str, np.ndarray], key: str) -> float:
    return float(np.asarray(payload[key], dtype=float).reshape(-1)[0])


def _vector(payload: dict[str, np.ndarray], key: str) -> np.ndarray:
    return np.asarray(payload[key], dtype=float).reshape(-1)


def _string(payload: dict[str, np.ndarray], key: str) -> str:
    value = payload[key]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr.reshape(-1)[0])


def test_signalobj_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("signalobj_exactness.mat")
    signal = SignalObj(_vector(payload, "time"), np.asarray(payload["data"], dtype=float), "sig", "time", "s", "u", ["x1", "x2"])
    signal_1 = signal.getSubSignal(1)
    signal_2 = SignalObj(np.arange(0.05, 0.5, 0.1), [0.0, 1.0, 0.0, -1.0, 0.0], "sig2", "time", "s", "u", ["x3"])

    filtered = signal.filter(_vector(payload, "filter_b"), _vector(payload, "filter_a"))
    derivative = signal.derivative
    integral = signal.integral()
    resampled = signal.resample(_scalar(payload, "resample_rate"))
    xcorr = signal.getSubSignal(1).xcorr(signal.getSubSignal(2), int(_scalar(payload, "xcorr_maxlag")))
    compatible_left, compatible_right = signal_1.makeCompatible(signal_2, holdVals=1)

    np.testing.assert_allclose(filtered.data, np.asarray(payload["filtered_data"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(derivative.data, np.asarray(payload["derivative_data"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(integral.data, np.asarray(payload["integral_data"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(resampled.time, _vector(payload, "resampled_time"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(resampled.data, np.asarray(payload["resampled_data"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(xcorr.time, _vector(payload, "xcorr_time"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(xcorr.data.reshape(-1), _vector(payload, "xcorr_data"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(compatible_left.time, _vector(payload, "compat_time"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(compatible_left.data.reshape(-1), _vector(payload, "compat_left_data"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(compatible_right.data.reshape(-1), _vector(payload, "compat_right_data"), rtol=1e-8, atol=1e-10)


def test_nspiketrain_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("nspiketrain_exactness.mat")
    nst = nspikeTrain(
        _vector(payload, "spikeTimes"),
        "nst",
        _scalar(payload, "binwidth"),
        _scalar(payload, "minTime"),
        _scalar(payload, "maxTime"),
        "time",
        "s",
        "spikes",
        "spk",
        0,
    )

    sig = nst.getSigRep(_scalar(payload, "binwidth"), _scalar(payload, "minTime"), _scalar(payload, "maxTime"))
    parts = nst.partitionNST([_scalar(payload, "minTime"), 0.2, _scalar(payload, "maxTime")])

    np.testing.assert_allclose(sig.time, _vector(payload, "sig_time"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sig.data[:, 0], _vector(payload, "sig_data"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(nst.getISIs(), _vector(payload, "isis"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(nst.avgFiringRate), _scalar(payload, "avgFiringRate"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(nst.B), _scalar(payload, "B"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(nst.An), _scalar(payload, "An"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(nst.burstIndex), _scalar(payload, "burstIndex"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(parts.getNST(1).spikeTimes, _vector(payload, "part1_spikes"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(parts.getNST(2).spikeTimes, _vector(payload, "part2_spikes"), rtol=1e-8, atol=1e-10)

    restore_train = nspikeTrain(_vector(payload, "spikeTimes"), "restore", 0.2, -0.1, 0.8, "time", "s", "spikes", "spk", -1)
    restore_train.setSigRep(0.1, -0.1, 0.8)
    restore_train.setMinTime(-0.3)
    restore_train.setMaxTime(1.1)
    restore_train.restoreToOriginal()

    np.testing.assert_allclose(float(restore_train.minTime), _scalar(payload, "restore_min_time"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(float(restore_train.maxTime), _scalar(payload, "restore_max_time"), rtol=1e-12, atol=1e-12)

    burst_train = nspikeTrain([0.0, 0.001, 0.002, 0.007, 0.507, 0.508, 0.509, 0.514], "bursting", 0.001, 0.0, 0.6, "time", "s", "spikes", "spk", 0)
    np.testing.assert_allclose(float(burst_train.avgSpikesPerBurst), _scalar(payload, "burst_avgSpikesPerBurst"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(burst_train.stdSpikesPerBurst), _scalar(payload, "burst_stdSpikesPerBurst"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(burst_train.numBursts), _scalar(payload, "burst_numBursts"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(burst_train.numSpikesPerBurst, _vector(payload, "burst_numSpikesPerBurst"), rtol=1e-8, atol=1e-10)


def test_cif_eval_surface_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("cif_exactness.mat")
    cif = CIF(
        beta=_vector(payload, "beta"),
        Xnames=["stim1", "stim2"],
        stimNames=["stim1", "stim2"],
        fitType="binomial",
    )

    stim_val = _vector(payload, "stimVal")

    np.testing.assert_allclose(cif.evalLambdaDelta(stim_val), _scalar(payload, "lambda_delta"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(cif.evalGradient(stim_val).reshape(-1), _vector(payload, "gradient"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(cif.evalGradientLog(stim_val).reshape(-1), _vector(payload, "gradient_log"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(cif.evalJacobian(stim_val), np.asarray(payload["jacobian"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(cif.evalJacobianLog(stim_val), np.asarray(payload["jacobian_log"], dtype=float), rtol=1e-8, atol=1e-10)


def test_analysis_fit_surface_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("analysis_exactness.mat")
    time = _vector(payload, "time")
    stim_data = _vector(payload, "stim_data")
    spike_times = _vector(payload, "spike_times")
    sample_rate = _scalar(payload, "sample_rate")

    stim = Covariate(time, stim_data, "Stimulus", "time", "s", "", ["stim"])
    spike_train = nspikeTrain(spike_times, "1", 0.1, 0.0, 1.0, "time", "s", "", "", -1)
    trial = Trial(nstColl([spike_train]), CovColl([stim]))
    cfg = TrialConfig([["Stimulus", "stim"]], sample_rate, [], [], name="stim")
    fit = Analysis.RunAnalysisForNeuron(trial, 1, ConfigColl([cfg]))

    np.testing.assert_allclose(fit.getCoeffs(1), _vector(payload, "coeffs"), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(fit.lambdaSignal.time, _vector(payload, "lambda_time"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fit.lambdaSignal.data[:, 0], _vector(payload, "lambda_data"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(fit.AIC[0]), _scalar(payload, "AIC"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(fit.BIC[0]), _scalar(payload, "BIC"), rtol=1e-8, atol=1e-10)
    assert np.isfinite(float(fit.logLL[0]))
    assert fit.fitType[0] == _string(payload, "distribution")


def test_point_process_lambda_trace_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("point_process_exactness.mat")

    time = np.arange(0.0, 50.0 + 0.001, 0.001, dtype=float)
    stim = Covariate(time, np.sin(2 * np.pi * 1.0 * time), "Stimulus", "time", "s", "Voltage", ["sin"])
    ens = Covariate(time, np.zeros_like(time), "Ensemble", "time", "s", "Spikes", ["n1"])
    _, lambda_cov = CIF.simulateCIF(
        -3.0,
        np.array([-1.0, -2.0, -4.0], dtype=float),
        np.array([1.0], dtype=float),
        np.array([0.0], dtype=float),
        stim,
        ens,
        numRealizations=5,
        simType="binomial",
        seed=int(_scalar(payload, "seed")),
        return_lambda=True,
    )

    np.testing.assert_allclose(lambda_cov.data[: _vector(payload, 'lambda_head').shape[0], 0], _vector(payload, "lambda_head"), rtol=1e-8, atol=1e-10)
