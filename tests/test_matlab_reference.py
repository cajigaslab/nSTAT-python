from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from nstat import Analysis, CIF, ConfigColl, CovColl, Covariate, FitResSummary, Trial, TrialConfig, nspikeTrain, nstColl, simulate_two_neuron_network


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "parity" / "fixtures" / "matlab_gold"

def _load_fixture(name: str):
    return loadmat(FIXTURE_ROOT / name, squeeze_me=True, struct_as_record=False)

def test_point_process_fixture_remains_consumable_without_matlab_runtime() -> None:
    payload = _load_fixture("point_process_exactness.mat")
    lambda_head = np.asarray(payload["lambda_head"], dtype=float).reshape(-1)
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
        seed=5,
        return_lambda=True,
        backend="python",
    )
    np.testing.assert_allclose(lambda_cov.data[: lambda_head.shape[0], 0], lambda_head, rtol=1e-8, atol=1e-10)

def test_network_fixture_remains_consumable_without_matlab_runtime() -> None:
    payload = _load_fixture("simulated_network_exactness.mat")
    native = simulate_two_neuron_network(seed=4, backend="python")

    np.testing.assert_allclose(native.actual_network, np.asarray(payload["actual_network"], dtype=float))
    np.testing.assert_allclose(native.lambda_delta[:5], np.asarray(payload["prob_head"], dtype=float), rtol=1e-8, atol=1e-10)
    dt = float(native.time[1] - native.time[0])
    native_state_head = np.column_stack([
        native.spikes.getNST(1).getSigRep(dt, float(native.time[0]), float(native.time[-1])).data[:5, 0],
        native.spikes.getNST(2).getSigRep(dt, float(native.time[0]), float(native.time[-1])).data[:5, 0],
    ])
    np.testing.assert_allclose(native_state_head, np.asarray(payload["state_head"], dtype=float), rtol=1e-8, atol=1e-10)
    native_counts = np.array([len(native.spikes.getNST(1).spikeTimes), len(native.spikes.getNST(2).spikeTimes)], dtype=float)
    assert np.all(np.abs(native_counts - np.asarray(payload["spike_counts"], dtype=float).reshape(-1)) <= 64.0)

def test_analysis_fixture_remains_consumable_without_matlab_runtime() -> None:
    payload = _load_fixture("analysis_exactness.mat")
    time = np.arange(0.0, 1.0 + 0.1, 0.1)
    stim = Covariate(time, np.sin(2 * np.pi * time), "Stimulus", "time", "s", "", ["stim"])
    spike_train = nspikeTrain([0.1, 0.4, 0.7], "1", 10.0, 0.0, 1.0, "time", "s", "", "", -1)
    trial = Trial(nstColl([spike_train]), CovColl([stim]))
    cfg = TrialConfig([["Stimulus", "stim"]], 10.0, [], [], name="stim")
    fit = Analysis.RunAnalysisForNeuron(trial, 1, ConfigColl([cfg]))
    summary = FitResSummary([fit])

    assert np.asarray(payload["coeffs"], dtype=float).reshape(-1).size >= 1
    assert np.asarray(payload["lambda_data"], dtype=float).reshape(-1).size >= 5
    assert np.isfinite(np.asarray(payload["AIC"], dtype=float).reshape(-1)[0])
    assert np.isfinite(np.asarray(payload["BIC"], dtype=float).reshape(-1)[0])
    assert np.isfinite(np.asarray(payload["summaryAIC"], dtype=float).reshape(-1)[0])
    assert np.isfinite(np.asarray(payload["summaryBIC"], dtype=float).reshape(-1)[0])
    assert np.isfinite(np.asarray(fit.AIC, dtype=float)[0])
    assert np.isfinite(np.asarray(fit.BIC, dtype=float)[0])
    assert np.isfinite(np.asarray(summary.AIC, dtype=float).reshape(-1)[0])
    assert np.isfinite(np.asarray(summary.BIC, dtype=float).reshape(-1)[0])
    assert np.isfinite(np.asarray(fit.logLL, dtype=float)[0])
