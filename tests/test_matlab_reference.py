from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nstat import Analysis, CIF, ConfigColl, CovColl, Covariate, Trial, TrialConfig, nspikeTrain, nstColl, simulate_two_neuron_network
from nstat.matlab_reference import (
    matlab_engine_available,
    run_analysis_reference,
    run_point_process_reference,
    run_simulated_network_reference,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MATLAB_REPO_ROOT = REPO_ROOT.parent / "nSTAT"


def test_matlab_engine_detection_is_boolean() -> None:
    assert isinstance(matlab_engine_available(), bool)


@pytest.mark.skipif(not MATLAB_REPO_ROOT.exists(), reason="MATLAB reference repo not available")
def test_matlab_reference_executes_only_when_engine_is_available() -> None:
    if not matlab_engine_available():
        pytest.skip("MATLAB Engine for Python is not installed")

    point_process = run_point_process_reference(matlab_repo=MATLAB_REPO_ROOT)
    network = run_simulated_network_reference(matlab_repo=MATLAB_REPO_ROOT)

    assert point_process["spike_counts"].shape == (5,)
    assert point_process["lambda_head"].shape == (10,)
    assert network["spike_counts"].shape == (2,)
    assert network["prob_head"].shape == (5, 2)
    assert network["state_head"].shape == (5, 2)
    np.testing.assert_allclose(network["actual_network"], np.array([[0.0, 1.0], [-4.0, 0.0]], dtype=float))


@pytest.mark.skipif(not MATLAB_REPO_ROOT.exists(), reason="MATLAB reference repo not available")
def test_native_point_process_simulation_matches_matlab_lambda_head_when_engine_is_available() -> None:
    if not matlab_engine_available():
        pytest.skip("MATLAB Engine for Python is not installed")

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
    )
    matlab_ref = run_point_process_reference(matlab_repo=MATLAB_REPO_ROOT, seed=5)

    np.testing.assert_allclose(lambda_cov.data[:10, 0], matlab_ref["lambda_head"], rtol=1e-6, atol=1e-8)


@pytest.mark.skipif(not MATLAB_REPO_ROOT.exists(), reason="MATLAB reference repo not available")
def test_native_network_simulation_preserves_matlab_connectivity_layout_when_engine_is_available() -> None:
    if not matlab_engine_available():
        pytest.skip("MATLAB Engine for Python is not installed")

    native = simulate_two_neuron_network(seed=4)
    matlab_ref = run_simulated_network_reference(matlab_repo=MATLAB_REPO_ROOT, seed=4)

    np.testing.assert_allclose(native.actual_network, matlab_ref["actual_network"])
    np.testing.assert_allclose(native.lambda_delta[:5], matlab_ref["prob_head"], rtol=1e-6, atol=1e-8)
    assert np.all((matlab_ref["state_head"] == 0.0) | (matlab_ref["state_head"] == 1.0))


@pytest.mark.skipif(not MATLAB_REPO_ROOT.exists(), reason="MATLAB reference repo not available")
def test_native_analysis_fit_matches_matlab_reference_when_engine_is_available() -> None:
    if not matlab_engine_available():
        pytest.skip("MATLAB Engine for Python is not installed")

    time = np.arange(0.0, 1.0 + 0.1, 0.1)
    stim = Covariate(time, np.sin(2 * np.pi * time), "Stimulus", "time", "s", "", ["stim"])
    spike_train = nspikeTrain([0.1, 0.4, 0.7], "1", 0.1, 0.0, 1.0, "time", "s", "", "", -1)
    trial = Trial(nstColl([spike_train]), CovColl([stim]))
    cfg = TrialConfig([["Stimulus", "stim"]], 10.0, [], [], name="stim")
    fit = Analysis.RunAnalysisForNeuron(trial, 1, ConfigColl([cfg]))
    matlab_ref = run_analysis_reference(matlab_repo=MATLAB_REPO_ROOT)

    np.testing.assert_allclose(fit.getCoeffs(1), matlab_ref["coeffs"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(fit.lambdaSignal.data[:5, 0], matlab_ref["lambda_head"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(np.asarray(fit.AIC, dtype=float)[:1], matlab_ref["aic"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(np.asarray(fit.BIC, dtype=float)[:1], matlab_ref["bic"], rtol=1e-6, atol=1e-8)
    assert np.isfinite(np.asarray(fit.logLL, dtype=float)[0])
