from __future__ import annotations

import numpy as np
from pathlib import Path

from nstat import Analysis, CIF, CIFModel, DecodingAlgorithms, FitResSummary, Trial, TrialConfig
from nstat.ConfigColl import ConfigColl
from nstat.CovColl import CovColl
from nstat.Covariate import Covariate
from nstat.Events import Events
from nstat.History import History
from nstat.FitResult import FitResult
from nstat.analysis import compHistEnsCoeff, compHistEnsCoeffForAll, computeGrangerCausalityMatrix, computeNeighbors, spikeTrigAvg
from nstat.nstColl import nstColl
from nstat.nspikeTrain import nspikeTrain
from nstat.paper_examples_full import run_experiment1


def _build_trial() -> Trial:
    time = np.arange(0.0, 1.0, 0.1)
    stim = Covariate(time, np.sin(2 * np.pi * time), "Stimulus", "time", "s", "", ["stim"])
    vel = Covariate(time, np.cos(2 * np.pi * time), "Velocity", "time", "s", "", ["vel"])
    spikes = nstColl(
        [
            nspikeTrain([0.1, 0.3, 0.7], "1", 10.0, 0.0, 0.9, makePlots=-1),
            nspikeTrain([0.2, 0.5, 0.8], "2", 10.0, 0.0, 0.9, makePlots=-1),
        ]
    )
    return Trial(spikes, CovColl([stim, vel]), Events([0.2], ["cue"]), History([0.0, 0.1, 0.2]))


def _build_dense_trial() -> Trial:
    time = np.arange(0.0, 2.0, 0.05)
    stim = Covariate(time, np.sin(2 * np.pi * time), "Stimulus", "time", "s", "", ["stim"])
    vel = Covariate(time, np.cos(2 * np.pi * time), "Velocity", "time", "s", "", ["vel"])
    spikes = nstColl(
        [
            nspikeTrain([0.10, 0.25, 0.55, 0.90, 1.10, 1.55, 1.75], "1", 20.0, 0.0, 1.95, makePlots=-1),
            nspikeTrain([0.15, 0.35, 0.60, 0.95, 1.25, 1.45, 1.80], "2", 20.0, 0.0, 1.95, makePlots=-1),
        ]
    )
    trial = Trial(spikes, CovColl([stim, vel]), Events([0.2], ["cue"]), History([0.0, 0.05, 0.10]))
    trial.setEnsCovHist([0.0, 0.05, 0.10])
    return trial


def test_analysis_returns_matlab_style_fitresult_surface() -> None:
    trial = _build_trial()
    configs = ConfigColl(
        [
            TrialConfig(covMask=[["Stimulus", "stim"]], sampleRate=10.0, history=[0.0, 0.1, 0.2], name="stim_hist"),
            TrialConfig(covMask=[["Velocity", "vel"]], sampleRate=10.0, name="vel_only"),
        ]
    )

    fit = Analysis.RunAnalysisForNeuron(trial, 1, configs)

    assert isinstance(fit, FitResult)
    assert fit.numResults == 2
    assert fit.configNames == ["stim_hist", "vel_only"]
    assert fit.lambdaSignal.dimension == 2
    assert fit.neuronNumber == 1.0
    assert len(fit.covLabels) == 2
    assert "stim" in fit.uniqueCovLabels
    coeffs, labels, se = fit.getCoeffs(1)
    assert coeffs.shape[0] >= 2
    hist_coeffs, hist_labels, hist_se = fit.getHistCoeffs(1)
    assert hist_coeffs.shape[0] == 2


def test_fitresult_roundtrip_and_summary_preserve_core_metadata() -> None:
    trial = _build_trial()
    configs = ConfigColl([TrialConfig(covMask=[["Stimulus", "stim"]], sampleRate=10.0, name="stim_only")])
    fits = Analysis.RunAnalysisForAllNeurons(trial, configs)

    rebuilt = FitResult.fromStructure(fits[0].toStructure())
    assert rebuilt.numResults == fits[0].numResults
    assert rebuilt.configNames == fits[0].configNames
    np.testing.assert_allclose(rebuilt.AIC, fits[0].AIC)

    summary = FitResSummary(fits)
    assert summary.numNeurons == 2
    assert summary.numResults == 1
    assert summary.fitNames == ["stim_only"]
    assert summary.AIC.shape == (2, 1)


def test_cif_instantiation_evaluation_and_simulate_from_lambda() -> None:
    cif = CIF([0.2, -0.1], ["stim", "vel"], ["stim"], fitType="poisson")
    design = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=float)
    rate = cif.evaluate(design, delta=0.1)
    assert rate.shape == (3,)
    assert np.all(rate > 0)

    time = np.array([0.0, 0.1, 0.2], dtype=float)
    cov = CIF.from_linear_terms(time, 0.1, np.array([0.2, -0.1]), design, 0.1, "lambda")
    sim = CIF.simulateCIFByThinningFromLambda(cov, numRealizations=2)
    assert sim.numSpikeTrains == 2

    sim_single, details = CIF.simulateCIFByThinningFromLambda(
        cov.getSubSignal(1),
        numRealizations=1,
        random_values=np.array([0.8, 0.6, 0.4, 0.2], dtype=float),
        thinning_values=np.array([0.1, 0.9, 0.3, 0.7], dtype=float),
        return_details=True,
    )
    assert sim_single.numSpikeTrains == 1
    assert int(details["proposal_count"]) >= 0
    assert details["candidate_spike_times"].ndim == 1

    model = CIFModel(time, np.array([5.0, 6.0, 7.0]), name="lambda")
    sim2 = model.simulate(num_realizations=1, seed=1)
    assert sim2.numSpikeTrains == 1


def test_cif_gamma_methods_and_copy_follow_matlab_surface() -> None:
    history = History([0.0, 0.1, 0.2])
    train = nspikeTrain([0.05, 0.15], "1", 10.0, 0.0, 0.3, makePlots=-1)
    cif = CIF([0.2, -0.1], ["stim"], ["stim"], fitType="poisson", histCoeffs=[0.3, -0.2], historyObj=history, nst=train)

    copied = cif.CIFCopy()
    assert copied is not cif
    assert copied.history is not cif.history
    assert copied.spikeTrain is not cif.spikeTrain
    assert copied.isSymBeta() is False

    stim_val = np.array([0.4], dtype=float)
    gamma = np.array([0.8, 1.2], dtype=float)
    ld = cif.evalLDGamma(stim_val, time_index=2, gamma=gamma)
    log_ld = cif.evalLogLDGamma(stim_val, time_index=2, gamma=gamma)
    grad = cif.evalGradientLDGamma(stim_val, time_index=2, gamma=gamma)
    grad_log = cif.evalGradientLogLDGamma(stim_val, time_index=2, gamma=gamma)
    jac = cif.evalJacobianLDGamma(stim_val, time_index=2, gamma=gamma)
    jac_log = cif.evalJacobianLogLDGamma(stim_val, time_index=2, gamma=gamma)

    assert ld > 0.0
    np.testing.assert_allclose(log_ld, np.log(ld))
    assert grad.shape == (1, 1)
    assert grad_log.shape == (1, 1)
    assert jac.shape == (1, 1)
    assert jac_log.shape == (1, 1)


def test_simulatecif_uses_temporal_fir_filtering_for_stimulus_drive() -> None:
    time = np.arange(0.0, 0.5, 0.1, dtype=float)
    stim_values = np.array([0.0, 1.0, 0.0, -1.0, 0.5], dtype=float)
    stim = Covariate(time, stim_values, "Stimulus", "time", "s", "", ["stim"])
    ens = Covariate(time, np.zeros_like(time), "Ensemble", "time", "s", "", ["ens"])

    _, lambda_cov = CIF.simulateCIF(
        -1.5,
        np.zeros(0, dtype=float),
        np.array([1.0, -0.5], dtype=float),
        np.array([0.0], dtype=float),
        stim,
        ens,
        numRealizations=1,
        simType="binomial",
        seed=1,
        return_lambda=True,
        backend="python",
    )

    expected_drive = np.convolve(stim_values, np.array([1.0, -0.5], dtype=float), mode="full")[: time.size]
    expected_eta = -1.5 + expected_drive
    expected_lambda = (1.0 / (1.0 + np.exp(-np.clip(expected_eta, -20.0, 20.0)))) / 0.1
    np.testing.assert_allclose(lambda_cov.data[:, 0], expected_lambda)


def test_simulatecif_accepts_multi_input_kernel_bank() -> None:
    time = np.arange(0.0, 0.4, 0.1, dtype=float)
    stim_values = np.column_stack(
        [
            np.array([1.0, 0.0, 0.5, 0.0], dtype=float),
            np.array([0.0, 0.25, 0.0, 0.25], dtype=float),
        ]
    )
    ens_values = np.column_stack(
        [
            np.array([0.0, 1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 1.0, 0.0], dtype=float),
        ]
    )
    stim = Covariate(time, stim_values, "Stimulus", "time", "s", "", ["x1", "x2"])
    ens = Covariate(time, ens_values, "Ensemble", "time", "s", "", ["n1", "n2"])

    _, lambda_cov = CIF.simulateCIF(
        -2.0,
        np.zeros(0, dtype=float),
        [np.array([1.0, 0.5], dtype=float), np.array([-0.25], dtype=float)],
        [np.array([0.75], dtype=float), np.array([-0.5], dtype=float)],
        stim,
        ens,
        numRealizations=1,
        simType="poisson",
        seed=2,
        return_lambda=True,
        backend="python",
    )

    expected_stim = (
        np.convolve(stim_values[:, 0], np.array([1.0, 0.5], dtype=float), mode="full")[: time.size]
        + np.convolve(stim_values[:, 1], np.array([-0.25], dtype=float), mode="full")[: time.size]
    )
    expected_ens = (
        np.convolve(ens_values[:, 0], np.array([0.75], dtype=float), mode="full")[: time.size]
        + np.convolve(ens_values[:, 1], np.array([-0.5], dtype=float), mode="full")[: time.size]
    )
    expected_lambda = np.exp(np.clip(-2.0 + expected_stim + expected_ens, -20.0, 20.0))
    np.testing.assert_allclose(lambda_cov.data[:, 0], expected_lambda)


def test_decoding_aliases_produce_state_and_covariance_outputs() -> None:
    obs = np.array([[1.0], [0.5], [0.2]], dtype=float)
    a = np.array([[1.0]], dtype=float)
    h = np.array([[1.0]], dtype=float)
    q = np.array([[0.01]], dtype=float)
    r = np.array([[0.04]], dtype=float)
    x0 = np.array([0.0], dtype=float)
    p0 = np.array([[1.0]], dtype=float)

    out = DecodingAlgorithms.PPDecodeFilterLinear(obs, a, h, q, r, x0, p0)
    assert out["state"].shape == (3, 1)
    assert out["cov"].shape == (3, 1, 1)


def test_analysis_helper_surfaces_match_matlab_workflow_names() -> None:
    trial = _build_dense_trial()
    fit, ensemble_cov, tcc = compHistEnsCoeff(trial, [0.0, 0.05, 0.10], 1, [2], None, 0)
    assert isinstance(fit, FitResult)
    assert ensemble_cov.numCov >= 1
    assert tcc.numConfigs == 1

    all_fits, all_ensemble_cov, all_tcc = compHistEnsCoeffForAll(trial, [0.0, 0.05, 0.10], 0)
    assert len(all_fits) == 2
    assert all_ensemble_cov is not None
    assert len(all_tcc) == 2

    neighbor_fit, neighbor_tcc = computeNeighbors(trial, 1, trial.sampleRate, [0.0, 0.05, 0.10], 0)
    assert isinstance(neighbor_fit, FitResult)
    assert neighbor_tcc.numConfigs == 3

    sta = spikeTrigAvg(trial, 1, 0.2)
    assert sta.numCov == trial.covarColl.numCov

    granger_results, gamma_mat, phi_mat, deviance_mat, sig_mat = computeGrangerCausalityMatrix(trial, "GLM", 0.95, 0)
    assert len(granger_results) == 2
    assert gamma_mat.shape == (2, 2)
    assert phi_mat.shape == (2, 2)
    assert deviance_mat.shape == (2, 2)
    assert sig_mat.shape == (2, 2)


def test_history_and_events_roundtrip_in_workflow_context() -> None:
    history = History([0.0, 0.2, 0.4], minTime=0.0, maxTime=1.0)
    rebuilt_history = History.fromStructure(history.toStructure())
    assert rebuilt_history is not None
    np.testing.assert_allclose(rebuilt_history.windowTimes, history.windowTimes)

    events = Events([0.1, 0.4], ["start", "stop"], "m")
    rebuilt_events = Events.fromStructure(events.toStructure())
    assert rebuilt_events is not None
    assert rebuilt_events.eventColor == "m"
    assert rebuilt_events.eventLabels == ["start", "stop"]


def test_paper_example_one_supports_synthetic_fallback(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NSTAT_ALLOW_SYNTHETIC_DATA", "1")

    summary, payload = run_experiment1(Path(tmp_path), return_payload=True)

    assert summary["const_condition_spikes"] > 0.0
    assert summary["decreasing_condition_spikes"] > 0.0
    assert payload["constant_spike_times_s"].size > 0
    assert payload["washout_spike_times_s"].size > 0
