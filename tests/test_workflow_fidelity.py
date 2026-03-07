from __future__ import annotations

import numpy as np

from nstat import Analysis, CIF, CIFModel, DecodingAlgorithms, FitResSummary, Trial, TrialConfig
from nstat.ConfigColl import ConfigColl
from nstat.CovColl import CovColl
from nstat.Covariate import Covariate
from nstat.Events import Events
from nstat.History import History
from nstat.FitResult import FitResult
from nstat.nstColl import nstColl
from nstat.nspikeTrain import nspikeTrain


def _build_trial() -> Trial:
    time = np.arange(0.0, 1.0, 0.1)
    stim = Covariate(time, np.sin(2 * np.pi * time), "Stimulus", "time", "s", "", ["stim"])
    vel = Covariate(time, np.cos(2 * np.pi * time), "Velocity", "time", "s", "", ["vel"])
    spikes = nstColl(
        [
            nspikeTrain([0.1, 0.3, 0.7], "1", 0.1, 0.0, 0.9, makePlots=-1),
            nspikeTrain([0.2, 0.5, 0.8], "2", 0.1, 0.0, 0.9, makePlots=-1),
        ]
    )
    return Trial(spikes, CovColl([stim, vel]), Events([0.2], ["cue"]), History([0.0, 0.1, 0.2]))


def test_analysis_returns_matlab_style_fitresult_surface() -> None:
    trial = _build_trial()
    configs = ConfigColl(
        [
            TrialConfig(covMask=[["Stimulus"]], sampleRate=10.0, history=[0.0, 0.1, 0.2], name="stim_hist"),
            TrialConfig(covMask=[["Velocity"]], sampleRate=10.0, name="vel_only"),
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
    assert fit.getCoeffs(1).shape[0] >= 2
    assert fit.getHistCoeffs(1).shape[0] == 2


def test_fitresult_roundtrip_and_summary_preserve_core_metadata() -> None:
    trial = _build_trial()
    configs = ConfigColl([TrialConfig(covMask=[["Stimulus"]], sampleRate=10.0, name="stim_only")])
    fits = Analysis.RunAnalysisForAllNeurons(trial, configs)

    rebuilt = FitResult.fromStructure(fits[0].toStructure())
    assert rebuilt.numResults == fits[0].numResults
    assert rebuilt.configNames == fits[0].configNames
    np.testing.assert_allclose(rebuilt.AIC, fits[0].AIC)

    summary = FitResSummary(fits)
    assert summary.numNeurons == 2
    assert summary.numResults == 1
    assert summary.fitNames == ["stim_only"]
    assert summary.AIC.shape == (1,)


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

    model = CIFModel(time, np.array([5.0, 6.0, 7.0]), name="lambda")
    sim2 = model.simulate(num_realizations=1, seed=1)
    assert sim2.numSpikeTrains == 1


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
