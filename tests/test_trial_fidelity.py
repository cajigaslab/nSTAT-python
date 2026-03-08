from __future__ import annotations

import numpy as np
import pytest

from nstat import Covariate, Events, History, Trial, TrialConfig, nspikeTrain
from nstat.ConfigColl import ConfigColl
from nstat.CovColl import CovColl
from nstat.nstColl import nstColl


def _make_covariates() -> tuple[Covariate, Covariate]:
    time = np.array([0.0, 0.5, 1.0], dtype=float)
    position = Covariate(time, np.column_stack([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]), "Position", "time", "s", "", ["x", "y"])
    stimulus = Covariate(time, [5.0, 6.0, 7.0], "Stimulus", "time", "s", "a.u.", ["stim"])
    return position, stimulus


def _make_spikes() -> tuple[nspikeTrain, nspikeTrain]:
    n1 = nspikeTrain([0.0, 0.5, 1.0], "n1", 0.5, 0.0, 1.0, makePlots=-1)
    n2 = nspikeTrain([0.25, 0.75], "n2", 0.5, 0.0, 1.0, makePlots=-1)
    return n1, n2


def test_covcoll_masking_selector_and_time_matrix() -> None:
    position, stimulus = _make_covariates()
    coll = CovColl([position, stimulus])

    assert coll.numCov == 2
    assert coll.getCovIndFromName("Position") == 1
    assert coll.getCovIndFromName("Stimulus") == 2

    coll.setMask([["Position", "x"], ["Stimulus"]])
    time, matrix, labels = coll.matrixWithTime()

    np.testing.assert_allclose(time, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(matrix, [[0.0, 5.0], [1.0, 6.0], [2.0, 7.0]])
    assert labels == ["x", "stim"]
    assert coll.getCovLabelsFromMask() == ["x", "stim"]

    coll.setCovShift(0.5)
    shifted = coll.getCov("Stimulus")
    np.testing.assert_allclose(shifted.time, [0.5, 1.0, 1.5])


def test_nstcoll_neighbors_mask_and_data_matrix() -> None:
    train1, train2 = _make_spikes()
    coll = nstColl()
    coll.addToColl([train1, train2])

    assert coll.numSpikeTrains == 2
    assert coll.getNSTnames() == ["n1", "n2"]
    assert coll.getUniqueNSTnames() == ["n1", "n2"]

    coll.setNeighbors()
    assert coll.getNeighbors(1) == [2]
    assert coll.getNeighbors(2) == [1]

    coll.setMask([1])
    assert coll.getIndFromMask() == [1]
    np.testing.assert_allclose(coll.getMaxBinSizeBinary(), 0.5)

    matrix = coll.dataToMatrix([1, 2], 0.5, 0.0, 1.0)
    np.testing.assert_allclose(matrix, [[1.0, 0.0], [1.0, 1.0], [1.0, 1.0]])


def test_trialconfig_and_configcoll_apply_and_roundtrip() -> None:
    position, stimulus = _make_covariates()
    train1, train2 = _make_spikes()
    trial = Trial(nstColl([train1, train2]), CovColl([position, stimulus]))

    cfg = TrialConfig(
        covMask=[["Position", "x"], ["Stimulus"]],
        sampleRate=2.0,
        history=[0.0, 0.5, 1.0],
        covLag=0.5,
        name="stim_pos",
    )
    cfg.setConfig(trial)

    assert round(trial.sampleRate, 3) == 2.0
    assert trial.isHistSet()
    assert trial.getCovLabelsFromMask() == ["x", "stim"]

    roundtrip = TrialConfig.fromStructure(cfg.toStructure())
    assert roundtrip.name == ""
    assert roundtrip.covLag == "stim_pos"
    assert roundtrip.ensCovMask == 0.5
    assert roundtrip.covariate_names == ["Position", "Stimulus"]

    cfg2 = TrialConfig(
        covMask=[["Stimulus"]],
        sampleRate=2.0,
        history=[],
        covLag=[],
        name="manual",
    )
    configs = ConfigColl([cfg, cfg2])
    assert configs.numConfigs == 2
    assert configs.getConfigNames() == ["stim_pos", "manual"]
    subset = configs.getSubsetConfigs([1, 2])
    assert subset.numConfigs == 2
    rebuilt = ConfigColl.fromStructure(configs.toStructure())
    assert rebuilt.getConfigNames() == ["Fit 1", "Fit 2"]
    assert rebuilt.getConfig(1).name == ""
    assert rebuilt.getConfig(1).covLag == "stim_pos"


def test_trial_partition_history_design_matrix_and_spike_vector() -> None:
    position, stimulus = _make_covariates()
    train1, train2 = _make_spikes()
    events = Events([0.25, 0.75], ["cue", "reward"], "g")
    hist = History([0.0, 0.5, 1.0])
    trial = Trial(nstColl([train1, train2]), CovColl([position, stimulus]), events, hist)

    assert trial.getEvents() is events
    assert trial.isHistSet()
    assert len(trial.getHistLabels()) == 2

    trial.setTrialPartition([0.0, 0.5, 1.0])
    np.testing.assert_allclose(trial.getTrialPartition(), [0.0, 0.5, 0.5, 1.0])
    trial.setTrialTimesFor("validation")
    np.testing.assert_allclose([trial.minTime, trial.maxTime], [0.5, 1.0])

    design = trial.getDesignMatrix(1)
    assert design.shape[1] == 5
    spikes = trial.getSpikeVector()
    assert spikes.shape[1] == 2
    np.testing.assert_allclose(trial.getSpikeVector(1).reshape(-1), spikes[:, 0])


def test_events_validation_and_history_collection_output() -> None:
    with pytest.raises(ValueError, match="Number of eventTimes"):
        Events([0.1, 0.2], ["one"])

    events = Events([0.1], ["cue"], "b")
    rebuilt = Events.fromStructure(events.toStructure())
    assert rebuilt is not None
    assert rebuilt.eventColor == "b"
    assert rebuilt.eventLabels == ["cue"]

    history = History([0.0, 0.5, 1.0])
    train, _ = _make_spikes()
    hist_cov = history.computeHistory(train)
    assert hist_cov.numCov == 1
    np.testing.assert_allclose(hist_cov.dataToMatrix().shape, (3, 2))
