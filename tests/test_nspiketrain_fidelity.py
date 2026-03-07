from __future__ import annotations

import numpy as np

from nstat.nspikeTrain import nspikeTrain


def test_nspiketrain_constructor_runs_statistics_without_numpy_mode_error() -> None:
    train = nspikeTrain([0.0, 0.5, 1.0], "neuron")

    assert np.isfinite(train.avgFiringRate)
    assert np.isfinite(train.burstIndex)
    assert train.Lstatistic is not None


def test_nspiketrain_sigrep_uses_matlab_style_centers_and_inclusive_last_bin() -> None:
    train = nspikeTrain([0.0, 0.5, 1.0], "neuron", 0.5, 0.0, 1.0, makePlots=-1)

    sig = train.getSigRep()

    np.testing.assert_allclose(sig.time, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(sig.data[:, 0], [1.0, 1.0, 1.0])
    assert train.isSigRepBinary()


def test_nspiketrain_windowing_and_binary_limit_follow_matlab_semantics() -> None:
    train = nspikeTrain([0.1, 0.4, 0.9], "neuron", makePlots=-1)

    np.testing.assert_allclose(train.getSpikeTimes(0.1, 0.4), [0.1, 0.4])
    np.testing.assert_allclose(train.getISIs(0.1, 0.9), [0.3, 0.5])
    np.testing.assert_allclose(train.getMaxBinSizeBinary(), 0.3)


def test_nspiketrain_partition_and_min_isi_follow_matlab_semantics() -> None:
    train = nspikeTrain([0.1, 0.4, 0.6, 1.1], "neuron", 0.1, 0.0, 1.2, makePlots=-1)

    np.testing.assert_allclose(train.getMinISI(), 0.2)
    parts = train.partitionNST([0.0, 0.5, 1.2], normalizeTime=0)

    assert parts.numSpikeTrains == 2
    np.testing.assert_allclose(parts.getNST(1).spikeTimes, [0.1, 0.4])
    np.testing.assert_allclose(parts.getNST(2).spikeTimes, [0.1, 0.6])

    normalized_parts = train.partitionNST([0.0, 0.5, 1.0], normalizeTime=1)
    assert normalized_parts.minTime == 0.0
    assert normalized_parts.maxTime == 1.0
    np.testing.assert_allclose(normalized_parts.getNST(1).spikeTimes, [0.2, 0.8])


def test_nspiketrain_setsigrep_restore_and_field_access_match_matlab_surface() -> None:
    train = nspikeTrain([0.2, 0.6], "neuron", 0.2, 0.0, 1.0, makePlots=-1)

    train.setSigRep(0.1, 0.0, 1.0)
    assert train.sampleRate == 10.0
    assert train.isSigRepBinary()

    train.setMinTime(-0.5)
    train.setMaxTime(1.5)
    assert train.getFieldVal("name") == "neuron"
    assert train.getFieldVal("missing") == []

    train.restoreToOriginal()
    assert train.sampleRate == 5.0
    np.testing.assert_allclose([train.minTime, train.maxTime], [0.2, 0.6])
