from __future__ import annotations

import matplotlib.pyplot as plt
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
    assert train.sampleRate == 10.0
    np.testing.assert_allclose([train.minTime, train.maxTime], [0.2, 0.6])


def test_nspiketrain_compute_statistics_matches_matlab_style_burst_metrics() -> None:
    train = nspikeTrain([0.0, 0.001, 0.002, 0.007, 0.507, 0.508, 0.509, 0.514], "bursting", 0.001, 0.0, 0.6, makePlots=0)

    assert np.isfinite(train.B)
    assert np.isfinite(train.An)
    assert np.isfinite(train.burstIndex)
    assert train.numBursts >= 1
    assert train.burstSig is not None
    assert train.burstTimes.size == train.numBursts
    assert train.numSpikesPerBurst.size == train.numBursts


def test_nspiketrain_partition_rounds_windows_and_uses_matlab_constructor_defaults() -> None:
    train = nspikeTrain([0.0004, 0.0014, 0.0096], "neuron", 0.001, 0.0, 0.01, makePlots=-1)

    parts = train.partitionNST([0.00049, 0.00151, 0.0101], normalizeTime=0)

    assert parts.numSpikeTrains == 2
    np.testing.assert_allclose(parts.getNST(1).spikeTimes, [0.0004, 0.0014])
    np.testing.assert_allclose(parts.getNST(2).spikeTimes, [0.0076])
    assert parts.getNST(1).sampleRate == 1000.0


def test_nspiketrain_isi_plot_helpers_execute_and_return_matplotlib_objects() -> None:
    train = nspikeTrain([0.1, 0.12, 0.15, 0.5, 0.8], "neuron", 0.001, 0.0, 1.0, makePlots=0)

    line = train.plotISISpectrumFunction()
    joint_ax = train.plotJointISIHistogram()
    counts = train.plotISIHistogram()
    prob_ax = train.plotProbPlot()
    fig = train.plotExponentialFit()

    assert hasattr(line, "get_xdata")
    assert hasattr(joint_ax, "loglog")
    assert counts.sum() == train.getISIs().size
    assert hasattr(prob_ax, "plot")
    assert len(fig.axes) == 2
    plt.close("all")
