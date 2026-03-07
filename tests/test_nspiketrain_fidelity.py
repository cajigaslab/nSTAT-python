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
