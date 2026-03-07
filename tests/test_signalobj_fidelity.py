from __future__ import annotations

import numpy as np

from nstat.Covariate import Covariate
from nstat.SignalObj import SignalObj


def test_signalobj_normalizes_channel_orientation_and_uses_one_based_selection() -> None:
    sig = SignalObj(
        [0.0, 0.5, 1.0],
        [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
        "stim",
        "time",
        "s",
        "a.u.",
        ["x", "y"],
    )

    assert sig.data.shape == (3, 2)
    np.testing.assert_allclose(sig.data[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(sig.data[:, 1], [10.0, 20.0, 30.0])

    sub = sig.getSubSignal(2)
    assert sub.dimension == 1
    assert sub.dataLabels == ["y"]
    np.testing.assert_allclose(sub.data[:, 0], [10.0, 20.0, 30.0])


def test_signalobj_time_window_padding_matches_matlab_style_hold_values() -> None:
    sig = SignalObj([0.0, 1.0], [5.0, 7.0], "stim")
    sig.setMinTime(-1.0, holdVals=1)
    sig.setMaxTime(2.0, holdVals=1)

    np.testing.assert_allclose(sig.time, [-1.0, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(sig.data[:, 0], [5.0, 5.0, 7.0, 7.0])


def test_covariate_compute_mean_plus_ci_uses_timewise_mean() -> None:
    cov = Covariate(
        [0.0, 1.0, 2.0],
        [[1.0, 3.0], [2.0, 4.0], [6.0, 8.0]],
        "lambda",
        "time",
        "s",
        "spikes/sec",
        ["trial1", "trial2"],
    )

    mean_cov = cov.computeMeanPlusCI(alphaVal=0.5)

    np.testing.assert_allclose(mean_cov.time, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(mean_cov.data[:, 0], [2.0, 3.0, 7.0])
    assert mean_cov.isConfIntervalSet()
    assert mean_cov.ci is not None
    assert len(mean_cov.ci) == 1
