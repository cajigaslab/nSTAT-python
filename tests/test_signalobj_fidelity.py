from __future__ import annotations

import numpy as np

from nstat.ConfidenceInterval import ConfidenceInterval
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


def test_signalobj_arithmetic_derivative_and_merge_preserve_matlab_style_shapes() -> None:
    sig = SignalObj([0.0, 1.0, 2.0], [[1.0, 2.0, 4.0], [2.0, 3.0, 5.0]], "stim", dataLabels=["x", "y"])
    offset = SignalObj([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], "offset", dataLabels=["x"])

    summed = sig + offset
    diffed = sig - 1.0
    scaled = 2.0 * sig
    merged = sig.getSubSignal(1).merge(sig.getSubSignal(2))

    np.testing.assert_allclose(summed.data[:, 0], [2.0, 3.0, 5.0])
    np.testing.assert_allclose(diffed.data[:, 1], [1.0, 2.0, 4.0])
    np.testing.assert_allclose(scaled.data[:, 0], [2.0, 4.0, 8.0])
    np.testing.assert_allclose(merged.data, sig.data)

    deriv = sig.derivative
    assert deriv.dimension == 2
    np.testing.assert_allclose(sig.derivativeAt(1.0), deriv.getValueAt(1.0))


def test_covariate_plus_minus_propagate_confidence_intervals() -> None:
    cov1 = Covariate([0.0, 1.0], [[1.0], [2.0]], "c1", dataLabels=["trial"])
    cov2 = Covariate([0.0, 1.0], [[0.5], [1.5]], "c2", dataLabels=["trial"])
    cov1.setConfInterval(ConfidenceInterval([0.0, 1.0], [[0.8, 1.2], [1.8, 2.2]]))
    cov2.setConfInterval(ConfidenceInterval([0.0, 1.0], [[0.4, 0.6], [1.4, 1.6]]))

    added = cov1 + cov2
    subtracted = cov1 - cov2

    assert added.isConfIntervalSet()
    assert subtracted.isConfIntervalSet()
    np.testing.assert_allclose(added.ci[0].bounds, [[1.2, 1.8], [3.2, 3.8]])
    np.testing.assert_allclose(subtracted.ci[0].bounds, [[0.2, 0.8], [0.2, 0.8]])
