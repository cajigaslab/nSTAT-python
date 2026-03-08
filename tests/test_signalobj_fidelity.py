from __future__ import annotations

import matplotlib.pyplot as plt
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


def test_signalobj_integral_matches_matlab_style_accumulator_and_labels() -> None:
    sig = SignalObj([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], "stim", "time", "s", "a.u.", ["x"])

    integrated = sig.integral()

    np.testing.assert_allclose(integrated.data[:, 0], [1.0, 3.0, 6.0])
    assert integrated.yunits == "a.u.*s"
    assert integrated.name.startswith("\\int_")
    assert integrated.dataLabels[0].startswith("\\int_")


def test_signalobj_makecompatible_and_correlation_helpers_follow_matlab_surface() -> None:
    s1 = SignalObj([0.0, 1.0, 2.0], [1.0, 0.0, -1.0], "s1", dataLabels=["x"])
    s2 = SignalObj([0.5, 1.5], [2.0, 4.0], "s2", dataLabels=["y"])

    s1c, s2c = s1.makeCompatible(s2, holdVals=1)

    np.testing.assert_allclose(s1c.time, s2c.time)
    assert s1c.sampleRate == s2c.sampleRate
    assert s1c.minTime == 0.0
    assert s1c.maxTime == 2.0
    np.testing.assert_allclose(s2c.data[:, 0], [2.0, 4.0, 4.0])

    acf = s1.autocorrelation()
    assert acf.name == "ACF(s1)"
    np.testing.assert_allclose(acf.time[acf.time.size // 2], 0.0)

    xcf = s1.crosscorrelation(SignalObj([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], "s3", dataLabels=["z"]))
    assert xcf.dimension == 1
    assert xcf.xlabelval == "Lag"

    xcorr_sig = s1.xcorr()
    assert xcorr_sig.xlabelval == "\\Delta \\tau"
    assert np.all(xcorr_sig.time >= 0.0)
    assert xcorr_sig.dimension == 1
    plt.close("all")


def test_signalobj_math_and_summary_methods_match_matlab_surface() -> None:
    sig = SignalObj([0.0, 1.0, 2.0], [[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]], "stim", dataLabels=["x", "y"], yunits="a.u.")

    abs_sig = abs(SignalObj([0.0, 1.0], [-1.0, 2.0], "signed", yunits="a.u.", dataLabels=["x"]))
    log_sig = SignalObj([1.0, 2.0], [1.0, np.e], "positive", yunits="Hz", dataLabels=["x"]).log()
    med = sig.median()
    mod = sig.mode()
    max_vals, max_idx, max_time = sig.max()
    min_vals, min_idx, min_time = sig.min()

    np.testing.assert_allclose(abs_sig.data[:, 0], [1.0, 2.0])
    assert abs_sig.name == "|signed|"
    np.testing.assert_allclose(log_sig.data[:, 0], [0.0, 1.0])
    assert log_sig.yunits == "ln(Hz)"
    np.testing.assert_allclose(med.data[0], [2.0, 2.0])
    np.testing.assert_allclose(mod.data[0], [1.0, 1.0])
    np.testing.assert_allclose(max_vals, [3.0, 4.0])
    np.testing.assert_array_equal(max_idx, [1, 2])
    np.testing.assert_allclose(max_time, [1.0, 2.0])
    np.testing.assert_allclose(min_vals, [1.0, 1.0])
    np.testing.assert_array_equal(min_idx, [0, 1])
    np.testing.assert_allclose(min_time, [0.0, 1.0])


def test_confidence_interval_line_plot_ignores_string_color_like_matlab() -> None:
    ci = ConfidenceInterval([0.0, 1.0], [[0.8, 1.2], [1.8, 2.2]], "CI", "time", "s", "a.u.", ["lo", "hi"], ["-.k"])

    fig1, ax1 = plt.subplots()
    lines_default = ci.plot(color="r", drawPatches=0, ax=ax1)
    default_colors = [line.get_color() for line in lines_default]

    fig2, ax2 = plt.subplots()
    lines_numeric = ci.plot(color=(0.2, 0.4, 0.6), drawPatches=0, ax=ax2)
    numeric_colors = [line.get_color() for line in lines_numeric]

    assert default_colors != ["r", "r"]
    assert numeric_colors == [(0.2, 0.4, 0.6), (0.2, 0.4, 0.6)]

    plt.close(fig1)
    plt.close(fig2)
