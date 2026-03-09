from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.io import loadmat

from nstat import (
    Analysis,
    CIF,
    ConfidenceInterval,
    ConfigColl,
    CovColl,
    Covariate,
    DecodingAlgorithms,
    Events,
    FitResult,
    FitResSummary,
    History,
    SignalObj,
    Trial,
    TrialConfig,
    nspikeTrain,
    nstColl,
    simulate_two_neuron_network,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "parity" / "fixtures" / "matlab_gold"


def _load_fixture(name: str) -> dict[str, np.ndarray]:
    return loadmat(FIXTURE_ROOT / name, squeeze_me=True, struct_as_record=False)


def _scalar(payload: dict[str, np.ndarray], key: str) -> float:
    return float(np.asarray(payload[key], dtype=float).reshape(-1)[0])


def _vector(payload: dict[str, np.ndarray], key: str) -> np.ndarray:
    return np.asarray(payload[key], dtype=float).reshape(-1)


def _string(payload: dict[str, np.ndarray], key: str) -> str:
    value = payload[key]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    if arr.shape == ():
        return str(arr.item())
    return str(arr.reshape(-1)[0])


def _string_list(payload: dict[str, np.ndarray], key: str) -> list[str]:
    value = payload[key]
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    arr = np.asarray(value, dtype=object)
    if arr.shape == ():
        return [str(arr.item())]
    return [str(item) for item in arr.reshape(-1)]


def _object_vectors(payload: dict[str, np.ndarray], key: str) -> list[np.ndarray]:
    value = np.asarray(payload[key], dtype=object)
    if value.shape == ():
        value = np.asarray([value.item()], dtype=object)
    if value.dtype != object:
        return [np.asarray(value, dtype=float).reshape(-1)]
    if value.ndim == 1 and all(not isinstance(item, (list, tuple, np.ndarray)) for item in value.reshape(-1)):
        return [np.asarray(value, dtype=float).reshape(-1)]
    return [np.asarray(item, dtype=float).reshape(-1) for item in value.reshape(-1)]


def test_signalobj_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("signalobj_exactness.mat")
    signal = SignalObj(_vector(payload, "time"), np.asarray(payload["data"], dtype=float), "sig", "time", "s", "u", ["x1", "x2"])
    signal_1 = signal.getSubSignal(1)
    signal_2 = SignalObj(np.arange(0.05, 0.5, 0.1), [0.0, 1.0, 0.0, -1.0, 0.0], "sig2", "time", "s", "u", ["x3"])

    filtered = signal.filter(_vector(payload, "filter_b"), _vector(payload, "filter_a"))
    derivative = signal.derivative
    integral = signal.integral()
    resampled = signal.resample(_scalar(payload, "resample_rate"))
    xcorr = signal.getSubSignal(1).xcorr(signal.getSubSignal(2), int(_scalar(payload, "xcorr_maxlag")))
    compatible_left, compatible_right = signal_1.makeCompatible(signal_2, holdVals=1)

    np.testing.assert_allclose(filtered.data, np.asarray(payload["filtered_data"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(derivative.data, np.asarray(payload["derivative_data"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(integral.data, np.asarray(payload["integral_data"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(resampled.time, _vector(payload, "resampled_time"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(resampled.data, np.asarray(payload["resampled_data"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(xcorr.time, _vector(payload, "xcorr_time"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(xcorr.data.reshape(-1), _vector(payload, "xcorr_data"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(compatible_left.time, _vector(payload, "compat_time"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(compatible_left.data.reshape(-1), _vector(payload, "compat_left_data"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(compatible_right.data.reshape(-1), _vector(payload, "compat_right_data"), rtol=1e-8, atol=1e-10)


def test_nspiketrain_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("nspiketrain_exactness.mat")
    nst = nspikeTrain(
        _vector(payload, "spikeTimes"),
        "nst",
        _scalar(payload, "binwidth"),
        _scalar(payload, "minTime"),
        _scalar(payload, "maxTime"),
        "time",
        "s",
        "spikes",
        "spk",
        0,
    )

    sig = nst.getSigRep(_scalar(payload, "binwidth"), _scalar(payload, "minTime"), _scalar(payload, "maxTime"))
    parts = nst.partitionNST([_scalar(payload, "minTime"), 0.2, _scalar(payload, "maxTime")])

    np.testing.assert_allclose(sig.time, _vector(payload, "sig_time"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sig.data[:, 0], _vector(payload, "sig_data"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(nst.getISIs(), _vector(payload, "isis"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(nst.avgFiringRate), _scalar(payload, "avgFiringRate"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(nst.B), _scalar(payload, "B"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(nst.An), _scalar(payload, "An"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(nst.burstIndex), _scalar(payload, "burstIndex"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(parts.getNST(1).spikeTimes, _vector(payload, "part1_spikes"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(parts.getNST(2).spikeTimes, _vector(payload, "part2_spikes"), rtol=1e-8, atol=1e-10)

    restore_train = nspikeTrain(_vector(payload, "spikeTimes"), "restore", 0.2, -0.1, 0.8, "time", "s", "spikes", "spk", -1)
    restore_train.setSigRep(0.1, -0.1, 0.8)
    restore_train.setMinTime(-0.3)
    restore_train.setMaxTime(1.1)
    restore_train.restoreToOriginal()

    np.testing.assert_allclose(float(restore_train.minTime), _scalar(payload, "restore_min_time"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(float(restore_train.maxTime), _scalar(payload, "restore_max_time"), rtol=1e-12, atol=1e-12)

    burst_train = nspikeTrain([0.0, 0.001, 0.002, 0.007, 0.507, 0.508, 0.509, 0.514], "bursting", 0.001, 0.0, 0.6, "time", "s", "spikes", "spk", 0)
    np.testing.assert_allclose(float(burst_train.avgSpikesPerBurst), _scalar(payload, "burst_avgSpikesPerBurst"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(burst_train.stdSpikesPerBurst), _scalar(payload, "burst_stdSpikesPerBurst"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(burst_train.numBursts), _scalar(payload, "burst_numBursts"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(burst_train.numSpikesPerBurst, _vector(payload, "burst_numSpikesPerBurst"), rtol=1e-8, atol=1e-10)

    fig, ax = plt.subplots()
    try:
        line = nst.plotISISpectrumFunction()
        np.testing.assert_allclose(np.asarray(line.get_xdata(), dtype=float), _vector(payload, "isi_spectrum_x"), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.asarray(line.get_ydata(), dtype=float), _vector(payload, "isi_spectrum_y"), rtol=1e-12, atol=1e-12)
    finally:
        plt.close("all")

    fig, ax = plt.subplots()
    try:
        joint_ax = nst.plotJointISIHistogram()
        joint_lines = list(joint_ax.lines)
        expected_styles = _string_list(payload, "joint_isi_style")
        assert len(joint_lines) == len(expected_styles)
        for line, expected_x, expected_y, expected_style in zip(
            joint_lines,
            _object_vectors(payload, "joint_isi_x"),
            _object_vectors(payload, "joint_isi_y"),
            expected_styles,
            strict=True,
        ):
            np.testing.assert_allclose(np.asarray(line.get_xdata(), dtype=float).reshape(-1), expected_x, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(np.asarray(line.get_ydata(), dtype=float).reshape(-1), expected_y, rtol=1e-12, atol=1e-12)
            assert str(line.get_linestyle()).lower() == str(expected_style).lower()
    finally:
        plt.close("all")

    fig, ax = plt.subplots()
    try:
        counts = nst.plotISIHistogram(handle=ax)
        np.testing.assert_allclose(np.asarray(counts, dtype=float).reshape(-1), _vector(payload, "isi_hist_counts"), rtol=1e-12, atol=1e-12)
        patches = list(ax.patches)
        assert patches
        np.testing.assert_allclose(np.asarray(patches[0].get_facecolor()[:3], dtype=float), _vector(payload, "isi_hist_face_color"), rtol=1e-12, atol=1e-12)
        expected_edge = _string(payload, "isi_hist_edge_color")
        if expected_edge.lower() == "none":
            edge = np.asarray(patches[0].get_edgecolor())
            assert edge.size == 0 or edge.reshape(-1, 4)[0, 3] == 0.0
        else:
            np.testing.assert_allclose(np.asarray(patches[0].get_edgecolor()[:3], dtype=float), _vector(payload, "isi_hist_edge_color"), rtol=1e-12, atol=1e-12)
    finally:
        plt.close("all")

    fig, ax = plt.subplots()
    try:
        prob_ax = nst.plotProbPlot(handle=ax)
        prob_lines = list(prob_ax.lines)
        expected_styles = _string_list(payload, "probplot_style")
        assert len(prob_lines) == len(expected_styles)
        for line, expected_x, expected_y, expected_style in zip(
            prob_lines,
            _object_vectors(payload, "probplot_x"),
            _object_vectors(payload, "probplot_y"),
            expected_styles,
            strict=True,
        ):
            np.testing.assert_allclose(np.asarray(line.get_xdata(), dtype=float).reshape(-1), expected_x, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(np.asarray(line.get_ydata(), dtype=float).reshape(-1), expected_y, rtol=1e-8, atol=1e-10)
            assert str(line.get_linestyle()).lower() == str(expected_style).lower()
    finally:
        plt.close("all")

    fig = nst.plotExponentialFit()
    try:
        assert len(fig.axes) == int(_scalar(payload, "expfit_num_axes"))
    finally:
        plt.close(fig)


def test_covariate_and_confidence_interval_match_matlab_gold_fixture() -> None:
    payload = _load_fixture("covariate_exactness.mat")
    time = _vector(payload, "time")
    replicates = np.asarray(payload["replicates"], dtype=float)

    cov = Covariate(time, replicates, "Stimulus", "time", "s", "a.u.", ["r1", "r2", "r3", "r4"])
    mean_cov = cov.computeMeanPlusCI(0.05)
    cov_single = Covariate(time, np.mean(replicates, axis=1), "StimulusSingle", "time", "s", "a.u.", ["stim"])
    cov_single.setConfInterval(
        ConfidenceInterval(
            time,
            np.asarray(payload["explicit_ci"], dtype=float),
            "CI",
            "time",
            "s",
            "a.u.",
        )
    )

    np.testing.assert_allclose(mean_cov.data[:, 0], _vector(payload, "mean_data"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(mean_cov.ci[0].bounds, np.asarray(payload["mean_ci"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(cov_single.ci[0].bounds, np.asarray(payload["explicit_ci"], dtype=float), rtol=1e-8, atol=1e-10)
    structure = cov_single.toStructure()
    np.testing.assert_allclose(np.asarray(structure["time"], dtype=float), _vector(payload, "structure_time"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(structure["data"], dtype=float).reshape(-1), _vector(payload, "structure_data"), rtol=1e-12, atol=1e-12)
    assert structure["name"] == _string(payload, "structure_name")
    assert list(structure["dataLabels"]) == _string_list(payload, "structure_dataLabels")
    assert list(structure["plotProps"]) == _string_list(payload, "structure_plotProps")
    assert isinstance(structure["ci"], dict)
    np.testing.assert_allclose(np.asarray(structure["ci"]["signals"]["values"], dtype=float), np.asarray(payload["structure_ci_values"], dtype=float), rtol=1e-12, atol=1e-12)
    assert structure["ci"]["name"] == _string(payload, "structure_ci_name")

    roundtrip = Covariate.fromStructure(structure)
    np.testing.assert_allclose(roundtrip.data.reshape(-1), _vector(payload, "roundtrip_data"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(roundtrip.ci[0].bounds, np.asarray(payload["roundtrip_ci"], dtype=float), rtol=1e-12, atol=1e-12)
    assert list(roundtrip.dataLabels) == _string_list(payload, "roundtrip_dataLabels")

    fig, ax = plt.subplots()
    try:
        cov_single.plot(handle=ax)
        line_handles = list(ax.lines)
        line_colors = np.asarray([mcolors.to_rgb(line.get_color()) for line in line_handles], dtype=float)
        np.testing.assert_allclose(line_colors, np.asarray(payload["plot_line_colors"], dtype=float), rtol=1e-12, atol=1e-12)
    finally:
        plt.close(fig)


def test_confidence_interval_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("confidence_interval_exactness.mat")
    ci = ConfidenceInterval(
        _vector(payload, "time"),
        np.asarray(payload["bounds"], dtype=float),
        "CI",
        "time",
        "s",
        "a.u.",
        ["lo", "hi"],
        ["-.k"],
    )
    ci.setColor(_string(payload, "color"))
    ci.setValue(_scalar(payload, "value"))
    structure = ci.toStructure()
    roundtrip = ConfidenceInterval.fromStructure(structure)

    np.testing.assert_allclose(ci.dataToMatrix(), np.asarray(payload["bounds"], dtype=float), rtol=1e-12, atol=1e-12)
    assert ci.name == _string(payload, "name")
    assert ci.color == _string(payload, "color")
    assert ci.plotProps == _string_list(payload, "plotProps")
    np.testing.assert_allclose(ci.value, _scalar(payload, "value"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(structure["time"], dtype=float), _vector(payload, "structure_time"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(structure["signals"]["values"], dtype=float), np.asarray(payload["structure_values"], dtype=float), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(roundtrip.dataToMatrix(), np.asarray(payload["roundtrip_bounds"], dtype=float), rtol=1e-12, atol=1e-12)
    assert roundtrip.color == _string(payload, "roundtrip_color")
    np.testing.assert_allclose(roundtrip.value, _scalar(payload, "roundtrip_value"), rtol=1e-12, atol=1e-12)
    assert roundtrip.name == _string(payload, "roundtrip_name")
    assert roundtrip.plotProps == _string_list(payload, "roundtrip_plotProps")

    fig, ax = plt.subplots()
    try:
        lines = ci.plot(_string(payload, "color"), 0.2, 0, ax=ax)
        actual_colors = np.asarray([mcolors.to_rgb(line.get_color()) for line in lines], dtype=float)
        np.testing.assert_allclose(actual_colors, np.asarray(payload["line_plot_colors"], dtype=float), rtol=1e-12, atol=1e-12)
    finally:
        plt.close(fig)

    fig, ax = plt.subplots()
    try:
        patch = ci.plot(np.asarray(payload["patch_face_color"], dtype=float), _scalar(payload, "patch_face_alpha"), 1, ax=ax)
        np.testing.assert_allclose(patch.get_facecolor()[0, :3], np.asarray(payload["patch_face_color"], dtype=float), rtol=1e-12, atol=1e-12)
        assert patch.get_edgecolor() is not None
        np.testing.assert_allclose(patch.get_alpha(), _scalar(payload, "patch_face_alpha"), rtol=1e-12, atol=1e-12)
        edge = _string(payload, "patch_edge_color")
        if edge == "none":
            edge_color = np.asarray(patch.get_edgecolor())
            assert edge_color.size == 0 or edge_color.reshape(-1, 4)[0, 3] == 0.0
    finally:
        plt.close(fig)


def test_nstcoll_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("nstcoll_exactness.mat")
    n1 = nspikeTrain(_vector(payload, "firstSpikeTimes"), "1", 10.0, 0.0, 0.5, "time", "s", "spikes", "spk", -1)
    n2 = nspikeTrain(_vector(payload, "secondSpikeTimes"), "2", 10.0, 0.0, 0.5, "time", "s", "spikes", "spk", -1)
    coll = nstColl([n1, n2])
    collapsed = coll.toSpikeTrain()
    coll.setNeighbors()

    np.testing.assert_equal(coll.numSpikeTrains, int(_scalar(payload, "numSpikeTrains")))
    assert coll.getNST(1).name == _string(payload, "firstName")
    np.testing.assert_allclose(coll.dataToMatrix([1, 2], 0.1, 0.0, 0.5), np.asarray(payload["dataMatrix"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(collapsed.spikeTimes, _vector(payload, "collapsedSpikeTimes"), rtol=1e-8, atol=1e-10)
    assert collapsed.name == _string(payload, "collapsedName")
    np.testing.assert_allclose(float(collapsed.minTime), _scalar(payload, "collapsedMinTime"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(float(collapsed.maxTime), _scalar(payload, "collapsedMaxTime"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(float(collapsed.sampleRate), _scalar(payload, "collapsedSampleRate"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(float(coll.getFirstSpikeTime()), _scalar(payload, "firstSpikeTime"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(float(coll.getLastSpikeTime()), _scalar(payload, "lastSpikeTime"), rtol=1e-12, atol=1e-12)
    assert coll.isSigRepBinary() == bool(_scalar(payload, "binarySigRep"))
    assert coll.BinarySigRep() == bool(_scalar(payload, "binarySigRep"))
    assert coll.getNSTnameFromInd(1) == _string(payload, "nstNameFromInd1")
    nst_from_name = coll.getNSTFromName("1")
    assert isinstance(nst_from_name, nspikeTrain)
    np.testing.assert_allclose(nst_from_name.spikeTimes, _vector(payload, "nstFromName1_spikeTimes"), rtol=1e-12, atol=1e-12)
    fieldVal, neuronNumbers = coll.getFieldVal("avgFiringRate")
    np.testing.assert_allclose(fieldVal, _vector(payload, "fieldVal_avgFiringRate"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(neuronNumbers, _vector(payload, "fieldVal_neuronNumbers"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(coll.getNeighbors(1), dtype=float), _vector(payload, "neighbors1"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(coll.getNeighbors(2), dtype=float), _vector(payload, "neighbors2"), rtol=1e-12, atol=1e-12)
    ensembleCov = coll.getEnsembleNeuronCovariates(1, [], [0.0, 0.1])
    assert ensembleCov.getAllCovLabels() == _string_list(payload, "ensemble_labels")
    np.testing.assert_allclose(ensembleCov.dataToMatrix(), np.asarray(payload["ensemble_matrix"], dtype=float).reshape(-1, 1), rtol=1e-12, atol=1e-12)
    psthCov = coll.psth(0.1, [1, 2], 0.0, 0.5)
    np.testing.assert_allclose(psthCov.time, _vector(payload, "psth_time"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(psthCov.data.reshape(-1), _vector(payload, "psth_data"), rtol=1e-12, atol=1e-12)


def test_trialconfig_and_configcoll_match_matlab_gold_fixture() -> None:
    payload = _load_fixture("config_exactness.mat")
    cfg = TrialConfig([["Position", "x"], ["Stimulus"]], 2.0, [0.0, 0.5, 1.0], [], [], 0.5, "stim_pos")
    cfg2 = TrialConfig([["Stimulus"]], 2.0, [], [], [], [], "manual")
    default_coll = ConfigColl()
    empty_coll = ConfigColl([])
    structure = cfg.toStructure()
    roundtrip = TrialConfig.fromStructure(structure)
    coll = ConfigColl([cfg, cfg2])
    subset = coll.getSubsetConfigs([1, 2])
    rebuilt = ConfigColl.fromStructure(coll.toStructure())
    renamed = ConfigColl([cfg, cfg2])
    renamed.setConfigNames("", [1])

    assert cfg.name == _string(payload, "cfg_name")
    np.testing.assert_allclose(float(cfg.sampleRate), _scalar(payload, "cfg_sampleRate"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(float(cfg.covLag), _scalar(payload, "cfg_covLag"), rtol=1e-12, atol=1e-12)
    assert roundtrip.name == _string(payload, "roundtrip_name")
    assert roundtrip.covLag == _string(payload, "roundtrip_covLag")
    np.testing.assert_allclose(float(roundtrip.ensCovMask), _scalar(payload, "roundtrip_ensCovMask"), rtol=1e-12, atol=1e-12)
    assert coll.getConfigNames() == _string_list(payload, "config_names")
    assert subset.getConfigNames() == _string_list(payload, "subset_names")
    assert rebuilt.getConfigNames() == _string_list(payload, "rebuilt_names")
    assert rebuilt.getConfig(1).name == _string(payload, "rebuilt_first_name")
    assert rebuilt.getConfig(1).covLag == _string(payload, "rebuilt_first_covLag")
    np.testing.assert_allclose(float(rebuilt.getConfig(1).ensCovMask), _scalar(payload, "rebuilt_first_ensCovMask"), rtol=1e-12, atol=1e-12)
    assert default_coll.numConfigs == int(_scalar(payload, "default_numConfigs"))
    assert default_coll.getConfigNames() == _string_list(payload, "default_names")
    assert empty_coll.numConfigs == int(_scalar(payload, "empty_numConfigs"))
    assert empty_coll.getConfigNames() == _string_list(payload, "empty_names")
    assert renamed.getConfigNames() == _string_list(payload, "renamed_names")
    with pytest.raises(AttributeError) as excinfo:
        ConfigColl("abc")
    assert _string(payload, "string_error_identifier") in {"", "MATLAB:structRefFromNonStruct"}
    assert "name" in str(excinfo.value)

    time = np.array([0.0, 0.5, 1.0], dtype=float)
    position = Covariate(time, np.column_stack([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]), "Position", "time", "s", "", ["x", "y"])
    stimulus = Covariate(time, [5.0, 6.0, 7.0], "Stimulus", "time", "s", "a.u.", ["stim"])
    n1 = nspikeTrain([0.0, 0.5, 1.0], "n1", 2.0, 0.0, 1.0, "time", "s", "spikes", "spk", -1)
    n2 = nspikeTrain([0.25, 0.75], "n2", 2.0, 0.0, 1.0, "time", "s", "spikes", "spk", -1)

    cfg_applied = TrialConfig([["Position", "x"], ["Stimulus"]], 4.0, [0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [[0, 1], [1, 0]], 0.25, "stim_pos")
    trial = Trial(nstColl([n1, n2]), CovColl([position, stimulus]))
    cfg_applied.setConfig(trial)

    np.testing.assert_allclose(float(trial.sampleRate), _scalar(payload, "applied_sampleRate"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(trial.flattenCovMask(), dtype=float), _vector(payload, "applied_flat_cov_mask"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(trial.history.windowTimes, dtype=float), _vector(payload, "applied_history_windowTimes"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(trial.ensCovHist.windowTimes, dtype=float), _vector(payload, "applied_ens_history_windowTimes"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(trial.ensCovMask, dtype=float), np.asarray(payload["applied_ens_mask"], dtype=float), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(trial.covarColl.getCov(1).time, dtype=float), _vector(payload, "applied_shifted_position_time"), rtol=1e-12, atol=1e-12)

    trial_from_coll = Trial(nstColl([n1, n2]), CovColl([position, stimulus]))
    ConfigColl([cfg_applied]).setConfig(trial_from_coll, 1)
    np.testing.assert_allclose(float(trial_from_coll.sampleRate), _scalar(payload, "applied_coll_sampleRate"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(trial_from_coll.flattenCovMask(), dtype=float), _vector(payload, "applied_coll_flat_cov_mask"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(trial_from_coll.history.windowTimes, dtype=float), _vector(payload, "applied_coll_history_windowTimes"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(trial_from_coll.ensCovHist.windowTimes, dtype=float), _vector(payload, "applied_coll_ens_history_windowTimes"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(trial_from_coll.ensCovMask, dtype=float), np.asarray(payload["applied_coll_ens_mask"], dtype=float), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(trial_from_coll.covarColl.getCov(1).time, dtype=float), _vector(payload, "applied_coll_shifted_position_time"), rtol=1e-12, atol=1e-12)


def test_covcoll_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("covcoll_exactness.mat")
    time = np.array([0.0, 0.5, 1.0], dtype=float)
    position = Covariate(time, np.column_stack([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]), "Position", "time", "s", "", ["x", "y"])
    stimulus = Covariate(time, [5.0, 6.0, 7.0], "Stimulus", "time", "s", "a.u.", ["stim"])
    coll = CovColl([position, stimulus])
    coll.setMask([["Position", "x"], ["Stimulus"]])

    np.testing.assert_allclose(coll.getCov(1).time, _vector(payload, "masked_time"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        coll.dataToMatrix(),
        np.asarray(payload["masked_matrix"], dtype=float).reshape(coll.dataToMatrix().shape),
        rtol=1e-12,
        atol=1e-12,
    )
    assert coll.getCovLabelsFromMask() == _string_list(payload, "masked_labels")

    data_structure = coll.dataToStructure()
    np.testing.assert_allclose(np.asarray(data_structure["time"], dtype=float).reshape(-1), _vector(payload, "data_structure_time"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(data_structure["signals"]["values"], dtype=float),
        np.asarray(payload["data_structure_values"], dtype=float).reshape(np.asarray(data_structure["signals"]["values"], dtype=float).shape),
        rtol=1e-12,
        atol=1e-12,
    )

    structure = coll.toStructure()
    assert int(structure["numCov"]) == int(_scalar(payload, "structure_numCov"))
    np.testing.assert_allclose(float(structure["minTime"]), _scalar(payload, "structure_minTime"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(float(structure["maxTime"]), _scalar(payload, "structure_maxTime"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(coll.covMask[0], _vector(payload, "post_structure_mask_1"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(coll.covMask[1], _vector(payload, "post_structure_mask_2"), rtol=1e-12, atol=1e-12)

    roundtrip = CovColl.fromStructure(structure)
    np.testing.assert_allclose(float(roundtrip.minTime), _scalar(payload, "roundtrip_minTime"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(float(roundtrip.maxTime), _scalar(payload, "roundtrip_maxTime"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(float(roundtrip.sampleRate), _scalar(payload, "roundtrip_sampleRate"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(roundtrip.dataToMatrix(), np.asarray(payload["roundtrip_matrix"], dtype=float), rtol=1e-12, atol=1e-12)
    assert roundtrip.getCovLabelsFromMask() == _string_list(payload, "roundtrip_labels")

    shifted = CovColl([position, stimulus])
    shifted.setCovShift(0.25)
    shifted.restrictToTimeWindow(0.25, 1.25)
    np.testing.assert_allclose(float(shifted.minTime), _scalar(payload, "shifted_minTime"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(float(shifted.maxTime), _scalar(payload, "shifted_maxTime"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(shifted.getCov(2).time, _vector(payload, "shifted_stim_time"), rtol=1e-12, atol=1e-12)

    assert coll.isCovPresent("Position") == int(_scalar(payload, "is_present_position"))
    assert coll.isCovPresent(2) == int(_scalar(payload, "is_present_last_index"))
    assert coll.copy().numCov == int(_scalar(payload, "copy_numCov"))


def test_events_match_matlab_gold_fixture() -> None:
    payload = _load_fixture("events_exactness.mat")
    events = Events(_vector(payload, "eventTimes"), _string_list(payload, "eventLabels"), _string(payload, "eventColor"))
    rebuilt = Events.fromStructure(events.toStructure())
    assert rebuilt is not None
    np.testing.assert_allclose(np.asarray(rebuilt.eventTimes, dtype=float), _vector(payload, "eventTimes"), rtol=1e-12, atol=1e-12)
    assert rebuilt.eventLabels == _string_list(payload, "eventLabels")
    assert rebuilt.eventColor == _string(payload, "eventColor")

    fig, ax = plt.subplots()
    try:
        ax.axis(np.asarray(payload["axis_limits"], dtype=float))
        returned_ax = events.plot(handle=ax)
        assert returned_ax is ax
        assert len(ax.lines) == _vector(payload, "eventTimes").size
        first_line = ax.lines[0]
        np.testing.assert_allclose(np.asarray(first_line.get_xdata(), dtype=float), _vector(payload, "plot_line_xdata"), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.asarray(first_line.get_ydata(), dtype=float), _vector(payload, "plot_line_ydata"), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.asarray(mcolors.to_rgb(first_line.get_color()), dtype=float), np.asarray(payload["plot_line_color"], dtype=float), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(float(first_line.get_linewidth()), _scalar(payload, "plot_line_width"), rtol=1e-12, atol=1e-12)
        text_strings = [text.get_text() for text in ax.texts]
        assert text_strings == _string_list(payload, "plot_label_strings")
        label_positions = np.asarray([(*text.get_position(), 0.0) for text in ax.texts], dtype=float).reshape(-1)
        np.testing.assert_allclose(label_positions, _vector(payload, "plot_label_positions"), rtol=1e-12, atol=1e-12)
    finally:
        plt.close(fig)


def test_history_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("history_exactness.mat")
    history = History(_vector(payload, "windowTimes"), _scalar(payload, "minTime"), _scalar(payload, "maxTime"))
    rebuilt = History.fromStructure(history.toStructure())
    assert rebuilt is not None
    np.testing.assert_allclose(rebuilt.windowTimes, _vector(payload, "roundtrip_windowTimes"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(float(rebuilt.minTime), _scalar(payload, "roundtrip_minTime"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(float(rebuilt.maxTime), _scalar(payload, "roundtrip_maxTime"), rtol=1e-12, atol=1e-12)

    n1 = nspikeTrain([0.0, 0.5, 1.0], "n1", 2.0, 0.0, 1.0, "time", "s", "spikes", "spk", -1)
    n2 = nspikeTrain([0.25, 0.75], "n2", 2.0, 0.0, 1.0, "time", "s", "spikes", "spk", -1)
    coll = nstColl([n1, n2])

    single_cov = history.computeHistory(n1, 1)
    coll_cov = history.computeHistory(coll, 2)
    np.testing.assert_allclose(single_cov.dataToMatrix(), np.asarray(payload["single_history_matrix"], dtype=float), rtol=1e-12, atol=1e-12)
    assert single_cov.getAllCovLabels() == _string_list(payload, "single_history_labels")
    assert single_cov.getCov(1).name == _string(payload, "single_history_name")
    np.testing.assert_allclose(coll_cov.dataToMatrix(), np.asarray(payload["coll_history_matrix"], dtype=float), rtol=1e-12, atol=1e-12)
    assert coll_cov.getAllCovLabels() == _string_list(payload, "coll_history_labels")
    assert [coll_cov.getCov(index).name for index in range(1, coll_cov.numCov + 1)] == _string_list(payload, "coll_cov_names")

    filter_bank = history.toFilter(_scalar(payload, "filter_delta"))
    expected_num = _object_vectors(payload, "filter_num")
    expected_den = _object_vectors(payload, "filter_den")
    assert filter_bank.numFilters == len(expected_num)
    for idx, expected in enumerate(expected_num):
        np.testing.assert_allclose(filter_bank[idx].numerator.reshape(-1), expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(filter_bank[idx].denominator.reshape(-1), expected_den[idx], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(float(filter_bank[idx].delta), _scalar(payload, "filter_delta"), rtol=1e-12, atol=1e-12)

    fig, ax = plt.subplots()
    try:
        lines = history.plot(handle=ax)
        assert len(lines) == len(_object_vectors(payload, "plot_x"))
        for line, xdata, ydata in zip(lines, _object_vectors(payload, "plot_x"), _object_vectors(payload, "plot_y")):
            np.testing.assert_allclose(np.asarray(line.get_xdata(), dtype=float).reshape(-1), xdata, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(np.asarray(line.get_ydata(), dtype=float).reshape(-1), ydata, rtol=1e-12, atol=1e-12)
    finally:
        plt.close(fig)


def test_cif_eval_surface_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("cif_exactness.mat")
    cif = CIF(
        beta=_vector(payload, "beta"),
        Xnames=["stim1", "stim2"],
        stimNames=["stim1", "stim2"],
        fitType="binomial",
    )

    stim_val = _vector(payload, "stimVal")

    np.testing.assert_allclose(cif.evalLambdaDelta(stim_val), _scalar(payload, "lambda_delta"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(cif.evalGradient(stim_val).reshape(-1), _vector(payload, "gradient"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(cif.evalGradientLog(stim_val).reshape(-1), _vector(payload, "gradient_log"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(cif.evalJacobian(stim_val), np.asarray(payload["jacobian"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(cif.evalJacobianLog(stim_val), np.asarray(payload["jacobian_log"], dtype=float), rtol=1e-8, atol=1e-10)

    poly_cif = CIF(
        beta=_vector(payload, "poly_beta"),
        Xnames=["1", "x", "y", "x^2", "y^2", "x*y"],
        stimNames=["x", "y"],
        fitType="binomial",
    )
    poly_stim = _vector(payload, "poly_stimVal")
    np.testing.assert_allclose(poly_cif.evalLambdaDelta(poly_stim), _scalar(payload, "poly_lambda_delta"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(poly_cif.evalGradient(poly_stim).reshape(-1), _vector(payload, "poly_gradient"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(poly_cif.evalGradientLog(poly_stim).reshape(-1), _vector(payload, "poly_gradient_log"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(poly_cif.evalJacobian(poly_stim), np.asarray(payload["poly_jacobian"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(poly_cif.evalJacobianLog(poly_stim), np.asarray(payload["poly_jacobian_log"], dtype=float), rtol=1e-8, atol=1e-10)


def test_analysis_fit_surface_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("analysis_exactness.mat")
    time = _vector(payload, "time")
    stim_data = _vector(payload, "stim_data")
    spike_times = _vector(payload, "spike_times")
    sample_rate = _scalar(payload, "sample_rate")

    stim = Covariate(time, stim_data, "Stimulus", "time", "s", "", ["stim"])
    spike_train = nspikeTrain(spike_times, "1", 0.1, 0.0, 1.0, "time", "s", "", "", -1)
    trial = Trial(nstColl([spike_train]), CovColl([stim]))
    cfg = TrialConfig([["Stimulus", "stim"]], sample_rate, [], [], name="stim")
    fit = Analysis.RunAnalysisForNeuron(trial, 1, ConfigColl([cfg]))
    summary = FitResSummary([fit])

    np.testing.assert_allclose(fit.getCoeffs(1), _vector(payload, "coeffs"), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(fit.lambdaSignal.time, _vector(payload, "lambda_time"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fit.lambdaSignal.data[:, 0], _vector(payload, "lambda_data"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(fit.AIC[0]), _scalar(payload, "AIC"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(fit.BIC[0]), _scalar(payload, "BIC"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(fit.logLL[0]), _scalar(payload, "logLL"), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(float(summary.AIC[0, 0]), _scalar(payload, "summaryAIC"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(summary.BIC[0, 0]), _scalar(payload, "summaryBIC"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(summary.logLL[0, 0]), _scalar(payload, "summarylogLL"), rtol=1e-6, atol=1e-8)
    ks_stats = fit.computeKSStats(1)
    np.testing.assert_allclose(float(ks_stats["ks_stat"]), _scalar(payload, "ks_stat"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(ks_stats["ks_pvalue"]), _scalar(payload, "ks_pvalue"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(ks_stats["within_conf_int"]), _scalar(payload, "ks_within_conf_int"), rtol=1e-8, atol=1e-10)
    residual = fit.computeFitResidual(1)
    np.testing.assert_allclose(residual.time, _vector(payload, "residual_time"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(residual.data[:, 0], _vector(payload, "residual_data"), rtol=1e-6, atol=1e-8)
    assert fit.fitType[0] == _string(payload, "distribution")


def test_analysis_multineuron_surface_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("analysis_multineuron_exactness.mat")
    time = _vector(payload, "time")
    stim_data = _vector(payload, "stim_data")
    stim = Covariate(time, stim_data, "Stimulus", "time", "s", "", ["stim"])
    spike_train_1 = nspikeTrain(_vector(payload, "spike_times_1"), "1", 0.1, 0.0, 1.0, "time", "s", "", "", -1)
    spike_train_2 = nspikeTrain(_vector(payload, "spike_times_2"), "2", 0.1, 0.0, 1.0, "time", "s", "", "", -1)
    trial = Trial(nstColl([spike_train_1, spike_train_2]), CovColl([stim]))
    cfg = TrialConfig([["Stimulus", "stim"]], 10, [], [], name="stim")
    fits = Analysis.RunAnalysisForAllNeurons(trial, ConfigColl([cfg]), makePlot=0)
    assert isinstance(fits, list)
    assert len(fits) == int(_scalar(payload, "num_fits"))

    np.testing.assert_allclose(fits[0].getCoeffs(1), _vector(payload, "fit1_coeffs"), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(fits[1].getCoeffs(1), _vector(payload, "fit2_coeffs"), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(float(fits[0].AIC[0]), _scalar(payload, "fit1_AIC"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(fits[1].AIC[0]), _scalar(payload, "fit2_AIC"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(fits[0].BIC[0]), _scalar(payload, "fit1_BIC"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(fits[1].BIC[0]), _scalar(payload, "fit2_BIC"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(fits[0].logLL[0]), _scalar(payload, "fit1_logLL"), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(float(fits[1].logLL[0]), _scalar(payload, "fit2_logLL"), rtol=1e-6, atol=1e-8)

    summary = FitResSummary(fits)
    np.testing.assert_allclose(summary.AIC, np.asarray(payload["summary_AIC"], dtype=float).reshape(summary.AIC.shape), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(summary.BIC, np.asarray(payload["summary_BIC"], dtype=float).reshape(summary.BIC.shape), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(summary.logLL, np.asarray(payload["summary_logLL"], dtype=float).reshape(summary.logLL.shape), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(summary.KSStats, np.asarray(payload["summary_KSStats"], dtype=float).reshape(summary.KSStats.shape), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(summary.KSPvalues, np.asarray(payload["summary_KSPvalues"], dtype=float).reshape(summary.KSPvalues.shape), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(summary.withinConfInt, np.asarray(payload["summary_withinConfInt"], dtype=float).reshape(summary.withinConfInt.shape), rtol=1e-8, atol=1e-10)


def test_analysis_discrete_ks_arrays_match_matlab_gold_fixture() -> None:
    payload = _load_fixture("ksdiscrete_exactness.mat")

    spike_train = nspikeTrain(
        _vector(payload, "spike_times"),
        "1",
        0.1,
        0.0,
        1.0,
        "time",
        "s",
        "",
        "",
        -1,
    )
    lambda_signal = Covariate(
        _vector(payload, "lambda_time"),
        _vector(payload, "lambda_data"),
        "\\lambda(t)",
        "time",
        "s",
        "Hz",
        ["\\lambda_{1}"],
    )

    Z, U, xAxis, KSSorted, ks_stat = Analysis.computeKSStats(
        spike_train,
        lambda_signal,
        1,
        random_values=_vector(payload, "uniform_draws"),
    )

    np.testing.assert_allclose(np.asarray(Z, dtype=float).reshape(-1), _vector(payload, "Z"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(np.asarray(U, dtype=float).reshape(-1), _vector(payload, "U"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(np.asarray(xAxis, dtype=float).reshape(-1), _vector(payload, "xAxis"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(np.asarray(KSSorted, dtype=float).reshape(-1), _vector(payload, "KSSorted"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(ks_stat), _scalar(payload, "compute_ks_stat"), rtol=1e-8, atol=1e-10)

    stim = Covariate(_vector(payload, "time"), _vector(payload, "stim_data"), "Stimulus", "time", "s", "", ["stim"])
    trial = Trial(nstColl([spike_train]), CovColl([stim]))
    cfg = TrialConfig([["Stimulus", "stim"]], _scalar(payload, "sample_rate"), [], [], name="stim")
    fit = Analysis.RunAnalysisForNeuron(trial, 1, ConfigColl([cfg]), makePlot=0)
    fit.setKSStats(Z, U, xAxis, KSSorted, np.asarray([ks_stat], dtype=float))
    np.testing.assert_allclose(float(fit.KSStats[0, 0]), _scalar(payload, "ks_stat"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(fit.KSPvalues[0]), _scalar(payload, "ks_pvalue"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(float(fit.withinConfInt[0]), _scalar(payload, "ks_within_conf_int"), rtol=1e-8, atol=1e-10)


def test_fit_summary_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("fit_summary_exactness.mat")
    time = np.arange(0.0, 1.0 + 0.1, 0.1)
    lambda_signal = Covariate(
        time,
        np.column_stack(
            [
                np.linspace(2.0, 7.0, time.size, dtype=float),
                np.linspace(3.0, 8.0, time.size, dtype=float),
            ]
        ),
        "\\lambda(t)",
        "time",
        "s",
        "Hz",
        ["stim", "stim_hist"],
    )
    st1 = nspikeTrain([0.1, 0.4, 0.7], "1", 0.1, 0.0, 1.0, "time", "s", "", "", -1)
    st2 = nspikeTrain([0.2, 0.5, 0.8], "2", 0.1, 0.0, 1.0, "time", "s", "", "", -1)
    stim_cfg = TrialConfig([["Stimulus", "stim"]], 10.0, [], [], name="stim")
    stim_hist_cfg = TrialConfig([["Stimulus", "stim"]], 10.0, [0.0, 0.1, 0.2], [], name="stim_hist")
    config_coll = ConfigColl([stim_cfg, stim_hist_cfg])
    fit1 = FitResult(
        st1,
        [["stim"], ["stim", "hist1", "hist2"]],
        [0, 2],
        [None, None],
        [None, None],
        lambda_signal,
        [np.array([0.5]), np.array([0.3, -0.1, -0.05])],
        np.array([1.0, 2.0]),
        [
            {"se": np.array([0.05]), "p": np.array([0.01])},
            {"se": np.array([0.04, 0.03, 0.02]), "p": np.array([0.02, 0.04, 0.06])},
        ],
        np.array([11.0, 7.0]),
        np.array([12.0, 8.0]),
        np.array([3.0, 5.0]),
        config_coll,
        [],
        [],
        "poisson",
    )
    fit2 = FitResult(
        st2,
        [["stim"], ["stim", "hist1", "hist2"]],
        [0, 2],
        [None, None],
        [None, None],
        lambda_signal,
        [np.array([0.4]), np.array([0.25, -0.08, -0.02])],
        np.array([1.5, 2.5]),
        [
            {"se": np.array([0.06]), "p": np.array([0.03])},
            {"se": np.array([0.05, 0.04, 0.03]), "p": np.array([0.01, 0.03, 0.07])},
        ],
        np.array([13.0, 9.0]),
        np.array([14.0, 10.0]),
        np.array([2.0, 4.0]),
        config_coll,
        [],
        [],
        "poisson",
    )
    fit1.KSStats[:, 0] = np.array([0.25, 0.50], dtype=float)
    fit1.KSPvalues[:] = np.array([0.90, 0.40], dtype=float)
    fit1.withinConfInt[:] = np.array([1.0, 1.0], dtype=float)
    fit2.KSStats[:, 0] = np.array([0.35, 0.55], dtype=float)
    fit2.KSPvalues[:] = np.array([0.80, 0.30], dtype=float)
    fit2.withinConfInt[:] = np.array([1.0, 0.0], dtype=float)
    summary = FitResSummary([fit1, fit2])

    assert summary.fitNames == _string_list(payload, "fitNames")
    np.testing.assert_allclose(np.asarray(summary.neuronNumbers, dtype=float), _vector(payload, "neuronNumbers"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(summary.AIC, np.asarray(payload["AIC"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(summary.BIC, np.asarray(payload["BIC"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(summary.logLL, np.asarray(payload["logLL"], dtype=float), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(summary.KSStats, np.asarray(payload["KSStats"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(summary.KSPvalues, np.asarray(payload["KSPvalues"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(summary.withinConfInt, np.asarray(payload["withinConfInt"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(summary.getDiffAIC(1), np.asarray(payload["diffAIC"], dtype=float).reshape(summary.getDiffAIC(1).shape), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(summary.getDiffBIC(1), np.asarray(payload["diffBIC"], dtype=float).reshape(summary.getDiffBIC(1).shape), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(summary.getDifflogLL(1), np.asarray(payload["difflogLL"], dtype=float).reshape(summary.getDifflogLL(1).shape), rtol=1e-6, atol=1e-8)


def test_point_process_lambda_trace_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("point_process_exactness.mat")

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
        seed=int(_scalar(payload, "seed")),
        return_lambda=True,
    )

    np.testing.assert_allclose(lambda_cov.data[: _vector(payload, 'lambda_head').shape[0], 0], _vector(payload, "lambda_head"), rtol=1e-8, atol=1e-10)


def test_point_process_deterministic_trace_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("point_process_exactness.mat")
    time = _vector(payload, "det_time")
    stim_values = _vector(payload, "det_stimulus")
    uniforms = _vector(payload, "det_uniforms").reshape(-1, 1)
    stim = Covariate(time, stim_values, "Stimulus", "time", "s", "Voltage", ["sin"])
    ens = Covariate(time, np.zeros_like(time), "Ensemble", "time", "s", "Spikes", ["n1"])
    _, lambda_cov, details = CIF.simulateCIF(
        -3.0,
        np.array([-1.0, -2.0, -4.0], dtype=float),
        np.array([1.0], dtype=float),
        np.array([0.0], dtype=float),
        stim,
        ens,
        numRealizations=1,
        simType="binomial",
        random_values=uniforms,
        return_lambda=True,
        return_details=True,
    )

    np.testing.assert_allclose(lambda_cov.data[:, 0], _vector(payload, "det_rate_hz"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(details["lambda_delta"][:, 0], _vector(payload, "det_lambda_delta"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(details["history_effect"][:, 0], _vector(payload, "det_history_effect"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(details["eta"][:, 0], _vector(payload, "det_eta"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(details["spike_indicator"][:, 0], _vector(payload, "det_spike_indicator"), rtol=1e-8, atol=1e-10)


def test_cif_thinning_from_lambda_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("thinning_exactness.mat")
    lambda_cov = Covariate(
        _vector(payload, "time"),
        _vector(payload, "lambda_data"),
        "\\lambda(t)",
        "time",
        "s",
        "Hz",
        ["\\lambda"],
    )
    spike_coll, details = CIF.simulateCIFByThinningFromLambda(
        lambda_cov,
        numRealizations=1,
        maxTimeRes=_scalar(payload, "maxTimeRes"),
        random_values=_vector(payload, "arrival_uniforms"),
        thinning_values=_vector(payload, "thinning_uniforms"),
        return_details=True,
    )

    np.testing.assert_allclose(float(details["lambda_bound"]), _scalar(payload, "lambda_bound"), rtol=1e-12, atol=1e-12)
    assert int(details["proposal_count"]) == int(_scalar(payload, "proposal_count"))
    np.testing.assert_allclose(details["arrival_uniforms"], _vector(payload, "arrival_uniforms"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(details["interarrival_times"], _vector(payload, "interarrival_times"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(details["candidate_spike_times"], _vector(payload, "candidate_spike_times"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(details["lambda_ratio"], _vector(payload, "lambda_ratio"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(details["thinning_uniforms"], _vector(payload, "thinning_uniforms"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(details["accepted_spike_times"], _vector(payload, "rounded_spike_times"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(spike_coll.getNST(1).spikeTimes, _vector(payload, "rounded_spike_times"), rtol=1e-8, atol=1e-10)


def test_decoding_predict_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("decoding_predict_exactness.mat")
    x_p, W_p = DecodingAlgorithms.PPDecode_predict(
        _vector(payload, "x_u"),
        np.asarray(payload["W_u"], dtype=float),
        np.asarray(payload["A"], dtype=float),
        np.asarray(payload["Q"], dtype=float),
    )

    np.testing.assert_allclose(x_p, _vector(payload, "x_p"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(W_p, np.asarray(payload["W_p"], dtype=float), rtol=1e-8, atol=1e-10)


def test_pp_fixed_interval_smoother_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("decoding_smoother_exactness.mat")
    x_pLag, W_pLag, x_uLag, W_uLag = DecodingAlgorithms.PP_fixedIntervalSmoother(
        _scalar(payload, "A"),
        _scalar(payload, "Q"),
        np.asarray(payload["dN"], dtype=float),
        int(_scalar(payload, "lags")),
        _scalar(payload, "mu"),
        _scalar(payload, "beta"),
        _string(payload, "fitType"),
        _scalar(payload, "delta"),
    )

    np.testing.assert_allclose(x_pLag, np.asarray(payload["x_pLag"], dtype=float).reshape(x_pLag.shape), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(W_pLag, np.asarray(payload["W_pLag"], dtype=float).reshape(W_pLag.shape), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(x_uLag, np.asarray(payload["x_uLag"], dtype=float).reshape(x_uLag.shape), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(W_uLag, np.asarray(payload["W_uLag"], dtype=float).reshape(W_uLag.shape), rtol=1e-8, atol=1e-10)


def test_pp_hybrid_filter_linear_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("hybrid_filter_exactness.mat")
    S_est, X, W, MU_u, X_s, W_s, pNGivenS = DecodingAlgorithms.PPHybridFilterLinear(
        [np.array([[float(_scalar(payload, "A1"))]], dtype=float), np.array([[float(_scalar(payload, "A2"))]], dtype=float)],
        [np.array([[float(_scalar(payload, "Q1"))]], dtype=float), np.array([[float(_scalar(payload, "Q2"))]], dtype=float)],
        np.asarray(payload["p_ij"], dtype=float),
        _vector(payload, "Mu0"),
        np.asarray(payload["dN"], dtype=float),
        float(_scalar(payload, "mu")),
        float(_scalar(payload, "beta")),
        _string(payload, "fitType"),
        _scalar(payload, "binwidth"),
    )

    np.testing.assert_allclose(S_est, _vector(payload, "S_est"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(X, np.asarray(payload["X"], dtype=float).reshape(X.shape), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(W, np.asarray(payload["W"], dtype=float).reshape(W.shape), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(MU_u, np.asarray(payload["MU_u"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(X_s[0], np.asarray(payload["X_s_1"], dtype=float).reshape(X_s[0].shape), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(X_s[1], np.asarray(payload["X_s_2"], dtype=float).reshape(X_s[1].shape), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(W_s[0], np.asarray(payload["W_s_1"], dtype=float).reshape(W_s[0].shape), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(W_s[1], np.asarray(payload["W_s_2"], dtype=float).reshape(W_s[1].shape), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(pNGivenS, np.asarray(payload["pNGivenS"], dtype=float), rtol=1e-8, atol=1e-10)


def test_nonlinear_ppdecodefilter_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("nonlinear_decode_exactness.mat")
    lambda_cifs = [
        CIF(_vector(payload, "beta1"), ["1", "x", "y", "x^2", "y^2", "x*y"], ["x", "y"], fitType="binomial"),
        CIF(_vector(payload, "beta2"), ["1", "x", "y", "x^2", "y^2", "x*y"], ["x", "y"], fitType="binomial"),
    ]

    x_p, W_p, x_u, W_u, *_ = DecodingAlgorithms.PPDecodeFilter(
        np.asarray(payload["A"], dtype=float),
        np.asarray(payload["Q"], dtype=float),
        np.asarray(payload["Px0"], dtype=float),
        np.asarray(payload["dN"], dtype=float),
        lambda_cifs,
        _scalar(payload, "delta"),
    )

    np.testing.assert_allclose(x_p, np.asarray(payload["x_p"], dtype=float), rtol=1e-3, atol=5e-4)
    np.testing.assert_allclose(W_p, np.asarray(payload["W_p"], dtype=float), rtol=1e-3, atol=5e-4)
    np.testing.assert_allclose(x_u, np.asarray(payload["x_u"], dtype=float), rtol=1e-3, atol=5e-4)
    np.testing.assert_allclose(W_u, np.asarray(payload["W_u"], dtype=float), rtol=1e-3, atol=5e-4)


def test_simulated_network_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("simulated_network_exactness.mat")
    native = simulate_two_neuron_network(seed=4)

    np.testing.assert_allclose(native.actual_network, np.asarray(payload["actual_network"], dtype=float), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(native.lambda_delta[:5], np.asarray(payload["prob_head"], dtype=float), rtol=1e-8, atol=1e-10)
    dt = float(native.time[1] - native.time[0])
    native_state_head = np.column_stack([
        native.spikes.getNST(1).getSigRep(dt, float(native.time[0]), float(native.time[-1])).data[:5, 0],
        native.spikes.getNST(2).getSigRep(dt, float(native.time[0]), float(native.time[-1])).data[:5, 0],
    ])
    np.testing.assert_allclose(native_state_head, np.asarray(payload["state_head"], dtype=float), rtol=1e-8, atol=1e-10)
    native_counts = np.array([
        len(native.spikes.getNST(1).spikeTimes),
        len(native.spikes.getNST(2).spikeTimes),
    ], dtype=float)
    matlab_counts = _vector(payload, "spike_counts")
    assert native_counts.shape == matlab_counts.shape
    assert np.all(np.abs(native_counts - matlab_counts) <= 64.0)


def test_simulated_network_deterministic_trace_matches_matlab_gold_fixture() -> None:
    payload = _load_fixture("simulated_network_exactness.mat")
    sim = simulate_two_neuron_network(
        duration_s=float(_vector(payload, "det_time")[-1]),
        dt=float(_vector(payload, "det_time")[1] - _vector(payload, "det_time")[0]),
        seed=None,
        uniform_values=np.asarray(payload["det_uniforms"], dtype=float),
    )

    np.testing.assert_allclose(sim.time, _vector(payload, "det_time"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sim.latent_drive, _vector(payload, "det_stimulus"), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(sim.lambda_delta, np.asarray(payload["det_probability"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(sim.spike_indicator, np.asarray(payload["det_state"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(sim.eta, np.asarray(payload["det_eta"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(sim.history_effect, np.asarray(payload["det_history_effect"], dtype=float), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(sim.ensemble_effect, np.asarray(payload["det_ensemble_effect"], dtype=float), rtol=1e-8, atol=1e-10)
