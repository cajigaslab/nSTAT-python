from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from nstat.history import HistoryBasis
from nstat.compat.matlab import History as MatlabHistory


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "History" / "basic.mat"


def _mat() -> dict[str, object]:
    return loadmat(str(FIXTURE), squeeze_me=False, struct_as_record=False)


def _vec(m: dict[str, object], key: str) -> np.ndarray:
    return np.asarray(m[key], dtype=float).reshape(-1)


def _scalar(m: dict[str, object], key: str) -> float:
    return float(np.asarray(m[key], dtype=float).reshape(-1)[0])


def test_history_native_design_and_structure_match_matlab_fixture() -> None:
    m = _mat()
    window_times = _vec(m, "window_times")
    spike_times = _vec(m, "spike_times")
    time_grid = _vec(m, "time_grid")

    hist = HistoryBasis(
        bin_edges_s=window_times,
        min_time_s=_scalar(m, "min_time"),
        max_time_s=_scalar(m, "max_time"),
    )

    design = hist.design_matrix(spike_times_s=spike_times, time_grid_s=time_grid)
    np.testing.assert_allclose(design, np.asarray(m["expected_design"], dtype=float), rtol=0.0, atol=1e-12)

    payload = hist.to_structure()
    np.testing.assert_allclose(np.asarray(payload["windowTimes"], dtype=float).reshape(-1), window_times, rtol=0.0, atol=1e-12)
    assert float(payload["minTime"]) == _scalar(m, "min_time")
    assert float(payload["maxTime"]) == _scalar(m, "max_time")

    restored = HistoryBasis.from_structure(payload)
    np.testing.assert_allclose(restored.windowTimes, window_times, rtol=0.0, atol=1e-12)
    assert restored.minTime == _scalar(m, "min_time")
    assert restored.maxTime == _scalar(m, "max_time")



def test_history_compat_roundtrip_setwindow_and_filter_match_matlab_fixture() -> None:
    m = _mat()
    window_times = _vec(m, "window_times")

    hist = MatlabHistory(
        bin_edges_s=window_times,
        min_time_s=_scalar(m, "min_time"),
        max_time_s=_scalar(m, "max_time"),
    )

    payload = hist.toStructure()
    assert {"windowTimes", "minTime", "maxTime"}.issubset(set(payload.keys()))
    np.testing.assert_allclose(np.asarray(payload["windowTimes"], dtype=float).reshape(-1), window_times, rtol=0.0, atol=1e-12)

    restored = MatlabHistory.fromStructure(payload)
    np.testing.assert_allclose(restored.windowTimes, window_times, rtol=0.0, atol=1e-12)
    assert restored.minTime == _scalar(m, "min_time")
    assert restored.maxTime == _scalar(m, "max_time")

    restored.setWindow(np.asarray(m["set_window_times"], dtype=float).reshape(-1))
    np.testing.assert_allclose(
        restored.windowTimes,
        _vec(m, "set_window_times"),
        rtol=0.0,
        atol=1e-12,
    )

    filt = hist.toFilter()
    np.testing.assert_allclose(filt.reshape(-1), _vec(m, "expected_filter"), rtol=0.0, atol=1e-12)

    filt_delta = hist.toFilter(delta=_scalar(m, "delta"))
    np.testing.assert_allclose(
        np.asarray(filt_delta, dtype=float),
        np.asarray(m["expected_filter_delta"], dtype=float),
        rtol=0.0,
        atol=1e-12,
    )
