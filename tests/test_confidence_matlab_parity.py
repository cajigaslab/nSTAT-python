from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from scipy.io import loadmat

from nstat.confidence import ConfidenceInterval
from nstat.compat.matlab import ConfidenceInterval as MatlabConfidenceInterval

matplotlib.use("Agg")


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ConfidenceInterval" / "basic.mat"


def _mat() -> dict[str, object]:
    return loadmat(str(FIXTURE), squeeze_me=False, struct_as_record=False)


def _vec(m: dict[str, object], key: str) -> np.ndarray:
    return np.asarray(m[key], dtype=float).reshape(-1)


def _scalar(m: dict[str, object], key: str) -> float:
    return float(np.asarray(m[key], dtype=float).reshape(-1)[0])


def _str(m: dict[str, object], key: str) -> str:
    return str(np.asarray(m[key], dtype=object).reshape(-1)[0])


def _cellvec(values: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(v, dtype=float).reshape(-1) for v in np.asarray(values, dtype=object).reshape(-1)]


def test_confidence_native_behavior_matches_matlab_fixture() -> None:
    m = _mat()
    time = _vec(m, "time")
    lower = _vec(m, "lower")
    upper = _vec(m, "upper")

    ci = ConfidenceInterval(time=time, lower=lower, upper=upper)
    assert ci.color == _str(m, "default_color")
    assert np.isclose(float(ci.value), _scalar(m, "default_value"), atol=1e-12)

    ci.set_color(_str(m, "set_color")).set_value(_scalar(m, "set_value"))
    assert ci.color == _str(m, "set_color")
    assert np.isclose(float(ci.value), _scalar(m, "set_value"), atol=1e-12)

    np.testing.assert_allclose(ci.width(), _vec(m, "width"), rtol=0.0, atol=1e-12)
    contains = ci.contains(_vec(m, "probe_values"))
    assert np.array_equal(contains, np.asarray(m["contains_probe"], dtype=bool).reshape(-1))

    payload = ci.to_structure()
    restored = ConfidenceInterval.from_structure(payload)
    np.testing.assert_allclose(restored.lower, lower, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(restored.upper, upper, rtol=0.0, atol=1e-12)



def test_confidence_compat_structure_roundtrip_and_plot_match_matlab_fixture() -> None:
    m = _mat()
    time = _vec(m, "time")
    lower = _vec(m, "lower")
    upper = _vec(m, "upper")

    ci = MatlabConfidenceInterval(time=time, lower=lower, upper=upper)
    assert ci.color == _str(m, "default_color")
    assert np.isclose(float(ci.value), _scalar(m, "default_value"), atol=1e-12)

    ci.setColor(_str(m, "set_color")).setValue(_scalar(m, "set_value"))
    assert ci.color == _str(m, "set_color")
    assert np.isclose(float(ci.value), _scalar(m, "set_value"), atol=1e-12)
    np.testing.assert_allclose(ci.lower, lower, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(ci.upper, upper, rtol=0.0, atol=1e-12)

    payload = ci.toStructure()
    assert "signals" in payload
    restored = MatlabConfidenceInterval.fromStructure(payload)
    np.testing.assert_allclose(restored.lower, np.asarray(m["roundtrip_data"], dtype=float)[:, 0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(restored.upper, np.asarray(m["roundtrip_data"], dtype=float)[:, 1], rtol=0.0, atol=1e-12)
    assert restored.color == _str(m, "roundtrip_color")
    assert np.isclose(float(restored.value), _scalar(m, "roundtrip_value"), atol=1e-12)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3), dpi=120)
    plt.sca(ax)
    lines = ci.plot(_str(m, "set_color"), 0.3, 0)
    expected_lines = int(_scalar(m, "line_count"))
    assert len(lines) == expected_lines
    expected_x = _cellvec(np.asarray(m["line_x_data"], dtype=object))
    expected_y = _cellvec(np.asarray(m["line_y_data"], dtype=object))
    for idx, line in enumerate(lines):
        np.testing.assert_allclose(np.asarray(line.get_xdata(), dtype=float).reshape(-1), expected_x[idx], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(line.get_ydata(), dtype=float).reshape(-1), expected_y[idx], rtol=0.0, atol=1e-12)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(4, 3), dpi=120)
    plt.sca(ax2)
    patches = ci.plot(_str(m, "set_color"), 0.2, 1)
    expected_patch_count = int(_scalar(m, "patch_count"))
    assert len(patches) == expected_patch_count
    expected_px = _cellvec(np.asarray(m["patch_x_data"], dtype=object))
    expected_py = _cellvec(np.asarray(m["patch_y_data"], dtype=object))
    for idx, patch in enumerate(patches):
        xy = np.asarray(patch.get_xy(), dtype=float)
        if xy.shape[0] == expected_px[idx].size + 1 and np.allclose(xy[0], xy[-1], atol=1e-12):
            xy = xy[:-1]
        np.testing.assert_allclose(xy[:, 0], expected_px[idx], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(xy[:, 1], expected_py[idx], rtol=0.0, atol=1e-12)
    plt.close(fig2)
