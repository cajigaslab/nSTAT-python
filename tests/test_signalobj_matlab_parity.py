from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from nstat.compat.matlab import SignalObj


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "SignalObj" / "basic.mat"


def _mat() -> dict[str, object]:
    return loadmat(str(FIXTURE), squeeze_me=False, struct_as_record=False)


def _arr(m: dict[str, object], key: str) -> np.ndarray:
    return np.asarray(m[key], dtype=float)


def _vec(m: dict[str, object], key: str) -> np.ndarray:
    return np.asarray(m[key], dtype=float).reshape(-1)


def _scalar(m: dict[str, object], key: str) -> float:
    return float(np.asarray(m[key], dtype=float).reshape(-1)[0])


def test_signalobj_compat_core_matches_matlab_fixture() -> None:
    m = _mat()
    sig = SignalObj(
        time=_vec(m, "time"),
        data=_arr(m, "data"),
        name="sig",
        x_label="time",
        x_units="s",
        y_units="unit",
    )

    np.testing.assert_allclose(sig.dataToMatrix(), _arr(m, "base_data"), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(sig.getTime(), _vec(m, "base_time"), rtol=0.0, atol=1e-12)
    assert np.isclose(sig.getSampleRate(), _scalar(m, "base_sample_rate"), atol=1e-12)

    deriv = sig.derivative()
    np.testing.assert_allclose(deriv.dataToMatrix(), _arr(m, "deriv_data"), rtol=0.0, atol=1e-10)

    sub = sig.getSubSignal([1])
    np.testing.assert_allclose(sub.dataToMatrix(), _arr(m, "sub_data"), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(sub.getTime(), _vec(m, "sub_time"), rtol=0.0, atol=1e-12)

    other = SignalObj(
        time=_vec(m, "time"),
        data=np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        name="sig2",
        x_label="time",
        x_units="s",
        y_units="unit",
    )
    merged = sig.merge(other)
    np.testing.assert_allclose(merged.dataToMatrix(), _arr(m, "merged_data"), rtol=0.0, atol=1e-12)



def test_signalobj_compat_resample_shift_align_and_roundtrip() -> None:
    m = _mat()
    sig = SignalObj(
        time=_vec(m, "time"),
        data=_arr(m, "data"),
        name="sig",
        x_label="time",
        x_units="s",
        y_units="unit",
    )

    resampled = sig.resample(_scalar(m, "resampled_sample_rate"))
    np.testing.assert_allclose(resampled.getTime(), _vec(m, "resampled_time"), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resampled.dataToMatrix(), _arr(m, "resampled_data"), rtol=0.0, atol=1e-8)

    shifted = sig.shift(0.1)
    np.testing.assert_allclose(shifted.getTime(), _vec(m, "shifted_time"), rtol=0.0, atol=1e-12)

    aligned = sig.copySignal()
    aligned.alignTime(0.5, 0.0)
    np.testing.assert_allclose(aligned.getTime(), _vec(m, "aligned_time"), rtol=0.0, atol=1e-12)

    # MATLAB returns 1-based indices; Python compat uses 0-based.
    nearest_idx_py = sig.findNearestTimeIndex(0.63)
    nearest_idx_mat = int(_scalar(m, "nearest_idx"))
    assert nearest_idx_py + 1 == nearest_idx_mat

    nearest_indices_py = np.asarray(sig.findNearestTimeIndices(np.array([0.0, 0.38, 0.99])), dtype=int)
    nearest_indices_mat = _vec(m, "nearest_indices").astype(int)
    assert np.array_equal(nearest_indices_py + 1, nearest_indices_mat)

    np.testing.assert_allclose(sig.getValueAt(0.5), _vec(m, "value_at_05"), rtol=0.0, atol=1e-12)

    mat_struct = np.asarray(m["sig_struct"], dtype=object).reshape(-1)[0]
    roundtrip = SignalObj.signalFromStruct(mat_struct)
    np.testing.assert_allclose(roundtrip.dataToMatrix(), _arr(m, "roundtrip_data"), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(roundtrip.getTime(), _vec(m, "roundtrip_time"), rtol=0.0, atol=1e-12)
