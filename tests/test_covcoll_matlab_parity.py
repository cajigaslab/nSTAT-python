from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from nstat.compat.matlab import CovColl as MatlabCovColl
from nstat.compat.matlab import Covariate as MatlabCovariate


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "CovColl" / "basic.mat"


def _mat() -> dict[str, object]:
    return loadmat(str(FIXTURE), squeeze_me=False, struct_as_record=False)


def _to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [_to_python(v) for v in value.reshape(-1)]
        if value.ndim == 0:
            return value.item()
        if value.size == 1:
            scalar = value.reshape(-1)[0]
            return scalar.item() if hasattr(scalar, "item") else scalar
        return value.tolist()
    if hasattr(value, "_fieldnames"):
        return {name: _to_python(getattr(value, name)) for name in value._fieldnames}
    return value


def _scalar(m: dict[str, object], key: str) -> float:
    return float(np.asarray(m[key], dtype=float).reshape(-1)[0])


def _int(m: dict[str, object], key: str) -> int:
    return int(np.asarray(m[key], dtype=float).reshape(-1)[0])


def _cellstr(values: Any) -> list[str]:
    arr = np.asarray(values, dtype=object).reshape(-1)
    out: list[str] = []
    for value in arr:
        parsed = _to_python(value)
        if isinstance(parsed, list):
            out.append("" if not parsed else str(parsed[0]))
        else:
            out.append(str(parsed))
    return out


def _build_covcoll() -> MatlabCovColl:
    time = np.arange(0.0, 1.0 + 1e-12, 0.1)
    cov1 = MatlabCovariate(time=time, data=np.sin(2.0 * np.pi * time), name="sine", labels=["sine"])
    cov2 = MatlabCovariate(time=time, data=np.column_stack([time, time**2]), name="poly", labels=["t", "t2"])
    return MatlabCovColl([cov1, cov2])


def test_covcoll_core_matches_matlab_fixture() -> None:
    m = _mat()
    coll = _build_covcoll()

    assert int(coll.numCov) == _int(m, "initial_numCov")
    assert [int(v) for v in coll.covDimensions] == np.asarray(m["initial_covDimensions"], dtype=int).reshape(-1).tolist()
    assert np.isclose(float(coll.sampleRate), _scalar(m, "initial_sampleRate"), atol=1e-12)
    assert np.isclose(float(coll.minTime), _scalar(m, "initial_minTime"), atol=1e-12)
    assert np.isclose(float(coll.maxTime), _scalar(m, "initial_maxTime"), atol=1e-12)
    assert coll.getAllCovLabels() == _cellstr(m["initial_labels"])

    np.testing.assert_array_equal(np.asarray(coll.covMask[0], dtype=int).reshape(-1), np.asarray([[1]], dtype=int).reshape(-1))
    np.testing.assert_array_equal(np.asarray(coll.covMask[1], dtype=int).reshape(-1), np.asarray([[1, 1]], dtype=int).reshape(-1))

    X, _labels = coll.dataToMatrix()
    np.testing.assert_allclose(X, np.asarray(m["initial_data_matrix"], dtype=float), rtol=0.0, atol=1e-12)

    # MATLAB indices are 1-based.
    assert [idx + 1 for idx in coll.getCovIndicesFromNames(["sine", "poly"])] == np.asarray(m["initial_cov_inds"], dtype=int).reshape(-1).tolist()
    assert coll.isCovPresent("sine") == bool(_int(m, "initial_is_cov_present"))

    shifted = _build_covcoll().copy()
    shifted.setCovShift(0.2)
    assert np.isclose(float(shifted.covShift), _scalar(m, "shift_covShift"), atol=1e-12)
    assert np.isclose(float(shifted.minTime), _scalar(m, "shift_minTime"), atol=1e-12)
    assert np.isclose(float(shifted.maxTime), _scalar(m, "shift_maxTime"), atol=1e-12)
    shifted.resetCovShift()
    assert np.isclose(float(shifted.covShift), _scalar(m, "reset_covShift"), atol=1e-12)
    assert np.isclose(float(shifted.minTime), _scalar(m, "reset_minTime"), atol=1e-12)
    assert np.isclose(float(shifted.maxTime), _scalar(m, "reset_maxTime"), atol=1e-12)

    sr = _build_covcoll().copy()
    sr.setSampleRate(5.0)
    assert np.isclose(float(sr.sampleRate), _scalar(m, "sr_sampleRate"), atol=1e-12)
    X_sr, _ = sr.dataToMatrix()
    np.testing.assert_allclose(X_sr, np.asarray(m["sr_data_matrix"], dtype=float), rtol=0.0, atol=1e-12)

    win = _build_covcoll().copy()
    win.restrictToTimeWindow(0.2, 0.8)
    assert np.isclose(float(win.minTime), _scalar(m, "win_minTime"), atol=1e-12)
    assert np.isclose(float(win.maxTime), _scalar(m, "win_maxTime"), atol=1e-12)
    X_win, _ = win.dataToMatrix()
    np.testing.assert_allclose(X_win, np.asarray(m["win_data_matrix"], dtype=float), rtol=0.0, atol=1e-12)


def test_covcoll_structure_and_removal_match_matlab_fixture() -> None:
    m = _mat()
    coll = _build_covcoll()
    payload = coll.toStructure()

    assert int(payload["numCov"]) == _int(m, "initial_numCov")
    assert np.asarray(payload["covDimensions"], dtype=int).reshape(-1).tolist() == np.asarray(m["initial_covDimensions"], dtype=int).reshape(-1).tolist()
    assert np.isclose(float(payload["sampleRate"]), _scalar(m, "initial_sampleRate"), atol=1e-12)
    assert np.isclose(float(payload["minTime"]), _scalar(m, "initial_minTime"), atol=1e-12)
    assert np.isclose(float(payload["maxTime"]), _scalar(m, "initial_maxTime"), atol=1e-12)
    assert "covArray" in payload

    mat_payload = _to_python(np.asarray(m["struct_payload"], dtype=object).reshape(-1)[0])
    restored = MatlabCovColl.fromStructure(mat_payload)
    assert int(restored.numCov) == _int(m, "roundtrip_numCov")
    assert [int(v) for v in restored.covDimensions] == np.asarray(m["roundtrip_covDimensions"], dtype=int).reshape(-1).tolist()
    assert np.isclose(float(restored.sampleRate), _scalar(m, "roundtrip_sampleRate"), atol=1e-12)
    assert np.isclose(float(restored.minTime), _scalar(m, "roundtrip_minTime"), atol=1e-12)
    assert np.isclose(float(restored.maxTime), _scalar(m, "roundtrip_maxTime"), atol=1e-12)
    assert restored.getAllCovLabels() == _cellstr(m["roundtrip_labels"])
    X_rt, _ = restored.dataToMatrix()
    np.testing.assert_allclose(X_rt, np.asarray(m["roundtrip_data_matrix"], dtype=float), rtol=0.0, atol=1e-12)

    removed = _build_covcoll().copy()
    removed.removeCovariate(1)  # MATLAB removed index 2 (1-based)
    assert int(removed.numCov) == _int(m, "removed_numCov")
    assert removed.getAllCovLabels() == _cellstr(m["removed_labels"])
    X_removed, _ = removed.dataToMatrix()
    np.testing.assert_allclose(X_removed, np.asarray(m["removed_data_matrix"], dtype=float), rtol=0.0, atol=1e-12)
