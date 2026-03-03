from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from nstat.compat.matlab import ConfidenceInterval as MatlabConfidenceInterval
from nstat.compat.matlab import Covariate as MatlabCovariate


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "Covariate" / "basic.mat"


def _mat() -> dict[str, object]:
    return loadmat(str(FIXTURE), squeeze_me=False, struct_as_record=False)


def _arr(m: dict[str, object], key: str) -> np.ndarray:
    return np.asarray(m[key], dtype=float)


def _vec(m: dict[str, object], key: str) -> np.ndarray:
    return np.asarray(m[key], dtype=float).reshape(-1)


def _scalar(m: dict[str, object], key: str) -> float:
    return float(np.asarray(m[key], dtype=float).reshape(-1)[0])


def _ci_matrix(cov: MatlabCovariate) -> np.ndarray:
    raw = cov.conf_interval
    if isinstance(raw, list):
        ci = raw[0]
    else:
        ci = raw
    assert ci is not None
    return np.column_stack([np.asarray(ci.lower, dtype=float), np.asarray(ci.upper, dtype=float)])


def test_covariate_compat_core_matches_matlab_fixture() -> None:
    m = _mat()
    time = _vec(m, "time")
    data = _arr(m, "data")

    cov = MatlabCovariate(
        time=time,
        data=data,
        name="stim",
        units="u",
        labels=["c1", "c2", "c3"],
        x_label="time",
        x_units="s",
        y_units="u",
    )

    np.testing.assert_allclose(cov.dataToMatrix(), _arr(m, "base_data"), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(cov.getTime(), _vec(m, "base_time"), rtol=0.0, atol=1e-12)

    std_rep = cov.getSigRep("standard")
    zm_rep = cov.getSigRep("zero-mean")
    np.testing.assert_allclose(std_rep.dataToMatrix(), _arr(m, "sigrep_standard"), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(zm_rep.dataToMatrix(), _arr(m, "sigrep_zero_mean"), rtol=0.0, atol=1e-12)

    sub_ind = cov.getSubSignal(2)
    sub_name = cov.getSubSignal("c3")
    np.testing.assert_allclose(sub_ind.dataToMatrix(), _arr(m, "sub_ind_data"), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(sub_name.dataToMatrix(), _arr(m, "sub_name_data"), rtol=0.0, atol=1e-12)

    mean_ci = cov.computeMeanPlusCI(0.10)
    np.testing.assert_allclose(mean_ci.dataToMatrix(), _arr(m, "mean_ci_data"), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(_ci_matrix(mean_ci), _arr(m, "mean_ci_interval"), rtol=0.0, atol=1e-12)

    cov_a = MatlabCovariate(time=time, data=data[:, 0], name="a", units="u", labels=["a"], x_label="time", x_units="s", y_units="u")
    ci_a = MatlabConfidenceInterval(time=time, lower=data[:, 0] - 0.10, upper=data[:, 0] + 0.20)
    cov_a.setConfInterval(ci_a)

    plus_scalar = cov_a.plus(0.5)
    minus_scalar = cov_a.minus(0.5)
    np.testing.assert_allclose(plus_scalar.dataToMatrix(), _arr(m, "plus_scalar_data"), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(_ci_matrix(plus_scalar), _arr(m, "plus_scalar_ci"), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(minus_scalar.dataToMatrix(), _arr(m, "minus_scalar_data"), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(_ci_matrix(minus_scalar), _arr(m, "minus_scalar_ci"), rtol=0.0, atol=1e-12)

    cov_no_ci_1 = MatlabCovariate(time=time, data=data[:, 0], name="n1", units="u", labels=["n1"], x_label="time", x_units="s", y_units="u")
    cov_no_ci_2 = MatlabCovariate(time=time, data=data[:, 0] + 0.25, name="n2", units="u", labels=["n2"], x_label="time", x_units="s", y_units="u")
    np.testing.assert_allclose(cov_no_ci_1.plus(cov_no_ci_2).dataToMatrix(), _arr(m, "plus_no_ci_data"), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(cov_no_ci_1.minus(cov_no_ci_2).dataToMatrix(), _arr(m, "minus_no_ci_data"), rtol=0.0, atol=1e-12)

    cov_b = MatlabCovariate(time=time, data=np.full(time.size, 0.5), name="b", units="u", labels=["b"], x_label="time", x_units="s", y_units="u")
    assert cov_b.isConfIntervalSet() == bool(_scalar(m, "is_ci_before"))
    cov_b.setConfInterval(ci_a)
    assert cov_b.isConfIntervalSet() == bool(_scalar(m, "is_ci_after"))

    filt = cov.filtfilt(np.array([0.2, 0.2]), np.array([1.0, -0.3]))
    np.testing.assert_allclose(filt.dataToMatrix(), _arr(m, "filt_data"), rtol=0.0, atol=2e-3)


def test_covariate_compat_structure_roundtrip_matches_matlab_fixture() -> None:
    m = _mat()
    mat_struct = np.asarray(m["cov_struct"], dtype=object).reshape(-1)[0]

    restored = MatlabCovariate.fromStructure(mat_struct)
    np.testing.assert_allclose(restored.dataToMatrix(), _arr(m, "roundtrip_data"), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(_ci_matrix(restored), _arr(m, "roundtrip_ci"), rtol=0.0, atol=1e-12)

    payload = restored.toStructure()
    reloaded = MatlabCovariate.fromStructure(payload)
    np.testing.assert_allclose(reloaded.dataToMatrix(), restored.dataToMatrix(), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(_ci_matrix(reloaded), _ci_matrix(restored), rtol=0.0, atol=1e-12)
