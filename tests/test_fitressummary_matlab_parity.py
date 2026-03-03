from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from nstat.compat.matlab import FitResSummary as MatlabFitResSummary
from nstat.compat.matlab import FitResult as MatlabFitResult


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "FitResSummary" / "basic.mat"


def _mat() -> dict[str, object]:
    return loadmat(str(FIXTURE), squeeze_me=False, struct_as_record=False)


def _vec(m: dict[str, object], key: str) -> np.ndarray:
    return np.asarray(m[key], dtype=float).reshape(-1)


def test_fitressummary_diff_metrics_match_matlab_fixture() -> None:
    m = _mat()

    logll = _vec(m, "logLL")
    f1 = MatlabFitResult(
        coefficients=np.array([0.4, -0.2], dtype=float),
        intercept=0.0,
        fit_type="binomial",
        log_likelihood=float(logll[0]),
        n_samples=5,
        n_parameters=0,
        parameter_labels=["stim1", "stim2"],
    )
    f2 = MatlabFitResult(
        coefficients=np.array([0.1, 0.3], dtype=float),
        intercept=0.0,
        fit_type="binomial",
        log_likelihood=float(logll[1]),
        n_samples=5,
        n_parameters=0,
        parameter_labels=["stim1", "stim2"],
    )

    summary = MatlabFitResSummary([f1, f2])

    d_aic = np.asarray(summary.getDiffAIC(1, False), dtype=float).reshape(-1)
    d_bic = np.asarray(summary.getDiffBIC(1, False), dtype=float).reshape(-1)
    d_logll = np.asarray(summary.getDifflogLL(1, False), dtype=float).reshape(-1)

    np.testing.assert_allclose(d_aic, _vec(m, "diff_aic"), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(d_bic, _vec(m, "diff_bic"), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(d_logll, _vec(m, "diff_logll"), rtol=0.0, atol=1e-12)


def test_fitressummary_index_helpers_match_matlab_fixture() -> None:
    m = _mat()
    f1 = MatlabFitResult(
        coefficients=np.array([0.4, -0.2], dtype=float),
        intercept=0.0,
        fit_type="binomial",
        log_likelihood=-1.6,
        n_samples=5,
        n_parameters=0,
        parameter_labels=["stim1", "stim2"],
    )
    f2 = MatlabFitResult(
        coefficients=np.array([0.1, 0.3], dtype=float),
        intercept=0.0,
        fit_type="binomial",
        log_likelihood=-1.4,
        n_samples=5,
        n_parameters=0,
        parameter_labels=["stim1", "stim2"],
    )
    summary = MatlabFitResSummary([f1, f2])

    coeff_idx, coeff_epoch, coeff_epochs = summary.getCoeffIndex(1, False)
    np.testing.assert_array_equal(np.asarray(coeff_idx, dtype=int).reshape(-1), np.asarray(m["coeff_index"], dtype=int).reshape(-1))
    assert int(coeff_epochs) == int(np.asarray(m["coeff_num_epochs"]).reshape(-1)[0])
    np.testing.assert_array_equal(np.asarray(coeff_epoch, dtype=int).reshape(-1), np.asarray(m["coeff_epoch_id"], dtype=int).reshape(-1))

    hist_idx, hist_epoch, hist_epochs = summary.getHistIndex(1, False)
    np.testing.assert_array_equal(np.asarray(hist_idx, dtype=int).reshape(-1), np.asarray(m["hist_index"], dtype=int).reshape(-1))
    np.testing.assert_array_equal(np.asarray(hist_epoch, dtype=int).reshape(-1), np.asarray(m["hist_epoch_id"], dtype=int).reshape(-1))
    assert int(hist_epochs) == int(np.asarray(m["hist_num_epochs"]).reshape(-1)[0])
