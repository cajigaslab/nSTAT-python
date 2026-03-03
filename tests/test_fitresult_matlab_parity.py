from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from nstat.compat.matlab import FitResult as MatlabFitResult


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "FitResult" / "basic.mat"


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
    return value


def _vec(m: dict[str, object], key: str) -> np.ndarray:
    return np.asarray(m[key], dtype=float).reshape(-1)


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


def test_fitresult_core_methods_match_matlab_fixture() -> None:
    m = _mat()
    X = np.asarray(m["X"], dtype=float)
    beta = _vec(m, "beta")

    fit = MatlabFitResult(
        coefficients=beta,
        intercept=0.0,
        fit_type="binomial",
        log_likelihood=-1.6,
        n_samples=int(X.shape[0]),
        n_parameters=2,
        parameter_labels=["stim1", "stim2"],
        xval_data=[X],
        xval_time=[_vec(m, "time")],
    )
    fit.addParamsToFit(
        {
            "plot_params": {
                "bAct": np.asarray(m["plot_bAct"], dtype=float),
                "seAct": np.asarray(m["plot_seAct"], dtype=float),
                "sigIndex": np.asarray(m["plot_sigIndex"], dtype=float),
                "xLabels": _cellstr(m["plot_xLabels"]),
                "numResultsCoeffPresent": np.ones(2, dtype=float),
            }
        }
    )

    lambda_eval = np.asarray(fit.evalLambda(1, X), dtype=float).reshape(-1)
    np.testing.assert_allclose(lambda_eval, _vec(m, "lambda_eval"), rtol=1e-12, atol=1e-12)

    coeff_index, epoch_id, num_epochs = fit.getCoeffIndex(1, False)
    np.testing.assert_array_equal(np.asarray(coeff_index, dtype=int).reshape(-1), np.asarray(m["coeff_index"], dtype=int).reshape(-1))
    np.testing.assert_array_equal(np.asarray(epoch_id, dtype=int).reshape(-1), np.array([0, 0], dtype=int))
    assert int(num_epochs) == 1

    coeff_mat, coeff_labels, coeff_se = fit.getCoeffs(1)
    np.testing.assert_allclose(np.asarray(coeff_mat, dtype=float), np.asarray(m["coeff_mat"], dtype=float), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(coeff_se, dtype=float), np.asarray(m["coeff_se"], dtype=float), rtol=0.0, atol=1e-12)
    assert [row[0] for row in coeff_labels] == _cellstr(m["coeff_labels"])

    p_vals, p_se, p_sig = fit.getParam(["stim1", "stim2"], 1)
    np.testing.assert_allclose(np.asarray(p_vals, dtype=float), np.asarray(m["param_vals"], dtype=float), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(p_se, dtype=float), np.asarray(m["param_se"], dtype=float), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(p_sig, dtype=float), np.asarray(m["param_sig"], dtype=float), rtol=0.0, atol=1e-12)

    assert bool(fit.isValDataPresent()) is bool(np.asarray(m["is_val_present"]).reshape(-1)[0])


def test_fitresult_plot_params_match_matlab_fixture() -> None:
    m = _mat()
    beta = _vec(m, "beta")

    fit = MatlabFitResult(
        coefficients=beta,
        intercept=0.0,
        fit_type="binomial",
        log_likelihood=-1.6,
        n_samples=5,
        n_parameters=2,
        parameter_labels=["stim1", "stim2"],
    )
    fit.addParamsToFit(
        {
            "plot_params": {
                "bAct": np.asarray(m["plot_bAct"], dtype=float),
                "seAct": np.asarray(m["plot_seAct"], dtype=float),
                "sigIndex": np.asarray(m["plot_sigIndex"], dtype=float),
                "xLabels": _cellstr(m["plot_xLabels"]),
                "numResultsCoeffPresent": np.ones(2, dtype=float),
            }
        }
    )
    params = fit.getPlotParams()

    np.testing.assert_allclose(np.asarray(params["bAct"], dtype=float), np.asarray(m["plot_bAct"], dtype=float), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(params["seAct"], dtype=float), np.asarray(m["plot_seAct"], dtype=float), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(params["sigIndex"], dtype=float), np.asarray(m["plot_sigIndex"], dtype=float), rtol=0.0, atol=1e-12)
    assert [str(v) for v in params["xLabels"]] == _cellstr(m["plot_xLabels"])
