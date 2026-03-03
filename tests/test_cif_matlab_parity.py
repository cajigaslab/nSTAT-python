from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from nstat.compat.matlab import CIF as MatlabCIF


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "CIF" / "basic.mat"


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


def _scalar(m: dict[str, object], key: str) -> float:
    return float(np.asarray(m[key], dtype=float).reshape(-1)[0])


def test_cif_poisson_and_binomial_derivatives_match_matlab_fixture() -> None:
    m = _mat()
    beta = np.asarray(m["beta"], dtype=float).reshape(-1)
    X = np.asarray(m["stim_vals"], dtype=float)

    poisson = MatlabCIF(coefficients=beta, intercept=0.0, link="poisson")
    binomial = MatlabCIF(coefficients=beta, intercept=0.0, link="binomial")

    np.testing.assert_allclose(
        np.asarray(poisson.evalLambdaDelta(X), dtype=float).reshape(-1),
        np.asarray(m["poisson_lambda_delta"], dtype=float).reshape(-1),
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(poisson.evalGradient(X), dtype=float),
        np.asarray(m["poisson_gradient"], dtype=float),
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(poisson.evalGradientLog(X), dtype=float),
        np.asarray(m["poisson_gradient_log"], dtype=float),
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(poisson.evalJacobian(X), dtype=float),
        np.transpose(np.asarray(m["poisson_jacobian"], dtype=float), (2, 0, 1)),
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(poisson.evalJacobianLog(X), dtype=float),
        np.transpose(np.asarray(m["poisson_jacobian_log"], dtype=float), (2, 0, 1)),
        rtol=1e-9,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(binomial.evalLambdaDelta(X), dtype=float).reshape(-1),
        np.asarray(m["binomial_lambda_delta"], dtype=float).reshape(-1),
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(binomial.evalGradient(X), dtype=float),
        np.asarray(m["binomial_gradient"], dtype=float),
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(binomial.evalGradientLog(X), dtype=float),
        np.asarray(m["binomial_gradient_log"], dtype=float),
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(binomial.evalJacobian(X), dtype=float),
        np.transpose(np.asarray(m["binomial_jacobian"], dtype=float), (2, 0, 1)),
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(binomial.evalJacobianLog(X), dtype=float),
        np.transpose(np.asarray(m["binomial_jacobian_log"], dtype=float), (2, 0, 1)),
        rtol=1e-9,
        atol=1e-12,
    )


def test_cif_copy_and_flags_match_matlab_fixture() -> None:
    m = _mat()
    beta = np.asarray(m["beta"], dtype=float).reshape(-1)
    poisson = MatlabCIF(coefficients=beta, intercept=0.0, link="poisson")

    copy_obj = poisson.CIFCopy()
    np.testing.assert_allclose(
        np.asarray(copy_obj.coefficients, dtype=float).reshape(-1),
        np.asarray(m["copy_b"], dtype=float).reshape(-1),
        rtol=0.0,
        atol=1e-12,
    )
    assert str(copy_obj.link) == str(_to_python(m["copy_fitType"]))
    assert bool(poisson.isSymBeta()) == bool(_scalar(m, "is_sym_beta"))
