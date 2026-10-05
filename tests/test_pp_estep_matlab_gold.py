"""``DecodingAlgorithms.PP_EStep`` against the MATLAB gold fixture.

Gold: ``tests/parity/fixtures/matlab_gold/pp_estep.mat``, captured from MATLAB
``nstat.decoding.PointProcessEM.PP_EStep`` by
``tools/parity/matlab/capture_pp_estep.m`` (rng(42) synthetic inputs; dx = 2
states, C = 3 cells, N = 150 bins; recaptured from the repaired MATLAB
``fix/pp-em`` @ ``a457b54``, pending upstream merge, which added c5 / c6 and
left c1-c4 bit-identical).  Cases:

* ``c1`` poisson, no history (``HkAll = zeros(N, 1, C)``, ``gamma = 0``)
* ``c2`` poisson, 2-window history, nonzero ``gamma``
* ``c3`` binomial, 4-window history, nonzero ``gamma``
* ``c4`` binomial, no history
* ``c5`` / ``c6`` poisson / binomial, square history (nW == C == 3,
  non-symmetric ``gamma``).  MATLAB's repaired PP_EStep orients each
  ``nW x C`` history slice of its log-likelihood by its columns; the former
  rows test (mirrored here until this fix) transposed a square slice and
  paired ``gamma(w, c)`` with ``H(c, w)`` -- the logll was off by ~1.

The MATLAB-computed ``HkAll`` (``History.computeHistory`` per cell, exactly as
``PP_EM`` builds it) is fed straight to Python, so the comparison isolates
``PP_EStep`` itself.  Every output is compared at every time step: ``x_K``
(dx x N), ``W_K`` (dx x dx x N), ``logll`` and every ``ExpectationSums`` field.

Before the fix, ``PP_EStep`` passed ``k + 1`` to the zero-based
``PPDecode_updateLinear`` (reading the next bin, ``IndexError`` on the last)
together with a MATLAB-style ``(nW, C, N)`` permuted history tensor that the
Python decoder does not accept (``ValueError`` whenever nW != C).

Tolerance: measured on macOS arm64 / Accelerate vs MATLAB R2025b, the largest
absolute errors are 1.0e-15 (x_K), 3.4e-16 (W_K), 8.5e-14 (sufficient
statistics, magnitudes 1-80), 2.6e-12 (logll, magnitudes 4-54; it sums
K * log det(Q) and trace(Q^-1 S) terms) and 0 (sumPPll).  ``rtol=1e-10``,
``atol=1e-12`` is >= ~100x above the worst element-wise error, leaving room
for BLAS/LAPACK last-ulp differences on other platforms.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from nstat.decoding_algorithms import DecodingAlgorithms

FIXTURE = Path(__file__).resolve().parent / "parity" / "fixtures" / "matlab_gold" / "pp_estep.mat"
RTOL = 1e-10
ATOL = 1e-12
CASES = {
    "c1": ("poisson", False),
    "c2": ("poisson", True),
    "c3": ("binomial", True),
    "c4": ("binomial", False),
    "c5": ("poisson", True),
    "c6": ("binomial", True),
}
SQUARE_CASES = ("c5", "c6")


@pytest.fixture(scope="module")
def gold() -> dict:
    return loadmat(FIXTURE, squeeze_me=False, struct_as_record=False)


def _case_inputs(gold: dict, case: str):
    def f(key: str) -> np.ndarray:
        return np.asarray(gold[f"{case}_{key}"], dtype=float)

    gamma = f("gamma")
    # MATLAB scalar gamma = 0 (no history) is passed as a Python scalar.
    gamma_arg = float(gamma.reshape(-1)[0]) if gamma.size == 1 else gamma
    fit_type = str(np.asarray(gold[f"{case}_fitType"]).reshape(-1)[0])
    return (
        f("A"), f("Q"), f("dN"), f("mu"), f("beta"), fit_type, gamma_arg,
        f("HkAll"), f("x0"), f("Px0"),
    )


def test_fixture_cases_are_the_documented_ones(gold) -> None:
    assert [str(np.asarray(c).reshape(-1)[0]) for c in gold["case_names"].reshape(-1)] == list(CASES)
    for case, (fit_type, has_history) in CASES.items():
        A, Q, dN, mu, beta, fit, gamma, HkAll, x0, Px0 = _case_inputs(gold, case)
        num_cells, N = dN.shape
        assert fit == fit_type
        assert HkAll.ndim == 3 and HkAll.shape[0] == N and HkAll.shape[2] == num_cells
        if has_history:
            # c2 / c3 have nW not in {1, C}; c5 / c6 are square (nW == C) with a
            # non-symmetric gamma, so a window/cell transpose cannot pass.
            if case in SQUARE_CASES:
                assert HkAll.shape[1] == num_cells
                assert np.max(np.abs(gamma - gamma.T)) > 0.1
            else:
                assert HkAll.shape[1] not in (1, num_cells)
            assert np.any(HkAll != 0) and np.ndim(gamma) == 2 and np.any(gamma != 0)
            assert gamma.shape == (HkAll.shape[1], num_cells)
        else:
            assert HkAll.shape[1] == 1 and not np.any(HkAll) and gamma == 0.0


@pytest.mark.parametrize("case", list(CASES))
def test_pp_estep_matches_matlab_gold_at_every_step(gold, case) -> None:
    A, Q, dN, mu, beta, fit, gamma, HkAll, x0, Px0 = _case_inputs(gold, case)
    x_K, W_K, logll, sums = DecodingAlgorithms.PP_EStep(A, Q, dN, mu, beta, fit, gamma, HkAll, x0, Px0)

    dx, N = A.shape[0], dN.shape[1]
    ml_x_K = np.asarray(gold[f"{case}_x_K"], dtype=float)
    ml_W_K = np.asarray(gold[f"{case}_W_K"], dtype=float)
    assert x_K.shape == ml_x_K.shape == (dx, N)
    assert W_K.shape == ml_W_K.shape == (dx, dx, N)
    np.testing.assert_allclose(x_K, ml_x_K, rtol=RTOL, atol=ATOL, err_msg="x_K")
    np.testing.assert_allclose(W_K, ml_W_K, rtol=RTOL, atol=ATOL, err_msg="W_K")
    np.testing.assert_allclose(
        logll, float(np.asarray(gold[f"{case}_logll"]).reshape(-1)[0]), rtol=RTOL, atol=ATOL, err_msg="logll"
    )

    prefix = f"{case}_ES_"
    ml_keys = sorted(k[len(prefix):] for k in gold if k.startswith(prefix))
    assert sorted(sums) == ml_keys
    for key in ml_keys:
        expected = np.asarray(gold[prefix + key], dtype=float)
        actual = np.asarray(sums[key], dtype=float)
        if expected.size == 1:  # MATLAB scalars load as 1 x 1
            actual, expected = actual.reshape(()), expected.reshape(())
        assert actual.shape == expected.shape, key
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, err_msg=f"ExpectationSums.{key}")
