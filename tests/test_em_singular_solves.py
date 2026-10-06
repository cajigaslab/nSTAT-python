"""The EM closed-form M-step / standard-error solves mirror MATLAB's ``/``
and ``\\`` on an exactly singular matrix (Inf/NaN, not a raised exception).

MATLAB's ``\\`` (mldivide) and ``/`` (mrdivide) on an exactly singular matrix
return +-Inf / NaN with MATLAB's "singular to working precision" warning
(not reproduced); ``np.linalg.solve`` / ``np.linalg.inv`` instead raise
``LinAlgError``. The closed-form M-step updates (``Ahat``, ``AtQinv``,
``x0hat`` in ``PP_MStep`` / ``PPLFP_MStep``) and the SE information blocks
(``Qinv``, ``Px0inv`` and the ``Rhat`` / ``Qhat`` / ``Px0hat`` solves of
``PP_ComputeParamStandardErrors`` / ``PPLFP_ComputeParamStandardErrors``)
previously used ``np.linalg.solve`` / ``np.linalg.inv`` directly; only the
Newton steps (``_matlab_mldivide``) already mirrored MATLAB here
(parity/matlab_defects.yml: em-newton-solve-reciprocal-pivot). This is
recorded as "not reached by any gold input" in that same ledger entry, so
this is a synthetic regression, not a MATLAB-captured one.
"""
from __future__ import annotations

import numpy as np
import pytest

from nstat.decoding_algorithms import (
    DecodingAlgorithms,
    _matlab_inv,
    _matlab_mldivide_matrix,
    _matlab_mrdivide,
)
from nstat.decoding.PPLFP import PPLFP


# A 2x2 exactly singular matrix. em-newton-solve-reciprocal-pivot's own
# example is the NEGATED matrix (-[1 1;1 1]\[1;2] == [-Inf; Inf]); checked
# directly against MATLAB R2026a that the POSITIVE matrix used here gives
# the identical result ([1 1;1 1]\[1;2] == [-Inf; Inf] too, not just its
# negation) -- this is not simply inferred from the negated case.
_SINGULAR = np.array([[1.0, 1.0], [1.0, 1.0]])


def test_matlab_mldivide_matrix_singular_gives_inf_not_raise() -> None:
    g = np.array([1.0, 2.0])
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(_SINGULAR, g)  # the pre-fix behaviour
    got = _matlab_mldivide_matrix(_SINGULAR, g)
    assert np.array_equal(got, np.array([-np.inf, np.inf]))


def test_matlab_mldivide_matrix_singular_matrix_rhs() -> None:
    G = np.eye(2)
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(_SINGULAR, G)
    got = _matlab_mldivide_matrix(_SINGULAR, G)
    assert np.all(~np.isfinite(got))


def test_matlab_mrdivide_singular_gives_inf_not_raise() -> None:
    A = np.array([[1.0, 2.0]])
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(_SINGULAR.T, A.T)
    got = _matlab_mrdivide(A, _SINGULAR)
    assert np.all(~np.isfinite(got))


def test_matlab_inv_singular_gives_inf_not_raise() -> None:
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.inv(_SINGULAR)
    got = _matlab_inv(_SINGULAR)
    assert np.all(~np.isfinite(got))


def _singular_mstep_problem(dx=2, C=3, nW=2, K=50, seed=7):
    """An M-step problem whose Sxkm1xkm1 is exactly singular (a constant
    state path: every column of x identical, so its outer-product sum has
    rank 1 for dx=2)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dx)
    x = np.tile(v[:, None], (1, K))  # every column identical -> rank-1 Sxkm1xkm1
    mu = np.linspace(-2.5, -2.0, C)
    beta = 0.8 * rng.standard_normal((dx, C))
    gamma = np.zeros((nW, C))
    wt = np.arange(nW + 1) * 0.001
    dN = (rng.random((C, K)) < 0.05).astype(float)
    from nstat.decoding_algorithms import _compute_history_terms

    HkAll = _compute_history_terms(dN, 0.001, wt)
    W_K = np.tile((0.02 * np.eye(dx))[:, :, None], (1, 1, K))
    ES = dict(
        Sxkm1xkm1=x @ x.T,  # exactly singular: rank 1
        Sxkxkm1=x[:, 1:] @ x[:, :-1].T,
        Sxkm1xk=x[:, :-1] @ x[:, 1:].T,
        Sxkxk=x @ x.T,
        sumXkTerms=0.02 * K * np.eye(dx),
        Sxkyk=x @ (np.array([[1.0, 0.5]])[:, :dx] @ x).T,
        sumYkTerms=0.1 * K * np.eye(1),
    )
    y = np.array([[1.0, 0.5]])[:, :dx] @ x
    return dict(x=x, dN=dN, wt=wt, HkAll=HkAll, mu=mu, beta=beta, gamma=gamma, W_K=W_K, ES=ES, dx=dx, C=C, K=K, y=y)


def test_pp_mstep_singular_sxkm1xkm1_gives_non_finite_ahat_not_raise() -> None:
    P = _singular_mstep_problem()
    assert np.linalg.matrix_rank(P["ES"]["Sxkm1xkm1"]) < P["dx"]
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(P["ES"]["Sxkm1xkm1"].T, P["ES"]["Sxkxkm1"].T)  # the pre-fix path

    cons = DecodingAlgorithms.PP_EMCreateConstraints(1, 0, 1, 0, 0, 0)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Ahat, Qhat, muhat_new, betahat_new, gammahat_new, x0hat, Px0hat = DecodingAlgorithms.PP_MStep(
            P["dN"], P["x"], P["W_K"], np.zeros(P["dx"]), 1e-9 * np.eye(P["dx"]), P["ES"], "binomial",
            P["mu"], P["beta"], P["gamma"], P["wt"], P["HkAll"], cons, "NewtonRaphson",
        )
    assert np.any(~np.isfinite(Ahat)), Ahat


def test_pplfp_mstep_singular_sxkm1xkm1_gives_non_finite_ahat_not_raise() -> None:
    P = _singular_mstep_problem()
    cons = PPLFP.PPLFP_EMCreateConstraints(1, 0, 1, 0, 1, 0, 0, 0, 0, 50, 0)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = PPLFP.PPLFP_MStep(
            P["dN"], P["y"], P["x"], P["W_K"], np.zeros(P["dx"]), 1e-9 * np.eye(P["dx"]), P["ES"], "binomial",
            P["mu"], P["beta"], P["gamma"], P["wt"], P["HkAll"], cons, "NewtonRaphson",
        )
    Ahat = result[0]
    assert np.any(~np.isfinite(Ahat)), Ahat
