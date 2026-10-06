"""EM numerics that mirror the repaired MATLAB exactly (nSTAT PR #135, ``fix/pp-em`` @ ``aa88a2b``).

* Newton-Raphson loop length: MATLAB's M-step loops are ``iter = 1; while
  (~converged && iter < maxIter)`` with ``maxIter = 100`` -- at most 99 Newton
  steps.  ``PP_MStep`` ran ``range(100)``.
"""
from __future__ import annotations

import numpy as np

from nstat.decoding_algorithms import DecodingAlgorithms


def _nonconverging_mstep_problem(K=200, x=0.5):
    """One cell that never spikes, on a known state (W_K = 1e-30): each Newton step
    moves mu and gamma by exactly -1 and beta by -1/x, so none converges."""
    dN = np.zeros((1, K))
    x_K = np.full((1, K), x)
    W_K = np.full((1, 1, K), 1e-30)
    HkAll = np.ones((K, 1, 1))  # one history window, count 1 in every bin
    ES = {"Sxkm1xkm1": np.array([[K * x * x]]), "Sxkxkm1": np.array([[K * x * x]]),
          "sumXkTerms": np.array([[1e-3]])}
    return dN, x_K, W_K, HkAll, ES


def test_newton_raphson_runs_at_most_99_steps() -> None:
    # MATLAB (and this port now): 99 steps of -1 from mu = -3, gamma = -0.5 and
    # of -1/x = -2 from beta = 0.2.  The port ran 100.
    dN, x_K, W_K, HkAll, ES = _nonconverging_mstep_problem()
    np.random.seed(0)
    _, _, mu, beta, gamma, _, _ = DecodingAlgorithms.PP_MStep(
        dN, x_K, W_K, np.zeros(1), np.eye(1), ES, "poisson", np.array([-3.0]), np.array([[0.2]]),
        np.array([[-0.5]]), np.array([0.0, 0.001]), HkAll, DecodingAlgorithms.PP_EMCreateConstraints(),
        "NewtonRaphson", 0.001)
    assert float(mu[0]) == -3.0 - 99
    assert float(np.ravel(gamma)[0]) == -0.5 - 99
    np.testing.assert_allclose(float(np.ravel(beta)[0]), 0.2 - 99 / 0.5, rtol=1e-12)
