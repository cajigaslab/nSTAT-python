from __future__ import annotations

import numpy as np

from nstat.CIF import CIF
from nstat.DecodingAlgorithms import DecodingAlgorithms
from nstat.History import History


def test_ppdecodefilterlinear_matches_matlab_style_shapes() -> None:
    a = 1.0
    q = 0.05
    dN = np.array([[0.0, 1.0, 0.0, 1.0]], dtype=float)
    mu = np.array([-1.0], dtype=float)
    beta = np.array([[0.75]], dtype=float)

    x_p, W_p, x_u, W_u, x_uT, W_uT, x_pT, W_pT = DecodingAlgorithms.PPDecodeFilterLinear(
        a,
        q,
        dN,
        mu,
        beta,
        "binomial",
        0.1,
    )

    assert x_p.shape == (1, 5)
    assert W_p.shape == (1, 1, 5)
    assert x_u.shape == (1, 4)
    assert W_u.shape == (1, 1, 4)
    assert x_uT.size == 0
    assert W_uT.shape == (0, 0, 0)
    assert x_pT.size == 0
    assert W_pT.shape == (0, 0, 0)
    assert np.all(np.isfinite(x_u))
    assert np.all(np.isfinite(W_u))


def test_ppdecodefilter_accepts_cif_collections_with_history() -> None:
    dN = np.array([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]], dtype=float)
    history = History([0.0, 0.1, 0.2])
    lambda_cifs = [
        CIF([0.1, 0.4], ["1", "x"], ["x"], fitType="binomial", histCoeffs=[0.2, 0.1], historyObj=history),
        CIF([-0.2, -0.3], ["1", "x"], ["x"], fitType="binomial", histCoeffs=[0.1, 0.05], historyObj=history),
    ]

    x_p, W_p, x_u, W_u, *_ = DecodingAlgorithms.PPDecodeFilter(1.0, 0.02, 0.1, dN, lambda_cifs, 0.1)

    assert x_p.shape == (1, 5)
    assert W_p.shape == (1, 1, 5)
    assert x_u.shape == (1, 4)
    assert W_u.shape == (1, 1, 4)
    assert np.all(np.isfinite(x_u))


def test_pphybridfilterlinear_returns_model_probabilities_and_state_banks() -> None:
    a = [np.array([[1.0]], dtype=float), np.array([[0.9]], dtype=float)]
    q = [np.array([[0.02]], dtype=float), np.array([[0.05]], dtype=float)]
    p_ij = np.array([[0.95, 0.05], [0.10, 0.90]], dtype=float)
    mu0 = np.array([0.6, 0.4], dtype=float)
    dN = np.array([[0.0, 1.0, 1.0, 0.0, 1.0]], dtype=float)
    mu = [np.array([-1.0], dtype=float), np.array([-0.5], dtype=float)]
    beta = [np.array([[0.5]], dtype=float), np.array([[1.1]], dtype=float)]

    S_est, X, W, MU_u, X_s, W_s, pNGivenS = DecodingAlgorithms.PPHybridFilterLinear(
        a,
        q,
        p_ij,
        mu0,
        dN,
        mu,
        beta,
        "binomial",
        0.1,
    )

    assert S_est.shape == (5,)
    assert X.shape == (1, 5)
    assert W.shape == (1, 1, 5)
    assert MU_u.shape == (2, 5)
    assert pNGivenS.shape == (2, 5)
    assert len(X_s) == 2
    assert len(W_s) == 2
    np.testing.assert_allclose(np.sum(MU_u, axis=0), np.ones(5), atol=1e-6)
