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


def test_ppdecodefilter_handles_symbolic_style_polynomial_cifs() -> None:
    dN = np.array([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]], dtype=float)
    lambda_cifs = [
        CIF([-2.0, -0.5, 0.3, -0.2, -0.1, 0.05], ["1", "x", "y", "x^2", "y^2", "x*y"], ["x", "y"], fitType="binomial"),
        CIF([-1.5, 0.4, -0.2, 0.15, -0.05, 0.02], ["1", "x", "y", "x^2", "y^2", "x*y"], ["x", "y"], fitType="binomial"),
    ]

    x_p, W_p, x_u, W_u, *_ = DecodingAlgorithms.PPDecodeFilter(np.eye(2), 0.01 * np.eye(2), 0.05 * np.eye(2), dN, lambda_cifs, 0.1)

    assert x_p.shape == (2, 5)
    assert W_p.shape == (2, 2, 5)
    assert x_u.shape == (2, 4)
    assert W_u.shape == (2, 2, 4)
    assert np.all(np.isfinite(x_u))
    assert np.all(np.isfinite(W_u))


def test_ppdecode_update_matches_matlab_facing_public_surface() -> None:
    dN = np.array([[0.0, 1.0, 0.0, 1.0]], dtype=float)
    lambda_cif = CIF([0.1, 0.4], ["1", "x"], ["x"], fitType="binomial")

    x_u, W_u, lambda_delta = DecodingAlgorithms.PPDecode_update(
        np.array([0.0], dtype=float),
        np.array([[1.0]], dtype=float),
        dN,
        lambda_cif,
        0.1,
        2,
    )

    assert x_u.shape == (1,)
    assert W_u.shape == (1, 1)
    assert lambda_delta.shape == (1, 1)
    assert np.all(np.isfinite(x_u))
    assert np.all(np.isfinite(W_u))
    assert np.all(lambda_delta > 0.0)


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


def test_kalman_helper_methods_and_confidence_intervals_are_available() -> None:
    A = np.array([[1.0]], dtype=float)
    C = np.array([[1.0]], dtype=float)
    Q = np.array([[0.05]], dtype=float)
    R = np.array([[0.02]], dtype=float)
    x0 = np.array([0.0], dtype=float)
    P0 = np.array([[1.0]], dtype=float)
    y = np.array([[0.0], [0.1], [0.2], [0.1]], dtype=float)

    x_p, P_p = DecodingAlgorithms.kalman_predict(x0, P0, A, Q)
    x_u, P_u, G = DecodingAlgorithms.kalman_update(x_p, P_p, C, R, y[0])
    assert x_p.shape == (1,)
    assert P_p.shape == (1, 1)
    assert x_u.shape == (1,)
    assert P_u.shape == (1, 1)
    assert G.shape == (1, 1)

    x_N, P_N, Ln, x_pred_hist, P_pred_hist, x_upd_hist, P_upd_hist = DecodingAlgorithms.kalman_smoother(A, C, Q, R, P0, x0, y)
    assert x_N.shape == (4, 1)
    assert P_N.shape == (4, 1, 1)
    assert Ln.shape == (3, 1, 1)
    assert x_pred_hist.shape == (4, 1)
    assert x_upd_hist.shape == (4, 1)

    x_pLag, P_pLag, x_uLag, P_uLag = DecodingAlgorithms.kalman_fixedIntervalSmoother(A, C, Q, R, P0, x0, y, 2)
    assert x_pLag.shape == (4, 1)
    assert P_pLag.shape == (4, 1, 1)
    assert x_uLag.shape == (4, 1)
    assert P_uLag.shape == (4, 1, 1)

    cis, stimulus = DecodingAlgorithms.ComputeStimulusCIs("poisson", x_N, P_N, 0.1, alphaVal=0.05)
    assert cis.shape == (4, 1, 2)
    assert stimulus.shape == (4, 1)
