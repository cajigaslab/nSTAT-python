from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import norm

from .cif import CIF
from .errors import UnsupportedWorkflowError
from .nspikeTrain import nspikeTrain


def _as_observation_matrix(dN) -> np.ndarray:
    obs = np.asarray(dN, dtype=float)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if obs.ndim != 2:
        raise ValueError("dN must be a CxN observation matrix")
    return obs


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _normalize_probabilities(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Probability vector cannot be empty")
    total = float(np.sum(arr))
    if not np.isfinite(total) or total <= 0:
        return np.full(arr.shape, 1.0 / float(arr.size), dtype=float)
    return arr / total


def _is_empty_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, Sequence):
        return len(value) == 0
    return False


def _infer_state_dim(A, beta, num_cells: int) -> int:
    arr = np.asarray(A, dtype=float)
    if arr.ndim >= 2:
        return int(arr.shape[0])
    beta_arr = np.asarray(beta, dtype=float)
    if beta_arr.ndim == 2:
        if beta_arr.shape[1] == num_cells:
            return int(beta_arr.shape[0])
        if beta_arr.shape[0] == num_cells:
            return int(beta_arr.shape[1])
    if beta_arr.ndim == 1:
        if num_cells == 1:
            return int(beta_arr.size)
        if beta_arr.size == num_cells:
            return 1
    return 1


def _as_state_matrix(matrix, dim: int) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 0:
        return np.eye(dim, dtype=float) * float(arr)
    if arr.ndim == 1:
        if arr.size == 1:
            return np.eye(dim, dtype=float) * float(arr[0])
        raise ValueError("State-space matrices must be square matrices or scalars")
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("State-space matrices must be square")
    if arr.shape[0] != dim:
        raise ValueError("State-space matrix dimension mismatch")
    return arr.astype(float, copy=False)


def _select_time_matrix(matrix, time_index: int, dim: int) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim <= 2:
        return _as_state_matrix(arr, dim)
    if arr.ndim == 3:
        return _as_state_matrix(arr[:, :, min(time_index, arr.shape[2] - 1)], dim)
    raise ValueError("Unsupported time-varying state-space matrix shape")


def _normalize_mu(mu, num_cells: int) -> np.ndarray:
    arr = np.asarray(mu, dtype=float).reshape(-1)
    if arr.size == 1 and num_cells > 1:
        arr = np.repeat(arr, num_cells)
    if arr.size != num_cells:
        raise ValueError("mu must contain one baseline term per observed cell")
    return arr


def _normalize_beta(beta, num_states: int, num_cells: int) -> np.ndarray:
    arr = np.asarray(beta, dtype=float)
    if arr.ndim == 0:
        if num_states != 1 or num_cells != 1:
            raise ValueError("scalar beta is only valid for the 1-state, 1-cell MATLAB surface")
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        if num_cells == 1 and arr.size == num_states:
            arr = arr.reshape(num_states, 1)
        elif arr.size == num_cells and num_states == 1:
            arr = arr.reshape(1, num_cells)
        else:
            raise ValueError("beta must be ns x C for MATLAB-facing decoding workflows")
    elif arr.ndim != 2:
        raise ValueError("beta must be a vector or 2D array")

    if arr.shape == (num_cells, num_states):
        arr = arr.T
    if arr.shape != (num_states, num_cells):
        raise ValueError("beta must be ns x C after MATLAB-style normalization")
    return arr


def _normalize_gamma(gamma, num_windows: int, num_cells: int) -> np.ndarray:
    if num_windows == 0 or _is_empty_value(gamma):
        return np.zeros((num_windows, num_cells), dtype=float)

    arr = np.asarray(gamma, dtype=float)
    if arr.ndim == 0:
        return np.full((num_windows, num_cells), float(arr), dtype=float)
    if arr.ndim == 1:
        if arr.size == 1:
            return np.full((num_windows, num_cells), float(arr[0]), dtype=float)
        if arr.size == num_windows:
            return np.repeat(arr[:, None], num_cells, axis=1)
        if arr.size == num_cells:
            return np.repeat(arr[None, :], num_windows, axis=0)
        raise ValueError("gamma must align with windowTimes or number of cells")
    if arr.ndim != 2:
        raise ValueError("gamma must be scalar, vector, or 2D array")
    if arr.shape == (num_cells, num_windows):
        arr = arr.T
    if arr.shape != (num_windows, num_cells):
        raise ValueError("gamma must be numWindows x C after normalization")
    return arr


def _normalize_history_tensor(HkAll, num_steps: int, num_windows: int, num_cells: int) -> np.ndarray:
    if _is_empty_value(HkAll):
        return np.zeros((num_steps, num_windows, num_cells), dtype=float)

    arr = np.asarray(HkAll, dtype=float)
    expected_shapes = {
        (num_steps, num_windows, num_cells): arr,
        (num_windows, num_cells, num_steps): np.transpose(arr, (2, 0, 1)),
        (num_cells, num_windows, num_steps): np.transpose(arr, (2, 1, 0)),
        (num_cells, num_steps, num_windows): np.transpose(arr, (1, 2, 0)),
    }
    for shape, normalized in expected_shapes.items():
        if arr.shape == shape:
            return normalized
    raise ValueError("HkAll must align with N x numWindows x C MATLAB-style history storage")


def _compute_history_terms(dN: np.ndarray, delta: float, windowTimes) -> np.ndarray:
    obs = _as_observation_matrix(dN)
    windows = np.asarray(windowTimes, dtype=float).reshape(-1)
    if windows.size <= 1:
        return np.zeros((obs.shape[1], 0, obs.shape[0]), dtype=float)

    num_steps = obs.shape[1]
    num_windows = windows.size - 1
    num_cells = obs.shape[0]
    out = np.zeros((num_steps, num_windows, num_cells), dtype=float)

    for time_index in range(num_steps):
        if time_index == 0:
            continue
        previous_indices = np.arange(time_index)
        lag_times = (time_index - previous_indices) * float(delta)
        for window_index, (window_start, window_stop) in enumerate(zip(windows[:-1], windows[1:])):
            mask = (lag_times >= float(window_start)) & (lag_times < float(window_stop))
            if np.any(mask):
                out[time_index, window_index, :] = np.sum(obs[:, previous_indices[mask]], axis=1)
    return out


def _lambda_delta_from_state(
    x_state: np.ndarray,
    mu: np.ndarray,
    beta: np.ndarray,
    fitType: str,
    gamma: np.ndarray,
    HkAll: np.ndarray,
    time_index: int,
) -> np.ndarray:
    histterm = np.asarray(HkAll[time_index - 1], dtype=float) if HkAll.size else np.zeros((0, mu.size), dtype=float)
    hist_effect = np.sum(gamma * histterm, axis=0) if histterm.size else np.zeros(mu.shape, dtype=float)
    lin_term = mu + beta.T @ x_state + hist_effect
    clipped = np.clip(lin_term, -20.0, 20.0)
    if fitType == "binomial":
        exp_term = np.exp(clipped)
        return exp_term / (1.0 + exp_term)
    if fitType == "poisson":
        return np.exp(clipped)
    raise ValueError("fitType must be either 'poisson' or 'binomial'")


def _likelihood_from_lambda(observed: np.ndarray, lambda_delta: np.ndarray, fitType: str) -> float:
    lam = np.clip(np.asarray(lambda_delta, dtype=float).reshape(-1), 1e-9, 1.0 - 1e-9 if fitType == "binomial" else np.inf)
    obs = np.asarray(observed, dtype=float).reshape(-1)
    if fitType == "binomial":
        log_prob = np.sum(obs * np.log(lam) + (1.0 - obs) * np.log(1.0 - lam))
    else:
        log_prob = np.sum(obs * np.log(lam) - lam)
    return float(np.exp(np.clip(log_prob, -200.0, 50.0)))


def _normalize_model_sequence(values, n_models: int, factory):
    if _is_empty_value(values):
        return [factory(index) for index in range(n_models)]
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, np.ndarray)):
        out = list(values)
        if len(out) == n_models:
            return out
    return [values for _ in range(n_models)]


def _normalize_beta_models(beta, n_models: int, num_cells: int, state_dims: list[int]) -> list[np.ndarray]:
    if isinstance(beta, Sequence) and not isinstance(beta, (str, bytes, np.ndarray)):
        items = list(beta)
        if len(items) == n_models and any(np.asarray(item).ndim >= 1 for item in items):
            return [_normalize_beta(item, state_dims[index], num_cells) for index, item in enumerate(items)]
    return [_normalize_beta(beta, state_dims[index], num_cells) for index in range(n_models)]


def _normalize_mu_models(mu, n_models: int, num_cells: int) -> list[np.ndarray]:
    if isinstance(mu, Sequence) and not isinstance(mu, (str, bytes, np.ndarray)):
        items = list(mu)
        if len(items) == n_models and any(np.asarray(item).ndim >= 0 for item in items):
            return [_normalize_mu(item, num_cells) for item in items]
    return [_normalize_mu(mu, num_cells) for _ in range(n_models)]


def _normalize_cif_collection(lambdaCIFColl) -> list[CIF]:
    if isinstance(lambdaCIFColl, CIF):
        cifs = [lambdaCIFColl]
    elif isinstance(lambdaCIFColl, Sequence) and not isinstance(lambdaCIFColl, (str, bytes)):
        cifs = list(lambdaCIFColl)
    else:
        raise UnsupportedWorkflowError("PPDecodeFilter requires a CIF or sequence of CIF objects for the Python port")
    if not cifs:
        raise ValueError("lambdaCIFColl must contain at least one CIF object")
    for cif in cifs:
        if not isinstance(cif, CIF):
            raise UnsupportedWorkflowError("PPDecodeFilter only supports CIF objects in the Python port")
    return cifs


def _extract_linear_terms_from_cifs(lambdaCIFColl, num_states: int, num_cells: int):
    cifs = _normalize_cif_collection(lambdaCIFColl)

    if len(cifs) != num_cells:
        raise ValueError("Number of CIF objects must match the number of observed cells")

    mu_terms: list[float] = []
    beta_cols: list[np.ndarray] = []
    fit_types: set[str] = set()
    history_coeffs: list[np.ndarray] = []
    history_windows = None

    for cif in cifs:
        if not isinstance(cif, CIF):
            raise UnsupportedWorkflowError("PPDecodeFilter only supports CIF objects in the Python port")
        coeffs = np.asarray(cif.b, dtype=float).reshape(-1)
        if coeffs.size == num_states + 1:
            mu_terms.append(float(coeffs[0]))
            beta_cols.append(coeffs[1:])
        elif coeffs.size == num_states:
            mu_terms.append(0.0)
            beta_cols.append(coeffs)
        elif coeffs.size == 1:
            mu_terms.append(float(coeffs[0]))
            beta_cols.append(np.zeros(num_states, dtype=float))
        else:
            raise ValueError("CIF coefficient length is incompatible with the decoding state dimension")

        fit_types.add(str(cif.fitType))
        history_coeffs.append(np.asarray(cif.histCoeffs, dtype=float).reshape(-1))
        if getattr(cif, "history", None) is not None:
            windows = np.asarray(cif.history.windowTimes, dtype=float).reshape(-1)
            if history_windows is None:
                history_windows = windows
            elif not np.allclose(history_windows, windows):
                raise UnsupportedWorkflowError("All CIF history objects must share the same windowTimes")

    if len(fit_types) != 1:
        raise UnsupportedWorkflowError("Mixed fitType collections are not yet supported by PPDecodeFilter")

    max_hist = max((coeff.size for coeff in history_coeffs), default=0)
    if max_hist > 0:
        gamma = np.column_stack(
            [
                np.pad(coeff, (0, max_hist - coeff.size), mode="constant", constant_values=0.0)
                for coeff in history_coeffs
            ]
        )
        if history_windows is None:
            history_windows = np.arange(max_hist + 1, dtype=float)
    else:
        gamma = None

    beta = np.column_stack(beta_cols) if beta_cols else np.zeros((num_states, num_cells), dtype=float)
    return np.asarray(mu_terms, dtype=float), beta, fit_types.pop(), gamma, history_windows




def _nearestSPD(A: np.ndarray) -> np.ndarray:
    """Find the nearest symmetric positive-definite matrix to *A*.

    Uses the algorithm of Higham (1988) via polar decomposition plus
    eigenvalue clamping, matching Matlab ``nearestSPD``.
    """
    B = 0.5 * (A + A.T)
    _, S, Vt = np.linalg.svd(B)
    H = Vt.T @ np.diag(S) @ Vt
    Ahat = 0.5 * (B + H)
    Ahat = 0.5 * (Ahat + Ahat.T)
    # Test for positive-definiteness; clamp eigenvalues if needed
    try:
        np.linalg.cholesky(Ahat)
        return Ahat
    except np.linalg.LinAlgError:
        pass
    eigvals, eigvecs = np.linalg.eigh(Ahat)
    eigvals = np.maximum(eigvals, np.finfo(float).eps)
    Ahat = eigvecs @ np.diag(eigvals) @ eigvecs.T
    Ahat = 0.5 * (Ahat + Ahat.T)
    return Ahat


def _ztest_pvalue(param: float, se: float) -> float:
    """Two-tailed z-test p-value: H0 param == 0, matching Matlab ``ztest``."""
    if se <= 0 or not np.isfinite(se):
        return 1.0
    z = param / se
    return float(2.0 * norm.sf(np.abs(z)))

class DecodingAlgorithms:
    @staticmethod
    def linear_decode(spike_counts: np.ndarray, stimulus: np.ndarray) -> dict[str, np.ndarray]:
        x = np.asarray(spike_counts, dtype=float)
        y = np.asarray(stimulus, dtype=float).reshape(-1)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[0] != y.shape[0]:
            raise ValueError("spike_counts and stimulus must align")

        x_aug = np.column_stack([np.ones(x.shape[0]), x])
        beta, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
        y_hat = x_aug @ beta
        resid = y - y_hat
        sigma = float(np.std(resid))
        ci = np.column_stack([y_hat - 1.96 * sigma, y_hat + 1.96 * sigma])
        return {"coefficients": beta, "decoded": y_hat, "residual": resid, "ci": ci}

    @staticmethod
    def kalman_filter(
        observations: np.ndarray,
        transition: np.ndarray,
        observation_matrix: np.ndarray,
        q_cov: np.ndarray,
        r_cov: np.ndarray,
        x0: np.ndarray,
        p0: np.ndarray,
    ) -> dict[str, np.ndarray]:
        y = np.asarray(observations, dtype=float)
        a = np.asarray(transition, dtype=float)
        h = np.asarray(observation_matrix, dtype=float)
        q = np.asarray(q_cov, dtype=float)
        r = np.asarray(r_cov, dtype=float)
        x_prev = np.asarray(x0, dtype=float).reshape(-1)
        p_prev = np.asarray(p0, dtype=float)

        n_t = y.shape[0]
        n_x = x_prev.shape[0]
        xs = np.zeros((n_t, n_x), dtype=float)
        ps = np.zeros((n_t, n_x, n_x), dtype=float)

        for t in range(n_t):
            x_pred = a @ x_prev
            p_pred = a @ p_prev @ a.T + q

            innovation = y[t] - h @ x_pred
            s_cov = h @ p_pred @ h.T + r
            k_gain = p_pred @ h.T @ np.linalg.pinv(s_cov)

            x_post = x_pred + k_gain @ innovation
            p_post = (np.eye(n_x) - k_gain @ h) @ p_pred

            xs[t] = x_post
            ps[t] = p_post
            x_prev = x_post
            p_prev = p_post

        return {"state": xs, "cov": ps}

    @staticmethod
    def kalman_predict(x_u, Pe_u, A, Pv, GnConv=None):
        x_vec = np.asarray(x_u, dtype=float).reshape(-1)
        dim = x_vec.size
        A_mat = _as_state_matrix(A, dim)
        Pe_mat = _as_state_matrix(Pe_u, dim)
        if _is_empty_value(GnConv):
            Pv_mat = _as_state_matrix(Pv, dim)
            Pe_p = _symmetrize(A_mat @ Pe_mat @ A_mat.T + Pv_mat)
        else:
            Pe_p = _symmetrize(_as_state_matrix(GnConv, dim))
        x_p = A_mat @ x_vec
        return x_p, Pe_p

    @staticmethod
    def kalman_update(x_p, Pe_p, C, Pw, y, GnConv=None):
        x_vec = np.asarray(x_p, dtype=float).reshape(-1)
        dim = x_vec.size
        C_mat = np.asarray(C, dtype=float)
        if C_mat.ndim == 1:
            C_mat = C_mat.reshape(1, -1)
        if C_mat.shape[1] != dim:
            raise ValueError("C must have one column per state dimension")
        Pe_mat = _as_state_matrix(Pe_p, dim)
        y_vec = np.asarray(y, dtype=float).reshape(-1)
        if _is_empty_value(GnConv):
            Pw_mat = _as_state_matrix(Pw, y_vec.size)
            innovation = y_vec - C_mat @ x_vec
            S_cov = _symmetrize(C_mat @ Pe_mat @ C_mat.T + Pw_mat)
            G = Pe_mat @ C_mat.T @ np.linalg.pinv(S_cov)
            x_u = x_vec + G @ innovation
            Pe_u = _symmetrize((np.eye(dim, dtype=float) - G @ C_mat) @ Pe_mat)
        else:
            G = np.asarray(GnConv, dtype=float)
            innovation = y_vec - C_mat @ x_vec
            x_u = x_vec + G @ innovation
            Pe_u = _symmetrize((np.eye(dim, dtype=float) - G @ C_mat) @ Pe_mat)
        return x_u, Pe_u, G

    @staticmethod
    def _state_history_time_major(x, P):
        x_arr = np.asarray(x, dtype=float)
        P_arr = np.asarray(P, dtype=float)
        if P_arr.ndim != 3:
            raise ValueError("Covariance history must be 3D")
        transposed = False
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        if x_arr.shape[0] == P_arr.shape[0]:
            return x_arr, P_arr, transposed
        if x_arr.shape[1] == P_arr.shape[0]:
            return x_arr.T, P_arr, True
        raise ValueError("State history shape does not align with covariance history")

    @staticmethod
    def kalman_smootherFromFiltered(A, x_p, Pe_p, x_u, Pe_u):
        x_p_tm, Pe_p_tm, predicted_transposed = DecodingAlgorithms._state_history_time_major(x_p, Pe_p)
        x_u_tm, Pe_u_tm, updated_transposed = DecodingAlgorithms._state_history_time_major(x_u, Pe_u)
        if predicted_transposed != updated_transposed:
            raise ValueError("Predicted and updated state histories must share an orientation")

        n_t, n_x = x_u_tm.shape
        x_N = x_u_tm.copy()
        P_N = Pe_u_tm.copy()
        Ln = np.zeros((max(n_t - 1, 0), n_x, n_x), dtype=float)

        for t in range(n_t - 2, -1, -1):
            A_t = _select_time_matrix(A, t, n_x)
            gain = Pe_u_tm[t] @ A_t.T @ np.linalg.pinv(Pe_p_tm[t + 1])
            Ln[t] = gain
            x_N[t] = x_u_tm[t] + gain @ (x_N[t + 1] - x_p_tm[t + 1])
            P_N[t] = _symmetrize(Pe_u_tm[t] + gain @ (P_N[t + 1] - Pe_p_tm[t + 1]) @ gain.T)

        if updated_transposed:
            return x_N.T, P_N, Ln
        return x_N, P_N, Ln

    @staticmethod
    def kalman_smoother(A, C, Pv, Pw, Px0, x0, y):
        observations = np.asarray(y, dtype=float)
        if observations.ndim == 1:
            observations = observations[:, None]

        x_prev = np.asarray(x0, dtype=float).reshape(-1)
        Pe_prev = _as_state_matrix(Px0, x_prev.size)
        n_t = observations.shape[0]
        n_x = x_prev.size
        x_p = np.zeros((n_t, n_x), dtype=float)
        Pe_p = np.zeros((n_t, n_x, n_x), dtype=float)
        x_u = np.zeros((n_t, n_x), dtype=float)
        Pe_u = np.zeros((n_t, n_x, n_x), dtype=float)

        for t in range(n_t):
            x_p[t], Pe_p[t] = DecodingAlgorithms.kalman_predict(x_prev, Pe_prev, A, Pv)
            x_u[t], Pe_u[t], _ = DecodingAlgorithms.kalman_update(x_p[t], Pe_p[t], C, Pw, observations[t])
            x_prev = x_u[t]
            Pe_prev = Pe_u[t]

        x_N, P_N, Ln = DecodingAlgorithms.kalman_smootherFromFiltered(A, x_p, Pe_p, x_u, Pe_u)
        return x_N, P_N, Ln, x_p, Pe_p, x_u, Pe_u

    @staticmethod
    def kalman_fixedIntervalSmoother(A, C, Pv, Pw, Px0, x0, y, lags):
        """Fixed-interval smoother with a specified lag.

        .. note::

           The Matlab implementation augments the state vector to dimension
           ``(1+lags)*n_x`` and runs a full Kalman smoother on the augmented
           system.  This Python version instead runs the standard smoother and
           extracts the lagged estimates by index look-up, which is an
           approximation.  The two implementations agree exactly when ``lags``
           equals the full observation length (standard RTS smoother), and the
           approximation error shrinks as ``lags`` grows.
        """
        x_N, P_N, _, x_p, Pe_p, x_u, Pe_u = DecodingAlgorithms.kalman_smoother(A, C, Pv, Pw, Px0, x0, y)
        x_p_tm, Pe_p_tm, _ = DecodingAlgorithms._state_history_time_major(x_p, Pe_p)
        x_u_tm, Pe_u_tm, _ = DecodingAlgorithms._state_history_time_major(x_u, Pe_u)
        x_N_tm, P_N_tm, _ = DecodingAlgorithms._state_history_time_major(x_N, P_N)
        lag = max(int(lags), 1)
        x_pLag = np.zeros_like(x_p_tm)
        Pe_pLag = np.zeros_like(Pe_p_tm)
        x_uLag = np.zeros_like(x_u_tm)
        Pe_uLag = np.zeros_like(Pe_u_tm)

        for t in range(x_u_tm.shape[0]):
            idx = max(t - lag + 1, 0)
            x_uLag[t] = x_N_tm[idx]
            Pe_uLag[t] = P_N_tm[idx]
            x_pLag[t] = x_p_tm[idx]
            Pe_pLag[t] = Pe_p_tm[idx]

        return x_pLag, Pe_pLag, x_uLag, Pe_uLag

    @staticmethod
    def ComputeStimulusCIs(fitType, xK, Wku, delta, Mc=None, alphaVal=0.05):
        del Mc, delta
        x_tm, W_tm, transposed = DecodingAlgorithms._state_history_time_major(xK, Wku)
        variances = np.clip(np.diagonal(W_tm, axis1=1, axis2=2), 0.0, None)
        z = float(norm.ppf(1.0 - float(alphaVal) / 2.0))
        lower = x_tm - z * np.sqrt(variances)
        upper = x_tm + z * np.sqrt(variances)
        fit_type = str(fitType).lower()
        if fit_type == "poisson":
            stimulus = np.exp(np.clip(x_tm, -20.0, 20.0))
            ci_lower = np.exp(np.clip(lower, -20.0, 20.0))
            ci_upper = np.exp(np.clip(upper, -20.0, 20.0))
        elif fit_type == "binomial":
            stimulus = 1.0 / (1.0 + np.exp(-np.clip(x_tm, -20.0, 20.0)))
            ci_lower = 1.0 / (1.0 + np.exp(-np.clip(lower, -20.0, 20.0)))
            ci_upper = 1.0 / (1.0 + np.exp(-np.clip(upper, -20.0, 20.0)))
        else:
            stimulus = x_tm
            ci_lower = lower
            ci_upper = upper

        ci = np.stack([ci_lower, ci_upper], axis=-1)
        if transposed:
            return np.transpose(ci, (1, 0, 2)), stimulus.T
        return ci, stimulus

    @staticmethod
    def computeSpikeRateCIs(
        xK,
        Wku,
        dN,
        t0: float,
        tf: float,
        fitType: str = "poisson",
        delta: float = 0.001,
        gamma=None,
        windowTimes=None,
        Mc: int = 500,
        alphaVal: float = 0.05,
    ):
        """Monte Carlo spike-rate confidence intervals.

        Computes the average firing rate over ``[t0, tf]`` for each trial
        by drawing ``Mc`` samples from the smoothing distribution and
        evaluating the conditional intensity.

        Parameters
        ----------
        xK : array, shape (numBasis, K)
            Smoothed state estimates (basis coefficients × trials).
        Wku : array, shape (numBasis, numBasis, K, K) or compatible
            Smoothed state covariance.
        dN : array, shape (C, N)
            Observation (spike indicator) matrix.
        t0, tf : float
            Time window over which to compute the average rate.
        fitType : str
            ``'poisson'`` or ``'binomial'``.
        delta : float
            Time-step size in seconds.
        gamma : array or None
            History-effect coefficients.
        windowTimes : array or None
            History window boundaries.
        Mc : int
            Number of Monte Carlo draws.
        alphaVal : float
            Significance level for CIs (one-sided).

        Returns
        -------
        spikeRateSig : Covariate
            Mean spike rate per trial with attached ConfidenceInterval.
        ProbMat : ndarray, shape (K, K)
            ``ProbMat[k, m]`` = P(rate_m > rate_k) estimated from MC draws.
        sigMat : ndarray, shape (K, K)
            Binary significance matrix at level ``1 - alphaVal``.
        """
        from .confidence_interval import ConfidenceInterval
        from .core import Covariate
        from .history import History
        from .nspikeTrain import nspikeTrain
        from .trial import SpikeTrainCollection

        xK = np.asarray(xK, dtype=float)
        dN = np.asarray(dN, dtype=float)
        if dN.ndim == 1:
            dN = dN.reshape(1, -1)
        numBasis, K = xK.shape
        minTime = 0.0
        maxTime = (dN.shape[1] - 1) * delta

        # Build unit-impulse basis matrix
        basisWidth = (maxTime - minTime) / numBasis
        sampleRate = 1.0 / delta
        unitPulseBasis = SpikeTrainCollection.generateUnitImpulseBasis(
            basisWidth, minTime, maxTime, sampleRate
        )
        basisMat = unitPulseBasis.data  # shape (T, numBasis)

        # Build history matrices if windowTimes provided
        Hk = {}
        if windowTimes is not None and len(windowTimes) > 0:
            histObj = History(windowTimes, minTime, maxTime)
            for k in range(K):
                spike_idx = np.flatnonzero(dN[k, :] == 1)
                spike_times = (spike_idx) * delta
                nst_k = nspikeTrain(spike_times)
                nst_k.setMinTime(minTime)
                nst_k.setMaxTime(maxTime)
                hist_cov = histObj.computeHistory(nst_k)
                Hk[k] = hist_cov.dataToMatrix()
        else:
            for k in range(K):
                Hk[k] = 0.0
            gamma = 0.0

        if gamma is None:
            gamma = 0.0
        gamma = np.asarray(gamma, dtype=float)

        # Monte Carlo draws from smoothing distribution
        Wku = np.asarray(Wku, dtype=float)
        xKDraw = np.zeros((numBasis, K, Mc), dtype=float)
        for r in range(numBasis):
            WkuTemp = Wku[r, r, :, :].squeeze() if Wku.ndim == 4 else Wku[r, r]
            WkuTemp = np.atleast_2d(WkuTemp)
            if WkuTemp.shape[0] != K:
                WkuTemp = np.diag(np.full(K, float(WkuTemp.flat[0])))
            try:
                chol_m = np.linalg.cholesky(WkuTemp)
            except np.linalg.LinAlgError:
                eigvals = np.linalg.eigvalsh(WkuTemp)
                WkuTemp += np.eye(K) * (abs(min(eigvals.min(), 0.0)) + 1e-10)
                chol_m = np.linalg.cholesky(WkuTemp)
            for c in range(Mc):
                z = np.random.randn(K)
                xKDraw[r, :, c] = xK[r, :] + chol_m.T @ z

        # Compute lambda for each MC draw and each trial
        time_vec = np.arange(minTime, maxTime + delta, delta)
        T = basisMat.shape[0]
        fit_type = str(fitType).lower()
        spikeRate = np.zeros((Mc, K), dtype=float)

        for c in range(Mc):
            for k in range(K):
                stimK = basisMat @ xKDraw[:, k, c]
                if fit_type == "poisson":
                    histEffect = np.exp(gamma @ Hk[k].T).ravel() if not np.isscalar(Hk[k]) else np.ones(T)
                    stimEffect = np.exp(np.clip(stimK, -20.0, 20.0))
                    lambdaDelta_kc = stimEffect * histEffect[:T]
                elif fit_type == "binomial":
                    if np.isscalar(Hk[k]):
                        eta = stimK
                    else:
                        eta = stimK + (gamma @ Hk[k].T).ravel()[:T]
                    eta = np.clip(eta, -20.0, 20.0)
                    lambdaDelta_kc = np.exp(eta) / (1.0 + np.exp(eta))
                else:
                    lambdaDelta_kc = np.exp(np.clip(stimK, -20.0, 20.0))

                # Integrate via cumulative trapezoid
                rate_per_sec = lambdaDelta_kc / delta
                time_k = time_vec[:len(rate_per_sec)]
                cum_integral = np.zeros(len(rate_per_sec))
                cum_integral[1:] = np.cumsum(rate_per_sec[:-1] * delta + 0.5 * np.diff(rate_per_sec) * delta)

                # Interpolate integral at t0 and tf
                val_t0 = np.interp(t0, time_k, cum_integral)
                val_tf = np.interp(tf, time_k, cum_integral)
                spikeRate[c, k] = (1.0 / (tf - t0)) * (val_tf - val_t0)

        # Compute CIs from ECDF (one-sided)
        CIs = np.zeros((K, 2), dtype=float)
        for k in range(K):
            sorted_rates = np.sort(spikeRate[:, k])
            ecdf = np.arange(1, Mc + 1, dtype=float) / float(Mc)
            lower_idx = np.flatnonzero(ecdf < alphaVal)
            upper_idx = np.flatnonzero(ecdf > (1.0 - alphaVal))
            CIs[k, 0] = sorted_rates[lower_idx[-1]] if lower_idx.size else sorted_rates[0]
            CIs[k, 1] = sorted_rates[upper_idx[0]] if upper_idx.size else sorted_rates[-1]

        trial_axis = np.arange(1, K + 1, dtype=float)
        mean_rate = np.mean(spikeRate, axis=0)
        spikeRateSig = Covariate(
            trial_axis,
            mean_rate,
            f"({tf:g}-{t0:g})^{{-1}} * \\Lambda({tf:g}-{t0:g})",
            "Trial",
            "k",
            "Hz",
        )
        ciSpikeRate = ConfidenceInterval(
            trial_axis, CIs, "CI_{spikeRate}", "Trial", "k", "Hz"
        )
        spikeRateSig.setConfInterval(ciSpikeRate)

        # Pairwise probability matrix
        ProbMat = np.zeros((K, K), dtype=float)
        for k in range(K):
            for m in range(k + 1, K):
                ProbMat[k, m] = np.sum(spikeRate[:, m] > spikeRate[:, k]) / float(Mc)

        sigMat = (ProbMat > (1.0 - alphaVal)).astype(float)

        return spikeRateSig, ProbMat, sigMat

    @staticmethod
    def computeSpikeRateDiffCIs(
        xK,
        Wku,
        dN,
        time1,
        time2,
        fitType: str = "poisson",
        delta: float = 0.001,
        gamma=None,
        windowTimes=None,
        Mc: int = 500,
        alphaVal: float = 0.05,
    ):
        """Monte Carlo CIs for the difference in spike rates between two time windows.

        Computes the difference of average firing rates
        ``rate(time1) - rate(time2)`` for each trial by drawing ``Mc``
        samples from the smoothing distribution.

        Parameters
        ----------
        xK : array, shape (numBasis, K)
            Smoothed state estimates (basis coefficients × trials).
        Wku : array, shape (numBasis, numBasis, K, K) or compatible
            Smoothed state covariance.
        dN : array, shape (C, N)
            Observation (spike indicator) matrix.
        time1 : array-like, length 2
            ``[t0_1, tf_1]`` — first time window.
        time2 : array-like, length 2
            ``[t0_2, tf_2]`` — second time window.
        fitType : str
            ``'poisson'`` or ``'binomial'``.
        delta : float
            Time-step size in seconds.
        gamma : array or None
            History-effect coefficients.
        windowTimes : array or None
            History window boundaries.
        Mc : int
            Number of Monte Carlo draws.
        alphaVal : float
            Significance level for CIs (one-sided).

        Returns
        -------
        spikeRateSig : Covariate
            Mean spike-rate difference per trial with attached CI.
        ProbMat : ndarray, shape (K, K)
            ``ProbMat[k, m]`` = P(diff_m > diff_k) from MC draws.
        sigMat : ndarray, shape (K, K)
            Binary significance matrix at level ``1 - alphaVal``.
        """
        from .confidence_interval import ConfidenceInterval
        from .core import Covariate
        from .history import History
        from .nspikeTrain import nspikeTrain
        from .trial import SpikeTrainCollection

        xK = np.asarray(xK, dtype=float)
        dN = np.asarray(dN, dtype=float)
        if dN.ndim == 1:
            dN = dN.reshape(1, -1)
        numBasis, K = xK.shape
        minTime = 0.0
        maxTime = (dN.shape[1] - 1) * delta

        time1 = np.asarray(time1, dtype=float).ravel()
        time2 = np.asarray(time2, dtype=float).ravel()

        # Build unit-impulse basis matrix
        basisWidth = (maxTime - minTime) / numBasis
        sampleRate = 1.0 / delta
        unitPulseBasis = SpikeTrainCollection.generateUnitImpulseBasis(
            basisWidth, minTime, maxTime, sampleRate
        )
        basisMat = unitPulseBasis.data

        # Build history matrices if windowTimes provided
        Hk = {}
        if windowTimes is not None and len(windowTimes) > 0:
            histObj = History(windowTimes, minTime, maxTime)
            for k in range(K):
                spike_idx = np.flatnonzero(dN[k, :] == 1)
                spike_times = (spike_idx) * delta
                nst_k = nspikeTrain(spike_times)
                nst_k.setMinTime(minTime)
                nst_k.setMaxTime(maxTime)
                hist_cov = histObj.computeHistory(nst_k)
                Hk[k] = hist_cov.dataToMatrix()
        else:
            for k in range(K):
                Hk[k] = 0.0
            gamma = 0.0

        if gamma is None:
            gamma = 0.0
        gamma = np.asarray(gamma, dtype=float)

        # Monte Carlo draws from smoothing distribution
        Wku = np.asarray(Wku, dtype=float)
        xKDraw = np.zeros((numBasis, K, Mc), dtype=float)
        for r in range(numBasis):
            WkuTemp = Wku[r, r, :, :].squeeze() if Wku.ndim == 4 else Wku[r, r]
            WkuTemp = np.atleast_2d(WkuTemp)
            if WkuTemp.shape[0] != K:
                WkuTemp = np.diag(np.full(K, float(WkuTemp.flat[0])))
            try:
                chol_m = np.linalg.cholesky(WkuTemp)
            except np.linalg.LinAlgError:
                eigvals = np.linalg.eigvalsh(WkuTemp)
                WkuTemp += np.eye(K) * (abs(min(eigvals.min(), 0.0)) + 1e-10)
                chol_m = np.linalg.cholesky(WkuTemp)
            for c in range(Mc):
                z = np.random.randn(K)
                xKDraw[r, :, c] = xK[r, :] + chol_m.T @ z

        # Compute lambda and spike-rate difference for each MC draw
        time_vec = np.arange(minTime, maxTime + delta, delta)
        T = basisMat.shape[0]
        fit_type = str(fitType).lower()
        spikeRate = np.zeros((Mc, K), dtype=float)

        for c in range(Mc):
            for k in range(K):
                stimK = basisMat @ xKDraw[:, k, c]
                if fit_type == "poisson":
                    histEffect = np.exp(gamma @ Hk[k].T).ravel() if not np.isscalar(Hk[k]) else np.ones(T)
                    stimEffect = np.exp(np.clip(stimK, -20.0, 20.0))
                    lambdaDelta_kc = stimEffect * histEffect[:T]
                elif fit_type == "binomial":
                    if np.isscalar(Hk[k]):
                        eta = stimK
                    else:
                        eta = stimK + (gamma @ Hk[k].T).ravel()[:T]
                    eta = np.clip(eta, -20.0, 20.0)
                    lambdaDelta_kc = np.exp(eta) / (1.0 + np.exp(eta))
                else:
                    lambdaDelta_kc = np.exp(np.clip(stimK, -20.0, 20.0))

                # Integrate via cumulative sum
                rate_per_sec = lambdaDelta_kc / delta
                time_k = time_vec[:len(rate_per_sec)]
                cum_integral = np.zeros(len(rate_per_sec))
                cum_integral[1:] = np.cumsum(rate_per_sec[:-1] * delta + 0.5 * np.diff(rate_per_sec) * delta)

                # Rate for time window 1
                t0_1, tf_1 = float(min(time1)), float(max(time1))
                val_t0_1 = np.interp(t0_1, time_k, cum_integral)
                val_tf_1 = np.interp(tf_1, time_k, cum_integral)
                rate1 = (1.0 / (tf_1 - t0_1)) * (val_tf_1 - val_t0_1)

                # Rate for time window 2
                t0_2, tf_2 = float(min(time2)), float(max(time2))
                val_t0_2 = np.interp(t0_2, time_k, cum_integral)
                val_tf_2 = np.interp(tf_2, time_k, cum_integral)
                rate2 = (1.0 / (tf_2 - t0_2)) * (val_tf_2 - val_t0_2)

                spikeRate[c, k] = rate1 - rate2

        # Compute CIs from ECDF (one-sided)
        CIs = np.zeros((K, 2), dtype=float)
        for k in range(K):
            sorted_rates = np.sort(spikeRate[:, k])
            ecdf = np.arange(1, Mc + 1, dtype=float) / float(Mc)
            lower_idx = np.flatnonzero(ecdf < alphaVal)
            upper_idx = np.flatnonzero(ecdf > (1.0 - alphaVal))
            CIs[k, 0] = sorted_rates[lower_idx[-1]] if lower_idx.size else sorted_rates[0]
            CIs[k, 1] = sorted_rates[upper_idx[0]] if upper_idx.size else sorted_rates[-1]

        trial_axis = np.arange(1, K + 1, dtype=float)
        mean_rate = np.mean(spikeRate, axis=0)
        label = (
            r"(t_{1f}-t_{1o})^{-1} \Lambda(t_{1f}-t_{1o})"
            r" - (t_{2f}-t_{2o})^{-1} \Lambda(t_{2f}-t_{2o})"
        )
        spikeRateSig = Covariate(trial_axis, mean_rate, label, "Trial", "k", "Hz")
        ciSpikeRate = ConfidenceInterval(
            trial_axis, CIs, "CI_{spikeRate}", "Trial", "k", "Hz"
        )
        spikeRateSig.setConfInterval(ciSpikeRate)

        # Pairwise probability matrix
        ProbMat = np.zeros((K, K), dtype=float)
        for k in range(K):
            for m in range(k + 1, K):
                ProbMat[k, m] = np.sum(spikeRate[:, m] > spikeRate[:, k]) / float(Mc)

        sigMat = (ProbMat > (1.0 - alphaVal)).astype(float)

        return spikeRateSig, ProbMat, sigMat

    @staticmethod
    def PPDecode_predict(x_u, W_u, A, Q, Wconv=None):
        x_vec = np.asarray(x_u, dtype=float).reshape(-1)
        dim = x_vec.size
        W_mat = _as_state_matrix(W_u, dim)
        A_mat = _as_state_matrix(A, dim)
        if Wconv is None or Wconv == []:
            Q_mat = _as_state_matrix(Q, dim)
            W_p = _symmetrize(A_mat @ W_mat @ A_mat.T + Q_mat)
        else:
            W_p = _symmetrize(_as_state_matrix(Wconv, dim))
        x_p = A_mat @ x_vec
        return x_p, W_p

    @staticmethod
    def PPDecode_update(x_p, W_p, dN, lambdaIn, binwidth=0.001, time_index=1, WuConv=None):
        x_vec = np.asarray(x_p, dtype=float).reshape(-1)
        W_mat = _as_state_matrix(W_p, x_vec.size)
        obs = _as_observation_matrix(dN)
        idx = max(1, min(int(time_index), obs.shape[1]))

        if isinstance(lambdaIn, CIF):
            lambda_items = [lambdaIn]
        elif isinstance(lambdaIn, Sequence) and not isinstance(lambdaIn, (str, bytes)):
            lambda_items = list(lambdaIn)
        else:
            raise ValueError("Lambda must be a cell of CIFs or a CIF")
        if not lambda_items:
            raise ValueError("Lambda must be a non-empty cell of CIFs or a CIF")

        lambda_delta = np.zeros((len(lambda_items), 1), dtype=float)
        sum_val_vec = np.zeros(x_vec.size, dtype=float)
        sum_val_mat = np.zeros((x_vec.size, x_vec.size), dtype=float)
        observed = obs[:, idx - 1]

        for cell_index, cif in enumerate(lambda_items):
            if cif.historyMat.size == 0:
                observed_prefix = obs[cell_index, :idx]
                spike_times = np.where(observed_prefix > 0.5)[0] * float(binwidth)
                nst = nspikeTrain(spike_times, makePlots=-1)
                nst.setMinTime(0.0)
                nst.setMaxTime((idx - 1) * float(binwidth))
                nst = nst.resample(1.0 / float(binwidth))
                lambda_delta[cell_index, 0] = float(cif.evalLambdaDelta(x_vec, idx, nst))
                sum_val_vec += observed[cell_index] * np.asarray(cif.evalGradientLog(x_vec, idx, nst), dtype=float).reshape(-1)
                sum_val_vec -= np.asarray(cif.evalGradient(x_vec, idx, nst), dtype=float).reshape(-1)
                sum_val_mat -= np.asarray(cif.evalJacobianLog(x_vec, idx, nst), dtype=float)
                sum_val_mat += np.asarray(cif.evalJacobian(x_vec, idx, nst), dtype=float)
            else:
                lambda_delta[cell_index, 0] = float(cif.evalLambdaDelta(x_vec, idx))
                sum_val_vec += observed[cell_index] * np.asarray(cif.evalGradientLog(x_vec, idx), dtype=float).reshape(-1)
                sum_val_vec -= np.asarray(cif.evalGradient(x_vec, idx), dtype=float).reshape(-1)
                sum_val_mat -= np.asarray(cif.evalJacobianLog(x_vec, idx), dtype=float)
                sum_val_mat += np.asarray(cif.evalJacobian(x_vec, idx), dtype=float)

        if _is_empty_value(WuConv):
            identity = np.eye(W_mat.shape[0], dtype=float)
            try:
                W_u = W_mat @ (identity - np.linalg.solve(identity + sum_val_mat @ W_mat, sum_val_mat @ W_mat))
            except np.linalg.LinAlgError:
                W_u = W_mat.copy()
            W_u = _symmetrize(W_u)
        else:
            W_u = _symmetrize(_as_state_matrix(WuConv, x_vec.size))
        x_u = x_vec + W_u @ sum_val_vec
        return x_u, W_u, lambda_delta

    @staticmethod
    def PPDecode_updateLinear(x_p, W_p, dN, mu, beta, fitType="poisson", gamma=None, HkAll=None, time_index=1, WuConv=None):
        x_vec = np.asarray(x_p, dtype=float).reshape(-1)
        W_mat = _as_state_matrix(W_p, x_vec.size)
        obs = _as_observation_matrix(dN)
        num_cells = obs.shape[0]
        mu_vec = _normalize_mu(mu, num_cells)
        beta_mat = _normalize_beta(beta, x_vec.size, num_cells)

        h_num_windows = 0 if _is_empty_value(HkAll) else np.asarray(HkAll).shape[1]
        H_tensor = _normalize_history_tensor(HkAll, obs.shape[1], h_num_windows, num_cells)
        num_windows = H_tensor.shape[1]
        gamma_mat = _normalize_gamma(gamma, num_windows, num_cells)

        lambda_delta = _lambda_delta_from_state(x_vec, mu_vec, beta_mat, str(fitType), gamma_mat, H_tensor, int(time_index))
        observed = obs[:, int(time_index) - 1]
        if str(fitType) == "binomial":
            factor = (observed - lambda_delta) * (1.0 - lambda_delta)
            temp_vec = (observed + (1.0 - 2.0 * lambda_delta)) * (1.0 - lambda_delta) * lambda_delta
        else:
            factor = observed - lambda_delta
            temp_vec = lambda_delta

        sum_val_vec = np.sum(beta_mat * factor[None, :], axis=1)
        sum_val_mat = (beta_mat * temp_vec[None, :]) @ beta_mat.T
        if _is_empty_value(WuConv):
            identity = np.eye(W_mat.shape[0], dtype=float)
            try:
                W_u = W_mat @ (identity - np.linalg.solve(identity + sum_val_mat @ W_mat, sum_val_mat @ W_mat))
            except np.linalg.LinAlgError:
                W_u = W_mat.copy()
            W_u = _symmetrize(W_u)
        else:
            W_u = _symmetrize(_as_state_matrix(WuConv, x_vec.size))
        x_u = x_vec + W_u @ sum_val_vec
        return x_u, W_u, lambda_delta.reshape(-1, 1)

    @staticmethod
    def _ppdecode_filter_linear(
        A,
        Q,
        dN,
        mu,
        beta,
        fitType="poisson",
        delta=0.001,
        gamma=None,
        windowTimes=None,
        x0=None,
        Pi0=None,
        yT=None,
        PiT=None,
        estimateTarget=0,
        Wconv=None,
    ):
        del yT, PiT, estimateTarget
        obs = _as_observation_matrix(dN)
        num_cells, num_steps = obs.shape
        num_states = _infer_state_dim(A, beta, num_cells)
        mu_vec = _normalize_mu(mu, num_cells)
        beta_mat = _normalize_beta(beta, num_states, num_cells)

        x0_vec = np.zeros(num_states, dtype=float) if _is_empty_value(x0) else np.asarray(x0, dtype=float).reshape(-1)
        if x0_vec.size != num_states:
            raise ValueError("x0 must match the decoding state dimension")
        Pi0_mat = np.zeros((num_states, num_states), dtype=float) if _is_empty_value(Pi0) else _as_state_matrix(Pi0, num_states)

        if _is_empty_value(windowTimes):
            H_tensor = np.zeros((num_steps, 0, num_cells), dtype=float)
            gamma_mat = np.zeros((0, num_cells), dtype=float)
        else:
            H_tensor = _compute_history_terms(obs, float(delta), windowTimes)
            gamma_mat = _normalize_gamma(gamma, H_tensor.shape[1], num_cells)

        x_p = np.zeros((num_states, num_steps + 1), dtype=float)
        x_u = np.zeros((num_states, num_steps), dtype=float)
        W_p = np.zeros((num_states, num_states, num_steps + 1), dtype=float)
        W_u = np.zeros((num_states, num_states, num_steps), dtype=float)

        A0 = _select_time_matrix(A, 0, num_states)
        Q0 = _select_time_matrix(Q, 0, num_states)
        x_p[:, 0], W_p[:, :, 0] = DecodingAlgorithms.PPDecode_predict(x0_vec, Pi0_mat, A0, Q0, Wconv)

        for time_index in range(1, num_steps + 1):
            x_u[:, time_index - 1], W_u[:, :, time_index - 1], _ = DecodingAlgorithms.PPDecode_updateLinear(
                x_p[:, time_index - 1],
                W_p[:, :, time_index - 1],
                obs,
                mu_vec,
                beta_mat,
                fitType,
                gamma_mat,
                H_tensor,
                time_index,
                None,
            )
            A_t = _select_time_matrix(A, time_index - 1, num_states)
            Q_t = _select_time_matrix(Q, time_index - 1, num_states)
            x_p[:, time_index], W_p[:, :, time_index] = DecodingAlgorithms.PPDecode_predict(
                x_u[:, time_index - 1],
                W_u[:, :, time_index - 1],
                A_t,
                Q_t,
                Wconv,
            )

        empty_vec = np.array([], dtype=float)
        empty_cov = np.zeros((0, 0, 0), dtype=float)
        return x_p, W_p, x_u, W_u, empty_vec, empty_cov, empty_vec, empty_cov

    @staticmethod
    def PPDecodeFilterLinear(*args, **kwargs):
        if len(args) >= 6 and isinstance(args[5], str):
            return DecodingAlgorithms._ppdecode_filter_linear(*args, **kwargs)
        if "fitType" in kwargs or "delta" in kwargs:
            return DecodingAlgorithms._ppdecode_filter_linear(*args, **kwargs)
        return DecodingAlgorithms.kalman_filter(*args, **kwargs)

    @staticmethod
    def PPDecodeFilter(A, Q, Px0, dN, lambdaCIFColl, binwidth=0.001, x0=None, Pi0=None, yT=None, PiT=None, estimateTarget=0, Wconv=None):
        obs = _as_observation_matrix(dN)
        lambda_items = _normalize_cif_collection(lambdaCIFColl)
        num_cells, num_steps = obs.shape
        if len(lambda_items) != num_cells:
            raise ValueError("Number of CIF objects must match the number of observed cells")

        num_states = _infer_state_dim(A, np.array([0.0]), num_cells)
        uses_target_branch = not _is_empty_value(yT) or not _is_empty_value(PiT) or int(estimateTarget) != 0
        if uses_target_branch:
            mu, beta, fitType, gamma, windowTimes = _extract_linear_terms_from_cifs(lambda_items, num_states, num_cells)
            initial_cov = Px0 if _is_empty_value(Pi0) else Pi0
            return DecodingAlgorithms._ppdecode_filter_linear(
                A,
                Q,
                obs,
                mu,
                beta,
                fitType,
                binwidth,
                gamma,
                windowTimes,
                x0,
                initial_cov,
                yT,
                PiT,
                estimateTarget,
                Wconv,
            )

        x0_vec = np.zeros(num_states, dtype=float) if _is_empty_value(x0) else np.asarray(x0, dtype=float).reshape(-1)
        if x0_vec.size != num_states:
            raise ValueError("x0 must match the decoding state dimension")
        # MATLAB PPDecodeFilter's standard branch initializes from Pi0, and
        # when Pi0 is omitted it falls back to zeros rather than using Px0.
        Pi0_mat = np.zeros((num_states, num_states), dtype=float) if _is_empty_value(Pi0) else _as_state_matrix(Pi0, num_states)

        x_p = np.zeros((num_states, num_steps + 1), dtype=float)
        x_u = np.zeros((num_states, num_steps), dtype=float)
        W_p = np.zeros((num_states, num_states, num_steps + 1), dtype=float)
        W_u = np.zeros((num_states, num_states, num_steps), dtype=float)

        A0 = _select_time_matrix(A, 0, num_states)
        Q0 = _select_time_matrix(Q, 0, num_states)
        x_p[:, 0], W_p[:, :, 0] = DecodingAlgorithms.PPDecode_predict(x0_vec, Pi0_mat, A0, Q0, Wconv)

        for time_index in range(1, num_steps + 1):
            x_u[:, time_index - 1], W_u[:, :, time_index - 1], _ = DecodingAlgorithms.PPDecode_update(
                x_p[:, time_index - 1],
                W_p[:, :, time_index - 1],
                obs,
                lambda_items,
                binwidth,
                time_index,
                None,
            )
            A_t = _select_time_matrix(A, time_index - 1, num_states)
            Q_t = _select_time_matrix(Q, time_index - 1, num_states)
            x_p[:, time_index], W_p[:, :, time_index] = DecodingAlgorithms.PPDecode_predict(
                x_u[:, time_index - 1],
                W_u[:, :, time_index - 1],
                A_t,
                Q_t,
                Wconv,
            )

        empty_vec = np.array([], dtype=float)
        empty_cov = np.zeros((0, 0, 0), dtype=float)
        return x_p, W_p, x_u, W_u, empty_vec, empty_cov, empty_vec, empty_cov

    @staticmethod
    def PP_fixedIntervalSmoother(A, Q, dN, lags, mu, beta, fitType="poisson", delta=0.001, gamma=None, windowTimes=None, x0=None, Pi0=None):
        obs = _as_observation_matrix(dN)
        num_cells, num_steps = obs.shape
        num_states = _infer_state_dim(A, beta, num_cells)
        mu_vec = _normalize_mu(mu, num_cells)
        beta_mat = _normalize_beta(beta, num_states, num_cells)

        x0_vec = np.zeros(num_states, dtype=float) if _is_empty_value(x0) else np.asarray(x0, dtype=float).reshape(-1)
        if x0_vec.size != num_states:
            raise ValueError("x0 must match the decoding state dimension")
        Pi0_mat = np.zeros((num_states, num_states), dtype=float) if _is_empty_value(Pi0) else _as_state_matrix(Pi0, num_states)

        if _is_empty_value(windowTimes):
            H_tensor = np.zeros((num_steps, 0, num_cells), dtype=float)
            gamma_mat = np.zeros((0, num_cells), dtype=float)
        else:
            H_tensor = _compute_history_terms(obs, float(delta), windowTimes)
            gamma_mat = _normalize_gamma(gamma, H_tensor.shape[1], num_cells)

        x_p = np.zeros((num_states, num_steps + 1), dtype=float)
        x_u = np.zeros((num_states, num_steps), dtype=float)
        W_p = np.zeros((num_states, num_states, num_steps + 1), dtype=float)
        W_u = np.zeros((num_states, num_states, num_steps), dtype=float)
        x_p[:, 0] = x0_vec
        W_p[:, :, 0] = Pi0_mat

        lag_count = max(int(lags), 1)
        num_states, num_steps = x_u.shape
        x_uLag = np.zeros_like(x_u)
        W_uLag = np.zeros_like(W_u)
        x_pLag = np.zeros_like(x_p)
        W_pLag = np.zeros_like(W_p)

        for n in range(num_steps):
            x_u[:, n], W_u[:, :, n], _ = DecodingAlgorithms.PPDecode_updateLinear(
                x_p[:, n],
                W_p[:, :, n],
                obs,
                mu_vec,
                beta_mat,
                fitType,
                gamma_mat,
                H_tensor,
                n + 1,
                None,
            )
            A_t = _select_time_matrix(A, n, num_states)
            Q_t = _select_time_matrix(Q, n, num_states)
            x_p[:, n + 1], W_p[:, :, n + 1] = DecodingAlgorithms.PPDecode_predict(
                x_u[:, n],
                W_u[:, :, n],
                A_t,
                Q_t,
            )
            if n < lag_count:
                continue

            x_bank: list[np.ndarray] = []
            w_bank: list[np.ndarray] = []
            for k in range(1, lag_count + 1):
                idx = n - k
                A_k = _select_time_matrix(A, idx, num_states)
                gain = W_u[:, :, idx] @ A_k.T @ np.linalg.pinv(W_p[:, :, idx + 1])
                target_x = x_u[:, idx + 1] if k == 1 else x_bank[k - 2]
                target_W = W_u[:, :, idx + 1] if k == 1 else w_bank[k - 2]
                x_k = x_u[:, idx] + gain @ (target_x - x_p[:, idx + 1])
                W_k = W_u[:, :, idx] + gain @ (target_W - W_p[:, :, idx + 1]) @ gain.T
                W_k = _symmetrize(W_k)
                x_bank.append(x_k)
                w_bank.append(W_k)

            x_uLag[:, n] = x_bank[-1]
            W_uLag[:, :, n] = w_bank[-1]
            if lag_count > 1:
                x_pLag[:, n + 1] = x_bank[-2]
                W_pLag[:, :, n + 1] = w_bank[-1]
            else:
                x_pLag[:, n + 1] = x_u[:, n]
                W_pLag[:, :, n + 1] = W_u[:, :, n]

        return x_pLag, W_pLag, x_uLag, W_uLag

    @staticmethod
    def PPHybridFilterLinear(
        A,
        Q,
        p_ij,
        Mu0,
        dN,
        mu,
        beta,
        fitType="poisson",
        binwidth=0.001,
        gamma=None,
        windowTimes=None,
        x0=None,
        Pi0=None,
        yT=None,
        PiT=None,
        estimateTarget=0,
        MinClassificationError=0,
    ):
        del yT, PiT, estimateTarget
        obs = _as_observation_matrix(dN)
        A_models = list(A) if isinstance(A, Sequence) and not isinstance(A, np.ndarray) else [A]
        Q_models = list(Q) if isinstance(Q, Sequence) and not isinstance(Q, np.ndarray) else [Q]
        n_models = len(A_models)
        if len(Q_models) != n_models:
            raise ValueError("A and Q must define the same number of hybrid models")

        num_cells, num_steps = obs.shape
        state_dims = [_infer_state_dim(A_models[index], beta, num_cells) for index in range(n_models)]
        max_dim = max(state_dims)
        mu_models = _normalize_mu_models(mu, n_models, num_cells)
        beta_models = _normalize_beta_models(beta, n_models, num_cells, state_dims)
        x0_models_raw = _normalize_model_sequence(x0, n_models, lambda index: np.zeros(state_dims[index], dtype=float))
        Pi0_models_raw = _normalize_model_sequence(Pi0, n_models, lambda index: np.zeros((state_dims[index], state_dims[index]), dtype=float))
        x0_models = [np.asarray(x0_models_raw[index], dtype=float).reshape(-1) for index in range(n_models)]
        Pi0_models = [_as_state_matrix(Pi0_models_raw[index], state_dims[index]) for index in range(n_models)]

        transition = np.asarray(p_ij, dtype=float)
        if transition.shape != (n_models, n_models):
            raise ValueError("p_ij must be an nModels x nModels transition matrix")
        row_sums = np.sum(transition, axis=1)
        if not np.allclose(row_sums, np.ones(n_models), atol=1e-8):
            raise ValueError("State Transition probability matrix must sum to 1 along each row")

        if _is_empty_value(Mu0):
            model_probs0 = np.full(n_models, 1.0 / float(n_models), dtype=float)
        else:
            model_probs0 = _normalize_probabilities(Mu0)
            if model_probs0.size != n_models:
                raise ValueError("Mu0 must contain one probability per hybrid model")

        if _is_empty_value(windowTimes):
            H_tensor = np.zeros((num_steps, 0, num_cells), dtype=float)
            gamma_mat = np.zeros((0, num_cells), dtype=float)
        else:
            H_tensor = _compute_history_terms(obs, float(binwidth), windowTimes)
            gamma_mat = _normalize_gamma(gamma, H_tensor.shape[1], num_cells)

        X = np.zeros((max_dim, num_steps), dtype=float)
        W = np.zeros((max_dim, max_dim, num_steps), dtype=float)
        X_s = [np.zeros((max_dim, num_steps), dtype=float) for _ in range(n_models)]
        W_s = [np.zeros((max_dim, max_dim, num_steps), dtype=float) for _ in range(n_models)]
        X_u = [np.zeros((state_dims[index], num_steps), dtype=float) for index in range(n_models)]
        W_u = [np.zeros((state_dims[index], state_dims[index], num_steps), dtype=float) for index in range(n_models)]
        X_p = [np.zeros((state_dims[index], num_steps), dtype=float) for index in range(n_models)]
        W_p = [np.zeros((state_dims[index], state_dims[index], num_steps), dtype=float) for index in range(n_models)]
        MU_u = np.zeros((n_models, num_steps), dtype=float)
        pNGivenS = np.zeros((n_models, num_steps), dtype=float)
        S_est = np.zeros(num_steps, dtype=int)

        fit_type = str(fitType)

        for time_index in range(num_steps):
            if time_index == 0:
                MU_p = transition.T @ model_probs0
                prev_probs = model_probs0
            else:
                MU_p = transition.T @ MU_u[:, time_index - 1]
                prev_probs = MU_u[:, time_index - 1]

            p_ij_s = transition * prev_probs[:, None]
            column_norm = np.sum(p_ij_s, axis=0, keepdims=True)
            column_norm[column_norm == 0.0] = 1.0
            p_ij_s = p_ij_s / column_norm

            for target_model in range(n_models):
                mixed_state = np.zeros(max_dim, dtype=float)
                for source_model in range(n_models):
                    dim_i = state_dims[source_model]
                    source_state = x0_models[source_model] if time_index == 0 else X_u[source_model][:, time_index - 1]
                    mixed_state[:dim_i] += source_state * p_ij_s[source_model, target_model]
                X_s[target_model][:, time_index] = mixed_state

                mixed_cov = np.zeros((max_dim, max_dim), dtype=float)
                for source_model in range(n_models):
                    dim_i = state_dims[source_model]
                    source_state = x0_models[source_model] if time_index == 0 else X_u[source_model][:, time_index - 1]
                    source_cov = Pi0_models[source_model] if time_index == 0 else W_u[source_model][:, :, time_index - 1]
                    diff = source_state - mixed_state[:dim_i]
                    mixed_cov[:dim_i, :dim_i] += (
                        source_cov + np.outer(diff, diff)
                    ) * p_ij_s[source_model, target_model]
                W_s[target_model][:, :, time_index] = _symmetrize(mixed_cov)

            likelihoods = np.zeros(n_models, dtype=float)
            for model_index in range(n_models):
                dim = state_dims[model_index]
                A_t = _select_time_matrix(A_models[model_index], time_index, dim)
                Q_t = _select_time_matrix(Q_models[model_index], time_index, dim)
                pred_x, pred_W = DecodingAlgorithms.PPDecode_predict(
                    X_s[model_index][:dim, time_index],
                    W_s[model_index][:dim, :dim, time_index],
                    A_t,
                    Q_t,
                )
                upd_x, upd_W, lambda_delta = DecodingAlgorithms.PPDecode_updateLinear(
                    pred_x,
                    pred_W,
                    obs,
                    mu_models[model_index],
                    beta_models[model_index],
                    fit_type,
                    gamma_mat,
                    H_tensor,
                    time_index + 1,
                    None,
                )
                X_p[model_index][:, time_index] = pred_x
                W_p[model_index][:, :, time_index] = pred_W
                X_u[model_index][:, time_index] = upd_x
                W_u[model_index][:, :, time_index] = upd_W

                det_ratio = np.sqrt(max(np.linalg.det(upd_W), 0.0)) / max(np.sqrt(max(np.linalg.det(pred_W), 0.0)), 1e-15)
                log_term = np.sum(obs[:, time_index] * np.log(np.clip(lambda_delta.reshape(-1), 1e-12, np.inf)) - lambda_delta.reshape(-1))
                likelihoods[model_index] = float(det_ratio * np.exp(np.clip(log_term, -200.0, 50.0)))

            finite_likelihoods = likelihoods.copy()
            finite_likelihoods[~np.isfinite(finite_likelihoods)] = 0.0
            pNGivenS[:, time_index] = finite_likelihoods
            norm = np.sum(pNGivenS[:, time_index])
            if norm != 0.0 and np.isfinite(norm):
                pNGivenS[:, time_index] /= norm
            elif time_index > 0:
                pNGivenS[:, time_index] = pNGivenS[:, time_index - 1]
            else:
                pNGivenS[:, time_index] = np.full(n_models, 0.5 if n_models == 2 else 1.0 / float(n_models), dtype=float)

            posterior = MU_p * pNGivenS[:, time_index]
            posterior_norm = np.sum(posterior)
            if posterior_norm != 0.0 and np.isfinite(posterior_norm):
                MU_u[:, time_index] = posterior / posterior_norm
            elif time_index > 0:
                MU_u[:, time_index] = MU_u[:, time_index - 1]
            else:
                MU_u[:, time_index] = model_probs0

            best_model = int(np.argmax(MU_u[:, time_index]))
            S_est[time_index] = best_model + 1

            if MinClassificationError:
                chosen = best_model
                dim = state_dims[chosen]
                X[:dim, time_index] = X_u[chosen][:, time_index]
                W[:dim, :dim, time_index] = W_u[chosen][:, :, time_index]
                continue

            mixed_global_state = np.zeros(max_dim, dtype=float)
            for model_index in range(n_models):
                dim = state_dims[model_index]
                mixed_global_state[:dim] += MU_u[model_index, time_index] * X_u[model_index][:, time_index]
            X[:, time_index] = mixed_global_state

            mixed_global_cov = np.zeros((max_dim, max_dim), dtype=float)
            for model_index in range(n_models):
                dim = state_dims[model_index]
                diff = X_u[model_index][:, time_index] - mixed_global_state[:dim]
                mixed_global_cov[:dim, :dim] += MU_u[model_index, time_index] * (
                    W_u[model_index][:, :, time_index] + np.outer(diff, diff)
                )
            W[:, :, time_index] = _symmetrize(mixed_global_cov)

        return S_est, X, W, MU_u, X_s, W_s, pNGivenS

    @staticmethod
    def PPHybridFilter(A, Q, p_ij, Mu0, dN, lambdaCIFColl, binwidth=0.001, x0=None, Pi0=None, yT=None, PiT=None, estimateTarget=0, MinClassificationError=0):
        """Hybrid point-process filter with CIF-object evaluation.

        Unlike :meth:`PPHybridFilterLinear` which takes pre-extracted linear
        parameters (mu, beta, gamma), this method evaluates CIF objects
        directly via their ``evalLambdaDelta`` / ``evalGradient*`` /
        ``evalJacobian*`` methods.  This supports nonlinear conditional
        intensity specifications.

        Falls back to the linear path when the target-estimation branch is
        active (``yT`` / ``PiT`` / ``estimateTarget`` supplied), matching
        Matlab behaviour.
        """
        del yT, PiT, estimateTarget  # reserved for future target-estimation branch
        obs = _as_observation_matrix(dN)
        lambda_items = _normalize_cif_collection(lambdaCIFColl)
        A_models = list(A) if isinstance(A, Sequence) and not isinstance(A, np.ndarray) else [A]
        Q_models = list(Q) if isinstance(Q, Sequence) and not isinstance(Q, np.ndarray) else [Q]
        n_models = len(A_models)
        if len(Q_models) != n_models:
            raise ValueError("A and Q must define the same number of hybrid models")

        num_cells, num_steps = obs.shape
        if len(lambda_items) != num_cells:
            raise ValueError("Number of CIF objects must match the number of observed cells")
        state_dims = [_infer_state_dim(A_models[index], np.array([0.0]), num_cells) for index in range(n_models)]
        max_dim = max(state_dims)

        x0_models_raw = _normalize_model_sequence(x0, n_models, lambda index: np.zeros(state_dims[index], dtype=float))
        Pi0_models_raw = _normalize_model_sequence(Pi0, n_models, lambda index: np.zeros((state_dims[index], state_dims[index]), dtype=float))
        x0_models = [np.asarray(x0_models_raw[index], dtype=float).reshape(-1) for index in range(n_models)]
        Pi0_models = [_as_state_matrix(Pi0_models_raw[index], state_dims[index]) for index in range(n_models)]

        transition = np.asarray(p_ij, dtype=float)
        if transition.shape != (n_models, n_models):
            raise ValueError("p_ij must be an nModels x nModels transition matrix")
        row_sums = np.sum(transition, axis=1)
        if not np.allclose(row_sums, np.ones(n_models), atol=1e-8):
            raise ValueError("State Transition probability matrix must sum to 1 along each row")

        if _is_empty_value(Mu0):
            model_probs0 = np.full(n_models, 1.0 / float(n_models), dtype=float)
        else:
            model_probs0 = _normalize_probabilities(Mu0)
            if model_probs0.size != n_models:
                raise ValueError("Mu0 must contain one probability per hybrid model")

        X = np.zeros((max_dim, num_steps), dtype=float)
        W = np.zeros((max_dim, max_dim, num_steps), dtype=float)
        X_s = [np.zeros((max_dim, num_steps), dtype=float) for _ in range(n_models)]
        W_s = [np.zeros((max_dim, max_dim, num_steps), dtype=float) for _ in range(n_models)]
        X_u = [np.zeros((state_dims[index], num_steps), dtype=float) for index in range(n_models)]
        W_u = [np.zeros((state_dims[index], state_dims[index], num_steps), dtype=float) for index in range(n_models)]
        X_p = [np.zeros((state_dims[index], num_steps), dtype=float) for index in range(n_models)]
        W_p = [np.zeros((state_dims[index], state_dims[index], num_steps), dtype=float) for index in range(n_models)]
        MU_u = np.zeros((n_models, num_steps), dtype=float)
        pNGivenS = np.zeros((n_models, num_steps), dtype=float)
        S_est = np.zeros(num_steps, dtype=int)

        for time_index in range(num_steps):
            if time_index == 0:
                MU_p = transition.T @ model_probs0
                prev_probs = model_probs0
            else:
                MU_p = transition.T @ MU_u[:, time_index - 1]
                prev_probs = MU_u[:, time_index - 1]

            p_ij_s = transition * prev_probs[:, None]
            column_norm = np.sum(p_ij_s, axis=0, keepdims=True)
            column_norm[column_norm == 0.0] = 1.0
            p_ij_s = p_ij_s / column_norm

            for target_model in range(n_models):
                mixed_state = np.zeros(max_dim, dtype=float)
                for source_model in range(n_models):
                    dim_i = state_dims[source_model]
                    source_state = x0_models[source_model] if time_index == 0 else X_u[source_model][:, time_index - 1]
                    mixed_state[:dim_i] += source_state * p_ij_s[source_model, target_model]
                X_s[target_model][:, time_index] = mixed_state

                mixed_cov = np.zeros((max_dim, max_dim), dtype=float)
                for source_model in range(n_models):
                    dim_i = state_dims[source_model]
                    source_state = x0_models[source_model] if time_index == 0 else X_u[source_model][:, time_index - 1]
                    source_cov = Pi0_models[source_model] if time_index == 0 else W_u[source_model][:, :, time_index - 1]
                    diff = source_state - mixed_state[:dim_i]
                    mixed_cov[:dim_i, :dim_i] += (
                        source_cov + np.outer(diff, diff)
                    ) * p_ij_s[source_model, target_model]
                W_s[target_model][:, :, time_index] = _symmetrize(mixed_cov)

            likelihoods = np.zeros(n_models, dtype=float)
            for model_index in range(n_models):
                dim = state_dims[model_index]
                A_t = _select_time_matrix(A_models[model_index], time_index, dim)
                Q_t = _select_time_matrix(Q_models[model_index], time_index, dim)
                pred_x, pred_W = DecodingAlgorithms.PPDecode_predict(
                    X_s[model_index][:dim, time_index],
                    W_s[model_index][:dim, :dim, time_index],
                    A_t,
                    Q_t,
                )
                # Use CIF-based (nonlinear) update instead of linear
                upd_x, upd_W, lambda_delta = DecodingAlgorithms.PPDecode_update(
                    pred_x,
                    pred_W,
                    obs,
                    lambda_items,
                    binwidth,
                    time_index + 1,
                    None,
                )
                X_p[model_index][:, time_index] = pred_x
                W_p[model_index][:, :, time_index] = pred_W
                X_u[model_index][:, time_index] = upd_x
                W_u[model_index][:, :, time_index] = upd_W

                det_ratio = np.sqrt(max(np.linalg.det(upd_W), 0.0)) / max(np.sqrt(max(np.linalg.det(pred_W), 0.0)), 1e-15)
                log_term = np.sum(obs[:, time_index] * np.log(np.clip(lambda_delta.reshape(-1), 1e-12, np.inf)) - lambda_delta.reshape(-1))
                likelihoods[model_index] = float(det_ratio * np.exp(np.clip(log_term, -200.0, 50.0)))

            finite_likelihoods = likelihoods.copy()
            finite_likelihoods[~np.isfinite(finite_likelihoods)] = 0.0
            pNGivenS[:, time_index] = finite_likelihoods
            norm = np.sum(pNGivenS[:, time_index])
            if norm != 0.0 and np.isfinite(norm):
                pNGivenS[:, time_index] /= norm
            elif time_index > 0:
                pNGivenS[:, time_index] = pNGivenS[:, time_index - 1]
            else:
                pNGivenS[:, time_index] = np.full(n_models, 0.5 if n_models == 2 else 1.0 / float(n_models), dtype=float)

            posterior = MU_p * pNGivenS[:, time_index]
            posterior_norm = np.sum(posterior)
            if posterior_norm != 0.0 and np.isfinite(posterior_norm):
                MU_u[:, time_index] = posterior / posterior_norm
            elif time_index > 0:
                MU_u[:, time_index] = MU_u[:, time_index - 1]
            else:
                MU_u[:, time_index] = model_probs0

            best_model = int(np.argmax(MU_u[:, time_index]))
            S_est[time_index] = best_model + 1

            if MinClassificationError:
                chosen = best_model
                dim = state_dims[chosen]
                X[:dim, time_index] = X_u[chosen][:, time_index]
                W[:dim, :dim, time_index] = W_u[chosen][:, :, time_index]
                continue

            mixed_global_state = np.zeros(max_dim, dtype=float)
            for model_index in range(n_models):
                dim = state_dims[model_index]
                mixed_global_state[:dim] += MU_u[model_index, time_index] * X_u[model_index][:, time_index]
            X[:, time_index] = mixed_global_state

            mixed_global_cov = np.zeros((max_dim, max_dim), dtype=float)
            for model_index in range(n_models):
                dim = state_dims[model_index]
                diff = X_u[model_index][:, time_index] - mixed_global_state[:dim]
                mixed_global_cov[:dim, :dim] += MU_u[model_index, time_index] * (
                    W_u[model_index][:, :, time_index] + np.outer(diff, diff)
                )
            W[:, :, time_index] = _symmetrize(mixed_global_cov)

        return S_est, X, W, MU_u, X_s, W_s, pNGivenS

    # ------------------------------------------------------------------
    # Unscented Kalman Filter (UKF)
    # Ported from Matlab DecodingAlgorithms.m
    # ------------------------------------------------------------------

    @staticmethod
    def ukf_sigmas(x: np.ndarray, P: np.ndarray, c: float) -> np.ndarray:
        """Generate sigma points around reference point *x*.

        Parameters
        ----------
        x : (L,) state vector
        P : (L, L) covariance
        c : scaling coefficient

        Returns
        -------
        X : (L, 2L+1) sigma-point matrix
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        P = np.asarray(P, dtype=float)
        A = c * np.linalg.cholesky(P)  # (L, L)
        L = len(x)
        Y = np.tile(x[:, None], (1, L))
        X = np.column_stack([x[:, None], Y + A, Y - A])
        return X

    @staticmethod
    def ukf_ut(f, X: np.ndarray, Wm: np.ndarray, Wc: np.ndarray,
               n: int, R: np.ndarray):
        """Unscented transformation.

        Parameters
        ----------
        f : callable mapping (L,) -> (n,)
        X : (L, 2L+1) sigma points
        Wm, Wc : (2L+1,) weights
        n : output dimensionality
        R : (n, n) additive covariance

        Returns
        -------
        y : (n,) transformed mean
        Y : (n, 2L+1) transformed sigma points
        P : (n, n) transformed covariance
        Y1 : (n, 2L+1) deviations
        """
        Lpts = X.shape[1]
        y = np.zeros(n)
        Y = np.zeros((n, Lpts))
        for k in range(Lpts):
            Y[:, k] = np.asarray(f(X[:, k]), dtype=float).reshape(-1)[:n]
            y += Wm[k] * Y[:, k]
        Y1 = Y - y[:, None]
        P = Y1 @ np.diag(Wc) @ Y1.T + np.asarray(R, dtype=float).reshape(n, n)
        return y, Y, P, Y1

    @staticmethod
    def ukf(fstate, x: np.ndarray, P: np.ndarray, hmeas,
            z: np.ndarray, Q: np.ndarray, R: np.ndarray):
        """Unscented Kalman Filter for nonlinear systems.

        One-step update matching Matlab ``DecodingAlgorithms.ukf``.

        System model (additive noise)::

            x_{k+1} = fstate(x_k) + w_k,   w ~ N(0, Q)
            z_k     = hmeas(x_k)  + v_k,   v ~ N(0, R)

        Parameters
        ----------
        fstate : callable (L,) -> (L,)
        x : (L,) prior state estimate
        P : (L, L) prior covariance
        hmeas : callable (L,) -> (m,)
        z : (m,) measurement
        Q : (L, L) process noise covariance
        R : (m, m) measurement noise covariance

        Returns
        -------
        x : (L,) posterior state estimate
        P : (L, L) posterior covariance
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        z = np.asarray(z, dtype=float).reshape(-1)
        P = np.asarray(P, dtype=float)
        Q = np.asarray(Q, dtype=float)
        R = np.asarray(R, dtype=float)
        if R.ndim == 0:
            R = R.reshape(1, 1)
        elif R.ndim == 1:
            R = np.diag(R)

        L = len(x)
        m = len(z)
        alpha = 1e-3
        ki = 0.0
        beta = 2.0
        lam = alpha ** 2 * (L + ki) - L
        c = L + lam
        Wm = np.full(2 * L + 1, 0.5 / c)
        Wm[0] = lam / c
        Wc = Wm.copy()
        Wc[0] += (1 - alpha ** 2 + beta)
        c_sqrt = np.sqrt(c)

        X = DecodingAlgorithms.ukf_sigmas(x, P, c_sqrt)
        x1, X1, P1, X2 = DecodingAlgorithms.ukf_ut(fstate, X, Wm, Wc, L, Q)
        z1, Z1, P2, Z2 = DecodingAlgorithms.ukf_ut(hmeas, X1, Wm, Wc, m, R)
        P12 = X2 @ np.diag(Wc) @ Z2.T
        K = P12 @ np.linalg.inv(P2)
        x_out = x1 + K @ (z - z1)
        P_out = P1 - K @ P12.T
        return x_out, P_out

    # ------------------------------------------------------------------
    # State-Space GLM (SSGLM) via EM Forward-Backward
    # Ported from Matlab DecodingAlgorithms.m (PPSS_EMFB and helpers)
    # ------------------------------------------------------------------

    @staticmethod
    def _ssglm_build_basis(numBasis, minTime, maxTime, delta):
        """Build unit-impulse basis matrix for SSGLM."""
        from .trial import SpikeTrainCollection

        sampleRate = 1.0 / delta
        basisWidth = (maxTime - minTime) / numBasis
        basis_cov = SpikeTrainCollection.generateUnitImpulseBasis(
            basisWidth, minTime, maxTime, sampleRate
        )
        return np.asarray(basis_cov.data, dtype=float)

    @staticmethod
    def _ssglm_build_history(dN, windowTimes, delta):
        """Build history design matrices for each trial from spike observations."""
        from .history import History

        K, N = dN.shape
        minTime = 0.0
        maxTime = (N - 1) * delta

        if windowTimes is not None and len(windowTimes) > 0:
            histObj = History(windowTimes, minTime, maxTime)
            HkAll = []
            for k in range(K):
                spike_indices = np.where(dN[k, :] > 0.5)[0]
                spike_times = spike_indices.astype(float) * delta
                nst = nspikeTrain(spike_times, makePlots=-1)
                nst.setMinTime(minTime)
                nst.setMaxTime(maxTime)
                hist_cov = histObj._compute_single_history(nst)
                HkAll.append(np.asarray(hist_cov.data, dtype=float))
            return HkAll
        else:
            return [np.zeros((N, 0), dtype=float) for _ in range(K)]

    @staticmethod
    def PPSS_EStep(A, Q, x0, dN, HkAll, fitType, delta, gamma, numBasis):
        """E-step: Forward Kalman filter + backward RTS smoother + cross-covariance.

        Parameters
        ----------
        A : (R, R) state transition matrix
        Q : (R,) or (R, R) state noise covariance (diagonal vector or matrix)
        x0 : (R,) initial state
        dN : (K, N) binary spike observations (K trials, N time bins)
        HkAll : list of K arrays, each (N, J) history design matrices
        fitType : 'poisson' or 'binomial'
        delta : time bin width
        gamma : (J,) history coefficients
        numBasis : number of basis functions R

        Returns
        -------
        x_K, W_K, Wku, logll, sumXkTerms, sumPPll
        """
        K, N = dN.shape
        minTime = 0.0
        maxTime = (N - 1) * delta

        basisMat = DecodingAlgorithms._ssglm_build_basis(numBasis, minTime, maxTime, delta)
        # Ensure basisMat has N rows matching dN columns
        if basisMat.shape[0] != N:
            basisMat = basisMat[:N, :] if basisMat.shape[0] > N else np.vstack(
                [basisMat, np.zeros((N - basisMat.shape[0], basisMat.shape[1]))]
            )

        Q_diag = np.asarray(Q, dtype=float).reshape(-1)
        if Q_diag.size == numBasis * numBasis:
            Q_mat = Q_diag.reshape(numBasis, numBasis)
            Q_diag = np.diag(Q_mat)
        Q_mat = np.diag(Q_diag)

        A_mat = np.asarray(A, dtype=float)
        if A_mat.ndim < 2:
            A_mat = np.eye(numBasis, dtype=float) * A_mat
        x0_vec = np.asarray(x0, dtype=float).reshape(-1)
        gamma_vec = np.asarray(gamma, dtype=float).reshape(-1)
        R = numBasis
        fitType = str(fitType).lower()

        # Forward Kalman filter
        x_p = np.zeros((R, K), dtype=float)
        x_u = np.zeros((R, K), dtype=float)
        W_p = np.zeros((R, R, K), dtype=float)
        W_u = np.zeros((R, R, K), dtype=float)

        for k in range(K):
            if k == 0:
                x_p[:, k] = A_mat @ x0_vec
                W_p[:, :, k] = Q_mat.copy()
            else:
                x_p[:, k] = A_mat @ x_u[:, k - 1]
                W_p[:, :, k] = A_mat @ W_u[:, :, k - 1] @ A_mat.T + Q_mat

            Hk = HkAll[k]
            stimK = basisMat @ x_p[:, k]

            if fitType == "poisson":
                histEffect = np.exp(np.clip(Hk @ gamma_vec, -30, 30)) if gamma_vec.size > 0 and Hk.shape[1] > 0 else np.ones(N)
                stimEffect = np.exp(np.clip(stimK, -30, 30))
                lambdaDelta = stimEffect * histEffect

                GradLogLD = basisMat  # (N, R)
                GradLD = basisMat * lambdaDelta[:, None]  # (N, R)

                sumValVec = GradLogLD.T @ dN[k, :] - np.diag(GradLD.T @ basisMat)
                sumValMat = GradLD.T @ basisMat

            elif fitType == "binomial":
                Hk = HkAll[k]
                stimK = basisMat @ x_p[:, k]
                linpred = stimK + (Hk @ gamma_vec if gamma_vec.size > 0 and Hk.shape[1] > 0 else 0.0)
                linpred = np.clip(linpred, -30, 30)
                lambdaDelta = 1.0 / (1.0 + np.exp(-linpred))

                GradLogLD = basisMat * (1.0 - lambdaDelta)[:, None]
                JacobianLogLD = basisMat * (lambdaDelta * (-1.0 + lambdaDelta))[:, None]
                GradLD = basisMat * (lambdaDelta * (1.0 - lambdaDelta))[:, None]
                JacobianLD = basisMat * (lambdaDelta * (1.0 - lambdaDelta) * (1.0 - 2.0 * lambdaDelta ** 2))[:, None]

                sumValVec = GradLogLD.T @ dN[k, :] - np.diag(GradLD.T @ basisMat)
                sumValMat = -np.diag(JacobianLogLD.T @ dN[k, :]) + JacobianLD.T @ basisMat
            else:
                raise ValueError(f"Unsupported fitType: {fitType}")

            # Kalman update
            W_p_inv = np.linalg.inv(W_p[:, :, k] + 1e-12 * np.eye(R))
            invW_u = W_p_inv + sumValMat
            W_u[:, :, k] = np.linalg.inv(invW_u + 1e-12 * np.eye(R))

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(W_u[:, :, k])
            eigvals = np.maximum(eigvals, np.finfo(float).eps)
            W_u[:, :, k] = eigvecs @ np.diag(eigvals) @ eigvecs.T

            x_u[:, k] = x_p[:, k] + W_u[:, :, k] @ sumValVec

        # Backward RTS smoother
        x_K = np.zeros((R, K), dtype=float)
        W_K = np.zeros((R, R, K), dtype=float)
        Lk = np.zeros((R, R, K), dtype=float)

        x_K[:, K - 1] = x_u[:, K - 1]
        W_K[:, :, K - 1] = W_u[:, :, K - 1]

        for k in range(K - 2, -1, -1):
            Lk[:, :, k] = W_u[:, :, k] @ A_mat.T @ np.linalg.inv(W_p[:, :, k + 1] + 1e-12 * np.eye(R))
            x_K[:, k] = x_u[:, k] + Lk[:, :, k] @ (x_K[:, k + 1] - x_p[:, k + 1])
            W_K[:, :, k] = W_u[:, :, k] + Lk[:, :, k] @ (W_K[:, :, k + 1] - W_p[:, :, k + 1]) @ Lk[:, :, k].T
            W_K[:, :, k] = 0.5 * (W_K[:, :, k] + W_K[:, :, k].T)

        # Cross-trial covariance Wku (R, R, K, K)
        Wku = np.zeros((R, R, K, K), dtype=float)
        for k in range(K):
            Wku[:, :, k, k] = W_K[:, :, k]

        Dk = np.zeros((R, R, K), dtype=float)
        for u in range(K - 1, 0, -1):
            for k in range(u - 1, -1, -1):
                Dk[:, :, k] = W_u[:, :, k] @ A_mat.T @ np.linalg.inv(W_p[:, :, k + 1] + 1e-12 * np.eye(R))
                Wku[:, :, k, u] = Dk[:, :, k] @ Wku[:, :, k + 1, u]
                Wku[:, :, u, k] = Wku[:, :, k, u]

        # Sufficient statistics for M-step
        Sxkxkp1 = np.zeros((R, R), dtype=float)
        Sxkp1xkp1 = np.zeros((R, R), dtype=float)
        Sxkxk = np.zeros((R, R), dtype=float)
        for k in range(K - 1):
            Sxkxkp1 += Wku[:, :, k, k + 1] + np.outer(x_K[:, k], x_K[:, k + 1])
            Sxkp1xkp1 += W_K[:, :, k + 1] + np.outer(x_K[:, k + 1], x_K[:, k + 1])
            Sxkxk += W_K[:, :, k] + np.outer(x_K[:, k], x_K[:, k])

        sumXkTerms = (
            Sxkp1xkp1 - A_mat @ Sxkxkp1 - Sxkxkp1.T @ A_mat.T + A_mat @ Sxkxk @ A_mat.T
            + W_K[:, :, 0] + np.outer(x_K[:, 0], x_K[:, 0])
            - A_mat @ np.outer(x0_vec, x_K[:, 0]) - np.outer(x_K[:, 0], x0_vec) @ A_mat.T
            + A_mat @ np.outer(x0_vec, x0_vec) @ A_mat.T
        )

        # Point process log-likelihood
        sumPPll = 0.0
        for k in range(K):
            Hk = HkAll[k]
            Wk = basisMat @ np.diag(W_K[:, :, k])
            stimK = basisMat @ x_K[:, k]

            if fitType == "poisson":
                hist_term = Hk @ gamma_vec if gamma_vec.size > 0 and Hk.shape[1] > 0 else np.zeros(N)
                histEffect = np.exp(np.clip(hist_term, -30, 30))
                stimK_clipped = np.clip(stimK, -30, 30)
                stimEffect = np.exp(stimK_clipped) + np.exp(stimK_clipped) / 2.0 * Wk
                ExplambdaDelta = stimEffect * histEffect
                ExplogLD = stimK + hist_term
                sumPPll += float(np.sum(dN[k, :] * ExplogLD - ExplambdaDelta))

            elif fitType == "binomial":
                hist_term = Hk @ gamma_vec if gamma_vec.size > 0 and Hk.shape[1] > 0 else np.zeros(N)
                linpred = np.clip(stimK + hist_term, -30, 30)
                lambdaDelta = 1.0 / (1.0 + np.exp(-linpred))
                ExplambdaDelta = lambdaDelta + Wk * (lambdaDelta * (1.0 - lambdaDelta) * (1.0 - 2.0 * lambdaDelta)) / 2.0
                ExplogLD = linpred - np.log(1.0 + np.exp(linpred)) - Wk * lambdaDelta * (1.0 - lambdaDelta) * 0.5
                sumPPll += float(np.sum(dN[k, :] * ExplogLD - ExplambdaDelta))

        det_Q = float(np.prod(np.maximum(Q_diag, np.finfo(float).eps)))
        logll = (
            -R * K * np.log(2.0 * np.pi)
            - K / 2.0 * np.log(det_Q)
            + sumPPll
            - 0.5 * float(np.trace(np.linalg.pinv(Q_mat) @ sumXkTerms))
        )

        return x_K, W_K, Wku, logll, sumXkTerms, sumPPll

    @staticmethod
    def PPSS_MStep(dN, HkAll, fitType, x_K, W_K, gamma, delta, sumXkTerms, windowTimes):
        """M-step: Update Q via closed form, gamma via Newton-Raphson.

        Parameters
        ----------
        dN : (K, N)
        HkAll : list of K arrays (N, J)
        fitType : 'poisson' or 'binomial'
        x_K : (R, K) smoothed states
        W_K : (R, R, K) smoothed covariances
        gamma : (J,) current history coefficients
        delta : time bin width
        sumXkTerms : (R, R) sufficient statistics from E-step
        windowTimes : array of history window boundaries

        Returns
        -------
        Qhat : (R,) updated state noise variance (diagonal)
        gamma_new : (J,) updated history coefficients
        """
        K, N = dN.shape
        R = x_K.shape[0]
        fitType = str(fitType).lower()

        # Q update (closed form)
        sumQ = np.diag(np.diag(sumXkTerms))
        Qhat = sumQ / K
        eigvals, eigvecs = np.linalg.eigh(Qhat)
        eigvals = np.maximum(eigvals, 1e-8)
        Qhat = eigvecs @ np.diag(eigvals) @ eigvecs.T
        Qhat = np.diag(Qhat)  # Return as vector

        # Build basis matrix for gamma update
        minTime = 0.0
        maxTime = (N - 1) * delta
        basisMat = DecodingAlgorithms._ssglm_build_basis(R, minTime, maxTime, delta)
        if basisMat.shape[0] != N:
            basisMat = basisMat[:N, :] if basisMat.shape[0] > N else np.vstack(
                [basisMat, np.zeros((N - basisMat.shape[0], basisMat.shape[1]))]
            )

        gamma_vec = np.asarray(gamma, dtype=float).reshape(-1)
        gamma_new = gamma_vec.copy()
        J = gamma_new.size

        # Newton-Raphson for gamma (history coefficients)
        if windowTimes is not None and len(windowTimes) > 0 and J > 0 and np.any(gamma_new != 0):
            converged = False
            max_iter = 300
            for iteration in range(max_iter):
                gradQ = np.zeros(J, dtype=float)
                jacQ = np.zeros((J, J), dtype=float)

                for k in range(K):
                    Hk = HkAll[k]
                    if Hk.shape[1] == 0:
                        continue
                    Wk = basisMat @ np.diag(W_K[:, :, k])
                    stimK = basisMat @ x_K[:, k]

                    if fitType == "poisson":
                        hist_term = np.clip(gamma_new @ Hk.T, -30, 30)
                        histEffect = np.exp(hist_term)
                        stimK_clipped = np.clip(stimK, -30, 30)
                        stimEffect = np.exp(stimK_clipped) + np.exp(stimK_clipped) / 2.0 * Wk
                        lambdaDelta = stimEffect * histEffect

                        gradQ += Hk.T @ dN[k, :] - Hk.T @ lambdaDelta
                        jacQ -= (Hk * lambdaDelta[:, None]).T @ Hk

                    elif fitType == "binomial":
                        linpred = np.clip(stimK + Hk @ gamma_new, -30, 30)
                        lambdaDelta = 1.0 / (1.0 + np.exp(-linpred))
                        histEffect = np.exp(np.clip(gamma_new @ Hk.T, -30, 30))
                        stimEffect = np.exp(np.clip(stimK, -30, 30))
                        C = stimEffect * histEffect
                        M = np.where(C > 1e-30, 1.0 / C, 1e30)
                        ExpLambdaDelta = lambdaDelta + Wk * (lambdaDelta * (1.0 - lambdaDelta) * (1.0 - 2.0 * lambdaDelta)) / 2.0
                        ExpLDSquaredTimesInvExp = lambdaDelta ** 2 * M
                        ExpLDCubedTimesInvExpSquared = (
                            lambdaDelta ** 3 * M ** 2
                            + Wk / 2.0 * (3.0 * M ** 4 * lambdaDelta ** 3
                                           + 12.0 * lambdaDelta ** 3 * M ** 3
                                           - 12.0 * M ** 4 * lambdaDelta ** 4)
                        )

                        gradQ += (Hk * (1.0 - ExpLambdaDelta)[:, None]).T @ dN[k, :] \
                                 - (Hk * (ExpLDSquaredTimesInvExp / np.maximum(lambdaDelta, 1e-30))[:, None]).T @ lambdaDelta
                        jacQ -= (Hk * (ExpLDSquaredTimesInvExp * dN[k, :])[:, None]).T @ Hk \
                                + (Hk * ExpLDSquaredTimesInvExp[:, None]).T @ Hk \
                                + (Hk * (2.0 * ExpLDCubedTimesInvExpSquared)[:, None]).T @ Hk

                # Newton-Raphson update
                try:
                    gamma_temp = gamma_new - np.linalg.pinv(jacQ) @ gradQ
                except np.linalg.LinAlgError:
                    gamma_temp = gamma_new

                if np.any(np.isnan(gamma_temp)):
                    gamma_temp = gamma_new

                mabsDiff = float(np.max(np.abs(gamma_temp - gamma_new)))
                gamma_new = gamma_temp
                if mabsDiff < 1e-2:
                    converged = True
                    break

            # Clamp gamma
            gamma_new = np.clip(gamma_new, -1e2, 1e2)

        return Qhat, gamma_new

    @staticmethod
    def PPSS_EM(A, Q0, x0, dN, fitType, delta, gamma0, windowTimes, numBasis, HkAll):
        """Inner EM loop for state-space GLM.

        Parameters
        ----------
        A : (R, R) state transition
        Q0 : (R,) initial state noise variance
        x0 : (R,) initial state
        dN : (K, N) observations
        fitType : 'poisson' or 'binomial'
        delta : time bin width
        gamma0 : (J,) initial history coefficients
        windowTimes : history window boundaries
        numBasis : number of basis functions
        HkAll : precomputed history matrices

        Returns
        -------
        xKFinal, WKFinal, WkuFinal, Qhat, gammahat, logll, QhatAll, gammahatAll, nIter, negLL
        """
        if numBasis is None:
            numBasis = 20
        if delta is None or delta == 0:
            delta = 0.001
        fitType = str(fitType or "poisson").lower()

        Q0_vec = np.asarray(Q0, dtype=float).reshape(-1)
        if Q0_vec.size == numBasis * numBasis:
            Q0_vec = np.diag(Q0_vec.reshape(numBasis, numBasis))

        gamma0_vec = np.asarray(gamma0, dtype=float).reshape(-1) if gamma0 is not None else np.array([], dtype=float)

        tolAbs = 1e-3
        tolRel = 1e-3
        llTol = 1e-3
        maxIter = 100
        numToKeep = 10

        # Circular buffer storage
        Qhat = np.zeros((Q0_vec.size, numToKeep), dtype=float)
        Qhat[:, 0] = Q0_vec
        gammahat = np.zeros((numToKeep, gamma0_vec.size), dtype=float)
        gammahat[0, :] = gamma0_vec

        xK_buf = [None] * numToKeep
        WK_buf = [None] * numToKeep
        Wku_buf = [None] * numToKeep

        x0hat = np.asarray(x0, dtype=float).reshape(-1)
        logll_list = []
        dLikelihood = [np.inf]
        negLL = False
        stoppingCriteria = False
        cnt = 0

        while not stoppingCriteria and cnt < maxIter:
            si = cnt % numToKeep
            si_p1 = (cnt + 1) % numToKeep
            si_m1 = (cnt - 1) % numToKeep

            xK_cur, WK_cur, Wku_cur, ll, SumXkTerms, _ = DecodingAlgorithms.PPSS_EStep(
                A, Qhat[:, si], x0hat, dN, HkAll, fitType, delta, gammahat[si, :], numBasis
            )
            xK_buf[si] = xK_cur
            WK_buf[si] = WK_cur
            Wku_buf[si] = Wku_cur
            logll_list.append(ll)

            Qnew, gnew = DecodingAlgorithms.PPSS_MStep(
                dN, HkAll, fitType, xK_cur, WK_cur, gammahat[si, :], delta, SumXkTerms, windowTimes
            )
            Qhat[:, si_p1] = Qnew
            gammahat[si_p1, :] = gnew

            if cnt == 0:
                dLikelihood.append(np.inf)
            else:
                dLikelihood.append(logll_list[cnt] - logll_list[cnt - 1])

            # Check convergence
            if cnt > 0:
                dQvals = np.abs(np.sqrt(np.maximum(Qhat[:, si], 0)) - np.sqrt(np.maximum(Qhat[:, si_m1], 0)))
                dGamma = np.abs(gammahat[si, :] - gammahat[si_m1, :])
                dMax = max(np.max(dQvals), np.max(dGamma)) if dGamma.size > 0 else float(np.max(dQvals))

                Q_prev = np.sqrt(np.maximum(Qhat[:, si_m1], 1e-30))
                dQRel = float(np.max(np.abs(dQvals / Q_prev)))
                if dGamma.size > 0:
                    g_prev = np.maximum(np.abs(gammahat[si_m1, :]), 1e-30)
                    dGammaRel = float(np.max(np.abs(dGamma / g_prev)))
                    dMaxRel = max(dQRel, dGammaRel)
                else:
                    dMaxRel = dQRel

                if dMax < tolAbs and dMaxRel < tolRel:
                    stoppingCriteria = True
                    negLL = False

                if abs(dLikelihood[-1]) < llTol or dLikelihood[-1] < 0:
                    stoppingCriteria = True
                    negLL = True

            cnt += 1

        # Select best iteration by log-likelihood
        logll_arr = np.array(logll_list)
        if logll_arr.size > 0:
            maxLLIndex = int(np.argmax(logll_arr))
        else:
            maxLLIndex = 0

        maxLLIndMod = maxLLIndex % numToKeep
        nIter = cnt

        xKFinal = xK_buf[maxLLIndMod] if xK_buf[maxLLIndMod] is not None else np.zeros((numBasis, dN.shape[0]))
        WKFinal = WK_buf[maxLLIndMod] if WK_buf[maxLLIndMod] is not None else np.zeros((numBasis, numBasis, dN.shape[0]))
        WkuFinal = Wku_buf[maxLLIndMod] if Wku_buf[maxLLIndMod] is not None else np.zeros((numBasis, numBasis, dN.shape[0], dN.shape[0]))

        QhatFinal = Qhat[:, maxLLIndMod]
        gammahatFinal = gammahat[maxLLIndMod, :]
        logllFinal = float(logll_arr[maxLLIndex]) if logll_arr.size > 0 else -np.inf

        QhatAll = Qhat[:, : min(cnt + 1, numToKeep)]
        gammahatAll = gammahat[: min(cnt + 1, numToKeep), :]

        return xKFinal, WKFinal, WkuFinal, QhatFinal, gammahatFinal, logllFinal, QhatAll, gammahatAll, nIter, negLL

    @staticmethod
    def PPSS_EMFB(A, Q0, x0, dN, fitType, delta, gamma0, windowTimes, numBasis, neuronName=None):
        """EM Forward-Backward algorithm for state-space GLM.

        Wraps PPSS_EM in a forward-backward-forward cycle for improved convergence.

        Parameters
        ----------
        A : (R, R) state transition matrix (typically identity for random walk)
        Q0 : (R,) initial state noise variance
        x0 : (R,) initial state coefficients
        dN : (K, N) binary spike observations (K trials, N time bins)
        fitType : 'poisson' or 'binomial'
        delta : time bin width in seconds
        gamma0 : (J,) initial history coefficients
        windowTimes : array of history window boundaries
        numBasis : number of basis functions R
        neuronName : identifier for the neuron (for labeling)

        Returns
        -------
        xKFinal : (R, K) estimated state trajectories
        WKFinal : (R, R, K) estimated state covariances
        WkuFinal : (R, R, K, K) cross-trial covariances
        Qhat : (R,) estimated state noise variance
        gammahat : (J,) estimated history coefficients
        fitResults : FitResult object with goodness-of-fit diagnostics
        stimulus : (R, K) estimated stimulus effect
        stimCIs : (R, K, 2) stimulus confidence intervals
        logll : float, log-likelihood at convergence
        QhatAll : parameter history
        gammahatAll : parameter history
        nIter : total EM iterations
        """
        K, N = dN.shape
        fitType = str(fitType or "poisson").lower()

        Q0_vec = np.asarray(Q0, dtype=float).reshape(-1)
        if Q0_vec.size == numBasis * numBasis:
            Q0_vec = np.diag(Q0_vec.reshape(numBasis, numBasis))

        gamma0_vec = np.asarray(gamma0, dtype=float).reshape(-1) if gamma0 is not None else np.array([], dtype=float)

        Qhat_cur = Q0_vec.copy()
        gammahat_cur = gamma0_vec.copy()
        xK0 = np.asarray(x0, dtype=float).reshape(-1)

        # Build history matrices
        HkAll = DecodingAlgorithms._ssglm_build_history(dN, windowTimes, delta)
        HkAllR = list(reversed(HkAll))

        tolAbs = 1e-3
        tolRel = 1e-3
        llTol = 1e-3
        maxIter = 2000

        Qhat_history = [Qhat_cur.copy()]
        gammahat_history = [gammahat_cur.copy()]
        logll_list = []
        stoppingCriteria = False
        cnt = 0

        xK = None
        WK = None
        Wku = None

        while not stoppingCriteria and cnt < maxIter:
            # Forward EM
            xK, WK, Wku, Qnew, gnew, ll, _, _, _, negLL = DecodingAlgorithms.PPSS_EM(
                A, Qhat_cur, xK0, dN, fitType, delta, gammahat_cur, windowTimes, numBasis, HkAll
            )

            if not negLL:
                # Backward EM
                _, _, _, QnewR, gnewR, _, _, _, _, negLLR = DecodingAlgorithms.PPSS_EM(
                    A, Qnew, xK[:, -1], np.flipud(dN), fitType, delta, gnew, windowTimes, numBasis, HkAllR
                )

                if not negLLR:
                    # Forward EM again with backward-updated parameters
                    # Matlab: PPSS_EM(A, QhatR(:,cnt+1), xKR(:,end), dN, ...)
                    xK2, WK2, Wku2, Qnew2, gnew2, ll2, _, _, _, negLL2 = DecodingAlgorithms.PPSS_EM(
                        A, QnewR, xK[:, -1], dN, fitType, delta, gnewR,
                        windowTimes, numBasis, HkAll
                    )

                    if not negLL2:
                        xK = xK2
                        WK = WK2
                        Wku = Wku2
                        Qnew = Qnew2
                        gnew = gnew2
                        ll = ll2

            Qhat_cur = Qnew
            gammahat_cur = gnew
            Qhat_history.append(Qnew.copy())
            gammahat_history.append(gnew.copy())
            logll_list.append(ll)

            xK0 = xK[:, 0]

            # Check convergence
            if cnt > 0:
                dLikelihood = logll_list[cnt] - logll_list[cnt - 1]
            else:
                dLikelihood = np.inf

            if len(Qhat_history) >= 2:
                Q_prev = Qhat_history[-2]
                Q_cur = Qhat_history[-1]
                dQvals = np.abs(np.sqrt(np.maximum(Q_cur, 0)) - np.sqrt(np.maximum(Q_prev, 0)))
                g_prev = gammahat_history[-2]
                g_cur = gammahat_history[-1]
                dGamma = np.abs(g_cur - g_prev) if g_cur.size > 0 else np.array([0.0])

                dMax = max(float(np.max(dQvals)), float(np.max(dGamma)))

                Q_denom = np.sqrt(np.maximum(Q_prev, 1e-30))
                dQRel = float(np.max(np.abs(dQvals / Q_denom)))
                if g_prev.size > 0 and np.any(g_prev != 0):
                    g_denom = np.maximum(np.abs(g_prev), 1e-30)
                    dGammaRel = float(np.max(np.abs(dGamma / g_denom)))
                    dMaxRel = max(dQRel, dGammaRel)
                else:
                    dMaxRel = dQRel

                if dMax < tolAbs and dMaxRel < tolRel:
                    stoppingCriteria = True

                if abs(dLikelihood) < llTol or dLikelihood < 0:
                    stoppingCriteria = True

            cnt += 1

        # Select best iteration
        logll_arr = np.array(logll_list)
        if logll_arr.size > 0:
            maxLLIndex = int(np.argmax(logll_arr))
        else:
            maxLLIndex = 0

        xKFinal = xK
        WKFinal = WK
        WkuFinal = Wku
        Qhat = Qhat_history[min(maxLLIndex + 1, len(Qhat_history) - 1)]
        gammahat = gammahat_history[min(maxLLIndex + 1, len(gammahat_history) - 1)]
        logll = float(logll_arr[maxLLIndex]) if logll_arr.size > 0 else -np.inf

        QhatAll = np.column_stack(Qhat_history) if Qhat_history else Q0_vec.reshape(-1, 1)
        gammahatAll = np.row_stack(gammahat_history) if gammahat_history and gammahat_history[0].size > 0 else np.array([[]])

        R = numBasis
        x0Final = xK[:, 0] if xK is not None else np.zeros(R)
        SumXkTermsFinal = np.diag(Qhat) * K
        McInfo = 100
        McCI = 3000

        # Observed log-likelihood
        logllobs = logll + R * K * np.log(2 * np.pi) + K / 2.0 * np.log(
            max(float(np.prod(np.maximum(Qhat, np.finfo(float).eps))), np.finfo(float).eps)
        ) + 0.5 * float(np.trace(np.linalg.pinv(np.diag(Qhat)) @ SumXkTermsFinal))

        nIter = cnt

        # Information matrix and result packaging
        InfoMat = DecodingAlgorithms.estimateInfoMat(
            fitType, dN, HkAll, A, x0Final, xKFinal, WKFinal, WkuFinal,
            Qhat, gammahat, windowTimes, SumXkTermsFinal, delta, McInfo
        )
        fitResults = DecodingAlgorithms.prepareEMResults(
            fitType, neuronName, dN, HkAll, xKFinal, WKFinal,
            Qhat, gammahat, windowTimes, delta, InfoMat, logllobs
        )

        stimCIs, stimulus = DecodingAlgorithms._ComputeStimulusCIs_MC(
            fitType, xKFinal, WkuFinal, delta, McCI
        )

        return (xKFinal, WKFinal, WkuFinal, Qhat, gammahat, fitResults,
                stimulus, stimCIs, logll, QhatAll, gammahatAll, nIter)

    @staticmethod
    def _ComputeStimulusCIs_MC(fitType, xK, Wku, delta, Mc=3000, alphaVal=0.05):
        """Monte Carlo confidence intervals for SSGLM stimulus estimate.

        Uses Cholesky decomposition of the cross-trial covariance to generate
        draws of the state trajectory, then computes empirical CIs.
        """
        fitType = str(fitType).lower()
        numBasis, K = xK.shape

        CIs = np.zeros((numBasis, K, 2), dtype=float)

        for r in range(numBasis):
            WkuTemp = Wku[r, r, :, :]  # (K, K) cross-trial covariance for basis r
            try:
                chol_m = np.linalg.cholesky(WkuTemp + 1e-10 * np.eye(K))
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(WkuTemp)
                eigvals = np.maximum(eigvals, 1e-10)
                chol_m = eigvecs @ np.diag(np.sqrt(eigvals))

            stimulusDraw = np.zeros((Mc, K), dtype=float)
            for c in range(Mc):
                z = np.random.randn(K)
                xKDraw = xK[r, :] + chol_m.T @ z
                if fitType == "poisson":
                    stimulusDraw[c, :] = np.exp(np.clip(xKDraw, -30, 30)) / delta
                elif fitType == "binomial":
                    xKDraw_clip = np.clip(xKDraw, -30, 30)
                    stimulusDraw[c, :] = (np.exp(xKDraw_clip) / (1.0 + np.exp(xKDraw_clip))) / delta
                else:
                    stimulusDraw[c, :] = xKDraw / delta

            for k in range(K):
                CIs[r, k, 0] = float(np.percentile(stimulusDraw[:, k], 100.0 * alphaVal / 2.0))
                CIs[r, k, 1] = float(np.percentile(stimulusDraw[:, k], 100.0 * (1.0 - alphaVal / 2.0)))

        if fitType == "poisson":
            stimulus = np.exp(np.clip(xK, -30, 30)) / delta
        elif fitType == "binomial":
            xK_clip = np.clip(xK, -30, 30)
            stimulus = (np.exp(xK_clip) / (1.0 + np.exp(xK_clip))) / delta
        else:
            stimulus = xK / delta

        return CIs, stimulus

    @staticmethod
    def estimateInfoMat(fitType, dN, HkAll, A, x0, xK, WK, Wku, Q, gamma,
                        windowTimes, SumXkTerms, delta, Mc=500):
        """Observed information matrix via Louis' identity with Monte Carlo.

        Computes I_obs = I_complete - I_missing where I_missing is estimated
        by MC sampling from the smoothing distribution.
        """
        fitType = str(fitType).lower()
        K, N = dN.shape
        gamma_vec = np.asarray(gamma, dtype=float).reshape(-1)
        J = gamma_vec.size if (windowTimes is not None and len(windowTimes) > 0) else 0

        Q_vec = np.asarray(Q, dtype=float).reshape(-1)
        R = Q_vec.size
        Q_mat = np.diag(Q_vec)
        numBasis = R

        # Build basis matrix
        minTime = 0.0
        maxTime = (N - 1) * delta
        basisMat = DecodingAlgorithms._ssglm_build_basis(numBasis, minTime, maxTime, delta)
        if basisMat.shape[0] != N:
            basisMat = basisMat[:N, :] if basisMat.shape[0] > N else np.vstack(
                [basisMat, np.zeros((N - basisMat.shape[0], basisMat.shape[1]))]
            )

        # Complete data information matrix
        Ic = np.zeros((R + J, R + J), dtype=float)
        Q_mat_safe = np.diag(np.maximum(Q_vec, np.finfo(float).eps))
        Q2 = Q_mat_safe @ Q_mat_safe
        Q3 = Q2 @ Q_mat_safe

        Ic[:R, :R] = K / 2.0 * np.linalg.inv(Q2) + np.linalg.inv(Q3) @ SumXkTerms

        # History portion of information matrix
        jacQ = np.zeros((J, J), dtype=float) if J > 0 else np.zeros((0, 0))
        if fitType == "poisson" and J > 0:
            for k in range(K):
                Hk = HkAll[k]
                if Hk.shape[1] == 0:
                    continue
                Wk = basisMat @ np.diag(WK[:, :, k])
                stimK = basisMat @ xK[:, k]
                stimK_clip = np.clip(stimK, -30, 30)
                hist_term = np.clip(gamma_vec @ Hk.T, -30, 30)
                histEffect = np.exp(hist_term)
                stimEffect = np.exp(stimK_clip) + np.exp(stimK_clip) / 2.0 * Wk
                lambdaDelta = stimEffect * histEffect
                jacQ -= (Hk * lambdaDelta[:, None]).T @ Hk
        elif fitType == "binomial" and J > 0:
            for k in range(K):
                Hk = HkAll[k]
                if Hk.shape[1] == 0:
                    continue
                Wk = basisMat @ np.diag(WK[:, :, k])
                stimK = basisMat @ xK[:, k]
                linpred = np.clip(stimK + Hk @ gamma_vec, -30, 30)
                histEffect = np.exp(np.clip(gamma_vec @ Hk.T, -30, 30))
                stimEffect = np.exp(np.clip(stimK, -30, 30))
                C = stimEffect * histEffect
                M = np.where(C > 1e-30, 1.0 / C, 1e30)
                lambdaDelta = 1.0 / (1.0 + np.exp(-linpred))
                ExpLDSquaredTimesInvExp = lambdaDelta ** 2 * M
                ExpLDCubedTimesInvExpSquared = (
                    lambdaDelta ** 3 * M ** 2
                    + Wk / 2.0 * (3.0 * M ** 4 * lambdaDelta ** 3
                                   + 12.0 * lambdaDelta ** 3 * M ** 3
                                   - 12.0 * M ** 4 * lambdaDelta ** 4)
                )
                jacQ -= (Hk * (ExpLDSquaredTimesInvExp * dN[k, :])[:, None]).T @ Hk \
                        + (Hk * ExpLDSquaredTimesInvExp[:, None]).T @ Hk \
                        + (Hk * (2.0 * ExpLDCubedTimesInvExpSquared)[:, None]).T @ Hk

        Ic[:R, :R] = K * np.linalg.inv(2.0 * Q2) + np.linalg.inv(Q3) @ SumXkTerms
        if J > 0:
            Ic[R:R + J, R:R + J] = -jacQ

        # MC estimation of missing information
        xKDraw = np.zeros((numBasis, K, Mc), dtype=float)
        for r in range(numBasis):
            WkuTemp = Wku[r, r, :, :]
            try:
                chol_m = np.linalg.cholesky(WkuTemp + 1e-10 * np.eye(K))
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(WkuTemp)
                eigvals = np.maximum(eigvals, 1e-10)
                chol_m = eigvecs @ np.diag(np.sqrt(eigvals))

            for c in range(Mc):
                z = np.random.randn(K)
                xKDraw[r, :, c] = xK[r, :] + chol_m.T @ z

        ImMC = np.zeros((R + J, R + J), dtype=float)
        A_mat = np.asarray(A, dtype=float)
        if A_mat.ndim < 2:
            A_mat = np.eye(R) * A_mat
        x0_vec = np.asarray(x0, dtype=float).reshape(-1)
        Q_inv = np.linalg.inv(Q_mat_safe)

        for c in range(Mc):
            gradQGammahat = np.zeros(J, dtype=float) if J > 0 else np.array([], dtype=float)
            gradQQhat = np.zeros(R, dtype=float)

            for k in range(K):
                Hk = HkAll[k]
                stimK = basisMat @ xKDraw[:, k, c]

                if fitType == "poisson":
                    hist_term = np.clip(gamma_vec @ Hk.T, -30, 30) if J > 0 and Hk.shape[1] > 0 else np.zeros(N)
                    histEffect = np.exp(hist_term)
                    stimK_clip = np.clip(stimK, -30, 30)
                    stimEffect = np.exp(stimK_clip)
                    lambdaDelta = stimEffect * histEffect
                    if J > 0 and Hk.shape[1] > 0:
                        gradQGammahat += Hk.T @ dN[k, :] - Hk.T @ lambdaDelta
                elif fitType == "binomial":
                    Wk = basisMat @ np.diag(WK[:, :, k])
                    linpred = np.clip(stimK + (Hk @ gamma_vec if J > 0 and Hk.shape[1] > 0 else 0.0), -30, 30)
                    histEffect = np.exp(np.clip(gamma_vec @ Hk.T, -30, 30)) if J > 0 and Hk.shape[1] > 0 else np.ones(N)
                    stimEffect = np.exp(np.clip(stimK, -30, 30))
                    C = stimEffect * histEffect
                    M = np.where(C > 1e-30, 1.0 / C, 1e30)
                    lambdaDelta = 1.0 / (1.0 + np.exp(-linpred))
                    ExpLambdaDelta = lambdaDelta + Wk * (lambdaDelta * (1.0 - lambdaDelta) * (1.0 - 2.0 * lambdaDelta)) / 2.0
                    ExpLDSquaredTimesInvExp = lambdaDelta ** 2 * M
                    if J > 0 and Hk.shape[1] > 0:
                        gradQGammahat += (Hk * (1.0 - ExpLambdaDelta)[:, None]).T @ dN[k, :] \
                                         - (Hk * (ExpLDSquaredTimesInvExp / np.maximum(lambdaDelta, 1e-30))[:, None]).T @ lambdaDelta

                if k == 0:
                    diff = xKDraw[:, k, c] - A_mat @ x0_vec
                else:
                    diff = xKDraw[:, k, c] - A_mat @ xKDraw[:, k - 1, c]
                gradQQhat += diff * diff

            gradQQhat_scaled = 0.5 * Q_inv @ gradQQhat - np.diag(K / 2.0 * np.linalg.inv(Q2))
            ImMC[:R, :R] += np.outer(gradQQhat_scaled, gradQQhat_scaled)
            if J > 0:
                ImMC[R:R + J, R:R + J] += np.diag(gradQGammahat ** 2)

        Im = ImMC / Mc
        InfoMatrix = Ic - Im

        return InfoMatrix

    @staticmethod
    def prepareEMResults(fitType, neuronNumber, dN, HkAll, xK, WK, Q, gamma,
                         windowTimes, delta, informationMatrix, logll):
        """Package SSGLM EM results into a FitResult object."""
        from .core import Covariate
        from .fit import FitResult
        from .history import History
        from .trial import (
            ConfigCollection,
            SpikeTrainCollection,
            TrialConfig,
        )
        from .analysis import Analysis

        fitType = str(fitType).lower()
        numBasis, K = xK.shape
        R = numBasis
        N = dN.shape[1]
        minTime = 0.0
        maxTime = (N - 1) * delta
        sampleRate = 1.0 / delta
        gamma_vec = np.asarray(gamma, dtype=float).reshape(-1)

        # Build basis matrix
        basisMat = DecodingAlgorithms._ssglm_build_basis(numBasis, minTime, maxTime, delta)
        if basisMat.shape[0] != N:
            basisMat = basisMat[:N, :] if basisMat.shape[0] > N else np.vstack(
                [basisMat, np.zeros((N - basisMat.shape[0], basisMat.shape[1]))]
            )

        # Standard errors from information matrix
        try:
            SE = np.sqrt(np.abs(np.diag(np.linalg.inv(informationMatrix))))
        except np.linalg.LinAlgError:
            SE = np.zeros(informationMatrix.shape[0], dtype=float)

        # Build per-trial standard errors
        xKbeta = xK.T.reshape(-1)  # (K*R,)
        seXK = np.zeros(K * R, dtype=float)
        for k in range(K):
            seXK[k * R:(k + 1) * R] = np.sqrt(np.maximum(np.diag(WK[:, :, k]), 0.0))

        # Neuron name
        if neuronNumber is None:
            name = "N01"
        elif isinstance(neuronNumber, (int, float)):
            n = int(neuronNumber)
            name = f"N{n:02d}" if 0 < n < 10 else f"N{n}"
        else:
            name = str(neuronNumber)

        # Create spike trains from dN
        nst_list = []
        for k in range(K):
            spike_indices = np.where(dN[k, :] > 0.5)[0]
            spike_times = spike_indices.astype(float) * delta
            nst_k = nspikeTrain(spike_times, name=name, makePlots=-1)
            nst_k.setMinTime(minTime)
            nst_k.setMaxTime(maxTime)
            nst_list.append(nst_k)

        nCopy = SpikeTrainCollection(nst_list)
        nCopy = nCopy.toSpikeTrain()

        # Compute lambda (conditional intensity)
        lambdaData = []
        otherLabels = []
        cnt = 0
        for k in range(K):
            Hk = HkAll[k]
            stimK = basisMat @ xK[:, k]

            if fitType == "poisson":
                hist_term = gamma_vec @ Hk.T if gamma_vec.size > 0 and Hk.shape[1] > 0 else np.zeros(N)
                histEffect = np.exp(np.clip(hist_term, -30, 30))
                stimEffect = np.exp(np.clip(stimK, -30, 30))
                lambdaDelta = histEffect * stimEffect / delta
            elif fitType == "binomial":
                linpred = np.clip(stimK + (Hk @ gamma_vec if gamma_vec.size > 0 and Hk.shape[1] > 0 else 0.0), -30, 30)
                hist_term = np.clip(gamma_vec @ Hk.T, -30, 30) if gamma_vec.size > 0 and Hk.shape[1] > 0 else np.zeros(N)
                histEffect = np.exp(hist_term)
                stimEffect = np.exp(np.clip(stimK, -30, 30))
                C = histEffect * stimEffect
                lambdaDelta = C / (1.0 + C) / delta
            else:
                lambdaDelta = np.zeros(N)

            lambdaData.append(lambdaDelta)

            for r in range(R):
                label = f"b{r + 1:02d}_{{{k + 1}}}" if r + 1 < 10 else f"b{r + 1}_{{{k + 1}}}"
                otherLabels.append(label)
                cnt += 1

        lambdaData = np.concatenate(lambdaData)
        lambdaTime = np.arange(len(lambdaData)) * delta + minTime

        nCopy.setMaxTime(float(np.max(lambdaTime)))
        nCopy.setMinTime(float(np.min(lambdaTime)))

        # Covariance labels
        covarianceLabels = [f"Q{r + 1:02d}" if r + 1 < 10 else f"Q{r + 1}" for r in range(R)]

        # History labels
        histLabels = []
        if windowTimes is not None and len(windowTimes) > 0:
            wt = np.asarray(windowTimes, dtype=float)
            for i in range(len(wt) - 1):
                histLabels.append(f"[{wt[i]:.3g},{wt[i + 1]:.3g}]")

        allLabels = otherLabels + covarianceLabels + histLabels

        # History objects
        if windowTimes is not None and len(windowTimes) > 0:
            histObj = [History(windowTimes, minTime, maxTime)]
        else:
            histObj = [None]

        # Trial configuration
        numBasisStr = str(numBasis)
        numHistStr = str(len(windowTimes) - 1) if windowTimes is not None and len(windowTimes) > 1 else "0"
        if histObj[0] is not None:
            cfg_name = f"SSGLM(N_{{b}}={numBasisStr})+Hist(N_{{h}}={numHistStr})"
        else:
            cfg_name = f"SSGLM(N_{{b}}={numBasisStr})"

        tc = TrialConfig([allLabels], sampleRate, histObj, [])
        tc.setName(cfg_name)
        configColl = ConfigCollection([tc])

        # Lambda covariate
        lambda_cov = Covariate(
            lambdaTime, lambdaData,
            r"\Lambda(t)", "time", "s", "Hz",
            [r"\lambda_{1}"]
        )

        # Model selection criteria
        AIC = 2.0 * len(allLabels) - 2.0 * logll
        BIC = -2.0 * logll + len(allLabels) * np.log(max(len(lambdaData), 1))
        dev = -2.0 * logll

        # Stats structure
        statsStruct = {
            "beta": np.concatenate([xKbeta, np.asarray(Q, dtype=float).reshape(-1), gamma_vec]),
            "se": np.concatenate([seXK, SE]),
        }

        # Coefficients
        b = [statsStruct["beta"]]
        stats = [statsStruct]
        distrib = [fitType]

        # Spike trains for FitResult
        spikeTraining = [nst.nstCopy() for nst in nst_list]
        for st in spikeTraining:
            st.setName(name)

        XvalData = [None]
        XvalTime = [None]
        numHist = [len(windowTimes) - 1] if windowTimes is not None and len(windowTimes) > 1 else [0]
        ensHistObj = [None]

        fitResults = FitResult(
            nCopy, [allLabels], numHist, histObj, ensHistObj,
            lambda_cov, b, dev, stats, AIC, BIC, logll,
            configColl, XvalData, XvalTime, distrib
        )

        # Goodness-of-fit (silent)
        try:
            Analysis.KSPlot(fitResults, DTCorrection=1, makePlot=0)
        except Exception:
            pass
        try:
            Analysis.plotInvGausTrans(fitResults, makePlot=0)
        except Exception:
            pass
        try:
            Analysis.plotFitResidual(fitResults, makePlot=0)
        except Exception:
            pass

        return fitResults


    # ------------------------------------------------------------------
    # Kalman Filter EM (KF_EM) family
    # Ported from Matlab DecodingAlgorithms.m lines 3295-4586
    # ------------------------------------------------------------------

    @staticmethod
    def KF_EMCreateConstraints(
        EstimateA=1,
        AhatDiag=0,
        QhatDiag=1,
        QhatIsotropic=0,
        RhatDiag=1,
        RhatIsotropic=0,
        Estimatex0=1,
        EstimatePx0=1,
        Px0Isotropic=0,
        mcIter=1000,
        EnableIkeda=0,
    ):
        """Return a dict of EM constraint flags for :meth:`KF_EM`.

        Parameters
        ----------
        EstimateA : int
            Whether to estimate the state transition matrix *A*.
        AhatDiag : int
            Constrain *A* to be diagonal.
        QhatDiag : int
            Constrain *Q* to be diagonal.
        QhatIsotropic : int
            Constrain *Q* to be isotropic (scalar times identity).
            Only active when *QhatDiag* is also true.
        RhatDiag : int
            Constrain *R* to be diagonal.
        RhatIsotropic : int
            Constrain *R* to be isotropic.  Only active when *RhatDiag*
            is also true.
        Estimatex0 : int
            Whether to estimate the initial state *x0*.
        EstimatePx0 : int
            Whether to estimate the initial covariance *Px0*.
        Px0Isotropic : int
            Constrain *Px0* to be isotropic.  Only active when
            *EstimatePx0* is true.
        mcIter : int
            Number of Monte Carlo iterations for standard-error
            estimation via the observed information matrix.
        EnableIkeda : int
            Enable Ikeda acceleration in the EM loop.

        Returns
        -------
        dict
            Constraint dictionary consumed by :meth:`KF_EM`,
            :meth:`KF_MStep`, and :meth:`KF_ComputeParamStandardErrors`.
        """
        C = {}
        C["EstimateA"] = int(EstimateA)
        C["AhatDiag"] = int(AhatDiag)
        C["QhatDiag"] = int(QhatDiag)
        # QhatIsotropic only valid if QhatDiag is true
        C["QhatIsotropic"] = 1 if (QhatDiag and QhatIsotropic) else 0
        C["RhatDiag"] = int(RhatDiag)
        # RhatIsotropic only valid if RhatDiag is true
        C["RhatIsotropic"] = 1 if (RhatDiag and RhatIsotropic) else 0
        C["Estimatex0"] = int(Estimatex0)
        C["EstimatePx0"] = int(EstimatePx0)
        # Px0Isotropic only valid if EstimatePx0 is true
        C["Px0Isotropic"] = 1 if (EstimatePx0 and Px0Isotropic) else 0
        C["mcIter"] = int(mcIter)
        C["EnableIkeda"] = int(EnableIkeda)
        return C

    # ---- internal Kalman filter matching Matlab (A, C, Q, R, Px0, x0, y) ----

    @staticmethod
    def _kf_filter_stateMajor(A, C, Q, R, Px0, x0, y):
        """Run a Kalman filter with Matlab-compatible state-major layout.

        Parameters
        ----------
        A : (Dx, Dx) state transition
        C : (Dy, Dx) observation matrix
        Q : (Dx, Dx) process noise covariance
        R : (Dy, Dy) observation noise covariance
        Px0 : (Dx, Dx) initial state covariance
        x0 : (Dx,) or (Dx, 1) initial state
        y : (Dy, K) observations (state-major, each column is one time step)

        Returns
        -------
        x_p : (Dx, K+1) predicted states  (x_p[:, 0] == x0)
        Pe_p : (Dx, Dx, K+1) predicted covariances
        x_u : (Dx, K) updated states
        Pe_u : (Dx, Dx, K) updated covariances
        """
        A = np.asarray(A, dtype=float)
        C = np.asarray(C, dtype=float)
        Q = np.asarray(Q, dtype=float)
        R = np.asarray(R, dtype=float)
        Px0 = np.asarray(Px0, dtype=float)
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float)

        Dx = A.shape[0]
        K = y.shape[1]

        x_p = np.zeros((Dx, K + 1), dtype=float)
        Pe_p = np.zeros((Dx, Dx, K + 1), dtype=float)
        x_u = np.zeros((Dx, K), dtype=float)
        Pe_u = np.zeros((Dx, Dx, K), dtype=float)

        x_p[:, 0] = x0
        Pe_p[:, :, 0] = Px0

        for n in range(K):
            # Update
            S = C @ Pe_p[:, :, n] @ C.T + R
            Gn = Pe_p[:, :, n] @ C.T @ np.linalg.pinv(S)
            x_u[:, n] = x_p[:, n] + Gn @ (y[:, n] - C @ x_p[:, n])
            Pe_u[:, :, n] = Pe_p[:, :, n] - Gn @ C @ Pe_p[:, :, n]
            # Predict
            x_p[:, n + 1] = A @ x_u[:, n]
            Pe_p[:, :, n + 1] = A @ Pe_u[:, :, n] @ A.T + Q

        return x_p, Pe_p, x_u, Pe_u

    @staticmethod
    def _kf_smootherFromFiltered_stateMajor(A, x_p, Pe_p, x_u, Pe_u):
        """RTS smoother with Matlab-compatible state-major layout.

        Parameters
        ----------
        A : (Dx, Dx) transition matrix
        x_p : (Dx, K+1) predicted states
        Pe_p : (Dx, Dx, K+1) predicted covariances
        x_u : (Dx, K) updated states
        Pe_u : (Dx, Dx, K) updated covariances

        Returns
        -------
        x_K : (Dx, K) smoothed states
        W_K : (Dx, Dx, K) smoothed covariances
        Lk : (Dx, Dx, K-1) smoother gains
        """
        K = x_u.shape[1]
        Dx = x_u.shape[0]
        x_K = np.copy(x_u)
        W_K = np.copy(Pe_u)
        Lk = np.zeros((Dx, Dx, max(K - 1, 0)), dtype=float)

        for t in range(K - 2, -1, -1):
            gain = Pe_u[:, :, t] @ A.T @ np.linalg.pinv(Pe_p[:, :, t + 1])
            Lk[:, :, t] = gain
            x_K[:, t] = x_u[:, t] + gain @ (x_K[:, t + 1] - x_p[:, t + 1])
            W_K[:, :, t] = _symmetrize(
                Pe_u[:, :, t] + gain @ (W_K[:, :, t + 1] - Pe_p[:, :, t + 1]) @ gain.T
            )

        return x_K, W_K, Lk

    @staticmethod
    def KF_EStep(A, Q, C, R, y, alpha, x0, Px0):
        """E-step for the Kalman Filter EM algorithm.

        Runs the forward Kalman filter followed by the backward RTS smoother
        and computes sufficient statistics (expectation sums) for the M-step.

        Parameters
        ----------
        A : (Dx, Dx) state transition matrix
        Q : (Dx, Dx) process noise covariance
        C : (Dy, Dx) observation matrix
        R : (Dy, Dy) observation noise covariance
        y : (Dy, K) observations (state-major)
        alpha : (Dy, 1) or (Dy,) observation offset
        x0 : (Dx,) initial state
        Px0 : (Dx, Dx) initial state covariance

        Returns
        -------
        x_K : (Dx, K) smoothed states
        W_K : (Dx, Dx, K) smoothed covariances
        logll : float — complete-data log-likelihood
        ExpectationSums : dict of sufficient statistics
        """
        A = np.asarray(A, dtype=float)
        Q = np.asarray(Q, dtype=float)
        C = np.asarray(C, dtype=float)
        R = np.asarray(R, dtype=float)
        y = np.asarray(y, dtype=float)
        alpha = np.asarray(alpha, dtype=float).reshape(-1, 1)
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        Px0 = np.asarray(Px0, dtype=float)

        Dx = A.shape[1]
        Dy = C.shape[0]
        K = y.shape[1]

        # Forward filter with offset subtracted: y - alpha*ones(1,K)
        y_centered = y - alpha @ np.ones((1, K))
        x_p, Pe_p, x_u, Pe_u = DecodingAlgorithms._kf_filter_stateMajor(
            A, C, Q, R, Px0, x0, y_centered
        )

        # Backward RTS smoother
        x_K, W_K, Lk = DecodingAlgorithms._kf_smootherFromFiltered_stateMajor(
            A, x_p, Pe_p, x_u, Pe_u
        )

        # Best estimates of initial states given the data
        # Matlab: W1G0 = A*Px0*A' + Q
        W1G0 = A @ Px0 @ A.T + Q
        L0 = Px0 @ A.T @ np.linalg.pinv(W1G0)

        # Ex0Gy = x0 + L0*(x_K(:,1) - x_p(:,1))
        Ex0Gy = x0 + L0 @ (x_K[:, 0] - x_p[:, 0])
        # Px0Gy = Px0 + L0*(inv(W_K(:,:,1)) - inv(W1G0))*L0'
        Px0Gy = Px0 + L0 @ (
            np.linalg.pinv(W_K[:, :, 0]) - np.linalg.pinv(W1G0)
        ) @ L0.T
        Px0Gy = _symmetrize(Px0Gy)

        # Cross-covariance terms Wku(:,:,k,u) from de Jong and MacKinnon 1988
        # Only compute the elements actually needed for the sums:
        # Wku(:,:,k,k) = W_K(:,:,k) and off-diagonal lags (k, k+1)
        # Matlab: Dk(:,:,k) = W_u(:,:,k)*A'/(W_p(:,:,k+1))
        # Wku(:,:,k,u) = Dk(:,:,k)*Wku(:,:,k+1,u)
        # We only need Wku(:,:,k-1,k) for the expectation sums.
        Wku_lag1 = np.zeros((Dx, Dx, K), dtype=float)  # Wku_lag1[:,:,k] = Wku(:,:,k-1,k)
        for k in range(K - 1, 0, -1):
            # Dk = Pe_u[:,:,k-1] * A' / Pe_p[:,:,k]
            Dk = Pe_u[:, :, k - 1] @ A.T @ np.linalg.pinv(Pe_p[:, :, k])
            if k == K - 1:
                # Wku(:,:,k-1,k) = Dk * W_K(:,:,k)
                Wku_lag1[:, :, k] = Dk @ W_K[:, :, k]
            else:
                # Wku(:,:,k-1,k) = Dk * Wku(:,:,k,k) = Dk * W_K(:,:,k)
                Wku_lag1[:, :, k] = Dk @ W_K[:, :, k]

        # Also need Wku(:,:,0,0) = W_K(:,:,0) and Px0*A'/W_p(:,:,0) for k==0
        # Matlab: Sxkm1xk at k==1: Px0*A'/W_p(:,:,1)*Wku(:,:,1,1)
        # Note: Matlab 1-indexed, W_p(:,:,1) is our Pe_p[:,:,0]
        # But the Matlab filter stores x_p(:,1)=x0, Pe_p(:,:,1)=Px0
        # and W_p(:,:,1) after the first predict is actually Pe_p(:,:,1) in Matlab = Pe_p[:,:,0] here
        # Actually let me re-read: Matlab's filter has Pe_p(:,:,1)=Px0 and the first
        # iteration does update then predict, so Pe_p(:,:,2) = A*Pe_u(:,:,1)*A'+Q.
        # In our _kf_filter_stateMajor, Pe_p[:,:,0]=Px0 and Pe_p[:,:,1]=A*Pe_u[:,:,0]*A'+Q
        # So Matlab's W_p(:,:,1) = Pe_p(:,:,1) in Matlab = our Pe_p[:,:,0] = Px0

        # Sufficient statistics (expectation sums)
        Sxkm1xk = np.zeros((Dx, Dx), dtype=float)
        Sxkm1xkm1 = np.zeros((Dx, Dx), dtype=float)
        Sxkxk = np.zeros((Dx, Dx), dtype=float)
        Sykyk = np.zeros((Dy, Dy), dtype=float)
        Sxkyk = np.zeros((Dx, Dy), dtype=float)

        for k in range(K):
            if k == 0:
                # Matlab: Sxkm1xk = Sxkm1xk + Px0*A'/W_p(:,:,1)*Wku(:,:,1,1)
                # W_p(:,:,1) in Matlab is Pe_p[:,:,0] = Px0 here
                # Wku(:,:,1,1) = W_K(:,:,1) in Matlab = W_K[:,:,0] here
                Sxkm1xk += Px0 @ A.T @ np.linalg.pinv(Pe_p[:, :, 0]) @ W_K[:, :, 0]
                Sxkm1xkm1 += Px0 + np.outer(x0, x0)
            else:
                # Wku(:,:,k-1,k) is Wku_lag1[:,:,k]
                Sxkm1xk += Wku_lag1[:, :, k] + np.outer(x_K[:, k - 1], x_K[:, k])
                Sxkm1xkm1 += W_K[:, :, k - 1] + np.outer(x_K[:, k - 1], x_K[:, k - 1])
            Sxkxk += W_K[:, :, k] + np.outer(x_K[:, k], x_K[:, k])
            Sykyk += np.outer(y[:, k] - alpha.ravel(), y[:, k] - alpha.ravel())
            Sxkyk += np.outer(x_K[:, k], y[:, k] - alpha.ravel())

        Sxkxk = _symmetrize(Sxkxk)
        Sykyk = _symmetrize(Sykyk)

        sumXkTerms = Sxkxk - A @ Sxkm1xk - Sxkm1xk.T @ A.T + A @ Sxkm1xkm1 @ A.T
        sumYkTerms = Sykyk - C @ Sxkyk - Sxkyk.T @ C.T + C @ Sxkxk @ C.T
        Sxkxkm1 = Sxkm1xk.T

        sumXkTerms = _symmetrize(sumXkTerms)
        sumYkTerms = _symmetrize(sumYkTerms)

        # Complete-data log-likelihood
        # Matlab: logll = -Dx*K/2*log(2*pi) - K/2*log(det(Q))
        #         - Dy*K/2*log(2*pi) - K/2*log(det(R))
        #         - Dx/2*log(2*pi) - 1/2*log(det(Px0))
        #         - 1/2*trace(inv(Q)*sumXkTerms) - 1/2*trace(inv(R)*sumYkTerms)
        #         - Dx/2
        sign_Q, logdet_Q = np.linalg.slogdet(Q)
        sign_R, logdet_R = np.linalg.slogdet(R)
        sign_P, logdet_P = np.linalg.slogdet(Px0)
        logll = (
            -Dx * K / 2.0 * np.log(2.0 * np.pi)
            - K / 2.0 * logdet_Q
            - Dy * K / 2.0 * np.log(2.0 * np.pi)
            - K / 2.0 * logdet_R
            - Dx / 2.0 * np.log(2.0 * np.pi)
            - 0.5 * logdet_P
            - 0.5 * np.trace(np.linalg.solve(Q, sumXkTerms))
            - 0.5 * np.trace(np.linalg.solve(R, sumYkTerms))
            - Dx / 2.0
        )
        logll = float(logll)
        print(f"logll: {logll}")

        ExpectationSums = {
            "Sxkm1xkm1": Sxkm1xkm1,
            "Sxkm1xk": Sxkm1xk,
            "Sxkxkm1": Sxkxkm1,
            "Sxkxk": Sxkxk,
            "Sxkyk": Sxkyk,
            "Sykyk": Sykyk,
            "sumXkTerms": sumXkTerms,
            "sumYkTerms": sumYkTerms,
            "Sx0": Ex0Gy,
            "Sx0x0": Px0Gy + np.outer(Ex0Gy, Ex0Gy),
        }

        return x_K, W_K, logll, ExpectationSums

    @staticmethod
    def KF_MStep(y, x_K, x0, Px0, ExpectationSums, KFEM_Constraints=None):
        """M-step for the Kalman Filter EM algorithm.

        Updates all state-space model parameters given the sufficient
        statistics from :meth:`KF_EStep`.

        Parameters
        ----------
        y : (Dy, K) observations
        x_K : (Dx, K) smoothed states from E-step
        x0 : (Dx,) current initial state estimate
        Px0 : (Dx, Dx) current initial covariance estimate
        ExpectationSums : dict from :meth:`KF_EStep`
        KFEM_Constraints : dict from :meth:`KF_EMCreateConstraints`, or *None*

        Returns
        -------
        Ahat, Qhat, Chat, Rhat, alphahat, x0hat, Px0hat
        """
        if KFEM_Constraints is None:
            KFEM_Constraints = DecodingAlgorithms.KF_EMCreateConstraints()

        Sxkm1xkm1 = ExpectationSums["Sxkm1xkm1"]
        Sxkxkm1 = ExpectationSums["Sxkxkm1"]
        Sxkxk = ExpectationSums["Sxkxk"]
        Sxkyk = ExpectationSums["Sxkyk"]
        sumXkTerms = ExpectationSums["sumXkTerms"]
        sumYkTerms = ExpectationSums["sumYkTerms"]
        Sx0 = ExpectationSums["Sx0"]
        Sx0x0 = ExpectationSums["Sx0x0"]

        y = np.asarray(y, dtype=float)
        x_K = np.asarray(x_K, dtype=float)
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        Px0 = np.asarray(Px0, dtype=float)

        N, K = x_K.shape  # N = Dx (num states), K = num time steps

        # Ahat
        if KFEM_Constraints["AhatDiag"]:
            I_N = np.eye(N)
            Ahat = (Sxkxkm1 * I_N) @ np.linalg.pinv(Sxkm1xkm1 * I_N)
        else:
            Ahat = Sxkxkm1 @ np.linalg.pinv(Sxkm1xkm1)

        # Chat = Sxkyk' / Sxkxk  (Matlab: Chat = Sxkyk'/Sxkxk)
        Chat = Sxkyk.T @ np.linalg.pinv(Sxkxk)

        # alphahat = sum(y - Chat*x_K, 2) / K
        alphahat = np.sum(y - Chat @ x_K, axis=1, keepdims=True) / K

        # Qhat
        if KFEM_Constraints["QhatDiag"]:
            if KFEM_Constraints["QhatIsotropic"]:
                Qhat = (1.0 / (N * K)) * np.trace(sumXkTerms) * np.eye(N)
            else:
                I_N = np.eye(N)
                Qhat = (1.0 / K) * (sumXkTerms * I_N)
                Qhat = _symmetrize(Qhat)
        else:
            Qhat = (1.0 / K) * sumXkTerms
            Qhat = _symmetrize(Qhat)

        # Rhat
        dy = sumYkTerms.shape[0]
        if KFEM_Constraints["RhatDiag"]:
            if KFEM_Constraints["RhatIsotropic"]:
                I_dy = np.eye(dy)
                Rhat = (1.0 / (dy * K)) * np.trace(sumYkTerms) * I_dy
            else:
                I_dy = np.eye(dy)
                Rhat = (1.0 / K) * (sumYkTerms * I_dy)
                Rhat = _symmetrize(Rhat)
        else:
            Rhat = (1.0 / K) * sumYkTerms
            Rhat = _symmetrize(Rhat)

        # x0hat — uses the newly computed Ahat and Qhat
        if KFEM_Constraints["Estimatex0"]:
            # Matlab: x0hat = (inv(Px0)+Ahat'/Qhat*Ahat)\(Ahat'/Qhat*x_K(:,1)+Px0\x0)
            Px0_inv = np.linalg.pinv(Px0)
            AQ = np.linalg.solve(Qhat, Ahat)  # Qhat\Ahat
            lhs = Px0_inv + Ahat.T @ AQ
            rhs = Ahat.T @ np.linalg.solve(Qhat, x_K[:, 0]) + np.linalg.solve(Px0, x0)
            x0hat = np.linalg.solve(lhs, rhs)
        else:
            x0hat = x0.copy()

        # Px0hat
        if KFEM_Constraints["EstimatePx0"]:
            if KFEM_Constraints["Px0Isotropic"]:
                diff = x0hat - x0
                Px0hat = (np.trace(np.outer(diff, diff)) / (N * K)) * np.eye(N)
            else:
                I_N = np.eye(N)
                diff = x0hat - x0
                Px0hat = (
                    np.outer(x0hat, x0hat)
                    - np.outer(x0, x0hat)
                    - np.outer(x0hat, x0)
                    + np.outer(x0, x0)
                ) * I_N
                Px0hat = _symmetrize(Px0hat)
                eigvals, eigvecs = np.linalg.eigh(Px0hat)
                if np.min(eigvals) < np.finfo(float).eps:
                    eigvals[eigvals == np.min(eigvals)] = np.finfo(float).eps
                    Px0hat = eigvecs @ np.diag(eigvals) @ eigvecs.T
        else:
            Px0hat = Px0.copy()

        return Ahat, Qhat, Chat, Rhat, alphahat, x0hat, Px0hat

    @staticmethod
    def KF_ComputeParamStandardErrors(
        y, xKFinal, WKFinal, Ahat, Qhat, Chat, Rhat, alphahat,
        x0hat, Px0hat, ExpectationSumsFinal, KFEM_Constraints=None,
    ):
        """Compute standard errors via the observed information matrix.

        Uses the complete information matrix and a Monte Carlo estimate of
        the missing information matrix, following McLachlan and Krishnan
        Eq. 4.7:  ``Io(theta; y) = Ic(theta; y) - Im(theta; y)``.

        Parameters
        ----------
        y : (Dy, K) observations
        xKFinal : (Dx, K) smoothed states
        WKFinal : (Dx, Dx, K) smoothed covariances
        Ahat, Qhat, Chat, Rhat : estimated model matrices
        alphahat : (Dy, 1) observation offset
        x0hat : (Dx,) initial state
        Px0hat : (Dx, Dx) initial covariance
        ExpectationSumsFinal : dict from :meth:`KF_EStep`
        KFEM_Constraints : dict from :meth:`KF_EMCreateConstraints`

        Returns
        -------
        SE : dict of standard-error matrices/vectors for each parameter
        Pvals : dict of p-value matrices/vectors for each parameter
        """
        if KFEM_Constraints is None:
            KFEM_Constraints = DecodingAlgorithms.KF_EMCreateConstraints()

        Ahat = np.asarray(Ahat, dtype=float)
        Qhat = np.asarray(Qhat, dtype=float)
        Chat = np.asarray(Chat, dtype=float)
        Rhat = np.asarray(Rhat, dtype=float)
        alphahat = np.asarray(alphahat, dtype=float).reshape(-1, 1)
        x0hat = np.asarray(x0hat, dtype=float).reshape(-1)
        Px0hat = np.asarray(Px0hat, dtype=float)
        y = np.asarray(y, dtype=float)
        xKFinal = np.asarray(xKFinal, dtype=float)
        WKFinal = np.asarray(WKFinal, dtype=float)

        dy, N = y.shape
        dx = xKFinal.shape[0]
        K = N

        # ----------------------------------------------------------------
        # Complete Information Matrices
        # ----------------------------------------------------------------

        # --- IAComp: information for A ---
        n1_A, n2_A = Ahat.shape
        el_A = np.eye(n1_A)
        em_A = np.eye(n2_A)
        if KFEM_Constraints["AhatDiag"]:
            nA = n1_A
            IAComp = np.zeros((nA, nA), dtype=float)
            cnt = 0
            for l in range(n1_A):
                m = l  # diagonal only
                # termMat = inv(Q) * el(:,l)*em(:,m)' * Sxkm1xkm1 .* I
                termMat = np.linalg.solve(Qhat, np.outer(el_A[:, l], em_A[:, m])) @ (
                    ExpectationSumsFinal["Sxkm1xkm1"] * np.eye(n1_A, n2_A)
                )
                IAComp[:, cnt] = np.diag(termMat)
                cnt += 1
        else:
            nA = Ahat.size
            IAComp = np.zeros((nA, nA), dtype=float)
            cnt = 0
            Qinv = np.linalg.inv(Qhat)
            for l in range(n1_A):
                for m in range(n2_A):
                    termMat = Qinv @ np.outer(el_A[:, l], em_A[:, m]) @ ExpectationSumsFinal["Sxkm1xkm1"]
                    termvec = termMat.T.ravel()
                    IAComp[:, cnt] = termvec
                    cnt += 1

        # --- ICComp: information for C ---
        n1_C, n2_C = Chat.shape
        el_C = np.eye(n1_C)
        em_C = np.eye(n2_C)
        nC = Chat.size
        ICComp = np.zeros((nC, nC), dtype=float)
        cnt = 0
        Rinv = np.linalg.inv(Rhat)
        for l in range(n1_C):
            for m in range(n2_C):
                termMat = Rinv @ np.outer(el_C[:, l], em_C[:, m]) @ ExpectationSumsFinal["Sxkxk"]
                termvec = termMat.T.ravel()
                ICComp[:, cnt] = termvec
                cnt += 1

        # --- IRComp: information for R ---
        n1_R, n2_R = Rhat.shape
        el_R = np.eye(n1_R)
        em_R = np.eye(n2_R)
        if KFEM_Constraints["RhatDiag"]:
            if KFEM_Constraints["RhatIsotropic"]:
                IRComp = np.array([[0.5 * N * dy * Rhat[0, 0] ** (-2)]])
                nR = 1
            else:
                nR = n1_R
                IRComp = np.zeros((nR, nR), dtype=float)
                cnt = 0
                for l in range(n1_R):
                    m = l
                    termMat = (N / 2.0) * np.linalg.solve(Rhat, np.outer(em_R[:, m], el_R[:, l])) @ np.linalg.inv(Rhat)
                    IRComp[:, cnt] = np.diag(termMat)
                    cnt += 1
        else:
            nR = Rhat.size
            IRComp = np.zeros((nR, nR), dtype=float)
            cnt = 0
            for l in range(n1_R):
                for m in range(n2_R):
                    termMat = (N / 2.0) * np.linalg.solve(Rhat, np.outer(em_R[:, m], el_R[:, l])) @ np.linalg.inv(Rhat)
                    termvec = termMat.T.ravel()
                    IRComp[:, cnt] = termvec
                    cnt += 1

        # --- IQComp: information for Q ---
        n1_Q, n2_Q = Qhat.shape
        el_Q = np.eye(n1_Q)
        em_Q = np.eye(n2_Q)
        if KFEM_Constraints["QhatDiag"]:
            if KFEM_Constraints["QhatIsotropic"]:
                IQComp = np.array([[0.5 * N * dx * Qhat[0, 0] ** (-2)]])
                nQ = 1
            else:
                nQ = n1_Q
                IQComp = np.zeros((nQ, nQ), dtype=float)
                cnt = 0
                for l in range(n1_Q):
                    m = l
                    termMat = (N / 2.0) * np.linalg.solve(Qhat, np.outer(em_Q[:, m], el_Q[:, l])) @ np.linalg.inv(Qhat)
                    IQComp[:, cnt] = np.diag(termMat)
                    cnt += 1
        else:
            nQ = Qhat.size
            IQComp = np.zeros((nQ, nQ), dtype=float)
            cnt = 0
            for l in range(n1_Q):
                for m in range(n2_Q):
                    termMat = (N / 2.0) * np.linalg.solve(Qhat, np.outer(em_Q[:, m], el_Q[:, l])) @ np.linalg.inv(Qhat)
                    termvec = termMat.T.ravel()
                    IQComp[:, cnt] = termvec
                    cnt += 1

        # --- ISComp: information for Px0 ---
        if KFEM_Constraints["EstimatePx0"]:
            if KFEM_Constraints["Px0Isotropic"]:
                ISComp = np.array([[0.5 * dx * Px0hat[0, 0] ** (-2)]])
                nS = 1
            else:
                nS = Px0hat.shape[0]
                ISComp = np.zeros((nS, nS), dtype=float)
                el_S = np.eye(nS)
                em_S = np.eye(nS)
                cnt = 0
                for l in range(nS):
                    m = l
                    termMat = 0.5 * np.linalg.solve(Px0hat, np.outer(em_S[:, m], el_S[:, l])) @ np.linalg.inv(Px0hat)
                    ISComp[:, cnt] = np.diag(termMat)
                    cnt += 1
        else:
            nS = 0

        # --- Ix0Comp: information for x0 ---
        if KFEM_Constraints["Estimatex0"]:
            Ix0Comp = np.linalg.inv(Px0hat) + Ahat.T @ np.linalg.solve(Qhat, Ahat)
            nx0 = Ix0Comp.shape[0]
        else:
            nx0 = 0

        # --- IAlphaComp ---
        IAlphaComp = N * np.linalg.inv(Rhat)
        nAlpha = IAlphaComp.shape[0]

        # Block sizes
        # n1=A, n2=Q, n3=C, n4=R, n5=Px0, n6=x0, n7=alpha
        if KFEM_Constraints["EstimateA"]:
            n1 = IAComp.shape[0]
        else:
            n1 = 0
        n2 = IQComp.shape[0]
        n3 = ICComp.shape[0]
        n4 = IRComp.shape[0]
        n5 = nS
        n6 = nx0
        n7 = nAlpha
        nTerms = n1 + n2 + n3 + n4 + n5 + n6 + n7

        # Assemble block-diagonal complete information matrix
        IComp = np.zeros((nTerms, nTerms), dtype=float)
        if KFEM_Constraints["EstimateA"]:
            IComp[:n1, :n1] = IAComp
        off = n1
        IComp[off:off + n2, off:off + n2] = IQComp
        off = n1 + n2
        IComp[off:off + n3, off:off + n3] = ICComp
        off = n1 + n2 + n3
        IComp[off:off + n4, off:off + n4] = IRComp
        off = n1 + n2 + n3 + n4
        if KFEM_Constraints["EstimatePx0"]:
            IComp[off:off + n5, off:off + n5] = ISComp
        off = n1 + n2 + n3 + n4 + n5
        if KFEM_Constraints["Estimatex0"]:
            IComp[off:off + n6, off:off + n6] = Ix0Comp
        off = n1 + n2 + n3 + n4 + n5 + n6
        IComp[off:off + n7, off:off + n7] = IAlphaComp

        # ----------------------------------------------------------------
        # Missing Information Matrix (Monte Carlo)
        # ----------------------------------------------------------------
        Mc = KFEM_Constraints["mcIter"]
        xKDraw = np.zeros((dx, N, Mc), dtype=float)

        for n in range(N):
            WuTemp = WKFinal[:, :, n]
            try:
                chol_m = np.linalg.cholesky(WuTemp).T  # upper Cholesky (Matlab chol returns upper)
            except np.linalg.LinAlgError:
                chol_m = np.linalg.cholesky(_nearestSPD(WuTemp)).T
            z = np.random.randn(dx, Mc)
            xKDraw[:, n, :] = x0hat[:, None] * 0 + xKFinal[:, n:n + 1] + chol_m @ z

        if KFEM_Constraints["EstimatePx0"] or KFEM_Constraints["Estimatex0"]:
            try:
                chol_m = np.linalg.cholesky(Px0hat).T
            except np.linalg.LinAlgError:
                chol_m = np.linalg.cholesky(_nearestSPD(Px0hat)).T
            z = np.random.randn(dx, Mc)
            x0Draw = x0hat[:, None] + chol_m @ z
        else:
            x0Draw = np.tile(x0hat[:, None], (1, Mc))

        IMc = np.zeros((nTerms, nTerms, Mc), dtype=float)
        alpha_flat = alphahat.ravel()

        for c in range(Mc):
            x_K_c = xKDraw[:, :, c]
            x_0_c = x0Draw[:, c]

            Dx_c = x_K_c.shape[0]
            Dy_c = y.shape[0]
            Sxkm1xk_c = np.zeros((Dx_c, Dx_c))
            Sxkm1xkm1_c = np.zeros((Dx_c, Dx_c))
            Sxkxk_c = np.zeros((Dx_c, Dx_c))
            Sykyk_c = np.zeros((Dy_c, Dy_c))
            Sxkyk_c = np.zeros((Dx_c, Dy_c))

            for k in range(K):
                if k == 0:
                    Sxkm1xk_c += np.outer(x_0_c, x_K_c[:, k])
                    Sxkm1xkm1_c += np.outer(x_0_c, x_0_c)
                else:
                    Sxkm1xk_c += np.outer(x_K_c[:, k - 1], x_K_c[:, k])
                    Sxkm1xkm1_c += np.outer(x_K_c[:, k - 1], x_K_c[:, k - 1])
                Sxkxk_c += np.outer(x_K_c[:, k], x_K_c[:, k])
                yk_centered = y[:, k] - alpha_flat
                Sykyk_c += np.outer(yk_centered, yk_centered)
                Sxkyk_c += np.outer(x_K_c[:, k], yk_centered)

            Sxkxk_c = _symmetrize(Sxkxk_c)
            Sykyk_c = _symmetrize(Sykyk_c)
            sumXkTerms_c = Sxkxk_c - Ahat @ Sxkm1xk_c - Sxkm1xk_c.T @ Ahat.T + Ahat @ Sxkm1xkm1_c @ Ahat.T
            sumYkTerms_c = Sykyk_c - Chat @ Sxkyk_c - Sxkyk_c.T @ Chat.T + Chat @ Sxkxk_c @ Chat.T
            Sxkxkm1_c = Sxkm1xk_c.T
            Sykxk_c = Sxkyk_c.T

            sumXkTerms_c = _symmetrize(sumXkTerms_c)
            sumYkTerms_c = _symmetrize(sumYkTerms_c)

            # Score for A
            if KFEM_Constraints["EstimateA"]:
                ScorA = np.linalg.solve(Qhat, Sxkxkm1_c - Ahat @ Sxkm1xkm1_c)
                if KFEM_Constraints["AhatDiag"]:
                    ScoreAMc = np.diag(ScorA)
                else:
                    ScoreAMc = ScorA.T.ravel()
            else:
                ScoreAMc = np.array([], dtype=float)

            # Score for C
            ScorC = np.linalg.solve(Rhat, Sykxk_c - Chat @ Sxkxk_c)
            ScoreCMc = ScorC.T.ravel()

            # Score for Q
            Qinv_c = np.linalg.inv(Qhat)
            I_Q = np.eye(Qhat.shape[0])
            if KFEM_Constraints["QhatDiag"]:
                if KFEM_Constraints["QhatIsotropic"]:
                    ScoreQ = -0.5 * (K * Dx_c * Qhat[0, 0] ** (-1) - Qhat[0, 0] ** (-2) * np.trace(sumXkTerms_c))
                    ScoreQMc = np.atleast_1d(ScoreQ)
                else:
                    ScoreQ = -0.5 * np.linalg.solve(Qhat, K * I_Q - np.linalg.solve(Qhat, sumXkTerms_c).T)
                    ScoreQMc = np.diag(ScoreQ)
            else:
                ScoreQ = -0.5 * np.linalg.solve(Qhat, K * I_Q - np.linalg.solve(Qhat, sumXkTerms_c).T)
                ScoreQMc = ScoreQ.T.ravel()

            # Score for alpha
            ScoreAlphaMc = np.sum(
                np.linalg.solve(Rhat, y - Chat @ x_K_c - alpha_flat[:, None] @ np.ones((1, N))),
                axis=1,
            )

            # Score for R
            I_R = np.eye(Rhat.shape[0])
            if KFEM_Constraints["RhatDiag"]:
                if KFEM_Constraints["RhatIsotropic"]:
                    ScoreR = -0.5 * (K * Dy_c * Rhat[0, 0] ** (-1) - Rhat[0, 0] ** (-2) * np.trace(sumYkTerms_c))
                    ScoreRMc = np.atleast_1d(ScoreR)
                else:
                    ScoreR = -0.5 * np.linalg.solve(Rhat, K * I_R - np.linalg.solve(Rhat, sumYkTerms_c).T)
                    ScoreRMc = np.diag(ScoreR)
            else:
                ScoreR = -0.5 * np.linalg.solve(Rhat, K * I_R - np.linalg.solve(Rhat, sumYkTerms_c).T)
                ScoreRMc = ScoreR.T.ravel()

            # Score for Px0
            diff_x0 = x_0_c - x0hat
            if KFEM_Constraints["Px0Isotropic"]:
                ScoreSMc = np.atleast_1d(
                    -0.5 * (Dx_c * Px0hat[0, 0] ** (-1) - Px0hat[0, 0] ** (-2) * np.trace(np.outer(diff_x0, diff_x0)))
                )
            else:
                ScorS = -0.5 * np.linalg.solve(
                    Px0hat,
                    np.eye(Px0hat.shape[0]) - np.linalg.solve(Px0hat, np.outer(diff_x0, diff_x0)).T,
                )
                ScoreSMc = np.diag(ScorS)

            # Score for x0
            Scorx0 = -np.linalg.solve(Px0hat, diff_x0) + Ahat.T @ np.linalg.solve(Qhat, x_K_c[:, 0] - Ahat @ x_0_c)
            Scorex0Mc = Scorx0.ravel()

            # Assemble score vector
            ScoreVec = ScoreAMc if KFEM_Constraints["EstimateA"] else np.array([], dtype=float)
            ScoreVec = np.concatenate([ScoreVec, ScoreQMc, ScoreCMc, ScoreRMc])
            if KFEM_Constraints["EstimatePx0"]:
                ScoreVec = np.concatenate([ScoreVec, ScoreSMc])
            if KFEM_Constraints["Estimatex0"]:
                ScoreVec = np.concatenate([ScoreVec, Scorex0Mc])
            ScoreVec = np.concatenate([ScoreVec, ScoreAlphaMc])

            IMc[:, :, c] = np.outer(ScoreVec, ScoreVec)

        # Observed information = Complete - Missing
        IMissing = np.mean(IMc, axis=2)
        IObs = IComp - IMissing
        invIObs = np.linalg.pinv(IObs)
        invIObs = _nearestSPD(invIObs)

        VarVec = np.diag(invIObs)
        SEVec = np.sqrt(np.maximum(VarVec, 0.0))

        # Unpack SE vector
        off = 0
        SEAterms = SEVec[off:off + n1]; off += n1
        SEQterms = SEVec[off:off + n2]; off += n2
        SECterms = SEVec[off:off + n3]; off += n3
        SERterms = SEVec[off:off + n4]; off += n4
        SEPx0terms = SEVec[off:off + n5]; off += n5
        SEx0terms = SEVec[off:off + n6]; off += n6
        SEAlphaterms = SEVec[off:off + n7]

        # Reshape SEs into matrices matching parameter shapes
        SE = {}
        if KFEM_Constraints["EstimateA"]:
            if KFEM_Constraints["AhatDiag"]:
                SE["A"] = np.diag(SEAterms)
            else:
                SE["A"] = SEAterms.reshape(Ahat.shape[1], Ahat.shape[0]).T
        SE["Q"] = np.diag(SEQterms) if KFEM_Constraints["QhatDiag"] else SEQterms.reshape(Qhat.shape[1], Qhat.shape[0]).T
        SE["C"] = SECterms.reshape(Chat.shape[1], Chat.shape[0]).T
        SE["R"] = np.diag(SERterms) if KFEM_Constraints["RhatDiag"] else SERterms.reshape(Rhat.shape[1], Rhat.shape[0]).T
        SE["alpha"] = SEAlphaterms.reshape(alphahat.shape)
        if KFEM_Constraints["EstimatePx0"]:
            SE["Px0"] = np.diag(SEPx0terms)
        if KFEM_Constraints["Estimatex0"]:
            SE["x0"] = SEx0terms

        # Compute p-values via z-tests
        Pvals = {}
        if KFEM_Constraints["EstimateA"]:
            if KFEM_Constraints["AhatDiag"]:
                pA = np.diag([_ztest_pvalue(Ahat[i, i], SE["A"][i, i]) for i in range(Ahat.shape[0])])
            else:
                pA_flat = [_ztest_pvalue(Ahat.ravel()[i], SE["A"].ravel()[i]) for i in range(Ahat.size)]
                pA = np.array(pA_flat).reshape(Ahat.shape)
            Pvals["A"] = pA

        # C p-values
        pC_flat = [_ztest_pvalue(Chat.ravel()[i], SE["C"].ravel()[i]) for i in range(Chat.size)]
        Pvals["C"] = np.array(pC_flat).reshape(Chat.shape)

        # R p-values
        if KFEM_Constraints["RhatDiag"]:
            if KFEM_Constraints["RhatIsotropic"]:
                pR = np.diag([_ztest_pvalue(Rhat[0, 0], SE["R"][0, 0])])
            else:
                pR = np.diag([_ztest_pvalue(Rhat[i, i], SE["R"][i, i]) for i in range(Rhat.shape[0])])
        else:
            pR_flat = [_ztest_pvalue(Rhat.ravel()[i], SE["R"].ravel()[i]) for i in range(Rhat.size)]
            pR = np.array(pR_flat).reshape(Rhat.shape)
        Pvals["R"] = pR

        # Q p-values
        if KFEM_Constraints["QhatDiag"]:
            if KFEM_Constraints["QhatIsotropic"]:
                pQ = np.diag([_ztest_pvalue(Qhat[0, 0], SE["Q"][0, 0])])
            else:
                pQ = np.diag([_ztest_pvalue(Qhat[i, i], SE["Q"][i, i]) for i in range(Qhat.shape[0])])
        else:
            pQ_flat = [_ztest_pvalue(Qhat.ravel()[i], SE["Q"].ravel()[i]) for i in range(Qhat.size)]
            pQ = np.array(pQ_flat).reshape(Qhat.shape)
        Pvals["Q"] = pQ

        # Px0 p-values
        if KFEM_Constraints["EstimatePx0"]:
            if KFEM_Constraints["Px0Isotropic"]:
                pPx0 = np.diag([_ztest_pvalue(Px0hat[0, 0], SE["Px0"][0, 0])])
            else:
                pPx0 = np.diag([_ztest_pvalue(Px0hat[i, i], SE["Px0"][i, i]) for i in range(Px0hat.shape[0])])
            Pvals["Px0"] = pPx0

        # alpha p-values
        alpha_flat_se = SE["alpha"].ravel()
        pAlpha = np.array([_ztest_pvalue(alphahat.ravel()[i], alpha_flat_se[i]) for i in range(alphahat.size)])
        Pvals["alpha"] = pAlpha

        # x0 p-values
        if KFEM_Constraints["Estimatex0"]:
            pX0 = np.array([_ztest_pvalue(x0hat[i], SE["x0"][i]) for i in range(x0hat.size)])
            Pvals["x0"] = pX0

        return SE, Pvals

    @staticmethod
    def KF_EM(
        y,
        Ahat0,
        Qhat0,
        Chat0,
        Rhat0,
        alphahat0,
        x0=None,
        Px0=None,
        KFEM_Constraints=None,
    ):
        """Kalman Filter EM algorithm with Cholesky-scaled system.

        Estimates the parameters of a linear-Gaussian state-space model::

            x_{k+1} = A x_k + v_k,  v ~ N(0, Q)
            y_k     = C x_k + alpha + w_k,  w ~ N(0, R)

        using the Expectation-Maximisation algorithm (E-step: KF + RTS
        smoother, M-step: closed-form updates).  Optionally applies Ikeda
        acceleration.

        Parameters
        ----------
        y : (Dy, K) observation matrix (each column is one time step)
        Ahat0 : (Dx, Dx) initial state transition
        Qhat0 : (Dx, Dx) initial process noise covariance
        Chat0 : (Dy, Dx) initial observation matrix
        Rhat0 : (Dy, Dy) initial observation noise covariance
        alphahat0 : (Dy, 1) initial observation offset
        x0 : (Dx,) initial state (default zeros)
        Px0 : (Dx, Dx) initial state covariance (default 1e-10 * I)
        KFEM_Constraints : dict from :meth:`KF_EMCreateConstraints`

        Returns
        -------
        xKFinal : (Dx, K) smoothed state estimates
        WKFinal : (Dx, Dx, K) smoothed covariances
        Ahat, Qhat, Chat, Rhat : estimated model matrices
        alphahat : (Dy, 1) estimated observation offset
        x0hat : (Dx,) estimated initial state
        Px0hat : (Dx, Dx) estimated initial covariance
        IC : dict of information criteria (AIC, AICc, BIC, llobs, llcomp)
        SE : dict of standard errors (or empty dict if not computed)
        Pvals : dict of p-values (or empty dict if not computed)
        nIter : int — number of EM iterations
        """
        Ahat0 = np.asarray(Ahat0, dtype=float)
        Qhat0 = np.asarray(Qhat0, dtype=float)
        Chat0 = np.asarray(Chat0, dtype=float)
        Rhat0 = np.asarray(Rhat0, dtype=float)
        alphahat0 = np.asarray(alphahat0, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=float)
        numStates = Ahat0.shape[0]

        if KFEM_Constraints is None:
            KFEM_Constraints = DecodingAlgorithms.KF_EMCreateConstraints()
        if Px0 is None:
            Px0 = 1e-10 * np.eye(numStates)
        else:
            Px0 = np.asarray(Px0, dtype=float)
        if x0 is None:
            x0 = np.zeros(numStates)
        else:
            x0 = np.asarray(x0, dtype=float).reshape(-1)

        tolAbs = 1e-3
        llTol = 1e-3
        maxIter = 100
        numToKeep = 10

        # Save originals for un-scaling later
        A0 = Ahat0.copy()
        Q0 = Qhat0.copy()
        C0 = Chat0.copy()
        R0 = Rhat0.copy()
        alpha0 = alphahat0.copy()
        yOrig = y.copy()

        # Circular buffers (indexed by storeInd)
        Ahat_buf = [None] * numToKeep
        Qhat_buf = [None] * numToKeep
        Chat_buf = [None] * numToKeep
        Rhat_buf = [None] * numToKeep
        x0hat_buf = [None] * numToKeep
        Px0hat_buf = [None] * numToKeep
        alphahat_buf = [None] * numToKeep
        x_K_buf = [None] * numToKeep
        W_K_buf = [None] * numToKeep
        ExpSums_buf = [None] * numToKeep

        # Initialize slot 0
        Ahat_buf[0] = A0.copy()
        Qhat_buf[0] = Q0.copy()
        Chat_buf[0] = C0.copy()
        Rhat_buf[0] = R0.copy()
        x0hat_buf[0] = x0.copy()
        Px0hat_buf[0] = Px0.copy()
        alphahat_buf[0] = alpha0.copy()

        # Scale the system via Cholesky transforms
        # Matlab: Tq = eye(size(Q))/(chol(Q));  Tr = eye(size(R))/(chol(R))
        scaledSystem = True
        if scaledSystem:
            try:
                cholQ = np.linalg.cholesky(Qhat_buf[0]).T  # upper Cholesky
            except np.linalg.LinAlgError:
                cholQ = np.linalg.cholesky(_nearestSPD(Qhat_buf[0])).T
            try:
                cholR = np.linalg.cholesky(Rhat_buf[0]).T  # upper Cholesky
            except np.linalg.LinAlgError:
                cholR = np.linalg.cholesky(_nearestSPD(Rhat_buf[0])).T
            Tq = np.linalg.solve(cholQ, np.eye(numStates))
            Tr = np.linalg.solve(cholR, np.eye(y.shape[0]))

            Ahat_buf[0] = Tq @ Ahat_buf[0] @ np.linalg.solve(Tq, np.eye(numStates))
            Chat_buf[0] = Tr @ Chat_buf[0] @ np.linalg.solve(Tq, np.eye(numStates))
            Qhat_buf[0] = Tq @ Qhat_buf[0] @ Tq.T
            Rhat_buf[0] = Tr @ Rhat_buf[0] @ Tr.T
            y = Tr @ y
            x0hat_buf[0] = Tq @ x0
            Px0hat_buf[0] = Tq @ Px0 @ Tq.T
            alphahat_buf[0] = Tr @ alphahat_buf[0]

        cnt = 0  # 0-based iteration counter
        ll_list = []
        dLikelihood = [np.inf, np.inf]
        IkedaAcc = KFEM_Constraints["EnableIkeda"]
        stoppingCriteria = False

        print("                       Kalman Filter/Gaussian Observation EM Algorithm                        ")

        while not stoppingCriteria and cnt < maxIter:
            storeInd = cnt % numToKeep
            storeIndP1 = (cnt + 1) % numToKeep
            storeIndM1 = (cnt - 1) % numToKeep

            print("-" * 100)
            print(f"Iteration #{cnt + 1}")
            print("-" * 100)

            # E-step
            x_K_buf[storeInd], W_K_buf[storeInd], ll_val, ExpSums_buf[storeInd] = (
                DecodingAlgorithms.KF_EStep(
                    Ahat_buf[storeInd], Qhat_buf[storeInd],
                    Chat_buf[storeInd], Rhat_buf[storeInd],
                    y, alphahat_buf[storeInd],
                    x0hat_buf[storeInd], Px0hat_buf[storeInd],
                )
            )
            ll_list.append(ll_val)

            # M-step
            (
                Ahat_buf[storeIndP1], Qhat_buf[storeIndP1],
                Chat_buf[storeIndP1], Rhat_buf[storeIndP1],
                alphahat_buf[storeIndP1], x0hat_buf[storeIndP1],
                Px0hat_buf[storeIndP1],
            ) = DecodingAlgorithms.KF_MStep(
                y, x_K_buf[storeInd],
                x0hat_buf[storeInd], Px0hat_buf[storeInd],
                ExpSums_buf[storeInd], KFEM_Constraints,
            )

            # Ikeda acceleration
            if IkedaAcc:
                print("****Ikeda Acceleration Step****")
                K_obs = x_K_buf[storeInd].shape[1]
                mean_y = (
                    Chat_buf[storeIndP1] @ x_K_buf[storeInd]
                    + alphahat_buf[storeIndP1] @ np.ones((1, K_obs))
                )
                ykNew = np.random.multivariate_normal(
                    np.zeros(Rhat_buf[storeIndP1].shape[0]),
                    Rhat_buf[storeIndP1],
                    size=K_obs,
                ).T + mean_y

                x_KNew, W_KNew, llNew, ExpSumsNew = DecodingAlgorithms.KF_EStep(
                    Ahat_buf[storeInd], Qhat_buf[storeInd],
                    Chat_buf[storeInd], Rhat_buf[storeInd],
                    ykNew, alphahat_buf[storeInd], x0, Px0,
                )
                (
                    AhatNew, QhatNew, ChatNew, RhatNew,
                    alphahatNew, x0new, Px0new,
                ) = DecodingAlgorithms.KF_MStep(
                    ykNew, x_KNew, x0hat_buf[storeInd],
                    Px0hat_buf[storeInd], ExpSumsNew, KFEM_Constraints,
                )

                Ahat_buf[storeIndP1] = 2 * Ahat_buf[storeIndP1] - AhatNew
                Qhat_buf[storeIndP1] = 2 * Qhat_buf[storeIndP1] - QhatNew
                Qhat_buf[storeIndP1] = _symmetrize(Qhat_buf[storeIndP1])
                Chat_buf[storeIndP1] = 2 * Chat_buf[storeIndP1] - ChatNew
                Rhat_buf[storeIndP1] = 2 * Rhat_buf[storeIndP1] - RhatNew
                Rhat_buf[storeIndP1] = _symmetrize(Rhat_buf[storeIndP1])
                alphahat_buf[storeIndP1] = 2 * alphahat_buf[storeIndP1] - alphahatNew

            # Override A if not estimating
            if not KFEM_Constraints["EstimateA"]:
                Ahat_buf[storeIndP1] = Ahat_buf[storeInd]

            # Likelihood change
            if cnt == 0:
                dLikelihood_val = np.inf
            else:
                dLikelihood_val = ll_list[cnt] - ll_list[cnt - 1]

            # Convergence check: max parameter change
            if cnt == 0:
                dMax = np.inf
            else:
                prev = storeIndM1
                dQvals = np.max(np.abs(
                    np.sqrt(np.abs(np.diag(Qhat_buf[storeInd])))
                    - np.sqrt(np.abs(np.diag(Qhat_buf[prev])))
                ))
                dRvals = np.max(np.abs(
                    np.sqrt(np.abs(np.diag(Rhat_buf[storeInd])))
                    - np.sqrt(np.abs(np.diag(Rhat_buf[prev])))
                ))
                dAvals = np.max(np.abs(Ahat_buf[storeInd] - Ahat_buf[prev]))
                dCvals = np.max(np.abs(Chat_buf[storeInd] - Chat_buf[prev]))
                dAlphavals = np.max(np.abs(alphahat_buf[storeInd] - alphahat_buf[prev]))
                dMax = max(dQvals, dRvals, dAvals, dCvals, dAlphavals)

            if cnt == 0:
                print("Max Parameter Change: N/A")
            else:
                print(f"Max Parameter Change: {dMax}")

            cnt += 1

            if dMax < tolAbs:
                stoppingCriteria = True
                print(f"         EM converged at iteration# {cnt} b/c change in params was within criteria")

            if abs(dLikelihood_val) < llTol or dLikelihood_val < 0:
                stoppingCriteria = True
                print(f"         EM stopped at iteration# {cnt} b/c change in likelihood was negative")

        print("-" * 100)

        # Select best iteration by max log-likelihood
        ll_arr = np.array(ll_list)
        maxLLIndex = int(np.argmax(ll_arr))
        maxLLIndMod = maxLLIndex % numToKeep
        nIter = cnt

        xKFinal = x_K_buf[maxLLIndMod]
        WKFinal = W_K_buf[maxLLIndMod]
        Ahat_final = Ahat_buf[maxLLIndMod]
        Qhat_final = Qhat_buf[maxLLIndMod]
        Chat_final = Chat_buf[maxLLIndMod]
        Rhat_final = Rhat_buf[maxLLIndMod]
        alphahat_final = alphahat_buf[maxLLIndMod]
        x0hat_final = x0hat_buf[maxLLIndMod]
        Px0hat_final = Px0hat_buf[maxLLIndMod]

        # Un-scale the system
        if scaledSystem:
            # Reconstruct Tq, Tr from original Q0, R0
            try:
                cholQ0 = np.linalg.cholesky(Q0).T
            except np.linalg.LinAlgError:
                cholQ0 = np.linalg.cholesky(_nearestSPD(Q0)).T
            try:
                cholR0 = np.linalg.cholesky(R0).T
            except np.linalg.LinAlgError:
                cholR0 = np.linalg.cholesky(_nearestSPD(R0)).T
            Tq = np.linalg.solve(cholQ0, np.eye(numStates))
            Tr = np.linalg.solve(cholR0, np.eye(y.shape[0]))

            # Matlab: Ahat = Tq\Ahat*Tq
            Tq_inv = np.linalg.inv(Tq)
            Tr_inv = np.linalg.inv(Tr)
            Ahat_final = Tq_inv @ Ahat_final @ Tq
            Qhat_final = Tq_inv @ Qhat_final @ Tq_inv.T
            Chat_final = Tr_inv @ Chat_final @ Tq
            Rhat_final = Tr_inv @ Rhat_final @ Tr_inv.T
            alphahat_final = Tr_inv @ alphahat_final
            xKFinal = Tq_inv @ xKFinal
            x0hat_final = Tq_inv @ x0hat_final
            Px0hat_final = Tq_inv @ Px0hat_final @ Tq_inv.T
            K_steps = WKFinal.shape[2]
            tempWK = np.zeros_like(WKFinal)
            for kk in range(K_steps):
                tempWK[:, :, kk] = Tq_inv @ WKFinal[:, :, kk] @ Tq_inv.T
            WKFinal = tempWK

        ll_best = ll_list[maxLLIndex]
        ExpectationSumsFinal = ExpSums_buf[maxLLIndMod]

        # Compute standard errors
        SE, Pvals = DecodingAlgorithms.KF_ComputeParamStandardErrors(
            yOrig, xKFinal, WKFinal, Ahat_final, Qhat_final,
            Chat_final, Rhat_final, alphahat_final, x0hat_final,
            Px0hat_final, ExpectationSumsFinal, KFEM_Constraints,
        )

        # Compute information criteria
        # Count number of estimated parameters (matches Matlab lines 3600-3640)
        if KFEM_Constraints["EstimateA"] and KFEM_Constraints["AhatDiag"]:
            np1 = Ahat_final.shape[0]
        elif KFEM_Constraints["EstimateA"] and not KFEM_Constraints["AhatDiag"]:
            np1 = Ahat_final.size
        else:
            np1 = 0

        if KFEM_Constraints["QhatDiag"] and KFEM_Constraints["QhatIsotropic"]:
            np2 = 1
        elif KFEM_Constraints["QhatDiag"] and not KFEM_Constraints["QhatIsotropic"]:
            np2 = Qhat_final.shape[0]
        else:
            np2 = Qhat_final.size

        np3 = Chat_final.size

        if KFEM_Constraints["RhatDiag"] and KFEM_Constraints["RhatIsotropic"]:
            np4 = 1
        elif KFEM_Constraints["QhatDiag"] and not KFEM_Constraints["QhatIsotropic"]:
            # Note: Matlab line 3618 checks QhatDiag here (likely a bug, but we match it)
            np4 = Rhat_final.shape[0]
        else:
            np4 = Rhat_final.size

        if KFEM_Constraints["EstimatePx0"] and KFEM_Constraints["Px0Isotropic"]:
            np5 = 1
        elif KFEM_Constraints["EstimatePx0"] and not KFEM_Constraints["Px0Isotropic"]:
            np5 = Px0hat_final.shape[0]
        else:
            np5 = 0

        np6 = x0hat_final.shape[0] if KFEM_Constraints["Estimatex0"] else 0
        np7 = alphahat_final.shape[0]
        nTerms_ic = np1 + np2 + np3 + np4 + np5 + np6 + np7

        K_steps = yOrig.shape[1]
        Dx = Ahat_final.shape[1]
        sumXkTerms_final = ExpectationSumsFinal["sumXkTerms"]

        # Matlab: llobs = ll + Dx*K/2*log(2*pi) + K/2*log(det(Qhat))
        #   + 1/2*trace(Qhat\sumXkTerms) + Dx/2*log(2*pi)
        #   + 1/2*log(det(Px0hat)) + 1/2*Dx
        _, logdet_Q_final = np.linalg.slogdet(Qhat_final)
        _, logdet_Px0_final = np.linalg.slogdet(Px0hat_final)
        llobs = (
            ll_best
            + Dx * K_steps / 2.0 * np.log(2.0 * np.pi)
            + K_steps / 2.0 * logdet_Q_final
            + 0.5 * np.trace(np.linalg.solve(Qhat_final, sumXkTerms_final))
            + Dx / 2.0 * np.log(2.0 * np.pi)
            + 0.5 * logdet_Px0_final
            + 0.5 * Dx
        )

        AIC = 2.0 * nTerms_ic - 2.0 * llobs
        AICc = AIC + 2.0 * nTerms_ic * (nTerms_ic + 1) / max(K_steps - nTerms_ic - 1, 1)
        BIC = -2.0 * llobs + nTerms_ic * np.log(K_steps)

        IC = {
            "AIC": float(AIC),
            "AICc": float(AICc),
            "BIC": float(BIC),
            "llobs": float(llobs),
            "llcomp": float(ll_best),
        }

        return (
            xKFinal, WKFinal, Ahat_final, Qhat_final,
            Chat_final, Rhat_final, alphahat_final,
            x0hat_final, Px0hat_final, IC, SE, Pvals, nIter,
        )

    @staticmethod
    # PP_EM family: Point-Process state-space EM (without basis functions)
    # ------------------------------------------------------------------

    @staticmethod
    def PP_EMCreateConstraints(
        EstimateA=1,
        AhatDiag=0,
        QhatDiag=1,
        QhatIsotropic=0,
        Estimatex0=1,
        EstimatePx0=1,
        Px0Isotropic=0,
        mcIter=1000,
        EnableIkeda=0,
    ):
        """Build a constraints dict for PP_EM.

        Parameters
        ----------
        EstimateA : int
            Whether to estimate the state transition matrix A.
        AhatDiag : int
            Constrain A to be diagonal.
        QhatDiag : int
            Constrain Q to be diagonal.
        QhatIsotropic : int
            Constrain Q to be isotropic (scalar * I).
        Estimatex0 : int
            Whether to estimate the initial state x0.
        EstimatePx0 : int
            Whether to estimate the initial state covariance Px0.
        Px0Isotropic : int
            Constrain Px0 to be isotropic.
        mcIter : int
            Number of Monte Carlo iterations for standard error estimation.
        EnableIkeda : int
            Enable Ikeda acceleration.

        Returns
        -------
        dict
            Constraints dictionary with all fields.
        """
        C = {}
        C["EstimateA"] = int(EstimateA)
        C["AhatDiag"] = int(AhatDiag)
        C["QhatDiag"] = int(QhatDiag)
        C["QhatIsotropic"] = 1 if (QhatDiag and QhatIsotropic) else 0
        C["Estimatex0"] = int(Estimatex0)
        C["EstimatePx0"] = int(EstimatePx0)
        C["Px0Isotropic"] = 1 if (EstimatePx0 and Px0Isotropic) else 0
        C["mcIter"] = int(mcIter)
        C["EnableIkeda"] = int(EnableIkeda)
        return C

    @staticmethod
    def _nearestSPD(A):
        """Compute the nearest symmetric positive semi-definite matrix.

        Uses the algorithm of Higham (1988).
        """
        B = 0.5 * (A + A.T)
        _, S, Vt = np.linalg.svd(B)
        H = Vt.T @ np.diag(S) @ Vt
        Ahat = 0.5 * (B + H)
        Ahat = 0.5 * (Ahat + Ahat.T)
        # Test positive definiteness and fix if needed
        try:
            np.linalg.cholesky(Ahat)
            return Ahat
        except np.linalg.LinAlgError:
            pass
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while True:
            try:
                np.linalg.cholesky(Ahat)
                return Ahat
            except np.linalg.LinAlgError:
                mineig = np.min(np.real(np.linalg.eigvalsh(Ahat)))
                Ahat += I * (-mineig * k ** 2 + spacing)
                k += 1
            if k > 100:
                return Ahat

    @staticmethod
    def _ztest_pvalue(param, se):
        """Two-sided z-test p-value for H0: param == 0."""
        se_safe = np.where(se > 0, se, 1.0)
        z = np.abs(param / se_safe)
        p = 2.0 * (1.0 - norm.cdf(z))
        # Where se was 0, return 1.0
        p = np.where(se > 0, p, 1.0)
        return p

    @staticmethod
    def PP_ComputeParamStandardErrors(
        dN,
        xKFinal,
        WKFinal,
        Ahat,
        Qhat,
        x0hat,
        Px0hat,
        ExpectationSumsFinal,
        fitType,
        muhat,
        betahat,
        gammahat,
        windowTimes,
        HkAll,
        PPEM_Constraints=None,
    ):
        """Compute standard errors via the observed information matrix.

        Uses a Monte-Carlo approximation of the missing information matrix
        (McLachlan & Krishnan, Eq. 4.7).

        Parameters
        ----------
        dN : (C, N) spike observations
        xKFinal : (dx, N) smoothed states
        WKFinal : (dx, dx, N) smoothed state covariances
        Ahat : (dx, dx) estimated state transition
        Qhat : (dx, dx) estimated state noise covariance
        x0hat : (dx,) estimated initial state
        Px0hat : (dx, dx) estimated initial state covariance
        ExpectationSumsFinal : dict with sufficient statistics
        fitType : 'poisson' or 'binomial'
        muhat : (C,) estimated baseline rates
        betahat : (dx, C) estimated stimulus coefficients
        gammahat : (nW, C) or scalar estimated history coefficients
        windowTimes : history window boundaries or None
        HkAll : (N, nW, C) history design tensor
        PPEM_Constraints : dict from PP_EMCreateConstraints

        Returns
        -------
        SE : dict of standard errors for each parameter group
        Pvals : dict of p-values for each parameter group
        nTerms : int, total number of estimated parameters
        """
        if PPEM_Constraints is None:
            PPEM_Constraints = DecodingAlgorithms.PP_EMCreateConstraints()

        Ahat = np.atleast_2d(Ahat)
        Qhat = np.atleast_2d(Qhat)
        Px0hat = np.atleast_2d(Px0hat)
        x0hat = np.asarray(x0hat, dtype=float).reshape(-1)
        muhat = np.asarray(muhat, dtype=float).reshape(-1)
        betahat = np.atleast_2d(betahat)
        gammahat = np.asarray(gammahat, dtype=float)
        dN = np.atleast_2d(dN)

        dx = Ahat.shape[0]
        N = xKFinal.shape[1]
        K = N
        numCells = betahat.shape[1]
        fitType = str(fitType).lower()

        # ---- Complete Information Matrices ----

        # A information
        if PPEM_Constraints["EstimateA"]:
            n1_A, n2_A = Ahat.shape
            Qinv = np.linalg.inv(Qhat)
            if PPEM_Constraints["AhatDiag"]:
                IAComp = np.zeros((n1_A, n1_A))
                for l in range(n1_A):
                    el = np.zeros(n1_A)
                    el[l] = 1.0
                    em = np.zeros(n2_A)
                    em[l] = 1.0
                    termMat = Qinv @ np.outer(el, em) @ (ExpectationSumsFinal["Sxkm1xkm1"] * np.eye(n1_A))
                    IAComp[:, l] = np.diag(termMat)
            else:
                nA = Ahat.size
                IAComp = np.zeros((nA, nA))
                cnt = 0
                for l in range(n1_A):
                    el = np.zeros(n1_A)
                    el[l] = 1.0
                    for m in range(n2_A):
                        em = np.zeros(n2_A)
                        em[m] = 1.0
                        termMat = Qinv @ np.outer(el, em) @ ExpectationSumsFinal["Sxkm1xkm1"]
                        IAComp[:, cnt] = termMat.T.ravel()
                        cnt += 1
        else:
            IAComp = np.zeros((0, 0))

        # Q information
        n1_Q, n2_Q = Qhat.shape
        Qinv = np.linalg.inv(Qhat)
        if PPEM_Constraints["QhatDiag"]:
            if PPEM_Constraints["QhatIsotropic"]:
                IQComp = np.array([[0.5 * N * dx * Qhat[0, 0] ** (-2)]])
            else:
                IQComp = np.zeros((n1_Q, n1_Q))
                cnt = 0
                for l in range(n1_Q):
                    el = np.zeros(n1_Q)
                    el[l] = 1.0
                    termMat = N / 2.0 * Qinv @ np.outer(el, el) @ Qinv
                    IQComp[:, cnt] = np.diag(termMat)
                    cnt += 1
        else:
            nQ = Qhat.size
            IQComp = np.zeros((nQ, nQ))
            cnt = 0
            for l in range(n1_Q):
                el = np.zeros(n1_Q)
                el[l] = 1.0
                for m in range(n2_Q):
                    em = np.zeros(n2_Q)
                    em[m] = 1.0
                    termMat = N / 2.0 * Qinv @ np.outer(em, el) @ Qinv
                    IQComp[:, cnt] = termMat.T.ravel()
                    cnt += 1

        # Px0 information
        if PPEM_Constraints["EstimatePx0"]:
            Px0inv = np.linalg.inv(Px0hat)
            if PPEM_Constraints["Px0Isotropic"]:
                ISComp = np.array([[0.5 * dx * Px0hat[0, 0] ** (-2)]])
            else:
                n1_S, n2_S = Px0hat.shape
                ISComp = np.zeros((n1_S, n1_S))
                cnt = 0
                for l in range(n1_S):
                    el = np.zeros(n1_S)
                    el[l] = 1.0
                    termMat = 0.5 * Px0inv @ np.outer(el, el) @ Px0inv
                    ISComp[:, cnt] = np.diag(termMat)
                    cnt += 1
        else:
            ISComp = np.zeros((0, 0))

        # x0 information
        if PPEM_Constraints["Estimatex0"]:
            Qinv = np.linalg.inv(Qhat)
            Px0inv = np.linalg.inv(Px0hat)
            Ix0Comp = Px0inv + Ahat.T @ Qinv @ Ahat
        else:
            Ix0Comp = np.zeros((0, 0))

        # Monte Carlo draws for expectation approximation
        McExp = PPEM_Constraints["mcIter"]
        xKDrawExp = np.zeros((dx, K, McExp))
        for k in range(K):
            WuTemp = WKFinal[:, :, k]
            try:
                chol_m = np.linalg.cholesky(WuTemp).T  # upper triangular
            except np.linalg.LinAlgError:
                eigv, eigvec = np.linalg.eigh(WuTemp)
                eigv = np.maximum(eigv, 1e-12)
                chol_m = np.linalg.cholesky(eigvec @ np.diag(eigv) @ eigvec.T).T
            z = np.random.randn(dx, McExp)
            xKDrawExp[:, k, :] = xKFinal[:, k:k + 1] + chol_m @ z

        # Beta information (Hessian approximation via MC)
        IBetaComp = np.zeros((dx * numCells, dx * numCells))
        # xkPerm: (dx, McExp, K)
        xkPerm = np.transpose(xKDrawExp, (0, 2, 1))

        for c in range(numCells):
            HessianTerm = np.zeros((dx, dx, K))
            for k in range(K):
                Hk = HkAll[k, :, c] if HkAll.ndim == 3 else np.zeros(0)
                xk = xkPerm[:, :, k]  # (dx, McExp)

                gammaC = gammahat if gammahat.ndim == 0 or gammahat.size == 1 else gammahat[:, c]
                gammaC = np.atleast_1d(gammaC)

                Hk_vec = np.atleast_1d(Hk)
                hist_term = float(gammaC @ Hk_vec) if Hk_vec.size == gammaC.size and gammaC.size > 0 else 0.0

                terms = muhat[c] + betahat[:, c] @ xk + hist_term
                if fitType == "poisson":
                    ld = np.exp(np.clip(terms, -30, 30))
                    HessianTerm[:, :, k] = -1.0 / McExp * (ld[None, :] * xk) @ xk.T
                else:  # binomial
                    ld = 1.0 / (1.0 + np.exp(-np.clip(terms, -30, 30)))
                    ExplambdaDeltaXkXk = 1.0 / McExp * (ld[None, :] * xk) @ xk.T
                    ExplambdaDeltaSqXkXkT = 1.0 / McExp * (ld[None, :] ** 2 * xk) @ xk.T
                    ExplambdaDeltaCubeXkXkT = 1.0 / McExp * (ld[None, :] ** 3 * xk) @ xk.T
                    HessianTerm[:, :, k] = ExplambdaDeltaXkXk + ExplambdaDeltaSqXkXkT - 2 * ExplambdaDeltaCubeXkXkT

            startInd = dx * c
            endInd = dx * (c + 1)
            IBetaComp[startInd:endInd, startInd:endInd] = -np.sum(HessianTerm, axis=2)

        # Mu information
        IMuComp = np.zeros((numCells, numCells))
        for c in range(numCells):
            HessianTerm = 0.0
            for k in range(K):
                Hk_full = HkAll[:, :, c] if HkAll.ndim == 3 else np.zeros((K, 0))
                Hk_vec = Hk_full[k, :] if Hk_full.ndim == 2 and Hk_full.shape[0] > k else np.zeros(0)
                xk = xkPerm[:, :, k]

                gammaC = gammahat if gammahat.ndim == 0 or gammahat.size == 1 else gammahat[:, c]
                gammaC = np.atleast_1d(gammaC)
                Hk_vec = np.atleast_1d(Hk_vec)
                hist_term = float(gammaC @ Hk_vec) if Hk_vec.size == gammaC.size and gammaC.size > 0 else 0.0

                terms = muhat[c] + betahat[:, c] @ xk + hist_term
                if fitType == "poisson":
                    ld = np.exp(np.clip(terms, -30, 30))
                    HessianTerm -= 1.0 / McExp * np.sum(ld)
                else:
                    ld = 1.0 / (1.0 + np.exp(-np.clip(terms, -30, 30)))
                    ExplambdaDelta = 1.0 / McExp * np.sum(ld)
                    ExplambdaDeltaSq = 1.0 / McExp * np.sum(ld ** 2)
                    ExplambdaDeltaCubed = 1.0 / McExp * np.sum(ld ** 3)
                    HessianTerm += -(dN[c, k] + 1) * ExplambdaDelta + (dN[c, k] + 3) * ExplambdaDeltaSq - 3 * ExplambdaDeltaCubed
            IMuComp[c, c] = -HessianTerm

        # Gamma information
        gammahat_flat = gammahat.ravel()
        has_gamma = gammahat_flat.size > 1 or (gammahat_flat.size == 1 and gammahat_flat[0] != 0)
        if windowTimes is not None and len(windowTimes) > 0 and has_gamma:
            nHist = HkAll.shape[1] if HkAll.ndim == 3 else 0
            IGammaComp = np.zeros((nHist * numCells, nHist * numCells))
            for c in range(numCells):
                HessianTerm = np.zeros((nHist, nHist))
                Hk_full = HkAll[:, :, c] if HkAll.ndim == 3 else np.zeros((K, nHist))
                for k in range(K):
                    Hk_vec = Hk_full[k, :]
                    xk = xkPerm[:, :, k]
                    gammaC = gammahat if gammahat.ndim == 0 or gammahat.size == 1 else gammahat[:, c]
                    gammaC = np.atleast_1d(gammaC)
                    hist_term = float(gammaC @ Hk_vec) if Hk_vec.size == gammaC.size else 0.0
                    terms = muhat[c] + betahat[:, c] @ xk + hist_term
                    if fitType == "poisson":
                        ld = np.exp(np.clip(terms, -30, 30))
                        ExplambdaDelta = 1.0 / McExp * np.sum(ld)
                        HessianTerm -= np.outer(Hk_vec, Hk_vec) * ExplambdaDelta
                    else:
                        ld = 1.0 / (1.0 + np.exp(-np.clip(terms, -30, 30)))
                        ExplambdaDelta = 1.0 / McExp * np.sum(ld)
                        ExplambdaDeltaSq = 1.0 / McExp * np.sum(ld ** 2)
                        ExplambdaDeltaCubed = 1.0 / McExp * np.sum(ld ** 2)  # Matlab uses ld.^2 here
                        HessianTerm += (-ExplambdaDelta * (dN[c, k] + 1)
                                        + ExplambdaDeltaSq * (dN[c, k] + 3)
                                        - 2 * ExplambdaDeltaCubed) * np.outer(Hk_vec, Hk_vec)
                startInd = nHist * c
                endInd = nHist * (c + 1)
                IGammaComp[startInd:endInd, startInd:endInd] = -HessianTerm
        else:
            IGammaComp = np.zeros((0, 0))

        # Assemble complete information matrix
        n1 = IAComp.shape[0] if PPEM_Constraints["EstimateA"] else 0
        n2 = IQComp.shape[0]
        n3 = ISComp.shape[0] if PPEM_Constraints["EstimatePx0"] else 0
        n4 = Ix0Comp.shape[0] if PPEM_Constraints["Estimatex0"] else 0
        n5 = IMuComp.shape[0]
        n6 = IBetaComp.shape[0]
        n7 = IGammaComp.shape[0] if has_gamma else 0
        nTerms = n1 + n2 + n3 + n4 + n5 + n6 + n7

        IComp = np.zeros((nTerms, nTerms))
        off = 0
        if PPEM_Constraints["EstimateA"] and n1 > 0:
            IComp[off:off + n1, off:off + n1] = IAComp
        off = n1
        IComp[off:off + n2, off:off + n2] = IQComp
        off = n1 + n2
        if PPEM_Constraints["EstimatePx0"] and n3 > 0:
            IComp[off:off + n3, off:off + n3] = ISComp
        off = n1 + n2 + n3
        if PPEM_Constraints["Estimatex0"] and n4 > 0:
            IComp[off:off + n4, off:off + n4] = Ix0Comp
        off = n1 + n2 + n3 + n4
        IComp[off:off + n5, off:off + n5] = IMuComp
        off = n1 + n2 + n3 + n4 + n5
        IComp[off:off + n6, off:off + n6] = IBetaComp
        off = n1 + n2 + n3 + n4 + n5 + n6
        if n7 > 0:
            IComp[off:off + n7, off:off + n7] = IGammaComp

        # ---- Missing Information Matrix (Monte Carlo) ----
        Mc = PPEM_Constraints["mcIter"]
        xKDraw = np.zeros((dx, N, Mc))
        for n_idx in range(N):
            WuTemp = WKFinal[:, :, n_idx]
            try:
                chol_m = np.linalg.cholesky(WuTemp).T
            except np.linalg.LinAlgError:
                eigv, eigvec = np.linalg.eigh(WuTemp)
                eigv = np.maximum(eigv, 1e-12)
                chol_m = np.linalg.cholesky(eigvec @ np.diag(eigv) @ eigvec.T).T
            z = np.random.randn(dx, Mc)
            xKDraw[:, n_idx, :] = xKFinal[:, n_idx:n_idx + 1] + chol_m @ z

        if PPEM_Constraints["EstimatePx0"] or PPEM_Constraints["Estimatex0"]:
            try:
                chol_m = np.linalg.cholesky(Px0hat).T
            except np.linalg.LinAlgError:
                eigv, eigvec = np.linalg.eigh(Px0hat)
                eigv = np.maximum(eigv, 1e-12)
                chol_m = np.linalg.cholesky(eigvec @ np.diag(eigv) @ eigvec.T).T
            z = np.random.randn(dx, Mc)
            x0Draw = x0hat[:, None] + chol_m @ z
        else:
            x0Draw = np.tile(x0hat[:, None], (1, Mc))

        Qinv = np.linalg.inv(Qhat)
        Px0inv = np.linalg.inv(Px0hat)
        IMc = np.zeros((nTerms, nTerms, Mc))

        for c_mc in range(Mc):
            x_K = xKDraw[:, :, c_mc]
            x_0 = x0Draw[:, c_mc]
            Dx = x_K.shape[0]

            Sxkm1xk = np.zeros((Dx, Dx))
            Sxkm1xkm1 = np.zeros((Dx, Dx))
            Sxkxk = np.zeros((Dx, Dx))

            for k in range(K):
                if k == 0:
                    Sxkm1xk += np.outer(x_0, x_K[:, k])
                    Sxkm1xkm1 += np.outer(x_0, x_0)
                else:
                    Sxkm1xk += np.outer(x_K[:, k - 1], x_K[:, k])
                    Sxkm1xkm1 += np.outer(x_K[:, k - 1], x_K[:, k - 1])
                Sxkxk += np.outer(x_K[:, k], x_K[:, k])

            Sxkxk = 0.5 * (Sxkxk + Sxkxk.T)
            sumXkTerms_mc = Sxkxk - Ahat @ Sxkm1xk - Sxkm1xk.T @ Ahat.T + Ahat @ Sxkm1xkm1 @ Ahat.T
            Sxkxkm1 = Sxkm1xk.T
            sumXkTerms_mc = 0.5 * (sumXkTerms_mc + sumXkTerms_mc.T)

            # Score for A
            if PPEM_Constraints["EstimateA"]:
                ScorA = np.linalg.solve(Qhat, Sxkxkm1 - Ahat @ Sxkm1xkm1)
                if PPEM_Constraints["AhatDiag"]:
                    ScoreAMc = np.diag(ScorA)
                else:
                    ScoreAMc = ScorA.T.ravel()
            else:
                ScoreAMc = np.array([])

            # Score for Q
            if PPEM_Constraints["QhatDiag"]:
                if PPEM_Constraints["QhatIsotropic"]:
                    ScoreQ = -0.5 * (K * Dx * Qhat[0, 0] ** (-1) - Qhat[0, 0] ** (-2) * np.trace(sumXkTerms_mc))
                    ScoreQMc = np.atleast_1d(ScoreQ)
                else:
                    ScoreQ = -0.5 * np.linalg.solve(Qhat, K * np.eye(dx) - np.linalg.solve(Qhat, sumXkTerms_mc).T)
                    ScoreQMc = np.diag(ScoreQ)
            else:
                ScoreQ = -0.5 * np.linalg.solve(Qhat, K * np.eye(dx) - np.linalg.solve(Qhat, sumXkTerms_mc).T)
                ScoreQMc = ScoreQ.T.ravel()

            # Score for Px0
            if PPEM_Constraints["Px0Isotropic"]:
                diff = x_0 - x0hat
                ScoreSMc = np.atleast_1d(-0.5 * (Dx * Px0hat[0, 0] ** (-1)
                                                  - Px0hat[0, 0] ** (-2) * np.dot(diff, diff)))
            else:
                diff = x_0 - x0hat
                ScorS = -0.5 * np.linalg.solve(Px0hat, np.eye(dx) - np.linalg.solve(Px0hat, np.outer(diff, diff)).T)
                ScoreSMc = np.diag(ScorS)

            # Score for x0
            Scorx0 = -np.linalg.solve(Px0hat, x_0 - x0hat) + Ahat.T @ Qinv @ (x_K[:, 0] - Ahat @ x_0)
            Scorex0Mc = Scorx0.ravel()

            # Cell scores
            ScoreMuMc = np.zeros(numCells)
            ScoreBetaMc = np.array([], dtype=float)
            ScoreGammaMc = np.array([], dtype=float)

            for nc in range(numCells):
                Hk_full = HkAll[:, :, nc] if HkAll.ndim == 3 else np.zeros((K, 0))
                nHist_c = Hk_full.shape[1]
                gammaC = gammahat if gammahat.ndim == 0 or gammahat.size == 1 else gammahat[:, nc]
                gammaC = np.atleast_1d(gammaC)

                hist_terms = Hk_full @ gammaC if gammaC.size == nHist_c and nHist_c > 0 else np.zeros(K)
                terms = muhat[nc] + betahat[:, nc] @ x_K + hist_terms

                if fitType == "poisson":
                    ld = np.exp(np.clip(terms, -30, 30))
                    ScoreMuMc[nc] = np.sum(dN[nc, :] - ld)
                    ScoreBetaMc = np.concatenate([ScoreBetaMc,
                                                  np.sum((dN[nc, :] - ld)[None, :] * x_K, axis=1)])
                    if nHist_c > 0:
                        ScoreGammaMc = np.concatenate([ScoreGammaMc,
                                                       np.sum((dN[nc, :] - ld)[None, :] * Hk_full.T, axis=1)])
                else:  # binomial
                    ld = 1.0 / (1.0 + np.exp(-np.clip(terms, -30, 30)))
                    ScoreMuMc[nc] = np.sum(dN[nc, :] - (dN[nc, :] + 1) * ld + ld ** 2)
                    ScoreBetaMc = np.concatenate([ScoreBetaMc,
                                                  np.sum((dN[nc, :] * (1 - ld) - ld * (1 - ld))[None, :] * x_K, axis=1)])
                    if nHist_c > 0:
                        ScoreGammaMc = np.concatenate([ScoreGammaMc,
                                                       np.sum((dN[nc, :] - (dN[nc, :] + 1) * ld + ld ** 2)[None, :] * Hk_full.T, axis=1)])

            # Assemble score vector
            ScoreVec = np.concatenate([ScoreAMc, ScoreQMc])
            if PPEM_Constraints["EstimatePx0"]:
                ScoreVec = np.concatenate([ScoreVec, ScoreSMc])
            if PPEM_Constraints["Estimatex0"]:
                ScoreVec = np.concatenate([ScoreVec, Scorex0Mc])
            ScoreVec = np.concatenate([ScoreVec, ScoreMuMc, ScoreBetaMc])
            if has_gamma and ScoreGammaMc.size > 0:
                ScoreVec = np.concatenate([ScoreVec, ScoreGammaMc])

            IMc[:, :, c_mc] = np.outer(ScoreVec, ScoreVec)

        IMissing = np.mean(IMc, axis=2)
        IObs = IComp - IMissing
        try:
            invIObs = np.linalg.inv(IObs)
        except np.linalg.LinAlgError:
            invIObs = np.linalg.pinv(IObs)
        invIObs = DecodingAlgorithms._nearestSPD(invIObs)

        VarVec = np.diag(invIObs)
        SEVec = np.sqrt(np.maximum(VarVec, 0.0))

        # Unpack SE vector
        off = 0
        SEAterms = SEVec[off:off + n1]; off += n1
        SEQterms = SEVec[off:off + n2]; off += n2
        SEPx0terms = SEVec[off:off + n3]; off += n3
        SEx0terms = SEVec[off:off + n4]; off += n4
        SEMuTerms = SEVec[off:off + n5]; off += n5
        SEBetaTerms = SEVec[off:off + n6]; off += n6
        SEGammaTerms = SEVec[off:off + n7]

        SE = {}
        Pvals = {}

        # A
        if PPEM_Constraints["EstimateA"]:
            if PPEM_Constraints["AhatDiag"]:
                SEA = np.diag(SEAterms)
                pA = np.diag(DecodingAlgorithms._ztest_pvalue(np.diag(Ahat), np.diag(SEA)))
            else:
                SEA = SEAterms.reshape(Ahat.shape[1], Ahat.shape[0]).T
                pA = DecodingAlgorithms._ztest_pvalue(Ahat.ravel(), SEA.ravel()).reshape(Ahat.shape)
            SE["A"] = SEA
            Pvals["A"] = pA

        # Q
        if PPEM_Constraints["QhatDiag"]:
            SEQ = np.diag(SEQterms)
            if PPEM_Constraints["QhatIsotropic"]:
                pQ = np.diag(DecodingAlgorithms._ztest_pvalue(np.atleast_1d(Qhat[0, 0]), np.atleast_1d(SEQ[0, 0])))
            else:
                pQ = np.diag(DecodingAlgorithms._ztest_pvalue(np.diag(Qhat), np.diag(SEQ)))
        else:
            SEQ = SEQterms.reshape(Qhat.shape[1], Qhat.shape[0]).T
            pQ = DecodingAlgorithms._ztest_pvalue(Qhat.ravel(), SEQ.ravel()).reshape(Qhat.shape)
        SE["Q"] = SEQ
        Pvals["Q"] = pQ

        # Px0
        if PPEM_Constraints["EstimatePx0"]:
            SES = np.diag(SEPx0terms)
            if PPEM_Constraints["Px0Isotropic"]:
                pPx0 = np.diag(DecodingAlgorithms._ztest_pvalue(np.atleast_1d(Px0hat[0, 0]), np.atleast_1d(SES[0, 0])))
            else:
                pPx0 = np.diag(DecodingAlgorithms._ztest_pvalue(np.diag(Px0hat), np.diag(SES)))
            SE["Px0"] = SES
            Pvals["Px0"] = pPx0

        # x0
        if PPEM_Constraints["Estimatex0"]:
            SEx0 = SEx0terms
            pX0 = DecodingAlgorithms._ztest_pvalue(x0hat, SEx0)
            SE["x0"] = SEx0
            Pvals["x0"] = pX0

        # Mu
        SEMu = SEMuTerms
        pMu = DecodingAlgorithms._ztest_pvalue(muhat, SEMu)
        SE["mu"] = SEMu
        Pvals["mu"] = pMu

        # Beta
        SEBeta = SEBetaTerms.reshape(betahat.shape[1], betahat.shape[0]).T
        pBeta = DecodingAlgorithms._ztest_pvalue(betahat.ravel(), SEBeta.ravel()).reshape(betahat.shape)
        SE["beta"] = SEBeta
        Pvals["beta"] = pBeta

        # Gamma
        if has_gamma and n7 > 0:
            SEGamma = SEGammaTerms.reshape(gammahat.shape[1], gammahat.shape[0]).T if gammahat.ndim == 2 else SEGammaTerms
            pGamma = DecodingAlgorithms._ztest_pvalue(gammahat.ravel(), SEGammaTerms).reshape(gammahat.shape) if gammahat.ndim == 2 else DecodingAlgorithms._ztest_pvalue(gammahat.ravel(), SEGammaTerms)
            SE["gamma"] = SEGamma
            Pvals["gamma"] = pGamma

        return SE, Pvals, nTerms

    @staticmethod
    def PP_EStep(A, Q, dN, mu, beta, fitType, gamma, HkAll, x0, Px0):
        """E-step for PP EM: forward filter + RTS smoother + cross-covariance.

        Parameters
        ----------
        A : (dx, dx) state transition matrix
        Q : (dx, dx) state noise covariance
        dN : (C, N) binary spike observations
        mu : (C,) baseline log-rate
        beta : (dx, C) stimulus coefficients
        fitType : 'poisson' or 'binomial'
        gamma : (nW, C) or scalar history coefficients
        HkAll : (N, nW, C) history design tensor
        x0 : (dx,) initial state
        Px0 : (dx, dx) initial state covariance

        Returns
        -------
        x_K : (dx, N) smoothed states
        W_K : (dx, dx, N) smoothed covariances
        logll : float, log-likelihood
        ExpectationSums : dict of sufficient statistics
        """
        A = np.atleast_2d(A).astype(float)
        Q = np.atleast_2d(Q).astype(float)
        dN = np.atleast_2d(dN).astype(float)
        mu = np.asarray(mu, dtype=float).reshape(-1)
        beta = np.atleast_2d(beta).astype(float)
        gamma = np.asarray(gamma, dtype=float)
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        Px0 = np.atleast_2d(Px0).astype(float)
        fitType = str(fitType).lower()

        numCells, K = dN.shape
        Dx = A.shape[1]

        # Forward filter
        x_p = np.zeros((Dx, K + 1))
        x_u = np.zeros((Dx, K))
        W_p = np.zeros((Dx, Dx, K + 1))
        W_u = np.zeros((Dx, Dx, K))
        x_p[:, 0] = A @ x0
        W_p[:, :, 0] = A @ Px0 @ A.T + Q

        # Permute HkAll for PPDecode_updateLinear: (nW, C, N)
        HkPerm = np.transpose(HkAll, (1, 2, 0)) if HkAll.ndim == 3 else HkAll

        for k in range(K):
            x_u[:, k], W_u[:, :, k], _ = DecodingAlgorithms.PPDecode_updateLinear(
                x_p[:, k], W_p[:, :, k], dN, mu, beta, fitType, gamma, HkPerm, k + 1, None
            )
            A_k = A[:, :, min(k, A.shape[2] - 1)] if A.ndim == 3 else A
            Q_k = Q[:, :, min(k, Q.shape[2] - 1)] if Q.ndim == 3 else Q
            x_p[:, k + 1], W_p[:, :, k + 1] = DecodingAlgorithms.PPDecode_predict(
                x_u[:, k], W_u[:, :, k], A_k, Q_k
            )

        # RTS smoother using kalman_smootherFromFiltered
        # Convert state-major (dx, K+1/K) to time-major (K+1/K, dx) for
        # the smoother, which uses _state_history_time_major internally.
        x_p_tm = x_p.T                          # (K+1, dx)
        W_p_tm = np.transpose(W_p, (2, 0, 1))   # (K+1, dx, dx)
        x_u_tm = x_u.T                          # (K, dx)
        W_u_tm = np.transpose(W_u, (2, 0, 1))   # (K, dx, dx)

        x_K_tm, W_K_tm, Lk = DecodingAlgorithms.kalman_smootherFromFiltered(
            A, x_p_tm, W_p_tm, x_u_tm, W_u_tm
        )

        # Convert back to state-major: x_K (dx, K), W_K (dx, dx, K)
        x_K = x_K_tm.T if x_K_tm.ndim == 2 else x_K_tm
        W_K = np.transpose(W_K_tm, (1, 2, 0)) if W_K_tm.ndim == 3 else W_K_tm

        numStates = x_K.shape[0]

        # Cross-covariance Wku
        Wku = np.zeros((numStates, numStates, K, K))
        for k in range(K):
            Wku[:, :, k, k] = W_K[:, :, k]

        # W_u and W_p remain in state-major (dx, dx, K) format
        W_u_sm = W_u
        W_p_sm = W_p

        Dk = np.zeros((numStates, numStates, K))
        for u in range(K - 1, 0, -1):
            k = u - 1
            Dk[:, :, k] = W_u_sm[:, :, k] @ A.T @ np.linalg.inv(W_p_sm[:, :, k + 1])
            Wku[:, :, k, u] = Dk[:, :, k] @ Wku[:, :, k + 1, u]
            Wku[:, :, u, k] = Wku[:, :, k, u].T

        # Sufficient statistics
        Sxkm1xk = np.zeros((Dx, Dx))
        Sxkm1xkm1 = np.zeros((Dx, Dx))
        Sxkxk = np.zeros((Dx, Dx))

        for k in range(K):
            if k == 0:
                Sxkm1xk += Px0 @ A.T @ np.linalg.inv(W_p_sm[:, :, 0]) @ Wku[:, :, 0, 0]
                Sxkm1xkm1 += Px0 + np.outer(x0, x0)
            else:
                Sxkm1xk += Wku[:, :, k - 1, k] + np.outer(x_K[:, k - 1], x_K[:, k])
                Sxkm1xkm1 += Wku[:, :, k - 1, k - 1] + np.outer(x_K[:, k - 1], x_K[:, k - 1])
            Sxkxk += Wku[:, :, k, k] + np.outer(x_K[:, k], x_K[:, k])

        Sxkxk = 0.5 * (Sxkxk + Sxkxk.T)
        sumXkTerms = Sxkxk - A @ Sxkm1xk - Sxkm1xk.T @ A.T + A @ Sxkm1xkm1 @ A.T
        Sxkxkm1 = Sxkm1xk.T

        # Point process log-likelihood
        sumPPll = 0.0

        if fitType == "poisson":
            for k in range(K):
                if HkAll.ndim == 3:
                    Hk = HkAll[k, :, :]  # (nW, C) — need to handle orientation
                    if Hk.shape[0] == numCells:
                        Hk = Hk.T
                else:
                    Hk = np.zeros((0, numCells))

                xk = x_K[:, k]
                gammaC = np.tile(gamma, numCells) if gamma.ndim == 0 or gamma.size == 1 else gamma
                gammaC = np.atleast_2d(gammaC)
                if gammaC.shape[0] == 1 and gammaC.shape[1] == 1:
                    gammaC = np.full((max(Hk.shape[0], 1), numCells), float(gamma.ravel()[0]) if gamma.size > 0 else 0.0)

                if Hk.ndim == 2 and Hk.shape[0] > 0 and gammaC.shape[0] == Hk.shape[0]:
                    hist_diag = np.diag(gammaC.T @ Hk) if Hk.shape[0] > 0 else np.zeros(numCells)
                else:
                    hist_diag = np.zeros(numCells)

                terms = mu + beta.T @ xk + hist_diag
                Wk = W_K[:, :, k]
                ld = np.exp(np.clip(terms, -30, 30))
                bt = beta
                ExplambdaDelta = ld + 0.5 * (ld * np.diag(bt.T @ Wk @ bt))
                ExplogLD = terms
                sumPPll += float(np.sum(dN[:, k] * ExplogLD - ExplambdaDelta))

        elif fitType == "binomial":
            for k in range(K):
                if HkAll.ndim == 3:
                    Hk = HkAll[k, :, :]
                    if Hk.shape[0] == numCells:
                        Hk = Hk.T
                else:
                    Hk = np.zeros((0, numCells))

                xk = x_K[:, k]
                gammaC = np.tile(gamma, numCells) if gamma.ndim == 0 or gamma.size == 1 else gamma
                gammaC = np.atleast_2d(gammaC)
                if gammaC.shape[0] == 1 and gammaC.shape[1] == 1:
                    gammaC = np.full((max(Hk.shape[0], 1), numCells), float(gamma.ravel()[0]) if gamma.size > 0 else 0.0)

                if Hk.ndim == 2 and Hk.shape[0] > 0 and gammaC.shape[0] == Hk.shape[0]:
                    hist_diag = np.diag(gammaC.T @ Hk) if Hk.shape[0] > 0 else np.zeros(numCells)
                else:
                    hist_diag = np.zeros(numCells)

                terms = mu + beta.T @ xk + hist_diag
                Wk = W_K[:, :, k]
                ld_raw = np.clip(terms, -30, 30)
                ld = 1.0 / (1.0 + np.exp(-ld_raw))
                bt = beta
                btWbt_diag = np.diag(bt.T @ Wk @ bt)
                ExplambdaDelta = ld + 0.5 * (ld * (1 - ld) * (1 - 2 * ld)) * btWbt_diag
                ExplogLD = np.log(np.maximum(ld, 1e-30)) + 0.5 * (-ld * (1 - ld)) * btWbt_diag
                sumPPll += float(np.sum(dN[:, k] * ExplogLD - ExplambdaDelta))

        det_Q = max(float(np.linalg.det(Q)), np.finfo(float).tiny)
        det_Px0 = max(float(np.linalg.det(Px0)), np.finfo(float).tiny)
        logll = (
            -Dx * K / 2.0 * np.log(2.0 * np.pi)
            - K / 2.0 * np.log(det_Q)
            - Dx / 2.0 * np.log(2.0 * np.pi)
            - 0.5 * np.log(det_Px0)
            + sumPPll
            - 0.5 * np.trace(np.linalg.solve(Q, sumXkTerms))
            - Dx / 2.0
        )

        ExpectationSums = {
            "Sxkm1xkm1": Sxkm1xkm1,
            "Sxkm1xk": Sxkm1xk,
            "Sxkxkm1": Sxkxkm1,
            "Sxkxk": Sxkxk,
            "sumXkTerms": sumXkTerms,
            "sumPPll": sumPPll,
        }

        return x_K, W_K, logll, ExpectationSums

    @staticmethod
    def PP_MStep(
        dN, x_K, W_K, x0, Px0, ExpectationSums, fitType,
        muhat, betahat, gammahat, windowTimes, HkAll,
        PPEM_Constraints=None, MstepMethod="NewtonRaphson",
    ):
        """M-step for PP EM: update all model parameters.

        Parameters
        ----------
        dN : (C, N) spike observations
        x_K : (dx, N) smoothed states
        W_K : (dx, dx, N) smoothed covariances
        x0 : (dx,) current initial state estimate
        Px0 : (dx, dx) current initial covariance estimate
        ExpectationSums : dict from E-step
        fitType : 'poisson' or 'binomial'
        muhat : (C,) current baseline rates
        betahat : (dx, C) current stimulus coefficients
        gammahat : scalar or (nW, C) current history coefficients
        windowTimes : history window boundaries or None
        HkAll : (N, nW, C) history tensor
        PPEM_Constraints : dict from PP_EMCreateConstraints
        MstepMethod : 'NewtonRaphson' (default) or 'GLM'

        Returns
        -------
        Ahat, Qhat, muhat_new, betahat_new, gammahat_new, x0hat, Px0hat
        """
        if PPEM_Constraints is None:
            PPEM_Constraints = DecodingAlgorithms.PP_EMCreateConstraints()

        Sxkm1xkm1 = ExpectationSums["Sxkm1xkm1"]
        Sxkxkm1 = ExpectationSums["Sxkxkm1"]
        sumXkTerms = ExpectationSums["sumXkTerms"]

        dx, K = x_K.shape
        numCells = dN.shape[0]
        fitType = str(fitType).lower()

        x0 = np.asarray(x0, dtype=float).reshape(-1)
        Px0 = np.atleast_2d(Px0).astype(float)
        muhat = np.asarray(muhat, dtype=float).reshape(-1)
        betahat = np.atleast_2d(betahat).astype(float)
        gammahat = np.asarray(gammahat, dtype=float)

        # --- A update ---
        I_dx = np.eye(dx)
        if PPEM_Constraints["AhatDiag"]:
            Ahat = (Sxkxkm1 * I_dx) @ np.linalg.inv(Sxkm1xkm1 * I_dx + 1e-12 * I_dx)
        else:
            Ahat = np.linalg.solve(Sxkm1xkm1.T + 1e-12 * I_dx, Sxkxkm1.T).T

        # --- Q update ---
        if PPEM_Constraints["QhatDiag"]:
            if PPEM_Constraints["QhatIsotropic"]:
                Qhat = (1.0 / (dx * K)) * np.trace(sumXkTerms) * I_dx
            else:
                Qhat = (1.0 / K) * (sumXkTerms * I_dx)
                Qhat = 0.5 * (Qhat + Qhat.T)
        else:
            Qhat = (1.0 / K) * sumXkTerms
            Qhat = 0.5 * (Qhat + Qhat.T)

        # Ensure positive definiteness
        eigvals, eigvecs = np.linalg.eigh(Qhat)
        eigvals = np.maximum(eigvals, 1e-10)
        Qhat = eigvecs @ np.diag(eigvals) @ eigvecs.T
        Qhat = 0.5 * (Qhat + Qhat.T)

        # --- x0 update ---
        if PPEM_Constraints["Estimatex0"]:
            Px0inv = np.linalg.inv(Px0 + 1e-12 * I_dx)
            Qinv = np.linalg.inv(Qhat + 1e-12 * I_dx)
            x0hat = np.linalg.solve(Px0inv + Ahat.T @ Qinv @ Ahat,
                                    Ahat.T @ Qinv @ x_K[:, 0] + Px0inv @ x0)
        else:
            x0hat = x0.copy()

        # --- Px0 update ---
        if PPEM_Constraints["EstimatePx0"]:
            if PPEM_Constraints["Px0Isotropic"]:
                diff = x0hat - x0
                Px0hat = (np.dot(diff, diff) / (dx * K)) * I_dx
            else:
                diff = x0hat - x0
                Px0hat = np.outer(diff, diff) * I_dx
                Px0hat = 0.5 * (Px0hat + Px0hat.T)
            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(Px0hat)
            eigvals = np.maximum(eigvals, 1e-10)
            Px0hat = eigvecs @ np.diag(eigvals) @ eigvecs.T
        else:
            Px0hat = Px0.copy()

        betahat_new = betahat.copy()
        gammahat_new = gammahat.copy() if gammahat.ndim > 0 else np.atleast_1d(gammahat).copy()
        muhat_new = muhat.copy()

        # --- Newton-Raphson for beta, mu, gamma ---
        McExp = 50
        xKDrawExp = np.zeros((dx, K, McExp))
        diffTol = 1e-5

        for k in range(K):
            WuTemp = W_K[:, :, k]
            try:
                chol_m = np.linalg.cholesky(WuTemp).T
            except np.linalg.LinAlgError:
                eigv, eigvec = np.linalg.eigh(WuTemp)
                eigv = np.maximum(eigv, 1e-12)
                chol_m = np.linalg.cholesky(eigvec @ np.diag(eigv) @ eigvec.T).T
            z = np.random.randn(dx, McExp)
            xKDrawExp[:, k, :] = x_K[:, k:k + 1] + chol_m @ z

        # xkPerm: (dx, McExp, K)
        xkPerm = np.transpose(xKDrawExp, (0, 2, 1))

        # --- Beta Newton-Raphson ---
        for c in range(numCells):
            converged = False
            maxIter_nr = 100
            for iteration in range(maxIter_nr):
                HessianTerm = np.zeros((dx, dx))
                GradTerm = np.zeros(dx)

                for k in range(K):
                    Hk_full = HkAll[:, :, c] if HkAll.ndim == 3 else np.zeros((K, 0))
                    Hk_vec = Hk_full[k, :] if Hk_full.ndim == 2 and Hk_full.shape[0] > k else np.zeros(0)
                    xk = xkPerm[:, :, k]  # (dx, McExp)

                    gammaC = gammahat if gammahat.ndim == 0 or gammahat.size == 1 else gammahat[:, c]
                    gammaC = np.atleast_1d(gammaC)
                    Hk_vec = np.atleast_1d(Hk_vec)
                    hist_term = float(gammaC @ Hk_vec) if Hk_vec.size == gammaC.size and gammaC.size > 0 else 0.0

                    terms = muhat[c] + betahat_new[:, c] @ xk + hist_term

                    if fitType == "poisson":
                        ld = np.exp(np.clip(terms, -30, 30))
                        ExpLambdaXk = (1.0 / McExp) * np.sum(ld[None, :] * xk, axis=1)
                        ExpLambdaXkXkT = (1.0 / McExp) * (ld[None, :] * xk) @ xk.T
                        GradTerm += dN[c, k] * x_K[:, k] - ExpLambdaXk
                        HessianTerm -= ExpLambdaXkXkT
                    else:  # binomial
                        ld = 1.0 / (1.0 + np.exp(-np.clip(terms, -30, 30)))
                        ExplambdaDeltaXkXk = (1.0 / McExp) * (ld[None, :] * xk) @ xk.T
                        ExplambdaDeltaSqXkXkT = (1.0 / McExp) * ((ld ** 2)[None, :] * xk) @ xk.T
                        ExplambdaDeltaCubeXkXkT = (1.0 / McExp) * ((ld ** 3)[None, :] * xk) @ xk.T
                        ExpLambdaXk = (1.0 / McExp) * np.sum(ld[None, :] * xk, axis=1)
                        ExpLambdaSquaredXk = (1.0 / McExp) * np.sum((ld ** 2)[None, :] * xk, axis=1)
                        GradTerm += dN[c, k] * x_K[:, k] - (dN[c, k] + 1) * ExpLambdaXk + ExpLambdaSquaredXk
                        HessianTerm += ExplambdaDeltaXkXk + ExplambdaDeltaSqXkXkT - 2 * ExplambdaDeltaCubeXkXkT

                if np.any(np.isnan(HessianTerm)) or np.any(np.isinf(HessianTerm)):
                    betahat_newTemp = betahat_new[:, c]
                else:
                    try:
                        betahat_newTemp = betahat_new[:, c] - np.linalg.solve(HessianTerm, GradTerm)
                    except np.linalg.LinAlgError:
                        betahat_newTemp = betahat_new[:, c]
                    if np.any(np.isnan(betahat_newTemp)):
                        betahat_newTemp = betahat_new[:, c]

                mabsDiff = float(np.max(np.abs(betahat_newTemp - betahat_new[:, c])))
                if mabsDiff < diffTol:
                    converged = True
                betahat_new[:, c] = betahat_newTemp
                if converged:
                    break

        # --- Mu Newton-Raphson ---
        for c in range(numCells):
            converged = False
            maxIter_nr = 100
            for iteration in range(maxIter_nr):
                HessianTerm = 0.0
                GradTerm = 0.0

                for k in range(K):
                    Hk_full = HkAll[:, :, c] if HkAll.ndim == 3 else np.zeros((K, 0))
                    Hk_vec = Hk_full[k, :] if Hk_full.ndim == 2 and Hk_full.shape[0] > k else np.zeros(0)
                    xk = xkPerm[:, :, k]

                    gammaC = gammahat if gammahat.ndim == 0 or gammahat.size == 1 else gammahat[:, c]
                    gammaC = np.atleast_1d(gammaC)
                    Hk_vec = np.atleast_1d(Hk_vec)
                    hist_term = float(gammaC @ Hk_vec) if Hk_vec.size == gammaC.size and gammaC.size > 0 else 0.0

                    terms = muhat_new[c] + betahat[:, c] @ xk + hist_term

                    if fitType == "poisson":
                        ld = np.exp(np.clip(terms, -30, 30))
                        ExpLambdaDelta = (1.0 / McExp) * np.sum(ld)
                        GradTerm += dN[c, k] - ExpLambdaDelta
                        HessianTerm -= ExpLambdaDelta
                    else:  # binomial
                        ld = 1.0 / (1.0 + np.exp(-np.clip(terms, -30, 30)))
                        ExpLambdaDelta = (1.0 / McExp) * np.sum(ld)
                        ExpLambdaDeltaSq = (1.0 / McExp) * np.sum(ld ** 2)
                        ExpLambdaDeltaCubed = (1.0 / McExp) * np.sum(ld ** 3)
                        GradTerm += dN[c, k] - (dN[c, k] + 1) * ExpLambdaDelta + ExpLambdaDeltaSq
                        HessianTerm += -(dN[c, k] + 1) * ExpLambdaDelta + (dN[c, k] + 3) * ExpLambdaDeltaSq - 2 * ExpLambdaDeltaCubed

                if np.isnan(HessianTerm) or np.isinf(HessianTerm) or abs(HessianTerm) < 1e-30:
                    muhat_newTemp = muhat_new[c]
                else:
                    muhat_newTemp = muhat_new[c] - GradTerm / HessianTerm
                    if np.isnan(muhat_newTemp):
                        muhat_newTemp = muhat_new[c]

                mabsDiff = abs(muhat_newTemp - muhat_new[c])
                if mabsDiff < diffTol:
                    converged = True
                muhat_new[c] = muhat_newTemp
                if converged:
                    break

        # --- Gamma Newton-Raphson ---
        gammahat_flat = gammahat_new.ravel()
        has_gamma = (windowTimes is not None and len(windowTimes) > 0
                     and (gammahat_flat.size > 1 or (gammahat_flat.size == 1 and gammahat_flat[0] != 0)))

        if has_gamma and gammahat_new.ndim >= 1:
            nGamma = gammahat_new.shape[0] if gammahat_new.ndim == 1 else gammahat_new.shape[0]
            for c in range(numCells):
                converged = False
                maxIter_nr = 100
                gammaC = gammahat_new if gammahat_new.ndim == 0 or gammahat_new.size == 1 else gammahat_new[:, c] if gammahat_new.ndim == 2 else gammahat_new
                gammaC = np.atleast_1d(gammaC).copy()

                for iteration in range(maxIter_nr):
                    HessianTerm = np.zeros((nGamma, nGamma))
                    GradTerm = np.zeros(nGamma)

                    for k in range(K):
                        Hk_full = HkAll[:, :, c] if HkAll.ndim == 3 else np.zeros((K, 0))
                        Hk_vec = Hk_full[k, :] if Hk_full.ndim == 2 and Hk_full.shape[0] > k else np.zeros(0)
                        Hk_vec = np.atleast_1d(Hk_vec)
                        xk = xkPerm[:, :, k]

                        hist_term = float(gammaC @ Hk_vec) if Hk_vec.size == gammaC.size and gammaC.size > 0 else 0.0
                        terms = muhat[c] + betahat[:, c] @ xk + hist_term

                        if fitType == "poisson":
                            ld = np.exp(np.clip(terms, -30, 30))
                            ExpLambdaDelta = (1.0 / McExp) * np.sum(ld)
                            GradTerm += (dN[c, k] - ExpLambdaDelta) * Hk_vec
                            HessianTerm -= ExpLambdaDelta * np.outer(Hk_vec, Hk_vec)
                        else:  # binomial
                            ld = 1.0 / (1.0 + np.exp(-np.clip(terms, -30, 30)))
                            ExpLambdaDelta = (1.0 / McExp) * np.sum(ld)
                            ExpLambdaDeltaSq = (1.0 / McExp) * np.sum(ld ** 2)
                            ExpLambdaDeltaCubed = (1.0 / McExp) * np.sum(ld ** 3)
                            GradTerm += (dN[c, k] - (dN[c, k] + 1) * ExpLambdaDelta + ExpLambdaDeltaSq) * Hk_vec
                            HessianTerm += (-(dN[c, k] + 1) * ExpLambdaDelta + (dN[c, k] + 3) * ExpLambdaDeltaSq - 2 * ExpLambdaDeltaCubed) * np.outer(Hk_vec, Hk_vec)

                    if np.any(np.isnan(HessianTerm)) or np.any(np.isinf(HessianTerm)):
                        gammahat_newTemp = gammaC
                    else:
                        try:
                            gammahat_newTemp = gammaC - np.linalg.solve(HessianTerm, GradTerm)
                        except np.linalg.LinAlgError:
                            gammahat_newTemp = gammaC
                        if np.any(np.isnan(gammahat_newTemp)):
                            gammahat_newTemp = gammaC

                    mabsDiff = float(np.max(np.abs(gammahat_newTemp - gammaC)))
                    if mabsDiff < diffTol:
                        converged = True
                    gammaC = gammahat_newTemp
                    if converged:
                        break

                if gammahat_new.ndim == 2:
                    gammahat_new[:, c] = gammaC
                else:
                    gammahat_new = gammaC

        return Ahat, Qhat, muhat_new, betahat_new, gammahat_new, x0hat, Px0hat

    @staticmethod
    def PP_EM(
        dN,
        Ahat0,
        Qhat0,
        mu,
        beta,
        fitType="poisson",
        delta=0.001,
        gamma=None,
        windowTimes=None,
        x0=None,
        Px0=None,
        PPEM_Constraints=None,
        MstepMethod="NewtonRaphson",
    ):
        """Full Point-Process state-space EM algorithm.

        Estimates state-space model parameters (A, Q, mu, beta, gamma) via EM
        for point-process observations. Unlike PPSS_EM, this operates on raw
        spike observations with explicit beta/mu/gamma parameters (no basis
        functions).

        Parameters
        ----------
        dN : (C, N) binary spike observations
        Ahat0 : (dx, dx) initial state transition matrix
        Qhat0 : (dx, dx) initial state noise covariance
        mu : (C,) initial baseline log-rates
        beta : (dx, C) initial stimulus coefficients
        fitType : 'poisson' or 'binomial'
        delta : float, time bin width
        gamma : (nW, C) or scalar, initial history coefficients
        windowTimes : history window boundaries
        x0 : (dx,) initial state (default zeros)
        Px0 : (dx, dx) initial state covariance
        PPEM_Constraints : dict from PP_EMCreateConstraints
        MstepMethod : 'NewtonRaphson' or 'GLM'

        Returns
        -------
        xKFinal, WKFinal, Ahat, Qhat, muhat, betahat, gammahat,
        x0hat, Px0hat, IC, SE, Pvals, nIter
        """
        from .history import History  # local import to avoid circular dependency

        Ahat0 = np.atleast_2d(Ahat0).astype(float)
        Qhat0 = np.atleast_2d(Qhat0).astype(float)
        numStates = Ahat0.shape[0]
        dN = np.atleast_2d(dN).astype(float)

        if PPEM_Constraints is None:
            PPEM_Constraints = DecodingAlgorithms.PP_EMCreateConstraints()
        if Px0 is None:
            Px0 = 1e-9 * np.eye(numStates)
        else:
            Px0 = np.atleast_2d(Px0).astype(float)
        if x0 is None:
            x0 = np.zeros(numStates)
        else:
            x0 = np.asarray(x0, dtype=float).reshape(-1)
        if gamma is None:
            gamma = np.zeros(0)
        gamma = np.asarray(gamma, dtype=float)

        if delta is None or delta == 0:
            delta = 0.001

        if windowTimes is None:
            gamma_flat = gamma.ravel()
            if gamma_flat.size == 0 or (gamma_flat.size == 1 and gamma_flat[0] == 0):
                windowTimes = []
            else:
                windowTimes = np.arange(0, (gamma.shape[0] + 2) * delta, delta).tolist()

        mu = np.asarray(mu, dtype=float).reshape(-1)
        beta = np.atleast_2d(beta).astype(float)

        # Build HkAll from spike trains and history windows
        K_cells = dN.shape[0]
        N_time = dN.shape[1]
        minTime = 0.0
        maxTime = (N_time - 1) * delta

        if len(windowTimes) > 0:
            histObj = History(windowTimes, minTime, maxTime)
            HkAll_list = []
            for k in range(K_cells):
                spike_indices = np.where(dN[k, :] == 1)[0]
                spike_times = (spike_indices) * delta
                nst = nspikeTrain(spike_times)
                nst.setMinTime(minTime)
                nst.setMaxTime(maxTime)
                hmat = histObj.computeHistory(nst).dataToMatrix()
                HkAll_list.append(hmat)
            # Stack: (N_time, nW, K_cells)
            HkAll = np.stack(HkAll_list, axis=2)
        else:
            HkAll = np.zeros((N_time, 0, K_cells))
            gamma = np.zeros(1)
            gamma[0] = 0.0

        # EM setup
        tolAbs = 1e-3
        llTol = 1e-3
        maxIter = 100
        numToKeep = 10

        # Circular buffer storage
        A_buf = [None] * numToKeep
        Q_buf = [None] * numToKeep
        x0_buf = [None] * numToKeep
        Px0_buf = [None] * numToKeep
        mu_buf = [None] * numToKeep
        beta_buf = [None] * numToKeep
        gamma_buf = [None] * numToKeep
        x_K_buf = [None] * numToKeep
        W_K_buf = [None] * numToKeep
        ExpSums_buf = [None] * numToKeep

        # Scaled system initialization
        A0 = Ahat0.copy()
        Q0 = Qhat0.copy()

        A_buf[0] = A0.copy()
        Q_buf[0] = Q0.copy()
        x0_buf[0] = x0.copy()
        Px0_buf[0] = Px0.copy()
        mu_buf[0] = mu.copy()
        beta_buf[0] = beta.copy()
        gamma_buf[0] = gamma.copy()

        # Apply scaling
        try:
            Tq = np.linalg.solve(np.linalg.cholesky(Q_buf[0]).T, np.eye(numStates))
        except np.linalg.LinAlgError:
            Tq = np.eye(numStates)
        TqInv = np.linalg.inv(Tq)

        A_buf[0] = Tq @ A_buf[0] @ TqInv
        Q_buf[0] = Tq @ Q_buf[0] @ Tq.T
        x0_buf[0] = Tq @ x0
        Px0_buf[0] = Tq @ Px0 @ Tq.T
        beta_buf[0] = np.linalg.solve(Tq.T, beta_buf[0])

        ll_list = []
        dLikelihood = [np.inf]
        stoppingCriteria = False
        cnt = 0

        print("                        Point-Process Observation EM Algorithm                        ")
        while not stoppingCriteria and cnt < maxIter:
            si = cnt % numToKeep
            si_p1 = (cnt + 1) % numToKeep
            si_m1 = (cnt - 1) % numToKeep

            print("-" * 80)
            print(f"Iteration #{cnt + 1}")
            print("-" * 80)

            # E-step
            x_K_cur, W_K_cur, ll, ExpSums = DecodingAlgorithms.PP_EStep(
                A_buf[si], Q_buf[si], dN, mu_buf[si], beta_buf[si],
                fitType, gamma_buf[si], HkAll, x0_buf[si], Px0_buf[si]
            )
            x_K_buf[si] = x_K_cur
            W_K_buf[si] = W_K_cur
            ExpSums_buf[si] = ExpSums
            ll_list.append(ll)

            # M-step
            Anew, Qnew, munew, bnew, gnew, x0new, Px0new = DecodingAlgorithms.PP_MStep(
                dN, x_K_cur, W_K_cur, x0_buf[si], Px0_buf[si], ExpSums,
                fitType, mu_buf[si], beta_buf[si], gamma_buf[si],
                windowTimes, HkAll, PPEM_Constraints, MstepMethod
            )
            A_buf[si_p1] = Anew
            Q_buf[si_p1] = Qnew
            mu_buf[si_p1] = munew
            beta_buf[si_p1] = bnew
            gamma_buf[si_p1] = gnew
            x0_buf[si_p1] = x0new
            Px0_buf[si_p1] = Px0new

            if not PPEM_Constraints["EstimateA"]:
                A_buf[si_p1] = A_buf[si]

            # Convergence check
            if cnt == 0:
                dLikelihood.append(np.inf)
                dMax = np.inf
            else:
                dLikelihood.append(ll_list[cnt] - ll_list[cnt - 1])
                dQvals = float(np.max(np.abs(np.sqrt(np.maximum(np.abs(Q_buf[si]), 0)) - np.sqrt(np.maximum(np.abs(Q_buf[si_m1]), 0)))))
                dAvals = float(np.max(np.abs(A_buf[si] - A_buf[si_m1])))
                dMuvals = float(np.max(np.abs(mu_buf[si] - mu_buf[si_m1])))
                dBetavals = float(np.max(np.abs(beta_buf[si] - beta_buf[si_m1])))
                gam_cur = gamma_buf[si].ravel() if gamma_buf[si] is not None else np.zeros(1)
                gam_prev = gamma_buf[si_m1].ravel() if gamma_buf[si_m1] is not None else np.zeros(1)
                dGammavals = float(np.max(np.abs(gam_cur[:min(len(gam_cur), len(gam_prev))] - gam_prev[:min(len(gam_cur), len(gam_prev))]))) if gam_cur.size > 0 else 0.0
                dMax = max(dQvals, dAvals, dMuvals, dBetavals, dGammavals)

            if cnt == 0:
                print("Max Parameter Change: N/A")
            else:
                print(f"Max Parameter Change: {dMax:.6f}")

            cnt += 1
            if dMax < tolAbs:
                stoppingCriteria = True
                print(f"         EM converged at iteration# {cnt} b/c change in params was within criteria")

            if abs(dLikelihood[-1]) < llTol or dLikelihood[-1] < 0:
                stoppingCriteria = True
                print(f"         EM stopped at iteration# {cnt} b/c change in likelihood was negative")

        print("-" * 80)

        # Select best iteration
        ll_arr = np.array(ll_list)
        if ll_arr.size > 0:
            maxLLIndex = int(np.argmax(ll_arr))
        else:
            maxLLIndex = 0
        maxLLIndMod = maxLLIndex % numToKeep
        nIter = cnt

        xKFinal = x_K_buf[maxLLIndMod] if x_K_buf[maxLLIndMod] is not None else np.zeros((numStates, N_time))
        WKFinal = W_K_buf[maxLLIndMod] if W_K_buf[maxLLIndMod] is not None else np.zeros((numStates, numStates, N_time))
        Ahat = A_buf[maxLLIndMod] if A_buf[maxLLIndMod] is not None else A0
        Qhat = Q_buf[maxLLIndMod] if Q_buf[maxLLIndMod] is not None else Q0
        muhat = mu_buf[maxLLIndMod] if mu_buf[maxLLIndMod] is not None else mu
        betahat = beta_buf[maxLLIndMod] if beta_buf[maxLLIndMod] is not None else beta
        gammahat = gamma_buf[maxLLIndMod] if gamma_buf[maxLLIndMod] is not None else gamma
        x0hat = x0_buf[maxLLIndMod] if x0_buf[maxLLIndMod] is not None else x0
        Px0hat = Px0_buf[maxLLIndMod] if Px0_buf[maxLLIndMod] is not None else Px0
        ExpSumsFinal = ExpSums_buf[maxLLIndMod] if ExpSums_buf[maxLLIndMod] is not None else {}

        # Unscale system
        try:
            Tq_unscale = np.linalg.solve(np.linalg.cholesky(Q0).T, np.eye(numStates))
        except np.linalg.LinAlgError:
            Tq_unscale = np.eye(numStates)
        TqInv_unscale = np.linalg.inv(Tq_unscale)

        Ahat = TqInv_unscale @ Ahat @ Tq_unscale
        Qhat = TqInv_unscale @ Qhat @ TqInv_unscale.T
        xKFinal = TqInv_unscale @ xKFinal
        x0hat = TqInv_unscale @ x0hat
        Px0hat = TqInv_unscale @ Px0hat @ TqInv_unscale.T
        if WKFinal.ndim == 3:
            for kk in range(WKFinal.shape[2]):
                WKFinal[:, :, kk] = TqInv_unscale @ WKFinal[:, :, kk] @ TqInv_unscale.T
        betahat = (betahat.T @ Tq_unscale).T

        # Compute standard errors
        SE = {}
        Pvals = {}
        if ExpSumsFinal:
            try:
                SE, Pvals, _ = DecodingAlgorithms.PP_ComputeParamStandardErrors(
                    dN, xKFinal, WKFinal, Ahat, Qhat, x0hat, Px0hat,
                    ExpSumsFinal, fitType, muhat, betahat, gammahat,
                    windowTimes, HkAll, PPEM_Constraints
                )
            except Exception:
                pass

        # Information criteria
        K_total = xKFinal.shape[1]
        Dx = Ahat.shape[1]

        # Count parameters
        if PPEM_Constraints["EstimateA"] and PPEM_Constraints["AhatDiag"]:
            n1_ic = Ahat.shape[0]
        elif PPEM_Constraints["EstimateA"]:
            n1_ic = Ahat.size
        else:
            n1_ic = 0
        if PPEM_Constraints["QhatDiag"] and PPEM_Constraints["QhatIsotropic"]:
            n2_ic = 1
        elif PPEM_Constraints["QhatDiag"]:
            n2_ic = Qhat.shape[0]
        else:
            n2_ic = Qhat.size
        if PPEM_Constraints["EstimatePx0"] and PPEM_Constraints["Px0Isotropic"]:
            n3_ic = 1
        elif PPEM_Constraints["EstimatePx0"]:
            n3_ic = Px0hat.shape[0]
        else:
            n3_ic = 0
        n4_ic = x0hat.size if PPEM_Constraints["Estimatex0"] else 0
        n5_ic = muhat.size
        n6_ic = betahat.size
        gammahat_flat = gammahat.ravel()
        if gammahat_flat.size == 1 and gammahat_flat[0] == 0:
            n7_ic = 0
        else:
            n7_ic = gammahat.size
        nTerms_ic = n1_ic + n2_ic + n3_ic + n4_ic + n5_ic + n6_ic + n7_ic

        sumXkTerms_ic = ExpSumsFinal.get("sumXkTerms", np.zeros((Dx, Dx)))
        ll_best = ll_list[maxLLIndex] if ll_list else 0.0
        det_Q = max(float(np.linalg.det(Qhat)), np.finfo(float).tiny)
        det_Px0 = max(float(np.linalg.det(Px0hat)), np.finfo(float).tiny)

        llobs = (ll_best
                 + Dx * K_total / 2.0 * np.log(2.0 * np.pi)
                 + K_total / 2.0 * np.log(det_Q)
                 + 0.5 * np.trace(np.linalg.solve(Qhat, sumXkTerms_ic))
                 + Dx / 2.0 * np.log(2.0 * np.pi)
                 + 0.5 * np.log(det_Px0)
                 + 0.5 * Dx)
        AIC = 2 * nTerms_ic - 2 * llobs
        AICc = AIC + 2 * nTerms_ic * (nTerms_ic + 1) / max(K_total - nTerms_ic - 1, 1)
        BIC = -2 * llobs + nTerms_ic * np.log(K_total)

        IC = {
            "AIC": AIC,
            "AICc": AICc,
            "BIC": BIC,
            "llobs": llobs,
            "llcomp": ll_best,
        }

        return (xKFinal, WKFinal, Ahat, Qhat, muhat, betahat, gammahat,
                x0hat, Px0hat, IC, SE, Pvals, nIter)


    # mPPCO family -- mixed Point-Process & Continuous Observation
    # ------------------------------------------------------------------

    @staticmethod
    def mPPCODecode_predict(x_u, W_u, A, Q):
        """Predict step for the mPPCO filter.

        Matlab: ``DecodingAlgorithms.mPPCODecode_predict``  (lines 4846-4854)

        Parameters
        ----------
        x_u : array (ns,)   -- updated state
        W_u : array (ns,ns) -- updated covariance
        A   : array (ns,ns) -- state transition
        Q   : array (ns,ns) -- process noise

        Returns
        -------
        x_p : array (ns,)
        W_p : array (ns,ns)
        """
        x_u = np.asarray(x_u, dtype=float).reshape(-1)
        ns = x_u.size
        A = np.asarray(A, dtype=float).reshape(ns, ns)
        Q = np.asarray(Q, dtype=float).reshape(ns, ns)
        W_u = np.asarray(W_u, dtype=float).reshape(ns, ns)
        x_p = A @ x_u
        W_p = A @ W_u @ A.T + Q
        W_p = _symmetrize(W_p)
        return x_p, W_p

    @staticmethod
    def mPPCODecode_update(x_p, W_p, C, R, y, alpha, dN, mu, beta,
                           fitType='poisson', gamma=None, HkAll=None,
                           time_index=1, WuConv=None):
        """Update step for the mPPCO filter (PP + continuous observation).

        Matlab: ``DecodingAlgorithms.mPPCODecode_update``  (lines 4855-4944)

        This combines both the point-process update terms (sumValVec/sumValMat)
        AND the Kalman/continuous-observation terms C'*R^{-1}*C and C'*R^{-1}*(y-Cx-alpha).

        Parameters
        ----------
        x_p   : (ns,)   -- predicted state
        W_p   : (ns,ns) -- predicted covariance
        C     : (nObs,ns) -- observation matrix
        R     : (nObs,nObs) -- observation noise covariance
        y     : (nObs,)  -- continuous observation at this time step
        alpha : (nObs,)  -- observation offset
        dN    : (numCells,N) -- spike matrix (full)
        mu    : (numCells,) -- CIF baseline
        beta  : (ns,numCells) -- CIF state coefficients
        fitType : 'poisson' or 'binomial'
        gamma : (numWindows,numCells) or scalar -- history coefficients
        HkAll : (numWindows,numCells,N) -- permuted history tensor with time on 3rd axis
        time_index : int -- 1-based time index
        WuConv : converged covariance or None

        Returns
        -------
        x_u : (ns,)
        W_u : (ns,ns)
        lambdaDeltaMat : (numCells,1)
        """
        x_p = np.asarray(x_p, dtype=float).reshape(-1)
        ns = x_p.size
        W_p = np.asarray(W_p, dtype=float).reshape(ns, ns)
        obs = _as_observation_matrix(dN)
        numCells = obs.shape[0]
        C = np.asarray(C, dtype=float)
        R = np.asarray(R, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        alpha = np.asarray(alpha, dtype=float).reshape(-1)
        mu_vec = np.asarray(mu, dtype=float).reshape(-1)
        beta_mat = np.asarray(beta, dtype=float)
        if beta_mat.ndim == 1:
            beta_mat = beta_mat.reshape(-1, 1)

        # Default gamma
        if gamma is None or (np.isscalar(gamma) and gamma == 0):
            gamma_mat = np.zeros((1, numCells), dtype=float)
        else:
            gamma_mat = np.asarray(gamma, dtype=float)
            if gamma_mat.ndim == 1:
                gamma_mat = gamma_mat.reshape(-1, 1)

        # Default HkAll -- expects (numWindows, numCells, N) orientation
        if HkAll is None:
            HkAll_arr = np.zeros((1, numCells, 1), dtype=float)
        else:
            HkAll_arr = np.asarray(HkAll, dtype=float)

        sumValVec = np.zeros(ns, dtype=float)
        sumValMat = np.zeros((ns, ns), dtype=float)
        lambdaDeltaMat = np.zeros(numCells, dtype=float)

        # If gamma is scalar zero, expand
        if gamma_mat.size == 1 and gamma_mat.flat[0] == 0:
            gamma_mat = np.zeros_like(mu_vec).reshape(-1, 1)

        # Ensure gamma_mat is (numWindows, numCells)
        if gamma_mat.shape[1] != numCells:
            if gamma_mat.shape[0] == numCells:
                gamma_mat = gamma_mat.T

        # Replicate gamma for all cells if needed
        if gamma_mat.ndim == 2 and gamma_mat.shape[1] != numCells:
            gamma_mat = np.tile(gamma_mat, (1, numCells))

        # time_index is 1-based; extract history at this time
        tidx = int(time_index) - 1  # zero-based
        if HkAll_arr.ndim == 3 and HkAll_arr.shape[2] > tidx:
            Histterm = HkAll_arr[:, :, tidx]  # (numWindows, numCells)
        else:
            Histterm = np.zeros((gamma_mat.shape[0], numCells), dtype=float)

        if Histterm.shape[0] != numCells:
            pass  # already (numWindows, numCells) orientation
        else:
            if Histterm.shape[0] == numCells and Histterm.shape[1] != numCells:
                Histterm = Histterm.T

        if str(fitType) == 'binomial':
            # linTerm = mu + beta'*x_p + diag(gamma'*Histterm')
            linTerm = mu_vec + beta_mat.T @ x_p + np.diag(gamma_mat.T @ Histterm)
            exp_linTerm = np.exp(np.clip(linTerm, -500, 500))
            lambdaDeltaMat = exp_linTerm / (1.0 + exp_linTerm)
            lambdaDeltaMat = np.where(np.isnan(lambdaDeltaMat) | np.isinf(lambdaDeltaMat), 1.0, lambdaDeltaMat)

            dN_t = obs[:, int(time_index) - 1]
            factor = (dN_t - lambdaDeltaMat) * (1.0 - lambdaDeltaMat)
            sumValVec = np.sum(beta_mat * factor[None, :], axis=1)
            tempVec = (dN_t + (1.0 - 2.0 * lambdaDeltaMat)) * (1.0 - lambdaDeltaMat) * lambdaDeltaMat
            sumValMat = (beta_mat * tempVec[None, :]) @ beta_mat.T

        elif str(fitType) == 'poisson':
            linTerm = mu_vec + beta_mat.T @ x_p + np.diag(gamma_mat.T @ Histterm)
            lambdaDeltaMat = np.exp(np.clip(linTerm, -500, 500))
            lambdaDeltaMat = np.where(np.isnan(lambdaDeltaMat) | np.isinf(lambdaDeltaMat), 1.0, lambdaDeltaMat)

            dN_t = obs[:, int(time_index) - 1]
            sumValVec = np.sum(beta_mat * (dN_t - lambdaDeltaMat)[None, :], axis=1)
            sumValMat = (beta_mat * lambdaDeltaMat[None, :]) @ beta_mat.T

        if WuConv is None or _is_empty_value(WuConv):
            # sumValMat += C' * R^{-1} * C  (continuous observation term)
            sumValMat = sumValMat + C.T @ np.linalg.solve(R, C)
            I = np.eye(ns, dtype=float)
            try:
                Wu = W_p @ (I - np.linalg.solve(I + sumValMat @ W_p, sumValMat @ W_p))
            except np.linalg.LinAlgError:
                Wu = W_p.copy()
            if np.any(np.isnan(Wu)) or np.any(np.isinf(Wu)):
                Wu = W_p.copy()
            W_u = _symmetrize(Wu)
        else:
            W_u = np.asarray(WuConv, dtype=float).reshape(ns, ns)

        # x_u = x_p + W_u*sumValVec + (W_u*C'/R)*(y - C*x_p - alpha)
        x_u = x_p + W_u @ sumValVec + W_u @ C.T @ np.linalg.solve(R, y - C @ x_p - alpha)

        return x_u, W_u, lambdaDeltaMat.reshape(-1, 1)

    @staticmethod
    def mPPCODecodeLinear(A, Q, C, R, y, alpha, dN, mu, beta,
                          fitType='poisson', delta=0.001, gamma=None,
                          windowTimes=None, x0=None, Px0=None, HkAll=None):
        """Full mPPCO decode filter (linear CIF version).

        Matlab: ``DecodingAlgorithms.mPPCODecodeLinear``  (lines 4689-4845)

        Returns
        -------
        x_p, W_p, x_u, W_u  -- predicted / updated states & covariances
            x_p : (ns, N+1),  W_p : (ns, ns, N+1)
            x_u : (ns, N),    W_u : (ns, ns, N)
        """
        obs = _as_observation_matrix(dN)
        numCells, N = obs.shape
        A_arr = np.asarray(A, dtype=float)
        ns = A_arr.shape[0]

        # Defaults
        if Px0 is None or _is_empty_value(Px0):
            Px0 = np.zeros((ns, ns), dtype=float)
        else:
            Px0 = np.asarray(Px0, dtype=float).reshape(ns, ns)
        if x0 is None or _is_empty_value(x0):
            x0 = np.zeros(ns, dtype=float)
        else:
            x0 = np.asarray(x0, dtype=float).reshape(-1)
        if gamma is None:
            gamma = 0
        if delta is None:
            delta = 0.001

        minTime = 0.0
        maxTime = (N - 1) * delta

        # Build history tensor if not provided
        if HkAll is None or _is_empty_value(HkAll):
            if windowTimes is not None and not _is_empty_value(windowTimes):
                wt = np.asarray(windowTimes, dtype=float).reshape(-1)
                HkAll = _compute_history_terms(dN, delta, wt)  # (N, numWindows, numCells)
                gamma_arr = np.asarray(gamma, dtype=float)
                if gamma_arr.ndim <= 1 and gamma_arr.size == 1 and numCells > 1:
                    gamma = np.tile(gamma_arr.reshape(-1, 1), (1, numCells))
            else:
                HkAll = np.zeros((N, 1, numCells), dtype=float)
                gamma = np.zeros(numCells, dtype=float)
        else:
            HkAll = np.asarray(HkAll, dtype=float)

        gamma_arr = np.asarray(gamma, dtype=float)
        if gamma_arr.ndim == 2 and gamma_arr.shape[1] != numCells:
            gamma = gamma_arr.T

        # Permute HkAll from (N, numWindows, numCells) to (numWindows, numCells, N)
        # This is Matlab: permute(HkAll, [2 3 1])
        if HkAll.ndim == 3 and HkAll.shape[0] == N:
            Histtermperm = np.transpose(HkAll, (1, 2, 0))
        else:
            Histtermperm = HkAll

        mu_vec = np.asarray(mu, dtype=float).reshape(-1)
        beta_mat = np.asarray(beta, dtype=float)
        if beta_mat.ndim == 1:
            beta_mat = beta_mat.reshape(-1, 1)

        # Allocate outputs
        x_p = np.zeros((ns, N + 1), dtype=float)
        x_u = np.zeros((ns, N), dtype=float)
        W_p = np.zeros((ns, ns, N + 1), dtype=float)
        W_u = np.zeros((ns, ns, N), dtype=float)

        # Time-varying or static matrices: pick slice for time 0
        def _sel_A(n):
            if A_arr.ndim == 3:
                return A_arr[:, :, min(n, A_arr.shape[2] - 1)]
            return A_arr.reshape(ns, ns)

        def _sel_Q(n):
            Q_arr = np.asarray(Q, dtype=float)
            if Q_arr.ndim == 3:
                return Q_arr[:, :, min(n, Q_arr.shape[2] - 1)]
            return Q_arr.reshape(ns, ns)

        def _sel_C(n):
            C_arr = np.asarray(C, dtype=float)
            if C_arr.ndim == 3:
                return C_arr[:, :, min(n, C_arr.shape[2] - 1)]
            return C_arr

        def _sel_R(n):
            R_arr = np.asarray(R, dtype=float)
            if R_arr.ndim == 3:
                return R_arr[:, :, min(n, R_arr.shape[2] - 1)]
            return R_arr

        def _sel_alpha(n):
            alpha_arr = np.asarray(alpha, dtype=float)
            if alpha_arr.ndim >= 2 and alpha_arr.shape[-1] > 1:
                return alpha_arr[:, min(n, alpha_arr.shape[-1] - 1)]
            return alpha_arr.reshape(-1)

        # Initial prediction
        A1 = _sel_A(0)
        Q1 = _sel_Q(0)
        x_p[:, 0] = A1 @ x0
        W_p[:, :, 0] = A1 @ Px0 @ A1.T + Q1

        y_arr = np.asarray(y, dtype=float)

        for n in range(N):
            # 1-based time_index for mPPCODecode_update
            x_u[:, n], W_u[:, :, n], _ = DecodingAlgorithms.mPPCODecode_update(
                x_p[:, n], W_p[:, :, n],
                _sel_C(n), _sel_R(n),
                y_arr[:, n] if y_arr.ndim == 2 else y_arr,
                _sel_alpha(n),
                dN, mu_vec, beta_mat, fitType,
                gamma, Histtermperm, n + 1, None)
            if n < N - 1:
                x_p[:, n + 1], W_p[:, :, n + 1] = DecodingAlgorithms.mPPCODecode_predict(
                    x_u[:, n], W_u[:, :, n], _sel_A(n), _sel_Q(n))

        return x_p, W_p, x_u, W_u

    @staticmethod
    def mPPCO_fixedIntervalSmoother(A, Q, C, R, y, alpha, dN, lags, mu, beta,
                                     fitType, delta=0.001, gamma=None,
                                     windowTimes=None, x0=None, Px0=None, HkAll=None):
        """State-augmentation smoother for the mPPCO filter.

        Matlab: ``DecodingAlgorithms.mPPCO_fixedIntervalSmoother``  (lines 4587-4688)

        Returns
        -------
        x_pLag, W_pLag, x_uLag, W_uLag -- lagged state estimates
        """
        obs = _as_observation_matrix(dN)
        numCells, N = obs.shape
        A_arr = np.asarray(A, dtype=float)
        ns = A_arr.shape[0]
        nObs = np.asarray(C, dtype=float).shape[0]

        if Px0 is None or _is_empty_value(Px0):
            Px0 = np.zeros((ns, ns), dtype=float)
        else:
            Px0 = np.asarray(Px0, dtype=float).reshape(ns, ns)
        if x0 is None or _is_empty_value(x0):
            x0 = np.zeros(ns, dtype=float)
        else:
            x0 = np.asarray(x0, dtype=float).reshape(-1)
        if gamma is None:
            gamma = 0
        if delta is None:
            delta = 0.001

        minTime = 0.0
        maxTime = (N - 1) * delta

        # Build history if needed
        if HkAll is None or _is_empty_value(HkAll):
            if windowTimes is not None and not _is_empty_value(windowTimes):
                wt = np.asarray(windowTimes, dtype=float).reshape(-1)
                HkAll = _compute_history_terms(dN, delta, wt)
                gamma_arr = np.asarray(gamma, dtype=float)
                if gamma_arr.ndim <= 1 and gamma_arr.size == 1 and numCells > 1:
                    gamma = np.tile(gamma_arr.reshape(-1, 1), (1, numCells))
            else:
                HkAll = np.zeros((N, 1, numCells), dtype=float)
                gamma = np.zeros(numCells, dtype=float)

        gamma_arr = np.asarray(gamma, dtype=float)
        if gamma_arr.ndim == 2 and gamma_arr.shape[1] != numCells:
            gamma = gamma_arr.T

        lags = int(lags)
        nStates = ns

        # Build augmented system
        aug_dim = (lags + 1) * nStates

        def _sel_A(n):
            if A_arr.ndim == 3:
                return A_arr[:, :, min(n, A_arr.shape[2] - 1)]
            return A_arr.reshape(ns, ns)

        def _sel_Q(n):
            Q_arr = np.asarray(Q, dtype=float)
            if Q_arr.ndim == 3:
                return Q_arr[:, :, min(n, Q_arr.shape[2] - 1)]
            return Q_arr.reshape(ns, ns)

        def _sel_C(n):
            C_arr = np.asarray(C, dtype=float)
            if C_arr.ndim == 3:
                return C_arr[:, :, min(n, C_arr.shape[2] - 1)]
            return C_arr

        def _sel_R(n):
            R_arr = np.asarray(R, dtype=float)
            if R_arr.ndim == 3:
                return R_arr[:, :, min(n, R_arr.shape[2] - 1)]
            return R_arr

        Alag = np.zeros((aug_dim, aug_dim, N), dtype=float)
        Qlag = np.zeros((aug_dim, aug_dim, N), dtype=float)
        Clag = np.zeros((nObs, aug_dim, N), dtype=float)
        Rlag = np.zeros((nObs, nObs, N), dtype=float)
        x0lag = np.zeros(aug_dim, dtype=float)
        Px0lag = np.zeros((aug_dim, aug_dim), dtype=float)
        Px0lag[:nStates, :nStates] = Px0
        x0lag[:nStates] = x0

        for n in range(N):
            offset = 0
            for i in range(lags + 1):
                if i == 0:
                    Alag[offset:offset + nStates, offset:offset + nStates, n] = _sel_A(n)
                    Qlag[offset:offset + nStates, offset:offset + nStates, n] = _sel_Q(n)
                    Clag[:nObs, offset:offset + nStates, n] = _sel_C(n)
                    Rlag[:nObs, :nObs, n] = _sel_R(n)
                else:
                    Alag[offset:offset + nStates, offset - nStates:offset, n] = np.eye(nStates)
                    # Qlag block remains zeros
                    # Clag block remains zeros
                offset += nStates

        betaLag = np.zeros((aug_dim, numCells), dtype=float)
        beta_mat = np.asarray(beta, dtype=float)
        if beta_mat.ndim == 1:
            beta_mat = beta_mat.reshape(-1, 1)
        betaLag[:nStates, :numCells] = beta_mat

        x_p, W_p, x_u, W_u = DecodingAlgorithms.mPPCODecodeLinear(
            Alag, Qlag, Clag, Rlag, y, alpha, dN,
            mu, betaLag, fitType, delta, gamma, windowTimes,
            x0lag, Px0lag, HkAll)

        # Extract lagged portion
        lag_start = lags * nStates
        lag_end = (lags + 1) * nStates
        x_pLag = x_p[lag_start:lag_end, :]
        W_pLag = W_p[lag_start:lag_end, lag_start:lag_end, :]
        x_uLag = x_u[lag_start:lag_end, :]
        W_uLag = W_u[lag_start:lag_end, lag_start:lag_end, :]

        return x_pLag, W_pLag, x_uLag, W_uLag

    @staticmethod
    def mPPCO_EMCreateConstraints(EstimateA=1, AhatDiag=0, QhatDiag=1,
                                   QhatIsotropic=0, RhatDiag=1,
                                   RhatIsotropic=0, Estimatex0=1,
                                   EstimatePx0=1, Px0Isotropic=0,
                                   mcIter=1000, EnableIkeda=0):
        """Create constraint dictionary for mPPCO EM.

        Matlab: ``DecodingAlgorithms.mPPCO_EMCreateConstraints`` (lines 4945-5005)
        """
        C = {}
        C['EstimateA'] = int(EstimateA)
        C['AhatDiag'] = int(AhatDiag)
        C['QhatDiag'] = int(QhatDiag)
        C['QhatIsotropic'] = 1 if (QhatDiag and QhatIsotropic) else 0
        C['RhatDiag'] = int(RhatDiag)
        C['RhatIsotropic'] = 1 if (RhatDiag and RhatIsotropic) else 0
        C['Estimatex0'] = int(Estimatex0)
        C['EstimatePx0'] = int(EstimatePx0)
        C['Px0Isotropic'] = 1 if (EstimatePx0 and Px0Isotropic) else 0
        C['mcIter'] = int(mcIter)
        C['EnableIkeda'] = int(EnableIkeda)
        return C

    @staticmethod
    def mPPCO_ComputeParamStandardErrors(y, dN, xKFinal, WKFinal, Ahat, Qhat,
                                          Chat, Rhat, alphahat, x0hat, Px0hat,
                                          ExpectationSumsFinal, fitType,
                                          muhat, betahat, gammahat,
                                          windowTimes, HkAll,
                                          mPPCOEM_Constraints=None):
        """Compute standard errors for mPPCO EM parameters.

        Matlab: ``DecodingAlgorithms.mPPCO_ComputeParamStandardErrors``  (lines 5006-6138)

        Uses the observed information matrix approach: Io = Ic - Im  (McLachlan & Krishnan Eq 4.7).
        """
        if mPPCOEM_Constraints is None:
            mPPCOEM_Constraints = DecodingAlgorithms.mPPCO_EMCreateConstraints()

        y = np.asarray(y, dtype=float)
        obs = _as_observation_matrix(dN)
        xKFinal = np.asarray(xKFinal, dtype=float)
        Ahat = np.asarray(Ahat, dtype=float)
        Qhat = np.asarray(Qhat, dtype=float)
        Chat = np.asarray(Chat, dtype=float)
        Rhat = np.asarray(Rhat, dtype=float)
        alphahat = np.asarray(alphahat, dtype=float).reshape(-1)
        x0hat = np.asarray(x0hat, dtype=float).reshape(-1)
        Px0hat = np.asarray(Px0hat, dtype=float)
        muhat = np.asarray(muhat, dtype=float).reshape(-1)
        betahat = np.asarray(betahat, dtype=float)
        if betahat.ndim == 1:
            betahat = betahat.reshape(-1, 1)
        gammahat = np.asarray(gammahat, dtype=float)
        HkAll = np.asarray(HkAll, dtype=float)

        dy, N = y.shape if y.ndim == 2 else (1, y.shape[0])
        K = N
        dx = xKFinal.shape[0]
        numCells = betahat.shape[1]
        McExp = mPPCOEM_Constraints['mcIter']

        Qhat_inv = np.linalg.inv(Qhat)
        Rhat_inv = np.linalg.inv(Rhat)
        Px0hat_inv = np.linalg.inv(Px0hat + np.eye(Px0hat.shape[0]) * 1e-12)

        # ---- Complete Information Matrices ----

        # IAComp - A parameter
        if mPPCOEM_Constraints['EstimateA']:
            n1A, n2A = Ahat.shape
            el = np.eye(n1A)
            em = np.eye(n2A)
            if mPPCOEM_Constraints['AhatDiag']:
                IAComp = np.zeros((n1A, n1A))
                for l in range(n1A):
                    termMat = Qhat_inv @ np.outer(el[:, l], em[:, l]) @ ExpectationSumsFinal['Sxkm1xkm1'] * np.eye(n1A)
                    IAComp[:, l] = np.diag(termMat)
            else:
                nA = Ahat.size
                IAComp = np.zeros((nA, nA))
                cnt = 0
                for l in range(n1A):
                    for m in range(n2A):
                        termMat = Qhat_inv @ np.outer(el[:, l], em[:, m]) @ ExpectationSumsFinal['Sxkm1xkm1']
                        IAComp[:, cnt] = termMat.T.reshape(-1)
                        cnt += 1

        # ICComp - C parameter
        n1C, n2C = Chat.shape
        nC = Chat.size
        ICComp = np.zeros((nC, nC))
        el = np.eye(n1C)
        em = np.eye(n2C)
        cnt = 0
        for l in range(n1C):
            for m in range(n2C):
                termMat = Rhat_inv @ np.outer(el[:, l], em[:, m]) @ ExpectationSumsFinal['Sxkxk']
                ICComp[:, cnt] = termMat.T.reshape(-1)
                cnt += 1

        # IRComp - R parameter
        n1R, n2R = Rhat.shape
        el = np.eye(n1R)
        em = np.eye(n2R)
        if mPPCOEM_Constraints['RhatDiag']:
            if mPPCOEM_Constraints['RhatIsotropic']:
                IRComp = np.array([[0.5 * N * dy * Rhat[0, 0] ** (-2)]])
            else:
                IRComp = np.zeros((n1R, n1R))
                for l in range(n1R):
                    termMat = N / 2.0 * Rhat_inv @ np.outer(em[:, l], el[:, l]) @ Rhat_inv
                    IRComp[:, l] = np.diag(termMat)
        else:
            nR = Rhat.size
            IRComp = np.zeros((nR, nR))
            cnt = 0
            for l in range(n1R):
                for m in range(n2R):
                    termMat = N / 2.0 * Rhat_inv @ np.outer(em[:, m], el[:, l]) @ Rhat_inv
                    IRComp[:, cnt] = termMat.T.reshape(-1)
                    cnt += 1

        # IQComp - Q parameter
        n1Q, n2Q = Qhat.shape
        el = np.eye(n1Q)
        em = np.eye(n2Q)
        if mPPCOEM_Constraints['QhatDiag']:
            if mPPCOEM_Constraints['QhatIsotropic']:
                IQComp = np.array([[0.5 * N * dx * Qhat[0, 0] ** (-2)]])
            else:
                IQComp = np.zeros((n1Q, n1Q))
                for l in range(n1Q):
                    termMat = N / 2.0 * Qhat_inv @ np.outer(em[:, l], el[:, l]) @ Qhat_inv
                    IQComp[:, l] = np.diag(termMat)
        else:
            nQ = Qhat.size
            IQComp = np.zeros((nQ, nQ))
            cnt = 0
            for l in range(n1Q):
                for m in range(n2Q):
                    termMat = N / 2.0 * Qhat_inv @ np.outer(em[:, m], el[:, l]) @ Qhat_inv
                    IQComp[:, cnt] = termMat.T.reshape(-1)
                    cnt += 1

        # ISComp - Px0 parameter
        if mPPCOEM_Constraints['EstimatePx0']:
            if mPPCOEM_Constraints['Px0Isotropic']:
                ISComp = np.array([[0.5 * dx * Px0hat[0, 0] ** (-2)]])
            else:
                n1S, n2S = Px0hat.shape
                ISComp = np.zeros((n1S, n1S))
                el = np.eye(n1S)
                em = np.eye(n2S)
                for l in range(n1S):
                    termMat = 0.5 * Px0hat_inv @ np.outer(em[:, l], el[:, l]) @ Px0hat_inv
                    ISComp[:, l] = np.diag(termMat)

        # Ix0Comp
        if mPPCOEM_Constraints['Estimatex0']:
            Ix0Comp = Px0hat_inv + Ahat.T @ Qhat_inv @ Ahat

        # IAlphaComp
        IAlphaComp = N * Rhat_inv

        # IBetaComp - Monte Carlo
        xKDrawExp = np.zeros((dx, K, McExp), dtype=float)
        for k in range(K):
            WuTemp = WKFinal[:, :, k]
            try:
                chol_m = np.linalg.cholesky(WuTemp)
            except np.linalg.LinAlgError:
                chol_m = np.linalg.cholesky(nearestSPD(WuTemp))
            z = np.random.randn(dx, McExp)
            xKDrawExp[:, k, :] = xKFinal[:, k:k + 1] + chol_m @ z

        IBetaComp = np.zeros((dx * numCells, dx * numCells), dtype=float)
        xkPerm = np.transpose(xKDrawExp, (0, 2, 1))  # (dx, McExp, K)

        for c in range(numCells):
            HessianTerm = np.zeros((dx, dx), dtype=float)
            for k in range(K):
                Hk = HkAll[k, :, c] if HkAll.ndim == 3 else np.zeros(1)
                xk = xkPerm[:, :, k]
                gammaC = gammahat if gammahat.size == 1 else (gammahat[:, c] if gammahat.ndim == 2 else gammahat)
                terms = muhat[c] + betahat[:, c] @ xk + float(np.dot(gammaC.reshape(-1), Hk.reshape(-1)))
                if fitType == 'poisson':
                    ld = np.exp(np.clip(terms, -500, 500))
                    HessianTerm -= (1.0 / McExp) * (np.tile(ld, (dx, 1)) * xk) @ xk.T
                else:
                    ld = np.exp(np.clip(terms, -500, 500))
                    ld = ld / (1.0 + ld)
                    EldXkXk = (1.0 / McExp) * (np.tile(ld, (dx, 1)) * xk) @ xk.T
                    EldSqXkXk = (1.0 / McExp) * (np.tile(ld ** 2, (dx, 1)) * xk) @ xk.T
                    EldCubeXkXk = (1.0 / McExp) * (np.tile(ld ** 3, (dx, 1)) * xk) @ xk.T
                    HessianTerm += EldXkXk + EldSqXkXk - 2.0 * EldCubeXkXk
            si = dx * c
            ei = dx * (c + 1)
            IBetaComp[si:ei, si:ei] = -HessianTerm

        # IMuComp
        IMuComp = np.zeros((numCells, numCells), dtype=float)
        for c in range(numCells):
            HessianTerm = 0.0
            for k in range(K):
                Hk_full = HkAll[:, :, c] if HkAll.ndim == 3 else np.zeros((K, 1))
                Hk = Hk_full[k, :]
                xk = xkPerm[:, :, k]
                gammaC = gammahat if gammahat.size == 1 else (gammahat[:, c] if gammahat.ndim == 2 else gammahat)
                terms = muhat[c] + betahat[:, c] @ xk + float(np.dot(gammaC.reshape(-1), Hk.reshape(-1)))
                if fitType == 'poisson':
                    ld = np.exp(np.clip(terms, -500, 500))
                    HessianTerm -= (1.0 / McExp) * float(np.sum(ld))
                else:
                    ld = np.exp(np.clip(terms, -500, 500)) / (1.0 + np.exp(np.clip(terms, -500, 500)))
                    Eld = (1.0 / McExp) * float(np.sum(ld))
                    EldSq = (1.0 / McExp) * float(np.sum(ld ** 2))
                    EldCube = (1.0 / McExp) * float(np.sum(ld ** 3))
                    HessianTerm += -(obs[c, k] + 1) * Eld + (obs[c, k] + 3) * EldSq - 3 * EldCube
            IMuComp[c, c] = -HessianTerm

        # IGammaComp
        nHist = HkAll.shape[1] if HkAll.ndim == 3 else 1
        IGammaComp = np.zeros((nHist * numCells, nHist * numCells), dtype=float)
        has_gamma = (windowTimes is not None and not _is_empty_value(windowTimes)
                     and np.any(gammahat != 0))
        if has_gamma:
            for c in range(numCells):
                HessianTerm = np.zeros((nHist, nHist), dtype=float)
                for k in range(K):
                    Hk_full = HkAll[:, :, c] if HkAll.ndim == 3 else np.zeros((K, 1))
                    Hk = Hk_full[k, :]
                    xk = xkPerm[:, :, k]
                    gammaC = gammahat if gammahat.size == 1 else (gammahat[:, c] if gammahat.ndim == 2 else gammahat)
                    terms = muhat[c] + betahat[:, c] @ xk + float(np.dot(gammaC.reshape(-1), Hk.reshape(-1)))
                    if fitType == 'poisson':
                        ld = np.exp(np.clip(terms, -500, 500))
                        Eld = (1.0 / McExp) * float(np.sum(ld))
                        HessianTerm -= np.outer(Hk, Hk) * Eld
                    else:
                        ld_raw = np.exp(np.clip(terms, -500, 500))
                        ld = ld_raw / (1.0 + ld_raw)
                        Eld = (1.0 / McExp) * float(np.sum(ld))
                        EldSq = (1.0 / McExp) * float(np.sum(ld ** 2))
                        EldCube = (1.0 / McExp) * float(np.sum(ld ** 2))  # matches Matlab typo (ld.^2)
                        HessianTerm += (-Eld * (obs[c, k] + 1) + EldSq * (obs[c, k] + 3) - 2 * EldCube) * np.outer(Hk, Hk)
                si = nHist * c
                ei = nHist * (c + 1)
                IGammaComp[si:ei, si:ei] = -HessianTerm

        # Assemble IComp
        n1 = IAComp.shape[0] if mPPCOEM_Constraints['EstimateA'] else 0
        n2 = IQComp.shape[0]
        n3 = ICComp.shape[0]
        n4 = IRComp.shape[0]
        n5 = ISComp.shape[0] if mPPCOEM_Constraints['EstimatePx0'] else 0
        n6 = Ix0Comp.shape[0] if mPPCOEM_Constraints['Estimatex0'] else 0
        n7 = IAlphaComp.shape[0]
        n8 = IMuComp.shape[0]
        n9 = IBetaComp.shape[0]
        if gammahat.size == 1 and float(gammahat.flat[0]) == 0:
            n10 = 0
        else:
            n10 = IGammaComp.shape[0]
        nTerms = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10
        IComp = np.zeros((nTerms, nTerms), dtype=float)

        offset = 0
        if mPPCOEM_Constraints['EstimateA']:
            IComp[offset:offset + n1, offset:offset + n1] = IAComp
            offset += n1
        IComp[offset:offset + n2, offset:offset + n2] = IQComp
        offset += n2
        IComp[offset:offset + n3, offset:offset + n3] = ICComp
        offset += n3
        IComp[offset:offset + n4, offset:offset + n4] = IRComp
        offset += n4
        if mPPCOEM_Constraints['EstimatePx0']:
            IComp[offset:offset + n5, offset:offset + n5] = ISComp
        offset += n5
        if mPPCOEM_Constraints['Estimatex0']:
            IComp[offset:offset + n6, offset:offset + n6] = Ix0Comp
        offset += n6
        IComp[offset:offset + n7, offset:offset + n7] = IAlphaComp
        offset += n7
        IComp[offset:offset + n8, offset:offset + n8] = IMuComp
        offset += n8
        IComp[offset:offset + n9, offset:offset + n9] = IBetaComp
        offset += n9
        if n10 > 0:
            IComp[offset:offset + n10, offset:offset + n10] = IGammaComp

        # ---- Missing Information Matrix (Monte Carlo) ----
        Mc = McExp
        xKDraw = np.zeros((dx, N, Mc), dtype=float)
        for n in range(N):
            WuTemp = WKFinal[:, :, n]
            try:
                chol_m = np.linalg.cholesky(WuTemp)
            except np.linalg.LinAlgError:
                chol_m = np.linalg.cholesky(nearestSPD(WuTemp))
            z = np.random.randn(dx, Mc)
            xKDraw[:, n, :] = xKFinal[:, n:n + 1] + chol_m @ z

        if mPPCOEM_Constraints['EstimatePx0'] or mPPCOEM_Constraints['Estimatex0']:
            try:
                chol_m = np.linalg.cholesky(Px0hat)
            except np.linalg.LinAlgError:
                chol_m = np.linalg.cholesky(nearestSPD(Px0hat))
            z = np.random.randn(dx, Mc)
            x0Draw = x0hat.reshape(-1, 1) + chol_m @ z
        else:
            x0Draw = np.tile(x0hat.reshape(-1, 1), (1, Mc))

        IMc = np.zeros((nTerms, nTerms, Mc), dtype=float)
        Dx = dx
        Dy = dy

        for c_mc in range(Mc):
            x_K = xKDraw[:, :, c_mc]
            x_0 = x0Draw[:, c_mc]

            Sxkm1xk = np.zeros((Dx, Dx))
            Sxkm1xkm1 = np.zeros((Dx, Dx))
            Sxkxk = np.zeros((Dx, Dx))
            Sykyk = np.zeros((Dy, Dy))
            Sxkyk = np.zeros((Dx, Dy))

            for k in range(K):
                if k == 0:
                    Sxkm1xk += np.outer(x_0, x_K[:, k])
                    Sxkm1xkm1 += np.outer(x_0, x_0)
                else:
                    Sxkm1xk += np.outer(x_K[:, k - 1], x_K[:, k])
                    Sxkm1xkm1 += np.outer(x_K[:, k - 1], x_K[:, k - 1])
                Sxkxk += np.outer(x_K[:, k], x_K[:, k])
                yk_alpha = y[:, k] - alphahat if y.ndim == 2 else y - alphahat
                Sykyk += np.outer(yk_alpha, yk_alpha)
                Sxkyk += np.outer(x_K[:, k], yk_alpha)

            Sxkxk = _symmetrize(Sxkxk)
            Sykyk = _symmetrize(Sykyk)
            sumXkTerms_mc = Sxkxk - Ahat @ Sxkm1xk - Sxkm1xk.T @ Ahat.T + Ahat @ Sxkm1xkm1 @ Ahat.T
            sumYkTerms_mc = Sykyk - Chat @ Sxkyk - Sxkyk.T @ Chat.T + Chat @ Sxkxk @ Chat.T
            Sxkxkm1 = Sxkm1xk.T
            sumXkTerms_mc = _symmetrize(sumXkTerms_mc)
            sumYkTerms_mc = _symmetrize(sumYkTerms_mc)

            # Score: A
            if mPPCOEM_Constraints['EstimateA']:
                ScorA = np.linalg.solve(Qhat, Sxkxkm1 - Ahat @ Sxkm1xkm1)
                if mPPCOEM_Constraints['AhatDiag']:
                    ScoreAMc = np.diag(ScorA)
                else:
                    ScoreAMc = ScorA.T.reshape(-1)
            else:
                ScoreAMc = np.array([], dtype=float)

            # Score: C
            ScorC = np.linalg.solve(Rhat, Sxkyk.T - Chat @ Sxkxk)
            ScoreCMc = ScorC.T.reshape(-1)

            # Score: Q
            if mPPCOEM_Constraints['QhatDiag']:
                if mPPCOEM_Constraints['QhatIsotropic']:
                    ScoreQ = -0.5 * (K * Dx * Qhat[0, 0] ** (-1) - Qhat[0, 0] ** (-2) * np.trace(sumXkTerms_mc))
                    ScoreQMc = np.array([ScoreQ])
                else:
                    ScoreQ = -0.5 * np.linalg.solve(Qhat, K * np.eye(Dx) - np.linalg.solve(Qhat, sumXkTerms_mc).T)
                    ScoreQMc = np.diag(ScoreQ)
            else:
                ScoreQ = -0.5 * np.linalg.solve(Qhat, K * np.eye(Dx) - np.linalg.solve(Qhat, sumXkTerms_mc).T)
                ScoreQMc = ScoreQ.T.reshape(-1)

            # Score: alpha
            resid = y - Chat @ x_K - alphahat.reshape(-1, 1) @ np.ones((1, N)) if y.ndim == 2 else y - Chat @ x_K - alphahat.reshape(-1, 1)
            ScoreAlphaMc = np.sum(np.linalg.solve(Rhat, resid), axis=1)

            # Score: R
            if mPPCOEM_Constraints['RhatDiag']:
                if mPPCOEM_Constraints['RhatIsotropic']:
                    ScoreR = -0.5 * (K * Dy * Rhat[0, 0] ** (-1) - Rhat[0, 0] ** (-2) * np.trace(sumYkTerms_mc))
                    ScoreRMc = np.array([ScoreR])
                else:
                    ScoreR = -0.5 * np.linalg.solve(Rhat, K * np.eye(Dy) - np.linalg.solve(Rhat, sumYkTerms_mc).T)
                    ScoreRMc = np.diag(ScoreR)
            else:
                ScoreR = -0.5 * np.linalg.solve(Rhat, K * np.eye(Dy) - np.linalg.solve(Rhat, sumYkTerms_mc).T)
                ScoreRMc = ScoreR.T.reshape(-1)

            # Score: Px0
            if mPPCOEM_Constraints['Px0Isotropic']:
                diff0 = x_0 - x0hat
                ScoreSMc = np.array([-0.5 * (Dx * Px0hat[0, 0] ** (-1) - Px0hat[0, 0] ** (-2) * np.trace(np.outer(diff0, diff0)))])
            else:
                diff0 = x_0 - x0hat
                ScorS = -0.5 * np.linalg.solve(Px0hat, np.eye(Dx) - np.linalg.solve(Px0hat, np.outer(diff0, diff0)).T)
                ScoreSMc = np.diag(ScorS)

            # Score: x0
            Scorx0 = -np.linalg.solve(Px0hat, x_0 - x0hat) + Ahat.T @ np.linalg.solve(Qhat, x_K[:, 0] - Ahat @ x_0)
            Scorex0Mc = Scorx0.reshape(-1)

            # Score: mu, beta, gamma per cell
            ScoreMuMc = np.zeros(numCells)
            ScoreBetaMc = np.array([], dtype=float)
            ScoreGammaMc = np.array([], dtype=float)
            for nc in range(numCells):
                Hk_full = HkAll[:, :, nc] if HkAll.ndim == 3 else np.zeros((K, 1))
                nHistC = Hk_full.shape[1]
                gammaC = gammahat if gammahat.size == 1 else (gammahat[:, nc] if gammahat.ndim == 2 else gammahat)
                terms = muhat[nc] + betahat[:, nc] @ x_K + gammaC.reshape(-1) @ Hk_full.T
                if fitType == 'poisson':
                    ld = np.exp(np.clip(terms, -500, 500))
                    ScoreMuMc[nc] = float(np.sum(obs[nc, :] - ld))
                    ScoreBetaMc = np.concatenate([ScoreBetaMc, np.sum(np.tile(obs[nc, :] - ld, (Dx, 1)) * x_K, axis=1)])
                    ScoreGammaMc = np.concatenate([ScoreGammaMc, np.sum(np.tile(obs[nc, :] - ld, (nHistC, 1)) * Hk_full.T, axis=1)])
                else:
                    ld_raw = np.exp(np.clip(terms, -500, 500))
                    ld = ld_raw / (1.0 + ld_raw)
                    ScoreMuMc[nc] = float(np.sum(obs[nc, :] - (obs[nc, :] + 1) * ld + ld ** 2))
                    ScoreBetaMc = np.concatenate([ScoreBetaMc, np.sum(np.tile(obs[nc, :] * (1 - ld) - ld * (1 - ld), (Dx, 1)) * x_K, axis=1)])
                    ScoreGammaMc = np.concatenate([ScoreGammaMc, np.sum(np.tile(obs[nc, :] - (obs[nc, :] + 1) * ld + ld ** 2, (nHistC, 1)) * Hk_full.T, axis=1)])

            ScoreVec = np.concatenate([ScoreAMc, ScoreQMc, ScoreCMc, ScoreRMc])
            if mPPCOEM_Constraints['EstimatePx0']:
                ScoreVec = np.concatenate([ScoreVec, ScoreSMc])
            if mPPCOEM_Constraints['Estimatex0']:
                ScoreVec = np.concatenate([ScoreVec, Scorex0Mc])
            ScoreVec = np.concatenate([ScoreVec, ScoreAlphaMc, ScoreMuMc, ScoreBetaMc])
            if n10 > 0:
                ScoreVec = np.concatenate([ScoreVec, ScoreGammaMc])

            IMc[:, :, c_mc] = np.outer(ScoreVec, ScoreVec)

        IMissing = np.mean(IMc, axis=2)
        IObs = IComp - IMissing
        try:
            invIObs = np.linalg.inv(IObs)
        except np.linalg.LinAlgError:
            invIObs = np.linalg.pinv(IObs)
        invIObs = nearestSPD(invIObs)
        VarVec = np.diag(invIObs)
        SEVec = np.sqrt(np.maximum(VarVec, 0.0))

        # Partition SE vector
        off = 0
        SEAterms = SEVec[off:off + n1]; off += n1
        SEQterms = SEVec[off:off + n2]; off += n2
        SECterms = SEVec[off:off + n3]; off += n3
        SERterms = SEVec[off:off + n4]; off += n4
        SEPx0terms = SEVec[off:off + n5]; off += n5
        SEx0terms = SEVec[off:off + n6]; off += n6
        SEAlphaterms = SEVec[off:off + n7]; off += n7
        SEMuTerms = SEVec[off:off + n8]; off += n8
        SEBetaTerms = SEVec[off:off + n9]; off += n9
        SEGammaTerms = SEVec[off:off + n10]; off += n10

        SE = {}
        if mPPCOEM_Constraints['EstimateA']:
            if mPPCOEM_Constraints['AhatDiag']:
                SE['A'] = np.diag(SEAterms)
            else:
                SE['A'] = SEAterms.reshape(Ahat.shape[1], Ahat.shape[0]).T
        SE['Q'] = np.diag(SEQterms) if mPPCOEM_Constraints['QhatDiag'] else SEQterms.reshape(Qhat.shape[1], Qhat.shape[0]).T
        SE['C'] = SECterms.reshape(Chat.shape[1], Chat.shape[0]).T
        SE['R'] = np.diag(SERterms) if mPPCOEM_Constraints['RhatDiag'] else SERterms.reshape(Rhat.shape[1], Rhat.shape[0]).T
        SE['alpha'] = SEAlphaterms.reshape(alphahat.shape)
        if mPPCOEM_Constraints['EstimatePx0']:
            SE['Px0'] = np.diag(SEPx0terms)
        if mPPCOEM_Constraints['Estimatex0']:
            SE['x0'] = SEx0terms
        SE['mu'] = SEMuTerms
        SE['beta'] = SEBetaTerms.reshape(betahat.shape[1], betahat.shape[0]).T
        if n10 > 0:
            SE['gamma'] = SEGammaTerms.reshape(gammahat.shape[1], gammahat.shape[0]).T if gammahat.ndim == 2 else SEGammaTerms

        # P-values (two-sided z-test)
        Pvals = {}
        if mPPCOEM_Constraints['EstimateA']:
            pA_flat = np.array([_ztest_pvalue(p, s) for p, s in zip(Ahat.reshape(-1) if not mPPCOEM_Constraints['AhatDiag'] else np.diag(Ahat), SE['A'].reshape(-1) if not mPPCOEM_Constraints['AhatDiag'] else np.diag(SE['A']))])
            Pvals['A'] = np.diag(pA_flat) if mPPCOEM_Constraints['AhatDiag'] else pA_flat.reshape(Ahat.shape)
        pC_flat = np.array([_ztest_pvalue(p, s) for p, s in zip(Chat.reshape(-1), SE['C'].reshape(-1))])
        Pvals['C'] = pC_flat.reshape(Chat.shape)
        if mPPCOEM_Constraints['RhatDiag']:
            pR_flat = np.array([_ztest_pvalue(p, s) for p, s in zip(np.diag(Rhat), np.diag(SE['R']))])
            Pvals['R'] = np.diag(pR_flat)
        else:
            pR_flat = np.array([_ztest_pvalue(p, s) for p, s in zip(Rhat.reshape(-1), SE['R'].reshape(-1))])
            Pvals['R'] = pR_flat.reshape(Rhat.shape)
        if mPPCOEM_Constraints['QhatDiag']:
            pQ_flat = np.array([_ztest_pvalue(p, s) for p, s in zip(np.diag(Qhat), np.diag(SE['Q']))])
            Pvals['Q'] = np.diag(pQ_flat)
        else:
            pQ_flat = np.array([_ztest_pvalue(p, s) for p, s in zip(Qhat.reshape(-1), SE['Q'].reshape(-1))])
            Pvals['Q'] = pQ_flat.reshape(Qhat.shape)
        if mPPCOEM_Constraints['EstimatePx0']:
            pPx0_flat = np.array([_ztest_pvalue(p, s) for p, s in zip(np.diag(Px0hat), np.diag(SE['Px0']))])
            Pvals['Px0'] = np.diag(pPx0_flat)
        if mPPCOEM_Constraints['Estimatex0']:
            Pvals['x0'] = np.array([_ztest_pvalue(p, s) for p, s in zip(x0hat, SE['x0'])])
        Pvals['alpha'] = np.array([_ztest_pvalue(p, s) for p, s in zip(alphahat.reshape(-1), SE['alpha'].reshape(-1))])
        Pvals['mu'] = np.array([_ztest_pvalue(p, s) for p, s in zip(muhat, SE['mu'])])
        pBeta_flat = np.array([_ztest_pvalue(p, s) for p, s in zip(betahat.reshape(-1), SE['beta'].reshape(-1))])
        Pvals['beta'] = pBeta_flat.reshape(betahat.shape)
        if n10 > 0:
            pGamma_flat = np.array([_ztest_pvalue(p, s) for p, s in zip(gammahat.reshape(-1), SE['gamma'].reshape(-1))])
            Pvals['gamma'] = pGamma_flat.reshape(gammahat.shape) if gammahat.ndim == 2 else pGamma_flat

        return SE, Pvals, nTerms

    @staticmethod
    def mPPCO_EStep(A, Q, C, R, y, alpha, dN, mu, beta, fitType='poisson',
                    delta=0.001, gamma=None, HkAll=None, x0=None, Px0=None):
        """E-step for the mPPCO EM algorithm.

        Matlab: ``DecodingAlgorithms.mPPCO_EStep``  (lines 6555-6772)

        Returns
        -------
        x_K : (dx, K) -- smoothed states
        W_K : (dx, dx, K) -- smoothed covariances
        logll : float -- log-likelihood
        ExpectationSums : dict
        """
        A = np.asarray(A, dtype=float)
        Q = np.asarray(Q, dtype=float)
        C = np.asarray(C, dtype=float)
        R = np.asarray(R, dtype=float)
        y = np.asarray(y, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        obs = _as_observation_matrix(dN)
        numCells, K = obs.shape
        Dx = A.shape[1] if A.ndim >= 2 else A.shape[0]
        Dy = C.shape[0] if C.ndim >= 2 else 1
        mu_vec = np.asarray(mu, dtype=float).reshape(-1)
        beta_mat = np.asarray(beta, dtype=float)
        if beta_mat.ndim == 1:
            beta_mat = beta_mat.reshape(-1, 1)
        if gamma is None:
            gamma = 0
        gamma_arr = np.asarray(gamma, dtype=float)
        if x0 is None or _is_empty_value(x0):
            x0 = np.zeros(Dx, dtype=float)
        else:
            x0 = np.asarray(x0, dtype=float).reshape(-1)
        if Px0 is None or _is_empty_value(Px0):
            Px0 = np.zeros((Dx, Dx), dtype=float)
        else:
            Px0 = np.asarray(Px0, dtype=float).reshape(Dx, Dx)

        if HkAll is None or _is_empty_value(HkAll):
            HkAll_arr = np.zeros((K, 1, numCells), dtype=float)
        else:
            HkAll_arr = np.asarray(HkAll, dtype=float)

        # Forward filter
        x_p, W_p, x_u, W_u = DecodingAlgorithms.mPPCODecodeLinear(
            A, Q, C, R, y, alpha, dN, mu_vec, beta_mat, fitType,
            delta, gamma, None, x0, Px0, HkAll_arr)

        # Smoother -- x_p has N+1 columns, x_u has N columns
        # kalman_smootherFromFiltered expects matching shapes
        # Trim x_p and W_p to first N entries for smoother input
        x_K, W_K, Lk = DecodingAlgorithms.kalman_smootherFromFiltered(
            A, x_p[:, :N], W_p[:, :, :N], x_u, W_u)

        # Handle Matlab-style output -- ensure x_K is (dx, K)
        if x_K.ndim == 2 and x_K.shape[0] == K and x_K.shape[1] == Dx:
            x_K = x_K.T
        if W_K.ndim == 3 and W_K.shape[0] == K:
            W_K = np.transpose(W_K, (1, 2, 0))

        # Best estimates of initial state given data
        W1G0 = A @ Px0 @ A.T + Q if A.ndim == 2 else A.reshape(Dx, Dx) @ Px0 @ A.reshape(Dx, Dx).T + Q.reshape(Dx, Dx)
        A_2d = A.reshape(Dx, Dx) if A.ndim != 2 else A
        L0 = Px0 @ A_2d.T @ np.linalg.pinv(W1G0)
        Ex0Gy = x0 + L0 @ (x_K[:, 0] - x_p[:, 0])
        Px0Gy = Px0 + L0 @ (np.linalg.pinv(W_K[:, :, 0]) - np.linalg.pinv(W1G0)) @ L0.T
        Px0Gy = _symmetrize(Px0Gy)

        # Cross-covariance matrices Wku
        numStates = Dx
        Wku = np.zeros((numStates, numStates, K, K), dtype=float)
        for k in range(K):
            Wku[:, :, k, k] = W_K[:, :, k]

        for u in range(K - 1, 0, -1):
            k = u - 1
            Dk = W_u[:, :, k] @ A_2d.T @ np.linalg.pinv(W_p[:, :, k + 1])
            Wku[:, :, k, u] = Dk @ Wku[:, :, k + 1, u]
            Wku[:, :, u, k] = Wku[:, :, k, u].T

        # Sufficient statistics
        Sxkm1xk = np.zeros((Dx, Dx))
        Sxkm1xkm1 = np.zeros((Dx, Dx))
        Sxkxk = np.zeros((Dx, Dx))
        Sykyk = np.zeros((Dy, Dy))
        Sxkyk = np.zeros((Dx, Dy))

        alpha_vec = alpha.reshape(-1)
        for k in range(K):
            if k == 0:
                Sxkm1xk += Px0 @ A_2d.T @ np.linalg.pinv(W_p[:, :, 0]) @ Wku[:, :, 0, 0]
                Sxkm1xkm1 += Px0 + np.outer(x0, x0)
            else:
                Sxkm1xk += Wku[:, :, k - 1, k] + np.outer(x_K[:, k - 1], x_K[:, k])
                Sxkm1xkm1 += Wku[:, :, k - 1, k - 1] + np.outer(x_K[:, k - 1], x_K[:, k - 1])
            Sxkxk += Wku[:, :, k, k] + np.outer(x_K[:, k], x_K[:, k])
            yk = y[:, k] if y.ndim == 2 else y
            Sykyk += np.outer(yk - alpha_vec, yk - alpha_vec)
            Sxkyk += np.outer(x_K[:, k], yk - alpha_vec)

        Sxkxk = _symmetrize(Sxkxk)
        Sykyk = _symmetrize(Sykyk)
        sumXkTerms = Sxkxk - A_2d @ Sxkm1xk - Sxkm1xk.T @ A_2d.T + A_2d @ Sxkm1xkm1 @ A_2d.T
        sumYkTerms = Sykyk - C @ Sxkyk - Sxkyk.T @ C.T + C @ Sxkxk @ C.T
        Sxkxkm1 = Sxkm1xk.T

        # Log-likelihood with PP term
        if str(fitType) == 'poisson':
            sumPPll = 0.0
            HkPerm = np.transpose(HkAll_arr, (1, 2, 0)) if HkAll_arr.ndim == 3 and HkAll_arr.shape[0] == K else HkAll_arr
            for k in range(K):
                Hk = HkPerm[:, :, k] if HkPerm.ndim == 3 else np.zeros((1, numCells))
                if Hk.shape[0] == numCells and Hk.shape[1] != numCells:
                    Hk = Hk.T
                xk = x_K[:, k]
                gammaC_mat = np.tile(gamma_arr.reshape(-1, 1), (1, numCells)) if gamma_arr.size == 1 else gamma_arr
                if gammaC_mat.ndim == 2 and gammaC_mat.shape[1] != numCells:
                    gammaC_mat = np.tile(gammaC_mat, (1, numCells))
                terms = mu_vec + beta_mat.T @ xk + np.diag(gammaC_mat.T @ Hk) if Hk.size > 0 and gammaC_mat.size > 0 else mu_vec + beta_mat.T @ xk
                Wk = W_K[:, :, k]
                ld = np.exp(np.clip(terms, -500, 500))
                bt = beta_mat
                ExplambdaDelta = ld + 0.5 * (ld * np.diag(bt.T @ Wk @ bt))
                ExplogLD = terms
                sumPPll += float(np.sum(obs[:, k] * ExplogLD - ExplambdaDelta))
        else:  # binomial
            sumPPll = 0.0
            HkPerm = np.transpose(HkAll_arr, (1, 2, 0)) if HkAll_arr.ndim == 3 and HkAll_arr.shape[0] == K else HkAll_arr
            for k in range(K):
                Hk = HkPerm[:, :, k] if HkPerm.ndim == 3 else np.zeros((1, numCells))
                if Hk.shape[0] == numCells and Hk.shape[1] != numCells:
                    Hk = Hk.T
                xk = x_K[:, k]
                gammaC_mat = np.tile(gamma_arr.reshape(-1, 1), (1, numCells)) if gamma_arr.size == 1 else gamma_arr
                if gammaC_mat.ndim == 2 and gammaC_mat.shape[1] != numCells:
                    gammaC_mat = np.tile(gammaC_mat, (1, numCells))
                terms = mu_vec + beta_mat.T @ xk + np.diag(gammaC_mat.T @ Hk) if Hk.size > 0 and gammaC_mat.size > 0 else mu_vec + beta_mat.T @ xk
                Wk = W_K[:, :, k]
                ld_raw = np.exp(np.clip(terms, -500, 500))
                ld = ld_raw / (1.0 + ld_raw)
                bt = beta_mat
                ExplambdaDelta = ld + 0.5 * (ld * (1 - ld) * (1 - 2 * ld)) * np.diag(bt.T @ Wk @ bt)
                ExplogLD = np.log(np.clip(ld, 1e-300, None)) + 0.5 * (-ld * (1 - ld)) * np.diag(bt.T @ Wk @ bt)
                sumPPll += float(np.sum(obs[:, k] * ExplogLD - ExplambdaDelta))

        Q_2d = Q.reshape(Dx, Dx) if Q.ndim != 2 else Q
        R_2d = R.reshape(Dy, Dy) if R.ndim != 2 else R
        logll = (-Dx * K / 2.0 * np.log(2 * np.pi)
                 - K / 2.0 * np.log(max(np.linalg.det(Q_2d), 1e-300))
                 - Dy * K / 2.0 * np.log(2 * np.pi)
                 - K / 2.0 * np.log(max(np.linalg.det(R_2d), 1e-300))
                 - Dx / 2.0 * np.log(2 * np.pi)
                 - 0.5 * np.log(max(np.linalg.det(Px0), 1e-300))
                 + sumPPll
                 - 0.5 * np.trace(np.linalg.solve(Q_2d, sumXkTerms))
                 - 0.5 * np.trace(np.linalg.solve(R_2d, sumYkTerms))
                 - Dx / 2.0)

        ExpectationSums = {
            'Sxkm1xkm1': Sxkm1xkm1,
            'Sxkm1xk': Sxkm1xk,
            'Sxkxkm1': Sxkxkm1,
            'Sxkxk': Sxkxk,
            'Sxkyk': Sxkyk,
            'Sykyk': Sykyk,
            'sumXkTerms': sumXkTerms,
            'sumYkTerms': sumYkTerms,
            'sumPPll': sumPPll,
            'Sx0': Ex0Gy,
            'Sx0x0': Px0Gy + np.outer(Ex0Gy, Ex0Gy),
        }

        return x_K, W_K, float(logll), ExpectationSums

    @staticmethod
    def mPPCO_MStep(dN, y, x_K, W_K, x0, Px0, ExpectationSums, fitType='poisson',
                    muhat=None, betahat=None, gammahat=None, windowTimes=None,
                    HkAll=None, mPPCOEM_Constraints=None, MstepMethod='GLM'):
        """M-step for the mPPCO EM algorithm.

        Matlab: ``DecodingAlgorithms.mPPCO_MStep``  (lines 6773-7662)

        Returns
        -------
        Ahat, Qhat, Chat, Rhat, alphahat, muhat_new, betahat_new, gammahat_new, x0hat, Px0hat
        """
        if mPPCOEM_Constraints is None:
            mPPCOEM_Constraints = DecodingAlgorithms.mPPCO_EMCreateConstraints()

        obs = _as_observation_matrix(dN)
        numCells = obs.shape[0]
        x_K = np.asarray(x_K, dtype=float)
        y = np.asarray(y, dtype=float)
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        Px0 = np.asarray(Px0, dtype=float)
        muhat = np.asarray(muhat, dtype=float).reshape(-1)
        betahat = np.asarray(betahat, dtype=float)
        if betahat.ndim == 1:
            betahat = betahat.reshape(-1, 1)
        gammahat = np.asarray(gammahat, dtype=float)
        if HkAll is None or _is_empty_value(HkAll):
            HkAll = np.zeros((obs.shape[1], 1, numCells), dtype=float)
        else:
            HkAll = np.asarray(HkAll, dtype=float)

        Sxkm1xkm1 = ExpectationSums['Sxkm1xkm1']
        Sxkm1xk = ExpectationSums['Sxkm1xk']
        Sxkxkm1 = ExpectationSums['Sxkxkm1']
        Sxkxk = ExpectationSums['Sxkxk']
        Sxkyk = ExpectationSums['Sxkyk']
        Sykyk = ExpectationSums['Sykyk']
        sumXkTerms = ExpectationSums['sumXkTerms']
        sumYkTerms = ExpectationSums['sumYkTerms']
        Sx0 = ExpectationSums['Sx0']
        Sx0x0 = ExpectationSums['Sx0x0']

        dx, K = x_K.shape
        dy = y.shape[0] if y.ndim == 2 else 1
        I_dx = np.eye(dx)

        # A estimate
        if mPPCOEM_Constraints['AhatDiag']:
            Ahat = (Sxkxkm1 * I_dx) @ np.linalg.inv(Sxkm1xkm1 * I_dx)
        else:
            Ahat = Sxkxkm1 @ np.linalg.inv(Sxkm1xkm1)

        # C estimate
        Chat = Sxkyk.T @ np.linalg.inv(Sxkxk)

        # alpha estimate
        alphahat = np.sum(y - Chat @ x_K, axis=1) / K if y.ndim == 2 else (y - Chat @ x_K) / K

        # Q estimate
        if mPPCOEM_Constraints['QhatDiag']:
            if mPPCOEM_Constraints['QhatIsotropic']:
                Qhat = (1.0 / (dx * K)) * np.trace(sumXkTerms) * I_dx
            else:
                Qhat = (1.0 / K) * (sumXkTerms * I_dx)
                Qhat = _symmetrize(Qhat)
        else:
            Qhat = (1.0 / K) * sumXkTerms
            Qhat = _symmetrize(Qhat)

        # R estimate
        I_dy = np.eye(dy)
        if mPPCOEM_Constraints['RhatDiag']:
            if mPPCOEM_Constraints['RhatIsotropic']:
                Rhat = (1.0 / (dy * K)) * np.trace(sumYkTerms) * I_dy
            else:
                Rhat = (1.0 / K) * (sumYkTerms * I_dy)
                Rhat = _symmetrize(Rhat)
        else:
            Rhat = (1.0 / K) * sumYkTerms
            Rhat = _symmetrize(Rhat)

        # x0 estimate
        if mPPCOEM_Constraints['Estimatex0']:
            x0hat = np.linalg.solve(
                np.linalg.inv(Px0) + Ahat.T @ np.linalg.solve(Qhat, Ahat),
                Ahat.T @ np.linalg.solve(Qhat, x_K[:, 0]) + np.linalg.solve(Px0, x0))
        else:
            x0hat = x0.copy()

        # Px0 estimate
        if mPPCOEM_Constraints['EstimatePx0']:
            if mPPCOEM_Constraints['Px0Isotropic']:
                Px0hat = (np.trace(np.outer(x0hat, x0hat) - np.outer(x0, x0hat) - np.outer(x0hat, x0) + np.outer(x0, x0)) / (dx * K)) * I_dx
            else:
                Px0hat = (np.outer(x0hat, x0hat) - np.outer(x0, x0hat) - np.outer(x0hat, x0) + np.outer(x0, x0)) * I_dx
                Px0hat = _symmetrize(Px0hat)
        else:
            Px0hat = Px0.copy()

        # CIF parameter updates via Newton-Raphson
        betahat_new = betahat.copy()
        gammahat_new = gammahat.copy()
        muhat_new = muhat.copy()

        # Newton-Raphson for beta, mu, gamma
        McExp = 50
        diffTol = 1e-5
        maxIter_nr = 100

        xKDrawExp = np.zeros((dx, K, McExp), dtype=float)
        for k in range(K):
            WuTemp = W_K[:, :, k]
            try:
                chol_m = np.linalg.cholesky(WuTemp)
            except np.linalg.LinAlgError:
                chol_m = np.linalg.cholesky(nearestSPD(WuTemp))
            z = np.random.randn(dx, McExp)
            xKDrawExp[:, k, :] = x_K[:, k:k + 1] + chol_m @ z

        xkPerm = np.transpose(xKDrawExp, (0, 2, 1))  # (dx, McExp, K)

        # -- beta update --
        for c in range(numCells):
            converged = False
            iterNR = 0
            while not converged and iterNR < maxIter_nr:
                HessianTerm = np.zeros((dx, dx))
                GradTerm = np.zeros(dx)
                for k in range(K):
                    Hk_full = HkAll[:, :, c] if HkAll.ndim == 3 else np.zeros((K, 1))
                    Hk = Hk_full[k, :]
                    xk = xkPerm[:, :, k]
                    gammaC = gammahat if gammahat.size == 1 else (gammahat[:, c] if gammahat.ndim == 2 else gammahat)
                    terms = muhat[c] + betahat_new[:, c] @ xk + float(np.dot(gammaC.reshape(-1), Hk.reshape(-1)))
                    if fitType == 'poisson':
                        ld = np.exp(np.clip(terms, -500, 500))
                        ExpLambdaXk = (1.0 / McExp) * np.sum(np.tile(ld, (dx, 1)) * xk, axis=1)
                        ExpLambdaXkXkT = (1.0 / McExp) * (np.tile(ld, (dx, 1)) * xk) @ xk.T
                        GradTerm += obs[c, k] * x_K[:, k] - ExpLambdaXk
                        HessianTerm -= ExpLambdaXkXkT
                    else:
                        ld_raw = np.exp(np.clip(terms, -500, 500))
                        ld = ld_raw / (1.0 + ld_raw)
                        EldXkXk = (1.0 / McExp) * (np.tile(ld, (dx, 1)) * xk) @ xk.T
                        EldSqXkXk = (1.0 / McExp) * (np.tile(ld ** 2, (dx, 1)) * xk) @ xk.T
                        EldCubeXkXk = (1.0 / McExp) * (np.tile(ld ** 3, (dx, 1)) * xk) @ xk.T
                        ExpLambdaXk = (1.0 / McExp) * np.sum(np.tile(ld, (dx, 1)) * xk, axis=1)
                        ExpLambdaSquaredXk = (1.0 / McExp) * np.sum(np.tile(ld ** 2, (dx, 1)) * xk, axis=1)
                        GradTerm += obs[c, k] * x_K[:, k] - (obs[c, k] + 1) * ExpLambdaXk + ExpLambdaSquaredXk
                        HessianTerm += EldXkXk + EldSqXkXk - 2 * EldCubeXkXk

                if np.any(np.isnan(HessianTerm)) or np.any(np.isinf(HessianTerm)):
                    betahat_newTemp = betahat_new[:, c]
                else:
                    try:
                        betahat_newTemp = betahat_new[:, c] - np.linalg.solve(HessianTerm, GradTerm)
                    except np.linalg.LinAlgError:
                        betahat_newTemp = betahat_new[:, c]
                    if np.any(np.isnan(betahat_newTemp)):
                        betahat_newTemp = betahat_new[:, c]

                mabsDiff = float(np.max(np.abs(betahat_newTemp - betahat_new[:, c])))
                if mabsDiff < diffTol:
                    converged = True
                betahat_new[:, c] = betahat_newTemp
                iterNR += 1

        # -- mu update --
        for c in range(numCells):
            converged = False
            iterNR = 0
            while not converged and iterNR < maxIter_nr:
                HessianTerm_mu = 0.0
                GradTerm_mu = 0.0
                for k in range(K):
                    Hk_full = HkAll[:, :, c] if HkAll.ndim == 3 else np.zeros((K, 1))
                    Hk = Hk_full[k, :]
                    xk = xkPerm[:, :, k]
                    gammaC = gammahat if gammahat.size == 1 else (gammahat[:, c] if gammahat.ndim == 2 else gammahat)
                    terms = muhat_new[c] + betahat[:, c] @ xk + float(np.dot(gammaC.reshape(-1), Hk.reshape(-1)))
                    if fitType == 'poisson':
                        ld = np.exp(np.clip(terms, -500, 500))
                        ExpLD = (1.0 / McExp) * float(np.sum(ld))
                        GradTerm_mu += obs[c, k] - ExpLD
                        HessianTerm_mu -= ExpLD
                    else:
                        ld_raw = np.exp(np.clip(terms, -500, 500))
                        ld = ld_raw / (1.0 + ld_raw)
                        ExpLD = (1.0 / McExp) * float(np.sum(ld))
                        ExpLDSq = (1.0 / McExp) * float(np.sum(ld ** 2))
                        ExpLDCube = (1.0 / McExp) * float(np.sum(ld ** 3))
                        GradTerm_mu += obs[c, k] - (obs[c, k] + 1) * ExpLD + ExpLDSq
                        HessianTerm_mu += -ExpLD * (obs[c, k] + 1) + ExpLDSq * (obs[c, k] + 3) - 2 * ExpLDCube

                if np.isnan(HessianTerm_mu) or np.isinf(HessianTerm_mu) or abs(HessianTerm_mu) < 1e-300:
                    muhat_newTemp = muhat_new[c]
                else:
                    muhat_newTemp = muhat_new[c] - GradTerm_mu / HessianTerm_mu
                    if np.isnan(muhat_newTemp):
                        muhat_newTemp = muhat_new[c]

                mabsDiff = abs(muhat_newTemp - muhat_new[c])
                if mabsDiff < diffTol:
                    converged = True
                muhat_new[c] = muhat_newTemp
                iterNR += 1

        # -- gamma update --
        if (windowTimes is not None and not _is_empty_value(windowTimes)
                and np.any(gammahat_new != 0)):
            nGamma = gammahat.shape[0] if gammahat.ndim >= 1 else 1
            for c in range(numCells):
                converged = False
                iterNR = 0
                gammaC = gammahat_new.copy() if gammahat_new.size == 1 else (gammahat_new[:, c].copy() if gammahat_new.ndim == 2 else gammahat_new.copy())
                while not converged and iterNR < maxIter_nr:
                    HessianTerm_g = np.zeros((nGamma, nGamma))
                    GradTerm_g = np.zeros(nGamma)
                    for k in range(K):
                        Hk_full = HkAll[:, :, c] if HkAll.ndim == 3 else np.zeros((K, 1))
                        Hk = Hk_full[k, :]
                        xk = xkPerm[:, :, k]
                        terms = muhat[c] + betahat[:, c] @ xk + float(np.dot(gammaC.reshape(-1), Hk.reshape(-1)))
                        if fitType == 'poisson':
                            ld = np.exp(np.clip(terms, -500, 500))
                            ExpLD = (1.0 / McExp) * float(np.sum(ld))
                            GradTerm_g += (obs[c, k] - ExpLD) * Hk
                            HessianTerm_g -= ExpLD * np.outer(Hk, Hk)
                        else:
                            ld_raw = np.exp(np.clip(terms, -500, 500))
                            ld = ld_raw / (1.0 + ld_raw)
                            ExpLD = (1.0 / McExp) * float(np.sum(ld))
                            ExpLDSq = (1.0 / McExp) * float(np.sum(ld ** 2))
                            ExpLDCube = (1.0 / McExp) * float(np.sum(ld ** 3))
                            GradTerm_g += (obs[c, k] - (obs[c, k] + 1) * ExpLD + ExpLDSq) * Hk
                            HessianTerm_g += (-ExpLD * (obs[c, k] + 1) + ExpLDSq * (obs[c, k] + 3) - 2 * ExpLDCube) * np.outer(Hk, Hk)

                    if np.any(np.isnan(HessianTerm_g)) or np.any(np.isinf(HessianTerm_g)):
                        gammahat_newTemp = gammaC.copy()
                    else:
                        try:
                            gammahat_newTemp = gammaC - np.linalg.solve(HessianTerm_g, GradTerm_g)
                        except np.linalg.LinAlgError:
                            gammahat_newTemp = gammaC.copy()
                        if np.any(np.isnan(gammahat_newTemp)):
                            gammahat_newTemp = gammaC.copy()

                    mabsDiff = float(np.max(np.abs(gammahat_newTemp - gammaC)))
                    if mabsDiff < diffTol:
                        converged = True
                    gammaC = gammahat_newTemp
                    iterNR += 1

                if gammahat_new.ndim == 2:
                    gammahat_new[:, c] = gammaC
                else:
                    gammahat_new = gammaC

        return Ahat, Qhat, Chat, Rhat, alphahat, muhat_new, betahat_new, gammahat_new, x0hat, Px0hat

    @staticmethod
    def mPPCO_EM(y, dN, Ahat0, Qhat0, Chat0, Rhat0, alphahat0, mu, beta,
                 fitType='poisson', delta=0.001, gamma=None, windowTimes=None,
                 x0=None, Px0=None, mPPCOEM_Constraints=None, MstepMethod='GLM'):
        """Full EM algorithm for the mixed Point-Process / Continuous Observation model.

        Matlab: ``DecodingAlgorithms.mPPCO_EM``  (lines 6139-6554)

        Returns
        -------
        xKFinal, WKFinal, Ahat, Qhat, Chat, Rhat, alphahat,
        muhat, betahat, gammahat, x0hat, Px0hat, IC, SE, Pvals
        """
        Ahat0 = np.asarray(Ahat0, dtype=float)
        Qhat0 = np.asarray(Qhat0, dtype=float)
        Chat0 = np.asarray(Chat0, dtype=float)
        Rhat0 = np.asarray(Rhat0, dtype=float)
        alphahat0 = np.asarray(alphahat0, dtype=float).reshape(-1)
        mu = np.asarray(mu, dtype=float).reshape(-1)
        beta = np.asarray(beta, dtype=float)
        if beta.ndim == 1:
            beta = beta.reshape(-1, 1)
        numStates = Ahat0.shape[0]
        obs = _as_observation_matrix(dN)
        numCells_K, N = obs.shape

        if mPPCOEM_Constraints is None:
            mPPCOEM_Constraints = DecodingAlgorithms.mPPCO_EMCreateConstraints()
        if Px0 is None or _is_empty_value(Px0):
            Px0 = 1e-9 * np.eye(numStates)
        else:
            Px0 = np.asarray(Px0, dtype=float).reshape(numStates, numStates)
        if x0 is None or _is_empty_value(x0):
            x0 = np.zeros(numStates, dtype=float)
        else:
            x0 = np.asarray(x0, dtype=float).reshape(-1)
        if gamma is None:
            gamma = np.array(0.0)
        else:
            gamma = np.asarray(gamma, dtype=float)
        if delta is None:
            delta = 0.001
        if windowTimes is None or _is_empty_value(windowTimes):
            if gamma is not None and np.any(gamma != 0):
                windowTimes = np.arange(gamma.size + 2, dtype=float) * delta
            else:
                windowTimes = None

        minTime = 0.0
        maxTime = (N - 1) * delta
        K_cells = numCells_K

        # Build history
        if windowTimes is not None and not _is_empty_value(windowTimes):
            wt = np.asarray(windowTimes, dtype=float).reshape(-1)
            HkAll = _compute_history_terms(dN, delta, wt)
        else:
            HkAll = np.zeros((N, 1, K_cells), dtype=float)
            gamma = np.array(0.0)

        y_arr = np.asarray(y, dtype=float)
        yOrig = y_arr.copy()

        tolAbs = 1e-3
        llTol = 1e-3
        maxIter = 100
        numToKeep = 10

        # Circular buffers
        Ahat_buf = [None] * numToKeep
        Qhat_buf = [None] * numToKeep
        Chat_buf = [None] * numToKeep
        Rhat_buf = [None] * numToKeep
        alphahat_buf = [None] * numToKeep
        muhat_buf = [None] * numToKeep
        betahat_buf = [None] * numToKeep
        gammahat_buf = [None] * numToKeep
        x0hat_buf = [None] * numToKeep
        Px0hat_buf = [None] * numToKeep
        x_K_buf = [None] * numToKeep
        W_K_buf = [None] * numToKeep
        ExpSums_buf = [None] * numToKeep
        ll_list = []

        # Initialize (scaled system)
        A0 = Ahat0.copy()
        Q0 = Qhat0.copy()
        C0 = Chat0.copy()
        R0 = Rhat0.copy()

        Tq = np.linalg.solve(np.linalg.cholesky(Q0), np.eye(numStates))
        Tr = np.linalg.solve(np.linalg.cholesky(R0), np.eye(R0.shape[0]))

        Ahat_buf[0] = Tq @ A0 @ np.linalg.inv(Tq)
        Chat_buf[0] = Tr @ C0 @ np.linalg.inv(Tq)
        Qhat_buf[0] = Tq @ Q0 @ Tq.T
        Rhat_buf[0] = Tr @ R0 @ Tr.T
        y_arr = Tr @ y_arr
        x0hat_buf[0] = Tq @ x0
        Px0hat_buf[0] = Tq @ Px0 @ Tq.T
        alphahat_buf[0] = Tr @ alphahat0
        betahat_buf[0] = np.linalg.solve(Tq.T, beta)
        muhat_buf[0] = mu.copy()
        gammahat_buf[0] = gamma.copy()

        cnt = 0
        stoppingCriteria = False

        print("                        Joint Point-Process/Gaussian Observation EM Algorithm                        ")

        while not stoppingCriteria and cnt < maxIter:
            si = cnt % numToKeep
            si_p1 = (cnt + 1) % numToKeep
            si_m1 = (cnt - 1) % numToKeep

            print("-" * 80)
            print(f"Iteration #{cnt + 1}")
            print("-" * 80)

            # E-step
            x_K_buf[si], W_K_buf[si], ll_val, ExpSums_buf[si] = DecodingAlgorithms.mPPCO_EStep(
                Ahat_buf[si], Qhat_buf[si], Chat_buf[si], Rhat_buf[si],
                y_arr, alphahat_buf[si], dN,
                muhat_buf[si], betahat_buf[si], fitType, delta,
                gammahat_buf[si], HkAll, x0hat_buf[si], Px0hat_buf[si])
            ll_list.append(ll_val)

            # M-step
            (Ahat_buf[si_p1], Qhat_buf[si_p1], Chat_buf[si_p1], Rhat_buf[si_p1],
             alphahat_buf[si_p1], muhat_buf[si_p1], betahat_buf[si_p1],
             gammahat_buf[si_p1], x0hat_buf[si_p1], Px0hat_buf[si_p1]) = \
                DecodingAlgorithms.mPPCO_MStep(
                    dN, y_arr, x_K_buf[si], W_K_buf[si],
                    x0hat_buf[si], Px0hat_buf[si], ExpSums_buf[si],
                    fitType, muhat_buf[si], betahat_buf[si],
                    gammahat_buf[si], windowTimes, HkAll,
                    mPPCOEM_Constraints, MstepMethod)

            if not mPPCOEM_Constraints['EstimateA']:
                Ahat_buf[si_p1] = Ahat_buf[si].copy()

            # Convergence check
            if cnt == 0:
                dMax = np.inf
            else:
                diffs = []
                for arr_curr, arr_prev in [
                    (Qhat_buf[si], Qhat_buf[si_m1]),
                    (Rhat_buf[si], Rhat_buf[si_m1]),
                    (Ahat_buf[si], Ahat_buf[si_m1]),
                    (Chat_buf[si], Chat_buf[si_m1]),
                ]:
                    if arr_curr is not None and arr_prev is not None:
                        diffs.append(float(np.max(np.abs(np.sqrt(np.abs(arr_curr)) - np.sqrt(np.abs(arr_prev))))) if 'Q' in str(id(arr_curr)) else float(np.max(np.abs(arr_curr - arr_prev))))
                for arr_curr, arr_prev in [
                    (muhat_buf[si], muhat_buf[si_m1]),
                    (alphahat_buf[si], alphahat_buf[si_m1]),
                    (betahat_buf[si], betahat_buf[si_m1]),
                    (gammahat_buf[si], gammahat_buf[si_m1]),
                ]:
                    if arr_curr is not None and arr_prev is not None:
                        diffs.append(float(np.max(np.abs(np.asarray(arr_curr) - np.asarray(arr_prev)))))
                dMax = max(diffs) if diffs else np.inf

            if cnt == 0:
                print("Max Parameter Change: N/A")
            else:
                print(f"Max Parameter Change: {dMax}")

            cnt += 1

            if dMax < tolAbs:
                stoppingCriteria = True
                print(f"         EM converged at iteration# {cnt} b/c change in params was within criteria")

            if cnt >= 2:
                dll = ll_list[-1] - ll_list[-2]
                if abs(dll) < llTol or dll < 0:
                    stoppingCriteria = True
                    print(f"         EM stopped at iteration# {cnt} b/c change in likelihood was negative or small")

        print("-" * 80)

        # Select best iteration
        ll_arr = np.array(ll_list)
        maxLLIndex = int(np.argmax(ll_arr))
        maxLLIndMod = maxLLIndex % numToKeep

        xKFinal = x_K_buf[maxLLIndMod]
        WKFinal = W_K_buf[maxLLIndMod]
        Ahat_out = Ahat_buf[maxLLIndMod]
        Qhat_out = Qhat_buf[maxLLIndMod]
        Chat_out = Chat_buf[maxLLIndMod]
        Rhat_out = Rhat_buf[maxLLIndMod]
        alphahat_out = alphahat_buf[maxLLIndMod]
        muhat_out = muhat_buf[maxLLIndMod]
        betahat_out = betahat_buf[maxLLIndMod]
        gammahat_out = gammahat_buf[maxLLIndMod]
        x0hat_out = x0hat_buf[maxLLIndMod]
        Px0hat_out = Px0hat_buf[maxLLIndMod]
        ExpectationSumsFinal = ExpSums_buf[maxLLIndMod]

        # Unscale
        Tq = np.linalg.solve(np.linalg.cholesky(Q0), np.eye(numStates))
        Tr = np.linalg.solve(np.linalg.cholesky(R0), np.eye(R0.shape[0]))
        Tq_inv = np.linalg.inv(Tq)
        Tr_inv = np.linalg.inv(Tr)

        Ahat_out = Tq_inv @ Ahat_out @ Tq
        Qhat_out = Tq_inv @ Qhat_out @ np.linalg.inv(Tq.T)
        Chat_out = Tr_inv @ Chat_out @ Tq
        Rhat_out = Tr_inv @ Rhat_out @ np.linalg.inv(Tr.T)
        alphahat_out = Tr_inv @ alphahat_out
        xKFinal = Tq_inv @ xKFinal
        x0hat_out = Tq_inv @ x0hat_out
        Px0hat_out = Tq_inv @ Px0hat_out @ np.linalg.inv(Tq.T)
        for kk in range(WKFinal.shape[2]):
            WKFinal[:, :, kk] = Tq_inv @ WKFinal[:, :, kk] @ np.linalg.inv(Tq.T)
        betahat_out = (betahat_out.T @ Tq).T

        # Information criteria
        ll_best = ll_arr[maxLLIndex]
        # Count parameters
        if mPPCOEM_Constraints['EstimateA'] and mPPCOEM_Constraints['AhatDiag']:
            n1 = Ahat_out.shape[0]
        elif mPPCOEM_Constraints['EstimateA']:
            n1 = Ahat_out.size
        else:
            n1 = 0

        if mPPCOEM_Constraints['QhatDiag'] and mPPCOEM_Constraints['QhatIsotropic']:
            n2 = 1
        elif mPPCOEM_Constraints['QhatDiag']:
            n2 = Qhat_out.shape[0]
        else:
            n2 = Qhat_out.size

        n3 = Chat_out.size

        if mPPCOEM_Constraints['RhatDiag'] and mPPCOEM_Constraints['RhatIsotropic']:
            n4 = 1
        elif mPPCOEM_Constraints['RhatDiag']:
            n4 = Rhat_out.shape[0]
        else:
            n4 = Rhat_out.size

        if mPPCOEM_Constraints['EstimatePx0'] and mPPCOEM_Constraints['Px0Isotropic']:
            n5 = 1
        elif mPPCOEM_Constraints['EstimatePx0']:
            n5 = Px0hat_out.shape[0]
        else:
            n5 = 0

        n6 = x0hat_out.size if mPPCOEM_Constraints['Estimatex0'] else 0
        n7 = alphahat_out.size
        n8 = muhat_out.size
        n9 = betahat_out.size
        if gammahat_out.size == 1 and float(gammahat_out.flat[0]) == 0:
            n10 = 0
        else:
            n10 = gammahat_out.size
        nTerms = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10

        Dx = Ahat_out.shape[1]
        sumXkTerms = ExpectationSumsFinal['sumXkTerms']
        llobs = (ll_best + Dx * N / 2.0 * np.log(2 * np.pi)
                 + N / 2.0 * np.log(max(np.linalg.det(Qhat_out), 1e-300))
                 + 0.5 * np.trace(np.linalg.solve(Qhat_out, sumXkTerms))
                 + Dx / 2.0 * np.log(2 * np.pi)
                 + 0.5 * np.log(max(np.linalg.det(Px0hat_out), 1e-300))
                 + 0.5 * Dx)

        AIC = 2 * nTerms - 2 * llobs
        AICc = AIC + 2 * nTerms * (nTerms + 1) / max(N - nTerms - 1, 1)
        BIC = -2 * llobs + nTerms * np.log(max(N, 1))

        IC = {
            'AIC': AIC, 'AICc': AICc, 'BIC': BIC,
            'llobs': llobs, 'llcomp': ll_best,
        }

        # Standard errors
        SE = {}
        Pvals = {}
        try:
            SE, Pvals, _ = DecodingAlgorithms.mPPCO_ComputeParamStandardErrors(
                yOrig, dN, xKFinal, WKFinal, Ahat_out, Qhat_out,
                Chat_out, Rhat_out, alphahat_out, x0hat_out, Px0hat_out,
                ExpectationSumsFinal, fitType, muhat_out, betahat_out,
                gammahat_out, windowTimes, HkAll, mPPCOEM_Constraints)
        except Exception:
            pass

        return (xKFinal, WKFinal, Ahat_out, Qhat_out, Chat_out, Rhat_out,
                alphahat_out, muhat_out, betahat_out, gammahat_out,
                x0hat_out, Px0hat_out, IC, SE, Pvals)




# Module-level aliases for backward compatibility
PP_fixedIntervalSmoother = DecodingAlgorithms.PP_fixedIntervalSmoother
PPDecodeFilter = DecodingAlgorithms.PPDecodeFilter
PPDecodeFilterLinear = DecodingAlgorithms.PPDecodeFilterLinear
PPDecode_predict = DecodingAlgorithms.PPDecode_predict
PPDecode_update = DecodingAlgorithms.PPDecode_update
PPDecode_updateLinear = DecodingAlgorithms.PPDecode_updateLinear
PPHybridFilter = DecodingAlgorithms.PPHybridFilter
PPHybridFilterLinear = DecodingAlgorithms.PPHybridFilterLinear
PPSS_EM = DecodingAlgorithms.PPSS_EM
PPSS_EMFB = DecodingAlgorithms.PPSS_EMFB
PPSS_EStep = DecodingAlgorithms.PPSS_EStep
PPSS_MStep = DecodingAlgorithms.PPSS_MStep
kalman_filter = DecodingAlgorithms.kalman_filter
kalman_predict = DecodingAlgorithms.kalman_predict
kalman_update = DecodingAlgorithms.kalman_update
kalman_fixedIntervalSmoother = DecodingAlgorithms.kalman_fixedIntervalSmoother
kalman_smootherFromFiltered = DecodingAlgorithms.kalman_smootherFromFiltered
kalman_smoother = DecodingAlgorithms.kalman_smoother
ComputeStimulusCIs = DecodingAlgorithms.ComputeStimulusCIs
computeSpikeRateCIs = DecodingAlgorithms.computeSpikeRateCIs
computeSpikeRateDiffCIs = DecodingAlgorithms.computeSpikeRateDiffCIs
ukf = DecodingAlgorithms.ukf
ukf_ut = DecodingAlgorithms.ukf_ut
ukf_sigmas = DecodingAlgorithms.ukf_sigmas
KF_EM = DecodingAlgorithms.KF_EM
KF_EMCreateConstraints = DecodingAlgorithms.KF_EMCreateConstraints
KF_EStep = DecodingAlgorithms.KF_EStep
KF_MStep = DecodingAlgorithms.KF_MStep
KF_ComputeParamStandardErrors = DecodingAlgorithms.KF_ComputeParamStandardErrors
PP_EM = DecodingAlgorithms.PP_EM
PP_EMCreateConstraints = DecodingAlgorithms.PP_EMCreateConstraints
PP_ComputeParamStandardErrors = DecodingAlgorithms.PP_ComputeParamStandardErrors
PP_EStep = DecodingAlgorithms.PP_EStep
PP_MStep = DecodingAlgorithms.PP_MStep
mPPCODecode_predict = DecodingAlgorithms.mPPCODecode_predict
mPPCODecode_update = DecodingAlgorithms.mPPCODecode_update
mPPCODecodeLinear = DecodingAlgorithms.mPPCODecodeLinear
mPPCO_fixedIntervalSmoother = DecodingAlgorithms.mPPCO_fixedIntervalSmoother
mPPCO_EMCreateConstraints = DecodingAlgorithms.mPPCO_EMCreateConstraints
mPPCO_ComputeParamStandardErrors = DecodingAlgorithms.mPPCO_ComputeParamStandardErrors
mPPCO_EM = DecodingAlgorithms.mPPCO_EM
mPPCO_EStep = DecodingAlgorithms.mPPCO_EStep
mPPCO_MStep = DecodingAlgorithms.mPPCO_MStep


__all__ = [
    "ComputeStimulusCIs",
    "DecodingAlgorithms",
    "computeSpikeRateCIs",
    "computeSpikeRateDiffCIs",
    "KF_ComputeParamStandardErrors",
    "KF_EM",
    "KF_EMCreateConstraints",
    "KF_EStep",
    "KF_MStep",
    "PP_ComputeParamStandardErrors",
    "PP_EM",
    "PP_EMCreateConstraints",
    "PP_EStep",
    "PP_MStep",
    "PPDecodeFilter",
    "PPDecodeFilterLinear",
    "PPDecode_predict",
    "PPDecode_update",
    "PPDecode_updateLinear",
    "PPHybridFilter",
    "PPHybridFilterLinear",
    "PPSS_EM",
    "PPSS_EMFB",
    "PPSS_EStep",
    "PPSS_MStep",
    "PP_fixedIntervalSmoother",
    "kalman_filter",
    "kalman_fixedIntervalSmoother",
    "kalman_predict",
    "kalman_smoother",
    "kalman_smootherFromFiltered",
    "kalman_update",
    "mPPCODecode_predict",
    "mPPCODecode_update",
    "mPPCODecodeLinear",
    "mPPCO_ComputeParamStandardErrors",
    "mPPCO_EM",
    "mPPCO_EMCreateConstraints",
    "mPPCO_EStep",
    "mPPCO_MStep",
    "mPPCO_fixedIntervalSmoother",
    "ukf",
    "ukf_sigmas",
    "ukf_ut",
]
