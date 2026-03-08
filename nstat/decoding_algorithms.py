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
        obs = _as_observation_matrix(dN)
        A_models = list(A) if isinstance(A, Sequence) and not isinstance(A, np.ndarray) else [A]
        num_states = _infer_state_dim(A_models[0], np.array([0.0]), obs.shape[0])
        mu, beta, fitType, gamma, windowTimes = _extract_linear_terms_from_cifs(lambdaCIFColl, num_states, obs.shape[0])
        return DecodingAlgorithms.PPHybridFilterLinear(
            A,
            Q,
            p_ij,
            Mu0,
            obs,
            mu,
            beta,
            fitType,
            binwidth,
            gamma,
            windowTimes,
            x0,
            Pi0,
            yT,
            PiT,
            estimateTarget,
            MinClassificationError,
        )


PP_fixedIntervalSmoother = DecodingAlgorithms.PP_fixedIntervalSmoother
PPDecodeFilter = DecodingAlgorithms.PPDecodeFilter
PPDecodeFilterLinear = DecodingAlgorithms.PPDecodeFilterLinear
PPDecode_predict = DecodingAlgorithms.PPDecode_predict
PPDecode_update = DecodingAlgorithms.PPDecode_update
PPDecode_updateLinear = DecodingAlgorithms.PPDecode_updateLinear
PPHybridFilter = DecodingAlgorithms.PPHybridFilter
PPHybridFilterLinear = DecodingAlgorithms.PPHybridFilterLinear
kalman_filter = DecodingAlgorithms.kalman_filter
kalman_predict = DecodingAlgorithms.kalman_predict
kalman_update = DecodingAlgorithms.kalman_update
kalman_fixedIntervalSmoother = DecodingAlgorithms.kalman_fixedIntervalSmoother
kalman_smootherFromFiltered = DecodingAlgorithms.kalman_smootherFromFiltered
kalman_smoother = DecodingAlgorithms.kalman_smoother
ComputeStimulusCIs = DecodingAlgorithms.ComputeStimulusCIs


__all__ = [
    "ComputeStimulusCIs",
    "DecodingAlgorithms",
    "PPDecodeFilter",
    "PPDecodeFilterLinear",
    "PPDecode_predict",
    "PPDecode_update",
    "PPDecode_updateLinear",
    "PPHybridFilter",
    "PPHybridFilterLinear",
    "PP_fixedIntervalSmoother",
    "kalman_filter",
    "kalman_fixedIntervalSmoother",
    "kalman_predict",
    "kalman_smoother",
    "kalman_smootherFromFiltered",
    "kalman_update",
]
