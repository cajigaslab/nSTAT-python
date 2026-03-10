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
ukf = DecodingAlgorithms.ukf
ukf_ut = DecodingAlgorithms.ukf_ut
ukf_sigmas = DecodingAlgorithms.ukf_sigmas


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
    "ukf",
    "ukf_sigmas",
    "ukf_ut",
]
