from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .cif import CIF
from .errors import UnsupportedWorkflowError


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
    if arr.ndim == 1:
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


def _extract_linear_terms_from_cifs(lambdaCIFColl, num_states: int, num_cells: int):
    if isinstance(lambdaCIFColl, CIF):
        cifs = [lambdaCIFColl]
    elif isinstance(lambdaCIFColl, Sequence) and not isinstance(lambdaCIFColl, (str, bytes)):
        cifs = list(lambdaCIFColl)
    else:
        raise UnsupportedWorkflowError("PPDecodeFilter requires a CIF or sequence of CIF objects for the Python port")

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
        num_states = _infer_state_dim(A, np.array([0.0]), obs.shape[0])
        mu, beta, fitType, gamma, windowTimes = _extract_linear_terms_from_cifs(lambdaCIFColl, num_states, obs.shape[0])
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

        num_cells = obs.shape[0]
        state_dims = [_infer_state_dim(A_models[index], beta, num_cells) for index in range(n_models)]
        mu_models = _normalize_mu_models(mu, n_models, num_cells)
        beta_models = _normalize_beta_models(beta, n_models, num_cells, state_dims)
        x0_models = _normalize_model_sequence(x0, n_models, lambda index: np.zeros(state_dims[index], dtype=float))
        Pi0_models = _normalize_model_sequence(Pi0, n_models, lambda index: np.zeros((state_dims[index], state_dims[index]), dtype=float))

        transition = np.asarray(p_ij, dtype=float)
        if transition.shape != (n_models, n_models):
            raise ValueError("p_ij must be an nModels x nModels transition matrix")
        model_probs = _normalize_probabilities(Mu0)
        if model_probs.size != n_models:
            raise ValueError("Mu0 must contain one probability per hybrid model")

        if _is_empty_value(windowTimes):
            H_tensor = np.zeros((obs.shape[1], 0, num_cells), dtype=float)
            gamma_mat = np.zeros((0, num_cells), dtype=float)
        else:
            H_tensor = _compute_history_terms(obs, float(binwidth), windowTimes)
            gamma_mat = _normalize_gamma(gamma, H_tensor.shape[1], num_cells)

        model_results = [
            DecodingAlgorithms._ppdecode_filter_linear(
                A_models[index],
                Q_models[index],
                obs,
                mu_models[index],
                beta_models[index],
                fitType,
                binwidth,
                gamma_mat,
                windowTimes,
                x0_models[index],
                Pi0_models[index],
            )
            for index in range(n_models)
        ]

        max_dim = max(state_dims)
        num_steps = obs.shape[1]
        X = np.zeros((max_dim, num_steps), dtype=float)
        W = np.zeros((max_dim, max_dim, num_steps), dtype=float)
        MU_u = np.zeros((n_models, num_steps), dtype=float)
        pNGivenS = np.zeros((n_models, num_steps), dtype=float)
        X_s = [result[2] for result in model_results]
        W_s = [result[3] for result in model_results]
        S_est = np.zeros(num_steps, dtype=int)

        for time_index in range(num_steps):
            predicted_probs = transition.T @ model_probs
            likelihoods = np.zeros(n_models, dtype=float)
            for model_index in range(n_models):
                x_state = model_results[model_index][2][:, time_index]
                lambda_delta = _lambda_delta_from_state(
                    x_state,
                    mu_models[model_index],
                    beta_models[model_index],
                    str(fitType),
                    gamma_mat,
                    H_tensor,
                    time_index + 1,
                )
                likelihoods[model_index] = _likelihood_from_lambda(obs[:, time_index], lambda_delta, str(fitType))

            weighted = likelihoods * predicted_probs
            model_probs = _normalize_probabilities(weighted)
            MU_u[:, time_index] = model_probs
            pNGivenS[:, time_index] = _normalize_probabilities(likelihoods)

            best_model = int(np.argmax(model_probs))
            S_est[time_index] = best_model + 1

            if MinClassificationError:
                chosen = best_model
                X[: state_dims[chosen], time_index] = model_results[chosen][2][:, time_index]
                W[: state_dims[chosen], : state_dims[chosen], time_index] = model_results[chosen][3][:, :, time_index]
                continue

            for model_index in range(n_models):
                dim = state_dims[model_index]
                X[:dim, time_index] += model_probs[model_index] * model_results[model_index][2][:, time_index]
                W[:dim, :dim, time_index] += model_probs[model_index] * model_results[model_index][3][:, :, time_index]

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


__all__ = ["DecodingAlgorithms"]
