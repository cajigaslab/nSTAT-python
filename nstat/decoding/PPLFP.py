"""Point-process log-likelihood filter (PPLFP) family.

This module hosts the Python port of the MATLAB ``PPLFP_*`` static
methods from ``DecodingAlgorithms.m``.  The PPLFP family is a
point-process state-space estimator that maximises the (log-)likelihood
of an observed spike train under a conditional-intensity model, with
companion routines for forward filtering, fixed-interval smoothing,
EM-based parameter learning, and parameter-uncertainty estimation.

Method roster (all ``@staticmethod`` on :class:`PPLFP`, mirroring the
MATLAB layout):

- :meth:`PPLFP.PPLFP_Decode_predict` — one-step state prediction.
- :meth:`PPLFP.PPLFP_Decode_update` — one-step measurement update.
- :meth:`PPLFP.PPLFP_DecodeLinear` — forward filter over the full epoch.
- :meth:`PPLFP.PPLFP_fixedIntervalSmoother` — RTS-style smoother.
- :meth:`PPLFP.PPLFP_EMCreateConstraints` — build EM constraint struct.
- :meth:`PPLFP.PPLFP_ComputeParamStandardErrors` — observed-information SE.
- :meth:`PPLFP.PPLFP_EM` — full EM driver (E-step / M-step loop).
- :meth:`PPLFP.PPLFP_EStep` — single expectation step.
- :meth:`PPLFP.PPLFP_MStep` — single maximisation step.

All methods currently raise :class:`NotImplementedError`; the bodies are
being filled in by the iter-29 parallel-porter pass.  The shims in
:mod:`nstat.decoding_algorithms` (``DecodingAlgorithms.PPLFP_*``) forward
here so that downstream code can adopt the canonical import path
incrementally.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import scipy.linalg  # noqa: F401  (used by ported method bodies)

from nstat.decoding_algorithms import (
    _as_observation_matrix,
    _as_state_matrix,
    _check_mstep_method,
    _em_glm_mstep,
    _expand_shared_se_gamma,
    _is_empty_value,
    _matlab_float_semantics,
    _matlab_mldivide,
    _matlab_nearest_spd,
    _matlab_ztest_p,
    _mc_state_draws,
    _normalize_beta,
    _normalize_gamma,
    _normalize_history_tensor,
    _normalize_mu,
    _symmetrize,
)

_PORT_MSG = "Port in progress: see iter 29 of parity push"

# PPLFP_EM / PPLFP_MStep progress messages go to this logger at INFO level --
# silent unless the caller configures logging, e.g.
# ``logging.basicConfig(level=logging.INFO)`` -- as in
# nstat.decoding_algorithms (KF_EM / PP_EM).
_logger = logging.getLogger(__name__)


def _is_empty_value(v: Any) -> bool:
    """MATLAB-style ``isempty`` check covering ``None``, ``[]``, empty arrays."""
    if v is None:
        return True
    if isinstance(v, (list, tuple)) and len(v) == 0:
        return True
    try:
        arr = np.asarray(v)
    except Exception:
        return False
    return arr.size == 0


def _nearest_spd(A):
    """Nearest symmetric positive-definite matrix (MATLAB ``nearestSPD``, as
    used on the inverse observed information of
    ``PPLFP_ComputeParamStandardErrors``); see
    :func:`nstat.decoding_algorithms._matlab_nearest_spd`.  (This port shifted
    by ``spacing(norm(A))`` instead of ``eps(mineig)``, stopped after 50
    passes and returned the unprojected ``B`` if the SVD failed.)"""
    return _matlab_nearest_spd(A)


def _unwhiten(T, S):
    """MATLAB ``(T\\S)/T'`` = ``T^-1 S T^-T``: a scaled-system covariance (or
    second-moment sum) mapped back to the original scale."""
    return np.linalg.solve(T, np.linalg.solve(T, S).T).T


class PPLFP:
    """Point-process log-likelihood filter family (MATLAB ``PPLFP_*``).

    Static-method namespace mirroring the MATLAB ``DecodingAlgorithms``
    PPLFP group.  Each method below is a placeholder pending its port;
    parallel porters fill them in one at a time without disturbing the
    public signature.
    """

    @staticmethod
    def PPLFP_Decode_predict(
        x_u: np.ndarray,
        W_u: np.ndarray,
        A: np.ndarray,
        Q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One-step PPLFP state prediction.

        Mirrors MATLAB ``PPLFP_Decode_predict`` in
        ``+nstat/+decoding/PPLFP.m`` (lines 293-301): propagates the
        posterior mean ``x_u`` and covariance ``W_u`` forward through the
        linear state model with transition matrix ``A`` and process-noise
        covariance ``Q``.

        Parameters
        ----------
        x_u : np.ndarray
            Posterior state estimate at the previous step (column vector
            ``(n_state,)`` or ``(n_state, 1)``).
        W_u : np.ndarray
            Posterior state covariance at the previous step
            ``(n_state, n_state)``.
        A : np.ndarray
            State-transition matrix ``(n_state, n_state)``.
        Q : np.ndarray
            Process-noise covariance ``(n_state, n_state)``.

        Returns
        -------
        x_p : np.ndarray
            One-step-ahead predicted state mean (same shape as ``x_u``).
        W_p : np.ndarray
            One-step-ahead predicted state covariance, symmetrised as
            ``0.5 * (W_p + W_p.T)``.

        Notes
        -----
        Follows the MATLAB form ``x_p = A * x_u``,
        ``W_p = A * W_u * A' + Q`` (Srinivasan et al. 2007, p. 529).
        The commented-out ``rcond`` guard in the MATLAB source is kept as
        a comment for parity audit; it is intentionally inactive.
        """
        x_p = A @ x_u
        W_p = A @ W_u @ A.T + Q
        # if rcond(W_p) < 1000*eps:
        #     W_p = W_u  # See Srinivasan et al. 2007 pg. 529
        W_p = 0.5 * (W_p + W_p.T)  # symmetrise to combat round-off drift
        return x_p, W_p

    @staticmethod
    def PPLFP_Decode_update(
        x_p: np.ndarray,
        W_p: np.ndarray,
        C: np.ndarray,
        R: np.ndarray,
        y: np.ndarray,
        alpha: np.ndarray,
        dN: np.ndarray,
        mu: np.ndarray,
        beta: np.ndarray,
        fitType: str = "poisson",
        gamma: Any = None,
        HkAll: Any = None,
        time_index: int = 0,
        WuConv: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One-step PPLFP measurement update (point-process + LFP).

        Direct port of MATLAB ``PPLFP_Decode_update`` in
        ``+nstat/+decoding/PPLFP.m`` (lines 302-388).  Combines the
        point-process spike-train update of ``PPDecode_updateLinear`` with
        an additional Gaussian LFP observation
        ``y = C * x + alpha + N(0, R)`` (Cajigas, Malik & Brown 2012,
        Section 2 hybrid filter).

        Parameters
        ----------
        x_p : np.ndarray
            Predicted state ``(n_state,)`` (or ``(n_state, 1)``).
        W_p : np.ndarray
            Predicted state covariance ``(n_state, n_state)``.
        C : np.ndarray
            LFP observation matrix ``(n_lfp, n_state)``.
        R : np.ndarray
            LFP observation noise covariance ``(n_lfp, n_lfp)``.
        y : np.ndarray
            LFP observation at this time step ``(n_lfp,)``.
        alpha : np.ndarray
            LFP observation offset ``(n_lfp,)``.
        dN : np.ndarray
            Binned spike counts ``(num_cells, N)``.
        mu : np.ndarray
            Baseline log-firing-rate per cell ``(num_cells,)``.
        beta : np.ndarray
            State-coupling coefficients ``(n_state, num_cells)``.
        fitType : str, optional
            Either ``"poisson"`` (default) or ``"binomial"``.
        gamma : array_like, optional
            History coefficients ``(n_windows, num_cells)``.  Scalar ``0``
            or empty means no history term.
        HkAll : array_like, optional
            History tensor ``(N, n_windows, num_cells)``.  Empty means no
            history term.
        time_index : int, optional
            Sample index to update on (0-based).  Defaults to 0.
        WuConv : array_like, optional
            Pre-converged update covariance.  When supplied, the
            measurement-information branch is skipped and ``W_u = WuConv``
            is reused (matches MATLAB ``WuConv`` short-circuit).

        Returns
        -------
        x_u : np.ndarray
            Posterior state estimate ``(n_state,)``.
        W_u : np.ndarray
            Posterior covariance ``(n_state, n_state)``.
        lambdaDeltaMat : np.ndarray
            Per-cell instantaneous firing probability (Poisson) or
            spike probability (binomial), shape ``(num_cells, 1)``.

        Notes
        -----
        MATLAB cross-reference (verbatim form):

        - ``linTerm = mu + beta' * x_p + diag(gamma' * Histterm')``
        - Poisson: ``lambdaDeltaMat = exp(linTerm)``
        - Binomial: ``lambdaDeltaMat = exp(linTerm) ./ (1 + exp(linTerm))``
        - ``sumValMat += C' * (R \\ C)`` (LFP information contribution).
        - Posterior:
          ``x_u = x_p + W_u * sumValVec + (W_u * C' / R) * (y - C * x_p - alpha)``.

        The original MATLAB body delegates the Woodbury gain step to
        ``nstat.decoding.internal.computeGainMatrix``; we inline the same
        Woodbury identity used by ``PPDecode_updateLinear`` for parity, and
        fall back to ``W_p`` if the linear solve is singular.
        """
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

        # MATLAB: linTerm = mu + beta'*x_p + diag(gamma'*Histterm')
        # Histterm has shape (num_cells, n_windows); diag(gamma'*Histterm')
        # gives the per-cell sum-of-products gamma(:,j)' * Histterm(j,:)'.
        lin_term = mu_vec + beta_mat.T @ x_vec
        if num_windows > 0:
            # gamma_mat: (n_windows, num_cells); H_tensor[time_index]: (n_windows, num_cells)
            history_term = H_tensor[int(time_index)]  # (n_windows, num_cells)
            lin_term = lin_term + np.sum(gamma_mat * history_term, axis=0)

        if str(fitType) == "binomial":
            exp_lin = np.exp(lin_term)
            lambda_delta = exp_lin / (1.0 + exp_lin)
        elif str(fitType) == "poisson":
            lambda_delta = np.exp(lin_term)
        else:
            raise ValueError(f"Unknown fitType {fitType!r}; expected 'poisson' or 'binomial'.")

        # MATLAB sanitises NaN / Inf entries by clamping to 1 (lines 340-345 / 365-370).
        bad = np.isnan(lambda_delta) | np.isinf(lambda_delta)
        if np.any(bad):
            lambda_delta = np.where(bad, 1.0, lambda_delta)

        observed = obs[:, int(time_index)]
        if str(fitType) == "binomial":
            factor = (observed - lambda_delta) * (1.0 - lambda_delta)
            temp_vec = (observed + (1.0 - 2.0 * lambda_delta)) * (1.0 - lambda_delta) * lambda_delta
        else:
            factor = observed - lambda_delta
            temp_vec = lambda_delta

        sum_val_vec = np.sum(beta_mat * factor[None, :], axis=1)
        sum_val_mat = (beta_mat * temp_vec[None, :]) @ beta_mat.T

        C_mat = np.atleast_2d(np.asarray(C, dtype=float))
        R_mat = np.atleast_2d(np.asarray(R, dtype=float))
        y_vec = np.asarray(y, dtype=float).reshape(-1)
        alpha_vec = np.asarray(alpha, dtype=float).reshape(-1)

        if _is_empty_value(WuConv):
            # MATLAB: sumValMat = sumValMat + C'*(R\C);
            R_inv_C = np.linalg.solve(R_mat, C_mat)
            sum_val_mat = sum_val_mat + C_mat.T @ R_inv_C

            # Woodbury form (parity with PPDecode_updateLinear / computeGainMatrix):
            # W_u = W_p * (I - (I + sumValMat*W_p)^{-1} * sumValMat * W_p)
            identity = np.eye(W_mat.shape[0], dtype=float)
            try:
                W_u = W_mat @ (
                    identity - np.linalg.solve(identity + sum_val_mat @ W_mat, sum_val_mat @ W_mat)
                )
                W_u = _symmetrize(W_u)
            except np.linalg.LinAlgError:
                # MATLAB: if isSingular -> W_u = W_p; W_u = 0.5*(W_u + W_u')
                W_u = _symmetrize(W_mat.copy())
        else:
            W_u = _symmetrize(_as_state_matrix(WuConv, x_vec.size))

        # x_u = x_p + W_u*sumValVec + (W_u*C'/R) * (y - C*x_p - alpha)
        # MATLAB ``(W_u*C')/R`` is right division: (W_u*C') * inv(R)
        # i.e. solve R^T * X^T = (W_u*C')^T  ->  X = (W_u*C') * R^{-1}.
        WCt = W_u @ C_mat.T
        right_solve = np.linalg.solve(R_mat.T, WCt.T).T  # (W_u*C') / R
        innovation = y_vec - C_mat @ x_vec - alpha_vec
        x_u = x_vec + W_u @ sum_val_vec + right_solve @ innovation

        return x_u, W_u, lambda_delta.reshape(-1, 1)

    @staticmethod
    def PPLFP_DecodeLinear(
        A: np.ndarray,
        Q: np.ndarray,
        C: np.ndarray,
        R: np.ndarray,
        y: np.ndarray,
        alpha: np.ndarray,
        dN: np.ndarray,
        mu: np.ndarray,
        beta: np.ndarray,
        fitType: str,
        delta: float | None = None,
        gamma: Any = None,
        windowTimes: Any = None,
        x0: Any = None,
        Px0: Any = None,
        HkAll: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward PPLFP filter over a full epoch.

        Python port of MATLAB ``PPLFP_DecodeLinear``
        (``+nstat/+decoding/PPLFP.m`` lines 136-292). Point-process adaptive
        filter that jointly assimilates continuous Gaussian observations
        ``y_t = C x_t + alpha + w_t`` (``w_t ~ N(0, R)``) and point-process
        spike observations ``dN`` whose conditional intensity is linear in
        the state (``'poisson'`` or ``'binomial'`` link). The state model is
        ``x_t = A x_{t-1} + v_t`` with ``v_t ~ N(0, Q)``.

        Parameters
        ----------
        A : np.ndarray
            State-transition matrix, shape ``(ns, ns)`` or ``(ns, ns, N)``.
        Q : np.ndarray
            Process-noise covariance, shape ``(ns, ns)`` or ``(ns, ns, N)``.
        C : np.ndarray
            Continuous-observation matrix, shape ``(ny, ns)`` or
            ``(ny, ns, N)``.
        R : np.ndarray
            Continuous-observation noise covariance, shape ``(ny, ny)`` or
            ``(ny, ny, N)``.
        y : np.ndarray
            Continuous observations, shape ``(ny, N)``.
        alpha : np.ndarray
            Offset for the continuous observations.
        dN : np.ndarray
            ``(numCells, N)`` binary spike-occurrence matrix.
        mu : np.ndarray
            ``(numCells,)`` baseline log-rate / logit.
        beta : np.ndarray
            ``(ns, numCells)`` linear CIF coefficients.
        fitType : str
            ``'poisson'`` or ``'binomial'``.
        delta : float, optional
            Seconds per time-step. Default ``0.001`` (MATLAB).
        gamma : array_like, optional
            History coefficients, shape ``(len(windowTimes)-1, numCells)``
            or ``(len(windowTimes)-1, 1)``. Default ``0``.
        windowTimes : array_like, optional
            History window edges (seconds).
        x0 : array_like, optional
            Initial state mean, shape ``(ns,)`` or ``(ns, 1)``. Default zeros.
        Px0 : array_like, optional
            Initial state covariance ``(ns, ns)``. Default zeros.
        HkAll : array_like, optional
            History-effect tensor; MATLAB shape ``(N, K, numCells)`` with
            ``K = len(windowTimes) - 1``.

        Returns
        -------
        x_p : np.ndarray
            One-step predicted state means, shape ``(ns, N+1)``.
        W_p : np.ndarray
            Predicted covariances, shape ``(ns, ns, N+1)``.
        x_u : np.ndarray
            Filter-updated state means, shape ``(ns, N)``.
        W_u : np.ndarray
            Filter-updated covariances, shape ``(ns, ns, N)``.

        Notes
        -----
        Trailing time-index layout (last axis is time) is preserved from
        MATLAB so element-wise comparison with ``matlab_gold/*.mat``
        fixtures is direct.
        """
        # Lazy local imports — match MATLAB-side namespacing and keep the
        # top-level import graph small.
        from nstat.history import History
        from nstat._spike_train_impl import nspikeTrain

        dN = _as_observation_matrix(dN)
        numCells, N = dN.shape

        A = np.asarray(A, dtype=float)
        Q = np.asarray(Q, dtype=float)
        C = np.asarray(C, dtype=float)
        R = np.asarray(R, dtype=float)
        y = np.asarray(y, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        mu = np.asarray(mu, dtype=float)
        beta = np.asarray(beta, dtype=float)

        ns = int(A.shape[0])

        # MATLAB default-handling block (lines 209-227).
        if _is_empty_value(Px0):
            Px0 = np.zeros((ns, ns), dtype=float)
        else:
            Px0 = np.asarray(Px0, dtype=float)
        if _is_empty_value(x0):
            x0 = np.zeros((ns, 1), dtype=float)
        else:
            x0 = np.asarray(x0, dtype=float).reshape(ns, 1)
        if _is_empty_value(windowTimes):
            windowTimes = []
        if _is_empty_value(gamma):
            gamma = 0
        if delta is None or (isinstance(delta, np.ndarray) and delta.size == 0):
            delta = 0.001
        delta = float(delta)

        minTime = 0.0
        maxTime = (dN.shape[1] - 1) * delta

        # --- History tensor management (MATLAB lines 234-261) -------------
        # MATLAB recomputes HkAll when both HkAll and windowTimes are
        # non-empty (effectively a refresh), otherwise installs a zero
        # tensor and zero gamma. We mirror that branching exactly.
        if not _is_empty_value(HkAll):
            if not _is_empty_value(windowTimes):
                histObj = History(windowTimes, minTime, maxTime)
                num_hist = max(len(np.asarray(windowTimes).reshape(-1)) - 1, 1)
                HkAll = np.zeros((N, num_hist, numCells), dtype=float)
                for c in range(numCells):
                    spike_times = np.flatnonzero(dN[c, :] == 1) * delta
                    nst_c = nspikeTrain(spike_times)
                    nst_c.setMinTime(minTime)
                    nst_c.setMaxTime(maxTime)
                    nst_c = nst_c.resample(1.0 / delta)
                    HkAll[:, :, c] = histObj.computeHistory(nst_c).dataToMatrix()
                gamma_arr = np.atleast_2d(np.asarray(gamma, dtype=float))
                if gamma_arr.shape[0] == 1 and gamma_arr.shape[1] not in (1, numCells):
                    # caller passed a row of history coefficients ->
                    # interpret as a column per MATLAB convention.
                    gamma_arr = gamma_arr.T
                if gamma_arr.shape[1] == 1 and numCells > 1:
                    # MATLAB note: "if more than 1 cell but only 1 gamma".
                    gammaNew = np.tile(gamma_arr, (1, numCells))
                else:
                    gammaNew = gamma_arr
                gamma = gammaNew
            else:
                HkAll = np.asarray(HkAll, dtype=float)
        else:
            # MATLAB lines 253-260: zero history + zero gamma per cell.
            HkAll = np.zeros((N, 1, numCells), dtype=float)
            gamma = np.zeros((1, numCells), dtype=float)

        # --- Allocate output tensors (MATLAB lines 266-269) ---------------
        ncols_A = A.shape[1]
        x_p = np.zeros((ncols_A, N + 1), dtype=float)
        x_u = np.zeros((ncols_A, N), dtype=float)
        W_p = np.zeros((ncols_A, ncols_A, N + 1), dtype=float)
        W_u = np.zeros((ncols_A, ncols_A, N), dtype=float)

        # MATLAB: A1 = A(:,:,min(size(A,3),1)) -> first slab if 3-D.
        A1 = A[:, :, 0] if A.ndim == 3 else A
        Q1 = Q[:, :, 0] if Q.ndim == 3 else Q

        x_p[:, 0:1] = A1 @ x0
        W_p[:, :, 0] = A1 @ Px0 @ A1.T + Q1

        # MATLAB ``permute(HkAll, [2 3 1])`` maps (N, K, C) -> (K, C, N).
        # PPLFP_Decode_update expects history with "time on 3rd index".
        Histtermperm = HkAll  # PPLFP_Decode_update expects (N, n_windows, num_cells)

        # --- Forward filter loop (MATLAB lines 275-289) -------------------
        for n in range(N):
            # Time-varying parameter slices (MATLAB ``min(size(.,3), n)``).
            if C.ndim == 3:
                C_n = C[:, :, min(C.shape[2] - 1, n)]
            else:
                C_n = C
            if R.ndim == 3:
                R_n = R[:, :, min(R.shape[2] - 1, n)]
            else:
                R_n = R
            if alpha.ndim == 3:
                alpha_n = alpha[:, :, min(alpha.shape[2] - 1, n)]
            elif alpha.ndim == 2:
                alpha_n = alpha
            else:
                alpha_n = alpha.reshape(-1, 1)

            y_n = (
                y[:, n : n + 1]
                if y.ndim >= 2
                else np.asarray([y[n]], dtype=float).reshape(-1, 1)
            )

            update_result = PPLFP.PPLFP_Decode_update(
                x_p[:, n : n + 1],
                W_p[:, :, n],
                C_n,
                R_n,
                y_n,
                alpha_n,
                dN,
                mu,
                beta,
                fitType,
                gamma,
                Histtermperm,
                n,
                None,
            )
            # MATLAB returns ``[x_u, W_u, lambdaDeltaMat]``; the call site
            # only consumes the first two. Tolerate either arity.
            if isinstance(update_result, tuple) and len(update_result) >= 2:
                xu_n, Wu_n = update_result[0], update_result[1]
            else:  # pragma: no cover
                xu_n, Wu_n = update_result

            x_u[:, n : n + 1] = np.asarray(xu_n, dtype=float).reshape(-1, 1)
            W_u[:, :, n] = Wu_n

            if n < N - 1:
                if A.ndim == 3:
                    A_n = A[:, :, min(A.shape[2] - 1, n)]
                else:
                    A_n = A
                if Q.ndim == 3:
                    Q_n = Q[:, :, min(Q.shape[2] - 1, n)]
                else:
                    Q_n = Q

                xp_next, Wp_next = PPLFP.PPLFP_Decode_predict(
                    x_u[:, n : n + 1],
                    W_u[:, :, n],
                    A_n,
                    Q_n,
                )
                x_p[:, n + 1 : n + 2] = np.asarray(xp_next, dtype=float).reshape(-1, 1)
                W_p[:, :, n + 1] = Wp_next

        return x_p, W_p, x_u, W_u

    @staticmethod
    def PPLFP_fixedIntervalSmoother(
        A,
        Q,
        C,
        R,
        y,
        alpha,
        dN,
        lags,
        mu,
        beta,
        fitType,
        delta: float = 0.001,
        gamma: Any = None,
        windowTimes: Any = None,
        x0: Any = None,
        Px0: Any = None,
        HkAll: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fixed-interval (state-augmentation) smoother for the PPLFP filter.

        Direct port of MATLAB ``PPLFP_fixedIntervalSmoother`` in
        ``+nstat/+decoding/PPLFP.m`` (lines 34-135).  Builds a
        ``(lags+1) * nStates``-dimensional augmented state-space system in
        which lagged copies of the state are stacked into a single vector,
        runs the linear-CIF PPLFP forward filter on that augmented system
        via :meth:`PPLFP_DecodeLinear`, and returns the trailing
        ``nStates`` block — the smoothed marginal of the state at lag
        ``lags`` relative to the current time index.

        Parameters
        ----------
        A, Q : np.ndarray
            State-transition and process-noise covariance matrices.  May be
            ``(ns, ns)`` (time-invariant) or ``(ns, ns, N)`` (time-varying).
        C, R : np.ndarray
            Linear-Gaussian observation matrix and noise covariance for the
            continuous LFP-power channel.  Either static or ``(*, *, N)``.
        y : np.ndarray
            ``(nObs, N)`` matrix of continuous observations.
        alpha : np.ndarray
            Observation offset; ``(nObs, 1)`` or ``(nObs, N)``.
        dN : np.ndarray
            ``(numCells, N)`` spike-train indicator matrix (0/1).
        lags : int
            Smoother lag; the returned estimate is the marginal of
            ``x_{n - lags}`` given measurements through time ``n``.
        mu : np.ndarray
            ``(numCells,)`` baseline log-firing rates.
        beta : np.ndarray
            ``(nStates, numCells)`` state-coefficients for the CIF.
        fitType : str
            ``'poisson'`` or ``'binomial'`` — controls the CIF link.
        delta : float, optional
            Seconds per time step.  Default ``0.001``.
        gamma : array_like, optional
            History coefficients ``(nWindows, numCells)``.
        windowTimes : array_like, optional
            Edges (in seconds) of the history-window partition.
        x0 : array_like, optional
            Initial state mean ``(ns,)``.
        Px0 : array_like, optional
            Initial state covariance ``(ns, ns)``.
        HkAll : array_like, optional
            Pre-computed history tensor ``(N, nWindows, numCells)``.

        Returns
        -------
        x_pLag : np.ndarray
            Predicted-state lag block, shape ``(nStates, N+1)``.
        W_pLag : np.ndarray
            Predicted-covariance lag block, shape
            ``(nStates, nStates, N+1)``.
        x_uLag : np.ndarray
            Updated-state lag block, shape ``(nStates, N)``.
        W_uLag : np.ndarray
            Updated-covariance lag block, shape ``(nStates, nStates, N)``.

        Notes
        -----
        Numerically equivalent to
        ``DecodingAlgorithms.mPPCO_fixedIntervalSmoother`` — the historical
        ``mPPCO_`` name was renamed to ``PPLFP_`` in commit 428c344.
        History construction follows the MATLAB ``History`` /
        ``nspikeTrain`` path (via ``_compute_history_terms``) when
        ``HkAll`` is not pre-supplied.
        """
        # Local lazy import — _compute_history_terms is module-private in
        # decoding_algorithms and only needed when windowTimes is provided.
        from nstat.decoding_algorithms import _compute_history_terms

        A_arr = np.asarray(A, dtype=float)
        Q_arr = np.asarray(Q, dtype=float)
        C_arr = np.asarray(C, dtype=float)
        R_arr = np.asarray(R, dtype=float)
        obs = _as_observation_matrix(dN)
        numCells, N = obs.shape
        nStates = A_arr.shape[1] if A_arr.ndim >= 2 else int(A_arr.size)
        nObs = C_arr.shape[0] if C_arr.ndim >= 2 else int(C_arr.size)
        ns = A_arr.shape[0]

        # --- MATLAB nargin-style defaults (lines 41-60) ---
        if _is_empty_value(HkAll):
            HkAll_local = None
        else:
            HkAll_local = np.asarray(HkAll, dtype=float)

        if _is_empty_value(Px0):
            Px0 = np.zeros((ns, ns), dtype=float)
        else:
            Px0 = np.asarray(Px0, dtype=float).reshape(ns, ns)

        if _is_empty_value(x0):
            x0 = np.zeros(ns, dtype=float)
        else:
            x0 = np.asarray(x0, dtype=float).reshape(-1)

        if _is_empty_value(windowTimes):
            windowTimes_local = None
        else:
            windowTimes_local = np.asarray(windowTimes, dtype=float).reshape(-1)

        if gamma is None:
            gamma = 0
        if delta is None:
            delta = 0.001

        # --- History construction (MATLAB lines 63-95) ---
        if windowTimes_local is not None and windowTimes_local.size > 0:
            HkAll_local = _compute_history_terms(obs, delta, windowTimes_local)
            gamma_arr = np.asarray(gamma, dtype=float)
            # MATLAB: if size(gamma,2)==1 && numCells>1, tile across cells
            if gamma_arr.ndim <= 1 and gamma_arr.size > 0 and numCells > 1:
                gamma = np.tile(gamma_arr.reshape(-1, 1), (1, numCells))
            elif (
                gamma_arr.ndim == 2
                and gamma_arr.shape[1] == 1
                and numCells > 1
            ):
                gamma = np.tile(gamma_arr, (1, numCells))
            else:
                gamma = gamma_arr
        else:
            # MATLAB else-branch (lines 84-92): HkAll(:,:,c)=zeros(N,1);
            # gammaNew(c)=0; gamma=gammaNew;
            HkAll_local = np.zeros((N, 1, numCells), dtype=float)
            gamma = np.zeros(numCells, dtype=float)

        # MATLAB lines 93-95: transpose gamma if its second dim isn't numCells
        gamma_arr = np.asarray(gamma, dtype=float)
        if gamma_arr.ndim == 2 and gamma_arr.shape[1] != numCells:
            gamma = gamma_arr.T

        # --- Augmented (lagged) system construction (MATLAB lines 98-123) ---
        lags = int(lags)
        aug_dim = (lags + 1) * nStates

        Alag = np.zeros((aug_dim, aug_dim, N), dtype=float)
        Qlag = np.zeros((aug_dim, aug_dim, N), dtype=float)
        Clag = np.zeros((nObs, aug_dim, N), dtype=float)
        Rlag = np.zeros((nObs, nObs, N), dtype=float)
        x0lag = np.zeros(int(np.size(x0)) * (lags + 1), dtype=float)
        Px0lag = np.zeros((aug_dim, aug_dim), dtype=float)
        Px0lag[:nStates, :nStates] = Px0
        x0lag[:nStates] = x0

        def _sel(arr3: np.ndarray, n: int) -> np.ndarray:
            """Index a 3-D MATLAB-style array on its last axis with clamping."""
            if arr3.ndim == 3:
                return arr3[:, :, min(n, arr3.shape[2] - 1)]
            return arr3

        for n in range(N):
            offset = 0
            for i in range(lags + 1):
                if i == 0:
                    Alag[offset:offset + nStates, offset:offset + nStates, n] = _sel(
                        A_arr, n
                    )
                    Qlag[offset:offset + nStates, offset:offset + nStates, n] = _sel(
                        Q_arr, n
                    )
                    Clag[:nObs, offset:offset + nStates, n] = _sel(C_arr, n)
                    Rlag[:nObs, :nObs, n] = _sel(R_arr, n)
                else:
                    # MATLAB: Alag block is identity from the previous lag slot.
                    Alag[
                        offset:offset + nStates,
                        offset - nStates:offset,
                        n,
                    ] = np.eye(nStates)
                    # Qlag and Clag blocks remain zero by construction.
                offset += nStates

        # --- betaLag (MATLAB lines 125-126) ---
        betaLag = np.zeros((aug_dim, numCells), dtype=float)
        beta_mat = np.asarray(beta, dtype=float)
        if beta_mat.ndim == 1:
            beta_mat = beta_mat.reshape(-1, 1)
        betaLag[:nStates, :numCells] = beta_mat

        # --- Run the linear-CIF PPLFP forward filter on the augmented system ---
        x_p, W_p, x_u, W_u = PPLFP.PPLFP_DecodeLinear(
            Alag,
            Qlag,
            Clag,
            Rlag,
            y,
            alpha,
            obs,
            mu,
            betaLag,
            fitType,
            delta,
            gamma,
            windowTimes_local,
            x0lag,
            Px0lag,
            HkAll_local,
        )

        # --- Extract the trailing lag block (MATLAB lines 130-133) ---
        lag_start = lags * nStates
        lag_end = (lags + 1) * nStates
        x_pLag = x_p[lag_start:lag_end, :]
        W_pLag = W_p[lag_start:lag_end, lag_start:lag_end, :]
        x_uLag = x_u[lag_start:lag_end, :]
        W_uLag = W_u[lag_start:lag_end, lag_start:lag_end, :]

        return x_pLag, W_pLag, x_uLag, W_uLag

    @staticmethod
    def PPLFP_EMCreateConstraints(
        EstimateA: int | bool = 1,
        AhatDiag: int | bool = 0,
        QhatDiag: int | bool = 1,
        QhatIsotropic: int | bool = 0,
        RhatDiag: int | bool = 1,
        RhatIsotropic: int | bool = 0,
        Estimatex0: int | bool = 0,
        EstimatePx0: int | bool = 0,
        Px0Isotropic: int | bool = 0,
        mcIter: int = 1000,
        EnableIkeda: int | bool = 0,
    ) -> dict:
        """Construct the PPLFP EM constraint struct.

        Python port of MATLAB
        ``+nstat/+decoding/PPLFP.m::PPLFP_EMCreateConstraints``
        (lines 389-449).

        Defaults follow the repaired MATLAB (fix/pp-em round 3, B8):
        ``EstimateA=1, AhatDiag=0, QhatDiag=1, QhatIsotropic=0, RhatDiag=1,
        RhatIsotropic=0, Estimatex0=0, EstimatePx0=0, Px0Isotropic=0,
        mcIter=1000, EnableIkeda=0``.  x0 and Px0 are NOT estimated by default
        (they were 1 and 1): the Px0 M-step ``(x0hat - x0)(x0hat - x0)' .* I``
        is a single-sample estimate that collapses to ~0 (or slightly
        negative) after one iteration and sends the E-step log-likelihood to
        +Inf / NaN (MATLAB report C2).  Pass ``Estimatex0=1, EstimatePx0=1``
        explicitly for the old behaviour.  Isotropic constraints are only
        honored when the corresponding diagonal/estimate flag is enabled,
        matching the MATLAB conditional gating.

        Parameters
        ----------
        EstimateA : bool/int, default 1
            Whether to estimate the state-transition matrix A.
        AhatDiag : bool/int, default 0
            Constrain Ahat to a diagonal matrix.
        QhatDiag : bool/int, default 1
            Constrain Qhat to a diagonal matrix.
        QhatIsotropic : bool/int, default 0
            Constrain Qhat to an isotropic (scalar * I) matrix.
            Only effective when ``QhatDiag`` is true.
        RhatDiag : bool/int, default 1
            Constrain Rhat to a diagonal matrix.
        RhatIsotropic : bool/int, default 0
            Constrain Rhat to an isotropic matrix. Only effective when
            ``RhatDiag`` is true.
        Estimatex0 : bool/int, default 0
            Whether to estimate the initial state x0.
        EstimatePx0 : bool/int, default 0
            Whether to estimate the initial state covariance Px0.
        Px0Isotropic : bool/int, default 0
            Constrain Px0hat to isotropic. Only effective when
            ``EstimatePx0`` is true.
        mcIter : int, default 1000
            Monte Carlo iterations for parameter-uncertainty estimation.
        EnableIkeda : bool/int, default 0
            Enable Ikeda-style stability constraint during EM.

        Returns
        -------
        C : dict
            Constraint struct mirroring MATLAB's ``C`` with fields:
            ``EstimateA``, ``AhatDiag``, ``QhatDiag``, ``QhatIsotropic``,
            ``RhatDiag``, ``RhatIsotropic``, ``Estimatex0``,
            ``EstimatePx0``, ``Px0Isotropic``, ``mcIter``,
            ``EnableIkeda``.
        """
        C: dict = {}
        C["EstimateA"] = EstimateA
        C["AhatDiag"] = AhatDiag
        C["QhatDiag"] = QhatDiag
        if QhatDiag and QhatIsotropic:
            C["QhatIsotropic"] = 1
        else:
            C["QhatIsotropic"] = 0
        C["RhatDiag"] = RhatDiag
        if RhatDiag and RhatIsotropic:
            C["RhatIsotropic"] = 1
        else:
            C["RhatIsotropic"] = 0
        C["Estimatex0"] = Estimatex0
        C["EstimatePx0"] = EstimatePx0
        if EstimatePx0 and Px0Isotropic:
            C["Px0Isotropic"] = 1
        else:
            C["Px0Isotropic"] = 0
        C["mcIter"] = mcIter
        C["EnableIkeda"] = EnableIkeda
        return C

    @staticmethod
    @_matlab_float_semantics
    def PPLFP_ComputeParamStandardErrors(
        y,
        dN,
        xKFinal,
        WKFinal,
        Ahat,
        Qhat,
        Chat,
        Rhat,
        alphahat,
        x0hat,
        Px0hat,
        ExpectationSumsFinal,
        fitType,
        muhat,
        betahat,
        gammahat,
        windowTimes,
        HkAll,
        PPLFP_EM_Constraints=None,
    ):
        """Observed-information standard errors for PPLFP parameters.

        MATLAB cross-reference: ``DecodingAlgorithms.PPLFP_ComputeParamStandardErrors``
        in ``+nstat/+decoding/PPLFP.m`` lines 450–1576.

        Uses the inverse observed information matrix to estimate standard
        errors of the EM-estimated parameters via McLachlan-Krishnan
        Eq. 4.7::

            Io(theta;y) = Ic(theta;y) - Im(theta;y)
                        = Ic(theta;y) - cov(Sc(X;theta) Sc(X;theta)')

        Ic is computed term by term; the covariance of the complete-data
        score is approximated by Monte Carlo.

        Returns
        -------
        SE : dict
            Standard-error structures, MATLAB ``SE`` struct.
        Pvals : dict
            Parameter p-values, MATLAB ``Pvals`` struct.
        nTerms : int
            Total number of parameters in the score vector.
        """
        # ------------------------------------------------------------------
        # Default constraints
        # ------------------------------------------------------------------
        if PPLFP_EM_Constraints is None:
            PPLFP_EM_Constraints = PPLFP.PPLFP_EMCreateConstraints()

        def _get(name, default):
            if isinstance(PPLFP_EM_Constraints, dict):
                return PPLFP_EM_Constraints.get(name, default)
            return getattr(PPLFP_EM_Constraints, name, default)

        EstimateA = int(_get("EstimateA", 1))
        AhatDiag = int(_get("AhatDiag", 0))
        RhatDiag = int(_get("RhatDiag", 0))
        RhatIsotropic = int(_get("RhatIsotropic", 0))
        QhatDiag = int(_get("QhatDiag", 0))
        QhatIsotropic = int(_get("QhatIsotropic", 0))
        EstimatePx0 = int(_get("EstimatePx0", 0))
        Px0Isotropic = int(_get("Px0Isotropic", 0))
        Estimatex0 = int(_get("Estimatex0", 0))
        mcIter = int(_get("mcIter", 500))

        # Coerce to numpy column-major mirroring MATLAB layouts.
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        dN = np.asarray(dN, dtype=float)
        if dN.ndim == 1:
            dN = dN.reshape(1, -1)
        xKFinal = np.asarray(xKFinal, dtype=float)
        WKFinal = np.asarray(WKFinal, dtype=float)
        Ahat = np.atleast_2d(np.asarray(Ahat, dtype=float))
        Qhat = np.atleast_2d(np.asarray(Qhat, dtype=float))
        Chat = np.atleast_2d(np.asarray(Chat, dtype=float))
        Rhat = np.atleast_2d(np.asarray(Rhat, dtype=float))
        alphahat = np.asarray(alphahat, dtype=float)
        if alphahat.ndim == 1:
            alphahat = alphahat.reshape(-1, 1)
        x0hat = np.asarray(x0hat, dtype=float)
        if x0hat.ndim == 1:
            x0hat = x0hat.reshape(-1, 1)
        Px0hat = np.atleast_2d(np.asarray(Px0hat, dtype=float))
        muhat = np.asarray(muhat, dtype=float).reshape(-1)
        betahat = np.atleast_2d(np.asarray(betahat, dtype=float))
        gammahat = np.asarray(gammahat, dtype=float)
        HkAll = np.asarray(HkAll, dtype=float)
        # MATLAB stores a one-cell N x nW x 1 history as N x nW; its HkAll(k,:,c)
        # slices read that directly (B6: single-cell history SEs).
        if betahat.shape[1] == 1 and HkAll.ndim == 2:
            HkAll = HkAll.reshape(HkAll.shape[0], HkAll.shape[1], 1)

        # Expectation sums (dict-like)
        def _es(name):
            if isinstance(ExpectationSumsFinal, dict):
                return np.asarray(ExpectationSumsFinal[name], dtype=float)
            return np.asarray(getattr(ExpectationSumsFinal, name), dtype=float)

        Sxkm1xkm1 = _es("Sxkm1xkm1")
        Sxkxk_ES = _es("Sxkxk")

        # Dimensions
        dy, N = y.shape
        dx = xKFinal.shape[0]
        K = y.shape[1]
        numCells = betahat.shape[1]
        # MATLAB F12 (fix/pp-em 1c051a9): a shared history-coefficient column
        # (numWindows x 1, a scalar for one window) with several cells is
        # expanded to numWindows x numCells, the rule PPLFP_EM applies before
        # it calls this routine (B9); the gamma blocks below are per cell.
        gammahat = _expand_shared_se_gamma(gammahat, windowTimes, numCells)

        # Pre-broadcast WKFinal into 3D (dx, dx, K).
        if WKFinal.ndim == 2:
            WKFinal_3 = np.tile(WKFinal[:, :, None], (1, 1, K))
        else:
            WKFinal_3 = WKFinal

        # ==================================================================
        # COMPLETE INFORMATION MATRICES
        # ==================================================================

        # ---- A complete information matrix ----
        IAComp = None
        if EstimateA == 1:
            n1A, n2A = Ahat.shape
            elA = np.eye(n1A)
            emA = np.eye(n2A)
            if AhatDiag == 1:
                size_a = int(np.size(np.diag(Ahat)))
                IAComp = np.zeros((size_a, size_a))
                cnt = 0
                for l in range(n1A):
                    m = l
                    qinv_el = np.linalg.solve(Qhat, elA[:, l : l + 1])
                    termMat = (qinv_el @ emA[:, m : m + 1].T) @ Sxkm1xkm1 * np.eye(n1A, n2A)
                    termvec = np.diag(termMat)
                    IAComp[:, cnt] = termvec
                    cnt += 1
            else:
                size_a = int(np.size(Ahat))
                IAComp = np.zeros((size_a, size_a))
                cnt = 0
                Qinv = np.linalg.inv(Qhat)
                for l in range(n1A):
                    for m in range(n2A):
                        termMat = Qinv @ elA[:, l : l + 1] @ emA[:, m : m + 1].T @ Sxkm1xkm1
                        termvec = termMat.T.flatten(order="F")
                        IAComp[:, cnt] = termvec
                        cnt += 1

        # ---- C complete information matrix ----
        n1C, n2C = Chat.shape
        elC = np.eye(n1C)
        emC = np.eye(n2C)
        ICComp = np.zeros((int(np.size(Chat)), int(np.size(Chat))))
        cnt = 0
        for l in range(n1C):
            for m in range(n2C):
                termMat = (
                    np.linalg.solve(Rhat, elC[:, l : l + 1]) @ emC[:, m : m + 1].T @ Sxkxk_ES
                )
                termvec = termMat.T.flatten(order="F")
                ICComp[:, cnt] = termvec
                cnt += 1

        # ---- R complete information matrix ----
        n1R, n2R = Rhat.shape
        elR = np.eye(n1R)
        emR = np.eye(n2R)
        if RhatDiag == 1:
            if RhatIsotropic == 1:
                IRComp = np.array([[0.5 * N * dy * Rhat[0, 0] ** (-2)]])
            else:
                size_r = int(np.size(np.diag(Rhat)))
                IRComp = np.zeros((size_r, size_r))
                cnt = 0
                for l in range(n1R):
                    m = l
                    left = np.linalg.solve(Rhat, emR[:, m : m + 1]) * (N / 2.0)
                    termMat_full = left @ elR[:, l : l + 1].T
                    termMat = np.linalg.solve(Rhat.T, termMat_full.T).T
                    termvec = np.diag(termMat)
                    IRComp[:, cnt] = termvec
                    cnt += 1
        else:
            # MATLAB pre-allocates diag-size but writes numel-size vectors here
            # (apparent MATLAB inconsistency); preserve diag-size and write what fits.
            size_r = int(np.size(np.diag(Rhat)))
            IRComp = np.zeros((size_r, size_r))
            cnt = 0
            for l in range(n1R):
                for m in range(n2R):
                    left = np.linalg.solve(Rhat, emR[:, m : m + 1]) * (N / 2.0)
                    termMat_full = left @ elR[:, l : l + 1].T
                    termMat = np.linalg.solve(Rhat.T, termMat_full.T).T
                    termvec = termMat.T.flatten(order="F")
                    if cnt < IRComp.shape[1]:
                        IRComp[: min(termvec.size, IRComp.shape[0]), cnt] = termvec[
                            : IRComp.shape[0]
                        ]
                    cnt += 1

        # ---- Q complete information matrix ----
        n1Q, n2Q = Qhat.shape
        elQ = np.eye(n1Q)
        emQ = np.eye(n2Q)
        if QhatDiag == 1:
            if QhatIsotropic == 1:
                IQComp = np.array([[0.5 * N * dx * Qhat[0, 0] ** (-2)]])
            else:
                size_q = int(np.size(np.diag(Qhat)))
                IQComp = np.zeros((size_q, size_q))
                cnt = 0
                for l in range(n1Q):
                    m = l
                    left = np.linalg.solve(Qhat, emQ[:, m : m + 1]) * (N / 2.0)
                    termMat_full = left @ elQ[:, l : l + 1].T
                    termMat = np.linalg.solve(Qhat.T, termMat_full.T).T
                    termvec = np.diag(termMat)
                    IQComp[:, cnt] = termvec
                    cnt += 1
        else:
            size_q = int(np.size(Qhat))
            IQComp = np.zeros((size_q, size_q))
            cnt = 0
            for l in range(n1Q):
                for m in range(n2Q):
                    left = np.linalg.solve(Qhat, emQ[:, m : m + 1]) * (N / 2.0)
                    termMat_full = left @ elQ[:, l : l + 1].T
                    termMat = np.linalg.solve(Qhat.T, termMat_full.T).T
                    termvec = termMat.T.flatten(order="F")
                    IQComp[:, cnt] = termvec
                    cnt += 1

        # ---- Px0 (S) complete information matrix ----
        ISComp = None
        if EstimatePx0 == 1:
            if Px0Isotropic == 1:
                ISComp = np.array([[0.5 * dx * Px0hat[0, 0] ** (-2)]])
            else:
                n1S, n2S = Px0hat.shape
                elS = np.eye(n1S)
                emS = np.eye(n2S)
                size_s = int(np.size(np.diag(Px0hat)))
                ISComp = np.zeros((size_s, size_s))
                cnt = 0
                for l in range(n1S):
                    m = l
                    left = 0.5 * np.linalg.solve(Px0hat, emS[:, m : m + 1])
                    termMat_full = left @ elS[:, l : l + 1].T
                    termMat = np.linalg.solve(Px0hat.T, termMat_full.T).T
                    termvec = np.diag(termMat)
                    ISComp[:, cnt] = termvec
                    cnt += 1

        # ---- x0 complete information matrix ----
        Ix0Comp = None
        if Estimatex0 == 1:
            term1 = np.linalg.solve(Px0hat.T, np.eye(Px0hat.shape[0])).T
            term2 = np.linalg.solve(Qhat.T, Ahat).T @ Ahat
            Ix0Comp = term1 + term2

        # ---- Alpha complete information matrix ----
        IAlphaComp = np.linalg.solve(Rhat.T, N * np.eye(Rhat.shape[0])).T

        # ==================================================================
        # Monte Carlo draws (expectation phase)
        # ==================================================================
        McExp = mcIter
        rng = np.random.default_rng()

        def _normal(d, n):
            return rng.standard_normal((d, n))

        # Draws through _mc_state_draws (MATLAB mcStateDraws, F9: the draw
        # was chol(W)*z, covariance R*R' instead of W).
        xKDrawExp = np.zeros((dx, K, McExp))
        for k in range(K):
            xKDrawExp[:, k, :] = _mc_state_draws(xKFinal[:, k], WKFinal_3[:, :, k], McExp, _normal, non_pd="zero")

        xkPerm = np.transpose(xKDrawExp, (0, 2, 1))

        # ---- Beta complete information matrix ----
        beta_rows = betahat.shape[0]
        IBetaComp = np.zeros((beta_rows * numCells, beta_rows * numCells))

        is_poisson = (fitType == "poisson")
        gamma_is_scalar = (gammahat.size == 1)

        def _gamma_for_cell(c):
            if gamma_is_scalar:
                return np.atleast_1d(gammahat.reshape(-1))
            return gammahat[:, c]

        for c in range(numCells):
            HessianTerm = np.zeros((dx, dx))
            for k in range(K):
                Hk_row = HkAll[k, :, c].reshape(1, -1)
                xk = xkPerm[:, :, k]
                gammaC = _gamma_for_cell(c)
                terms = (
                    muhat[c]
                    + betahat[:, c].reshape(1, -1) @ xk
                    + (gammaC.reshape(1, -1) @ Hk_row.T)
                )
                terms = terms.reshape(-1)
                if is_poisson:
                    ld = np.exp(terms)
                    HessianTerm = HessianTerm - (1.0 / McExp) * ((ld[None, :] * xk) @ xk.T)
                else:
                    ld = np.exp(terms) / (1.0 + np.exp(terms))
                    Ex1 = (1.0 / McExp) * ((ld[None, :] * xk) @ xk.T)
                    Ex2 = (1.0 / McExp) * (((ld ** 2)[None, :] * xk) @ xk.T)
                    Ex3 = (1.0 / McExp) * (((ld ** 3)[None, :] * xk) @ xk.T)
                    # MATLAB (repaired, A4): the beta Hessian of
                    # dN*log(p) - p, logistic p, is
                    # (-(dN+1)p + (dN+3)p^2 - 2p^3) x x'.  The former
                    # Ex1 + Ex2 - 2*Ex3 had the wrong sign and form.
                    HessianTerm = (HessianTerm - (dN[c, k] + 1) * Ex1
                                   + (dN[c, k] + 3) * Ex2 - 2.0 * Ex3)
            si = beta_rows * c
            ei = beta_rows * (c + 1)
            IBetaComp[si:ei, si:ei] = -HessianTerm

        # ---- Mu complete information matrix ----
        IMuComp = np.zeros((numCells, numCells))
        for c in range(numCells):
            HessianTerm = 0.0
            for k in range(K):
                Hk_row = HkAll[k, :, c].reshape(1, -1)
                xk = xkPerm[:, :, k]
                gammaC = _gamma_for_cell(c)
                terms = (
                    muhat[c]
                    + betahat[:, c].reshape(1, -1) @ xk
                    + (gammaC.reshape(1, -1) @ Hk_row.T)
                )
                terms = terms.reshape(-1)
                if is_poisson:
                    ld = np.exp(terms)
                    HessianTerm = HessianTerm - (1.0 / McExp) * np.sum(ld)
                else:
                    ld = np.exp(terms) / (1.0 + np.exp(terms))
                    Ed = (1.0 / McExp) * np.sum(ld)
                    Ed2 = (1.0 / McExp) * np.sum(ld ** 2)
                    Ed3 = (1.0 / McExp) * np.sum(ld ** 3)
                    # MATLAB (repaired, A1): cubic coefficient -2, not -3.
                    HessianTerm = (
                        HessianTerm
                        - (dN[c, k] + 1) * Ed
                        + (dN[c, k] + 3) * Ed2
                        - 2.0 * Ed3
                    )
            IMuComp[c, c] = -HessianTerm

        # ---- Gamma complete information matrix ----
        nHist = HkAll.shape[1]
        IGammaComp = np.zeros((max(gammahat.size, 1), max(gammahat.size, 1)))
        gamma_active = (
            (windowTimes is not None)
            and (np.size(windowTimes) > 0)
            and bool(np.any(gammahat != 0))
        )
        if gamma_active:
            IGammaComp = np.zeros((nHist * numCells, nHist * numCells))
            for c in range(numCells):
                HessianTerm = np.zeros((nHist, nHist))
                for k in range(K):
                    Hk_row = HkAll[k, :, c].reshape(1, -1)
                    xk = xkPerm[:, :, k]
                    gammaC = _gamma_for_cell(c)
                    terms = (
                        muhat[c]
                        + betahat[:, c].reshape(1, -1) @ xk
                        + (gammaC.reshape(1, -1) @ Hk_row.T)
                    )
                    terms = terms.reshape(-1)
                    if is_poisson:
                        ld = np.exp(terms)
                        Ed = (1.0 / McExp) * np.sum(ld)
                        HessianTerm = HessianTerm - (Hk_row.T @ Hk_row) * Ed
                    else:
                        ld = np.exp(terms) / (1.0 + np.exp(terms))
                        Ed = (1.0 / McExp) * np.sum(ld)
                        Ed2 = (1.0 / McExp) * np.sum(ld ** 2)
                        Ed3 = (1.0 / McExp) * np.sum(ld ** 3)
                        HessianTerm = HessianTerm + (
                            -Ed * (dN[c, k] + 1)
                            + Ed2 * (dN[c, k] + 3)
                            - 2.0 * Ed3
                        ) * (Hk_row.T @ Hk_row)
                si = nHist * c
                ei = nHist * (c + 1)
                IGammaComp[si:ei, si:ei] = -HessianTerm

        # ==================================================================
        # Assemble the complete-information block matrix
        # ==================================================================
        n1 = IAComp.shape[0] if (EstimateA == 1 and IAComp is not None) else 0
        n2 = IQComp.shape[0]
        n3 = ICComp.shape[0]
        n4 = IRComp.shape[0]
        n5 = ISComp.shape[0] if (EstimatePx0 == 1 and ISComp is not None) else 0
        n6 = Ix0Comp.shape[0] if (Estimatex0 == 1 and Ix0Comp is not None) else 0
        n7 = IAlphaComp.shape[0]
        n8 = IMuComp.shape[0]
        n9 = IBetaComp.shape[0]
        if gamma_is_scalar:
            # MATLAB: if(numel(gammahat)==1 && gammahat==0) n10=0; else n10=size(IGammaComp,1)
            _gscal = float(np.asarray(gammahat).reshape(-1)[0])
            n10 = 0 if _gscal == 0.0 else IGammaComp.shape[0]
        else:
            n10 = IGammaComp.shape[0]

        nTerms = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10
        IComp = np.zeros((nTerms, nTerms))
        if EstimateA == 1 and n1 > 0:
            IComp[0:n1, 0:n1] = IAComp
        off = n1
        IComp[off : off + n2, off : off + n2] = IQComp
        off += n2
        IComp[off : off + n3, off : off + n3] = ICComp
        off += n3
        IComp[off : off + n4, off : off + n4] = IRComp
        off += n4
        if EstimatePx0 == 1 and n5 > 0:
            IComp[off : off + n5, off : off + n5] = ISComp
        off += n5
        if Estimatex0 == 1 and n6 > 0:
            IComp[off : off + n6, off : off + n6] = Ix0Comp
        off += n6
        IComp[off : off + n7, off : off + n7] = IAlphaComp
        off += n7
        IComp[off : off + n8, off : off + n8] = IMuComp
        off += n8
        IComp[off : off + n9, off : off + n9] = IBetaComp
        off += n9
        if n10 > 0:
            IComp[off : off + n10, off : off + n10] = IGammaComp

        # ==================================================================
        # MISSING INFORMATION MATRIX (Monte Carlo)
        # ==================================================================
        # Draws through _mc_state_draws (MATLAB mcStateDraws, F9).
        Mc = mcIter
        xKDraw = np.zeros((dx, N, Mc))
        for n_idx in range(N):
            xKDraw[:, n_idx, :] = _mc_state_draws(xKFinal[:, n_idx], WKFinal_3[:, :, n_idx], Mc, _normal,
                                                  non_pd="zero")

        if EstimatePx0 or Estimatex0:
            x0Draw = _mc_state_draws(x0hat, Px0hat, Mc, _normal, non_pd="zero")
        else:
            x0Draw = np.tile(x0hat, (1, Mc))

        IMc = np.zeros((nTerms, nTerms, Mc))

        for c in range(Mc):
            x_K = xKDraw[:, :, c]
            x_0 = x0Draw[:, c : c + 1]
            Dx = x_K.shape[0]
            Dy = y.shape[0]

            Sxkm1xk = np.zeros((Dx, Dx))
            Sxkm1xkm1_loc = np.zeros((Dx, Dx))
            Sxkxk_loc = np.zeros((Dx, Dx))
            Sykyk = np.zeros((Dy, Dy))
            Sxkyk = np.zeros((Dx, Dy))

            for k in range(K):
                xk_col = x_K[:, k : k + 1]
                if k == 0:
                    Sxkm1xk = Sxkm1xk + x_0 @ xk_col.T
                    Sxkm1xkm1_loc = Sxkm1xkm1_loc + x_0 @ x_0.T
                else:
                    xkm1 = x_K[:, k - 1 : k]
                    Sxkm1xk = Sxkm1xk + xkm1 @ xk_col.T
                    Sxkm1xkm1_loc = Sxkm1xkm1_loc + xkm1 @ xkm1.T
                Sxkxk_loc = Sxkxk_loc + xk_col @ xk_col.T
                yk_col = y[:, k : k + 1]
                ymalpha = yk_col - alphahat
                Sykyk = Sykyk + ymalpha @ ymalpha.T
                Sxkyk = Sxkyk + xk_col @ ymalpha.T

            Sxkxk_loc = 0.5 * (Sxkxk_loc + Sxkxk_loc.T)
            Sykyk = 0.5 * (Sykyk + Sykyk.T)
            sumXkTerms = (
                Sxkxk_loc
                - Ahat @ Sxkm1xk
                - Sxkm1xk.T @ Ahat.T
                + Ahat @ Sxkm1xkm1_loc @ Ahat.T
            )
            sumYkTerms = (
                Sykyk
                - Chat @ Sxkyk
                - Sxkyk.T @ Chat.T
                + Chat @ Sxkxk_loc @ Chat.T
            )
            Sxkxkm1 = Sxkm1xk.T
            Sykxk = Sxkyk.T
            sumXkTerms = 0.5 * (sumXkTerms + sumXkTerms.T)
            sumYkTerms = 0.5 * (sumYkTerms + sumYkTerms.T)

            if EstimateA == 1:
                ScorA = np.linalg.solve(Qhat, Sxkxkm1 - Ahat @ Sxkm1xkm1_loc)
                if AhatDiag == 1:
                    ScoreAMc = np.diag(ScorA).reshape(-1, 1)
                else:
                    ScoreAMc = ScorA.T.flatten(order="F").reshape(-1, 1)
            else:
                ScoreAMc = np.zeros((0, 1))

            ScorC = np.linalg.solve(Rhat, Sykxk - Chat @ Sxkxk_loc)
            ScoreCMc = ScorC.T.flatten(order="F").reshape(-1, 1)

            if QhatDiag:
                if QhatIsotropic:
                    ScoreQ_val = -0.5 * (
                        K * Dx * Qhat[0, 0] ** (-1)
                        - Qhat[0, 0] ** (-2) * np.trace(sumXkTerms)
                    )
                    ScoreQMc = np.array([[ScoreQ_val]])
                else:
                    inner = (
                        K * np.eye(Qhat.shape[0])
                        - np.linalg.solve(Qhat.T, sumXkTerms.T).T
                    )
                    ScoreQ = -0.5 * np.linalg.solve(Qhat, inner)
                    ScoreQMc = np.diag(ScoreQ).reshape(-1, 1)
            else:
                inner = (
                    K * np.eye(Qhat.shape[0])
                    - np.linalg.solve(Qhat.T, sumXkTerms.T).T
                )
                ScoreQ = -0.5 * np.linalg.solve(Qhat, inner)
                ScoreQMc = ScoreQ.T.flatten(order="F").reshape(-1, 1)

            ScoreAlphaMc = np.sum(
                np.linalg.solve(Rhat, y - Chat @ x_K - alphahat @ np.ones((1, N))),
                axis=1,
            ).reshape(-1, 1)

            if RhatDiag:
                if RhatIsotropic:
                    ScoreR_val = -0.5 * (
                        K * Dy * Rhat[0, 0] ** (-1)
                        - Rhat[0, 0] ** (-2) * np.trace(sumYkTerms)
                    )
                    ScoreRMc = np.array([[ScoreR_val]])
                else:
                    inner = (
                        K * np.eye(Rhat.shape[0])
                        - np.linalg.solve(Rhat.T, sumYkTerms.T).T
                    )
                    ScoreR = -0.5 * np.linalg.solve(Rhat, inner)
                    ScoreRMc = np.diag(ScoreR).reshape(-1, 1)
            else:
                inner = (
                    K * np.eye(Rhat.shape[0])
                    - np.linalg.solve(Rhat.T, sumYkTerms.T).T
                )
                ScoreR = -0.5 * np.linalg.solve(Rhat, inner)
                ScoreRMc = ScoreR.T.flatten(order="F").reshape(-1, 1)

            if Px0Isotropic == 1:
                diff = x_0 - x0hat
                ScoreSMc = np.array(
                    [
                        [
                            -0.5
                            * (
                                Dx * Px0hat[0, 0] ** (-1)
                                - Px0hat[0, 0] ** (-2) * np.trace(diff @ diff.T)
                            )
                        ]
                    ]
                )
            else:
                diff = x_0 - x0hat
                outer = diff @ diff.T
                inner = np.eye(Px0hat.shape[0]) - np.linalg.solve(Px0hat.T, outer.T).T
                ScorS = -0.5 * np.linalg.solve(Px0hat, inner)
                ScoreSMc = np.diag(ScorS).reshape(-1, 1)

            # x0 score
            Scorx0 = -np.linalg.solve(Px0hat, x_0 - x0hat) + (
                np.linalg.solve(Qhat.T, Ahat).T @ (x_K[:, 0:1] - Ahat @ x_0)
            )
            Scorex0Mc = Scorx0.T.flatten(order="F").reshape(-1, 1)

            ScoreMuMc = np.zeros((numCells, 1))
            ScoreBetaList = []
            ScoreGammaList = []

            for nc in range(numCells):
                Hk_full = HkAll[:, :, nc]  # (K, nHist)
                gammaC = _gamma_for_cell(nc)
                terms_row = (
                    muhat[nc]
                    + betahat[:, nc].reshape(1, -1) @ x_K
                    + gammaC.reshape(1, -1) @ Hk_full.T
                )
                terms_row = terms_row.reshape(-1)
                if is_poisson:
                    ld = np.exp(terms_row)
                    ScoreMuMc[nc, 0] = np.sum(dN[nc, :] - ld)
                    sb = np.sum(((dN[nc, :] - ld)[None, :]) * x_K, axis=1).reshape(-1, 1)
                    ScoreBetaList.append(sb)
                    sg = np.sum(
                        ((dN[nc, :] - ld)[None, :]) * Hk_full.T, axis=1
                    ).reshape(-1, 1)
                    ScoreGammaList.append(sg)
                else:
                    ld = np.exp(terms_row) / (1.0 + np.exp(terms_row))
                    ScoreMuMc[nc, 0] = np.sum(
                        dN[nc, :] - (dN[nc, :] + 1) * ld + ld ** 2
                    )
                    sb = np.sum(
                        ((dN[nc, :] * (1 - ld) - ld * (1 - ld))[None, :]) * x_K,
                        axis=1,
                    ).reshape(-1, 1)
                    ScoreBetaList.append(sb)
                    sg = np.sum(
                        ((dN[nc, :] - (dN[nc, :] + 1) * ld + ld ** 2)[None, :])
                        * Hk_full.T,
                        axis=1,
                    ).reshape(-1, 1)
                    ScoreGammaList.append(sg)

            ScoreBetaMc = (
                np.vstack(ScoreBetaList) if ScoreBetaList else np.zeros((0, 1))
            )
            ScoreGammaMc = (
                np.vstack(ScoreGammaList) if ScoreGammaList else np.zeros((0, 1))
            )

            parts = [ScoreAMc, ScoreQMc, ScoreCMc, ScoreRMc]
            if EstimatePx0 == 1:
                parts.append(ScoreSMc)
            if Estimatex0 == 1:
                parts.append(Scorex0Mc)
            parts.append(ScoreAlphaMc)
            parts.append(ScoreMuMc)
            parts.append(ScoreBetaMc)
            include_gamma = (
                (gamma_is_scalar and float(np.asarray(gammahat).reshape(-1)[0]) != 0.0)
                or (not gamma_is_scalar)
            )
            if include_gamma:
                parts.append(ScoreGammaMc)
            ScoreVec = np.vstack(parts)
            IMc[:, :, c] = ScoreVec @ ScoreVec.T

        IMissing = (1.0 / Mc) * np.sum(IMc, axis=2)
        IObs = IComp - IMissing

        # ------------------------------------------------------------------
        # Invert IObs and project to nearest SPD
        # ------------------------------------------------------------------
        try:
            invIObs = np.linalg.solve(IObs, np.eye(IObs.shape[0]))
        except np.linalg.LinAlgError:
            # Python-only: an exactly singular IObs.  MATLAB's eye/IObs warns and
            # returns Inf, and nearestSPD's svd then errors.
            invIObs = np.linalg.pinv(IObs)

        invIObs = _nearest_spd(invIObs)
        VarVec = np.diag(invIObs)
        SEVec = np.sqrt(VarVec)  # MATLAB sqrt(VarVec); positive after nearestSPD (was sqrt(|VarVec|))

        idx = 0
        SEAterms = SEVec[idx : idx + n1]
        idx += n1
        SEQterms = SEVec[idx : idx + n2]
        idx += n2
        SECterms = SEVec[idx : idx + n3]
        idx += n3
        SERterms = SEVec[idx : idx + n4]
        idx += n4
        SEPx0terms = SEVec[idx : idx + n5]
        idx += n5
        SEx0terms = SEVec[idx : idx + n6]
        idx += n6
        SEAlphaterms = SEVec[idx : idx + n7]
        idx += n7
        SEMuTerms = SEVec[idx : idx + n8]
        idx += n8
        SEBetaTerms = SEVec[idx : idx + n9]
        idx += n9
        SEGammaTerms = SEVec[idx : idx + n10]

        SE = {}
        SES = None
        if EstimatePx0 == 1 and n5 > 0:
            SES = np.diag(SEPx0terms)
        if Estimatex0 == 1 and n6 > 0:
            SEx0 = SEx0terms
        else:
            SEx0 = None

        if EstimateA == 1 and n1 > 0:
            if AhatDiag == 1:
                SEA = np.diag(SEAterms)
            else:
                # The A / C / full-Q parameters (information columns and
                # scores) are ordered row by row, as in MATLAB, whose
                # reshape(SEterms, ncol, nrow)' unpacks them row-major.  The
                # former reshape(ncol, nrow).T (NumPy is row-major) unpacked
                # them column-major: SE.A and SE.C came back transposed.
                SEA = SEAterms.reshape(Ahat.shape)
            SE["A"] = SEA

        SEC = SECterms.reshape(Chat.shape)
        SEAlpha = SEAlphaterms.reshape(alphahat.shape[1], alphahat.shape[0]).T

        if RhatDiag == 1:
            SER = np.diag(SERterms)
        else:
            SER = SERterms[: Rhat.size].reshape(Rhat.shape[1], Rhat.shape[0]).T

        if QhatDiag == 1:
            SEQ = np.diag(SEQterms)
        else:
            SEQ = SEQterms.reshape(Qhat.shape)  # row-major (see SE.A)

        SE["Q"] = SEQ
        SE["C"] = SEC
        SE["R"] = SER
        SE["alpha"] = SEAlpha
        if EstimatePx0 == 1 and SES is not None:
            SE["Px0"] = SES
        if Estimatex0 == 1 and SEx0 is not None:
            SE["x0"] = SEx0
        SEMu = SEMuTerms
        SEBeta = SEBetaTerms.reshape(betahat.shape[1], betahat.shape[0]).T
        SE["mu"] = SEMu
        SE["beta"] = SEBeta
        SEGamma = None
        if (gamma_is_scalar and float(np.asarray(gammahat).reshape(-1)[0]) != 0.0) or (not gamma_is_scalar):
            if n10 > 0:
                if gammahat.ndim > 1:
                    SEGamma = SEGammaTerms.reshape(
                        gammahat.shape[1], gammahat.shape[0]
                    ).T
                else:
                    SEGamma = SEGammaTerms.reshape(1, gammahat.size).T
                SE["gamma"] = SEGamma

        # ------------------------------------------------------------------
        # p-values via two-sided z-tests (mu = 0)
        # ------------------------------------------------------------------
        def _ztest_p(params, ses):
            # MATLAB ztest(param, 0, se): 2*normcdf(-|param/se|) (see
            # _matlab_ztest_p; this port used 2*(1 - normcdf(|z|)), which
            # underflows to 0 for |z| > 8.3, and returned 1 for se <= 0).
            return _matlab_ztest_p(np.asarray(params, dtype=float).reshape(-1),
                                   np.asarray(ses, dtype=float).reshape(-1))

        Pvals = {}

        if EstimateA == 1 and n1 > 0:
            if AhatDiag == 1:
                vp = np.diag(Ahat)
                vs = np.diag(SE["A"])
                p_a = _ztest_p(vp, vs)
                Pvals["A"] = np.diag(p_a)
            else:
                vp = Ahat.flatten(order="F")
                vs = SE["A"].flatten(order="F")
                p_a = _ztest_p(vp, vs)
                Pvals["A"] = p_a.reshape(Ahat.shape, order="F")

        vp = Chat.flatten(order="F")
        vs = SE["C"].flatten(order="F")
        p_c = _ztest_p(vp, vs)
        Pvals["C"] = p_c.reshape(Chat.shape, order="F")

        if RhatDiag == 1:
            if RhatIsotropic == 1:
                p_r = _ztest_p([Rhat[0, 0]], [SER[0, 0]])
                Pvals["R"] = np.diag(p_r)
            else:
                p_r = _ztest_p(np.diag(Rhat), np.diag(SER))
                Pvals["R"] = np.diag(p_r)
        else:
            vp = Rhat.flatten(order="F")
            vs = SE["R"].flatten(order="F")
            p_r = _ztest_p(vp, vs)
            Pvals["R"] = p_r.reshape(Rhat.shape, order="F")

        if QhatDiag == 1:
            if QhatIsotropic == 1:
                p_q = _ztest_p([Qhat[0, 0]], [SEQ[0, 0]])
                Pvals["Q"] = np.diag(p_q)
            else:
                p_q = _ztest_p(np.diag(Qhat), np.diag(SEQ))
                Pvals["Q"] = np.diag(p_q)
        else:
            vp = Qhat.flatten(order="F")
            vs = SE["Q"].flatten(order="F")
            p_q = _ztest_p(vp, vs)
            Pvals["Q"] = p_q.reshape(Qhat.shape, order="F")

        if EstimatePx0 == 1 and SES is not None:
            if Px0Isotropic == 1:
                p_p = _ztest_p([Px0hat[0, 0]], [SES[0, 0]])
                Pvals["Px0"] = np.diag(p_p)
            else:
                p_p = _ztest_p(np.diag(Px0hat), np.diag(SES))
                Pvals["Px0"] = np.diag(p_p)

        Pvals["alpha"] = _ztest_p(
            alphahat.reshape(-1), SE["alpha"].reshape(-1)
        ).reshape(-1, 1)

        if Estimatex0 == 1 and SEx0 is not None:
            Pvals["x0"] = _ztest_p(x0hat.reshape(-1), SEx0.reshape(-1)).reshape(-1, 1)

        Pvals["mu"] = _ztest_p(muhat, SEMu).reshape(-1, 1)

        vp = betahat.flatten(order="F")
        vs = SE["beta"].flatten(order="F")
        p_b = _ztest_p(vp, vs)
        Pvals["beta"] = p_b.reshape(betahat.shape, order="F")

        include_gamma = (
            (gamma_is_scalar and float(np.asarray(gammahat).reshape(-1)[0]) != 0.0)
            or (not gamma_is_scalar)
        )
        if include_gamma and SEGamma is not None:
            vp = gammahat.flatten(order="F")
            vs = SEGamma.flatten(order="F")
            p_g = _ztest_p(vp, vs)
            Pvals["gamma"] = (
                p_g.reshape(gammahat.shape, order="F")
                if gammahat.ndim > 1
                else p_g
            )

        return SE, Pvals, nTerms

    @staticmethod
    @_matlab_float_semantics
    def PPLFP_EM(
        y,
        dN,
        Ahat0,
        Qhat0,
        Chat0,
        Rhat0,
        alphahat0,
        mu,
        beta,
        fitType: str = "poisson",
        delta: float = 0.001,
        gamma=None,
        windowTimes=None,
        x0=None,
        Px0=None,
        PPLFP_EM_Constraints=None,
        MstepMethod: str = "NewtonRaphson",
    ):
        """Full PPLFP EM driver (E-step / M-step loop).

        Defaults (repaired MATLAB, B8): ``fitType='poisson'``,
        ``delta=0.001``, no history, ``x0 = 0``, ``Px0 = 1e-9 I``,
        ``PPLFP_EM_Constraints = PPLFP_EMCreateConstraints()`` (x0 / Px0 not
        estimated -- that estimator collapses Px0 and drives the log-likelihood
        to +Inf / NaN) and ``MstepMethod='NewtonRaphson'`` (was ``'GLM'``): the
        Monte-Carlo Newton-Raphson M-step maximises the expected complete-data
        log-likelihood over the smoothed state posterior, whereas the 'GLM'
        M-step is a plug-in fit on the smoothed means that ignores W_K,
        inflates beta and drifts.  'GLM' remains selectable.  With ``gamma``
        given and ``windowTimes`` empty there is one history window per
        coefficient (``0:delta:size(gamma,1)*delta``) and a nonzero shared
        gamma column is expanded to every cell.  EM stops before the M-step if
        the E-step log-likelihood is not finite and returns the best finite
        iterate.

        Direct port of MATLAB ``PPLFP_EM`` in
        ``+nstat/+decoding/PPLFP.m`` (lines 1577-1989).  Structurally
        mirrors :func:`mPPCO_EM` in :mod:`nstat.decoding_algorithms` but
        calls the PPLFP-flavoured E-/M-step helpers.

        Parameters mirror the MATLAB signature in name and order.

        Returns
        -------
        xKFinal, WKFinal, Ahat, Qhat, Chat, Rhat, alphahat, muhat,
        betahat, gammahat, x0hat, Px0hat, IC, SE, Pvals

        Notes
        -----
        - The MATLAB GUI plotting block (subplot panels of LL / Q / R) is
          intentionally skipped — non-portable side-effect not required
          for numerical parity.
        - The optional Ikeda-acceleration branch is engaged when
          ``PPLFP_EM_Constraints['EnableIkeda']`` is truthy.
        - ``MstepMethod`` and ``delta`` are forwarded to :meth:`PPLFP_MStep`
          (``delta`` sets the GLM M-step's time base, MATLAB R4c).
        - The MATLAB convergence test uses elementwise sqrt(Q)/sqrt(R)
          which assumes the scaled (whitened) system; we preserve that.
        """
        # ---- Defaults (mirror MATLAB ``nargin<...`` cascade) ----------
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

        if PPLFP_EM_Constraints is None:
            PPLFP_EM_Constraints = PPLFP.PPLFP_EMCreateConstraints()

        if Px0 is None or _is_empty_value(Px0):
            # MATLAB: 10e-10 == 1e-9
            Px0 = 1e-9 * np.eye(numStates)
        else:
            Px0 = np.asarray(Px0, dtype=float).reshape(numStates, numStates)

        if x0 is None or _is_empty_value(x0):
            x0 = np.zeros(numStates, dtype=float)
        else:
            x0 = np.asarray(x0, dtype=float).reshape(-1)

        dN_arr = np.asarray(dN, dtype=float)
        if dN_arr.ndim == 1:
            dN_arr = dN_arr.reshape(1, -1)
        K_cells, N = dN_arr.shape

        # MATLAB (PPLFP.m PPLFP_EM, repaired B9; FIX #98): an empty or
        # scalar-zero gamma with no windowTimes means no history; otherwise one
        # default window per history coefficient, windowTimes =
        # 0:deltaW:size(gamma,1)*deltaW (deltaW = delta, 0.001 if empty), built
        # element-exactly; a nonzero shared gamma column is expanded to every
        # cell.  See decoding_algorithms._em_history_windows.  (The former rule
        # 0:delta:(length(gamma)+1)*delta gave length(gamma)+1 windows.)
        from nstat.decoding_algorithms import _em_history_windows

        gamma_arr, windowTimes = _em_history_windows(gamma, windowTimes, delta, K_cells)

        if delta is None:
            delta = 0.001

        # ---- Build history covariate HkAll (N, p_hist, K_cells) -------
        if windowTimes is not None and not _is_empty_value(windowTimes):
            from nstat.decoding_algorithms import _compute_history_terms

            HkAll = _compute_history_terms(
                dN_arr,
                delta,
                np.asarray(windowTimes, dtype=float).reshape(-1),
            )
        else:
            HkAll = np.zeros((N, 1, K_cells), dtype=float)
            gamma_arr = np.array(0.0)

        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(1, -1)
        yOrig = y_arr.copy()  # the original-scale y, for the SE call (MATLAB F8)

        # ---- EM tolerance settings ------------------------------------
        # MATLAB references ``nstat.Defaults.EM_TolAbs`` / ``EM_LogLTol``;
        # this Python port uses the same numerical constants as the
        # mPPCO_EM port for parity.
        tolAbs = 1e-3
        llTol = 1e-3
        maxIter = 100
        numToKeep = 10

        # ---- Circular history buffers (parity with MATLAB cells) ------
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
        ll_list: list[float] = []

        # ---- Scaled-system whitening (MATLAB ``scaledSystem==1``) -----
        A0 = Ahat0.copy()
        Q0 = Qhat0.copy()
        C0 = Chat0.copy()
        R0 = Rhat0.copy()

        scaledSystem = True
        if scaledSystem:
            # Tq = inv(chol(Q0, 'lower')), Tr = inv(chol(R0, 'lower')) -- NumPy's
            # cholesky is the lower factor -- so Tq*Q0*Tq' = I and
            # Tr*R0*Tr' = I (MATLAB G1, repaired after fix/pp-em 8dbd0e4).
            # With the former upper factor R (R'R = Q0) the transform did not
            # whiten a non-diagonal Q0 / R0, the diagonal / isotropic Q and R
            # constraints acted on a mixed parameterisation, and with the
            # default constraints EM returned the initial parameters.
            # Identical for a diagonal Q0 / R0.
            chol_Q0 = np.linalg.cholesky(Q0)
            chol_R0 = np.linalg.cholesky(R0)
            Tq = np.linalg.solve(chol_Q0, np.eye(numStates))
            Tr = np.linalg.solve(chol_R0, np.eye(R0.shape[0]))

            # MATLAB: Ahat{1} = Tq*Ahat{1}/Tq;
            Ahat_buf[0] = Tq @ A0 @ np.linalg.inv(Tq)
            Chat_buf[0] = Tr @ C0 @ np.linalg.inv(Tq)
            Qhat_buf[0] = Tq @ Q0 @ Tq.T
            Rhat_buf[0] = Tr @ R0 @ Tr.T
            y_arr = Tr @ y_arr
            x0hat_buf[0] = Tq @ x0
            Px0hat_buf[0] = Tq @ Px0 @ Tq.T
            alphahat_buf[0] = Tr @ alphahat0
            # MATLAB: betahat{1}=(betahat{1}'/Tq)'  ==> solve(Tq.T, beta)
            betahat_buf[0] = np.linalg.solve(Tq.T, beta)
        else:
            Ahat_buf[0] = A0
            Qhat_buf[0] = Q0
            Chat_buf[0] = C0
            Rhat_buf[0] = R0
            x0hat_buf[0] = x0
            Px0hat_buf[0] = Px0
            alphahat_buf[0] = alphahat0
            betahat_buf[0] = beta
        muhat_buf[0] = mu.copy()
        gammahat_buf[0] = gamma_arr.copy()

        IkedaAcc = bool(PPLFP_EM_Constraints.get("EnableIkeda", 0))

        cnt = 0
        stoppingCriteria = False
        dLikelihood: list[float] = [np.inf]

        _logger.info(" Joint Point-Process/Gaussian Observation EM Algorithm ")

        # ---- Main EM loop ---------------------------------------------
        while not stoppingCriteria and cnt < maxIter:
            si = cnt % numToKeep
            si_p1 = (cnt + 1) % numToKeep
            si_m1 = (cnt - 1) % numToKeep

            _logger.info("-" * 104)
            _logger.info("Iteration #%s", cnt + 1)
            _logger.info("-" * 104)

            # E-step
            x_K_buf[si], W_K_buf[si], ll_val, ExpSums_buf[si] = PPLFP.PPLFP_EStep(
                Ahat_buf[si],
                Qhat_buf[si],
                Chat_buf[si],
                Rhat_buf[si],
                y_arr,
                alphahat_buf[si],
                dN_arr,
                muhat_buf[si],
                betahat_buf[si],
                fitType,
                delta,
                gammahat_buf[si],
                HkAll,
                x0hat_buf[si],
                Px0hat_buf[si],
            )
            ll_list.append(float(ll_val))

            # MATLAB (repaired, B5): stop before the M-step when the E-step
            # log-likelihood is not a finite real number (e.g. Px0hat
            # collapsing under EstimatePx0=1); the best finite iterate is
            # selected below.  NaN comparisons are False, so the likelihood
            # stopping rule alone kept iterating on NaN.
            if not np.isfinite(ll_list[-1]):
                _logger.info(
                    " EM stopped at iteration# %s b/c the E-step log-likelihood was not a finite "
                    "real number (%s)", cnt + 1, ll_list[-1],
                )
                break

            # M-step
            (
                Ahat_buf[si_p1],
                Qhat_buf[si_p1],
                Chat_buf[si_p1],
                Rhat_buf[si_p1],
                alphahat_buf[si_p1],
                muhat_buf[si_p1],
                betahat_buf[si_p1],
                gammahat_buf[si_p1],
                x0hat_buf[si_p1],
                Px0hat_buf[si_p1],
            ) = PPLFP.PPLFP_MStep(
                dN_arr,
                y_arr,
                x_K_buf[si],
                W_K_buf[si],
                x0hat_buf[si],
                Px0hat_buf[si],
                ExpSums_buf[si],
                fitType,
                muhat_buf[si],
                betahat_buf[si],
                gammahat_buf[si],
                windowTimes,
                HkAll,
                PPLFP_EM_Constraints,
                MstepMethod,
                delta,
            )

            # ---- Optional Ikeda acceleration --------------------------
            if IkedaAcc:
                _logger.info("****Ikeda Acceleration Step****")
                rng = np.random.default_rng()
                mean_y = (
                    Chat_buf[si_p1] @ x_K_buf[si]
                    + alphahat_buf[si_p1].reshape(-1, 1)
                    @ np.ones((1, x_K_buf[si].shape[1]))
                )
                R_for_draw = Rhat_buf[si_p1]
                ykNew = mean_y + rng.multivariate_normal(
                    np.zeros(R_for_draw.shape[0]),
                    R_for_draw,
                    size=mean_y.shape[1],
                ).T

                dNNew = dN_arr

                (x_KNew, W_KNew, _llNew, ExpectationSumsNew) = PPLFP.PPLFP_EStep(
                    Ahat_buf[si],
                    Qhat_buf[si],
                    Chat_buf[si],
                    Rhat_buf[si],
                    ykNew,
                    alphahat_buf[si],
                    dNNew,
                    muhat_buf[si],
                    betahat_buf[si],
                    fitType,
                    delta,
                    gammahat_buf[si],
                    HkAll,
                    x0,
                    Px0,
                )

                (
                    AhatNew,
                    QhatNew,
                    ChatNew,
                    RhatNew,
                    alphahatNew,
                    _muhatNew,
                    _betahatNew,
                    _gammahatNew,
                    _x0new,
                    _Px0new,
                ) = PPLFP.PPLFP_MStep(
                    dNNew,
                    ykNew,
                    x_KNew,
                    W_KNew,
                    x0hat_buf[si],
                    Px0hat_buf[si],
                    ExpectationSumsNew,
                    fitType,
                    muhat_buf[si],
                    betahat_buf[si],
                    gammahat_buf[si],
                    windowTimes,
                    HkAll,
                    PPLFP_EM_Constraints,
                    MstepMethod,
                    delta,
                )

                Ahat_buf[si_p1] = 2 * Ahat_buf[si_p1] - AhatNew
                Qhat_buf[si_p1] = 2 * Qhat_buf[si_p1] - QhatNew
                Qhat_buf[si_p1] = (Qhat_buf[si_p1] + Qhat_buf[si_p1].T) / 2
                Chat_buf[si_p1] = 2 * Chat_buf[si_p1] - ChatNew
                Rhat_buf[si_p1] = 2 * Rhat_buf[si_p1] - RhatNew
                Rhat_buf[si_p1] = (Rhat_buf[si_p1] + Rhat_buf[si_p1].T) / 2
                alphahat_buf[si_p1] = 2 * alphahat_buf[si_p1] - alphahatNew

            # Honor EstimateA == 0 -> freeze A
            if not PPLFP_EM_Constraints.get("EstimateA", 1):
                Ahat_buf[si_p1] = Ahat_buf[si].copy()

            # ---- Likelihood delta -------------------------------------
            if cnt == 0:
                dLikelihood.append(np.inf)
            else:
                dLikelihood.append(ll_list[-1] - ll_list[-2])

            # ---- Parameter-change convergence test --------------------
            if cnt == 0:
                dMax = np.inf
            else:
                diffs: list[float] = []
                # MATLAB max(max(abs(sqrt(Qhat{k}) - sqrt(Qhat{k-1})))) (and R):
                # the elementwise sqrt is complex for a negative (off-diagonal)
                # entry.  This port took sqrt(|.|) (different when an entry
                # changes sign) inside a try / except that could not fire.
                for M_cur, M_prev in ((Qhat_buf[si], Qhat_buf[si_m1]), (Rhat_buf[si], Rhat_buf[si_m1])):
                    diffs.append(float(np.max(np.abs(
                        np.sqrt(np.asarray(M_cur, dtype=complex)) - np.sqrt(np.asarray(M_prev, dtype=complex))))))
                diffs.append(float(np.max(np.abs(Ahat_buf[si] - Ahat_buf[si_m1]))))
                diffs.append(float(np.max(np.abs(Chat_buf[si] - Chat_buf[si_m1]))))
                diffs.append(
                    float(
                        np.max(
                            np.abs(
                                np.asarray(muhat_buf[si])
                                - np.asarray(muhat_buf[si_m1])
                            )
                        )
                    )
                )
                diffs.append(
                    float(
                        np.max(
                            np.abs(
                                np.asarray(alphahat_buf[si])
                                - np.asarray(alphahat_buf[si_m1])
                            )
                        )
                    )
                )
                diffs.append(
                    float(
                        np.max(
                            np.abs(
                                np.asarray(betahat_buf[si])
                                - np.asarray(betahat_buf[si_m1])
                            )
                        )
                    )
                )
                diffs.append(
                    float(
                        np.max(
                            np.abs(
                                np.asarray(gammahat_buf[si])
                                - np.asarray(gammahat_buf[si_m1])
                            )
                        )
                    )
                )
                dMax = max(diffs)

            if cnt == 0:
                _logger.info("Max Parameter Change: N/A")
            else:
                _logger.info("Max Parameter Change: %s", dMax)

            cnt += 1

            if dMax < tolAbs:
                stoppingCriteria = True
                _logger.info(
                    "         EM converged at iteration# %s b/c change in params was within criteria",
                    cnt,
                )

            if abs(dLikelihood[-1]) < llTol or dLikelihood[-1] < 0:
                stoppingCriteria = True
                _logger.info(
                    "         EM stopped at iteration# %s b/c change in likelihood was negative",
                    cnt,
                )

        _logger.info("-" * 104)

        # ---- Select best iteration ------------------------------------
        ll_arr = np.asarray(ll_list, dtype=float)
        if ll_arr.size == 0:
            raise RuntimeError("PPLFP_EM: no EM iterations executed")
        # Best FINITE iterate, first max (MATLAB, repaired: non-finite logll ->
        # NaN, find(llSel == max(llSel), 1, 'first'); none finite -> 1).
        finite = np.isfinite(ll_arr)
        maxLLIndex = int(np.argmax(np.where(finite, ll_arr, -np.inf))) if np.any(finite) else 0
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

        # ---- Reverse the scaling (MATLAB final ``scaledSystem==1``) ---
        if scaledSystem:
            # The same Tq = inv(chol(Q0, 'lower')), Tr = inv(chol(R0, 'lower'))
            # as the forward scaling (MATLAB G1).
            chol_Q0 = np.linalg.cholesky(Q0)
            chol_R0 = np.linalg.cholesky(R0)
            Tq = np.linalg.solve(chol_Q0, np.eye(numStates))
            Tr = np.linalg.solve(chol_R0, np.eye(R0.shape[0]))

            # Tq \ X  -> np.linalg.solve(Tq, X)
            Ahat_out = np.linalg.solve(Tq, Ahat_out) @ Tq
            # Covariances: MATLAB (T\S)/T' = T^-1 S T^-T.  This used
            # solve(T.T, (T\S).T).T = T^-1 S T^-1, which is the same only for
            # a symmetric T (diagonal Q0 / R0); with a non-diagonal Q0 or R0
            # it returned non-symmetric Qhat, Rhat, Px0hat and WKFinal.
            Qhat_out = _unwhiten(Tq, Qhat_out)
            Chat_out = np.linalg.solve(Tr, Chat_out) @ Tq
            Rhat_out = _unwhiten(Tr, Rhat_out)
            alphahat_out = np.linalg.solve(Tr, alphahat_out)
            xKFinal = np.linalg.solve(Tq, xKFinal)
            x0hat_out = np.linalg.solve(Tq, x0hat_out)
            Px0hat_out = _unwhiten(Tq, Px0hat_out)

            tempWK = np.zeros_like(WKFinal)
            for kk in range(WKFinal.shape[2]):
                tempWK[:, :, kk] = _unwhiten(Tq, WKFinal[:, :, kk])
            WKFinal = tempWK

            # MATLAB: betahat=(betahat'*Tq)'  -> betahat = Tq.T @ betahat
            betahat_out = (betahat_out.T @ Tq).T

        ll_best = float(ll_arr[maxLLIndex])

        # ---- Standard errors ------------------------------------------
        # MATLAB computes them (when SE/Pvals are requested) without a guard.
        # This used to unpack the routine's three outputs into two inside an
        # ``except Exception: pass`` block, so SE = Pvals = {} on every call.
        # MATLAB F8 (fix/pp-em bac99f9): EM runs on the SCALED system (x_s =
        # Tq x, y_s = Tr y) and every estimate passed here is back on the
        # original scale, but the SE call used to receive the scaled y and
        # the scaled expectation sums.  The SE routine reads y (alpha, C and R
        # scores), ES.Sxkm1xkm1 (A information) and ES.Sxkxk (C information),
        # so SE.A / SE.C / SE.R mixed scales whenever Q0 or R0 != I.  Pass the
        # original y and map those two sums back: S = (Tq\S_s)/Tq'.  (A copy:
        # the information criteria below read the buffer's own sums.)
        ySE = y_arr
        ESforSE = dict(ExpectationSumsFinal)
        if scaledSystem:
            ySE = yOrig
            ESforSE["Sxkm1xkm1"] = _unwhiten(Tq, np.asarray(ExpectationSumsFinal["Sxkm1xkm1"], dtype=float))
            ESforSE["Sxkxk"] = _unwhiten(Tq, np.asarray(ExpectationSumsFinal["Sxkxk"], dtype=float))
        SE, Pvals, _ = PPLFP.PPLFP_ComputeParamStandardErrors(
            ySE,
            dN_arr,
            xKFinal,
            WKFinal,
            Ahat_out,
            Qhat_out,
            Chat_out,
            Rhat_out,
            alphahat_out,
            x0hat_out,
            Px0hat_out,
            ESforSE,
            fitType,
            muhat_out,
            betahat_out,
            gammahat_out,
            windowTimes,
            HkAll,
            PPLFP_EM_Constraints,
        )

        # ---- Information criteria (parity with MATLAB) ---------------
        if PPLFP_EM_Constraints.get("EstimateA", 0) and PPLFP_EM_Constraints.get(
            "AhatDiag", 0
        ):
            n1 = Ahat_out.shape[0]
        elif PPLFP_EM_Constraints.get("EstimateA", 0):
            n1 = Ahat_out.size
        else:
            n1 = 0

        if PPLFP_EM_Constraints.get("QhatDiag", 0) and PPLFP_EM_Constraints.get(
            "QhatIsotropic", 0
        ):
            n2 = 1
        elif PPLFP_EM_Constraints.get("QhatDiag", 0):
            n2 = Qhat_out.shape[0]
        else:
            n2 = Qhat_out.size

        n3 = Chat_out.size

        # MATLAB F11 (fix/pp-em 564c207): the diagonal-R branch tests R's own
        # flags.  It tested QhatDiag / QhatIsotropic (a transcription slip this
        # port mirrored), so RhatDiag = 0 with QhatDiag = 1 counted dy instead
        # of numel(Rhat) and RhatDiag = 1 with QhatDiag = 0 counted numel(Rhat)
        # instead of dy.  Only IC used the count; the default flags (both
        # diagonal, not isotropic) were unaffected.
        if PPLFP_EM_Constraints.get("RhatDiag", 0) and PPLFP_EM_Constraints.get(
            "RhatIsotropic", 0
        ):
            n4 = 1
        elif PPLFP_EM_Constraints.get("RhatDiag", 0) and not PPLFP_EM_Constraints.get(
            "RhatIsotropic", 0
        ):
            n4 = Rhat_out.shape[0]
        else:
            n4 = Rhat_out.size

        if PPLFP_EM_Constraints.get("EstimatePx0", 0) and PPLFP_EM_Constraints.get(
            "Px0Isotropic", 0
        ):
            n5 = 1
        elif PPLFP_EM_Constraints.get("EstimatePx0", 0):
            n5 = Px0hat_out.shape[0]
        else:
            n5 = 0

        if PPLFP_EM_Constraints.get("Estimatex0", 0):
            n6 = np.asarray(x0hat_out).size
        else:
            n6 = 0

        n7 = np.asarray(alphahat_out).size
        n8 = np.asarray(muhat_out).size
        n9 = np.asarray(betahat_out).size
        ghat = np.asarray(gammahat_out)
        if ghat.size == 1:
            n10 = 0 if float(ghat.flat[0]) == 0 else 1
        else:
            n10 = ghat.size

        nTerms = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10

        K_T = y_arr.shape[1]
        Dx = Ahat_out.shape[1]
        sumXkTerms = np.asarray(ExpectationSumsFinal["sumXkTerms"], dtype=float)
        # MATLAB F10 (fix/pp-em 8843a94): ll_best (the best E-step's expected
        # complete-data log-likelihood) and sumXkTerms belong to the SCALED
        # system (x_s = Tq x, y_s = Tr y), while Qhat and Px0hat are back on
        # the original scale, so llobs mixed scales and AIC / BIC depended on
        # the units of x and of y.  Map both to the original scale:
        # S = (Tq\S_s)/Tq'; each of the K+1 state log-densities gains
        # log|det Tq| and each of the K observation log-densities log|det Tr|.
        # llobs is then E[log p(dN, y | x)] on y's original scale, and
        # IC.llcomp the expected complete-data log-likelihood PPLFP_EStep
        # returns in the original coordinates at the returned estimates.
        llcomp = ll_best
        if scaledSystem:
            sumXkTerms = _unwhiten(Tq, sumXkTerms)
            llcomp = (ll_best + (K_T + 1) * np.log(np.abs(np.linalg.det(Tq)))
                      + K_T * np.log(np.abs(np.linalg.det(Tr))))
        # MATLAB log(det(Qhat)), log(det(Px0hat)), 2n(n+1)/(K-n-1) and
        # n*log(K): no floors or guards (this port floored both determinants
        # at 1e-300, set AICc = Inf for K - n - 1 == 0 and used log(max(K, 1))).
        detQ = np.linalg.det(Qhat_out)
        detPx0 = np.linalg.det(Px0hat_out)
        llobs = (
            llcomp
            + Dx * K_T / 2.0 * np.log(2 * np.pi)
            + K_T / 2.0 * np.log(detQ)
            + 0.5 * np.trace(np.linalg.solve(Qhat_out, sumXkTerms))
            + Dx / 2.0 * np.log(2 * np.pi)
            + 0.5 * np.log(detPx0)
            + 0.5 * Dx
        )
        AIC = 2 * nTerms - 2 * llobs
        AICc = AIC + np.divide(2.0 * nTerms * (nTerms + 1), np.float64(K_T - nTerms - 1))
        BIC = -2 * llobs + nTerms * np.log(K_T)

        IC = {
            "AIC": AIC,
            "AICc": AICc,
            "BIC": BIC,
            "llobs": llobs,
            "llcomp": llcomp,
        }

        return (
            xKFinal,
            WKFinal,
            Ahat_out,
            Qhat_out,
            Chat_out,
            Rhat_out,
            alphahat_out,
            muhat_out,
            betahat_out,
            gammahat_out,
            x0hat_out,
            Px0hat_out,
            IC,
            SE,
            Pvals,
        )

    @staticmethod
    @_matlab_float_semantics
    def PPLFP_EStep(A, Q, C, R, y, alpha, dN, mu, beta, fitType='poisson',
                     delta=0.001, gamma=None, HkAll=None, x0=None, Px0=None):
        """E-step for the PPLFP EM algorithm.

        MATLAB cross-reference: ``PPLFP.PPLFP_EStep``
        (``+nstat/+decoding/PPLFP.m`` lines 1990-2206).

        Runs the PPLFP forward filter + RTS smoother on the trial, then
        accumulates the sufficient statistics required by the M-step
        (``Sxkm1xkm1``, ``Sxkm1xk``, ``Sxkxk``, ``Sykyk``, ``Sxkyk``),
        the linearised Gaussian terms (``sumXkTerms``, ``sumYkTerms``),
        the conditional-intensity contribution to the complete-data
        log-likelihood (``sumPPll``), and the best estimates of the
        initial state (``Sx0``, ``Sx0x0``).

        Parameters
        ----------
        A, Q : (Dx, Dx) ndarray
            Linear state transition matrix and process-noise covariance.
        C, R : (Dy, Dx) and (Dy, Dy) ndarray
            Continuous observation matrix and noise covariance.
        y : (Dy, K) ndarray
            Continuous observations (LFP signal).
        alpha : (Dy,) ndarray
            Continuous-observation offset.
        dN : (numCells, K) ndarray
            Binned spike counts.
        mu : (numCells,) ndarray
            CIF baseline log-rates.
        beta : (Dx, numCells) ndarray
            CIF state-coupling matrix.
        fitType : {'poisson', 'binomial'}
            CIF link selection.
        delta : float
            Bin width (seconds).
        gamma : ndarray or scalar
            CIF history-effect coefficients.
        HkAll : ndarray
            Per-cell history-basis design (shape ``(K, Nh, numCells)``
            after MATLAB-style permute).
        x0, Px0 : ndarray
            Prior mean / covariance on the initial state.

        Returns
        -------
        x_K : (Dx, K) ndarray -- smoothed state mean.
        W_K : (Dx, Dx, K) ndarray -- smoothed state covariance.
        logll : float -- complete-data log-likelihood lower bound.
        ExpectationSums : dict -- sufficient statistics for the M-step.
        """
        # Lazy imports to avoid circular references at module-import time.
        from nstat.decoding_algorithms import (
            DecodingAlgorithms,
            _as_observation_matrix,
            _is_empty_value,
            _symmetrize,
        )

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

        A_2d = A.reshape(Dx, Dx) if A.ndim != 2 else A
        Q_2d = Q.reshape(Dx, Dx) if Q.ndim != 2 else Q
        R_2d = R.reshape(Dy, Dy) if R.ndim != 2 else R

        # Forward filter (PPLFP filter pass).
        x_p, W_p, x_u, W_u = PPLFP.PPLFP_DecodeLinear(
            A, Q, C, R, y, alpha, dN, mu_vec, beta_mat, fitType,
            delta, gamma, None, x0, Px0, HkAll_arr)

        x_p = np.asarray(x_p, dtype=float)
        W_p = np.asarray(W_p, dtype=float)
        x_u = np.asarray(x_u, dtype=float)
        W_u = np.asarray(W_u, dtype=float)

        # Smoother (RTS): PPLFP_DecodeLinear returns predictions of length K+1
        # (one extra forward step) and updates of length K, both in MATLAB
        # column-major layout: x_* is (Dx, T), W_* is (Dx, Dx, T).  Convert to
        # the smoother's time-major (T, Dx) and (T, Dx, Dx) before calling and
        # convert back after.
        N = K
        x_p_in = x_p[:, :N] if x_p.shape[1] >= N else x_p
        W_p_in = W_p[:, :, :N] if W_p.shape[2] >= N else W_p
        x_u_in = x_u[:, :N] if x_u.shape[1] >= N else x_u
        W_u_in = W_u[:, :, :N] if W_u.shape[2] >= N else W_u

        x_p_tm = x_p_in.T  # (N, Dx)
        x_u_tm = x_u_in.T
        W_p_tm = np.transpose(W_p_in, (2, 0, 1))  # (N, Dx, Dx)
        W_u_tm = np.transpose(W_u_in, (2, 0, 1))

        x_K_tm, W_K_tm, _Lk = DecodingAlgorithms.kalman_smootherFromFiltered(
            A_2d, x_p_tm, W_p_tm, x_u_tm, W_u_tm,
        )
        x_K_tm = np.asarray(x_K_tm, dtype=float)
        W_K_tm = np.asarray(W_K_tm, dtype=float)

        # Convert back to MATLAB column-major (Dx, K) / (Dx, Dx, K).
        if x_K_tm.ndim == 2 and x_K_tm.shape == (K, Dx):
            x_K = x_K_tm.T
        elif x_K_tm.ndim == 2 and x_K_tm.shape == (Dx, K):
            x_K = x_K_tm
        else:
            x_K = x_K_tm
        if W_K_tm.ndim == 3 and W_K_tm.shape[0] == K:
            W_K = np.transpose(W_K_tm, (1, 2, 0))
        else:
            W_K = W_K_tm

        # Best estimates of the initial state given the data.
        # MATLAB: W1G0 = A*Px0*A' + Q; L0 = Px0*A' / W1G0.
        # MATLAB B/W1G0 = B * inv(W1G0); use solve via transposes for stability.
        W1G0 = A_2d @ Px0 @ A_2d.T + Q_2d
        # L0 = (Px0 @ A.T) / W1G0  ->  (W1G0.T) L0.T = (Px0 A.T).T
        try:
            L0 = np.linalg.solve(W1G0.T, (Px0 @ A_2d.T).T).T
        except np.linalg.LinAlgError:
            L0 = (Px0 @ A_2d.T) @ np.linalg.pinv(W1G0)
        Ex0Gy = x0 + L0 @ (x_K[:, 0] - x_p[:, 0])
        # Px0Gy = Px0 + L0 * (inv(W_K[:,:,0]) - inv(W1G0)) * L0.T
        try:
            inv_WK0 = np.linalg.solve(W_K[:, :, 0], np.eye(Dx))
        except np.linalg.LinAlgError:
            inv_WK0 = np.linalg.pinv(W_K[:, :, 0])
        try:
            inv_W1G0 = np.linalg.solve(W1G0, np.eye(Dx))
        except np.linalg.LinAlgError:
            inv_W1G0 = np.linalg.pinv(W1G0)
        Px0Gy = Px0 + L0 @ (inv_WK0 - inv_W1G0) @ L0.T
        Px0Gy = _symmetrize(Px0Gy)

        # Cross-covariance Wku (de Jong & MacKinnon 1988): Tk = A.
        # MATLAB inner loop is a single iteration (k = u-1); we mirror that.
        numStates = Dx
        Wku = np.zeros((numStates, numStates, K, K), dtype=float)
        for k in range(K):
            Wku[:, :, k, k] = W_K[:, :, k]
        for u in range(K - 1, 0, -1):
            k = u - 1
            # Dk = W_u[:,:,k] * A.T / W_p[:,:,k+1]
            try:
                Dk = np.linalg.solve(
                    W_p[:, :, k + 1].T, (W_u[:, :, k] @ A_2d.T).T
                ).T
            except np.linalg.LinAlgError:
                Dk = W_u[:, :, k] @ A_2d.T @ np.linalg.pinv(W_p[:, :, k + 1])
            Wku[:, :, k, u] = Dk @ Wku[:, :, k + 1, u]
            Wku[:, :, u, k] = Wku[:, :, k, u].T

        # Sufficient statistics.
        Sxkm1xk = np.zeros((Dx, Dx))
        Sxkm1xkm1 = np.zeros((Dx, Dx))
        Sxkxk = np.zeros((Dx, Dx))
        Sykyk = np.zeros((Dy, Dy))
        Sxkyk = np.zeros((Dx, Dy))

        alpha_vec = alpha.reshape(-1)
        for k in range(K):
            if k == 0:
                # Px0*A' / W_p(:,:,1) * Wku(:,:,1,1)
                try:
                    PxA_div_Wp0 = np.linalg.solve(
                        W_p[:, :, 0].T, (Px0 @ A_2d.T).T
                    ).T
                except np.linalg.LinAlgError:
                    PxA_div_Wp0 = (Px0 @ A_2d.T) @ np.linalg.pinv(W_p[:, :, 0])
                Sxkm1xk += PxA_div_Wp0 @ Wku[:, :, 0, 0]
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

        # Point-process log-likelihood contribution (vectorised over cells).
        if str(fitType) == 'poisson':
            sumPPll = 0.0
            HkPerm = (np.transpose(HkAll_arr, (1, 2, 0))
                      if HkAll_arr.ndim == 3 and HkAll_arr.shape[0] == K
                      else HkAll_arr)
            for k in range(K):
                Hk = HkPerm[:, :, k] if HkPerm.ndim == 3 else np.zeros((1, numCells))
                if Hk.shape[0] == numCells and Hk.shape[1] != numCells:
                    Hk = Hk.T
                xk = x_K[:, k]
                gammaC_mat = (np.tile(gamma_arr.reshape(-1, 1), (1, numCells))
                              if gamma_arr.size == 1 else gamma_arr)
                if gammaC_mat.ndim == 2 and gammaC_mat.shape[1] != numCells:
                    gammaC_mat = np.tile(gammaC_mat, (1, numCells))
                if Hk.size > 0 and gammaC_mat.size > 0:
                    terms = mu_vec + beta_mat.T @ xk + np.diag(gammaC_mat.T @ Hk)
                else:
                    terms = mu_vec + beta_mat.T @ xk
                Wk = W_K[:, :, k]
                # MATLAB: ld = exp(terms); no clip.  Preserve overflow behaviour.
                with np.errstate(over='ignore'):
                    ld = np.exp(terms)
                bt = beta_mat
                ExplambdaDelta = ld + 0.5 * (ld * np.diag(bt.T @ Wk @ bt))
                ExplogLD = terms
                sumPPll += float(np.sum(obs[:, k] * ExplogLD - ExplambdaDelta))
        else:  # binomial link
            sumPPll = 0.0
            HkPerm = (np.transpose(HkAll_arr, (1, 2, 0))
                      if HkAll_arr.ndim == 3 and HkAll_arr.shape[0] == K
                      else HkAll_arr)
            for k in range(K):
                Hk = HkPerm[:, :, k] if HkPerm.ndim == 3 else np.zeros((1, numCells))
                if Hk.shape[0] == numCells and Hk.shape[1] != numCells:
                    Hk = Hk.T
                xk = x_K[:, k]
                gammaC_mat = (np.tile(gamma_arr.reshape(-1, 1), (1, numCells))
                              if gamma_arr.size == 1 else gamma_arr)
                if gammaC_mat.ndim == 2 and gammaC_mat.shape[1] != numCells:
                    gammaC_mat = np.tile(gammaC_mat, (1, numCells))
                if Hk.size > 0 and gammaC_mat.size > 0:
                    terms = mu_vec + beta_mat.T @ xk + np.diag(gammaC_mat.T @ Hk)
                else:
                    terms = mu_vec + beta_mat.T @ xk
                Wk = W_K[:, :, k]
                # MATLAB: ld = exp(terms)./(1+exp(terms)).
                with np.errstate(over='ignore'):
                    ld_raw = np.exp(terms)
                ld = ld_raw / (1.0 + ld_raw)
                bt = beta_mat
                ExplambdaDelta = ld + 0.5 * (ld * (1 - ld) * (1 - 2 * ld)) * np.diag(bt.T @ Wk @ bt)
                with np.errstate(divide='ignore'):
                    ExplogLD = (np.log(ld)
                                + 0.5 * (-ld * (1 - ld)) * np.diag(bt.T @ Wk @ bt))
                sumPPll += float(np.sum(obs[:, k] * ExplogLD - ExplambdaDelta))

        # Complete-data log-likelihood lower bound (MATLAB lines 2169-2173).
        # MATLAB log(det(.)), no floor: a non-positive determinant gives -Inf
        # or NaN (MATLAB: complex), and PPLFP_EM stops before the M-step.  This
        # port replaced it by log(max(|det|, 1e-300)), which kept a collapsed
        # Px0 iterating where MATLAB stops.
        def _safe_logdet(M):
            return float(np.log(np.linalg.det(M)))

        logll = (-Dx * K / 2.0 * np.log(2 * np.pi)
                 - K / 2.0 * _safe_logdet(Q_2d)
                 - Dy * K / 2.0 * np.log(2 * np.pi)
                 - K / 2.0 * _safe_logdet(R_2d)
                 - Dx / 2.0 * np.log(2 * np.pi)
                 - 0.5 * _safe_logdet(Px0)
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
    @_matlab_float_semantics
    def PPLFP_MStep(
        dN,
        y,
        x_K,
        W_K,
        x0,
        Px0,
        ExpectationSums,
        fitType,
        muhat,
        betahat,
        gammahat,
        windowTimes,
        HkAll,
        PPLFP_EM_Constraints=None,
        MstepMethod=None,
        delta=0.001,
    ):
        """PPLFP EM maximisation step.

        Python port of MATLAB ``PPLFP_MStep`` (repaired, fix/pp-em @ 8dbd0e4).
        Closed-form updates for state-space params ``(A, Q, C, R, alpha,
        x0, Px0)`` from sufficient statistics in ``ExpectationSums``, then
        CIF param updates ``(mu, beta, gamma)`` via either a GLM fit on the
        smoothed means (``MstepMethod='GLM'``, see
        :func:`nstat.decoding_algorithms._em_glm_mstep`) or Monte-Carlo
        Newton-Raphson (``MstepMethod='NewtonRaphson'``, the default since
        the repaired MATLAB, B8; it was 'GLM' -- see PPLFP_EM).  Any other
        ``MstepMethod`` raises ``ValueError`` (MATLAB runs Newton-Raphson for
        it).  ``delta`` (seconds per bin, MATLAB's optional 16th input,
        default 0.001, R4c) is the GLM M-step's time base; ``PPLFP_EM`` passes
        its ``delta``.
        """
        MstepMethod = _check_mstep_method(MstepMethod)
        if delta is None:
            delta = 0.001
        if PPLFP_EM_Constraints is None:
            PPLFP_EM_Constraints = PPLFP.PPLFP_EMCreateConstraints()

        def _get(name):
            if isinstance(ExpectationSums, dict):
                return ExpectationSums[name]
            return getattr(ExpectationSums, name)

        Sxkm1xkm1 = np.asarray(_get("Sxkm1xkm1"), dtype=float)
        Sxkxkm1 = np.asarray(_get("Sxkxkm1"), dtype=float)
        Sxkxk = np.asarray(_get("Sxkxk"), dtype=float)
        Sxkyk = np.asarray(_get("Sxkyk"), dtype=float)
        sumXkTerms = np.asarray(_get("sumXkTerms"), dtype=float)
        sumYkTerms = np.asarray(_get("sumYkTerms"), dtype=float)

        x_K = np.asarray(x_K, dtype=float)
        if x_K.ndim == 1:
            x_K = x_K.reshape(1, -1)
        dx, K = x_K.shape

        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(1, -1)

        dN = np.asarray(dN)
        if dN.ndim == 1:
            dN = dN.reshape(1, -1)
        numCells = dN.shape[0]

        x0 = np.asarray(x0, dtype=float).reshape(dx, 1)
        Px0 = np.asarray(Px0, dtype=float)

        muhat = np.asarray(muhat, dtype=float).reshape(-1)
        betahat = np.asarray(betahat, dtype=float)
        if betahat.ndim == 1:
            betahat = betahat.reshape(-1, 1)
        gammahat_arr = np.asarray(gammahat, dtype=float)

        def _gc(name, default=0):
            if isinstance(PPLFP_EM_Constraints, dict):
                return PPLFP_EM_Constraints.get(name, default)
            return getattr(PPLFP_EM_Constraints, name, default)

        # ---- Ahat: MATLAB Sxkxkm1 / Sxkm1xkm1 -------------------------
        # MATLAB mrdivide: Ahat = (Sxkxkm1.*I)/(Sxkm1xkm1.*I) or
        # Sxkxkm1/Sxkm1xkm1, Chat = Sxkyk'/Sxkxk (X/B = solve(B', X')').  This
        # port used least squares (lstsq), whose minimum-norm answer differs
        # from MATLAB's on a singular system and at round-off otherwise.
        if _gc("AhatDiag", 0) == 1:
            I_dx = np.eye(dx)
            num = Sxkxkm1 * I_dx
            den = Sxkm1xkm1 * I_dx
            Ahat = np.linalg.solve(den.T, num.T).T
        else:
            Ahat = np.linalg.solve(Sxkm1xkm1.T, Sxkxkm1.T).T

        # ---- Chat: MATLAB Sxkyk' / Sxkxk ------------------------------
        Chat = np.linalg.solve(Sxkxk.T, Sxkyk).T

        alphahat = np.sum(y - Chat @ x_K, axis=1, keepdims=True) / K

        # ---- Qhat ----------------------------------------------------
        if _gc("QhatDiag", 0) == 1:
            if _gc("QhatIsotropic", 0) == 1:
                Qhat = (1.0 / (dx * K)) * np.trace(sumXkTerms) * np.eye(dx)
            else:
                Qhat = (1.0 / K) * (sumXkTerms * np.eye(dx))
                Qhat = (Qhat + Qhat.T) / 2.0
        else:
            Qhat = (1.0 / K) * sumXkTerms
            Qhat = (Qhat + Qhat.T) / 2.0

        # ---- Rhat ----------------------------------------------------
        dy = sumYkTerms.shape[0]
        if _gc("RhatDiag", 0) == 1:
            if _gc("RhatIsotropic", 0) == 1:
                Rhat = (1.0 / (dy * K)) * np.trace(sumYkTerms) * np.eye(dy)
            else:
                Rhat = (1.0 / K) * (sumYkTerms * np.eye(dy))
                Rhat = (Rhat + Rhat.T) / 2.0
        else:
            Rhat = (1.0 / K) * sumYkTerms
            Rhat = (Rhat + Rhat.T) / 2.0

        # ---- x0hat ---------------------------------------------------
        if _gc("Estimatex0", 0):
            Px0_inv = np.linalg.inv(Px0)
            # MATLAB A'/Q -> A' * inv(Q) (mrdivide; this port used lstsq)
            AtQinv = np.linalg.solve(Qhat.T, Ahat).T
            lhs = Px0_inv + AtQinv @ Ahat
            rhs = AtQinv @ x_K[:, 0:1] + np.linalg.solve(Px0, x0)
            x0hat = np.linalg.solve(lhs, rhs)
        else:
            x0hat = x0.copy()

        # ---- Px0hat --------------------------------------------------
        if _gc("EstimatePx0", 0) == 1:
            xh = x0hat.reshape(dx, 1)
            x0c = x0.reshape(dx, 1)
            inner = xh @ xh.T - x0c @ xh.T - xh @ x0c.T + x0c @ x0c.T
            if _gc("Px0Isotropic", 0) == 1:
                Px0hat = (np.trace(inner) / (dx * K)) * np.eye(dx)
            else:
                Px0hat = inner * np.eye(dx)
                Px0hat = (Px0hat + Px0hat.T) / 2.0
        else:
            Px0hat = Px0.copy()

        betahat_new = betahat.copy()
        gammahat_new = gammahat_arr.copy()
        muhat_new = muhat.copy()

        if fitType == "poisson":
            algorithm = "GLM"
        else:
            algorithm = "BNLRCG"

        if MstepMethod == "GLM":
            # MATLAB's GLM M-step (repaired: the getCoeffs() tuple is unpacked
            # -- it crashed here -- and coefficients are mapped BY LABEL, F3 /
            # R4a / F1, with keep-previous for unestimable ones; delta time
            # base, R4c; scoped warnings).  The former positional read took
            # neurons as labels and v1 as mu.  See _em_glm_mstep.
            muhat_new, betahat_new, gammahat_new = _em_glm_mstep(
                dN, x_K, fitType, muhat, betahat, gammahat_arr, windowTimes, delta)
            return (
                Ahat, Qhat, Chat, Rhat, alphahat,
                muhat_new, betahat_new, gammahat_new, x0hat, Px0hat,
            )

        # ============== Newton-Raphson branch =========================
        _logger.info("****M-step for beta****")
        McExp = 50
        rng = np.random.default_rng()

        def _normal(d, n):
            return rng.standard_normal((d, n))

        # Draws through _mc_state_draws (MATLAB mcStateDraws, F9: the draw
        # was chol(W)*z, covariance R*R' instead of W).
        W_K_arr = np.asarray(W_K, dtype=float)
        xKDrawExp = np.zeros((dx, K, McExp))
        for k in range(K):
            xKDrawExp[:, k, :] = _mc_state_draws(x_K[:, k], W_K_arr[:, :, k], McExp, _normal, non_pd="zero")

        # MATLAB permute(xKDrawExp,[1 3 2]) -> (dx, McExp, K)
        xkPerm = np.transpose(xKDrawExp, (0, 2, 1))
        diffTol = 1e-5

        HkAll_arr = np.asarray(HkAll, dtype=float)

        def _Hk_for_cell(c):
            # MATLAB R4d (fix/pp-em 605d9bc): no re-orientation.  HkAll(:,:,c)
            # is (numTimeSteps x numWindows) by construction; the old
            # `if size(Hk,1)==numCells, Hk = Hk'` fired when numTimeSteps ==
            # numCells and then indexed Hk(k,:) on the transpose.
            return HkAll_arr[:, :, c]

        def _gammaC(gamma_src, c):
            if np.size(gamma_src) == 1:
                return np.atleast_1d(
                    np.asarray(gamma_src).ravel()
                ).astype(float)
            return gamma_src[:, c]

        # ----- stimulus coefficients (beta) ---------------------------
        for c in range(numCells):
            converged = False
            it = 1
            maxIter = 100
            # One record per neuron, listing the Newton iterations (formerly
            # written incrementally to stdout on a single line).
            iters_done: list[int] = []
            while (not converged) and it < maxIter:
                iters_done.append(it)
                HessianTerm = np.zeros((dx, dx))
                GradTerm = np.zeros((dx, 1))
                Hk = _Hk_for_cell(c)
                gammaC = _gammaC(gammahat_arr, c)
                for k in range(K):
                    xk = xkPerm[:, :, k]
                    terms = (
                        muhat[c]
                        + betahat_new[:, c] @ xk
                        + gammaC @ Hk[k, :].reshape(-1, 1)
                    )
                    terms = np.asarray(terms).reshape(-1)
                    if fitType == "poisson":
                        ld = np.exp(terms)
                        ld_tile = np.tile(ld, (xk.shape[0], 1))
                        ExpLambdaXk = (1.0 / McExp) * np.sum(
                            ld_tile * xk, axis=1, keepdims=True,
                        )
                        ExpLambdaXkXkT = (1.0 / McExp) * (
                            (ld_tile * xk) @ xk.T
                        )
                        GradTerm = (
                            GradTerm
                            + dN[c, k] * x_K[:, k:k + 1]
                            - ExpLambdaXk
                        )
                        HessianTerm = HessianTerm - ExpLambdaXkXkT
                    elif fitType == "binomial":
                        e = np.exp(terms)
                        ld = e / (1.0 + e)
                        ld_tile = np.tile(ld, (xk.shape[0], 1))
                        ld2_tile = np.tile(ld ** 2, (xk.shape[0], 1))
                        ld3_tile = np.tile(ld ** 3, (xk.shape[0], 1))
                        ExplambdaDeltaXkXk = (1.0 / McExp) * (
                            (ld_tile * xk) @ xk.T
                        )
                        ExplambdaDeltaSqXkXkT = (1.0 / McExp) * (
                            (ld2_tile * xk) @ xk.T
                        )
                        ExplambdaDeltaCubeXkXkT = (1.0 / McExp) * (
                            (ld3_tile * xk) @ xk.T
                        )
                        ExpLambdaXk = (1.0 / McExp) * np.sum(
                            ld_tile * xk, axis=1, keepdims=True,
                        )
                        ExpLambdaSquaredXk = (1.0 / McExp) * np.sum(
                            ld2_tile * xk, axis=1, keepdims=True,
                        )
                        GradTerm = (
                            GradTerm
                            + dN[c, k] * x_K[:, k:k + 1]
                            - (dN[c, k] + 1) * ExpLambdaXk
                            + ExpLambdaSquaredXk
                        )
                        # Jacobian of GradTerm (MATLAB PPLFP_MStep, repaired C5):
                        # (-(dN+1)p + (dN+3)p^2 - 2p^3) x x'.  The former
                        # E[p xx'] + E[p^2 xx'] - 2E[p^3 xx'] had the wrong sign.
                        HessianTerm = (
                            HessianTerm
                            - (dN[c, k] + 1) * ExplambdaDeltaXkXk
                            + (dN[c, k] + 3) * ExplambdaDeltaSqXkXkT
                            - 2 * ExplambdaDeltaCubeXkXkT
                        )
                if (
                    np.isnan(HessianTerm).any()
                    or np.isinf(HessianTerm).any()
                ):
                    betahat_newTemp = betahat_new[:, c:c + 1].copy()
                else:
                    try:
                        step = _matlab_mldivide(HessianTerm, GradTerm)
                    except np.linalg.LinAlgError:
                        # An exactly singular Hessian (e.g. every lambda of a
                        # term underflowed to 0): MATLAB's HessianTerm\GradTerm
                        # warns and returns NaN there, so the previous value is
                        # kept, as in PP_MStep.  This port took the lstsq
                        # minimum-norm step instead.
                        step = np.full_like(GradTerm, np.nan)
                    betahat_newTemp = betahat_new[:, c:c + 1] - step
                    if np.isnan(betahat_newTemp).any():
                        betahat_newTemp = betahat_new[:, c:c + 1].copy()
                mabsDiff = np.max(
                    np.abs(
                        betahat_newTemp.ravel() - betahat_new[:, c]
                    )
                )
                if mabsDiff < diffTol:
                    converged = True
                betahat_new[:, c] = betahat_newTemp.ravel()
                it += 1
            _logger.info("neuron:%s iter: %s", c + 1, ",".join(str(i) for i in iters_done))

        # ----- CIF means (mu) -----------------------------------------
        for c in range(numCells):
            converged = False
            it = 1
            maxIter = 100
            while (not converged) and it < maxIter:
                HessianTerm = 0.0
                GradTerm = 0.0
                Hk = _Hk_for_cell(c)
                gammaC = _gammaC(gammahat_arr, c)
                for k in range(K):
                    xk = xkPerm[:, :, k]
                    terms = (
                        muhat_new[c]
                        + betahat[:, c] @ xk
                        + gammaC @ Hk[k, :].reshape(-1, 1)
                    )
                    terms = np.asarray(terms).reshape(-1)
                    if fitType == "poisson":
                        ld = np.exp(terms)
                        ExpLambdaDelta = (1.0 / McExp) * np.sum(ld)
                        GradTerm = GradTerm + (
                            dN[c, k] - ExpLambdaDelta
                        )
                        HessianTerm = HessianTerm - ExpLambdaDelta
                    elif fitType == "binomial":
                        e = np.exp(terms)
                        ld = e / (1.0 + e)
                        ExpLambdaDelta = (1.0 / McExp) * np.sum(ld)
                        ExpLambdaDeltaSq = (1.0 / McExp) * np.sum(ld ** 2)
                        ExpLambdaDeltaCubed = (1.0 / McExp) * np.sum(ld ** 3)
                        GradTerm = GradTerm + (
                            dN[c, k]
                            - (dN[c, k] + 1) * ExpLambdaDelta
                            + ExpLambdaDeltaSq
                        )
                        HessianTerm = HessianTerm + (
                            -ExpLambdaDelta * (dN[c, k] + 1)
                            + ExpLambdaDeltaSq * (dN[c, k] + 3)
                            - 2 * ExpLambdaDeltaCubed
                        )
                if np.isnan(HessianTerm) or np.isinf(HessianTerm):
                    muhat_newTemp = muhat_new[c]
                else:
                    # MATLAB HessianTerm\GradTerm: a zero Hessian gives +-Inf,
                    # which MATLAB accepts, or NaN (0/0: keep).  This port kept
                    # the previous value for HessianTerm == 0.
                    muhat_newTemp = muhat_new[c] - np.divide(GradTerm, HessianTerm)
                    if np.isnan(muhat_newTemp):
                        muhat_newTemp = muhat_new[c]
                mabsDiff = abs(muhat_newTemp - muhat_new[c])
                if mabsDiff < diffTol:
                    converged = True
                muhat_new[c] = muhat_newTemp
                it += 1

        # ----- history coefficients (gamma) ---------------------------
        windowTimes_arr = (
            np.asarray(windowTimes)
            if windowTimes is not None else np.array([])
        )
        has_gamma = (
            windowTimes_arr.size > 0
            and gammahat_new.size > 0
            and np.any(gammahat_new != 0)
        )
        if has_gamma:
            # A single (scalar) history coefficient -- one cell, one window --
            # is MATLAB's 1 x 1 gammahat_new(:,c); slice it as such and restore
            # the caller's shape afterwards (a 0-d gamma used to raise here).
            gamma_shape = gammahat_new.shape
            if gammahat_new.size == 1:
                gammahat_new = gammahat_new.reshape(1, 1)
            for c in range(numCells):
                converged = False
                it = 1
                maxIter = 100
                while (not converged) and it < maxIter:
                    n_hist = (
                        gammahat_arr.shape[0]
                        if gammahat_arr.ndim >= 1 else 1
                    )
                    HessianTerm = np.zeros((n_hist, n_hist))
                    GradTerm = np.zeros((n_hist, 1))
                    Hk = _Hk_for_cell(c)
                    gammaC = _gammaC(gammahat_new, c)
                    for k in range(K):
                        xk = xkPerm[:, :, k]
                        Hk_row = Hk[k, :].reshape(-1, 1)
                        terms = (
                            muhat[c]
                            + betahat[:, c] @ xk
                            + gammaC @ Hk_row
                        )
                        terms = np.asarray(terms).reshape(-1)
                        if fitType == "poisson":
                            ld = np.exp(terms)
                            ExpLambdaDelta = (1.0 / McExp) * np.sum(ld)
                            GradTerm = (
                                GradTerm
                                + (dN[c, k] - ExpLambdaDelta) * Hk_row
                            )
                            HessianTerm = (
                                HessianTerm
                                - ExpLambdaDelta * (Hk_row @ Hk_row.T)
                            )
                        elif fitType == "binomial":
                            e = np.exp(terms)
                            ld = e / (1.0 + e)
                            ExpLambdaDelta = (1.0 / McExp) * np.sum(ld)
                            ExpLambdaDeltaSq = (1.0 / McExp) * np.sum(
                                ld ** 2,
                            )
                            ExpLambdaDeltaCubed = (
                                (1.0 / McExp) * np.sum(ld ** 3)
                            )
                            GradTerm = GradTerm + (
                                dN[c, k]
                                - (dN[c, k] + 1) * ExpLambdaDelta
                                + ExpLambdaDeltaSq
                            ) * Hk_row
                            HessianTerm = HessianTerm + (
                                -ExpLambdaDelta * (dN[c, k] + 1)
                                + ExpLambdaDeltaSq * (dN[c, k] + 3)
                                - 2 * ExpLambdaDeltaCubed
                            ) * (Hk_row @ Hk_row.T)
                    if (
                        np.isnan(HessianTerm).any()
                        or np.isinf(HessianTerm).any()
                    ):
                        gammahat_newTemp = (
                            gammahat_new[:, c:c + 1].copy()
                        )
                    else:
                        try:
                            step = _matlab_mldivide(HessianTerm, GradTerm)
                        except np.linalg.LinAlgError:
                            # Exactly singular: MATLAB returns NaN, so the
                            # previous value is kept (see beta; was lstsq).
                            step = np.full_like(GradTerm, np.nan)
                        gammahat_newTemp = (
                            gammahat_new[:, c:c + 1] - step
                        )
                        if np.isnan(gammahat_newTemp).any():
                            gammahat_newTemp = (
                                gammahat_new[:, c:c + 1].copy()
                            )
                    mabsDiff = np.max(
                        np.abs(
                            gammahat_newTemp.ravel()
                            - gammahat_new[:, c]
                        )
                    )
                    if mabsDiff < diffTol:
                        converged = True
                    gammahat_new[:, c] = gammahat_newTemp.ravel()
                    it += 1
            if gammahat_new.size == 1:
                gammahat_new = gammahat_new.reshape(gamma_shape)

        return (
            Ahat, Qhat, Chat, Rhat, alphahat,
            muhat_new, betahat_new, gammahat_new, x0hat, Px0hat,
        )


__all__ = [
    "PPLFP_Decode_predict",
    "PPLFP_Decode_update",
    "PPLFP_DecodeLinear",
    "PPLFP_fixedIntervalSmoother",
    "PPLFP_EMCreateConstraints",
    "PPLFP_ComputeParamStandardErrors",
    "PPLFP_EM",
    "PPLFP_EStep",
    "PPLFP_MStep",
]
