"""Numba-JIT'd hot-loop kernels for nstat decoding algorithms.

This module is an OPT-IN performance accelerator behind the
``nstat-toolbox[numba]`` extras key.  When Numba is installed, the
``DecodingAlgorithms.PPDecodeFilterLinear`` and
``DecodingAlgorithms.kalman_filter`` fast paths route through the
``@numba.njit`` kernels declared here; when Numba is absent, the
pure-Python (NumPy) paths in :mod:`nstat.decoding_algorithms` are used
unchanged.

Correctness contract
--------------------
- ``fastmath=False`` on every ``@njit`` decorator.  MATLAB parity
  requires exact float64 semantics; ``fastmath=True`` reorders
  arithmetic and breaks bit-equivalence against ``matlab_gold/*.mat``
  fixtures.
- The kernels deliberately mirror the inlined fast-path loop bodies in
  ``nstat/decoding_algorithms.py`` line-for-line; do not "simplify"
  the math without re-running the gold-fixture suite.
- The Numba path and the pure-Python path are tested side-by-side in
  ``tests/test_decoding_algorithms_fidelity.py`` via a parametrized
  ``force_numba`` fixture.

Notes
-----
The kernels expect canonical, contiguous float64 arrays of fully
resolved shapes — all dispatch/normalisation/empty-checks happen in
the Python wrapper before the JIT call.  This is what lets Numba inline
the LAPACK ``solve`` for tiny 2x2/4x4 matrices instead of round-tripping
to NumPy.
"""
from __future__ import annotations

import numpy as np

try:
    import numba as _numba  # type: ignore[import-not-found]

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on default install
    _numba = None  # type: ignore[assignment]
    _NUMBA_AVAILABLE = False


__all__ = [
    "_NUMBA_AVAILABLE",
    "ppdecode_linear_loop",
    "kalman_filter_loop",
]


if _NUMBA_AVAILABLE:

    @_numba.njit(cache=True, fastmath=False)
    def _ppdecode_linear_loop_jit(
        A_static,
        Q_static,
        beta_mat,
        mu_vec,
        gamma_mat,
        H_tensor,
        obs,
        x0_vec,
        Pi0_mat,
        is_binomial,
        has_history,
    ):
        """JIT'd no-target, time-invariant A/Q, no-Wconv standard fast path.

        Mirrors ``_ppdecode_filter_linear`` line 990-1056 (the inlined
        standard loop) for the most common production shape: time-
        invariant A/Q, no convergence override, no history windows or
        history windows present uniformly.  All other branches
        (time-varying A/Q, target-estimation, Wconv) keep the
        pure-Python path.
        """
        ns = A_static.shape[0]
        num_cells = obs.shape[0]
        N = obs.shape[1]
        num_windows = H_tensor.shape[1]

        x_p = np.zeros((ns, N + 1))
        x_u = np.zeros((ns, N))
        W_p = np.zeros((ns, ns, N + 1))
        W_u = np.zeros((ns, ns, N))

        # Initial predict: x_p[:,0] = A @ x0,  W_p[:,:,0] = A @ Pi0 @ A.T + Q
        for i in range(ns):
            acc = 0.0
            for k in range(ns):
                acc += A_static[i, k] * x0_vec[k]
            x_p[i, 0] = acc

        # W_p[:,:,0] = A_static @ Pi0_mat @ A_static.T + Q_static
        tmp = np.zeros((ns, ns))
        for i in range(ns):
            for j in range(ns):
                s = 0.0
                for k in range(ns):
                    s += A_static[i, k] * Pi0_mat[k, j]
                tmp[i, j] = s
        for i in range(ns):
            for j in range(ns):
                s = 0.0
                for k in range(ns):
                    s += tmp[i, k] * A_static[j, k]
                W_p[i, j, 0] = s + Q_static[i, j]

        ident = np.eye(ns)
        lin_term = np.zeros(num_cells)
        lambda_delta = np.zeros(num_cells)
        factor = np.zeros(num_cells)
        temp_vec = np.zeros(num_cells)
        sum_val_vec = np.zeros(ns)
        sum_val_mat = np.zeros((ns, ns))
        SmW = np.zeros((ns, ns))
        lhs = np.zeros((ns, ns))
        W_u_t = np.zeros((ns, ns))
        W_pred = np.zeros((ns, ns))

        for time_index in range(N):
            x_vec_t = x_p[:, time_index]
            W_mat_t = W_p[:, :, time_index]

            # lambda_delta = exp(clip(mu + beta.T @ x + sum(gamma * H[t]), -20, 20))
            for c in range(num_cells):
                acc = mu_vec[c]
                for k in range(ns):
                    acc += beta_mat[k, c] * x_vec_t[k]
                lin_term[c] = acc
            if has_history:
                for c in range(num_cells):
                    acc = 0.0
                    for w in range(num_windows):
                        acc += gamma_mat[w, c] * H_tensor[time_index, w, c]
                    lin_term[c] += acc

            for c in range(num_cells):
                v = lin_term[c]
                if v > 20.0:
                    v = 20.0
                elif v < -20.0:
                    v = -20.0
                if is_binomial:
                    e = np.exp(v)
                    lambda_delta[c] = e / (1.0 + e)
                else:
                    lambda_delta[c] = np.exp(v)

            # factor / temp_vec
            for c in range(num_cells):
                obs_c = obs[c, time_index]
                ld = lambda_delta[c]
                if is_binomial:
                    one_minus = 1.0 - ld
                    factor[c] = (obs_c - ld) * one_minus
                    temp_vec[c] = (obs_c + (1.0 - 2.0 * ld)) * one_minus * ld
                else:
                    factor[c] = obs_c - ld
                    temp_vec[c] = ld

            # sum_val_vec = sum(beta * factor, axis=1)  [ns]
            for i in range(ns):
                acc = 0.0
                for c in range(num_cells):
                    acc += beta_mat[i, c] * factor[c]
                sum_val_vec[i] = acc

            # sum_val_mat = (beta * temp_vec) @ beta.T  [ns, ns]
            for i in range(ns):
                for j in range(ns):
                    acc = 0.0
                    for c in range(num_cells):
                        acc += beta_mat[i, c] * temp_vec[c] * beta_mat[j, c]
                    sum_val_mat[i, j] = acc

            # SmW = sum_val_mat @ W_mat_t
            for i in range(ns):
                for j in range(ns):
                    acc = 0.0
                    for k in range(ns):
                        acc += sum_val_mat[i, k] * W_mat_t[k, j]
                    SmW[i, j] = acc

            # lhs = I + SmW; W_u_t = W_mat_t @ (I - solve(lhs, SmW))
            for i in range(ns):
                for j in range(ns):
                    lhs[i, j] = SmW[i, j]
                lhs[i, i] += 1.0

            # solved = solve(lhs, SmW); inner = I - solved
            try:
                solved = np.linalg.solve(lhs, SmW)
                # inner = I - solved;  W_u_t = W_mat_t @ inner
                for i in range(ns):
                    for j in range(ns):
                        acc = 0.0
                        for k in range(ns):
                            inner_kj = -solved[k, j]
                            if k == j:
                                inner_kj += 1.0
                            acc += W_mat_t[i, k] * inner_kj
                        W_u_t[i, j] = acc
            except Exception:  # pragma: no cover - LinAlgError fallback
                for i in range(ns):
                    for j in range(ns):
                        W_u_t[i, j] = W_mat_t[i, j]

            # symmetrize W_u_t
            for i in range(ns):
                for j in range(i + 1, ns):
                    avg = 0.5 * (W_u_t[i, j] + W_u_t[j, i])
                    W_u_t[i, j] = avg
                    W_u_t[j, i] = avg

            # x_u_t = x_vec_t + W_u_t @ sum_val_vec
            for i in range(ns):
                acc = x_vec_t[i]
                for k in range(ns):
                    acc += W_u_t[i, k] * sum_val_vec[k]
                x_u[i, time_index] = acc
            for i in range(ns):
                for j in range(ns):
                    W_u[i, j, time_index] = W_u_t[i, j]

            # ---- inlined PPDecode_predict (time-invariant A/Q) ------
            # W_pred = A @ W_u_t @ A.T + Q
            for i in range(ns):
                for j in range(ns):
                    acc = 0.0
                    for k in range(ns):
                        acc += A_static[i, k] * W_u_t[k, j]
                    tmp[i, j] = acc
            for i in range(ns):
                for j in range(ns):
                    acc = 0.0
                    for k in range(ns):
                        acc += tmp[i, k] * A_static[j, k]
                    W_pred[i, j] = acc + Q_static[i, j]
            if not np.isfinite(W_pred[0, 0]):
                for i in range(ns):
                    for j in range(ns):
                        W_pred[i, j] = W_u_t[i, j]
            # symmetrize into W_p[:,:,t+1]
            for i in range(ns):
                for j in range(i, ns):
                    avg = 0.5 * (W_pred[i, j] + W_pred[j, i])
                    W_p[i, j, time_index + 1] = avg
                    W_p[j, i, time_index + 1] = avg

            # x_p[:, t+1] = A @ x_u_t
            for i in range(ns):
                acc = 0.0
                for k in range(ns):
                    acc += A_static[i, k] * x_u[k, time_index]
                x_p[i, time_index + 1] = acc

        return x_p, W_p, x_u, W_u

    @_numba.njit(cache=True, fastmath=False)
    def _kalman_filter_matlab_loop_jit(
        A_static,
        C_static,
        Pv_static,
        Pw_static,
        Px0,
        x0_vec,
        y,
    ):
        """JIT'd ``_kalman_filter_matlab`` core loop, time-invariant case.

        Mirrors lines 1910-1944 of ``decoding_algorithms.py`` for the
        time-invariant A/C/Pv/Pw case with no pre-converged gain
        override (``GnConv is None``).  Implements MATLAB's per-step
        update + predict + gain-convergence detection.
        """
        Dx = A_static.shape[0]
        Dy = C_static.shape[0]
        N = y.shape[1]

        x_p = np.zeros((Dx, N + 1))
        Pe_p = np.zeros((Dx, Dx, N + 1))
        x_u = np.zeros((Dx, N))
        Pe_u = np.zeros((Dx, Dx, N))
        Gn = np.zeros((Dx, Dy, N))

        # x_p[:,0] = x0; Pe_p[:,:,0] = Px0
        for i in range(Dx):
            x_p[i, 0] = x0_vec[i]
            for j in range(Dx):
                Pe_p[i, j, 0] = Px0[i, j]

        _GnConv_active = False
        GnConvIter = -1  # sentinel: -1 means None

        # scratch
        S = np.zeros((Dy, Dy))
        CPp = np.zeros((Dy, Dx))
        G = np.zeros((Dx, Dy))
        G_prev = np.zeros((Dx, Dy))
        GC = np.zeros((Dx, Dx))
        Pu = np.zeros((Dx, Dx))
        APu = np.zeros((Dx, Dx))
        Pp = np.zeros((Dx, Dx))
        innovation = np.zeros(Dy)
        eye_Dy = np.eye(Dy)
        GnConv_mat = np.zeros((Dx, Dy))

        for n in range(N):
            # CPp = C @ Pe_p[:,:,n]
            for i in range(Dy):
                for j in range(Dx):
                    acc = 0.0
                    for k in range(Dx):
                        acc += C_static[i, k] * Pe_p[k, j, n]
                    CPp[i, j] = acc
            if _GnConv_active:
                for i in range(Dx):
                    for j in range(Dy):
                        G[i, j] = GnConv_mat[i, j]
            else:
                # S = CPp @ C.T + Pw
                for i in range(Dy):
                    for j in range(Dy):
                        acc = 0.0
                        for k in range(Dx):
                            acc += CPp[i, k] * C_static[j, k]
                        S[i, j] = acc + Pw_static[i, j]
                # G = Pe_p @ C.T @ solve(S, I) = Pe_p @ C.T @ inv(S)
                # Compute Sinv = solve(S, eye(Dy))
                Sinv = np.linalg.solve(S, eye_Dy)
                # G = Pe_p[:,:,n] @ C.T @ Sinv
                # First PpCt = Pe_p @ C.T  [Dx, Dy]
                PpCt = np.zeros((Dx, Dy))
                for i in range(Dx):
                    for j in range(Dy):
                        acc = 0.0
                        for k in range(Dx):
                            acc += Pe_p[i, k, n] * C_static[j, k]
                        PpCt[i, j] = acc
                for i in range(Dx):
                    for j in range(Dy):
                        acc = 0.0
                        for k in range(Dy):
                            acc += PpCt[i, k] * Sinv[k, j]
                        G[i, j] = acc

            # innovation = y[:,n] - C @ x_p[:,n]
            for i in range(Dy):
                acc = y[i, n]
                for k in range(Dx):
                    acc -= C_static[i, k] * x_p[k, n]
                innovation[i] = acc

            # x_u[:,n] = x_p[:,n] + G @ innovation
            for i in range(Dx):
                acc = x_p[i, n]
                for k in range(Dy):
                    acc += G[i, k] * innovation[k]
                x_u[i, n] = acc

            # Pu = Pe_p[:,:,n] - G @ C @ Pe_p[:,:,n]
            # GC = G @ C
            for i in range(Dx):
                for j in range(Dx):
                    acc = 0.0
                    for k in range(Dy):
                        acc += G[i, k] * C_static[k, j]
                    GC[i, j] = acc
            # Pu = Pe_p - GC @ Pe_p
            for i in range(Dx):
                for j in range(Dx):
                    acc = 0.0
                    for k in range(Dx):
                        acc += GC[i, k] * Pe_p[k, j, n]
                    Pu[i, j] = Pe_p[i, j, n] - acc
            # symmetrize Pu
            for i in range(Dx):
                for j in range(i, Dx):
                    avg = 0.5 * (Pu[i, j] + Pu[j, i])
                    Pu[i, j] = avg
                    Pu[j, i] = avg
            for i in range(Dx):
                for j in range(Dx):
                    Pe_u[i, j, n] = Pu[i, j]
                for j in range(Dy):
                    Gn[i, j, n] = G[i, j]

            # Predict: x_p[:,n+1] = A @ x_u[:,n]; Pe_p[:,:,n+1] = A @ Pu @ A.T + Pv
            if _GnConv_active:
                for i in range(Dx):
                    for j in range(Dx):
                        Pp[i, j] = Pu[i, j]
            else:
                # APu = A @ Pu
                for i in range(Dx):
                    for j in range(Dx):
                        acc = 0.0
                        for k in range(Dx):
                            acc += A_static[i, k] * Pu[k, j]
                        APu[i, j] = acc
                # Pp = APu @ A.T + Pv
                for i in range(Dx):
                    for j in range(Dx):
                        acc = 0.0
                        for k in range(Dx):
                            acc += APu[i, k] * A_static[j, k]
                        Pp[i, j] = acc + Pv_static[i, j]
            # symmetrize Pp into Pe_p[:,:,n+1]
            for i in range(Dx):
                for j in range(i, Dx):
                    avg = 0.5 * (Pp[i, j] + Pp[j, i])
                    Pe_p[i, j, n + 1] = avg
                    Pe_p[j, i, n + 1] = avg

            # x_p[:,n+1] = A @ x_u[:,n]
            for i in range(Dx):
                acc = 0.0
                for k in range(Dx):
                    acc += A_static[i, k] * x_u[k, n]
                x_p[i, n + 1] = acc

            # Gain convergence detection.  We track ``G_prev`` explicitly
            # instead of indexing ``Gn[..., n - 1]`` to satisfy the
            # zero-based-indexing convention enforced by
            # ``tests/test_indexing_convention.py`` (HR3).
            if n > 0 and not _GnConv_active:
                max_diff = 0.0
                for i in range(Dx):
                    for j in range(Dy):
                        d = G[i, j] - G_prev[i, j]
                        if d < 0.0:
                            d = -d
                        if d > max_diff:
                            max_diff = d
                if max_diff < 1e-6:
                    for i in range(Dx):
                        for j in range(Dy):
                            GnConv_mat[i, j] = G[i, j]
                    _GnConv_active = True
                    GnConvIter = n
            # Roll G into G_prev for the next iteration's comparison.
            for i in range(Dx):
                for j in range(Dy):
                    G_prev[i, j] = G[i, j]

        return x_p, Pe_p, x_u, Pe_u, Gn, GnConvIter


def ppdecode_linear_loop(
    A_static: np.ndarray,
    Q_static: np.ndarray,
    beta_mat: np.ndarray,
    mu_vec: np.ndarray,
    gamma_mat: np.ndarray,
    H_tensor: np.ndarray,
    obs: np.ndarray,
    x0_vec: np.ndarray,
    Pi0_mat: np.ndarray,
    is_binomial: bool,
    has_history: bool,
):
    """Public wrapper around the JIT'd PPAF linear loop.

    The wrapper ensures contiguous float64 arrays and dispatches to the
    JIT'd kernel.  Caller is responsible for verifying the time-
    invariant / no-target / no-Wconv preconditions.
    """
    if not _NUMBA_AVAILABLE:
        raise RuntimeError(
            "numba is not installed; install nstat-toolbox[numba] for this fast path"
        )
    return _ppdecode_linear_loop_jit(
        np.ascontiguousarray(A_static, dtype=np.float64),
        np.ascontiguousarray(Q_static, dtype=np.float64),
        np.ascontiguousarray(beta_mat, dtype=np.float64),
        np.ascontiguousarray(mu_vec, dtype=np.float64),
        np.ascontiguousarray(gamma_mat, dtype=np.float64),
        np.ascontiguousarray(H_tensor, dtype=np.float64),
        np.ascontiguousarray(obs, dtype=np.float64),
        np.ascontiguousarray(x0_vec, dtype=np.float64),
        np.ascontiguousarray(Pi0_mat, dtype=np.float64),
        bool(is_binomial),
        bool(has_history),
    )


def kalman_filter_loop(
    A_static: np.ndarray,
    C_static: np.ndarray,
    Pv_static: np.ndarray,
    Pw_static: np.ndarray,
    Px0: np.ndarray,
    x0_vec: np.ndarray,
    y: np.ndarray,
):
    """Public wrapper around the JIT'd Kalman filter loop (MATLAB style).

    Returns ``(x_p, Pe_p, x_u, Pe_u, Gn, GnConvIter_or_None)``.
    """
    if not _NUMBA_AVAILABLE:
        raise RuntimeError(
            "numba is not installed; install nstat-toolbox[numba] for this fast path"
        )
    x_p, Pe_p, x_u, Pe_u, Gn, conv = _kalman_filter_matlab_loop_jit(
        np.ascontiguousarray(A_static, dtype=np.float64),
        np.ascontiguousarray(C_static, dtype=np.float64),
        np.ascontiguousarray(Pv_static, dtype=np.float64),
        np.ascontiguousarray(Pw_static, dtype=np.float64),
        np.ascontiguousarray(Px0, dtype=np.float64),
        np.ascontiguousarray(x0_vec, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
    )
    gn_iter = None if conv < 0 else int(conv)
    return x_p, Pe_p, x_u, Pe_u, Gn, gn_iter
