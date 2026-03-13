#!/usr/bin/env python3
"""Example 05 — Cross-language parity test.

Runs the Example 05 decoding algorithms (PPDecodeFilterLinear and
PPHybridFilterLinear) in *both* Python and MATLAB with IDENTICAL inputs
and compares every numerical output.

Since the example scripts use different RNG implementations (MATLAB vs
NumPy), we generate deterministic test inputs shared between both
languages rather than replicate the full simulation pipeline.

Comparison points:
  - PPDecodeFilterLinear: x_p, W_p, x_u, W_u (1-D scalar case)
  - PPDecodeFilterLinear: x_p, W_p, x_u, W_u (4-D reach case)
  - PPDecodeFilterLinear: goal-directed decode
  - PPHybridFilterLinear: S_est, X_est, MU_u

Requirements:
    - MATLAB Engine API for Python (``pip install matlabengine``)
    - nSTAT MATLAB repo at ``MATLAB_NSTAT`` path below

Usage::

    python tests/test_example05_parity.py
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MATLAB_NSTAT = Path("/Users/iahncajigas/Library/CloudStorage/Dropbox/Claude/nSTAT")
TOL = 1e-8
TOL_LOOSE = 1e-5  # for accumulated filter differences


def _generate_test_inputs():
    """Generate deterministic test inputs shared between MATLAB and Python."""
    rng = np.random.default_rng(42)

    # --- Part A: 1-D scalar decoding ---
    delta = 0.001
    T = 500  # short for speed
    n_cells = 10
    A_1d = np.array([[1.0]])
    Q_1d = np.array([[0.001]])
    x0_1d = np.array([0.0])
    Pi0_1d = 0.5 * np.eye(1)
    b0_1d = np.log(10.0 * delta) * np.ones(n_cells) + 0.1 * np.arange(n_cells)
    beta_1d = 0.5 * np.ones((1, n_cells)) + 0.05 * np.arange(n_cells).reshape(1, -1)

    # Deterministic spike data: simple pattern
    dN_1d = np.zeros((n_cells, T), dtype=float)
    for c in range(n_cells):
        spike_times = np.arange(c * 10 + 5, T, 50 + c * 3)
        dN_1d[c, spike_times] = 1.0

    # --- Part B: 4-D reach decoding ---
    ns = 4
    n_cells_b = 15
    A_4d = np.array([
        [1, 0, delta, 0],
        [0, 1, 0, delta],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    Q_4d = 0.001 * np.eye(ns, dtype=float)
    x0_4d = np.array([0.0, 0.0, 0.1, 0.1])
    Pi0_4d = 0.1 * np.eye(ns)
    b0_4d = -3.0 * np.ones(n_cells_b) + 0.1 * np.arange(n_cells_b)
    beta_4d = np.zeros((ns, n_cells_b), dtype=float)
    for c in range(n_cells_b):
        beta_4d[0, c] = 0.5 * (c % 3)
        beta_4d[1, c] = 0.3 * ((c + 1) % 3)
        beta_4d[2, c] = 1.0 * ((c + 2) % 4 - 1.5)
        beta_4d[3, c] = 0.8 * ((c + 3) % 4 - 1.5)

    T_b = 300
    dN_4d = np.zeros((n_cells_b, T_b), dtype=float)
    for c in range(n_cells_b):
        spike_times = np.arange(c * 5 + 3, T_b, 30 + c * 2)
        dN_4d[c, spike_times] = 1.0

    # Goal-directed
    yT = np.array([0.1, 0.1, 0.0, 0.0])
    PiT = 0.01 * np.eye(ns)

    # --- Part C: Hybrid filter ---
    # NOTE: MATLAB's PPHybridFilterLinear uses a SINGLE shared mu and beta
    # for all models (it copies `beta` to each model internally).  Python
    # enhanced this to support per-model parameters.  For parity, we use
    # the same mu/beta for both modes.
    n_cells_c = 8
    T_c = 200
    A_reach = A_4d.copy()
    Q_reach = 0.001 * np.eye(ns)
    A_hold = A_4d.copy()
    A_hold[2, 2] = 0.95
    A_hold[3, 3] = 0.95
    Q_hold = 0.0005 * np.eye(ns)

    # Shared mu and beta across both discrete modes (MATLAB requirement)
    b0_shared = -3.0 * np.ones(n_cells_c) + 0.1 * np.arange(n_cells_c)
    beta_c = np.zeros((ns, n_cells_c), dtype=float)
    for c in range(n_cells_c):
        beta_c[0, c] = 0.3 * (c % 3)
        beta_c[1, c] = 0.2 * ((c + 1) % 3)
        beta_c[2, c] = 0.5 * ((c + 2) % 4 - 1.5)
        beta_c[3, c] = 0.4 * ((c + 3) % 4 - 1.5)

    dN_c = np.zeros((n_cells_c, T_c), dtype=float)
    for c in range(n_cells_c):
        spike_times = np.arange(c * 7 + 2, T_c, 25 + c)
        dN_c[c, spike_times] = 1.0

    p_ij = np.array([[0.985, 0.015], [0.02, 0.98]], dtype=float)
    Mu0 = np.array([0.5, 0.5])
    x0_c = [x0_4d.copy(), x0_4d.copy()]
    Pi0_c = [0.5 * np.eye(ns), 0.5 * np.eye(ns)]

    return {
        # Part A
        "A_1d": A_1d, "Q_1d": Q_1d, "dN_1d": dN_1d, "b0_1d": b0_1d,
        "beta_1d": beta_1d, "delta": delta, "x0_1d": x0_1d, "Pi0_1d": Pi0_1d,
        "T": T, "n_cells": n_cells,
        # Part B
        "A_4d": A_4d, "Q_4d": Q_4d, "dN_4d": dN_4d, "b0_4d": b0_4d,
        "beta_4d": beta_4d, "x0_4d": x0_4d, "Pi0_4d": Pi0_4d,
        "T_b": T_b, "n_cells_b": n_cells_b,
        "yT": yT, "PiT": PiT,
        # Part C
        "A_reach": A_reach, "Q_reach": Q_reach,
        "A_hold": A_hold, "Q_hold": Q_hold,
        "dN_c": dN_c, "b0_shared": b0_shared,
        "beta_c": beta_c, "p_ij": p_ij, "Mu0": Mu0,
        "x0_c": x0_c, "Pi0_c": Pi0_c,
        "T_c": T_c, "n_cells_c": n_cells_c,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MATLAB side
# ═══════════════════════════════════════════════════════════════════════════
def run_matlab(inputs):
    """Run decoding algorithms in MATLAB with shared inputs."""
    import matlab.engine
    import matlab

    eng = matlab.engine.start_matlab()
    eng.addpath(str(MATLAB_NSTAT), nargout=0)

    def _to_ml(arr):
        """Convert numpy array to MATLAB double."""
        return matlab.double(arr.tolist())

    # Push shared inputs to MATLAB workspace
    eng.workspace["A_1d"] = _to_ml(inputs["A_1d"])
    eng.workspace["Q_1d"] = _to_ml(inputs["Q_1d"])
    eng.workspace["dN_1d"] = _to_ml(inputs["dN_1d"])
    eng.workspace["b0_1d"] = _to_ml(inputs["b0_1d"].reshape(1, -1))
    eng.workspace["beta_1d"] = _to_ml(inputs["beta_1d"])
    eng.workspace["delta"] = float(inputs["delta"])
    eng.workspace["x0_1d"] = _to_ml(inputs["x0_1d"].reshape(-1, 1))
    eng.workspace["Pi0_1d"] = _to_ml(inputs["Pi0_1d"])

    # Part A: 1-D decode
    eng.eval("""
        [x_p_1d, W_p_1d, x_u_1d, W_u_1d] = DecodingAlgorithms.PPDecodeFilterLinear( ...
            A_1d, Q_1d, dN_1d, b0_1d', beta_1d, 'binomial', delta, [], [], x0_1d, Pi0_1d);
    """, nargout=0)

    x_p_1d = np.array(eng.eval("x_p_1d")).flatten()
    x_u_1d = np.array(eng.eval("x_u_1d")).flatten()
    W_p_1d = np.array(eng.eval("squeeze(W_p_1d)")).flatten()
    W_u_1d = np.array(eng.eval("squeeze(W_u_1d)")).flatten()

    # Part B: 4-D decode (free)
    eng.workspace["A_4d"] = _to_ml(inputs["A_4d"])
    eng.workspace["Q_4d"] = _to_ml(inputs["Q_4d"])
    eng.workspace["dN_4d"] = _to_ml(inputs["dN_4d"])
    eng.workspace["b0_4d"] = _to_ml(inputs["b0_4d"].reshape(1, -1))
    eng.workspace["beta_4d"] = _to_ml(inputs["beta_4d"])
    eng.workspace["x0_4d"] = _to_ml(inputs["x0_4d"].reshape(-1, 1))
    eng.workspace["Pi0_4d"] = _to_ml(inputs["Pi0_4d"])

    eng.eval("""
        [x_p_4d, W_p_4d, x_u_4d, W_u_4d] = DecodingAlgorithms.PPDecodeFilterLinear( ...
            A_4d, Q_4d, dN_4d, b0_4d', beta_4d, 'binomial', delta, [], [], x0_4d, Pi0_4d);
    """, nargout=0)

    x_u_4d = np.array(eng.eval("x_u_4d"))  # (4, T)
    x_u_4d_row0 = x_u_4d[0, :] if x_u_4d.ndim > 1 else x_u_4d.flatten()
    x_u_4d_row3 = x_u_4d[3, :] if x_u_4d.ndim > 1 else x_u_4d.flatten()

    # Part B: 4-D decode (goal)
    eng.workspace["yT"] = _to_ml(inputs["yT"].reshape(-1, 1))
    eng.workspace["PiT"] = _to_ml(inputs["PiT"])

    eng.eval("""
        [~, ~, x_u_goal, W_u_goal] = DecodingAlgorithms.PPDecodeFilterLinear( ...
            A_4d, Q_4d, dN_4d, b0_4d', beta_4d, 'binomial', delta, [], [], ...
            x0_4d, Pi0_4d, yT, PiT, 0);
    """, nargout=0)

    x_u_goal = np.array(eng.eval("x_u_goal"))
    x_u_goal_row0 = x_u_goal[0, :] if x_u_goal.ndim > 1 else x_u_goal.flatten()

    # Part C: Hybrid filter
    eng.workspace["A_reach"] = _to_ml(inputs["A_reach"])
    eng.workspace["Q_reach"] = _to_ml(inputs["Q_reach"])
    eng.workspace["A_hold"] = _to_ml(inputs["A_hold"])
    eng.workspace["Q_hold"] = _to_ml(inputs["Q_hold"])
    eng.workspace["dN_c"] = _to_ml(inputs["dN_c"])
    eng.workspace["b0_shared"] = _to_ml(inputs["b0_shared"].reshape(-1, 1))
    eng.workspace["beta_c"] = _to_ml(inputs["beta_c"])
    eng.workspace["p_ij"] = _to_ml(inputs["p_ij"])
    eng.workspace["Mu0"] = _to_ml(inputs["Mu0"].reshape(1, -1))
    eng.workspace["x0_c1"] = _to_ml(inputs["x0_c"][0].reshape(-1, 1))
    eng.workspace["x0_c2"] = _to_ml(inputs["x0_c"][1].reshape(-1, 1))
    eng.workspace["Pi0_c1"] = _to_ml(inputs["Pi0_c"][0])
    eng.workspace["Pi0_c2"] = _to_ml(inputs["Pi0_c"][1])

    eng.eval("""
        A_cell = {A_reach, A_hold};
        Q_cell = {Q_reach, Q_hold};
        x0_cell = {x0_c1, x0_c2};
        Pi0_cell = {Pi0_c1, Pi0_c2};
        % MATLAB PPHybridFilterLinear expects single mu vector and beta matrix
        % (shared across all discrete models — not per-model cell arrays)
        [S_est, X_est, W_est, MU_u] = DecodingAlgorithms.PPHybridFilterLinear( ...
            A_cell, Q_cell, p_ij, Mu0', dN_c, b0_shared, beta_c, ...
            'binomial', delta, [], [], x0_cell, Pi0_cell);
    """, nargout=0)

    S_est = np.array(eng.eval("S_est")).flatten().astype(int)
    X_est = np.array(eng.eval("X_est"))
    MU_u = np.array(eng.eval("MU_u"))

    eng.quit()

    return {
        # Part A
        "x_p_1d": x_p_1d,
        "x_u_1d": x_u_1d,
        "W_p_1d": W_p_1d,
        "W_u_1d": W_u_1d,
        # Part B
        "x_u_4d_row0": x_u_4d_row0,
        "x_u_4d_row3": x_u_4d_row3,
        "x_u_goal_row0": x_u_goal_row0,
        # Part C
        "S_est": S_est,
        "X_est_row0": X_est[0, :] if X_est.ndim > 1 else X_est.flatten(),
        "X_est_row1": X_est[1, :] if X_est.ndim > 1 else X_est.flatten(),
        "MU_u_row0": MU_u[0, :] if MU_u.ndim > 1 else MU_u.flatten(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Python side
# ═══════════════════════════════════════════════════════════════════════════
def run_python(inputs):
    """Run decoding algorithms in Python with shared inputs."""
    from nstat import DecodingAlgorithms

    # Part A: 1-D decode
    x_p, W_p, x_u, W_u, _, _, _, _ = DecodingAlgorithms.PPDecodeFilterLinear(
        inputs["A_1d"], inputs["Q_1d"], inputs["dN_1d"],
        inputs["b0_1d"], inputs["beta_1d"], "binomial",
        inputs["delta"], None, None,
        inputs["x0_1d"], inputs["Pi0_1d"],
    )

    x_p_1d = x_p.flatten()
    x_u_1d = x_u.flatten()
    W_p_1d = W_p.flatten()
    W_u_1d = W_u.flatten()

    # Part B: 4-D decode (free)
    x_p4, W_p4, x_u4, W_u4, _, _, _, _ = DecodingAlgorithms.PPDecodeFilterLinear(
        inputs["A_4d"], inputs["Q_4d"], inputs["dN_4d"],
        inputs["b0_4d"], inputs["beta_4d"], "binomial",
        inputs["delta"], None, None,
        inputs["x0_4d"], inputs["Pi0_4d"],
    )

    # Part B: 4-D decode (goal)
    _, _, x_u_goal, _, _, _, _, _ = DecodingAlgorithms.PPDecodeFilterLinear(
        inputs["A_4d"], inputs["Q_4d"], inputs["dN_4d"],
        inputs["b0_4d"], inputs["beta_4d"], "binomial",
        inputs["delta"], None, None,
        inputs["x0_4d"], inputs["Pi0_4d"],
        inputs["yT"], inputs["PiT"], 0,
    )

    # Part C: Hybrid filter — use shared mu/beta to match MATLAB
    S_est, X_est, W_est, MU_u, _, _, _ = DecodingAlgorithms.PPHybridFilterLinear(
        [inputs["A_reach"], inputs["A_hold"]],
        [inputs["Q_reach"], inputs["Q_hold"]],
        inputs["p_ij"],
        inputs["Mu0"],
        inputs["dN_c"],
        inputs["b0_shared"],   # single shared mu
        inputs["beta_c"],       # single shared beta
        "binomial",
        inputs["delta"],
        None, None,
        inputs["x0_c"],
        inputs["Pi0_c"],
    )

    return {
        # Part A
        "x_p_1d": x_p_1d,
        "x_u_1d": x_u_1d,
        "W_p_1d": W_p_1d,
        "W_u_1d": W_u_1d,
        # Part B
        "x_u_4d_row0": x_u4[0, :],
        "x_u_4d_row3": x_u4[3, :],
        "x_u_goal_row0": x_u_goal[0, :],
        # Part C
        "S_est": S_est.flatten().astype(int),
        "X_est_row0": X_est[0, :],
        "X_est_row1": X_est[1, :],
        "MU_u_row0": MU_u[0, :],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Compare
# ═══════════════════════════════════════════════════════════════════════════
def compare(py: dict, ml: dict):
    """Compare Python and MATLAB results; return (passed, total)."""
    passed = 0
    total = 0

    def check(name, py_val, ml_val, tol=TOL):
        nonlocal passed, total
        total += 1
        py_a = np.asarray(py_val, dtype=float).flatten()
        ml_a = np.asarray(ml_val, dtype=float).flatten()
        if py_a.shape != ml_a.shape:
            print(f"  ✗ {name}  SHAPE MISMATCH py={py_a.shape} ml={ml_a.shape}")
            return
        if py_a.size == 0 and ml_a.size == 0:
            print(f"  ✓ {name}  (both empty)")
            passed += 1
            return
        delta = np.max(np.abs(py_a - ml_a))
        ok = delta < tol
        sym = "✓" if ok else "✗"
        extra = ""
        if not ok:
            idx = int(np.argmax(np.abs(py_a - ml_a)))
            extra = f" at [{idx}] py={py_a[idx]:.6f} ml={ml_a[idx]:.6f}"
        print(f"  {sym} {name}  (max Δ = {delta:.2e}){extra}")
        if ok:
            passed += 1

    # ─────── Part A: 1-D Scalar Decode ───────
    print("\n═══ Part A: PPDecodeFilterLinear (1-D) ═══")
    check("x_p (predicted)", py["x_p_1d"], ml["x_p_1d"], tol=TOL_LOOSE)
    check("x_u (updated)", py["x_u_1d"], ml["x_u_1d"], tol=TOL_LOOSE)
    check("W_p (pred cov)", py["W_p_1d"], ml["W_p_1d"], tol=TOL_LOOSE)
    check("W_u (upd cov)", py["W_u_1d"], ml["W_u_1d"], tol=TOL_LOOSE)

    # ─────── Part B: 4-D Reach Decode ───────
    print("\n═══ Part B: PPDecodeFilterLinear (4-D free) ═══")
    check("x_u row0 (x-pos)", py["x_u_4d_row0"], ml["x_u_4d_row0"], tol=TOL_LOOSE)
    check("x_u row3 (vy)", py["x_u_4d_row3"], ml["x_u_4d_row3"], tol=TOL_LOOSE)

    print("\n═══ Part B: PPDecodeFilterLinear (4-D goal) ═══")
    check("x_u_goal row0", py["x_u_goal_row0"], ml["x_u_goal_row0"], tol=TOL_LOOSE)

    # ─────── Part C: Hybrid Filter ───────
    print("\n═══ Part C: PPHybridFilterLinear ═══")
    check("S_est (discrete state)", py["S_est"], ml["S_est"])
    check("X_est row0 (x-pos)", py["X_est_row0"], ml["X_est_row0"], tol=TOL_LOOSE)
    check("X_est row1 (y-pos)", py["X_est_row1"], ml["X_est_row1"], tol=TOL_LOOSE)
    check("MU_u row0 (mode prob)", py["MU_u_row0"], ml["MU_u_row0"], tol=TOL_LOOSE)

    return passed, total


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating shared test inputs …")
    inputs = _generate_test_inputs()
    print(f"  Part A: {inputs['n_cells']} cells, {inputs['T']} time steps")
    print(f"  Part B: {inputs['n_cells_b']} cells, {inputs['T_b']} time steps")
    print(f"  Part C: {inputs['n_cells_c']} cells, {inputs['T_c']} time steps")

    t0 = _time.time()
    print("\nStarting MATLAB engine …")
    ml = run_matlab(inputs)
    t_ml = _time.time() - t0
    print(f"  MATLAB done in {t_ml:.1f}s")

    t1 = _time.time()
    print("\nRunning Example 05 in Python …")
    py = run_python(inputs)
    t_py = _time.time() - t1
    print(f"  Python done in {t_py:.1f}s")

    passed, total = compare(py, ml)

    print(f"\n{'═' * 60}")
    status = "✓ ALL PASS" if passed == total else f"✗ {total - passed} FAILED"
    print(f"  EXAMPLE 05 PARITY: {passed}/{total} passed  {status}")
    print(f"  MATLAB: {t_ml:.1f}s | Python: {t_py:.1f}s")
    print(f"{'═' * 60}")

    sys.exit(0 if passed == total else 1)
