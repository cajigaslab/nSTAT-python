#!/usr/bin/env python3
"""Example 05 — Stimulus Decoding With PPAF and PPHF.

This example demonstrates neural decoding using point-process adaptive filters
(PPAF) and point-process hybrid filters (PPHF) from the nSTAT toolbox.

The example has three parts:

Part A — Univariate Sinusoidal Stimulus (Figures 1–2):
  1. Define 20-cell population with logistic (binomial) tuning to a 1-D
     sinusoidal stimulus.
  2. Simulate spike observations from the binomial CIF.
  3. Decode the stimulus using ``PPDecodeFilterLinear`` (PPAF).

Part B — 4-State Arm Reach with PPAF (Figures 3–4):
  4. Simulate reaching trajectories (position + velocity, 4-D state) using
     minimum-jerk dynamics (cosine acceleration toward target).
  5. Encode with 20-cell velocity-tuned population (binomial CIF).
  6. Decode with PPAF (free) and PPAF + Goal; compare across 20 simulations.

Part C — Hybrid Filter (Figures 5–6):
  7. Load fixture trajectory with 6-D state (pos + vel + accel) and 2 discrete
     movement modes (not-moving / moving) from ``paperHybridFilterExample.mat``.
  8. Simulate 40-cell population with velocity-tuned binomial CIF.
  9. Decode joint discrete + continuous state via ``PPHybridFilterLinear``
     (both goal-directed and free), averaged over 20 simulations.

Paper mapping:
  Sections 2.3.6–2.3.7 (decoding); Figs. 8, 9, 14 plus hybrid extension.

Expected outputs:
  - Figure 1: CIF tuning curves and simulated spike raster.
  - Figure 2: Decoded stimulus vs true (with 95% confidence band).
  - Figure 3: Reach trajectory, position/velocity traces, neural raster, CIF.
  - Figure 4: PPAF decoding overlaid trajectories (free=green, goal=blue).
  - Figure 5: Hybrid fixture setup (reach path, traces, raster, discrete state).
  - Figure 6: Hybrid decoding summary (state est, probabilities, decoded path).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nstat import DecodingAlgorithms  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
#  Helper: simulate binomial spikes from linear-logistic CIF
# ──────────────────────────────────────────────────────────────────────────────


def _simulate_binomial_spikes_from_lambda(lambdaRate, delta, rng):
    """Simulate spikes from precomputed lambda rates via thinning.

    Parameters
    ----------
    lambdaRate : (C, T) array — firing rates [spikes/sec] per cell per time
    delta : float — bin width in seconds
    rng : numpy Generator

    Returns
    -------
    dN : (C, T) array — binary spike indicators
    """
    prob = lambdaRate * delta  # convert rate to probability per bin
    prob = np.clip(prob, 0.0, 1.0)
    return (rng.random(prob.shape) < prob).astype(float)


def _logistic_cif(dataMat, coeffs, delta):
    """Compute binomial CIF rates matching MATLAB's logistic link.

    Parameters
    ----------
    dataMat : (T, p) — design matrix [1, covariates]
    coeffs : (C, p) — per-cell coefficients [mu, betas]
    delta : float — bin width

    Returns
    -------
    lambdaRate : (C, T) — firing rates in spikes/sec
    """
    C = coeffs.shape[0]
    T = dataMat.shape[0]
    lambdaRate = np.zeros((C, T))
    for c in range(C):
        eta = dataMat @ coeffs[c, :]
        expEta = np.exp(np.clip(eta, -20.0, 20.0))
        p = expEta / (1.0 + expEta)
        lambdaRate[c, :] = p / delta
    return lambdaRate


# ──────────────────────────────────────────────────────────────────────────────
#  Part A — Univariate sinusoidal stimulus
# ──────────────────────────────────────────────────────────────────────────────


def _run_part_a(seed=0, n_cells=20):
    """Encode/decode a 1-D sinusoidal stimulus with 20-cell binomial CIF.

    Matches MATLAB: rng(0,'twister'), delta=0.001, f=2Hz, b0~N(log(10*delta),1),
    b1~N(0,1), logistic CIF, PPDecodeFilterLinear with A=1, Q=std(diff(stim)).
    """
    rng = np.random.default_rng(seed)
    delta = 0.001
    tmax = 1.0
    time = np.arange(0.0, tmax + delta, delta)
    T = len(time)
    f = 2.0

    # Encoding model — matches MATLAB exactly
    b1 = rng.standard_normal(n_cells)
    b0 = np.log(10.0 * delta) + rng.standard_normal(n_cells)
    stimSignal = np.sin(2.0 * np.pi * f * time)

    # Compute CIF and simulate spikes per cell
    dN = np.zeros((n_cells, T))
    lambdaAll = np.zeros((n_cells, T))
    for c in range(n_cells):
        eta = b1[c] * stimSignal + b0[c]
        expEta = np.exp(np.clip(eta, -20.0, 20.0))
        p = expEta / (1.0 + expEta)
        lambdaAll[c, :] = p / delta
        dN[c, :] = (rng.random(T) < p).astype(float)

    # State-space model: x(t+1) = A*x(t) + w
    A = np.array([[1.0]])
    Q_val = float(np.std(np.diff(stimSignal)))
    Q = np.array([[Q_val]])
    x0 = np.array([0.0])
    Pi0 = 0.5 * np.eye(1)

    # Decode
    beta = b1.reshape(1, -1)  # (1, C)
    x_p, W_p, x_u, W_u, _, _, _, _ = DecodingAlgorithms.PPDecodeFilterLinear(
        A, Q, dN, b0, beta, "binomial", delta, None, None, x0, Pi0
    )

    x_decoded = x_u[0, :]
    sigma = np.sqrt(np.maximum(W_u[0, 0, :], 0.0))
    z_val = 1.96
    ci_low = np.minimum(x_decoded - z_val * sigma, x_decoded + z_val * sigma)
    ci_high = np.maximum(x_decoded - z_val * sigma, x_decoded + z_val * sigma)
    rmse = float(np.sqrt(np.mean((x_decoded - stimSignal) ** 2)))

    return {
        "time": time,
        "stimSignal": stimSignal,
        "x_decoded": x_decoded,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "dN": dN,
        "lambdaAll": lambdaAll,
        "b0": b0,
        "b1": b1,
        "rmse": rmse,
        "n_cells": n_cells,
        "delta": delta,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Part B — 4-state arm reach with PPAF
# ──────────────────────────────────────────────────────────────────────────────


def _simulate_reach_minjerk(delta, T_total):
    """Simulate a 2-D minimum-jerk reach from x0 to xT.

    Uses MATLAB's cosine-acceleration dynamics:
        xState(:,k) = A * xState(:,k-1) + (delta/2)*(pi/T)^2 * cos(pi*t/T)
                       * [0; 0; xT(1)-x0(1); xT(2)-x0(2)]

    Returns time, xState (4×T), A (4×4).
    """
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    xT_target = np.array([-0.35, 0.2, 0.0, 0.0])
    time = np.arange(0.0, T_total + delta, delta)
    T = len(time)

    A = np.array([
        [1, 0, delta, 0],
        [0, 1, 0, delta],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)

    xState = np.zeros((4, T), dtype=float)
    xState[:, 0] = x0
    accel_dir = np.array([0.0, 0.0, xT_target[0] - x0[0], xT_target[1] - x0[1]])
    for k in range(1, T):
        accel = (delta / 2.0) * (np.pi / T_total) ** 2 * np.cos(np.pi * time[k] / T_total)
        xState[:, k] = A @ xState[:, k - 1] + accel * accel_dir

    return time, xState, A


def _run_part_b(seed=0, n_cells=20, n_sims=20):
    """Compare PPAF free vs goal-directed decoding for arm reach.

    Matches MATLAB: single trajectory, 20 re-randomized encoding simulations.
    """
    rng = np.random.default_rng(seed)
    delta = 0.001  # 1 ms bins (matches MATLAB)
    T_total = 2.0  # 2-second reach

    # ── Generate minimum-jerk reach trajectory ──
    time, xState, A = _simulate_reach_minjerk(delta, T_total)
    T = xState.shape[1]
    ns = 4

    # Target = final state
    yT = xState[:, -1].copy()

    # Q from trajectory variance (MATLAB: diag(var(diff(xState,[],2),[],2))*100)
    Q = np.diag(np.var(np.diff(xState, axis=1), axis=1)) * 100.0

    # Initial/target covariances (MATLAB: very tight)
    r, p = 1e-6, 1e-6
    pi0 = np.diag([r, r, p, p])
    piT = np.diag([r, r, p, p])

    # ── Run 20 repeated simulations ──
    # Same trajectory, re-randomized encoding + spikes each time
    all_runs_goal = []
    all_runs_free = []
    example_run = None

    for sim_idx in range(n_sims):
        # MATLAB: bCoeffs = 10*(rand(numCells,2)-0.5);  Uniform[-5,5]
        bCoeffs = 10.0 * (rng.random((n_cells, 2)) - 0.5)
        # MATLAB: muCoeffs = log(10*delta) + randn(numCells,1)
        muCoeffs = np.log(10.0 * delta) + rng.standard_normal(n_cells)

        # beta: 4×C with zeros for position, bCoeffs for velocity
        beta = np.zeros((ns, n_cells), dtype=float)
        beta[2, :] = bCoeffs[:, 0]  # vx tuning
        beta[3, :] = bCoeffs[:, 1]  # vy tuning

        # Simulate spikes
        dN = _simulate_binomial_spikes(xState, muCoeffs, beta, rng)

        # Initial state
        x0 = np.array([0.0, 0.0, 0.0, 0.0])

        # --- Goal-directed decode ---
        _, _, x_u_goal, _, _, _, _, _ = DecodingAlgorithms.PPDecodeFilterLinear(
            A, Q, dN, muCoeffs, beta, "binomial", delta,
            None, None, x0, pi0, yT, piT, 0
        )

        # --- Free decode (no goal) ---
        _, _, x_u_free, _, _, _, _, _ = DecodingAlgorithms.PPDecodeFilterLinear(
            A, Q, dN, muCoeffs, beta, "binomial", delta,
            None, None, x0,
        )

        all_runs_goal.append(x_u_goal)
        all_runs_free.append(x_u_free)

        if sim_idx == 0:
            example_run = {
                "time": time,
                "xState": xState,
                "dN": dN,
                "muCoeffs": muCoeffs,
                "bCoeffs": bCoeffs,
                "beta": beta,
            }

    return {
        "all_runs_goal": all_runs_goal,
        "all_runs_free": all_runs_free,
        "example": example_run,
        "n_cells": n_cells,
        "n_sims": n_sims,
        "xState": xState,
        "time": time,
        "delta": delta,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Part C — Hybrid filter
# ──────────────────────────────────────────────────────────────────────────────


def _load_hybrid_fixture():
    """Load the MATLAB hybrid filter fixture (paperHybridFilterExample.mat).

    Returns a dict with: time, delta, X (6×T), mstate (T,),
    A (list of 2), Q (list of 2), p_ij (2×2), Px0 (list of 2), ind.
    """
    import scipy.io as sio

    # Search for fixture in multiple locations
    candidates = [
        REPO_ROOT / "nstat" / "data" / "paperHybridFilterExample.mat",
        REPO_ROOT / "helpfiles" / "paperHybridFilterExample.mat",
    ]
    mat_path = None
    for p in candidates:
        if p.exists() and p.stat().st_size > 200:  # skip LFS pointers
            mat_path = p
            break

    if mat_path is None:
        raise FileNotFoundError(
            "Cannot find paperHybridFilterExample.mat fixture. "
            "Ensure it is in nstat/data/ or helpfiles/."
        )

    f = sio.loadmat(str(mat_path))
    time = f["time"].ravel().astype(float)
    delta = float(f["delta"].ravel()[0])
    X = f["X"].astype(float)  # (6, T)
    mstate = f["mstate"].ravel().astype(int)  # (T,), values 1 or 2
    p_ij = f["p_ij"].astype(float)  # (2, 2)

    # Cell arrays → Python lists
    A_cell = f["A"]
    Q_cell = f["Q"]
    Px0_cell = f["Px0"]
    ind_cell = f["ind"]

    A = [A_cell[0, i].astype(float) for i in range(A_cell.shape[1])]
    Q = [Q_cell[0, i].astype(float) for i in range(Q_cell.shape[1])]
    Px0 = [Px0_cell[0, i].astype(float) for i in range(Px0_cell.shape[1])]
    # ind: convert from MATLAB 1-indexed to Python 0-indexed
    ind = [ind_cell[0, i].ravel().astype(int) - 1 for i in range(ind_cell.shape[1])]

    return {
        "setup": setup_data,
        "time": time,
        "delta": delta,
        "X": X,
        "mstate": mstate,
        "A": A,
        "Q": Q,
        "p_ij": p_ij,
        "Px0": Px0,
        "ind": ind,
    }


def _run_part_c(seed=0, n_cells=40, n_sims=20):
    """PPHybridFilterLinear: joint discrete/continuous state decoding.

    Loads fixture trajectory, runs 20 simulations with re-randomized encoding,
    comparing goal-directed vs free hybrid decoding.
    """
    rng = np.random.default_rng(seed)

    # ── Load fixture ──
    fix = _load_hybrid_fixture()
    time = fix["time"]
    delta = fix["delta"]
    X = fix["X"]  # (6, T)
    mstate = fix["mstate"]  # (T,)
    p_ij = fix["p_ij"]
    A_models = fix["A"]  # [A_hold (2×2), A_reach (6×6)]
    Q_models_orig = fix["Q"]  # [Q_hold (2×2), Q_reach (6×6)]
    Px0 = fix["Px0"]  # [Px0_hold (2×2), Px0_reach (6×6)]
    ind = fix["ind"]  # [[0,1], [0,1,2,3,4,5]]
    T = X.shape[1]

    # ── Recompute Q from trajectory variance (matching MATLAB) ──
    nonMovingInd = np.where((X[4, :] == 0) & (X[5, :] == 0))[0]
    movingInd = np.setdiff1d(np.arange(T), nonMovingInd)

    Q_reach = np.diag(np.var(np.diff(X[:, movingInd], axis=1), axis=1))
    Q_reach[:4, :4] = 0.0  # Zero out pos/vel noise; only accel has noise
    varNV = np.diag(np.var(np.diff(X[:, nonMovingInd], axis=1), axis=1))
    Q_hold = varNV[:2, :2]

    Q_models = [Q_hold, Q_reach]

    # State dimensions
    dim_hold = A_models[0].shape[0]  # 2
    dim_reach = A_models[1].shape[0]  # 6

    # ── Run 20 repeated simulations ──
    X_estAll = np.zeros((dim_reach, T, n_sims), dtype=float)
    X_estNTAll = np.zeros((dim_reach, T, n_sims), dtype=float)
    S_estAll = np.zeros((n_sims, T), dtype=int)
    S_estNTAll = np.zeros((n_sims, T), dtype=int)
    MU_estAll = np.zeros((2, T, n_sims), dtype=float)
    MU_estNTAll = np.zeros((2, T, n_sims), dtype=float)
    example_dN = None

    for n in range(n_sims):
        # MATLAB: muCoeffs = log(10*delta) + randn(numCells,1)
        muCoeffs = np.log(10.0 * delta) + rng.standard_normal(n_cells)
        # MATLAB: coeffs = [muCoeffs, zeros(C,2), 10*(rand(C,2)-0.5), zeros(C,2)]
        # = [mu, 0, 0, b_vx, b_vy, 0, 0] — tuned to velocities (states 3-4)
        bCoeffs_vx = 10.0 * (rng.random(n_cells) - 0.5)
        bCoeffs_vy = 10.0 * (rng.random(n_cells) - 0.5)

        # Full beta: 6×C matrix
        beta_full = np.zeros((6, n_cells), dtype=float)
        beta_full[2, :] = bCoeffs_vx  # vx tuning
        beta_full[3, :] = bCoeffs_vy  # vy tuning

        # Simulate spikes from full state trajectory
        dN = _simulate_binomial_spikes(X, muCoeffs, beta_full, rng)

        if n == 0:
            example_dN = dN.copy()

        # ── Initial conditions per mode ──
        x0_list = [X[ind[0], 0], X[ind[1], 0]]
        Pi0_list = Px0

        # ── Target conditions per mode ──
        yT_list = [X[ind[0], -1], X[ind[1], -1]]
        piT_list = [1e-9 * np.eye(dim_hold), 1e-9 * np.eye(dim_reach)]

        # beta per mode: hold uses 2-dim subset, reach uses full 6-dim
        beta_hold = beta_full[ind[0], :]  # (2, C) — will be zeros
        beta_reach = beta_full[ind[1], :]  # (6, C) — has vx/vy tuning

        mu0 = np.array([0.5, 0.5])

        # --- Goal-directed hybrid decode ---
        S_est, X_est, _, MU_est, _, _, _ = DecodingAlgorithms.PPHybridFilterLinear(
            A_models, Q_models, p_ij, mu0, dN,
            [muCoeffs, muCoeffs],
            [beta_hold, beta_reach],
            "binomial", delta, None, None,
            x0_list, Pi0_list,
            yT_list, piT_list,
        )

        # --- Free hybrid decode (no target) ---
        S_estNT, X_estNT, _, MU_estNT, _, _, _ = DecodingAlgorithms.PPHybridFilterLinear(
            A_models, Q_models, p_ij, mu0, dN,
            [muCoeffs, muCoeffs],
            [beta_hold, beta_reach],
            "binomial", delta, None, None,
            x0_list, Pi0_list,
        )

        X_estAll[:, :, n] = X_est
        X_estNTAll[:, :, n] = X_estNT
        S_estAll[n, :] = S_est
        S_estNTAll[n, :] = S_estNT
        MU_estAll[:, :, n] = MU_est
        MU_estNTAll[:, :, n] = MU_estNT

        print(f"  Hybrid sim {n + 1}/{n_sims} done")

    return {
        "time": time,
        "X": X,
        "mstate": mstate,
        "dN": example_dN,
        "X_estAll": X_estAll,
        "X_estNTAll": X_estNTAll,
        "S_estAll": S_estAll,
        "S_estNTAll": S_estNTAll,
        "MU_estAll": MU_estAll,
        "MU_estNTAll": MU_estNTAll,
        "n_cells": n_cells,
        "n_sims": n_sims,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────────────────────


def _plot_part_a(result):
    """Figure 1: stimulus + CIF + raster (3×1). Figure 2: decoded vs true."""
    time = result["time"]
    stimSignal = result["stimSignal"]
    dN = result["dN"]
    n_cells = result["n_cells"]

    # ── Figure 1: stimulus, CIF, spike raster (3 panels, matching MATLAB) ──
    fig1, axes1 = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    # (3,1,1): Driving stimulus
    axes1[0].plot(time, stimSignal, "k", linewidth=1.5)
    axes1[0].set_ylabel("Stimulus")
    axes1[0].set_title("Driving Stimulus", fontweight="bold", fontsize=14,
                        fontfamily="Arial")
    axes1[0].tick_params(labelbottom=False)

    # (3,1,2): CIFs overlaid in black
    lambdaAll = result["lambdaAll"]
    for c in range(n_cells):
        axes1[1].plot(time, lambdaAll[c, :], "k", linewidth=0.5)
    axes1[1].set_ylabel("Firing Rate [spikes/sec]")
    axes1[1].set_title("Conditional Intensity Functions", fontweight="bold",
                        fontsize=14, fontfamily="Arial")
    axes1[1].tick_params(labelbottom=False)

    # (3,1,3): Spike raster
    for c in range(n_cells):
        spike_t = time[dN[c, :] > 0]
        axes1[2].plot(spike_t, np.full_like(spike_t, c + 1), "|", color="k",
                      markersize=2)
    axes1[2].set_ylabel("Cell Number")
    axes1[2].set_xlabel("time [s]")
    axes1[2].set_ylim(0.5, n_cells + 0.5)
    axes1[2].set_yticks(np.arange(0, n_cells + 1, 10))
    axes1[2].set_title("Point Process Sample Paths", fontweight="bold",
                        fontsize=14, fontfamily="Arial")
    fig1.tight_layout()

    # ── Figure 2: Decoding results (MATLAB: black=decoded, blue=actual) ──
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 9))
    ax2.fill_between(
        time, result["ci_low"], result["ci_high"],
        color="0.75", alpha=0.4, label="95% CI"
    )
    ax2.plot(time, result["x_decoded"], "k-", linewidth=4, label="Decoded")
    ax2.plot(time, x_true, "b-", linewidth=4, label="Actual")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("")
    ax2.set_title(f"Decoded Stimulus $\\pm$ 95% CIs with {result['n_cells']} cells",
                  fontweight="bold", fontsize=18, fontfamily="Arial")
    ax2.legend(["Decoded", "Actual"], loc="upper right")
    fig2.tight_layout()

    return fig1, fig2


def _plot_part_b(result):
    """Figure 3: Reach setup (4×2 layout). Figure 4: Overlaid decoded trajectories."""
    ex = result["example"]
    time = ex["time"]
    xState = ex["xState"]
    dN = ex["dN"]
    delta = result["delta"]
    n_cells = result["n_cells"]

    # ── Figure 3: Reach trajectory and population setup (4×2 layout) ──
    fig3 = plt.figure(figsize=(14, 9))

    # Top-left [1,3]: 2D reach path (in cm)
    ax_path = fig3.add_subplot(4, 2, (1, 3))
    ax_path.plot(100 * xState[0, :], 100 * xState[1, :], "k", linewidth=2)
    ax_path.plot(100 * xState[0, 0], 100 * xState[1, 0], "bo", markersize=14)
    ax_path.plot(100 * xState[0, -1], 100 * xState[1, -1], "ro", markersize=14)
    ax_path.legend(["Path", "Start", "Finish"], loc="upper right")
    ax_path.set_xlabel("X Position [cm]")
    ax_path.set_ylabel("Y Position [cm]")
    ax_path.set_title("Reach Path", fontweight="bold", fontsize=14)

    # Middle-left [5]: position vs time (in cm)
    ax_pos = fig3.add_subplot(4, 2, 5)
    h1, = ax_pos.plot(time, 100 * xState[0, :], "k", linewidth=2)
    h2, = ax_pos.plot(time, 100 * xState[1, :], "k-.", linewidth=2)
    ax_pos.legend([h1, h2], ["x", "y"], loc="upper right")
    ax_pos.set_xlabel("time [s]")
    ax_pos.set_ylabel("Position [cm]")

    # Lower-left [7]: velocity vs time (in cm/s)
    ax_vel = fig3.add_subplot(4, 2, 7)
    h1, = ax_vel.plot(time, 100 * xState[2, :], "k", linewidth=2)
    h2, = ax_vel.plot(time, 100 * xState[3, :], "k-.", linewidth=2)
    ax_vel.legend([h1, h2], ["$v_x$", "$v_y$"], loc="upper right")
    ax_vel.set_xlabel("time [s]")
    ax_vel.set_ylabel("Velocity [cm/s]")

    # Top-right [2,4]: neural raster
    ax_raster = fig3.add_subplot(4, 2, (2, 4))
    for c in range(n_cells):
        spike_t = time[dN[c, :] > 0]
        ax_raster.plot(spike_t, np.full_like(spike_t, c + 1), "|", color="k", markersize=2)
    ax_raster.set_ylabel("Cell Number")
    ax_raster.set_xticks([])
    ax_raster.set_xticklabels([])
    ax_raster.set_title("Neural Raster", fontweight="bold", fontsize=14)

    # Bottom-right [6,8]: CIF curves
    ax_cif = fig3.add_subplot(4, 2, (6, 8))
    muCoeffs = ex["muCoeffs"]
    beta = ex["beta"]
    for c in range(n_cells):
        eta = muCoeffs[c] + beta[:, c] @ xState
        exp_eta = np.exp(np.clip(eta, -20, 20))
        lam = (exp_eta / (1.0 + exp_eta)) / delta
        ax_cif.plot(time, lam, "k", linewidth=0.5)
    ax_cif.set_title("Neural Conditional Intensity Functions",
                     fontweight="bold", fontsize=14)
    ax_cif.set_xlabel("time [s]")
    ax_cif.set_ylabel("Firing Rate [spikes/sec]")

    fig3.tight_layout()

    # ── Figure 4: Overlaid decoded trajectories (4×2 layout, 20 runs) ──
    fig4 = plt.figure(figsize=(14, 9))

    # Top [1:4]: 2D estimated vs actual reach paths
    ax_2d = fig4.add_subplot(4, 2, (1, 4))
    ax_2d.plot(100 * xState[0, :], 100 * xState[1, :], "k", linewidth=3)
    ax_2d.set_title("Estimated vs. Actual Reach Paths",
                    fontweight="bold", fontsize=12)

    for sim_idx in range(result["n_sims"]):
        x_u_goal = result["all_runs_goal"][sim_idx]
        x_u_free = result["all_runs_free"][sim_idx]
        ax_2d.plot(100 * x_u_goal[0, :], 100 * x_u_goal[1, :], "b", linewidth=0.5)
        ax_2d.plot(100 * x_u_free[0, :], 100 * x_u_free[1, :], "g", linewidth=0.5)
    ax_2d.set_xlabel("x [cm]")
    ax_2d.set_ylabel("y [cm]")

    # Bottom panels: per-state traces
    state_labels = ["x(t) [cm]", "y(t) [cm]", "$v_x$(t) [cm/s]", "$v_y$(t) [cm/s]"]
    subplot_indices = [5, 6, 7, 8]
    scale = 100.0  # meters → cm

    for d, (sp_idx, ylabel) in enumerate(zip(subplot_indices, state_labels)):
        ax = fig4.add_subplot(4, 2, sp_idx)
        ax.plot(time, scale * xState[d, :], "k", linewidth=3)

        for sim_idx in range(result["n_sims"]):
            x_u_goal = result["all_runs_goal"][sim_idx]
            x_u_free = result["all_runs_free"][sim_idx]
            hB, = ax.plot(time, scale * x_u_goal[d, :], "b", linewidth=0.5)
            hC, = ax.plot(time, scale * x_u_free[d, :], "g", linewidth=0.5)

        ax.set_ylabel(ylabel)
        if d >= 2:
            ax.set_xlabel("time [s]")
        else:
            ax.set_xticklabels([])

        # Add legend on y(t) panel (subplot 6), matching MATLAB
        if d == 1:
            hA, = ax.plot([], [], "k", linewidth=3)
            ax.legend([hA, hB, hC], ["Actual", "PPAF+Goal", "PPAF"],
                      loc="lower right", fontsize=8)

    fig4.tight_layout()

    return fig3, fig4


def _plot_part_c(result):
    """Figure 5: Hybrid setup (4×2 layout). Figure 6: Hybrid decoding summary (4×3)."""
    time = result["time"]
    X = result["X"]  # (6, T)
    mstate = result["mstate"]
    dN = result["dN"]
    n_cells = result["n_cells"]

    # ── Figure 5: Setup — reach path, traces, raster, discrete state (4×2) ──
    fig5 = plt.figure(figsize=(14, 9))

    # Top-left [1,3]: 2D reach path
    ax_path = fig5.add_subplot(4, 2, (1, 3))
    ax_path.plot(100 * X[0, :], 100 * X[1, :], "k", linewidth=2)
    ax_path.plot(100 * X[0, 0], 100 * X[1, 0], "bo", markersize=16)
    ax_path.plot(100 * X[0, -1], 100 * X[1, -1], "ro", markersize=16)
    ax_path.set_xlabel("X [cm]")
    ax_path.set_ylabel("Y [cm]")
    ax_path.set_title("Reach Path", fontweight="bold", fontsize=14)

    # Middle-left [5]: position vs time
    ax_pos = fig5.add_subplot(4, 2, 5)
    h1, = ax_pos.plot(time, 100 * X[0, :], "k", linewidth=2)
    h2, = ax_pos.plot(time, 100 * X[1, :], "k-.", linewidth=2)
    ax_pos.legend([h1, h2], ["x", "y"], loc="upper right")
    ax_pos.set_xlabel("time [s]")
    ax_pos.set_ylabel("Position [cm]")

    # Lower-left [7]: velocity vs time
    ax_vel = fig5.add_subplot(4, 2, 7)
    h1, = ax_vel.plot(time, 100 * X[2, :], "k", linewidth=2)
    h2, = ax_vel.plot(time, 100 * X[3, :], "k-.", linewidth=2)
    ax_vel.legend([h1, h2], ["$v_x$", "$v_y$"], loc="upper right")
    ax_vel.set_xlabel("time [s]")
    ax_vel.set_ylabel("Velocity [cm/s]")

    # Top-right [2,4]: neural raster (show ALL cells, matching MATLAB)
    ax_raster = fig5.add_subplot(4, 2, (2, 4))
    for c in range(dN.shape[0]):
        spike_t = time[dN[c, :] > 0]
        ax_raster.plot(spike_t, np.full_like(spike_t, c + 1), "|", color="k", markersize=2)
    ax_raster.set_ylabel("Cell Number")
    ax_raster.set_yticklabels([])
    ax_raster.set_xticks([])
    ax_raster.set_xticklabels([])
    ax_raster.set_title("Neural Raster", fontweight="bold", fontsize=14)

    # Bottom-right [6,8]: discrete movement state
    ax_state = fig5.add_subplot(4, 2, (6, 8))
    ax_state.plot(time, mstate, "k", linewidth=2)
    ax_state.set_ylim(0, 3)
    ax_state.set_yticks([1, 2])
    ax_state.set_yticklabels(["N", "M"])
    ax_state.set_xlabel("time [s]")
    ax_state.set_ylabel("state")
    ax_state.set_title("Discrete Movement State", fontweight="bold", fontsize=14)

    fig5.tight_layout()

    # ── Figure 6: Hybrid decoding results (4×3 layout, averaged over 20 sims) ──
    fig6 = plt.figure(figsize=(14, 9))

    # Mean across simulations
    mS_est = np.mean(result["S_estAll"], axis=0)
    mS_estNT = np.mean(result["S_estNTAll"], axis=0)
    mMU_est = np.mean(result["MU_estAll"][1, :, :], axis=1)   # P(M|data) for goal
    mMU_estNT = np.mean(result["MU_estNTAll"][1, :, :], axis=1)  # P(M|data) for free
    mX_est = np.mean(100 * result["X_estAll"], axis=2)
    mX_estNT = np.mean(100 * result["X_estNTAll"], axis=2)

    # Left column: state estimation + probability
    # [1,4]: Estimated vs actual state
    ax_s = fig6.add_subplot(4, 3, (1, 4))
    ax_s.plot(time, mstate, "k", linewidth=3)
    ax_s.plot(time, mS_est, "b", linewidth=3)
    ax_s.plot(time, mS_estNT, "g", linewidth=3)
    ax_s.set_yticks([1, 2.1])
    ax_s.set_yticklabels(["N", "M"])
    ax_s.set_xticklabels([])
    ax_s.set_ylabel("state")
    ax_s.set_title("Estimated vs. Actual State", fontweight="bold", fontsize=12)

    # [7,10]: P(s(t)=M | data)
    ax_prob = fig6.add_subplot(4, 3, (7, 10))
    ax_prob.plot(time, mMU_est, "b", linewidth=3)
    ax_prob.plot(time, mMU_estNT, "g", linewidth=3)
    ax_prob.set_xlim(time[0], time[-1])
    ax_prob.set_ylim(0, 1.1)
    ax_prob.set_xlabel("time [s]")
    ax_prob.set_ylabel("P(s(t)=M | data)")
    ax_prob.set_title("Probability of State", fontweight="bold", fontsize=12)

    # Right top [2,3,5,6]: 2D estimated vs actual reach path
    ax_2d = fig6.add_subplot(4, 3, (2, 6))
    ax_2d.plot(100 * X[0, :], 100 * X[1, :], "k", linewidth=1)
    ax_2d.plot(mX_est[0, :], mX_est[1, :], "b", linewidth=3)
    ax_2d.plot(mX_estNT[0, :], mX_estNT[1, :], "g", linewidth=3)
    ax_2d.plot(100 * X[0, 0], 100 * X[1, 0], "bo", markersize=14)
    ax_2d.plot(100 * X[0, -1], 100 * X[1, -1], "ro", markersize=14)
    ax_2d.set_xlabel("x [cm]")
    ax_2d.set_ylabel("y [cm]")
    ax_2d.set_title("Estimated vs. Actual Reach Path",
                    fontweight="bold", fontsize=12)

    # Bottom panels: per-state traces
    # [8]: x(t)
    ax_x = fig6.add_subplot(4, 3, 8)
    ax_x.plot(time, 100 * X[0, :], "k", linewidth=3)
    ax_x.plot(time, mX_est[0, :], "b", linewidth=3)
    ax_x.plot(time, mX_estNT[0, :], "g", linewidth=3)
    ax_x.set_ylabel("x(t) [cm]")
    ax_x.set_xticklabels([])
    ax_x.set_title("X Position", fontweight="bold", fontsize=12)

    # [9]: y(t) with legend
    ax_y = fig6.add_subplot(4, 3, 9)
    h1, = ax_y.plot(time, 100 * X[1, :], "k", linewidth=3)
    h2, = ax_y.plot(time, mX_est[1, :], "b", linewidth=3)
    h3, = ax_y.plot(time, mX_estNT[1, :], "g", linewidth=3)
    ax_y.legend([h1, h2, h3], ["Actual", "PPAF+Goal", "PPAF"],
                loc="lower right", fontsize=8)
    ax_y.set_ylabel("y(t) [cm]")
    ax_y.set_xticklabels([])
    ax_y.set_title("Y Position", fontweight="bold", fontsize=12)

    # [11]: vx(t)
    ax_vx = fig6.add_subplot(4, 3, 11)
    ax_vx.plot(time, 100 * X[2, :], "k", linewidth=3)
    ax_vx.plot(time, mX_est[2, :], "b", linewidth=3)
    ax_vx.plot(time, mX_estNT[2, :], "g", linewidth=3)
    ax_vx.set_ylabel("$v_x$(t) [cm/s]")
    ax_vx.set_xlabel("time [s]")
    ax_vx.set_title("X Velocity", fontweight="bold", fontsize=12)

    # [12]: vy(t)
    ax_vy = fig6.add_subplot(4, 3, 12)
    ax_vy.plot(time, 100 * X[3, :], "k", linewidth=3)
    ax_vy.plot(time, mX_est[3, :], "b", linewidth=3)
    ax_vy.plot(time, mX_estNT[3, :], "g", linewidth=3)
    ax_vy.set_ylabel("$v_y$(t) [cm/s]")
    ax_vy.set_xlabel("time [s]")
    ax_vy.set_title("Y Velocity", fontweight="bold", fontsize=12)

    fig6.tight_layout()

    return fig5, fig6


# ──────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ──────────────────────────────────────────────────────────────────────────────


def run_example05(*, export_figures=False, export_dir=None, show=False):
    """Run Example 05: PPAF and PPHF decoding.

    Mirrors MATLAB ``example05_decoding_ppaf_pphf.m``:

    Part A — Univariate stimulus decoding (Figs 1–2):
      1. 20-cell sinusoidal-tuned population, binomial CIF.
      2. PPDecodeFilterLinear decoding with 95% CIs.

    Part B — Arm-reach PPAF:
      4. Simulate 4-state minimum-jerk reaching movement.
      5. Encode with 20-cell velocity-tuned population.
      6. Decode with PPAF (free) and PPAF+Goal; 20 overlaid simulations.

    Part C — Hybrid filter:
      7. Load 6-state fixture trajectory with 2 discrete modes.
      8. Simulate 40-cell population with velocity tuning.
      9. Decode joint discrete/continuous state via PPHybridFilterLinear
         (both goal-directed and free), averaged over 20 simulations.
    """
    print("=" * 70)
    print("Example 05: Stimulus Decoding with PPAF and PPHF")
    print("=" * 70)

    # --- Part A ---
    print("\n--- Part A: Univariate Sinusoidal Stimulus ---")
    result_a = _run_part_a()
    print(f"  {result_a['n_cells']} cells, decode RMSE = {result_a['rmse']:.4f}")

    # --- Part B ---
    print("\n--- Part B: Arm Reach PPAF (20 simulations) ---")
    result_b = _run_part_b()
    print(f"  {result_b['n_sims']} simulations, {result_b['n_cells']} cells")

    # --- Part C: Hybrid filter ---
    print("\n--- Part C: Hybrid Filter (20 simulations) ---")
    result_c = _run_part_c()
    print(f"  {result_c['n_cells']} cells, {result_c['n_sims']} simulations")

    # Summary
    summary = {
        "experiment5": {
            "num_cells": float(result_a["n_cells"]),
            "decode_rmse": result_a["rmse"],
        },
        "experiment5b": {
            "num_cells": float(result_b["n_cells"]),
            "n_sims": float(result_b["n_sims"]),
        },
        "experiment6": {
            "num_cells": float(result_c["n_cells"]),
            "n_sims": float(result_c["n_sims"]),
        },
    }
    print("\n" + json.dumps(summary, indent=2))

    # --- Figures ---
    fig1, fig2 = _plot_part_a(result_a)
    fig3, fig4 = _plot_part_b(result_b)
    fig5, fig6 = _plot_part_c(result_c)
    figures = [fig1, fig2, fig3, fig4, fig5, fig6]

    if export_figures:
        if export_dir is None:
            export_dir = THIS_DIR / "figures" / "example05"
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        fig_names = [
            "fig01_univariate_setup", "fig02_univariate_decoding",
            "fig03_reach_and_population_setup", "fig04_ppaf_goal_vs_free",
            "fig05_hybrid_setup", "fig06_hybrid_decoding_summary",
        ]
        for i, fig in enumerate(figures):
            path = export_dir / f"{fig_names[i]}.png"
            fig.savefig(path, dpi=250, facecolor="w", edgecolor="none")
            print(f"  Saved: {path}")

    if show:
        plt.show()
    else:
        plt.close("all")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example 05: Stimulus Decoding With PPAF and PPHF"
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--export-figures", action="store_true")
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--show", action="store_true",
                        help="Display figures interactively")
    args = parser.parse_args()

    result = run_example05(
        export_figures=args.export_figures,
        export_dir=args.export_dir,
        show=args.show,
    )
    if args.output_json:
        args.output_json.write_text(json.dumps(result, indent=2),
                                    encoding="utf-8")
