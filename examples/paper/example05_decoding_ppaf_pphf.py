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
  4. Simulate reaching trajectories (position + velocity, 4-D state).
  5. Encode with 20-cell cosine-tuning population (binomial CIF).
  6. Decode with PPAF (free) and PPAF + Goal; compare across 20 simulations.

Part C — Hybrid Filter (Figures 5–6):
  7. Simulate 40-cell population with 2 discrete reach-states (rest / reach)
     that modulate baseline firing rate, plus velocity-tuned continuous state.
  8. Decode joint discrete + continuous state via ``PPHybridFilterLinear``.

Paper mapping:
  Section 2.5 (point-process adaptive filter) and Section 2.6 (hybrid filter).

Expected outputs:
  - Figure 1: CIF tuning curves and simulated spike raster.
  - Figure 2: Decoded stimulus vs true (with 95% confidence band).
  - Figure 3: Reach trajectory and population spike raster.
  - Figure 4: PPAF comparison (free vs goal-informed, 20 runs box plot).
  - Figure 5: Hybrid filter setup (state sequence, spike raster).
  - Figure 6: Hybrid decoding results (state probabilities, decoded kinematics).
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


def _simulate_binomial_spikes(x, mu, beta, rng):
    """Simulate spikes from binomial CIF: p_c = sigmoid(mu_c + beta_c @ x).

    Parameters
    ----------
    x : (ns, T) array — stimulus/state trajectory
    mu : (C,) array — baseline log-odds per cell
    beta : (ns, C) array — tuning coefficients
    rng : numpy Generator

    Returns
    -------
    dN : (C, T) array — binary spike indicators
    """
    ns, T = x.shape
    C = mu.size
    dN = np.zeros((C, T), dtype=float)
    for t in range(T):
        eta = mu + beta.T @ x[:, t]  # (C,)
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -20.0, 20.0)))
        dN[:, t] = (rng.random(C) < p).astype(float)
    return dN


# ──────────────────────────────────────────────────────────────────────────────
#  Part A — Univariate sinusoidal stimulus
# ──────────────────────────────────────────────────────────────────────────────


def _run_part_a(seed=11, n_cells=20):
    """Encode/decode a 1-D sinusoidal stimulus with 20-cell binomial CIF."""
    rng = np.random.default_rng(seed)
    delta = 0.001  # 1 ms bins
    time = np.arange(0.0, 1.0 + delta, delta)
    T = len(time)

    # True stimulus: sinusoidal
    x_true = np.sin(2.0 * np.pi * 2.0 * time)  # (T,)

    # ── Encoding model: logistic CIF ──
    # MATLAB: b0 = log(10*delta) + randn(C,1);  b1 = randn(C,1);
    b0 = np.log(10.0 * delta) + rng.standard_normal(n_cells)
    b1 = rng.standard_normal(n_cells)

    # Simulate spikes
    x_2d = x_true.reshape(1, -1)  # (1, T) — scalar state
    beta = b1.reshape(1, -1)  # (1, C) — stimulus coefficients
    dN = _simulate_binomial_spikes(x_2d, b0, beta, rng)

    # ── State-space model ──
    # x(t+1) = A * x(t) + w,  w ~ N(0, Q)
    # MATLAB: Q = std(stim.data(2:end) - stim.data(1:end-1));  A = 1;
    A = np.array([[1.0]])
    Q_val = float(np.std(np.diff(x_true)))
    Q = np.array([[Q_val]])
    x0 = np.array([0.0])
    Pi0 = 0.5 * np.eye(1)

    # ── Decode with PPDecodeFilterLinear ──
    # dN is (C, T) — the API expects (num_cells, num_steps)
    x_p, W_p, x_u, W_u, _, _, _, _ = DecodingAlgorithms.PPDecodeFilterLinear(
        A, Q, dN, b0, beta, "binomial", delta, None, None, x0, Pi0
    )

    # Extract decoded signal and 95% CI (±1.96σ, matching MATLAB zVal=1.96)
    x_decoded = x_u[0, :]  # (T,)
    sigma = np.sqrt(np.maximum(W_u[0, 0, :], 0.0))
    z_val = 1.96
    ci_low = np.minimum(x_decoded - z_val * sigma, x_decoded + z_val * sigma)
    ci_high = np.maximum(x_decoded - z_val * sigma, x_decoded + z_val * sigma)
    rmse = float(np.sqrt(np.mean((x_decoded - x_true) ** 2)))

    return {
        "time": time,
        "x_true": x_true,
        "x_decoded": x_decoded,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "dN": dN,
        "b0": b0,
        "b1": b1,
        "rmse": rmse,
        "n_cells": n_cells,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Part B — 4-state arm reach with PPAF
# ──────────────────────────────────────────────────────────────────────────────


def _simulate_reach(delta, T_total, rng):
    """Simulate a 2-D reaching trajectory with 4-D state [x, y, vx, vy].

    Uses a simple sinusoidal trajectory to mimic a reaching task.
    """
    time = np.arange(0.0, T_total + delta, delta)
    T = len(time)

    # Smooth trajectory
    x_pos = 0.25 * np.sin(2.0 * np.pi * 0.15 * time)
    y_pos = 0.20 * np.cos(2.0 * np.pi * 0.10 * time)
    vx = np.gradient(x_pos, delta)
    vy = np.gradient(y_pos, delta)

    state = np.vstack([x_pos, y_pos, vx, vy])  # (4, T)
    return time, state


def _run_part_b(seed=19, n_cells=20, n_sims=20):
    """Compare PPAF free vs goal-directed decoding for arm reach."""
    rng = np.random.default_rng(seed)
    delta = 0.01  # 10 ms bins
    ns = 4  # state dimension

    # State-space model (constant-velocity kinematic model)
    A = np.array([
        [1, 0, delta, 0],
        [0, 1, 0, delta],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    Q = 0.001 * np.eye(ns, dtype=float)

    # Encoding model: cosine tuning to velocity
    # mu_c ~ N(-3.0, 0.2)
    # beta_c = [0, 0, w_vx, w_vy] — velocity tuned
    b0 = rng.normal(-3.0, 0.2, n_cells)
    beta = np.zeros((ns, n_cells), dtype=float)
    for c in range(n_cells):
        beta[2, c] = 3.0 * rng.normal(0.0, 1.0)  # vx weight
        beta[3, c] = 3.0 * rng.normal(0.0, 1.0)  # vy weight

    # Run multiple simulations to compare free vs goal-directed
    rmse_free = np.zeros((n_sims, ns), dtype=float)
    rmse_goal = np.zeros((n_sims, ns), dtype=float)

    # Store one example run for plotting
    example_run = None

    for sim_idx in range(n_sims):
        sim_rng = np.random.default_rng(seed + sim_idx + 100)
        time, state = _simulate_reach(delta, 10.0, sim_rng)
        T = state.shape[1]

        # Simulate spikes
        dN = _simulate_binomial_spikes(state, b0, beta, sim_rng)

        # Initial conditions
        x0 = state[:, 0]
        Pi0 = 0.1 * np.eye(ns)

        # --- Free decode (no goal) ---
        x_p_free, _, x_u_free, W_u_free, _, _, _, _ = (
            DecodingAlgorithms.PPDecodeFilterLinear(
                A, Q, dN, b0, beta, "binomial", delta,
                None, None, x0, Pi0
            )
        )

        # --- Goal-directed decode ---
        yT = state[:, -1]  # target = final state
        PiT = 0.01 * np.eye(ns)  # tight target uncertainty
        x_p_goal, _, x_u_goal, W_u_goal, _, _, _, _ = (
            DecodingAlgorithms.PPDecodeFilterLinear(
                A, Q, dN, b0, beta, "binomial", delta,
                None, None, x0, Pi0, yT, PiT, 0
            )
        )

        # Compute RMSE per state dimension
        for d in range(ns):
            rmse_free[sim_idx, d] = np.sqrt(np.mean((x_u_free[d, :] - state[d, :]) ** 2))
            rmse_goal[sim_idx, d] = np.sqrt(np.mean((x_u_goal[d, :] - state[d, :]) ** 2))

        if sim_idx == 0:
            example_run = {
                "time": time,
                "state": state,
                "dN": dN,
                "x_u_free": x_u_free,
                "x_u_goal": x_u_goal,
                "W_u_free": W_u_free,
                "W_u_goal": W_u_goal,
            }

    return {
        "rmse_free": rmse_free,
        "rmse_goal": rmse_goal,
        "example": example_run,
        "n_cells": n_cells,
        "n_sims": n_sims,
        "state_labels": ["x", "y", "vx", "vy"],
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Part C — Hybrid filter
# ──────────────────────────────────────────────────────────────────────────────


def _run_part_c(seed=37, n_cells=40):
    """PPHybridFilterLinear: joint discrete/continuous state decoding."""
    rng = np.random.default_rng(seed)
    delta = 0.01  # 10 ms bins
    ns = 4  # continuous state dimension (x, y, vx, vy)

    # ── Simulate trajectory ──
    time = np.arange(0.0, 10.0, delta, dtype=float)
    T = len(time)
    x_pos = 0.3 * np.sin(2.0 * np.pi * 0.15 * time)
    y_pos = 0.25 * np.cos(2.0 * np.pi * 0.10 * time)
    vx = np.gradient(x_pos, delta)
    vy = np.gradient(y_pos, delta)
    state = np.vstack([x_pos, y_pos, vx, vy])  # (4, T)

    # Discrete state: alternating reach / hold (period ~6s)
    true_mode = np.where(np.sin(2.0 * np.pi * time / 6.0) > 0.0, 1, 2).astype(int)
    # Add stochastic flips
    flip = rng.random(T) < 0.01
    true_mode[flip] = 3 - true_mode[flip]

    # ── State-space models (one per mode) ──
    A_reach = np.array([
        [1, 0, delta, 0],
        [0, 1, 0, delta],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    Q_reach = 0.001 * np.eye(ns)

    # Hold state: damped velocity
    A_hold = np.array([
        [1, 0, delta, 0],
        [0, 1, 0, delta],
        [0, 0, 0.95, 0],
        [0, 0, 0, 0.95],
    ], dtype=float)
    Q_hold = 0.0005 * np.eye(ns)

    # ── Encoding model ──
    # Neurons tuned to ALL state dimensions (position + velocity).
    # Mode-dependent baseline: mode 1 (reach) has different rate than mode 2 (hold).
    b0_mode1 = rng.normal(-3.5, 0.2, n_cells)  # reach baseline
    b0_mode2 = rng.normal(-2.5, 0.2, n_cells)  # hold baseline

    # Full state tuning: position + velocity
    beta_mat = np.zeros((ns, n_cells), dtype=float)
    beta_mat[0, :] = rng.normal(0.0, 2.0, n_cells)  # x position
    beta_mat[1, :] = rng.normal(0.0, 2.0, n_cells)  # y position
    beta_mat[2, :] = rng.normal(0.0, 3.0, n_cells)  # vx
    beta_mat[3, :] = rng.normal(0.0, 3.0, n_cells)  # vy

    # Simulate spikes with mode-dependent baseline (binomial)
    dN = np.zeros((n_cells, T), dtype=float)
    for t in range(T):
        b0 = b0_mode1 if true_mode[t] == 1 else b0_mode2
        eta = b0 + beta_mat.T @ state[:, t]
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -20.0, 20.0)))
        dN[:, t] = (rng.random(n_cells) < p).astype(float)

    # ── Transition matrix ──
    p_ij = np.array([[0.985, 0.015], [0.02, 0.98]], dtype=float)

    # ── Decode with PPHybridFilterLinear ──
    Mu0 = np.array([0.5, 0.5])
    x0 = [state[:, 0], state[:, 0]]
    Pi0 = [0.5 * np.eye(ns), 0.5 * np.eye(ns)]

    S_est, X_est, W_est, MU_u, _, _, _ = DecodingAlgorithms.PPHybridFilterLinear(
        [A_reach, A_hold],
        [Q_reach, Q_hold],
        p_ij,
        Mu0,
        dN,
        [b0_mode1, b0_mode2],
        [beta_mat, beta_mat],
        "binomial",
        delta,
        None,  # gamma
        None,  # windowTimes
        x0,
        Pi0,
    )

    # Classification accuracy
    state_acc = float(np.mean(S_est == true_mode))

    # Position RMSE
    rmse_x = float(np.sqrt(np.mean((X_est[0, :] - x_pos) ** 2)))
    rmse_y = float(np.sqrt(np.mean((X_est[1, :] - y_pos) ** 2)))

    return {
        "time": time,
        "state": state,
        "true_mode": true_mode,
        "S_est": S_est,
        "X_est": X_est,
        "MU_u": MU_u,
        "dN": dN,
        "state_acc": state_acc,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "n_cells": n_cells,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────────────────────


def _plot_part_a(result):
    """Figure 1: CIF setup & raster. Figure 2: Decoded vs true stimulus."""
    time = result["time"]
    x_true = result["x_true"]
    dN = result["dN"]
    delta = time[1] - time[0]

    # ── Figure 1: stimulus, CIF, spike raster (3 panels, matching MATLAB) ──
    fig1, axes1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Top: driving stimulus
    axes1[0].plot(time, x_true, "k-", linewidth=1.5)
    axes1[0].set_ylabel("Stimulus")
    axes1[0].set_title("Driving Stimulus", fontweight="bold", fontsize=14)
    axes1[0].tick_params(labelbottom=False)

    # Middle: conditional intensity functions (firing rates in spikes/sec)
    b0 = result["b0"]
    b1 = result["b1"]
    n_cells = dN.shape[0]
    for c in range(n_cells):
        eta = b1[c] * x_true + b0[c]
        exp_eta = np.exp(eta)
        lam = (exp_eta / (1.0 + exp_eta)) / delta  # probability → rate (Hz)
        axes1[1].plot(time, lam, "k-", linewidth=1.0)
    axes1[1].set_ylabel("Firing Rate [spikes/sec]")
    axes1[1].set_title("Conditional Intensity Functions", fontweight="bold", fontsize=14)
    axes1[1].tick_params(labelbottom=False)

    # Bottom: spike raster
    for c in range(n_cells):
        spike_times = time[dN[c, :] > 0]
        axes1[2].plot(spike_times, np.full_like(spike_times, c + 1), "|", color="k", markersize=2)
    axes1[2].set_ylabel("Cell Number")
    axes1[2].set_xlabel("time [s]")
    axes1[2].set_ylim(0.5, n_cells + 0.5)
    axes1[2].set_yticks(np.arange(0, n_cells + 1, 10))
    axes1[2].set_title("Point Process Sample Paths", fontweight="bold", fontsize=14)
    fig1.tight_layout()

    # ── Figure 2: Decoding results (MATLAB: black=decoded, blue=actual) ──
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
    ax2.fill_between(
        time, result["ci_low"], result["ci_high"],
        color="0.75", alpha=0.4, label="95% CI"
    )
    ax2.plot(time, result["x_decoded"], "k-", linewidth=2.0, label="Decoded")
    ax2.plot(time, x_true, "b-", linewidth=2.0, label="Actual")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("")
    ax2.set_title(f"Decoded Stimulus $\\pm$ 95% CIs with {result['n_cells']} cells",
                  fontweight="bold", fontsize=14)
    ax2.legend(loc="upper right")
    fig2.tight_layout()

    return fig1, fig2


def _plot_part_b(result):
    """Figure 3: Reach trajectory & encoding. Figure 4: RMSE comparison."""
    ex = result["example"]
    time = ex["time"]
    state = ex["state"]

    # ── Figure 3: Example reach with decoded trajectories ──
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
    labels = result["state_labels"]
    ylabels = ["x (m)", "y (m)", "vx (m/s)", "vy (m/s)"]

    for d, (ax, lab, ylab) in enumerate(zip(axes3.ravel(), labels, ylabels)):
        ax.plot(time, state[d, :], "k-", linewidth=1.0, label="True")
        ax.plot(time, ex["x_u_free"][d, :], "b-", linewidth=0.7, alpha=0.8, label="PPAF free")
        ax.plot(time, ex["x_u_goal"][d, :], "r-", linewidth=0.7, alpha=0.8, label="PPAF+Goal")
        ax.set_ylabel(ylab)
        ax.set_title(f"State: {lab}")
        if d >= 2:
            ax.set_xlabel("Time (s)")
        if d == 0:
            ax.legend(loc="upper right", fontsize=8)

    fig3.suptitle("Part B: Arm Reach — PPAF Decoding (Example Run)", fontsize=12)
    fig3.tight_layout()

    # ── Figure 4: RMSE box plot (free vs goal) ──
    fig4, axes4 = plt.subplots(1, 4, figsize=(14, 4))
    for d, (ax, lab) in enumerate(zip(axes4, labels)):
        data = [result["rmse_free"][:, d], result["rmse_goal"][:, d]]
        bp = ax.boxplot(data, labels=["Free", "Goal"])
        ax.set_title(f"RMSE: {lab}")
        ax.set_ylabel("RMSE")

    fig4.suptitle(
        f"Part B: PPAF Free vs Goal ({result['n_sims']} simulations, {result['n_cells']} cells)",
        fontsize=12,
    )
    fig4.tight_layout()

    return fig3, fig4


def _plot_part_c(result):
    """Figure 5: Hybrid setup. Figure 6: Hybrid decoding results."""
    time = result["time"]

    # ── Figure 5: Setup — state sequence + raster ──
    fig5, axes5 = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    # Top: discrete state
    axes5[0].plot(time, result["true_mode"], "k-", linewidth=1.0, label="True mode")
    axes5[0].set_ylabel("Discrete State")
    axes5[0].set_yticks([1, 2])
    axes5[0].set_yticklabels(["Reach", "Hold"])
    axes5[0].set_title("Part C: Hybrid Filter Setup")
    axes5[0].legend()

    # Bottom: spike raster (first 20 cells)
    dN = result["dN"]
    n_show = min(20, dN.shape[0])
    for c in range(n_show):
        idx = np.where(dN[c, :] > 0)[0]
        spike_t = time[idx]
        axes5[1].plot(spike_t, np.full_like(spike_t, c + 1), "|", color="k", markersize=2)
    axes5[1].set_ylabel("Neuron")
    axes5[1].set_xlabel("Time (s)")
    axes5[1].set_ylim(0.5, n_show + 0.5)
    fig5.tight_layout()

    # ── Figure 6: Decoding results ──
    fig6, axes6 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Top: model probabilities
    axes6[0].plot(time, result["MU_u"][0, :], "b-", linewidth=0.5, label="P(Reach)")
    axes6[0].plot(time, result["MU_u"][1, :], "r-", linewidth=0.5, label="P(Hold)")
    axes6[0].axhline(0.5, color="gray", linestyle="--", linewidth=0.5)
    axes6[0].set_ylabel("Model Prob")
    axes6[0].set_title(
        f"PPHybridFilterLinear — State Accuracy: {result['state_acc']:.1%}"
    )
    axes6[0].legend(loc="upper right", fontsize=8)

    # Middle: decoded x-position
    axes6[1].plot(time, result["state"][0, :], "k-", linewidth=1.0, label="True")
    axes6[1].plot(time, result["X_est"][0, :], "b-", linewidth=0.7, alpha=0.8, label="Decoded")
    axes6[1].set_ylabel("x (m)")
    axes6[1].legend(loc="upper right", fontsize=8)

    # Bottom: decoded y-position
    axes6[2].plot(time, result["state"][1, :], "k-", linewidth=1.0, label="True")
    axes6[2].plot(time, result["X_est"][1, :], "r-", linewidth=0.7, alpha=0.8, label="Decoded")
    axes6[2].set_ylabel("y (m)")
    axes6[2].set_xlabel("Time (s)")
    axes6[2].legend(loc="upper right", fontsize=8)

    fig6.suptitle(
        f"Hybrid Decoding (RMSE: x={result['rmse_x']:.4f}, y={result['rmse_y']:.4f})",
        fontsize=12,
    )
    fig6.tight_layout()

    return fig5, fig6


# ──────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ──────────────────────────────────────────────────────────────────────────────


def run_example05(*, export_figures=False, export_dir=None, show=False):
    """Run Example 05: PPAF and PPHF decoding.

    Analysis workflow (mirrors Matlab ``example05_decoding_ppaf_pphf.m``):

    Part A — Univariate stimulus decoding:
      1. Define 20-cell population with sinusoidal tuning.
      2. Simulate spikes from binomial CIF.
      3. Decode stimulus via PPDecodeFilterLinear.

    Part B — Arm-reach PPAF:
      4. Simulate 4-state reaching movements (position + velocity).
      5. Encode with 20-cell cosine-tuning population.
      6. Decode with PPAF (free) and PPAF+Goal; compare across 20 runs.

    Part C — Hybrid filter:
      7. Simulate 40-cell population with discrete state modulation.
      8. Decode joint discrete/continuous state via PPHybridFilterLinear.
    """
    print("=" * 70)
    print("Example 05: Stimulus Decoding with PPAF and PPHF")
    print("=" * 70)

    # --- Part A: Univariate sinusoidal stimulus ---
    print("\n--- Part A: Univariate Sinusoidal Stimulus ---")
    result_a = _run_part_a()
    print(f"  {result_a['n_cells']} cells, decode RMSE = {result_a['rmse']:.4f}")

    # --- Part B: Arm-reach PPAF ---
    print("\n--- Part B: Arm Reach PPAF (20 simulations) ---")
    result_b = _run_part_b()
    mean_free = result_b["rmse_free"].mean(axis=0)
    mean_goal = result_b["rmse_goal"].mean(axis=0)
    print(f"  Mean RMSE (free):  x={mean_free[0]:.4f}, y={mean_free[1]:.4f}, "
          f"vx={mean_free[2]:.4f}, vy={mean_free[3]:.4f}")
    print(f"  Mean RMSE (goal):  x={mean_goal[0]:.4f}, y={mean_goal[1]:.4f}, "
          f"vx={mean_goal[2]:.4f}, vy={mean_goal[3]:.4f}")

    # --- Part C: Hybrid filter ---
    print("\n--- Part C: Hybrid Filter ---")
    result_c = _run_part_c()
    print(f"  {result_c['n_cells']} cells, state accuracy = {result_c['state_acc']:.1%}")
    print(f"  Position RMSE: x={result_c['rmse_x']:.4f}, y={result_c['rmse_y']:.4f}")

    # Summary
    summary = {
        "experiment5": {
            "num_cells": float(result_a["n_cells"]),
            "decode_rmse": result_a["rmse"],
        },
        "experiment5b": {
            "num_cells": float(result_b["n_cells"]),
            "n_sims": float(result_b["n_sims"]),
            "mean_rmse_free_x": float(mean_free[0]),
            "mean_rmse_goal_x": float(mean_goal[0]),
        },
        "experiment6": {
            "num_cells": float(result_c["n_cells"]),
            "state_accuracy": result_c["state_acc"],
            "decode_rmse_x": result_c["rmse_x"],
            "decode_rmse_y": result_c["rmse_y"],
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
        for i, fig in enumerate(figures, 1):
            fig_names = [
                "fig01_univariate_setup", "fig02_univariate_decoding",
                "fig03_reach_and_population_setup", "fig04_ppaf_goal_vs_free",
                "fig05_hybrid_setup", "fig06_hybrid_decoding_summary",
            ]
            path = export_dir / f"{fig_names[i - 1]}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
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
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    args = parser.parse_args()

    result = run_example05(
        export_figures=args.export_figures,
        export_dir=args.export_dir,
        show=args.show,
    )
    if args.output_json:
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
