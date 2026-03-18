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
  4. Simulate minimum-energy reaching trajectory (position + velocity, 4-D).
  5. Encode with 20-cell velocity-tuned population (binomial CIF).
  6. Decode with PPAF (free) and PPAF + Goal; overlay 20 simulations.

Part C — Hybrid Filter (Figures 5–6):
  7. Load hybrid-filter trajectory fixture with 2 discrete movement states.
  8. Encode with 40-cell population (velocity-tuned, binomial CIF).
  9. Decode joint discrete + continuous state via ``PPHybridFilterLinear``
     over 20 simulations; average results.

Paper mapping:
  Sections 2.3.6–2.3.7 (decoding); Figs. 8, 9, 14 plus hybrid extension.

Expected outputs:
  - Figure 1: Driving stimulus, CIFs, and simulated spike raster.
  - Figure 2: Decoded stimulus vs. true with 95% CIs.
  - Figure 3: Reach path, neural raster, position/velocity traces, CIFs.
  - Figure 4: 20-simulation overlaid decoded reach paths (PPAF vs PPAF+Goal).
  - Figure 5: Hybrid setup — reach path, raster, kinematics, discrete state.
  - Figure 6: Hybrid 20-sim averaged decoding — state, reach path, kinematics.
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


def _generate_reach_trajectory(delta, T_total, x0, xT):
    """Generate minimum-energy reaching trajectory matching MATLAB.

    Uses the forcing function:
        x(k) = A*x(k-1) + (delta/2)*(pi/T)^2*cos(pi*t/T)*[0; 0; dx; dy]
    """
    time = np.arange(0.0, T_total + delta / 2, delta)
    nT = len(time)

    A = np.array([
        [1, 0, delta, 0],
        [0, 1, 0, delta],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)

    xState = np.zeros((4, nT))
    xState[:, 0] = x0
    for k in range(1, nT):
        forcing = (delta / 2.0) * (np.pi / T_total) ** 2 * np.cos(np.pi * time[k] / T_total) * \
            np.array([0.0, 0.0, xT[0] - x0[0], xT[1] - x0[1]])
        xState[:, k] = A @ xState[:, k - 1] + forcing

    return time, xState, A


def _run_part_b(seed=0, n_cells=20, n_sims=20):
    """Arm reaching simulation and PPAF decoding — matches MATLAB exactly.

    Generates minimum-energy reach, encodes with velocity-tuned binomial CIF,
    decodes with PPAF (free) and PPAF+Goal over 20 simulations.
    """
    rng = np.random.default_rng(seed)
    delta = 0.001
    T_total = 2.0

    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    xT_target = np.array([-0.35, 0.2, 0.0, 0.0])

    time, xState, A = _generate_reach_trajectory(delta, T_total, x0, xT_target)
    nT = len(time)
    xT_actual = xState[:, -1]

    # Process noise: Qreach = diag(var(diff(xState))) * 100
    Qreach = np.diag(np.var(np.diff(xState, axis=1), axis=1)) * 100

    # First simulation: generate CIFs and spike data for Figure 3
    bCoeffs = 10.0 * (rng.random((n_cells, 2)) - 0.5)
    muCoeffs = np.log(10.0 * delta) + rng.standard_normal(n_cells)
    coeffs = np.column_stack([muCoeffs, bCoeffs])  # (C, 3)
    dataMat = np.column_stack([np.ones(nT), xState[2, :], xState[3, :]])  # (T, 3): [1, vx, vy]

    # Compute CIF for all cells
    lambdaAll = _logistic_cif(dataMat, coeffs, delta)

    # Simulate spikes
    dN_setup = _simulate_binomial_spikes_from_lambda(lambdaAll, delta, rng)

    # Store setup data for Figure 3
    setup_data = {
        "time": time,
        "xState": xState,
        "lambdaAll": lambdaAll,
        "dN": dN_setup,
        "n_cells": n_cells,
    }

    # 20 repeated simulations for Figure 4
    all_x_u_goal = []  # PPAF+Goal decoded paths
    all_x_u_free = []  # PPAF free decoded paths

    for k in range(n_sims):
        bCoeffs_k = 10.0 * (rng.random((n_cells, 2)) - 0.5)
        muCoeffs_k = np.log(10.0 * delta) + rng.standard_normal(n_cells)
        coeffs_k = np.column_stack([muCoeffs_k, bCoeffs_k])

        lambdaK = _logistic_cif(dataMat, coeffs_k, delta)
        dN_k = _simulate_binomial_spikes_from_lambda(lambdaK, delta, rng)
        dN_k = np.minimum(dN_k, 1.0)  # cap at 1

        # beta for decoding: (4, C) — zeros for position, bCoeffs for velocity
        beta_k = np.zeros((4, n_cells))
        beta_k[2, :] = bCoeffs_k[:, 0]
        beta_k[3, :] = bCoeffs_k[:, 1]

        Pi0 = np.diag([1e-6, 1e-6, 1e-6, 1e-6])
        PiT = np.diag([1e-6, 1e-6, 1e-6, 1e-6])

        # PPAF+Goal
        _, _, x_u_goal, _, _, _, _, _ = DecodingAlgorithms.PPDecodeFilterLinear(
            A, Qreach, dN_k, muCoeffs_k, beta_k, "binomial", delta,
            None, None, x0, Pi0, xT_actual, PiT, 0
        )

        # PPAF free
        _, _, x_u_free, _, _, _, _, _ = DecodingAlgorithms.PPDecodeFilterLinear(
            A, Qreach, dN_k, muCoeffs_k, beta_k, "binomial", delta,
            None, None, x0, Pi0
        )

        all_x_u_goal.append(x_u_goal)
        all_x_u_free.append(x_u_free)

    return {
        "setup": setup_data,
        "time": time,
        "xState": xState,
        "all_x_u_goal": all_x_u_goal,
        "all_x_u_free": all_x_u_free,
        "n_cells": n_cells,
        "n_sims": n_sims,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Part C — Hybrid filter
# ──────────────────────────────────────────────────────────────────────────────


def _load_hybrid_fixture():
    """Load the hybrid filter trajectory fixture (HDF5 preferred, .mat fallback)."""
    # Prefer HDF5 (needs h5py; fall back to .mat via scipy if unavailable)
    h5_path = REPO_ROOT / "data_cache" / "nstat_data" / "paperHybridFilterExample.h5"
    try:
        import h5py  # noqa: F811
    except ImportError:
        h5py = None  # type: ignore[assignment]
    if h5py is not None and h5_path.exists():
        with h5py.File(str(h5_path), "r") as f:
            d = {
                "time": f["time"][:],
                "delta": float(f["delta"][()]),
                "X": f["X"][:],
                "mstate": f["mstate"][:].astype(int),
                "p_ij": f["p_ij"][:],
                "A": [f[f"A/{i}"][:] for i in range(2)],
                "Q": [f[f"Q/{i}"][:] for i in range(2)],
                "Px0": [f[f"Px0/{i}"][:] for i in range(2)],
                "ind": [f[f"ind/{i}"][:].astype(int) for i in range(2)],
            }
        return d

    # Fallback: .mat file (scipy required)
    import scipy.io as sio
    candidates = [
        REPO_ROOT / "data_cache" / "nstat_data" / "paperHybridFilterExample.mat",
        REPO_ROOT.parent / "nSTAT_currentRelease_Local" / "helpfiles" / "paperHybridFilterExample.mat",
    ]
    for path in candidates:
        if path.exists():
            d = sio.loadmat(str(path), squeeze_me=True)
            # Normalize cell arrays to lists
            d["A"] = [d["A"][i] for i in range(2)]
            d["Q"] = [d["Q"][i] for i in range(2)]
            d["Px0"] = [d["Px0"][i] for i in range(2)]
            d["ind"] = [d["ind"][i].flatten().astype(int) for i in range(2)]
            return d
    raise FileNotFoundError(
        "Cannot find paperHybridFilterExample.h5 or .mat. "
        "Run the MATLAB export script or copy the data to data_cache/nstat_data/."
    )


def _run_part_c(seed=0, n_cells=40, n_sims=20):
    """PPHybridFilterLinear: joint discrete/continuous state decoding.

    Loads trajectory fixture, encodes with velocity-tuned binomial CIF,
    runs 20 simulations and averages decoded results — matching MATLAB.
    """
    rng = np.random.default_rng(seed)

    fixture = _load_hybrid_fixture()
    time = fixture["time"]
    delta = float(fixture["delta"])
    X = fixture["X"]  # (6, T)
    mstate = fixture["mstate"].astype(int)
    A_list = list(fixture["A"])
    Q_list = list(fixture["Q"])
    p_ij = fixture["p_ij"]
    ind = [np.asarray(v).flatten().astype(int) - 1 for v in fixture["ind"]]  # 0-based
    Px0 = list(fixture["Px0"])

    nT = len(time)

    # Clamp hold-state noise (matches MATLAB: minCovVal = 1e-12)
    Q_list[0] = 1e-12 * np.eye(Q_list[0].shape[0])

    # Compute actual process noise from trajectory
    nonMovingInd = np.intersect1d(np.where(X[4, :] == 0)[0], np.where(X[5, :] == 0)[0])
    movingInd = np.setdiff1d(np.arange(nT), nonMovingInd)
    if len(movingInd) > 1:
        Q_list[1] = np.diag(np.var(np.diff(X[:, movingInd], axis=1), axis=1))
        Q_list[1][:4, :4] = 0.0
    if len(nonMovingInd) > 1:
        varNV = np.diag(np.var(np.diff(X[:, nonMovingInd], axis=1), axis=1))
        n0 = Q_list[0].shape[0]
        Q_list[0] = varNV[:n0, :n0]

    # Setup encoding: first simulation for Figure 5
    muCoeffs_0 = np.log(10.0 * delta) + rng.standard_normal(n_cells)
    coeffs_0 = np.column_stack([
        muCoeffs_0,
        np.zeros((n_cells, 2)),
        10.0 * (rng.random((n_cells, 2)) - 0.5),
        np.zeros((n_cells, 2)),
    ])  # (C, 7): [mu, 0, 0, b_vx, b_vy, 0, 0]

    dataMat = np.column_stack([np.ones(nT), X.T])  # (T, 7)
    lambdaAll_0 = _logistic_cif(dataMat, coeffs_0, delta)
    dN_0 = _simulate_binomial_spikes_from_lambda(lambdaAll_0, delta, rng)

    # Setup data for Figure 5
    setup_data = {
        "time": time,
        "X": X,
        "mstate": mstate,
        "dN": dN_0,
        "n_cells": n_cells,
    }

    # 20 repeated simulations for Figure 6
    X_estAll = np.zeros((X.shape[0], nT, n_sims))
    X_estNTAll = np.zeros((X.shape[0], nT, n_sims))
    S_estAll = np.zeros((n_sims, nT))
    S_estNTAll = np.zeros((n_sims, nT))
    MU_estAll = []
    MU_estNTAll = []

    for n in range(n_sims):
        muCoeffs_n = np.log(10.0 * delta) + rng.standard_normal(n_cells)
        coeffs_n = np.column_stack([
            muCoeffs_n,
            np.zeros((n_cells, 2)),
            10.0 * (rng.random((n_cells, 2)) - 0.5),
            np.zeros((n_cells, 2)),
        ])

        lambdaAll_n = _logistic_cif(dataMat, coeffs_n, delta)
        dN_n = _simulate_binomial_spikes_from_lambda(lambdaAll_n, delta, rng)
        dN_n = np.minimum(dN_n, 1.0)

        mu0 = 0.5 * np.ones(p_ij.shape[0])
        beta_full = coeffs_n[:, 1:].T  # (6, C)
        # Per-model beta slices: model 0 uses ind[0] states, model 1 uses ind[1]
        beta_list = [beta_full[ind[0], :], beta_full[ind[1], :]]

        x0_list = [X[ind[0], 0], X[ind[1], 0]]
        yT_list = [X[ind[0], -1], X[ind[1], -1]]
        PiT_list = [
            1e-9 * np.eye(len(ind[0])),
            1e-9 * np.eye(len(ind[1])),
        ]

        # PPAF+Goal (with target)
        S_est, X_est, _, MU_est, _, _, _ = DecodingAlgorithms.PPHybridFilterLinear(
            A_list, Q_list, p_ij, mu0, dN_n,
            muCoeffs_n, beta_list, "binomial", delta,
            None, None, x0_list, Px0, yT_list, PiT_list
        )

        # PPAF free (no target)
        S_estNT, X_estNT, _, MU_estNT, _, _, _ = DecodingAlgorithms.PPHybridFilterLinear(
            A_list, Q_list, p_ij, mu0, dN_n,
            muCoeffs_n, beta_list, "binomial", delta,
            None, None, x0_list, Px0
        )

        X_estAll[:, :, n] = X_est
        X_estNTAll[:, :, n] = X_estNT
        S_estAll[n, :] = S_est
        S_estNTAll[n, :] = S_estNT
        MU_estAll.append(MU_est)
        MU_estNTAll.append(MU_estNT)

    MU_estAll = np.array(MU_estAll)      # (n_sims, n_modes, T)
    MU_estNTAll = np.array(MU_estNTAll)  # (n_sims, n_modes, T)

    state_acc = float(np.mean(np.mean(S_estAll, axis=0).round() == mstate))
    rmse_x = float(np.sqrt(np.mean((np.mean(X_estAll[0, :, :], axis=1) - X[0, :]) ** 2)))
    rmse_y = float(np.sqrt(np.mean((np.mean(X_estAll[1, :, :], axis=1) - X[1, :]) ** 2)))

    return {
        "setup": setup_data,
        "time": time,
        "X": X,
        "mstate": mstate,
        "X_estAll": X_estAll,
        "X_estNTAll": X_estNTAll,
        "S_estAll": S_estAll,
        "S_estNTAll": S_estNTAll,
        "MU_estAll": MU_estAll,
        "MU_estNTAll": MU_estNTAll,
        "state_acc": state_acc,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
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

    # ── Figure 1: 3×1 matching MATLAB subplot(3,1,...) ──
    fig1, axes1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

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

    # ── Figure 2: Decoded vs true (single axes, MATLAB style) ──
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
    ax2.fill_between(time, result["ci_low"], result["ci_high"],
                     color="0.75", alpha=0.4, label="95% CI")
    ax2.plot(time, result["x_decoded"], "k-", linewidth=2.0, label="Decoded")
    ax2.plot(time, stimSignal, "b-", linewidth=2.0, label="Actual")
    ax2.set_xlabel("time [s]")
    ax2.set_title(
        f"Decoded Stimulus $\\pm$ 95% CIs with {n_cells} cells",
        fontweight="bold", fontsize=14, fontfamily="Arial",
    )
    ax2.legend(loc="upper right")
    fig2.tight_layout()

    return fig1, fig2


def _plot_part_b(result):
    """Figure 3: Reach setup (4×2). Figure 4: 20-sim decoded paths (4×2)."""
    time = result["time"]
    xState = result["xState"]
    setup = result["setup"]

    # ── Figure 3: 4×2 setup — matches MATLAB subplot(4,2,...) ──
    fig3 = plt.figure(figsize=(14, 9))

    # (4,2,[1,3]): 2D reach path with start/end markers
    ax_path = fig3.add_subplot(4, 2, (1, 3))
    ax_path.plot(100 * xState[0, :], 100 * xState[1, :], "k", linewidth=2)
    ax_path.plot(100 * xState[0, 0], 100 * xState[1, 0], "bo", markersize=14)
    ax_path.plot(100 * xState[0, -1], 100 * xState[1, -1], "ro", markersize=14)
    ax_path.set_xlabel("X Position [cm]")
    ax_path.set_ylabel("Y Position [cm]")
    ax_path.set_title("Reach Path", fontweight="bold", fontsize=14,
                       fontfamily="Arial")
    ax_path.legend(["Path", "Start", "Finish"], loc="upper right")

    # (4,2,[2,4]): Spike raster
    ax_raster = fig3.add_subplot(4, 2, (2, 4))
    dN = setup["dN"]
    n_cells = setup["n_cells"]
    for c in range(n_cells):
        spike_t = time[dN[c, :] > 0]
        ax_raster.plot(spike_t, np.full_like(spike_t, c + 1), "|", color="k",
                       markersize=2)
    ax_raster.set_ylim(0.5, n_cells + 0.5)
    ax_raster.set_xticklabels([])
    ax_raster.set_title("Neural Raster", fontweight="bold", fontsize=14,
                         fontfamily="Arial")
    ax_raster.set_xlabel("time [s]")
    ax_raster.set_ylabel("Cell Number")

    # (4,2,5): Position traces x, y
    ax_pos = fig3.add_subplot(4, 2, 5)
    h1, = ax_pos.plot(time, 100 * xState[0, :], "k", linewidth=2)
    h2, = ax_pos.plot(time, 100 * xState[1, :], "k-.", linewidth=2)
    ax_pos.legend([h1, h2], ["x", "y"], loc="upper right")
    ax_pos.set_xlabel("time [s]")
    ax_pos.set_ylabel("Position [cm]")

    # (4,2,7): Velocity traces vx, vy
    ax_vel = fig3.add_subplot(4, 2, 7)
    h1, = ax_vel.plot(time, 100 * xState[2, :], "k", linewidth=2)
    h2, = ax_vel.plot(time, 100 * xState[3, :], "k-.", linewidth=2)
    ax_vel.legend([h1, h2], ["v_x", "v_y"], loc="upper right")
    ax_vel.set_xlabel("time [s]")
    ax_vel.set_ylabel("Velocity [cm/s]")

    # (4,2,[6,8]): CIFs overlaid in black
    ax_cif = fig3.add_subplot(4, 2, (6, 8))
    lambdaAll = setup["lambdaAll"]
    for c in range(n_cells):
        ax_cif.plot(time, lambdaAll[c, :], "k", linewidth=0.5)
    ax_cif.set_title("Neural Conditional Intensity Functions",
                      fontweight="bold", fontsize=14, fontfamily="Arial")
    ax_cif.set_xlabel("time [s]")
    ax_cif.set_ylabel("Firing Rate [spikes/sec]")

    fig3.tight_layout()

    # ── Figure 4: 4×2 overlaid decoded paths — matches MATLAB ──
    fig4 = plt.figure(figsize=(14, 9))
    all_goal = result["all_x_u_goal"]
    all_free = result["all_x_u_free"]

    # (4,2,1:4): 2D reach paths overlaid
    ax_paths = fig4.add_subplot(4, 2, (1, 4))
    ax_paths.plot(100 * xState[0, :], 100 * xState[1, :], "k", linewidth=3)
    ax_paths.set_title("Estimated vs. Actual Reach Paths", fontweight="bold",
                        fontsize=12, fontfamily="Arial")
    for k in range(len(all_goal)):
        ax_paths.plot(100 * all_goal[k][0, :], 100 * all_goal[k][1, :], "b",
                      linewidth=0.5, alpha=0.7)
        ax_paths.plot(100 * all_free[k][0, :], 100 * all_free[k][1, :], "g",
                      linewidth=0.5, alpha=0.7)
    ax_paths.set_xlabel("x [cm]")
    ax_paths.set_ylabel("y [cm]")

    # (4,2,5): x(t)
    ax_x = fig4.add_subplot(4, 2, 5)
    ax_x.plot(time, 100 * xState[0, :], "k", linewidth=3)
    for k in range(len(all_goal)):
        ax_x.plot(time, 100 * all_goal[k][0, :], "b", linewidth=0.5, alpha=0.5)
        ax_x.plot(time, 100 * all_free[k][0, :], "g", linewidth=0.5, alpha=0.5)
    ax_x.set_ylabel("x(t) [cm]")
    ax_x.set_xticklabels([])

    # (4,2,6): y(t) with legend
    ax_y = fig4.add_subplot(4, 2, 6)
    hA, = ax_y.plot(time, 100 * xState[1, :], "k", linewidth=3)
    hB, = ax_y.plot(time, 100 * all_goal[0][1, :], "b", linewidth=0.5)
    hC, = ax_y.plot(time, 100 * all_free[0][1, :], "g", linewidth=0.5)
    for k in range(1, len(all_goal)):
        ax_y.plot(time, 100 * all_goal[k][1, :], "b", linewidth=0.5, alpha=0.5)
        ax_y.plot(time, 100 * all_free[k][1, :], "g", linewidth=0.5, alpha=0.5)
    ax_y.legend([hA, hB, hC], ["Actual", "PPAF+Goal", "PPAF"], loc="lower right")
    ax_y.set_ylabel("y(t) [cm]")
    ax_y.set_xticklabels([])

    # (4,2,7): vx(t)
    ax_vx = fig4.add_subplot(4, 2, 7)
    ax_vx.plot(time, 100 * xState[2, :], "k", linewidth=3)
    for k in range(len(all_goal)):
        ax_vx.plot(time, 100 * all_goal[k][2, :], "b", linewidth=0.5, alpha=0.5)
        ax_vx.plot(time, 100 * all_free[k][2, :], "g", linewidth=0.5, alpha=0.5)
    ax_vx.set_ylabel("v_x(t) [cm/s]")
    ax_vx.set_xlabel("time [s]")

    # (4,2,8): vy(t)
    ax_vy = fig4.add_subplot(4, 2, 8)
    ax_vy.plot(time, 100 * xState[3, :], "k", linewidth=3)
    for k in range(len(all_goal)):
        ax_vy.plot(time, 100 * all_goal[k][3, :], "b", linewidth=0.5, alpha=0.5)
        ax_vy.plot(time, 100 * all_free[k][3, :], "g", linewidth=0.5, alpha=0.5)
    ax_vy.set_ylabel("v_y(t) [cm/s]")
    ax_vy.set_xlabel("time [s]")

    fig4.tight_layout()

    return fig3, fig4


def _plot_part_c(result):
    """Figure 5: Hybrid setup (4×2). Figure 6: 20-sim decode (4×3)."""
    time = result["time"]
    X = result["X"]
    mstate = result["mstate"]
    setup = result["setup"]

    # ── Figure 5: 4×2 setup — matches MATLAB subplot(4,2,...) ──
    fig5 = plt.figure(figsize=(14, 9))

    # (4,2,[1,3]): Reach path with markers
    ax_path = fig5.add_subplot(4, 2, (1, 3))
    ax_path.plot(100 * X[0, :], 100 * X[1, :], "k", linewidth=2)
    ax_path.plot(100 * X[0, 0], 100 * X[1, 0], "bo", markersize=16)
    ax_path.plot(100 * X[0, -1], 100 * X[1, -1], "ro", markersize=16)
    ax_path.set_xlabel("X [cm]")
    ax_path.set_ylabel("Y [cm]")
    ax_path.set_title("Reach Path", fontweight="bold", fontsize=14,
                       fontfamily="Arial")

    # (4,2,[2,4]): Spike raster
    ax_raster = fig5.add_subplot(4, 2, (2, 4))
    dN = setup["dN"]
    n_cells = setup["n_cells"]
    for c in range(n_cells):
        spike_t = time[dN[c, :] > 0]
        ax_raster.plot(spike_t, np.full_like(spike_t, c + 1), "|", color="k",
                       markersize=2)
    ax_raster.set_ylim(0.5, n_cells + 0.5)
    ax_raster.set_xticklabels([])
    ax_raster.set_title("Neural Raster", fontweight="bold", fontsize=14,
                         fontfamily="Arial")
    ax_raster.set_xlabel("time [s]")
    ax_raster.set_ylabel("Cell Number")

    # (4,2,5): Position traces
    ax_pos = fig5.add_subplot(4, 2, 5)
    h1, = ax_pos.plot(time, 100 * X[0, :], "k", linewidth=2)
    h2, = ax_pos.plot(time, 100 * X[1, :], "k-.", linewidth=2)
    ax_pos.legend([h1, h2], ["x", "y"], loc="upper right")
    ax_pos.set_xlabel("time [s]")
    ax_pos.set_ylabel("Position [cm]")

    # (4,2,7): Velocity traces
    ax_vel = fig5.add_subplot(4, 2, 7)
    h1, = ax_vel.plot(time, 100 * X[2, :], "k", linewidth=2)
    h2, = ax_vel.plot(time, 100 * X[3, :], "k-.", linewidth=2)
    ax_vel.legend([h1, h2], ["v_x", "v_y"], loc="upper right")
    ax_vel.set_xlabel("time [s]")
    ax_vel.set_ylabel("Velocity [cm/s]")

    # (4,2,[6,8]): Discrete movement state
    ax_state = fig5.add_subplot(4, 2, (6, 8))
    ax_state.plot(time, mstate, "k", linewidth=2)
    ax_state.set_ylim(0, 3)
    ax_state.set_yticks([1, 2])
    ax_state.set_yticklabels(["N", "M"])
    ax_state.set_xlabel("time [s]")
    ax_state.set_ylabel("state")
    ax_state.set_title("Discrete Movement State", fontweight="bold",
                        fontsize=14, fontfamily="Arial")

    fig5.tight_layout()

    # ── Figure 6: 4×3 averaged decode — matches MATLAB subplot(4,3,...) ──
    fig6 = plt.figure(figsize=(14, 9))

    mS_est = np.mean(result["S_estAll"], axis=0)
    mS_estNT = np.mean(result["S_estNTAll"], axis=0)
    mMU_est = np.mean(result["MU_estAll"], axis=0)    # (n_modes, T)
    mMU_estNT = np.mean(result["MU_estNTAll"], axis=0)
    mX_est = 100 * np.mean(result["X_estAll"], axis=2)    # (6, T) in cm
    mX_estNT = 100 * np.mean(result["X_estNTAll"], axis=2)

    # (4,3,[1,4]): Estimated vs actual state
    ax_s = fig6.add_subplot(4, 3, (1, 4))
    ax_s.plot(time, mstate, "k", linewidth=3)
    ax_s.plot(time, mS_est, "b", linewidth=3)
    ax_s.plot(time, mS_estNT, "g", linewidth=3)
    ax_s.set_xticklabels([])
    ax_s.set_yticks([1, 2.1])
    ax_s.set_yticklabels(["N", "M"])
    ax_s.set_ylabel("state")
    ax_s.set_title("Estimated vs. Actual State", fontweight="bold",
                    fontsize=12, fontfamily="Arial")

    # (4,3,[7,10]): P(s(t)=M|data)
    ax_p = fig6.add_subplot(4, 3, (7, 10))
    ax_p.plot(time, mMU_est[1, :], "b", linewidth=3)
    ax_p.plot(time, mMU_estNT[1, :], "g", linewidth=3)
    ax_p.set_xlim(time[0], time[-1])
    ax_p.set_ylim(0, 1.1)
    ax_p.set_xlabel("time [s]")
    ax_p.set_ylabel("P(s(t)=M | data)")
    ax_p.set_title("Probability of State", fontweight="bold", fontsize=12,
                    fontfamily="Arial")

    # (4,3,[2,3,5,6]): 2D reach path
    ax_2d = fig6.add_subplot(4, 3, (2, 6))
    ax_2d.plot(100 * X[0, :], 100 * X[1, :], "k", linewidth=1)
    ax_2d.plot(mX_est[0, :], mX_est[1, :], "b", linewidth=3)
    ax_2d.plot(mX_estNT[0, :], mX_estNT[1, :], "g", linewidth=3)
    ax_2d.plot(100 * X[0, 0], 100 * X[1, 0], "bo", markersize=14)
    ax_2d.plot(100 * X[0, -1], 100 * X[1, -1], "ro", markersize=14)
    ax_2d.set_xlabel("x [cm]")
    ax_2d.set_ylabel("y [cm]")
    ax_2d.set_title("Estimated vs. Actual Reach Path", fontweight="bold",
                     fontsize=12, fontfamily="Arial")

    # (4,3,8): X position
    ax_xp = fig6.add_subplot(4, 3, 8)
    ax_xp.plot(time, 100 * X[0, :], "k", linewidth=3)
    ax_xp.plot(time, mX_est[0, :], "b", linewidth=3)
    ax_xp.plot(time, mX_estNT[0, :], "g", linewidth=3)
    ax_xp.set_ylabel("x(t) [cm]")
    ax_xp.set_xticklabels([])
    ax_xp.set_title("X Position", fontweight="bold", fontsize=12,
                     fontfamily="Arial")

    # (4,3,9): Y position with legend
    ax_yp = fig6.add_subplot(4, 3, 9)
    h1, = ax_yp.plot(time, 100 * X[1, :], "k", linewidth=3)
    h2, = ax_yp.plot(time, mX_est[1, :], "b", linewidth=3)
    h3, = ax_yp.plot(time, mX_estNT[1, :], "g", linewidth=3)
    ax_yp.legend([h1, h2, h3], ["Actual", "PPAF+Goal", "PPAF"],
                 loc="lower right")
    ax_yp.set_ylabel("y(t) [cm]")
    ax_yp.set_xticklabels([])
    ax_yp.set_title("Y Position", fontweight="bold", fontsize=12,
                     fontfamily="Arial")

    # (4,3,11): X velocity
    ax_xv = fig6.add_subplot(4, 3, 11)
    ax_xv.plot(time, 100 * X[2, :], "k", linewidth=3)
    ax_xv.plot(time, mX_est[2, :], "b", linewidth=3)
    ax_xv.plot(time, mX_estNT[2, :], "g", linewidth=3)
    ax_xv.set_ylabel("v_x(t) [cm/s]")
    ax_xv.set_xlabel("time [s]")
    ax_xv.set_title("X Velocity", fontweight="bold", fontsize=12,
                     fontfamily="Arial")

    # (4,3,12): Y velocity
    ax_yv = fig6.add_subplot(4, 3, 12)
    ax_yv.plot(time, 100 * X[3, :], "k", linewidth=3)
    ax_yv.plot(time, mX_est[3, :], "b", linewidth=3)
    ax_yv.plot(time, mX_estNT[3, :], "g", linewidth=3)
    ax_yv.set_ylabel("v_y(t) [cm/s]")
    ax_yv.set_xlabel("time [s]")
    ax_yv.set_title("Y Velocity", fontweight="bold", fontsize=12,
                     fontfamily="Arial")

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

    Part B — Arm-reach PPAF (Figs 3–4):
      3. Minimum-energy reaching trajectory (4-D state).
      4. Velocity-tuned 20-cell population, binomial CIF.
      5. 20 simulations: PPAF free vs PPAF+Goal, overlaid decoded paths.

    Part C — Hybrid filter (Figs 5–6):
      6. Fixture trajectory with 2 discrete movement states.
      7. 40-cell velocity-tuned population, binomial CIF.
      8. 20 simulations: PPHybridFilterLinear, averaged decode results.
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
    print(f"  {result_b['n_cells']} cells, {result_b['n_sims']} sims completed")

    # --- Part C ---
    print("\n--- Part C: Hybrid Filter (20 simulations) ---")
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
