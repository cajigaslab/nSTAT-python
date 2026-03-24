#!/usr/bin/env python3
"""Example 03 — PSTH and State-Space GLM Dynamics.

This example demonstrates:
  1) Simulating spike trains from a known sinusoidal CIF.
  2) Computing PSTH (histogram) and comparing with GLM-PSTH.
  3) State-space GLM (SSGLM) estimation with EM algorithm.
  4) Across-trial learning dynamics and stimulus-effect surfaces.

The example has two parts:
  Part A (PSTH): Simulate 20 trials from a sinusoidal CIF,
      load real data from ``data/PSTH/Results.mat``, compare histogram
      PSTH vs GLM-PSTH.
  Part B (SSGLM): Simulate 50-trial dataset with across-trial gain
      modulation, load precomputed SSGLM fit from
      ``data/SSGLMExampleData.mat``, visualise learning dynamics and
      3-D stimulus-effect surfaces.

Expected outputs:
  - Figure 1: Simulated CIF + simulated/real raster examples.
  - Figure 2: PSTH comparison (histogram vs GLM).
  - Figure 3: SSGLM simulation summary.
  - Figure 4: SSGLM vs PSTH model diagnostics.
  - Figure 5: Stimulus-effect surfaces (3-D).
  - Figure 6: Learning-trial comparison and significance matrix.

Paper mapping:
  Section 2.3.3 (PSTH) and Section 2.4 (SSGLM).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nstat import (  # noqa: E402
    Analysis,
    Covariate,
    CovariateCollection,
    FitResult,
    Trial,
    TrialConfig,
    ConfigCollection,
)
from nstat.cif import CIF  # noqa: E402
from nstat.confidence_interval import ConfidenceInterval  # noqa: E402
from nstat.core import nspikeTrain  # noqa: E402
from nstat.data_manager import ensure_example_data  # noqa: E402
from nstat.decoding_algorithms import DecodingAlgorithms  # noqa: E402
from nstat.trial import SpikeTrainCollection  # noqa: E402


# =====================================================================
# Helper: load Matlab FitResult struct into Python FitResult
# =====================================================================
def _load_matlab_fitresult(mat_struct, spike_trains):
    """Convert a Matlab FitResult structured array to a Python FitResult.

    Parameters
    ----------
    mat_struct : numpy structured array
        The `fR` or `psthR` field from the .mat file.
    spike_trains : list[nspikeTrain]
        The spike trains corresponding to this FitResult (since Matlab
        MCOS objects cannot be deserialized by scipy).
    """
    # Extract lambda signal
    lam = mat_struct["lambda"].item()
    lam_time = np.asarray(lam["time"].item(), dtype=float).ravel()
    lam_data = np.asarray(lam["data"].item(), dtype=float)
    lam_name = str(lam["name"].item()) if lam["name"].size else "\\lambda"

    lambda_cov = Covariate(
        lam_time, lam_data, lam_name, "time", "s", "spikes/sec",
    )

    # Extract scalar statistics
    b_raw = np.asarray(mat_struct["b"].item(), dtype=float).reshape(-1)
    AIC_val = float(np.asarray(mat_struct["AIC"].item(), dtype=float).ravel()[0])
    BIC_val = float(np.asarray(mat_struct["BIC"].item(), dtype=float).ravel()[0])
    logLL_val = float(np.asarray(mat_struct["logLL"].item(), dtype=float).ravel()[0])
    config_name = str(mat_struct["configNames"].item())

    # Extract covariate labels
    cov_labels_raw = mat_struct["covLabels"].item()
    if isinstance(cov_labels_raw, np.ndarray):
        cov_labels = [str(x) for x in cov_labels_raw.ravel()]
    elif isinstance(cov_labels_raw, str):
        cov_labels = [cov_labels_raw]
    else:
        cov_labels = list(cov_labels_raw) if cov_labels_raw is not None else []

    num_hist_raw = mat_struct["numHist"].item()
    num_hist = [int(num_hist_raw)] if np.isscalar(num_hist_raw) else [int(x) for x in np.asarray(num_hist_raw).ravel()]

    cfgs = ConfigCollection([TrialConfig(name=config_name)])

    return FitResult(
        spike_trains,
        [cov_labels],        # covLabels (list of lists)
        num_hist,            # numHist
        [],                  # histObjects
        [],                  # ensHistObjects
        lambda_cov,          # lambda_signal
        [b_raw],             # b
        [0.0],               # dev
        [None],              # stats
        [AIC_val],           # AIC
        [BIC_val],           # BIC
        [logLL_val],         # logLL
        cfgs,                # configColl
        [],                  # XvalData
        [],                  # XvalTime
        "poisson",           # distribution
    )


# =====================================================================
# Part A: PSTH Analysis
# =====================================================================
def run_part_a(data_dir, export_dir=None):
    """Simulate and real PSTH + GLM-PSTH analysis."""
    print("=== Part A: PSTH Analysis ===")

    # ------------------------------------------------------------------
    # 1. Define sinusoidal CIF: lambda(t) = sigmoid(sin(2*pi*f*t) + mu) / dt
    # ------------------------------------------------------------------
    delta = 0.001
    tmax = 1.0
    time = np.arange(0.0, tmax + delta, delta)
    f = 2
    mu = -3

    lambdaRaw = np.sin(2 * np.pi * f * time) + mu
    lambdaData = np.exp(lambdaRaw) / (1 + np.exp(lambdaRaw)) * (1 / delta)
    lambdaCov = Covariate(
        time, lambdaData, "\\lambda(t)", "time", "s", "spikes/sec",
        ["\\lambda_{1}"],
    )

    # ------------------------------------------------------------------
    # 2. Simulate 20 spike trains via CIF thinning
    # ------------------------------------------------------------------
    numRealizations = 20
    spikeCollSim = CIF.simulateCIFByThinningFromLambda(
        lambdaCov, numRealizations, seed=0,
    )
    spikeCollSim.setMinTime(0.0)
    spikeCollSim.setMaxTime(tmax)
    print(f"  Simulated {numRealizations} spike trains")

    # ------------------------------------------------------------------
    # 3. Load real PSTH data from Results.mat
    # ------------------------------------------------------------------
    psth_path = data_dir / "PSTH" / "Results.mat"
    psthData = loadmat(str(psth_path), squeeze_me=False)
    Results = psthData["Results"][0, 0]
    Data = Results["Data"][0, 0]
    STC = Data["Spike_times_STC"][0, 0]
    SUA = STC["balanced_SUA"][0, 0]
    numTrials = int(SUA["Nr_trials"][0, 0])
    spikeTimesArr = SUA["spike_times"]  # shape (16, numTrials, 8)

    # Cell 6 (Matlab 1-indexed)
    trains6 = []
    for iTrial in range(numTrials):
        st = spikeTimesArr[0, iTrial, 5].ravel()  # cell index 5 = cell 6
        nst = nspikeTrain(st, name="6", minTime=0.0, maxTime=2.0, makePlots=-1)
        trains6.append(nst)
    spikeCollReal1 = SpikeTrainCollection(trains6)
    spikeCollReal1.setMinTime(0.0)
    spikeCollReal1.setMaxTime(2.0)

    # Cell 1 (Matlab 1-indexed)
    trains1 = []
    for iTrial in range(numTrials):
        st = spikeTimesArr[0, iTrial, 0].ravel()  # cell index 0 = cell 1
        nst = nspikeTrain(st, name="1", minTime=0.0, maxTime=2.0, makePlots=-1)
        trains1.append(nst)
    spikeCollReal2 = SpikeTrainCollection(trains1)
    spikeCollReal2.setMinTime(0.0)
    spikeCollReal2.setMaxTime(2.0)
    print(f"  Loaded real data: {numTrials} trials, cells 6 and 1")

    # ------------------------------------------------------------------
    # Figure 1: Simulated CIF + simulated/real rasters (2x2)
    # ------------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 9))

    # Top-left: CIF
    ax = axes1[0, 0]
    ax.plot(time, lambdaData, "b", linewidth=2, label=r"$\lambda_i$")
    ax.set_title("Simulated Conditional Intensity Function (CIF)",
                 fontweight="bold", fontsize=14, fontfamily="Arial")
    ax.set_xlabel("time [s]", fontsize=12, fontweight="bold", fontfamily="Arial")
    ax.set_ylabel(r"$\lambda(t)$ [spikes/sec]", fontsize=14, fontweight="bold",
                  fontfamily="Arial")
    ax.legend(loc="upper right")

    # Bottom-left: simulated raster
    ax = axes1[1, 0]
    spikeCollSim.plot(handle=ax)
    ax.set_yticks(range(0, numRealizations + 1, 5))
    ax.set_title(f"{numRealizations} Simulated Point Process Sample Paths",
                 fontweight="bold", fontsize=14, fontfamily="Arial")
    ax.set_xlabel("time [s]", fontsize=12, fontweight="bold", fontfamily="Arial")
    ax.set_ylabel("Trial [k]", fontsize=12, fontweight="bold", fontfamily="Arial")

    # Top-right: real cell 6 raster
    ax = axes1[0, 1]
    spikeCollReal1.plot(handle=ax)
    ax.set_yticks(range(0, numTrials + 1, 2))
    ax.set_title("Response to Moving Visual Stimulus (Neuron 6)",
                 fontweight="bold", fontsize=14, fontname="Arial")
    ax.set_xlabel("time [s]", fontname="Arial", fontsize=12, fontweight="bold")
    ax.set_ylabel("Trial [k]", fontname="Arial", fontsize=12, fontweight="bold")

    # Bottom-right: real cell 1 raster
    ax = axes1[1, 1]
    spikeCollReal2.plot(handle=ax)
    ax.set_yticks(range(0, numTrials + 1, 2))
    ax.set_title("Response to Moving Visual Stimulus (Neuron 1)",
                 fontweight="bold", fontsize=14, fontname="Arial")
    ax.set_xlabel("time [s]", fontname="Arial", fontsize=12, fontweight="bold")
    ax.set_ylabel("Trial [k]", fontname="Arial", fontsize=12, fontweight="bold")

    fig1.tight_layout()

    # ------------------------------------------------------------------
    # 4. Compute PSTH and GLM-PSTH
    # ------------------------------------------------------------------
    binsize = 0.05
    psthSim = spikeCollSim.psth(binsize)
    psthGLMSim, _, _ = spikeCollSim.psthGLM(binsize)

    psthReal1 = spikeCollReal1.psth(binsize)
    psthGLMReal1, _, _ = spikeCollReal1.psthGLM(binsize)

    psthReal2 = spikeCollReal2.psth(binsize)
    psthGLMReal2, _, _ = spikeCollReal2.psthGLM(binsize)
    print("  PSTH and GLM-PSTH computed for all 3 collections")

    # ------------------------------------------------------------------
    # Figure 2: PSTH comparison (2x3)
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 9))

    # Top row: rasters
    spikeCollSim.plot(handle=axes2[0, 0])
    axes2[0, 0].set_yticks(range(0, numRealizations + 1, 2))
    axes2[0, 0].set_xlabel("time [s]")
    axes2[0, 0].set_ylabel("Trial [k]")

    spikeCollReal1.plot(handle=axes2[0, 1])
    axes2[0, 1].set_yticks(range(0, numTrials + 1, 2))
    axes2[0, 1].set_xlabel("time [s]")
    axes2[0, 1].set_ylabel("Trial [k]")

    spikeCollReal2.plot(handle=axes2[0, 2])
    axes2[0, 2].set_yticks(range(0, numTrials + 1, 2))
    axes2[0, 2].set_xlabel("time [s]")
    axes2[0, 2].set_ylabel("Trial [k]")

    # Bottom row: PSTH comparisons
    ax = axes2[1, 0]
    h_true, = ax.plot(time, lambdaData, "b", linewidth=4, label="true")
    # MATLAB z-order: true, then GLM, then PSTH (markers on top)
    glm_time = np.asarray(psthGLMSim.time, dtype=float).ravel()
    glm_data = np.asarray(psthGLMSim.data, dtype=float).ravel()
    h_glm, = ax.plot(glm_time, glm_data, "k", linewidth=4, label="PSTH_{glm}")
    psth_time = np.asarray(psthSim.time, dtype=float).ravel()
    psth_data = np.asarray(psthSim.data, dtype=float).ravel()
    h_psth, = ax.plot(psth_time, psth_data, "rx", linewidth=4, label="PSTH")
    ax.legend(handles=[h_true, h_psth, h_glm])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("[spikes/sec]")

    ax = axes2[1, 1]
    psth_t1 = np.asarray(psthReal1.time, dtype=float).ravel()
    psth_d1 = np.asarray(psthReal1.data, dtype=float).ravel()
    glm_t1 = np.asarray(psthGLMReal1.time, dtype=float).ravel()
    glm_d1 = np.asarray(psthGLMReal1.data, dtype=float).ravel()
    h2, = ax.plot(psth_t1, psth_d1, "rx", linewidth=4, label="PSTH")
    h3, = ax.plot(glm_t1, glm_d1, "k", linewidth=4, label="PSTH_{glm}")
    ax.legend(handles=[h2, h3])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("[spikes/sec]")

    ax = axes2[1, 2]
    psth_t2 = np.asarray(psthReal2.time, dtype=float).ravel()
    psth_d2 = np.asarray(psthReal2.data, dtype=float).ravel()
    glm_t2 = np.asarray(psthGLMReal2.time, dtype=float).ravel()
    glm_d2 = np.asarray(psthGLMReal2.data, dtype=float).ravel()
    h2, = ax.plot(psth_t2, psth_d2, "rx", linewidth=4, label="PSTH")
    h3, = ax.plot(glm_t2, glm_d2, "k", linewidth=4, label="PSTH_{glm}")
    ax.legend(handles=[h2, h3])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("[spikes/sec]")

    fig2.tight_layout()

    figures = {"fig01_simulated_and_real_rasters": fig1, "fig02_psth_comparison": fig2}
    return figures, spikeCollSim, lambdaCov


# =====================================================================
# Part B: SSGLM Analysis
# =====================================================================
def run_part_b(data_dir, export_dir=None):
    """SSGLM simulation, diagnostics, stimulus surfaces, learning trial."""
    print("\n=== Part B: SSGLM Analysis ===")

    # ------------------------------------------------------------------
    # 1. Simulate 50-trial CIF with across-trial stimulus gain
    # ------------------------------------------------------------------
    delta = 0.001
    tmax = 1.0
    time = np.arange(0.0, tmax + delta, delta)
    f = 2
    numRealizations = 50
    b0 = -3

    # Linearly increasing stimulus gain across trials
    b1 = 3 * np.arange(1, numRealizations + 1) / numRealizations

    # Simulate each trial using CIF.simulateCIF
    trains = []
    for iTrial in range(numRealizations):
        u = np.sin(2 * np.pi * f * time)
        stim = Covariate(time, u, "Stimulus", "time", "s", "V", ["sin"])
        ens = Covariate(time, np.zeros_like(time), "Ensemble", "time", "s",
                        "Spikes", ["n1"])

        histCoeffs = [-4, -1, -0.5]

        sC, lambdaTemp = CIF.simulateCIF(
            b0, histCoeffs, [b1[iTrial]], [0],
            stim, ens, 1, "binomial",
            seed=iTrial, return_lambda=True,
        )
        nst = sC.getNST(1)
        nst = nst.nstCopy()
        nst.resample(1 / delta)
        trains.append(nst)

    spikeColl = SpikeTrainCollection(trains)
    spikeColl.setMinTime(0.0)
    spikeColl.setMaxTime(tmax)
    print(f"  Simulated {numRealizations} spike trains with CIF.simulateCIF")

    # Compute true CIF surface: sigma(b0 + b1[k]*u(t)) / delta
    u = np.sin(2 * np.pi * f * time)
    stimDataEta = np.outer(u, b1)  # (T, K)
    stimData = np.exp(stimDataEta + b0)
    stimData = stimData / (1 + stimData) / delta  # binomial link

    # ------------------------------------------------------------------
    # Figure 3: SSGLM simulation summary (3x2)
    # ------------------------------------------------------------------
    fig3, axes3 = plt.subplots(3, 2, figsize=(14, 9))

    # (1,1): Within-trial stimulus
    ax = axes3[0, 0]
    ax.plot(time, u, "k", linewidth=3)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("Stimulus")
    ax.set_title("Within Trial Stimulus", fontweight="bold", fontsize=14)

    # (1,2): Across-trial gain
    ax = axes3[0, 1]
    ax.plot(np.arange(1, numRealizations + 1), b1, "k", linewidth=3)
    ax.set_xlabel("Trial [k]")
    ax.set_ylabel("Stimulus Gain")
    ax.set_title("Across Trial Stimulus Gain", fontweight="bold", fontsize=14)

    # (2,1)+(2,2): Raster spanning both columns
    axes3[1, 1].remove()
    ax = axes3[1, 0]
    ax.set_position(
        [axes3[1, 0].get_position().x0,
         axes3[1, 0].get_position().y0,
         axes3[1, 1].get_position().x1 - axes3[1, 0].get_position().x0,
         axes3[1, 0].get_position().height]
    )
    spikeColl.plot(handle=ax)
    ax.set_yticks(range(0, numRealizations + 1, 10))
    ax.set_xlabel("time [s]")
    ax.set_ylabel("Trial [k]")
    ax.set_title("Simulated Neural Raster", fontweight="bold", fontsize=14)

    # (3,1)+(3,2): True CIF heatmap spanning both columns
    axes3[2, 1].remove()
    ax = axes3[2, 0]
    ax.set_position(
        [axes3[2, 0].get_position().x0,
         axes3[2, 0].get_position().y0,
         ax.get_position().width * 2.1,
         axes3[2, 0].get_position().height]
    )
    ax.imshow(stimData.T, aspect="auto", origin="lower",
              extent=[time[0], time[-1], 1, numRealizations],
              cmap="jet")  # MATLAB default colormap for imagesc
    ax.set_xlabel("time [s]")
    ax.set_ylabel("Trial [k]")
    ax.set_title("True Conditional Intensity Function", fontweight="bold",
                 fontsize=14)
    ax.set_yticks(range(0, numRealizations + 1, 10))

    fig3.tight_layout()

    # ------------------------------------------------------------------
    # 2. Compute PSTH-GLM and prepare data matrices
    #    (Matlab: psthGLM + dN before loading precomputed SSGLM)
    # ------------------------------------------------------------------
    numBasis = 25
    basisWidth = (tmax - 0.0) / numBasis
    windowTimes = np.arange(0.0, 0.004, delta)
    fitType = "poisson"

    spikeColl.resample(1 / delta)
    spikeColl.setMaxTime(tmax)

    # MATLAB: dN = spikeColl.dataToMatrix'  →  (K, T)
    # Python dataToMatrix() returns (T, K), so transpose to match.
    dN = spikeColl.dataToMatrix().T  # (K, T)
    if dN.ndim == 1:
        dN = dN.reshape(1, -1)
    dN = np.asarray(dN, dtype=float)
    dN[dN > 1] = 1

    psthSig, _, _ = spikeColl.psthGLM(basisWidth, windowTimes, fitType)
    print("  Computed psthGLM on 50-trial collection")

    # ------------------------------------------------------------------
    # 3. Load precomputed SSGLM data
    # ------------------------------------------------------------------
    ssglm_path = data_dir / "SSGLMExampleData.mat"
    ssglm = loadmat(str(ssglm_path), squeeze_me=True)

    xK = np.asarray(ssglm["xK"], dtype=float)            # (25, 50)
    WkuFinal = np.asarray(ssglm["WkuFinal"], dtype=float) # (25, 25, 50, 50)
    stimulus_true = np.asarray(ssglm["stimulus"], dtype=float)  # (25, 50)
    stimCIs = np.asarray(ssglm["stimCIs"], dtype=float)         # (25, 50, 2)
    gammahat = np.asarray(ssglm["gammahat"], dtype=float)       # (3,)
    K = xK.shape[1]
    print(f"  Loaded precomputed SSGLM: {numBasis} basis x {K} trials")

    # ------------------------------------------------------------------
    # 4. Reconstruct FitResult objects from loaded data
    # ------------------------------------------------------------------
    ssglm_fit = _load_matlab_fitresult(ssglm["fR"], trains)
    psth_fit = _load_matlab_fitresult(ssglm["psthR"], trains)

    tCompare = psth_fit.mergeResults(ssglm_fit)
    tCompare.lambda_signal.setDataLabels(
        ["\\lambda_{PSTH}", "\\lambda_{SSGLM}"]
    )

    # ------------------------------------------------------------------
    # Figure 4: SSGLM vs PSTH diagnostics (2x2)
    # ------------------------------------------------------------------
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 9))
    tCompare.KSPlot(handle=axes4[0, 0])
    tCompare.plotResidual(handle=axes4[0, 1])
    tCompare.plotInvGausTrans(handle=axes4[1, 0])
    tCompare.plotSeqCorr(handle=axes4[1, 1])
    fig4.tight_layout()
    print("  Figure 4: SSGLM vs PSTH diagnostics")

    # ------------------------------------------------------------------
    # 5. Compute stimulus effect surfaces
    # ------------------------------------------------------------------
    sampleRate = 1 / delta

    unitPulseBasis = SpikeTrainCollection.generateUnitImpulseBasis(
        basisWidth, 0.0, tmax, sampleRate,
    )
    basisMat = np.asarray(unitPulseBasis.data, dtype=float)  # (T, numBasis)
    basis_time = np.asarray(unitPulseBasis.time, dtype=float).ravel()

    # True stimulus effect (Poisson link, matching fitType for analysis)
    u_basis = np.sin(2 * np.pi * f * basis_time)
    actStimEffect = np.exp(np.outer(u_basis, b1) + b0) / delta  # (T, K)

    # PSTH surface (constant across trials — replicate fresh psthGLM output)
    psthSig_data = np.asarray(psthSig.data, dtype=float).ravel()
    psthSurface2D = np.tile(psthSig_data[:, None], (1, numRealizations))

    # SSGLM estimated CIF from basis coefficients
    estStimEffect = np.exp(basisMat @ xK) / delta  # (T, K)

    # ------------------------------------------------------------------
    # Figure 5: True/PSTH/SSGLM stimulus effect surfaces
    # MATLAB: mesh(trial, time, data) with view([90 -90]) renders as a
    # top-down colored heatmap (MATLAB applies its colormap to Z-values).
    # Python equivalent: pcolormesh with viridis (≈MATLAB parula default).
    # MATLAB orientation: trial on x-axis, time on y-axis (view [90 -90]).
    # ------------------------------------------------------------------
    fig5, axes5 = plt.subplots(3, 1, figsize=(14, 9))
    trial_axis = np.arange(1, numRealizations + 1)
    T_act = min(actStimEffect.shape[0], len(basis_time))

    surfaces = [
        ("True Stimulus Effect", actStimEffect[:T_act, :]),
        ("PSTH Estimated Stimulus Effect", psthSurface2D[:T_act, :]),
        ("SSGLM Estimated Stimulus Effect", estStimEffect[:T_act, :]),
    ]
    for ax, (title, data) in zip(axes5, surfaces):
        # MATLAB mesh(trial, time, data) viewed from above: x=trial, y=time
        ax.pcolormesh(trial_axis, basis_time[:T_act], data,
                       shading="gouraud", cmap="viridis")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontweight="bold", fontsize=14, fontfamily="Arial")

    fig5.tight_layout()
    print("  Figure 5: Stimulus effect surfaces (top-down heatmap)")

    # ------------------------------------------------------------------
    # 6. Learning-trial analysis: spike rate CIs
    # ------------------------------------------------------------------
    tRate, probMat, sigMat = DecodingAlgorithms.computeSpikeRateCIs(
        xK, WkuFinal, dN, 0, tmax, fitType, delta, gammahat, windowTimes,
    )

    # Find first learning trial (first column where significance appears)
    sig_cols = np.where(sigMat[0, :] == 1)[0]
    lt = int(sig_cols[0]) if sig_cols.size > 0 else 2
    if lt < 2:
        lt = 2

    # ------------------------------------------------------------------
    # Figure 6: Learning trial comparison + significance matrix (2x3)
    # ------------------------------------------------------------------
    fig6 = plt.figure(figsize=(14, 9))

    # (1,1): average spike rate with learning trial marker
    ax1 = fig6.add_subplot(2, 3, 1)
    rate_time = np.asarray(tRate.time, dtype=float).ravel()
    rate_data = np.asarray(tRate.data, dtype=float).ravel()
    ax1.plot(rate_time, rate_data, "k", linewidth=4)
    ylims = ax1.get_ylim()
    ax1.plot([lt, lt], ylims, "r", linewidth=2)
    ax1.set_xlabel("Trial [k]")
    ax1.set_ylabel("Average Firing Rate [spikes/sec]")
    ax1.set_title(f"Learning Trial: {lt}", fontweight="bold", fontsize=12)

    # (1,2)+(1,3)+(2,2)+(2,3): significance matrix
    ax2 = fig6.add_subplot(2, 3, (2, 6))
    ax2.imshow(probMat, cmap="gray_r", aspect="auto")
    kTrials = sigMat.shape[0]
    for k in range(kTrials):
        for m in range(k + 1, kTrials):
            if sigMat[k, m] == 1:
                ax2.plot(m, k, "r*", markersize=6)
    ax2.xaxis.set_ticks_position("top")
    ax2.xaxis.set_label_position("top")
    ax2.yaxis.set_ticks_position("right")
    ax2.yaxis.set_label_position("right")
    ax2.set_xlabel("Trial Number")
    ax2.set_ylabel("Trial Number")

    # (2,1): CIF comparison for trial 1 vs learning trial
    ax3 = fig6.add_subplot(2, 3, 4)
    stim1_data = basisMat @ stimulus_true[:, 0]
    stimlt_data = basisMat @ stimulus_true[:, lt - 1]
    ci1_lo = basisMat @ stimCIs[:, 0, 0]
    ci1_hi = basisMat @ stimCIs[:, 0, 1]
    cilt_lo = basisMat @ stimCIs[:, lt - 1, 0]
    cilt_hi = basisMat @ stimCIs[:, lt - 1, 1]

    ax3.fill_between(basis_time, ci1_lo, ci1_hi, alpha=0.3, color="gray")
    ax3.fill_between(basis_time, cilt_lo, cilt_hi, alpha=0.3, color="red")
    h1, = ax3.plot(basis_time, stim1_data, "k", linewidth=4,
                   label=r"$\lambda_1(t)$")
    h2, = ax3.plot(basis_time, stimlt_data, "r", linewidth=4,
                   label=rf"$\lambda_{{{lt}}}(t)$")
    ax3.legend(handles=[h1, h2])
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("Firing Rate [spikes/sec]")
    ax3.set_title("Learning Trial Vs. Baseline Trial\nwith 95% CIs",
                  fontweight="bold", fontsize=12)

    fig6.tight_layout()
    print(f"  Figure 6: Learning trial = {lt}")

    figures = {
        "fig03_ssglm_simulation_summary": fig3,
        "fig04_ssglm_fit_diagnostics": fig4,
        "fig05_stimulus_effect_surfaces": fig5,
        "fig06_learning_trial_comparison": fig6,
    }
    return figures


# =====================================================================
# Main
# =====================================================================
def run_example03(*, export_figures: bool = False, export_dir: Path | None = None):
    """Run Example 03: PSTH and SSGLM dynamics."""
    data_dir = ensure_example_data(download=True)

    if export_dir is None:
        export_dir = THIS_DIR / "figures" / "example03"

    figs_a, _, _ = run_part_a(data_dir, export_dir)
    figs_b = run_part_b(data_dir, export_dir)

    all_figs = {**figs_a, **figs_b}

    if export_figures:
        export_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in all_figs.items():
            path = export_dir / f"{name}.png"
            fig.savefig(str(path), dpi=250, facecolor="w", edgecolor="none")
            print(f"  Saved {path}")

    plt.show()
    print(f"\nExample 03 complete. Generated {len(all_figs)} figure(s).")
    return all_figs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example 03: PSTH and SSGLM Dynamics"
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--export-figures", action="store_true")
    parser.add_argument("--export-dir", type=Path, default=None)
    args = parser.parse_args()

    run_example03(
        export_figures=args.export_figures,
        export_dir=args.export_dir,
    )
