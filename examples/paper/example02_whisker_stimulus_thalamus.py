#!/usr/bin/env python3
"""Example 02 — Whisker Stimulus GLM With Lag and History Selection.

This example demonstrates:
  1) Fitting an explicit-stimulus point-process GLM to thalamic spike data.
  2) Cross-correlation analysis to identify optimal stimulus lag.
  3) History-order selection via AIC/BIC sweeps.
  4) Model comparison: baseline vs stimulus vs stimulus+history.

Data provenance:
  Uses ``data/Explicit Stimulus/Dir3/Neuron1/Stim2/trngdataBis.mat``
  (whisker displacement ``t``, binary spike indicator ``y``, 1000 Hz).

Expected outputs:
  - Figure 1: Data overview (raster, stimulus displacement, velocity).
  - Figure 2: Lag selection (CCF), history diagnostics, KS plot, coefficients.

Paper mapping:
  Section 2.3.2 (thalamic whisker-stimulus analysis); Figs. 4 and 11.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# Ensure nstat is importable when running from the examples/paper directory.
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import nstat  # noqa: E402
from nstat import (  # noqa: E402
    Analysis,
    ConfigColl,
    CovColl,
    nspikeTrain,
    nstColl,
    Trial,
    TrialConfig,
)
from nstat.signal import Covariate  # noqa: E402
from nstat.data_manager import ensure_example_data  # noqa: E402


# =========================================================================
# Helper: export figure
# =========================================================================
def _maybe_export(fig, export_dir: Path | None, name: str, dpi: int = 250):
    """Save figure to disk if export_dir is set."""
    saved = []
    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)
        png_path = export_dir / f"{name}.png"
        fig.savefig(png_path, dpi=dpi, facecolor="w", edgecolor="none")
        saved.append(png_path)
        print(f"  Saved {png_path}")
    return saved


# =========================================================================
# Main example function
# =========================================================================
def run_example02(*, export_figures: bool = False, export_dir: Path | None = None,
                  visible: bool = True):
    """Run Example 02: Whisker stimulus GLM with lag and history selection.

    Mirrors Matlab example02_whisker_stimulus_thalamus.m exactly:
      1. Load trngdataBis.mat (struct with fields t=stimulus, y=spike indicator).
      2. Construct nSTAT objects (nspikeTrain, Covariate, Trial).
      3. Fit baseline-only GLM; compute residual cross-covariance with stimulus.
      4. Identify optimal lag from peak xcov; shift stimulus by that lag.
      5. Sweep history windows via Analysis.computeHistLagForAll with logspace grid.
      6. Select optimal history order from min(AIC_idx, BIC_idx).
      7. Fit 3 nested models: baseline, baseline+stim, baseline+stim+hist.
      8. Generate 2 figures with Matlab-matching subplot layouts.
    """
    if not visible:
        matplotlib.use("Agg")

    data_dir = ensure_example_data(download=True)
    figure_files: list[Path] = []

    sampleRate = 1000  # Hz

    # ==================================================================
    # Load data from trngdataBis.mat
    # ==================================================================
    print("=== Example 02: Whisker Stimulus GLM ===")

    mat_path = (data_dir / "Explicit Stimulus" / "Dir3" / "Neuron1"
                / "Stim2" / "trngdataBis.mat")
    d = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    # Extract stimulus signal and spike indicator from struct
    # Matlab: data.t is stimulus, data.y is binary spike indicator
    if hasattr(d.get("data", None), "t"):
        stimData = np.asarray(d["data"].t, dtype=float).reshape(-1)
        yData = np.asarray(d["data"].y, dtype=float).reshape(-1)
    else:
        # Fallback: try direct keys
        stimData = np.asarray(d["t"], dtype=float).reshape(-1)
        yData = np.asarray(d["y"], dtype=float).reshape(-1)

    # Construct time vector at 1 ms resolution
    time = np.arange(0, len(stimData)) * (1.0 / sampleRate)

    # Extract spike times from binary indicator
    spikeTimes = time[yData == 1]
    print(f"  Data length: {len(stimData)} samples ({time[-1]:.1f} s)")
    print(f"  Total spikes: {len(spikeTimes)}")

    # ==================================================================
    # Create nSTAT objects
    # ==================================================================
    # Stimulus covariate (divided by 10, matching Matlab: stimData ./ 10)
    stim = Covariate(
        time, stimData / 10.0,
        "Stimulus", "time", "s", "mm",
        dataLabels=["stim"],
    )
    # Constant baseline covariate
    baseline = Covariate(
        time, np.ones((len(time), 1)),
        "Baseline", "time", "s", "",
        dataLabels=["constant"],
    )

    nst = nspikeTrain(spikeTimes)
    spikeColl = nstColl(nst)
    trial = Trial(spikeColl, CovColl([stim, baseline]))

    # ==================================================================
    # Figure 1: Data overview — raster, stimulus, velocity (3x1 layout)
    # ==================================================================
    fig1, axes1 = plt.subplots(3, 1, figsize=(14, 9))
    viewWindow = 21.0  # First 21 seconds, matching Matlab

    # Subplot 1: Neural raster (first 21 s)
    ax = axes1[0]
    nstView = nspikeTrain(spikeTimes)
    nstView.setMaxTime(viewWindow)
    nstView.plot(handle=ax)
    ax.set_yticks([0, 1])
    xticks = np.arange(0, int(max(time)) + 1, 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([])  # No x-labels on top/middle subplots
    ax.set_xlabel("")
    ax.set_ylabel("spikes", fontsize=12, fontweight="bold", fontfamily="Arial")
    ax.set_title("Neural Raster", fontweight="bold", fontsize=16, fontfamily="Arial")
    ax.spines["top"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)

    # Subplot 2: Stimulus displacement (first 21 s, black line matching MATLAB)
    ax = axes1[1]
    stimView = stim.getSigInTimeWindow(0, viewWindow)
    ax.plot(stimView.time, stimView.data[:, 0], "k")
    ax.legend().set_visible(False) if ax.get_legend() else None
    ax.set_yticks(np.arange(0, 1.25, 0.25))
    ax.set_xticks(xticks)
    ax.set_xticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("Displacement [mm]", fontsize=12, fontweight="bold", fontfamily="Arial")
    ax.set_title("Stimulus - Whisker Displacement", fontweight="bold", fontsize=16, fontfamily="Arial")
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    # Subplot 3: Stimulus velocity (derivative, first 21 s, black line matching MATLAB)
    ax = axes1[2]
    stimDeriv = stim.derivative
    stimDerivView = stimDeriv.getSigInTimeWindow(0, viewWindow)
    ax.plot(stimDerivView.time, stimDerivView.data[:, 0], "k")
    ax.set_yticks(np.arange(-80, 81, 40))
    ax.set_xticks(xticks)
    ax.set_xlim(0, viewWindow)
    ax.set_ylim(-80, 80)
    ax.set_ylabel("Displacement Velocity [mm/s]", fontsize=12, fontweight="bold", fontfamily="Arial")
    ax.set_xlabel("time [s]", fontsize=12, fontweight="bold", fontfamily="Arial")
    ax.set_title("Displacement Velocity", fontweight="bold", fontsize=16, fontfamily="Arial")
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    fig1.tight_layout()
    figure_files.extend(_maybe_export(
        fig1, export_dir, "fig01_data_overview"))

    # ==================================================================
    # Fit baseline-only model
    # ==================================================================
    print("\n--- Fitting baseline-only model ---")
    cfgBase = TrialConfig([("Baseline", "constant")], sampleRate, [], [])
    cfgBase.setName("Baseline")
    baselineResults = Analysis.RunAnalysisForAllNeurons(
        trial, ConfigColl([cfgBase]), 0)

    # ==================================================================
    # Compute residual cross-covariance with stimulus to find optimal lag
    # ==================================================================
    print("--- Computing residual cross-covariance ---")
    residual = baselineResults.computeFitResidual()
    xcovSig = residual.xcov(stim)

    # Window to positive lags [0, 1] s (matching Matlab)
    xcovWindowed = xcovSig.windowedSignal([0, 1])

    # Find peak lag — findGlobalPeak returns (times, values)
    peakTimes, peakVals = xcovWindowed.findGlobalPeak("maxima")
    shiftTime = float(peakTimes[0])
    peakVal = float(peakVals[0])
    print(f"  Peak xcov at lag = {shiftTime:.4f} s (value = {peakVal:.4f})")

    # ==================================================================
    # Shift stimulus by optimal lag and build new Trial
    # ==================================================================
    # Matlab: stimShifted = Covariate(time, stimData, ...).shift(shiftTime)
    # Note: Matlab uses raw stimData (not /10) with units 'V' for the shifted version
    stimShifted = Covariate(
        time, stimData,
        "Stimulus", "time", "s", "V",
        dataLabels=["stim"],
    )
    stimShifted = stimShifted.shift(shiftTime)

    baselineMu = Covariate(
        time, np.ones((len(time), 1)),
        "Baseline", "time", "s", "",
        dataLabels=["\\mu"],
    )

    trialShifted = Trial(
        nstColl(nspikeTrain(spikeTimes)),
        CovColl([stimShifted, baselineMu]),
    )

    # ==================================================================
    # History model-order search via computeHistLagForAll
    # ==================================================================
    print("\n--- Sweeping history windows ---")
    delta = 1.0 / sampleRate
    maxWindow = 1.0
    numWindows = 32

    # Construct log-spaced history window boundaries (matching Matlab)
    logVals = np.logspace(np.log10(delta), np.log10(maxWindow), numWindows)
    windowTimes = np.concatenate([[0.0], logVals])
    # Round to nearest ms and remove duplicates
    windowTimes = np.unique(np.round(windowTimes * sampleRate) / sampleRate)

    print(f"  Window boundaries: {len(windowTimes)} unique values")
    print(f"  Range: [{windowTimes[0]:.4f}, {windowTimes[-1]:.4f}] s")

    historySweep = Analysis.computeHistLagForAll(
        trialShifted, windowTimes,
        CovLabels=[("Baseline", "\\mu"), ("Stimulus", "stim")],
        Algorithm="GLM",
        batchMode=0,
        sampleRate=sampleRate,
        makePlot=0,
    )

    # ==================================================================
    # Select optimal history order
    # ==================================================================
    # historySweep is a list of FitResult objects (one per neuron)
    sweep = historySweep[0]
    aicArr = np.asarray(sweep.AIC, dtype=float)
    bicArr = np.asarray(sweep.BIC, dtype=float)
    ksArr = np.asarray(sweep.KSStats, dtype=float).ravel()

    # Delta AIC/BIC relative to no-history model (index 0)
    dAIC = aicArr[1:] - aicArr[0]
    dBIC = bicArr[1:] - bicArr[0]

    # Find index of minimum delta (offset by +1 since we skipped index 0)
    aicIdx = int(np.argmin(dAIC)) + 1 if dAIC.size > 0 else None
    bicIdx = int(np.argmin(dBIC)) + 1 if dBIC.size > 0 else None
    ksIdx = int(np.argmin(ksArr)) if ksArr.size > 0 else 0

    # Take minimum of AIC and BIC optimal indices
    candidates = []
    if aicIdx is not None and aicIdx > 0:
        candidates.append(aicIdx)
    if bicIdx is not None and bicIdx > 0:
        candidates.append(bicIdx)
    windowIndex = min(candidates) if candidates else ksIdx

    if windowIndex > len(windowTimes):
        windowIndex = ksIdx

    # Extract selected history windows
    # windowIndex is 0-based; MATLAB uses windowTimes(1:windowIndex) with 1-based
    # indexing, which includes windowIndex elements.  Python equivalent is [:windowIndex+1].
    if windowIndex > 1:
        selectedHistory = list(windowTimes[:windowIndex + 1])
    else:
        selectedHistory = []

    print(f"  AIC optimal index: {aicIdx}")
    print(f"  BIC optimal index: {bicIdx}")
    print(f"  KS optimal index:  {ksIdx}")
    print(f"  Selected window index: {windowIndex}")
    print(f"  Selected history: {len(selectedHistory)} windows")

    # ==================================================================
    # Final 3-model comparison
    # ==================================================================
    print("\n--- Fitting 3 nested models ---")

    cfg1 = TrialConfig([("Baseline", "\\mu")], sampleRate, [], [])
    cfg1.setName("Baseline")

    cfg2 = TrialConfig(
        [("Baseline", "\\mu"), ("Stimulus", "stim")],
        sampleRate, [], [],
    )
    cfg2.setName("Baseline+Stimulus")

    cfg3 = TrialConfig(
        [("Baseline", "\\mu"), ("Stimulus", "stim")],
        sampleRate, selectedHistory, [],
    )
    cfg3.setName("Baseline+Stimulus+Hist")

    modelCompare = Analysis.RunAnalysisForAllNeurons(
        trialShifted, ConfigColl([cfg1, cfg2, cfg3]), 0)
    modelCompare.lambda_signal.setDataLabels([
        "\\lambda_{const}",
        "\\lambda_{const+stim}",
        "\\lambda_{const+stim+hist}",
    ])

    print(f"  AIC: {modelCompare.AIC}")
    print(f"  BIC: {modelCompare.BIC}")

    # ==================================================================
    # Figure 2: Lag selection, history diagnostics, KS, coefficients
    # (Matlab uses subplot(7,2,...) layout)
    # ==================================================================
    fig2 = plt.figure(figsize=(14, 9))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(7, 2, figure=fig2, hspace=0.5, wspace=0.3)

    numResults = len(ksArr)

    # --- Left column, rows 1-3: Cross-correlation function ---
    ax_xcov = fig2.add_subplot(gs[0:3, 0])
    xcovWindowed.plot(handle=ax_xcov)
    ax_xcov.plot(shiftTime, peakVal, "ro", linewidth=3,
                 markerfacecolor="r", markeredgecolor="r")
    ax_xcov.set_title(
        f"Cross Correlation Function - Peak at t={shiftTime:g} sec",
        fontweight="bold", fontsize=12, fontfamily="Arial")
    ax_xcov.set_xlabel("Lag [s]", fontsize=12, fontweight="bold", fontfamily="Arial")
    ax_xcov.set_ylabel("")

    # --- Right column, row 1: KS statistic vs Q ---
    ax_ks_sweep = fig2.add_subplot(gs[0, 1])
    xvals = np.arange(numResults)
    ax_ks_sweep.plot(xvals, ksArr, ".-")
    if windowIndex < numResults:
        ax_ks_sweep.plot(xvals[windowIndex], ksArr[windowIndex], "r*")
    ax_ks_sweep.set_xlim(xvals[0], xvals[-1])
    ax_ks_sweep.set_xticks(np.arange(0, numResults, 5))
    ax_ks_sweep.set_xticklabels([])
    ax_ks_sweep.tick_params(length=4, which="major")
    ax_ks_sweep.minorticks_on()
    ax_ks_sweep.set_ylabel("KS Statistic")
    ax_ks_sweep.set_title("Model Selection via change\nin KS Statistic, AIC, and BIC",
                          fontweight="bold", fontsize=12, fontfamily="Arial")

    # --- Right column, row 2: Delta AIC vs Q ---
    ax_daic = fig2.add_subplot(gs[1, 1])
    dAIC_full = aicArr - aicArr[0]
    ax_daic.plot(np.arange(len(dAIC_full)), dAIC_full, ".-")
    if windowIndex < len(dAIC_full):
        ax_daic.plot(windowIndex, dAIC_full[windowIndex], "r*")
    ax_daic.set_xlim(0, numResults - 1)
    ax_daic.set_xticks(np.arange(0, numResults, 5))
    ax_daic.set_xticklabels([])
    ax_daic.tick_params(length=4, which="major")
    ax_daic.minorticks_on()
    ax_daic.set_ylabel(r"$\Delta$ AIC")

    # --- Right column, row 3: Delta BIC vs Q ---
    ax_dbic = fig2.add_subplot(gs[2, 1])
    dBIC_full = bicArr - bicArr[0]
    ax_dbic.plot(np.arange(len(dBIC_full)), dBIC_full, ".-")
    if windowIndex < len(dBIC_full):
        ax_dbic.plot(windowIndex, dBIC_full[windowIndex], "r*")
    ax_dbic.set_xlim(0, numResults - 1)
    ax_dbic.set_xticks(np.arange(0, numResults, 5))
    ax_dbic.tick_params(length=4, which="major")
    ax_dbic.minorticks_on()
    ax_dbic.set_xlabel("# History Windows, Q",
                       fontsize=12, fontweight="bold", fontfamily="Arial")
    ax_dbic.set_ylabel(r"$\Delta$ BIC",
                       fontsize=12, fontweight="bold", fontfamily="Arial")

    # --- Left column, rows 5-7: KS plot (3 models) ---
    ax_ks = fig2.add_subplot(gs[4:7, 0])
    modelCompare.KSPlot(handle=ax_ks)

    # --- Right column, rows 5-7: Coefficient comparison ---
    ax_coeff = fig2.add_subplot(gs[4:7, 1])
    modelCompare.plotCoeffs(handle=ax_coeff)
    if ax_coeff.get_legend():
        ax_coeff.get_legend().set_visible(False)
    figure_files.extend(_maybe_export(
        fig2, export_dir, "fig02_lag_and_model_comparison"))

    if visible:
        plt.show()

    print(f"\nExample 02 complete. Generated {len(figure_files)} figure(s).")
    return figure_files


# =========================================================================
# CLI entry point
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example 02: Whisker Stimulus GLM")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT,
                        help="Repository root (default: auto-detected).")
    parser.add_argument("--export-figures", action="store_true",
                        help="Export figures to disk.")
    parser.add_argument("--export-dir", type=Path, default=None,
                        help="Directory for exported figures.")
    parser.add_argument("--no-display", action="store_true",
                        help="Run without displaying figures (headless).")
    args = parser.parse_args()

    export_dir = args.export_dir
    if args.export_figures and export_dir is None:
        export_dir = THIS_DIR / "figures" / "example02"

    run_example02(
        export_figures=args.export_figures,
        export_dir=export_dir if args.export_figures else None,
        visible=not args.no_display,
    )
