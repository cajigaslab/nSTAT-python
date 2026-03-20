#!/usr/bin/env python3
"""Example 01 — mEPSC Poisson Models Under Constant and Washout Magnesium.

This example demonstrates:
  1) Homogeneous Poisson modeling for constant Mg2+ conditions.
  2) Piecewise baseline modeling under Mg2+ washout conditions.
  3) Model comparison using KS plots, time-rescaling diagnostics, and
     estimated conditional intensity functions.

Data provenance:
  Uses installer-downloaded nSTAT example data from ``data/mEPSCs``:
  ``epsc2.txt``, ``washout1.txt``, ``washout2.txt``

Expected outputs:
  - Figure 1: Constant Mg2+ raster + diagnostics + lambda estimate.
  - Figure 2: Constant vs decreasing Mg2+ raster overview.
  - Figure 3: Piecewise model diagnostics and lambda comparison.

Paper mapping:
  Section 2.3.1 (mEPSC analysis); Figs. 3 and 10 (nSTAT paper, 2012).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
# Helper: load mEPSC spike times from text file
# =========================================================================
def _load_mepsc_times_seconds(path: Path) -> np.ndarray:
    """Load spike times from mEPSC text file, returning times in seconds."""
    data = np.loadtxt(path, skiprows=1)
    # Column 2 is spike time in milliseconds at 1000 Hz
    times_ms = data[:, 1] if data.ndim == 2 else data
    return times_ms / 1000.0


def _matlab_colon(start: float, step: float, stop: float) -> np.ndarray:
    """Replicate MATLAB ``start:step:stop`` exactly.

    ``np.arange`` accumulates floating-point error over many steps and can
    produce off-by-one length mismatches.  This function computes the element
    count first (like MATLAB's colon operator), then multiplies by integer
    indices — giving bit-exact parity.
    """
    n = int(np.floor((stop - start) / step)) + 1
    return start + np.arange(n) * step


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
def run_example01(*, export_figures: bool = False, export_dir: Path | None = None,
                  visible: bool = True):
    """Run Example 01: mEPSC Poisson models."""

    if not visible:
        matplotlib.use("Agg")

    data_dir = ensure_example_data(download=True)
    mepsc_dir = data_dir / "mEPSCs"
    figure_files: list[Path] = []

    sampleRate = 1000  # Hz

    # ==================================================================
    # Part 1: Constant magnesium concentration — Homogeneous Poisson model
    # ==================================================================
    print("=== Part 1: Constant Mg2+ — Homogeneous Poisson ===")

    epsc2 = _load_mepsc_times_seconds(mepsc_dir / "epsc2.txt")

    # Create spike train and time vector
    nstConst = nspikeTrain(epsc2)
    timeConst = _matlab_colon(0, 1.0 / sampleRate, nstConst.maxTime)

    # Create baseline covariate
    baseline = Covariate(
        timeConst,
        np.ones((len(timeConst), 1)),
        "Baseline", "time", "s", "",
        dataLabels=["\\mu"],
    )
    covarColl = CovColl([baseline])
    spikeCollConst = nstColl(nstConst)
    trialConst = Trial(spikeCollConst, covarColl)

    # Configure: single constant-rate model
    tcConst = TrialConfig([("Baseline", "\\mu")], sampleRate, [])
    tcConst.setName("Constant Baseline")
    configConst = ConfigColl([tcConst])

    # Fit GLM
    resultConst = Analysis.RunAnalysisForAllNeurons(trialConst, configConst, 0)
    resultConst.lambda_signal.setDataLabels(["\\lambda_{const}"])

    print(f"  Spikes: {len(epsc2)}")
    print(f"  AIC: {resultConst.AIC}")
    print(f"  BIC: {resultConst.BIC}")

    # --- Figure 1: Constant Mg2+ diagnostics (Matlab-matching 2x2 layout) ---
    # Matlab uses subplot(2,2,...) with: raster, InvGausTrans, KSPlot, lambda
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 9))

    # (2,2,1): Neural raster
    ax = axes1[0, 0]
    spikeCollConst.plot(handle=ax)
    ax.set_title("Neural Raster with constant Mg$^{2+}$ Concentration",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("time [s]", fontname="Arial", fontsize=12, fontweight="bold")
    ax.set_ylabel("mEPSCs", fontname="Arial", fontsize=12, fontweight="bold")
    ax.set_yticks([0, 1])

    # (2,2,2): Inverse Gaussian transform (ACF)
    resultConst.plotInvGausTrans(fit_num=None, handle=axes1[0, 1])

    # (2,2,3): KS plot
    resultConst.KSPlot(fit_num=None, handle=axes1[1, 0])

    # (2,2,4): Lambda estimate
    ax = axes1[1, 1]
    lam = resultConst.lambda_signal
    ax.plot(np.asarray(lam.time, dtype=float),
            np.asarray(lam.data[:, 0], dtype=float),
            "b", linewidth=2)
    ax.set_xlabel("time [s]", fontname="Arial", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"$\lambda(t)$ [Hz]", fontname="Arial", fontsize=12, fontweight="bold")
    ax.legend(["$\\lambda_{const}$"], loc="upper right", fontsize=14)

    fig1.tight_layout()
    figure_files.extend(_maybe_export(fig1, export_dir, "fig01_constant_mg_summary"))

    # ==================================================================
    # Part 2: Varying magnesium concentration — Piecewise baseline model
    # ==================================================================
    print("\n=== Part 2: Decreasing Mg2+ — Piecewise Baseline ===")

    washout1 = _load_mepsc_times_seconds(mepsc_dir / "washout1.txt")
    washout2 = _load_mepsc_times_seconds(mepsc_dir / "washout2.txt")

    spikeTimes1 = 260.0 + washout1
    spikeTimes2 = np.sort(washout2) + 745.0
    nstWashout = nspikeTrain(np.concatenate([spikeTimes1, spikeTimes2]))
    timeWashout = _matlab_colon(260.0, 1.0 / sampleRate, nstWashout.maxTime)

    # --- Figure 2: Constant vs Decreasing Mg2+ rasters ---
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 9))

    ax = axes2[0]
    nstConst.plot(handle=ax)
    ax.set_yticks([0, 1])
    ax.set_ylabel("mEPSCs", fontname="Arial", fontsize=12, fontweight="bold")
    ax.set_title("Neural Raster with constant Mg$^{2+}$ Concentration",
                 fontweight="bold", fontsize=12)

    ax = axes2[1]
    nstWashout.plot(handle=ax)
    ax.set_yticks([0, 1])
    ax.set_ylabel("mEPSCs", fontname="Arial", fontsize=12, fontweight="bold")
    ax.set_title("Neural Raster with decreasing Mg$^{2+}$ Concentration",
                 fontweight="bold", fontsize=12)

    fig2.tight_layout()
    figure_files.extend(_maybe_export(fig2, export_dir, "fig02_washout_raster_overview"))

    # ==================================================================
    # Part 3: Piecewise baseline model and model comparison
    # ==================================================================
    print("\n=== Part 3: Piecewise Baseline Model Comparison ===")

    # Build piecewise indicator covariates
    # Matlab: timeInd1 = find(time < 495, 1, 'last')  → last 1-based index < 495
    # Equivalent Python: first 0-based index >= 495 (searchsorted side='left'),
    # so rate1[:idx] covers [260, 494.999] and rate2[idx:] starts at 495.
    timeInd1 = np.searchsorted(timeWashout, 495.0, side="left")
    timeInd2 = np.searchsorted(timeWashout, 765.0, side="left")
    N = len(timeWashout)

    constantRate = np.ones((N, 1))
    rate1 = np.zeros((N, 1))
    rate2 = np.zeros((N, 1))
    rate3 = np.zeros((N, 1))
    rate1[:timeInd1] = 1.0
    rate2[timeInd1:timeInd2] = 1.0
    rate3[timeInd2:] = 1.0

    baselineWashout = Covariate(
        timeWashout,
        np.column_stack([constantRate, rate1, rate2, rate3]),
        "Baseline", "time", "s", "",
        dataLabels=["\\mu", "\\mu_{1}", "\\mu_{2}", "\\mu_{3}"],
    )

    spikeCollWashout = nstColl(nstWashout)
    trialWashout = Trial(spikeCollWashout, CovColl([baselineWashout]))

    # Configure: (1) constant baseline, (2) piecewise baseline
    tc1 = TrialConfig([("Baseline", "\\mu")], sampleRate, [])
    tc1.setName("Constant Baseline")
    tc2 = TrialConfig([("Baseline", "\\mu_{1}", "\\mu_{2}", "\\mu_{3}")], sampleRate, [])
    tc2.setName("Diff Baseline")
    configWashout = ConfigColl([tc1, tc2])

    resultWashout = Analysis.RunAnalysisForAllNeurons(trialWashout, configWashout, 0)
    resultWashout.lambda_signal.setDataLabels(["\\lambda_{const}", "\\lambda_{const-epoch}"])

    print(f"  AIC: {resultWashout.AIC}")
    print(f"  BIC: {resultWashout.BIC}")

    # --- Figure 3: Piecewise model diagnostics (Matlab-matching 2x2 layout) ---
    # Matlab uses subplot(2,2,...) with: raster+epoch lines, InvGausTrans, KSPlot, lambda comparison
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 9))

    # (2,2,1): Neural raster with epoch boundary lines
    ax = axes3[0, 0]
    spikeCollWashout.plot(handle=ax)
    ax.set_title("Neural Raster with decreasing Mg$^{2+}$ Concentration",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("time [s]", fontsize=12, fontweight="bold")
    ax.set_yticklabels([])
    ax.axvline(495, color="r", linewidth=4)
    ax.axvline(765, color="r", linewidth=4)

    # (2,2,2): Inverse Gaussian transform (ACF) — all fits
    resultWashout.plotInvGausTrans(fit_num=None, handle=axes3[0, 1])

    # (2,2,3): KS plot — all fits
    resultWashout.KSPlot(fit_num=None, handle=axes3[1, 0])

    # (2,2,4): Lambda comparison (two models overlaid)
    ax = axes3[1, 1]
    lam = resultWashout.lambda_signal
    t = np.asarray(lam.time, dtype=float)
    ax.plot(t, np.asarray(lam.data[:, 0], dtype=float), "b", linewidth=2)
    if lam.data.shape[1] > 1:
        ax.plot(t, np.asarray(lam.data[:, 1], dtype=float), "g", linewidth=2)
    ax.set_ylim(0, 5)
    ax.set_xlabel("time [s]", fontname="Arial", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"$\lambda(t)$ [Hz]", fontname="Arial", fontsize=12, fontweight="bold")
    ax.legend(["$\\lambda_{const}$", "$\\lambda_{const-epoch}$"],
              loc="upper right", fontsize=14)

    fig3.tight_layout()
    figure_files.extend(_maybe_export(fig3, export_dir, "fig03_piecewise_baseline_comparison"))

    if visible:
        plt.show()

    print(f"\nExample 01 complete. Generated {len(figure_files)} figure(s).")
    return figure_files


# =========================================================================
# CLI entry point
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example 01: mEPSC Poisson Models")
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
        export_dir = THIS_DIR / "figures" / "example01"

    run_example01(
        export_figures=args.export_figures,
        export_dir=export_dir if args.export_figures else None,
        visible=not args.no_display,
    )
