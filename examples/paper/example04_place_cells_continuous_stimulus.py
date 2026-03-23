#!/usr/bin/env python3
"""Example 04 — Place-Cell Receptive Fields (Gaussian vs Zernike).

This example demonstrates:
  1) Loading hippocampal place-cell data from two animals.
  2) Visualising spike locations overlaid on the animal's path.
  3) Loading precomputed Gaussian and Zernike polynomial receptive-field fits.
  4) Comparing model families via KS, AIC, and BIC statistics using FitSummary.
  5) Generating 2-D heatmaps of place fields for all neurons.
  6) Generating 3-D mesh comparison for selected example cells.

Data provenance:
  Uses ``data/PlaceCellDataAnimal{1,2}.mat`` (trajectories + spike times)
  and ``PlaceCellAnimal{1,2}Results.mat`` (precomputed FitResult structures).

Expected outputs:
  - Figure 1: Example cells — spike locations over path (4 cells per animal).
  - Figure 2: Population model-comparison statistics (dKS, dAIC, dBIC).
  - Figure 3: Gaussian receptive-field heatmaps (Animal 1).
  - Figure 4: Zernike receptive-field heatmaps (Animal 1).
  - Figure 5: Gaussian receptive-field heatmaps (Animal 2).
  - Figure 6: Zernike receptive-field heatmaps (Animal 2).
  - Figure 7: 3-D mesh comparison for selected example cells.

Paper mapping:
  Section 2.3.5 (place-cell continuous-stimulus analysis).
"""
from __future__ import annotations

import argparse
import math
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
    Covariate,
    FitResult,
    FitSummary,
    TrialConfig,
    ConfigCollection,
)
from nstat.core import nspikeTrain  # noqa: E402
from nstat.data_manager import ensure_example_data  # noqa: E402
from nstat.zernike import zernike_basis_from_cartesian  # noqa: E402


# =====================================================================
# Helpers
# =====================================================================
def _load_animal_data(path):
    """Load place-cell trajectory and spike data from a .mat file."""
    d = loadmat(str(path), squeeze_me=True)
    x = np.asarray(d["x"], dtype=float).ravel()
    y = np.asarray(d["y"], dtype=float).ravel()
    time = np.asarray(d["time"], dtype=float).ravel()
    neurons = np.asarray(d["neuron"], dtype=object).ravel()
    return x, y, time, neurons


def _load_animal_results(path, x, y, time, neurons):
    """Load precomputed FitResult structures and reconstruct Python FitResults."""
    d = loadmat(str(path), squeeze_me=True)
    res_structs = np.asarray(d["resStruct"], dtype=object).ravel()
    fit_results = []

    for i, rs in enumerate(res_structs):
        # Extract lambda signal
        lam = rs["lambda"].item()
        lam_time = np.asarray(lam["time"].item(), dtype=float).ravel()
        lam_data = np.asarray(lam["data"].item(), dtype=float)
        lam_name = str(lam["name"].item()) if lam["name"].size else "\\lambda"

        lambda_cov = Covariate(
            lam_time, lam_data, lam_name, "time", "s", "spikes/sec",
        )

        # Extract coefficients
        b_raw = rs["b"].item()
        if b_raw.dtype == object:
            b_list = [np.asarray(b_raw[j], dtype=float).ravel() for j in range(b_raw.size)]
        else:
            b_list = [np.asarray(b_raw, dtype=float).ravel()]

        numResults = int(np.asarray(rs["numResults"].item()).ravel()[0])

        # Extract AIC/BIC/logLL
        AIC = np.asarray(rs["AIC"].item(), dtype=float).ravel()
        BIC = np.asarray(rs["BIC"].item(), dtype=float).ravel()
        logLL = np.asarray(rs["logLL"].item(), dtype=float).ravel()

        # Config names
        cn_raw = rs["configNames"].item()
        if isinstance(cn_raw, np.ndarray):
            config_names = [str(c) for c in cn_raw.ravel()]
        else:
            config_names = [str(cn_raw)]

        # Covariate labels
        if "covLabels" in rs.dtype.names:
            cl_raw = rs["covLabels"].item()
            cl = cl_raw
        else:
            cl = []
        if isinstance(cl, np.ndarray) and cl.dtype == object:
            cov_labels = []
            for j in range(cl.size):
                item = cl[j]
                if isinstance(item, np.ndarray):
                    cov_labels.append([str(x) for x in item.ravel()])
                else:
                    cov_labels.append([str(item)])
        elif isinstance(cl, str):
            cov_labels = [[cl]] * numResults
        else:
            cov_labels = [config_names] * numResults

        # Create spike train for this neuron
        st = np.asarray(neurons[i]["spikeTimes"].item(), dtype=float).ravel()
        nst = nspikeTrain(st, name=str(i + 1),
                          minTime=float(time[0]), maxTime=float(time[-1]),
                          makePlots=-1)

        # numHist
        if "numHist" in rs.dtype.names:
            nh = rs["numHist"].item()
            num_hist = list(np.asarray(nh, dtype=int).ravel())
        else:
            num_hist = [0] * numResults

        cfgs = ConfigCollection([TrialConfig(name=n) for n in config_names])

        fr = FitResult(
            nst,
            cov_labels,
            num_hist,
            [],              # histObjects
            [],              # ensHistObjects
            lambda_cov,
            b_list,
            [0.0] * numResults,  # dev
            [None] * numResults, # stats
            AIC,
            BIC,
            logLL,
            cfgs,
            [],              # XvalData
            [],              # XvalTime
            "poisson",
        )

        # Load KS statistics if available
        if "KSStats" in rs.dtype.names:
            ks_struct = rs["KSStats"].item()
            if hasattr(ks_struct, "dtype") and ks_struct.dtype.names:
                ks_stat = np.asarray(ks_struct["ks_stat"].item(), dtype=float).ravel()
                pval = np.asarray(ks_struct["pValue"].item(), dtype=float).ravel()
                within = np.asarray(ks_struct["withinConfInt"].item(), dtype=float).ravel()
                if ks_stat.size >= numResults:
                    fr.KSStats = ks_stat[:numResults].reshape(numResults, 1)
                    fr.KSPvalues = pval[:numResults]
                    fr.withinConfInt = within[:numResults]

        fit_results.append(fr)

    return fit_results


def _compute_place_field(coeffs, grid_design, grid_shape, sample_rate=1.0):
    """Compute predicted firing rate on a spatial grid.

    Matches Matlab ``FitResult.evalLambda`` which computes
    ``exp(X * b) * sampleRate`` to convert from conditional intensity
    (per bin) to firing rate (Hz).
    """
    eta = grid_design @ coeffs
    rate = np.exp(eta) * sample_rate
    return rate.reshape(grid_shape)


# =====================================================================
# Main example
# =====================================================================
def run_example04(*, export_figures: bool = False, export_dir: Path | None = None):
    """Run Example 04: Place-cell receptive fields."""
    print("=== Example 04: Place-Cell Receptive Fields ===")

    data_dir = ensure_example_data(download=True)
    if export_dir is None:
        export_dir = THIS_DIR / "figures" / "example04"

    # ==================================================================
    # 1. Load data for both animals
    # ==================================================================
    x1, y1, t1, neurons1 = _load_animal_data(
        data_dir / "Place Cells" / "PlaceCellDataAnimal1.mat")
    x2, y2, t2, neurons2 = _load_animal_data(
        data_dir / "Place Cells" / "PlaceCellDataAnimal2.mat")
    nCells1 = len(neurons1)
    nCells2 = len(neurons2)
    print(f"  Animal 1: {nCells1} cells, {len(t1)} time points")
    print(f"  Animal 2: {nCells2} cells, {len(t2)} time points")

    # ==================================================================
    # 2. Load precomputed FitResults
    # ==================================================================
    fitResults1 = _load_animal_results(
        data_dir / "PlaceCellAnimal1Results.mat", x1, y1, t1, neurons1)
    fitResults2 = _load_animal_results(
        data_dir / "PlaceCellAnimal2Results.mat", x2, y2, t2, neurons2)
    print(f"  Loaded {len(fitResults1)} + {len(fitResults2)} FitResult objects")

    # ==================================================================
    # 3. Build FitSummary for each animal
    # ==================================================================
    summary1 = FitSummary(fitResults1)
    summary2 = FitSummary(fitResults2)

    # Delta statistics
    # dKS: direct subtraction Gaussian - Zernike (matches Matlab line 81-83)
    dKS1 = summary1.KSStats[:, 0] - summary1.KSStats[:, 1]
    dKS2 = summary2.KSStats[:, 0] - summary2.KSStats[:, 1]

    # dAIC/dBIC: Matlab uses getDiffAIC(1) / getDiffBIC(1) which computes
    # Zernike - Gaussian (other columns minus reference column).
    dAIC1 = summary1.AIC[:, 1] - summary1.AIC[:, 0]
    dBIC1 = summary1.BIC[:, 1] - summary1.BIC[:, 0]

    dAIC2 = summary2.AIC[:, 1] - summary2.AIC[:, 0]
    dBIC2 = summary2.BIC[:, 1] - summary2.BIC[:, 0]

    dAIC_all = np.concatenate([dAIC1, dAIC2])
    dBIC_all = np.concatenate([dBIC1, dBIC2])
    dKS_all = np.concatenate([dKS1, dKS2])

    print(f"  Mean dKS  (Gauss-Zern): {np.nanmean(dKS_all):.4f}")
    print(f"  Mean dAIC (Zern-Gauss): {np.nanmean(dAIC_all):.2f}")
    print(f"  Mean dBIC (Zern-Gauss): {np.nanmean(dBIC_all):.2f}")

    # ==================================================================
    # Figure 1: Example cells — spike locations over path (2x2)
    # ==================================================================
    exampleCells = [1, 20, 24, 48]  # 0-indexed (MATLAB: [2 21 25 49])
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
    for i, cidx in enumerate(exampleCells):
        ax = axes1.flat[i]
        h1, = ax.plot(x1, y1, "b-", linewidth=0.5)
        n = neurons1[min(cidx, nCells1 - 1)]
        xn = np.asarray(n["xN"].item(), dtype=float).ravel()
        yn = np.asarray(n["yN"].item(), dtype=float).ravel()
        h2, = ax.plot(xn, yn, "r.", markersize=7)
        ax.set_title(f"Cell#{cidx + 1}", fontweight="bold", fontsize=12)
        ax.set_xticks(np.arange(-1, 1.5, 0.5))
        ax.set_yticks(np.arange(-1, 1.5, 0.5))
        ax.set_aspect("equal")
        ax.axis("square")
        if i == 3:
            ax.legend([h1, h2], ["Animal Path", "Location at time of spike"])
    fig1.tight_layout()

    # ==================================================================
    # Figure 2: Population statistics (1x3 box plots)
    # ==================================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))

    axes2[0].boxplot([dKS1[np.isfinite(dKS1)], dKS2[np.isfinite(dKS2)]],
                     tick_labels=["Animal 1", "Animal 2"],
                     vert=True)
    axes2[0].set_title(r"$\Delta$ KS Statistic", fontsize=14, fontweight="bold",
                       fontfamily="Arial")

    axes2[1].boxplot([dAIC1[np.isfinite(dAIC1)], dAIC2[np.isfinite(dAIC2)]],
                     tick_labels=["Animal 1", "Animal 2"],
                     vert=True)
    axes2[1].set_title(r"$\Delta$ AIC", fontsize=14, fontweight="bold",
                       fontfamily="Arial")

    axes2[2].boxplot([dBIC1[np.isfinite(dBIC1)], dBIC2[np.isfinite(dBIC2)]],
                     tick_labels=["Animal 1", "Animal 2"],
                     vert=True)
    axes2[2].set_title(r"$\Delta$ BIC", fontsize=14, fontweight="bold",
                       fontfamily="Arial")

    fig2.tight_layout()

    # ==================================================================
    # 4. Build spatial grids and design matrices for heatmaps
    # ==================================================================
    grid_res = 201  # Matlab: meshgrid(-1:0.01:1) → 201 points
    xGrid = np.linspace(-1, 1, grid_res)
    yGrid = np.linspace(-1, 1, grid_res)
    xx, yy = np.meshgrid(xGrid, yGrid)
    yy = np.flipud(yy)   # Matlab: y increases bottom-to-top
    xx = np.fliplr(xx)    # Matlab: x increases right-to-left
    xf, yf = xx.ravel(), yy.ravel()

    # Gaussian design: [1, x, y, x^2, y^2, xy] (intercept prepended)
    gridDesignGauss = np.column_stack([
        np.ones(xf.size), xf, yf, xf**2, yf**2, xf * yf
    ])

    # Zernike design: [1, z1, z2, ..., z9] (intercept prepended)
    zBasis = zernike_basis_from_cartesian(xf, yf, fill_value=0.0)
    gridDesignZern = np.column_stack([np.ones(xf.size), zBasis])

    # ==================================================================
    # Figures 3-6: Place field heatmaps
    # ==================================================================
    def _plot_heatmaps(fit_results, nCells, title_prefix, design_gauss,
                       design_zern, grid_shape):
        nRows = math.ceil(nCells / 7)
        nCols = 7

        figG, axesG = plt.subplots(nRows, nCols, figsize=(14, 2 * nRows))
        figZ, axesZ = plt.subplots(nRows, nCols, figsize=(14, 2 * nRows))
        if nRows == 1:
            axesG = axesG[np.newaxis, :]
            axesZ = axesZ[np.newaxis, :]

        for i in range(nCells):
            row, col = divmod(i, nCols)
            fr = fit_results[i]
            sr = float(fr.lambda_signal.sampleRate)
            coeffs_g = np.asarray(fr.b[0], dtype=float).ravel()
            coeffs_z = np.asarray(fr.b[1], dtype=float).ravel() if fr.numResults > 1 else coeffs_g

            # Gaussian field
            ax = axesG[row, col]
            try:
                field_g = _compute_place_field(coeffs_g, design_gauss[:, :coeffs_g.size], grid_shape, sr)
                ax.pcolormesh(xx, yy, field_g, shading="gouraud", cmap="jet")
            except Exception:
                pass
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

            # Zernike field
            ax = axesZ[row, col]
            try:
                field_z = _compute_place_field(coeffs_z, design_zern[:, :coeffs_z.size], grid_shape, sr)
                ax.pcolormesh(xx, yy, field_z, shading="gouraud", cmap="jet")
            except Exception:
                pass
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

        # Hide unused subplots
        for i in range(nCells, nRows * nCols):
            row, col = divmod(i, nCols)
            axesG[row, col].set_visible(False)
            axesZ[row, col].set_visible(False)

        # MATLAB: sgtitle('Gaussian Place Fields - Animal#N')
        animal_num = title_prefix.replace("Animal ", "")
        figG.suptitle(f"Gaussian Place Fields - Animal#{animal_num}",
                      fontweight="bold", fontsize=12)
        figZ.suptitle(f"Zernike Place Fields - Animal#{animal_num}",
                      fontweight="bold", fontsize=12)
        figG.tight_layout()
        figZ.tight_layout()
        return figG, figZ

    figG1, figZ1 = _plot_heatmaps(
        fitResults1, nCells1, "Animal 1",
        gridDesignGauss, gridDesignZern, xx.shape,
    )
    figG2, figZ2 = _plot_heatmaps(
        fitResults2, nCells2, "Animal 2",
        gridDesignGauss, gridDesignZern, xx.shape,
    )
    print("  Figures 3-6: Place field heatmaps")

    # ==================================================================
    # Figure 7: 3-D mesh comparison for an example cell
    # ==================================================================
    exampleCell = min(24, nCells1 - 1)  # 0-indexed → cell 25 in Matlab
    fr_ex = fitResults1[exampleCell]
    sr_ex = float(fr_ex.lambda_signal.sampleRate)
    coeffs_g = np.asarray(fr_ex.b[0], dtype=float).ravel()
    coeffs_z = np.asarray(fr_ex.b[1], dtype=float).ravel()

    field_g = _compute_place_field(
        coeffs_g, gridDesignGauss[:, :coeffs_g.size], xx.shape, sr_ex)
    field_z = _compute_place_field(
        coeffs_z, gridDesignZern[:, :coeffs_z.size], xx.shape, sr_ex)

    fig7 = plt.figure(figsize=(14, 9))
    ax3d = fig7.add_subplot(111, projection="3d")
    # MATLAB: mesh(xGrid, yGrid, lambda, 'FaceAlpha',0.2, 'EdgeAlpha',0.2, 'EdgeColor','b'/'g')
    ax3d.plot_surface(xx, yy, field_g, alpha=0.2, edgecolor="b",
                      facecolor="blue", linewidth=0.2, rstride=5, cstride=5,
                      shade=False)
    ax3d.plot_surface(xx, yy, field_z, alpha=0.2, edgecolor="g",
                      facecolor="green", linewidth=0.2, rstride=5, cstride=5,
                      shade=False)
    # Overlay animal path at z=0 (MATLAB: 'k')
    ax3d.plot(x1, y1, np.zeros_like(x1), "k-", linewidth=0.3)
    # Overlay spike locations (MATLAB: 'r.')
    n_ex = neurons1[exampleCell]
    xn_ex = np.asarray(n_ex["xN"].item(), dtype=float).ravel()
    yn_ex = np.asarray(n_ex["yN"].item(), dtype=float).ravel()
    ax3d.scatter(xn_ex, yn_ex, np.zeros_like(xn_ex), c="r", s=5)
    ax3d.set_xlabel("x position")
    ax3d.set_ylabel("y position")
    ax3d.set_title(f"Animal#1, Cell#{exampleCell + 1}",
                   fontweight="bold", fontsize=12)
    # MATLAB legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor="blue", edgecolor="b", alpha=0.3,
              label=r"$\lambda_{Gaussian}$"),
        Patch(facecolor="green", edgecolor="g", alpha=0.3,
              label=r"$\lambda_{Zernike}$"),
        Line2D([0], [0], color="k", linewidth=0.5, label="Animal Path"),
        Line2D([0], [0], marker=".", color="r", linestyle="",
               label="Spike Locations"),
    ]
    ax3d.legend(handles=legend_elements, loc="best")

    print(f"  Figure 7: 3D mesh for cell {exampleCell + 1}")

    # ==================================================================
    # Save figures
    # ==================================================================
    all_figs = {
        "fig01_example_cells_path_overlay": fig1,
        "fig02_model_summary_statistics": fig2,
        "fig03_gaussian_place_fields_animal1": figG1,
        "fig04_zernike_place_fields_animal1": figZ1,
        "fig05_gaussian_place_fields_animal2": figG2,
        "fig06_zernike_place_fields_animal2": figZ2,
        "fig07_example_cell_mesh_comparison": fig7,
    }

    if export_figures:
        export_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in all_figs.items():
            path = export_dir / f"{name}.png"
            fig.savefig(str(path), dpi=250, facecolor="w", edgecolor="none")
            print(f"  Saved {path}")

    plt.show()
    print(f"\nExample 04 complete. Generated {len(all_figs)} figure(s).")
    return all_figs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example 04: Place-Cell Receptive Fields"
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--export-figures", action="store_true")
    parser.add_argument("--export-dir", type=Path, default=None)
    args = parser.parse_args()

    run_example04(
        export_figures=args.export_figures,
        export_dir=args.export_dir,
    )
