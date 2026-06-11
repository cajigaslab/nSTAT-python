#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"


LANGUAGE_METADATA = {
    "language_info": {
        "name": "python",
    }
}


def _write_notebook(path: Path, *, topic: str, expected_figures: int, markdown_note: str, code_cells: list[str]) -> None:
    notebook = new_notebook(
        cells=[
            new_markdown_cell(markdown_note),
            *[new_code_cell(dedent(cell).strip() + "\n") for cell in code_cells],
        ],
        metadata={
            **LANGUAGE_METADATA,
            "nstat": {
                "expected_figures": expected_figures,
                "run_group": "smoke",
                "style": "python-example",
                "topic": topic,
            },
        },
    )
    path.write_text(nbformat.writes(notebook), encoding="utf-8")


NOTE = """\
<!-- parity-note -->
## MATLAB Parity Note
- Source MATLAB helpfile: `nSTATPaperExamples.mlx`
- Fidelity status: `high_fidelity`
- Remaining justified differences: The notebook now executes the canonical paper-example workflows through the standalone Python implementations and real figshare-backed datasets; exact numerical traces and figure styling still vary modestly because the Python GLM/decoder stack and plotting defaults are not byte-identical to MATLAB.
"""


CODE = [
    """
    # nSTAT-python notebook example: nSTATPaperExamples
    from pathlib import Path
    import sys

    REPO_ROOT = Path.cwd().resolve().parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    SRC_PATH = (REPO_ROOT / "src").resolve()
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from nstat.notebook_data import notebook_example_data_dir
    from nstat.notebook_figures import FigureTracker
    from nstat.paper_examples_full import (
        run_experiment1,
        run_experiment2,
        run_experiment3,
        run_experiment3b,
        run_experiment4,
        run_experiment5,
        run_experiment5b,
        run_experiment6,
    )

    DATA_DIR = notebook_example_data_dir(allow_synthetic=True)
    OUTPUT_ROOT = REPO_ROOT / "output" / "notebook_images"
    __tracker = FigureTracker(topic="nSTATPaperExamples", output_root=OUTPUT_ROOT, expected_count=26)


    def _fig(label: str, *, figsize=(8.5, 4.5)):
        fig = __tracker.new_figure(label)
        fig.clear()
        fig.set_size_inches(*figsize)
        return fig


    plt.close("all")
    exp1_summary, exp1 = run_experiment1(DATA_DIR, return_payload=True)
    exp2_summary, exp2 = run_experiment2(DATA_DIR, return_payload=True)
    exp3_summary, exp3 = run_experiment3(return_payload=True)
    exp3b_summary, exp3b = run_experiment3b(DATA_DIR, return_payload=True)
    exp4_summary, exp4 = run_experiment4(DATA_DIR, return_payload=True)
    exp5_summary, exp5 = run_experiment5(return_payload=True)
    exp5b_summary, exp5b = run_experiment5b(return_payload=True)
    exp6_summary, exp6 = run_experiment6(REPO_ROOT, return_payload=True)
    print({"dataset_root": str(DATA_DIR), "paper_examples_loaded": 8})
    """,
    """
    # SECTION 1: Experiment 1
    print(exp1_summary)
    """,
    """
    # SECTION 2: Constant Magnesium Concentration - Constant rate poisson
    fig = _fig("experiment1 constant rate", figsize=(9.0, 4.0))
    ax = fig.subplots(1, 1)
    ax.plot(exp1["constant_time_s"], exp1["constant_rate_hz"], color="tab:blue", linewidth=1.4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rate (Hz)")
    ax.set_title("Constant Mg condition: homogeneous Poisson fit")
    """,
    """
    # SECTION 3: Varying Magnesium Concentration - Piecewise Constant rate poisson
    print({"decreasing_condition_spikes": exp1_summary["decreasing_condition_spikes"], "piecewise_model_aic": round(float(exp1_summary["piecewise_model_aic"]), 3)})
    """,
    """
    # SECTION 4: Data Visualization
    fig = _fig("experiment1 washout raster and rates", figsize=(10.0, 5.5))
    axs = fig.subplots(2, 1, sharex=True)
    spike_times = np.asarray(exp1["washout_spike_times_s"], dtype=float)
    axs[0].vlines(spike_times, 0.0, 1.0, color="k", linewidth=0.3)
    axs[0].set_ylim(0.0, 1.0)
    axs[0].set_ylabel("spikes")
    axs[0].set_title("Decreasing Mg condition raster")
    axs[1].plot(exp1["washout_time_s"], exp1["washout_observed_rate_hz"], color="0.3", linewidth=1.0, label="Observed")
    axs[1].plot(exp1["washout_time_s"], exp1["washout_piecewise_rate_hz"], color="tab:green", linewidth=1.3, label="Piecewise")
    axs[1].plot(exp1["washout_time_s"], exp1["washout_piecewise_history_rate_hz"], color="tab:red", linewidth=1.3, label="Piecewise+Hist")
    for edge in exp1["washout_segment_edges_s"][1:-1]:
        axs[1].axvline(edge, color="tab:red", linestyle="--", linewidth=0.9)
    axs[1].set_xlabel("time (s)")
    axs[1].set_ylabel("rate (Hz)")
    axs[1].legend(loc="upper left", frameon=False, fontsize=8)
    """,
    """
    # SECTION 5: Define Covariates for the analysis
    fig = _fig("experiment1 constant ks", figsize=(6.0, 5.0))
    ax = fig.subplots(1, 1)
    ax.plot(exp1["constant_ks_ideal"], exp1["constant_ks_empirical"], color="tab:blue", linewidth=1.4)
    ax.plot([0.0, 1.0], [0.0, 1.0], color="0.25", linestyle="--", linewidth=1.0)
    ax.fill_between(exp1["constant_ks_ideal"], np.clip(exp1["constant_ks_ideal"] - exp1["constant_ks_ci"], 0.0, 1.0), np.clip(exp1["constant_ks_ideal"] + exp1["constant_ks_ci"], 0.0, 1.0), color="0.85")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("theoretical CDF")
    ax.set_ylabel("empirical CDF")
    ax.set_title("Constant-condition KS plot")
    """,
    """
    # SECTION 6: Define how we want to analyze the data
    fig = _fig("experiment1 constant acf", figsize=(7.0, 4.0))
    ax = fig.subplots(1, 1)
    ax.vlines(exp1["constant_acf_lags_s"], 0.0, exp1["constant_acf_values"], color="tab:purple", linewidth=1.0)
    ax.axhline(exp1_summary["constant_acf_ci"], color="tab:red", linewidth=1.0)
    ax.axhline(-exp1_summary["constant_acf_ci"], color="tab:red", linewidth=1.0)
    ax.set_xlabel("lag")
    ax.set_ylabel("autocorrelation")
    ax.set_title("Sequential correlation under constant Mg")
    """,
    """
    # SECTION 7: Compare constant-rate and piecewise-rate fits
    fig = _fig("experiment1 model summary", figsize=(7.5, 4.0))
    ax = fig.subplots(1, 1)
    names = ["Const", "Piecewise", "Piecewise+Hist"]
    aics = [exp1_summary["const_model_aic"], exp1_summary["piecewise_model_aic"], exp1_summary["piecewise_history_model_aic"]]
    ax.bar(np.arange(3), aics, color=["0.6", "tab:green", "tab:red"])
    ax.set_xticks(np.arange(3), names)
    ax.set_ylabel("AIC")
    ax.set_title("Experiment 1 model comparison")
    """,
    """
    # SECTION 8: Experiment 2
    print(exp2_summary)
    """,
    """
    # SECTION 9: Load the explicit-stimulus dataset
    fig = _fig("experiment2 stimulus and spikes", figsize=(10.0, 5.5))
    axs = fig.subplots(2, 1, sharex=True)
    spike_times = np.asarray(exp2["time_s"], dtype=float)[np.asarray(exp2["spike_indicator"], dtype=float) > 0.5]
    axs[0].vlines(spike_times, 0.0, 1.0, color="k", linewidth=0.35)
    axs[0].set_ylim(0.0, 1.0)
    axs[0].set_ylabel("spikes")
    axs[1].plot(exp2["time_s"], exp2["stimulus"], color="tab:blue", linewidth=1.2)
    axs[1].set_ylabel("stimulus")
    axs[1].set_xlabel("time (s)")
    """,
    """
    # SECTION 10: Stimulus-lag search
    fig = _fig("experiment2 xcorr", figsize=(7.0, 4.0))
    ax = fig.subplots(1, 1)
    ax.plot(1000.0 * np.asarray(exp2["xcorr_lags_s"], dtype=float), exp2["xcorr_values"], color="tab:purple", linewidth=1.3)
    ax.set_xlabel("lag (ms)")
    ax.set_ylabel("cross-covariance")
    ax.set_title("Stimulus lag search")
    """,
    """
    # SECTION 11: Model comparison with stimulus effects
    fig = _fig("experiment2 aic bic", figsize=(8.5, 4.0))
    axs = fig.subplots(1, 2)
    model_names = ["Baseline", "Stim", "Stim+Hist"]
    axs[0].bar(np.arange(3), [exp2_summary["model1_aic"], exp2_summary["model2_aic"], exp2_summary["model3_aic"]], color=["0.65", "tab:blue", "tab:green"])
    axs[0].set_xticks(np.arange(3), model_names, rotation=15)
    axs[0].set_title("AIC")
    axs[1].bar(np.arange(3), [exp2_summary["model1_bic"], exp2_summary["model2_bic"], exp2_summary["model3_bic"]], color=["0.65", "tab:blue", "tab:green"])
    axs[1].set_xticks(np.arange(3), model_names, rotation=15)
    axs[1].set_title("BIC")
    """,
    """
    # SECTION 12: KS diagnostics
    fig = _fig("experiment2 ks compare", figsize=(6.5, 5.0))
    ax = fig.subplots(1, 1)
    ideal = np.asarray(exp2["ks_ideal"], dtype=float)
    ax.plot(ideal, ideal, color="0.25", linestyle="--", linewidth=1.0)
    ax.plot(ideal, exp2["ks_const_empirical"], color="tab:blue", linewidth=1.2, label="Baseline")
    ax.plot(ideal, exp2["ks_stim_empirical"], color="tab:orange", linewidth=1.2, label="Stim")
    ax.plot(ideal, exp2["ks_hist_empirical"], color="tab:green", linewidth=1.2, label="Stim+Hist")
    ax.fill_between(ideal, np.clip(ideal - exp2["ks_ci"], 0.0, 1.0), np.clip(ideal + exp2["ks_ci"], 0.0, 1.0), color="0.88")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_title("Experiment 2 KS diagnostics")
    """,
    """
    # SECTION 13: History-window scan
    fig = _fig("experiment2 history scan", figsize=(8.5, 7.0))
    axs = fig.subplots(3, 1, sharex=True)
    windows = np.asarray(exp2["history_windows"], dtype=float)
    axs[0].plot(windows, exp2["ks_stats"], marker="o", color="tab:purple", linewidth=1.2)
    axs[0].set_ylabel("KS")
    axs[1].plot(windows, exp2["delta_aic"], marker="o", color="tab:green", linewidth=1.2)
    axs[1].set_ylabel("Delta AIC")
    axs[2].plot(windows, exp2["delta_bic"], marker="o", color="tab:brown", linewidth=1.2)
    axs[2].set_ylabel("Delta BIC")
    axs[2].set_xlabel("history windows")
    """,
    """
    # SECTION 14: Coefficient summaries
    fig = _fig("experiment2 coefficients", figsize=(9.0, 4.5))
    ax = fig.subplots(1, 1)
    xpos = np.arange(len(exp2["coef_names"]))
    coef_values = np.asarray(exp2["coef_values"], dtype=float)
    lower = np.asarray(exp2["coef_lower"], dtype=float)
    upper = np.asarray(exp2["coef_upper"], dtype=float)
    ax.errorbar(xpos, coef_values, yerr=np.vstack([coef_values - lower, upper - coef_values]), fmt="o", color="tab:blue", capsize=3)
    ax.set_xticks(xpos, exp2["coef_names"], rotation=30)
    ax.set_ylabel("coefficient value")
    ax.set_title("Experiment 2 coefficient intervals")
    """,
    """
    # SECTION 15: Experiment 3
    print(exp3_summary)
    """,
    """
    # SECTION 16: Simulated PSTH setup
    fig = _fig("experiment3 true rate", figsize=(9.0, 4.0))
    ax = fig.subplots(1, 1)
    ax.plot(exp3["time_s"], exp3["true_rate_hz"], color="tab:blue", linewidth=1.3)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rate (Hz)")
    ax.set_title("Experiment 3 true conditional intensity")
    """,
    """
    # SECTION 17: PSTH estimate
    fig = _fig("experiment3 psth", figsize=(9.0, 5.0))
    axs = fig.subplots(2, 1, sharex=True)
    for row, spikes in enumerate(exp3["raster_spike_times"][:10], start=1):
        axs[0].vlines(spikes, row - 0.4, row + 0.4, color="k", linewidth=0.3)
    axs[0].set_ylabel("trial")
    axs[1].plot(exp3["psth_bin_centers_s"], exp3["psth_rate_hz"], color="tab:red", linewidth=1.4)
    axs[1].set_ylabel("PSTH (Hz)")
    axs[1].set_xlabel("time (s)")
    """,
    """
    # SECTION 18: Experiment 3b
    print(exp3b_summary)
    """,
    """
    # SECTION 19: SSGLM state estimates
    fig = _fig("experiment3b state estimates", figsize=(10.0, 5.0))
    axs = fig.subplots(2, 1, sharex=True)
    axs[0].imshow(exp3b["stimulus"], aspect="auto", cmap="viridis")
    axs[0].set_title("True stimulus")
    axs[1].imshow(exp3b["xk"], aspect="auto", cmap="viridis")
    axs[1].set_title("Decoded state")
    axs[1].set_xlabel("time bin")
    """,
    """
    # SECTION 20: SSGLM confidence intervals
    fig = _fig("experiment3b ci width", figsize=(8.5, 4.5))
    axs = fig.subplots(1, 2)
    axs[0].plot(np.mean(exp3b["ci_width"], axis=0), color="tab:orange", linewidth=1.3)
    axs[0].set_title("Mean CI width over time")
    axs[1].plot(np.mean(exp3b["qhat_all"], axis=0), marker="o", color="tab:blue", linewidth=1.2)
    axs[1].set_title("Mean Qhat across models")
    """,
    """
    # SECTION 21: SSGLM gamma summaries
    fig = _fig("experiment3b gamma", figsize=(8.5, 4.5))
    axs = fig.subplots(1, 2)
    axs[0].bar(np.arange(len(exp3b["gammahat"])), exp3b["gammahat"], color="tab:green")
    axs[0].set_title("gammahat")
    axs[1].plot(np.asarray(exp3b["gammahat_all"], dtype=float), marker="o", color="tab:red", linewidth=1.2)
    axs[1].set_title("gammahatAll")
    """,
    """
    # SECTION 22: Experiment 4
    print(exp4_summary)
    """,
    """
    # SECTION 23: Place-cell model comparison for Animal 1
    fig = _fig("experiment4 animal1 delta aic", figsize=(7.5, 4.0))
    ax = fig.subplots(1, 1)
    ax.bar(np.arange(len(exp4["animal1"]["selected_indices"])), exp4["animal1"]["delta_aic"], color="tab:blue")
    ax.set_xticks(np.arange(len(exp4["animal1"]["selected_indices"])), [str(int(v) + 1) for v in exp4["animal1"]["selected_indices"]])
    ax.set_ylabel("Gaussian - Zernike AIC")
    ax.set_title("Animal 1 place-cell comparison")
    """,
    """
    # SECTION 24: Place-cell model comparison for Animal 2
    fig = _fig("experiment4 animal2 delta bic", figsize=(7.5, 4.0))
    ax = fig.subplots(1, 1)
    ax.bar(np.arange(len(exp4["animal2"]["selected_indices"])), exp4["animal2"]["delta_bic"], color="tab:green")
    ax.set_xticks(np.arange(len(exp4["animal2"]["selected_indices"])), [str(int(v) + 1) for v in exp4["animal2"]["selected_indices"]])
    ax.set_ylabel("Gaussian - Zernike BIC")
    ax.set_title("Animal 2 place-cell comparison")
    """,
    """
    # SECTION 25: Place-field mesh for representative neuron
    fig = _fig("experiment4 gaussian mesh", figsize=(9.0, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(exp4["mesh"]["grid_x"], exp4["mesh"]["grid_y"], exp4["mesh"]["gaussian_field"], cmap="Blues", linewidth=0.0, antialiased=True)
    ax.set_title("Gaussian place-field estimate")
    """,
    """
    # SECTION 26: Zernike place-field mesh
    fig = _fig("experiment4 zernike mesh", figsize=(9.0, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(exp4["mesh"]["grid_x"], exp4["mesh"]["grid_y"], exp4["mesh"]["zernike_field"], cmap="Greens", linewidth=0.0, antialiased=True)
    ax.set_title("Zernike place-field estimate")
    """,
    """
    # SECTION 27: Experiment 5
    print(exp5_summary)
    """,
    """
    # SECTION 28: 1-D decoding workflow
    fig = _fig("experiment5 stimulus decode", figsize=(9.0, 4.5))
    ax = fig.subplots(1, 1)
    ax.plot(exp5["time_s"], exp5["stimulus"], color="0.3", linewidth=1.0, label="True")
    ax.plot(exp5["time_s"], exp5["decoded"], color="tab:blue", linewidth=1.4, label="Decoded")
    ax.fill_between(exp5["time_s"], exp5["ci_low"], exp5["ci_high"], color="0.85")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_xlabel("time (s)")
    ax.set_title("Experiment 5 adaptive decoding")
    """,
    """
    # SECTION 29: Experiment 5b
    print(exp5b_summary)
    """,
    """
    # SECTION 30: Goal-directed 2-D decode
    fig = _fig("experiment5b goal decode", figsize=(9.5, 4.5))
    axs = fig.subplots(2, 1, sharex=True)
    axs[0].plot(exp5b["time_s"], exp5b["x_true"], color="0.3", linewidth=1.0, label="True x")
    axs[0].plot(exp5b["time_s"], exp5b["dx_goal"], color="tab:blue", linewidth=1.2, label="Decoded x")
    axs[0].legend(loc="upper right", frameon=False, fontsize=8)
    axs[1].plot(exp5b["time_s"], exp5b["y_true"], color="0.3", linewidth=1.0, label="True y")
    axs[1].plot(exp5b["time_s"], exp5b["dy_goal"], color="tab:orange", linewidth=1.2, label="Decoded y")
    axs[1].legend(loc="upper right", frameon=False, fontsize=8)
    axs[1].set_xlabel("time (s)")
    """,
    """
    # SECTION 31: Free-model 2-D decode
    fig = _fig("experiment5b free decode", figsize=(9.5, 4.5))
    axs = fig.subplots(2, 1, sharex=True)
    axs[0].plot(exp5b["time_s"], exp5b["x_true"], color="0.3", linewidth=1.0, label="True x")
    axs[0].plot(exp5b["time_s"], exp5b["dx_free"], color="tab:green", linewidth=1.2, label="Decoded x")
    axs[0].legend(loc="upper right", frameon=False, fontsize=8)
    axs[1].plot(exp5b["time_s"], exp5b["y_true"], color="0.3", linewidth=1.0, label="True y")
    axs[1].plot(exp5b["time_s"], exp5b["dy_free"], color="tab:red", linewidth=1.2, label="Decoded y")
    axs[1].legend(loc="upper right", frameon=False, fontsize=8)
    axs[1].set_xlabel("time (s)")
    """,
    """
    # SECTION 32: Experiment 6
    print(exp6_summary)
    """,
    """
    # SECTION 33: Hybrid-filter simulation
    fig = _fig("experiment6 state probabilities", figsize=(9.5, 4.5))
    axs = fig.subplots(2, 1, sharex=True)
    axs[0].plot(exp6["time_s"], exp6["state_true"], color="0.2", linewidth=1.0)
    axs[0].set_ylabel("true state")
    axs[1].plot(exp6["time_s"], exp6["state_prob_1"], color="tab:blue", linewidth=1.2, label="P(state=1)")
    axs[1].plot(exp6["time_s"], exp6["state_prob_2"], color="tab:orange", linewidth=1.2, label="P(state=2)")
    axs[1].legend(loc="upper right", frameon=False, fontsize=8)
    axs[1].set_xlabel("time (s)")
    """,
    """
    # SECTION 34: Hybrid-filter decoded positions
    fig = _fig("experiment6 decoded positions", figsize=(9.5, 4.5))
    axs = fig.subplots(2, 1, sharex=True)
    axs[0].plot(exp6["time_s"], exp6["x_pos"], color="0.3", linewidth=1.0, label="True x")
    axs[0].plot(exp6["time_s"], exp6["decoded_x"], color="tab:blue", linewidth=1.2, label="Decoded x")
    axs[0].legend(loc="upper right", frameon=False, fontsize=8)
    axs[1].plot(exp6["time_s"], exp6["y_pos"], color="0.3", linewidth=1.0, label="True y")
    axs[1].plot(exp6["time_s"], exp6["decoded_y"], color="tab:orange", linewidth=1.2, label="Decoded y")
    axs[1].legend(loc="upper right", frameon=False, fontsize=8)
    axs[1].set_xlabel("time (s)")
    """,
    """
    # SECTION 35: Canonical paper-example gallery summary
    fig = _fig("paper gallery summary", figsize=(8.5, 4.5))
    ax = fig.subplots(1, 1)
    rmses = [exp5_summary["decode_rmse"], exp5b_summary["decode_rmse_x"], exp5b_summary["decode_rmse_y"], exp6_summary["decode_rmse_x"], exp6_summary["decode_rmse_y"]]
    labels = ["Exp5", "Exp5b x", "Exp5b y", "Exp6 x", "Exp6 y"]
    ax.bar(np.arange(len(labels)), rmses, color=["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:orange"])
    ax.set_xticks(np.arange(len(labels)), labels, rotation=20)
    ax.set_ylabel("RMSE")
    ax.set_title("Decoding summary across paper examples")
    """,
    """
    # SECTION 36: Dataset-backed parity summary
    fig = _fig("paper dataset summary", figsize=(8.5, 4.5))
    ax = fig.subplots(1, 1)
    counts = [
        exp1_summary["decreasing_condition_spikes"],
        exp2_summary["n_samples"],
        exp3_summary["num_trials"],
        exp4_summary["num_cells_fit"],
        exp6_summary["num_cells"],
    ]
    labels = ["Exp1 spikes", "Exp2 samples", "Exp3 trials", "Exp4 cells", "Exp6 cells"]
    ax.bar(np.arange(len(labels)), counts, color="0.65")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=20)
    ax.set_title("Paper-example dataset scale")
    """,
    """
    # SECTION 37: Final summary
    print(
        {
            "experiment1_piecewise_history_aic": round(float(exp1_summary["piecewise_history_model_aic"]), 3),
            "experiment2_peak_lag_ms": round(float(exp2_summary["peak_lag_seconds"]) * 1000.0, 1),
            "experiment4_mean_delta_aic": round(float(exp4_summary["mean_delta_aic_gaussian_minus_zernike"]), 3),
            "experiment6_state_accuracy": round(float(exp6_summary["state_accuracy"]), 3),
        }
    )
    __tracker.finalize()
    """,
]


def main() -> int:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    _write_notebook(
        NOTEBOOK_DIR / "nSTATPaperExamples.ipynb",
        topic="nSTATPaperExamples",
        expected_figures=26,
        markdown_note=NOTE,
        code_cells=CODE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
