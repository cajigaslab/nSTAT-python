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


def _write_notebook(
    path: Path,
    *,
    topic: str,
    expected_figures: int,
    markdown_note: str,
    code_cells: list[str],
) -> None:
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


EXPLICIT_STIMULUS_NOTE = """\
<!-- parity-note -->
## MATLAB Parity Note
- Source MATLAB helpfile: `ExplicitStimulusWhiskerData.mlx`
- Fidelity status: `high_fidelity`
- Remaining justified differences: The notebook now reproduces the dataset-backed lag search, stimulus-effect, and history-effect workflow with real figures; exact KS traces and coefficient values still vary modestly from MATLAB because the Python GLM backend and plotting defaults are different.
"""


EXPLICIT_STIMULUS_CODE = [
    """
    # nSTAT-python notebook example: ExplicitStimulusWhiskerData
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

    from nstat.data_manager import ensure_example_data
    from nstat.notebook_figures import FigureTracker
    from nstat.paper_examples_full import run_experiment2

    np.random.seed(0)
    DATA_DIR = ensure_example_data(download=True)
    OUTPUT_ROOT = REPO_ROOT / "output" / "notebook_images"
    __tracker = FigureTracker(topic='ExplicitStimulusWhiskerData', output_root=OUTPUT_ROOT, expected_count=9)


    def _prepare_figure(matlab_line: str, *, figsize=(8.0, 4.5)):
        fig = __tracker.new_figure(matlab_line)
        fig.clear()
        fig.set_size_inches(*figsize)
        return fig


    def _plot_spike_indicator(ax, time_s, spike_indicator):
        spike_times = np.asarray(time_s, dtype=float)[np.asarray(spike_indicator, dtype=float) > 0.5]
        if spike_times.size:
            ax.vlines(spike_times, 0.0, 1.0, color="k", linewidth=0.35)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("spikes")


    def _plot_ks(ax, ideal, empirical, ci, *, label, color):
        ideal_arr = np.asarray(ideal, dtype=float)
        empirical_arr = np.asarray(empirical, dtype=float)
        ci_arr = np.asarray(ci, dtype=float)
        ax.plot(ideal_arr, ideal_arr, color="0.2", linewidth=1.0, linestyle="--", label="45° line")
        ax.plot(ideal_arr, empirical_arr, color=color, linewidth=1.5, label=label)
        ax.fill_between(
            ideal_arr,
            np.clip(ideal_arr - ci_arr, 0.0, 1.0),
            np.clip(ideal_arr + ci_arr, 0.0, 1.0),
            color="0.8",
            alpha=0.35,
            label="95% CI",
        )
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Empirical quantiles")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

    """,
    """
    # SECTION 0: EXPLICIT STIMULUS EXAMPLE - WHISKER STIMULATION/THALAMIC NEURON
    # This notebook follows the MATLAB helpfile workflow for explicit whisker-stimulation analysis.
    plt.close("all")
    summary, payload = run_experiment2(DATA_DIR, return_payload=True)
    model_names = ["Baseline", "Baseline+Stimulus", "Baseline+Stimulus+History"]
    best_history_idx = int(np.argmin(np.asarray(payload["delta_bic"], dtype=float)))
    best_history_window = int(np.asarray(payload["history_windows"], dtype=float)[best_history_idx])
    print(
        {
            "n_samples": int(summary["n_samples"]),
            "peak_lag_ms": round(float(summary["peak_lag_seconds"]) * 1000.0, 1),
            "best_history_window_bins": best_history_window,
        }
    )
    """,
    """
    # SECTION 1: Load the data
    fig = _prepare_figure("trial.plot", figsize=(10.0, 6.0))
    axs = fig.subplots(2, 1, sharex=True)
    _plot_spike_indicator(axs[0], payload["time_s"], payload["spike_indicator"])
    axs[0].set_title("Observed spike train")
    axs[1].plot(payload["time_s"], payload["stimulus"], color="tab:blue", linewidth=1.25)
    axs[1].set_title("Whisker stimulus")
    axs[1].set_ylabel("stimulus")
    axs[1].set_xlabel("time (s)")

    fig = _prepare_figure("stim.getSigInTimeWindow(0,21).plot", figsize=(10.0, 5.5))
    axs = fig.subplots(2, 1, sharex=True)
    axs[0].plot(payload["time_s"], payload["stimulus"], color="tab:blue", linewidth=1.4)
    axs[0].set_title("Stimulus over the analysis window")
    axs[0].set_ylabel("stimulus")
    axs[1].plot(payload["time_s"], payload["velocity"], color="tab:orange", linewidth=1.2)
    axs[1].set_title("Stimulus derivative")
    axs[1].set_ylabel("d(stimulus)/dt")
    axs[1].set_xlabel("time (s)")
    """,
    """
    # SECTION 2: Fit a constant baseline
    fig = _prepare_figure("results.plotResults", figsize=(6.0, 5.5))
    ax = fig.subplots(1, 1)
    _plot_ks(ax, payload["ks_ideal"], payload["ks_const_empirical"], payload["ks_ci"], label="Baseline model", color="tab:blue")
    ax.set_title("Baseline model KS plot")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    """,
    """
    # SECTION 3: Find Stimulus Lag
    fig = _prepare_figure("results.Residual.xcov(stim).windowedSignal([0,1]).plot", figsize=(8.5, 4.5))
    ax = fig.subplots(1, 1)
    lags_ms = 1000.0 * np.asarray(payload["xcorr_lags_s"], dtype=float)
    xcorr_vals = np.asarray(payload["xcorr_values"], dtype=float)
    peak_idx = int(np.argmax(xcorr_vals))
    ax.plot(lags_ms, xcorr_vals, color="tab:purple", linewidth=1.4)
    ax.axvline(lags_ms[peak_idx], color="tab:red", linestyle="--", linewidth=1.0)
    ax.scatter([lags_ms[peak_idx]], [xcorr_vals[peak_idx]], color="tab:red", zorder=3)
    ax.set_title("Cross-covariance used to identify the stimulus lag")
    ax.set_xlabel("lag (ms)")
    ax.set_ylabel("cross-covariance")
    """,
    """
    # SECTION 4: Compare constant rate model with model including stimulus effect
    fig = _prepare_figure("results.plotResults", figsize=(8.5, 4.5))
    axs = fig.subplots(1, 2)
    aic_vals = np.asarray([summary["model1_aic"], summary["model2_aic"], summary["model3_aic"]], dtype=float)
    bic_vals = np.asarray([summary["model1_bic"], summary["model2_bic"], summary["model3_bic"]], dtype=float)
    xloc = np.arange(len(model_names))
    axs[0].bar(xloc, aic_vals, color=["0.7", "tab:blue", "tab:green"])
    axs[0].set_xticks(xloc, model_names, rotation=15)
    axs[0].set_title("AIC")
    axs[1].bar(xloc, bic_vals, color=["0.7", "tab:blue", "tab:green"])
    axs[1].set_xticks(xloc, model_names, rotation=15)
    axs[1].set_title("BIC")

    fig = _prepare_figure("results.plotResults", figsize=(7.0, 5.5))
    ax = fig.subplots(1, 1)
    _plot_ks(ax, payload["ks_ideal"], payload["ks_const_empirical"], payload["ks_ci"], label="Baseline", color="tab:blue")
    ax.plot(np.asarray(payload["ks_ideal"], dtype=float), np.asarray(payload["ks_stim_empirical"], dtype=float), color="tab:orange", linewidth=1.5, label="Baseline+Stimulus")
    ax.set_title("Baseline vs stimulus-augmented model")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    """,
    """
    # SECTION 5: History Effect
    fig = _prepare_figure("Summary.plotSummary", figsize=(9.0, 7.0))
    axs = fig.subplots(3, 1, sharex=True)
    history_windows = np.asarray(payload["history_windows"], dtype=float)
    axs[0].plot(history_windows, payload["ks_stats"], marker="o", color="tab:purple", linewidth=1.2)
    axs[0].scatter([history_windows[best_history_idx]], [payload["ks_stats"][best_history_idx]], color="tab:red", zorder=3)
    axs[0].set_ylabel("KS statistic")
    axs[0].set_title("History-window scan")
    axs[1].plot(history_windows, payload["delta_aic"], marker="o", color="tab:green", linewidth=1.2)
    axs[1].scatter([history_windows[best_history_idx]], [payload["delta_aic"][best_history_idx]], color="tab:red", zorder=3)
    axs[1].set_ylabel("ΔAIC")
    axs[2].plot(history_windows, payload["delta_bic"], marker="o", color="tab:brown", linewidth=1.2)
    axs[2].scatter([history_windows[best_history_idx]], [payload["delta_bic"][best_history_idx]], color="tab:red", zorder=3)
    axs[2].set_ylabel("ΔBIC")
    axs[2].set_xlabel("history window count")

    fig = _prepare_figure("plot(x,dBIC,'.')", figsize=(8.0, 4.5))
    ax = fig.subplots(1, 1)
    ax.plot(history_windows, payload["delta_bic"], marker="o", color="tab:brown", linewidth=1.4)
    ax.axvline(history_windows[best_history_idx], color="tab:red", linestyle="--", linewidth=1.0)
    ax.set_title("BIC improvement across history-window choices")
    ax.set_xlabel("history window count")
    ax.set_ylabel("ΔBIC relative to first history model")
    """,
    """
    # SECTION 6: Compare Baseline, Baseline+Stimulus Model, Baseline+History+Stimulus
    fig = _prepare_figure("plot(historyCoeffs)", figsize=(9.5, 5.0))
    axs = fig.subplots(1, 2, width_ratios=[1.6, 1.0])
    coeff_names = list(payload["coef_names"])
    coeff_vals = np.asarray(payload["coef_values"], dtype=float)
    coeff_low = np.asarray(payload["coef_lower"], dtype=float)
    coeff_high = np.asarray(payload["coef_upper"], dtype=float)
    ypos = np.arange(len(coeff_names))
    axs[0].hlines(ypos, coeff_low, coeff_high, color="0.6", linewidth=2.0)
    axs[0].plot(coeff_vals, ypos, "o", color="tab:green")
    axs[0].axvline(0.0, color="0.2", linewidth=1.0)
    axs[0].set_yticks(ypos, coeff_names)
    axs[0].set_title("Full-model coefficient intervals")
    axs[0].set_xlabel("coefficient value")
    axs[1].axis("off")
    axs[1].text(
        0.0,
        0.98,
        "\\n".join(
            [
                f"Peak lag: {1000.0 * float(summary['peak_lag_seconds']):.1f} ms",
                f"Best history window: {best_history_window} bins",
                f"Baseline AIC: {summary['model1_aic']:.1f}",
                f"Stimulus AIC: {summary['model2_aic']:.1f}",
                f"History AIC: {summary['model3_aic']:.1f}",
            ]
        ),
        va="top",
        family="monospace",
        fontsize=9,
    )

    fig = _prepare_figure("results.plotResults", figsize=(7.0, 5.5))
    ax = fig.subplots(1, 1)
    _plot_ks(ax, payload["ks_ideal"], payload["ks_const_empirical"], payload["ks_ci"], label="Baseline", color="tab:blue")
    ax.plot(np.asarray(payload["ks_ideal"], dtype=float), np.asarray(payload["ks_stim_empirical"], dtype=float), color="tab:orange", linewidth=1.5, label="Baseline+Stimulus")
    ax.plot(np.asarray(payload["ks_ideal"], dtype=float), np.asarray(payload["ks_hist_empirical"], dtype=float), color="tab:green", linewidth=1.5, label="Baseline+Stimulus+History")
    ax.set_title("Final KS comparison across the three models")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    __tracker.finalize()
    """,
]


VALIDATION_NOTE = """\
<!-- parity-note -->
## MATLAB Parity Note
- Source MATLAB helpfile: `ValidationDataSet.mlx`
- Fidelity status: `high_fidelity`
- Remaining justified differences: The notebook now reproduces the constant-rate and piecewise-rate validation workflows with real `Trial`/`Analysis` objects and figure outputs; the Python port uses shorter deterministic simulations than MATLAB so the notebook remains stable in CI.
"""


VALIDATION_CODE = [
    """
    # nSTAT-python notebook example: ValidationDataSet
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

    from nstat import Analysis, ConfigColl, CovColl, Covariate, FitResSummary, Trial, TrialConfig, nspikeTrain, nstColl
    from nstat.notebook_figures import FigureTracker

    np.random.seed(0)
    OUTPUT_ROOT = REPO_ROOT / "output" / "notebook_images"
    __tracker = FigureTracker(topic='ValidationDataSet', output_root=OUTPUT_ROOT, expected_count=10)


    def _prepare_figure(matlab_line: str, *, figsize=(8.0, 4.5)):
        fig = __tracker.new_figure(matlab_line)
        fig.clear()
        fig.set_size_inches(*figsize)
        return fig


    def _lambda_columns(fit_result):
        time = np.asarray(fit_result.lambda_signal.time, dtype=float)
        data = np.asarray(fit_result.lambda_signal.data, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        return time, data


    def _simulate_constant_case(seed=0, *, p=0.01, n_samples=20001, delta=0.001):
        rng = np.random.default_rng(seed)
        total_time = n_samples * delta
        time = np.linspace(0.0, total_time, n_samples)
        lambda_hz = n_samples * p / total_time
        mu = float(np.log(lambda_hz * delta / (1.0 - lambda_hz * delta)))
        trains = []
        for idx in range(2):
            spike_mask = rng.random(n_samples) < p
            spike_times = time[spike_mask]
            train = nspikeTrain(spike_times, str(idx + 1), delta, 0.0, total_time, makePlots=-1)
            trains.append(train)
        spike_coll = nstColl(trains)
        cov = Covariate(time, np.ones((time.shape[0], 1), dtype=float), "Baseline", "time", "s", "", ["mu"])
        trial = Trial(spike_coll, CovColl([cov]))
        cfg = ConfigColl([TrialConfig([["Baseline", "mu"]], 1.0 / delta, [], [], name="Baseline")])
        return {
            "time_s": time,
            "delta": delta,
            "lambda_hz": lambda_hz,
            "mu": mu,
            "trial": trial,
            "cfg": cfg,
            "trains": trains,
        }


    def _simulate_piecewise_case(seed=1, *, p1=0.001, p2=0.01, n1=20000, n2=20000, delta=0.001):
        rng = np.random.default_rng(seed)
        t1 = np.linspace(0.0, n1 * delta, n1 + 1)
        t2 = np.linspace(n1 * delta, (n1 + n2) * delta, n2 + 1)[1:]
        total_time = float(t2[-1])
        lambda1_hz = n1 * p1 / (n1 * delta)
        lambda2_hz = n2 * p2 / (n2 * delta)
        lambda_const_hz = (n1 * p1 + n2 * p2) / total_time
        trains = []
        for idx in range(2):
            spikes1 = t1[:-1][rng.random(n1) < p1]
            spikes2 = t2[rng.random(n2) < p2]
            spike_times = np.concatenate([spikes1, spikes2])
            train = nspikeTrain(spike_times, str(idx + 1), delta, 0.0, total_time, makePlots=-1)
            trains.append(train)
        time = np.concatenate([t1[:-1], t2])
        cov_data = np.column_stack(
            [
                np.ones(time.shape[0], dtype=float),
                (time <= float(t1[-1])).astype(float),
                (time > float(t1[-1])).astype(float),
            ]
        )
        cov = Covariate(time, cov_data, "Baseline", "time", "s", "", ["muConst", "mu1", "mu2"])
        trial = Trial(nstColl(trains), CovColl([cov]))
        cfg = ConfigColl(
            [
                TrialConfig([["Baseline", "muConst"]], 1.0 / delta, [], [], name="Baseline"),
                TrialConfig([["Baseline", "mu1", "mu2"]], 1.0 / delta, [], [], name="Variable"),
            ]
        )
        return {
            "time_s": time,
            "delta": delta,
            "edge_time_s": float(t1[-1]),
            "lambda1_hz": lambda1_hz,
            "lambda2_hz": lambda2_hz,
            "lambda_const_hz": lambda_const_hz,
            "trial": trial,
            "cfg": cfg,
            "trains": trains,
        }


    def _plot_isi_hist(ax, train, lambda_hz, *, title):
        isi = np.asarray(train.getISIs(), dtype=float)
        if isi.size:
            ax.hist(isi, bins=25, density=True, color="0.8", edgecolor="0.3")
            x = np.linspace(0.0, float(np.max(isi)), 200)
            ax.plot(x, lambda_hz * np.exp(-lambda_hz * x), color="tab:red", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("ISI (s)")
        ax.set_ylabel("density")

    """,
    """
    # SECTION 0: Software Validation Data Set
    # This notebook follows the MATLAB validation helpfile with deterministic simulations for CI-stable execution.
    plt.close("all")
    constant_case = _simulate_constant_case()
    piecewise_case = _simulate_piecewise_case()
    print(
        {
            "constant_lambda_hz": round(float(constant_case["lambda_hz"]), 4),
            "piecewise_lambda1_hz": round(float(piecewise_case["lambda1_hz"]), 4),
            "piecewise_lambda2_hz": round(float(piecewise_case["lambda2_hz"]), 4),
        }
    )
    """,
    """
    # SECTION 1: Case #1: Constant Rate Poisson Process
    # First we verify that the analysis recovers a constant Poisson rate from simulated spike trains.
    pass
    """,
    """
    # SECTION 2: Generate constant-rate neural firing activity
    constant_time = np.asarray(constant_case["time_s"], dtype=float)
    constant_trains = list(constant_case["trains"])
    pass
    """,
    """
    # SECTION 3: Sanity check the ISI distribution
    fig = _prepare_figure("nst{1}.plotISIHistogram", figsize=(10.0, 4.0))
    axs = fig.subplots(1, 2)
    _plot_isi_hist(axs[0], constant_trains[0], constant_case["lambda_hz"], title="Neuron 1 ISI histogram")
    _plot_isi_hist(axs[1], constant_trains[1], constant_case["lambda_hz"], title="Neuron 2 ISI histogram")
    """,
    """
    # SECTION 4: Setup the constant-rate analysis
    constant_results = Analysis.RunAnalysisForAllNeurons(constant_case["trial"], constant_case["cfg"], 0)
    constant_intercepts = np.asarray([fit.getCoeffs(1)[0] for fit in constant_results], dtype=float)

    fig = _prepare_figure("plot(mu,'ro', 'MarkerSize',10)", figsize=(7.5, 4.5))
    ax = fig.subplots(1, 1)
    xloc = np.arange(1, constant_intercepts.size + 1)
    ax.bar(xloc, constant_intercepts, color="tab:blue", alpha=0.85, label="Estimated μ")
    ax.axhline(constant_case["mu"], color="tab:red", linestyle="--", linewidth=1.4, label="True μ")
    ax.set_xticks(xloc, [f"Neuron {idx}" for idx in xloc])
    ax.set_ylabel("μ coefficient")
    ax.set_title("Estimated constant-rate coefficient")
    ax.legend(loc="best", frameon=False)
    """,
    """
    # SECTION 5: Run the constant-rate analysis
    fig = _prepare_figure("results{1}.lambda.plot", figsize=(10.0, 4.5))
    axs = fig.subplots(1, 2, sharey=True)
    for idx, ax in enumerate(axs):
        fit = constant_results[idx]
        time_s, lambda_cols = _lambda_columns(fit)
        ax.plot(time_s, lambda_cols[:, 0], color="tab:blue", linewidth=1.25, label="Estimated λ(t)")
        ax.axhline(constant_case["lambda_hz"], color="tab:red", linestyle="--", linewidth=1.25, label="True λ")
        ax.set_title(f"Neuron {idx + 1}")
        ax.set_xlabel("time (s)")
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("rate (Hz)")
    axs[1].legend(loc="best", frameon=False, fontsize=8)
    """,
    """
    # SECTION 6: Case #2: Piece-wise Constant Rate Poisson Process
    # Next we compare a single-rate model against a two-epoch rate model.
    piecewise_time = np.asarray(piecewise_case["time_s"], dtype=float)
    piecewise_trains = list(piecewise_case["trains"])
    pass
    """,
    """
    # SECTION 7: Generate the piecewise-rate spike trains
    fig = _prepare_figure("plot(spikeTimes1, spikeTimes2)", figsize=(10.0, 4.5))
    axs = fig.subplots(2, 1, sharex=True)
    for row, train in enumerate(piecewise_trains, start=1):
        spikes = np.asarray(train.getSpikeTimes(), dtype=float)
        if spikes.size:
            axs[row - 1].vlines(spikes, row - 0.35, row + 0.35, color="k", linewidth=0.4)
        axs[row - 1].axvline(piecewise_case["edge_time_s"], color="tab:red", linestyle="--", linewidth=1.0)
        axs[row - 1].set_ylim(row - 0.5, row + 0.5)
        axs[row - 1].set_ylabel(f"N{row}")
    axs[-1].set_xlabel("time (s)")

    fig = _prepare_figure("plot(truePiecewiseRate)", figsize=(8.5, 4.0))
    ax = fig.subplots(1, 1)
    ax.plot(piecewise_time, np.where(piecewise_time <= piecewise_case["edge_time_s"], piecewise_case["lambda1_hz"], piecewise_case["lambda2_hz"]), color="tab:green", linewidth=1.6, label="True variable rate")
    ax.plot(piecewise_time, np.full_like(piecewise_time, piecewise_case["lambda_const_hz"]), color="tab:blue", linewidth=1.2, linestyle="--", label="True constant-rate surrogate")
    ax.axvline(piecewise_case["edge_time_s"], color="tab:red", linestyle="--", linewidth=1.0)
    ax.set_title("Ground-truth rates for the two-epoch simulation")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rate (Hz)")
    ax.legend(loc="best", frameon=False, fontsize=8)
    """,
    """
    # SECTION 8: Setup the piecewise-rate analysis
    piecewise_results = Analysis.RunAnalysisForAllNeurons(piecewise_case["trial"], piecewise_case["cfg"], 0)
    pass
    """,
    """
    # SECTION 9: Run the piecewise-rate analysis
    fig = _prepare_figure("results{1}.lambda.plot", figsize=(10.0, 4.5))
    axs = fig.subplots(1, 2, sharey=True)
    for idx, ax in enumerate(axs):
        fit = piecewise_results[idx]
        time_s, lambda_cols = _lambda_columns(fit)
        ax.plot(time_s, lambda_cols[:, 0], color="tab:blue", linewidth=1.2, label="Baseline model")
        ax.plot(time_s, lambda_cols[:, 1], color="tab:green", linewidth=1.2, label="Variable model")
        ax.plot(
            time_s,
            np.where(
                time_s <= piecewise_case["edge_time_s"],
                piecewise_case["lambda1_hz"],
                piecewise_case["lambda2_hz"],
            ),
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label="True rate",
        )
        ax.axvline(piecewise_case["edge_time_s"], color="0.3", linestyle=":", linewidth=1.0)
        ax.set_title(f"Neuron {idx + 1}")
        ax.set_xlabel("time (s)")
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("rate (Hz)")
    axs[1].legend(loc="best", frameon=False, fontsize=8)
    """,
    """
    # SECTION 10: Compare the results across the two neurons
    summary = FitResSummary(piecewise_results)
    fig = _prepare_figure("Summary.plotSummary", figsize=(8.5, 4.5))
    axs = fig.subplots(1, 2)
    xloc = np.arange(len(summary.fitNames))
    axs[0].bar(xloc, summary.AIC, color=["tab:blue", "tab:green"])
    axs[0].set_xticks(xloc, summary.fitNames)
    axs[0].set_title("Mean AIC across neurons")
    axs[1].bar(xloc, summary.BIC, color=["tab:blue", "tab:green"])
    axs[1].set_xticks(xloc, summary.fitNames)
    axs[1].set_title("Mean BIC across neurons")

    fig = _prepare_figure("Summary.getDifflogLL", figsize=(7.5, 4.5))
    ax = fig.subplots(1, 1)
    neuron_ids = np.arange(1, len(piecewise_results) + 1)
    base_logll = np.asarray([fit.logLL[0] for fit in piecewise_results], dtype=float)
    var_logll = np.asarray([fit.logLL[1] for fit in piecewise_results], dtype=float)
    ax.bar(neuron_ids - 0.15, base_logll, width=0.3, color="tab:blue", label="Baseline")
    ax.bar(neuron_ids + 0.15, var_logll, width=0.3, color="tab:green", label="Variable")
    ax.set_xticks(neuron_ids, [f"Neuron {idx}" for idx in neuron_ids])
    ax.set_ylabel("log-likelihood")
    ax.set_title("Per-neuron log-likelihood comparison")
    ax.legend(loc="best", frameon=False, fontsize=8)
    __tracker.finalize()
    """,
]


def main() -> int:
    _write_notebook(
        NOTEBOOK_DIR / "ExplicitStimulusWhiskerData.ipynb",
        topic="ExplicitStimulusWhiskerData",
        expected_figures=9,
        markdown_note=EXPLICIT_STIMULUS_NOTE,
        code_cells=EXPLICIT_STIMULUS_CODE,
    )
    _write_notebook(
        NOTEBOOK_DIR / "ValidationDataSet.ipynb",
        topic="ValidationDataSet",
        expected_figures=10,
        markdown_note=VALIDATION_NOTE,
        code_cells=VALIDATION_CODE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
