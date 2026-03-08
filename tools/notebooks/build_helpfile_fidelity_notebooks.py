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

    from nstat.notebook_data import notebook_example_data_dir
    from nstat.notebook_figures import FigureTracker
    from nstat.paper_examples_full import run_experiment2

    np.random.seed(0)
    DATA_DIR = notebook_example_data_dir(allow_synthetic=True)
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
- Remaining justified differences: The notebook now reproduces the constant-rate and piecewise-rate validation workflows with real `Trial`/`Analysis` objects and figure outputs; local execution uses the MATLAB-scale simulation sizes, while CI switches to a documented shorter deterministic fast path for stability.
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

    import os

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


    CI_FAST_PATH = os.environ.get("CI", "").strip().lower() in {"1", "true", "yes"}


    def _simulate_constant_case(seed=0, *, p=0.01, n_samples=None, delta=0.001):
        if n_samples is None:
            n_samples = 20001 if CI_FAST_PATH else 100001
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


    def _simulate_piecewise_case(seed=1, *, p1=0.001, p2=0.01, n1=None, n2=None, delta=0.001):
        if n1 is None:
            n1 = 20000 if CI_FAST_PATH else 100000
        if n2 is None:
            n2 = 20000 if CI_FAST_PATH else 100000
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
    # This notebook follows the MATLAB validation helpfile; CI uses a documented short fast path while local runs use MATLAB-scale sample counts.
    plt.close("all")
    constant_case = _simulate_constant_case()
    piecewise_case = _simulate_piecewise_case()
    print(
        {
            "ci_fast_path": CI_FAST_PATH,
            "constant_lambda_hz": round(float(constant_case["lambda_hz"]), 4),
            "piecewise_lambda1_hz": round(float(piecewise_case["lambda1_hz"]), 4),
            "piecewise_lambda2_hz": round(float(piecewise_case["lambda2_hz"]), 4),
        }
    )
    """,
    """
    # SECTION 1: Case #1: Constant Rate Poisson Process
    # First we verify that the analysis recovers a constant Poisson rate from simulated spike trains.
    """,
    """
    # SECTION 2: Generate constant-rate neural firing activity
    constant_time = np.asarray(constant_case["time_s"], dtype=float)
    constant_trains = list(constant_case["trains"])
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


HIPPOCAMPAL_NOTE = """\
<!-- parity-note -->
## MATLAB Parity Note
- Source MATLAB helpfile: `HippocampalPlaceCellExample.mlx`
- Fidelity status: `high_fidelity`
- Remaining justified differences: The notebook now reproduces the dataset-backed place-cell model-comparison and field-visualization workflow with the same normalized 10-term Zernike basis used by MATLAB; exact AIC/BIC values and surface styling still vary modestly because the Python GLM solver and plotting backend are not byte-identical to MATLAB.
"""


HIPPOCAMPAL_CODE = [
    """
    # nSTAT-python notebook example: HippocampalPlaceCellExample
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
    from nstat.paper_examples_full import run_experiment4

    np.random.seed(0)
    DATA_DIR = notebook_example_data_dir(allow_synthetic=True)
    OUTPUT_ROOT = REPO_ROOT / "output" / "notebook_images"
    __tracker = FigureTracker(topic='HippocampalPlaceCellExample', output_root=OUTPUT_ROOT, expected_count=11)


    def _prepare_figure(matlab_line: str, *, figsize=(8.0, 4.5)):
        fig = __tracker.new_figure(matlab_line)
        fig.clear()
        fig.set_size_inches(*figsize)
        return fig


    def _interp_spike_positions(time_s, x_pos, y_pos, spike_times):
        spike_times = np.asarray(spike_times, dtype=float)
        return (
            np.interp(spike_times, np.asarray(time_s, dtype=float), np.asarray(x_pos, dtype=float)),
            np.interp(spike_times, np.asarray(time_s, dtype=float), np.asarray(y_pos, dtype=float)),
        )


    def _plot_field_grid(fig, animal_key, field_key, title):
        animal = payload[animal_key]
        grid_x = np.asarray(animal["grid_x"], dtype=float)
        grid_y = np.asarray(animal["grid_y"], dtype=float)
        fields = np.asarray(animal[field_key], dtype=float)
        labels = np.asarray(animal["selected_indices"], dtype=int) + 1
        axs = fig.subplots(2, 2, squeeze=False)
        for ax, field, label in zip(axs.ravel(), fields, labels, strict=False):
            image = ax.imshow(
                field,
                origin="lower",
                extent=[float(grid_x.min()), float(grid_x.max()), float(grid_y.min()), float(grid_y.max())],
                aspect="equal",
                cmap="viridis",
            )
            ax.set_title(f"Cell {label}")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(title)
        fig.colorbar(image, ax=axs.ravel().tolist(), shrink=0.78)

    """,
    """
    # SECTION 0: HIPPOCAMPAL PLACE CELL - RECEPTIVE FIELD ESTIMATION
    # This notebook mirrors the MATLAB place-cell helpfile using the dataset-backed Python workflow.
    plt.close("all")
    summary, payload = run_experiment4(DATA_DIR, return_payload=True)
    print(
        {
            "num_cells_fit": int(summary["num_cells_fit"]),
            "mean_delta_aic": round(float(summary["mean_delta_aic_gaussian_minus_zernike"]), 3),
            "mean_delta_bic": round(float(summary["mean_delta_bic_gaussian_minus_zernike"]), 3),
        }
    )
    """,
    """
    # SECTION 1: Example Data
    mesh = payload["mesh"]
    spike_x, spike_y = _interp_spike_positions(mesh["time_s"], mesh["x_pos"], mesh["y_pos"], mesh["spike_times"])
    fig = _prepare_figure("figure(1)", figsize=(6.0, 6.0))
    ax = fig.subplots(1, 1)
    ax.plot(mesh["x_pos"], mesh["y_pos"], color="tab:blue", linewidth=0.8, alpha=0.5)
    ax.scatter(spike_x, spike_y, s=9, color="tab:red", alpha=0.7)
    ax.set_title(f"Animal 1, Cell {int(mesh['cell_index']) + 1}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    """,
    """
    # SECTION 2: Analyze All Cells
    fig = _prepare_figure("Summary.plotSummary", figsize=(7.5, 4.5))
    ax = fig.subplots(1, 1)
    animal1 = payload["animal1"]
    labels = [f"Cell {int(idx) + 1}" for idx in np.asarray(animal1["selected_indices"], dtype=int)]
    ax.bar(np.arange(len(labels)), animal1["delta_aic"], color="tab:purple")
    ax.axhline(0.0, color="0.2", linewidth=1.0)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=20)
    ax.set_ylabel("Gaussian - Zernike AIC")
    ax.set_title("Animal 1 model comparison")

    fig = _prepare_figure("Summary.plotSummary", figsize=(7.5, 4.5))
    ax = fig.subplots(1, 1)
    ax.bar(np.arange(len(labels)), animal1["delta_bic"], color="tab:green")
    ax.axhline(0.0, color="0.2", linewidth=1.0)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=20)
    ax.set_ylabel("Gaussian - Zernike BIC")
    ax.set_title("Animal 1 model comparison")
    """,
    """
    # SECTION 3: View Summary Statistics
    fig = _prepare_figure("Summary.plotSummary", figsize=(7.5, 4.5))
    ax = fig.subplots(1, 1)
    animal2 = payload["animal2"]
    labels = [f"Cell {int(idx) + 1}" for idx in np.asarray(animal2["selected_indices"], dtype=int)]
    ax.bar(np.arange(len(labels)), animal2["delta_aic"], color="tab:purple")
    ax.axhline(0.0, color="0.2", linewidth=1.0)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=20)
    ax.set_ylabel("Gaussian - Zernike AIC")
    ax.set_title("Animal 2 model comparison")

    fig = _prepare_figure("Summary.plotSummary", figsize=(7.5, 4.5))
    ax = fig.subplots(1, 1)
    ax.bar(np.arange(len(labels)), animal2["delta_bic"], color="tab:green")
    ax.axhline(0.0, color="0.2", linewidth=1.0)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=20)
    ax.set_ylabel("Gaussian - Zernike BIC")
    ax.set_title("Animal 2 model comparison")
    """,
    """
    # SECTION 4: Visualize the results
    fig = _prepare_figure("h4=figure(4)", figsize=(8.5, 8.0))
    _plot_field_grid(fig, "animal1", "gaussian_fields", "Gaussian place fields - Animal 1")

    fig = _prepare_figure("h5=figure(5)", figsize=(8.5, 8.0))
    _plot_field_grid(fig, "animal1", "zernike_fields", "Zernike place fields - Animal 1")

    fig = _prepare_figure("h6=figure(6)", figsize=(8.5, 8.0))
    _plot_field_grid(fig, "animal2", "gaussian_fields", "Gaussian place fields - Animal 2")

    fig = _prepare_figure("h7=figure(7)", figsize=(8.5, 8.0))
    _plot_field_grid(fig, "animal2", "zernike_fields", "Zernike place fields - Animal 2")

    fig = _prepare_figure("figure(8)", figsize=(7.0, 5.5))
    ax = fig.subplots(1, 1)
    ax.imshow(
        mesh["gaussian_field"],
        origin="lower",
        extent=[float(np.min(mesh["grid_x"])), float(np.max(mesh["grid_x"])), float(np.min(mesh["grid_y"])), float(np.max(mesh["grid_y"]))],
        aspect="equal",
        cmap="viridis",
    )
    ax.plot(mesh["x_pos"], mesh["y_pos"], color="white", linewidth=0.5, alpha=0.35)
    ax.scatter(spike_x, spike_y, s=8, color="tab:red", alpha=0.7)
    ax.set_title(f"Gaussian receptive field - Cell {int(mesh['cell_index']) + 1}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig = _prepare_figure("figure(9)", figsize=(7.0, 5.5))
    ax = fig.subplots(1, 1)
    ax.imshow(
        mesh["zernike_field"],
        origin="lower",
        extent=[float(np.min(mesh["grid_x"])), float(np.max(mesh["grid_x"])), float(np.min(mesh["grid_y"])), float(np.max(mesh["grid_y"]))],
        aspect="equal",
        cmap="viridis",
    )
    ax.plot(mesh["x_pos"], mesh["y_pos"], color="white", linewidth=0.5, alpha=0.35)
    ax.scatter(spike_x, spike_y, s=8, color="tab:red", alpha=0.7)
    ax.set_title(f"Zernike receptive field - Cell {int(mesh['cell_index']) + 1}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig = _prepare_figure("figure(10)", figsize=(9.0, 4.5))
    axs = fig.subplots(1, 2)
    axs[0].hist(np.concatenate([payload["animal1"]["delta_aic"], payload["animal2"]["delta_aic"]]), bins=8, color="tab:purple", alpha=0.8)
    axs[0].axvline(0.0, color="0.2", linewidth=1.0)
    axs[0].set_title("Distribution of ΔAIC")
    axs[1].hist(np.concatenate([payload["animal1"]["delta_bic"], payload["animal2"]["delta_bic"]]), bins=8, color="tab:green", alpha=0.8)
    axs[1].axvline(0.0, color="0.2", linewidth=1.0)
    axs[1].set_title("Distribution of ΔBIC")

    fig = _prepare_figure("figure(11)", figsize=(6.5, 4.5))
    ax = fig.subplots(1, 1)
    ax.axis("off")
    ax.text(
        0.0,
        0.95,
        "\\n".join(
            [
                f"Cells analyzed: {int(summary['num_cells_fit'])}",
                f"Mean Gaussian-Zernike ΔAIC: {summary['mean_delta_aic_gaussian_minus_zernike']:.2f}",
                f"Mean Gaussian-Zernike ΔBIC: {summary['mean_delta_bic_gaussian_minus_zernike']:.2f}",
                "Negative values favor the Zernike model.",
            ]
        ),
        va="top",
        family="monospace",
        fontsize=10,
    )
    __tracker.finalize()
    """,
]


HYBRID_NOTE = """\
<!-- parity-note -->
## MATLAB Parity Note
- Source MATLAB helpfile: `HybridFilterExample.mlx`
- Fidelity status: `high_fidelity`
- Remaining justified differences: The notebook now reproduces the hybrid-filter simulation, single-run decoding, and averaged summary figures with real outputs; the Python port still uses the current hybrid-filter implementation instead of every MATLAB-specific reporting branch.
"""


HYBRID_CODE = [
    """
    # nSTAT-python notebook example: HybridFilterExample
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

    from nstat.notebook_figures import FigureTracker
    from nstat.paper_examples_full import run_experiment6

    np.random.seed(0)
    OUTPUT_ROOT = REPO_ROOT / "output" / "notebook_images"
    __tracker = FigureTracker(topic='HybridFilterExample', output_root=OUTPUT_ROOT, expected_count=3)


    def _prepare_figure(matlab_line: str, *, figsize=(8.0, 4.5)):
        fig = __tracker.new_figure(matlab_line)
        fig.clear()
        fig.set_size_inches(*figsize)
        return fig


    def _plot_raster(ax, time_s, spikes, *, max_cells=18):
        n_cells = min(int(spikes.shape[1]), max_cells)
        for row in range(n_cells):
            spike_times = np.asarray(time_s, dtype=float)[np.asarray(spikes[:, row], dtype=float) > 0.5]
            if spike_times.size:
                ax.vlines(spike_times, row + 0.6, row + 1.4, color="k", linewidth=0.35)
        ax.set_ylim(0.5, n_cells + 0.5)
        ax.set_ylabel("cell")

    """,
    """
    # SECTION 0: Hybrid Point Process Filter Example
    # This notebook mirrors the MATLAB hybrid-filter helpfile with executable figures.
    plt.close("all")
    summary, payload = run_experiment6(REPO_ROOT, return_payload=True)
    batch_payloads = [run_experiment6(REPO_ROOT, seed=37 + idx, return_payload=True)[1] for idx in range(4)]
    mean_state_prob_2 = np.mean([row["state_prob_2"] for row in batch_payloads], axis=0)
    mean_decoded_x = np.mean([row["decoded_x"] for row in batch_payloads], axis=0)
    mean_decoded_y = np.mean([row["decoded_y"] for row in batch_payloads], axis=0)
    print(
        {
            "num_samples": int(summary["num_samples"]),
            "num_cells": int(summary["num_cells"]),
            "state_accuracy": round(float(summary["state_accuracy"]), 3),
        }
    )
    """,
    """
    # SECTION 1: Problem Statement
    # We infer both a discrete movement state and a continuous reach trajectory from point-process observations.
    """,
    """
    # SECTION 2: Hybrid state-space setup
    # The Python port keeps the same two-state problem structure as MATLAB: a low-motion state and a movement state.
    """,
    """
    # SECTION 3: Generated Simulated Arm Reach
    fig = _prepare_figure("fig1=figure('OuterPosition',[scrsz(3)*.1 scrsz(4)*.1 ...", figsize=(10.0, 9.0))
    axs = fig.subplots(4, 2)
    axs[0, 0].plot(100.0 * payload["x_pos"], 100.0 * payload["y_pos"], color="k", linewidth=1.8)
    axs[0, 0].scatter([100.0 * payload["x_pos"][0]], [100.0 * payload["y_pos"][0]], color="tab:blue", s=35, label="Start")
    axs[0, 0].scatter([100.0 * payload["x_pos"][-1]], [100.0 * payload["y_pos"][-1]], color="tab:red", s=35, label="Finish")
    axs[0, 0].set_title("Reach path")
    axs[0, 0].set_xlabel("X [cm]")
    axs[0, 0].set_ylabel("Y [cm]")
    axs[0, 0].legend(loc="best", frameon=False, fontsize=8)
    _plot_raster(axs[0, 1], payload["time_s"], payload["spikes"])
    axs[0, 1].set_title("Neural raster")
    axs[1, 0].plot(payload["time_s"], payload["state_true"], color="k", linewidth=1.8)
    axs[1, 0].set_yticks([1, 2], ["N", "M"])
    axs[1, 0].set_title("Discrete movement state")
    axs[1, 1].plot(payload["time_s"], 100.0 * payload["x_pos"], color="tab:blue", linewidth=1.3, label="x")
    axs[1, 1].plot(payload["time_s"], 100.0 * payload["y_pos"], color="tab:orange", linewidth=1.3, label="y")
    axs[1, 1].set_title("Position")
    axs[1, 1].legend(loc="best", frameon=False, fontsize=8)
    axs[2, 0].plot(payload["time_s"], 100.0 * payload["x_vel"], color="tab:blue", linewidth=1.3, label="vx")
    axs[2, 0].plot(payload["time_s"], 100.0 * payload["y_vel"], color="tab:orange", linewidth=1.3, label="vy")
    axs[2, 0].set_title("Velocity")
    axs[2, 0].legend(loc="best", frameon=False, fontsize=8)
    axs[2, 1].plot(payload["time_s"], np.mean(payload["spikes"], axis=1), color="tab:green", linewidth=1.2)
    axs[2, 1].set_title("Population spike fraction")
    axs[3, 0].plot(payload["time_s"], np.cumsum(payload["spikes"], axis=0)[:, 0], color="tab:purple", linewidth=1.1)
    axs[3, 0].set_title("Example cumulative spike count")
    axs[3, 1].axis("off")
    axs[3, 1].text(
        0.0,
        0.95,
        "\\n".join(
            [
                f"Cells: {int(summary['num_cells'])}",
                f"State accuracy: {summary['state_accuracy']:.3f}",
                f"Decode RMSE X: {summary['decode_rmse_x']:.3f}",
                f"Decode RMSE Y: {summary['decode_rmse_y']:.3f}",
            ]
        ),
        va="top",
        family="monospace",
        fontsize=9,
    )
    """,
    """
    # SECTION 4: Simulate Neural Firing
    # The simulated spike population depends on the latent state and the movement dynamics.
    """,
    """
    # SECTION 5: Run the hybrid filter
    fig = _prepare_figure("subplot(4,3,[1 4])", figsize=(11.0, 9.0))
    axs = fig.subplots(4, 3)
    decoded_vx = np.gradient(payload["decoded_x"], payload["time_s"])
    decoded_vy = np.gradient(payload["decoded_y"], payload["time_s"])
    axs[0, 0].plot(payload["time_s"], payload["state_true"], color="k", linewidth=1.8, label="True")
    axs[0, 0].plot(payload["time_s"], payload["state_hat"], color="tab:blue", linewidth=1.0, label="Estimated")
    axs[0, 0].set_yticks([1, 2], ["N", "M"])
    axs[0, 0].set_title("State estimate")
    axs[0, 0].legend(loc="best", frameon=False, fontsize=8)
    axs[0, 1].plot(payload["time_s"], payload["state_prob_2"], color="tab:blue", linewidth=1.2)
    axs[0, 1].set_title("Pr(Movement)")
    axs[0, 2].plot(100.0 * payload["x_pos"], 100.0 * payload["y_pos"], color="k", linewidth=1.6, label="True")
    axs[0, 2].plot(100.0 * payload["decoded_x"], 100.0 * payload["decoded_y"], color="tab:blue", linewidth=1.2, label="Decoded")
    axs[0, 2].set_title("Movement path")
    axs[0, 2].legend(loc="best", frameon=False, fontsize=8)
    axs[1, 0].plot(payload["time_s"], 100.0 * payload["x_pos"], color="k", linewidth=1.6)
    axs[1, 0].plot(payload["time_s"], 100.0 * payload["decoded_x"], color="tab:blue", linewidth=1.2)
    axs[1, 0].set_title("X position")
    axs[1, 1].plot(payload["time_s"], 100.0 * payload["y_pos"], color="k", linewidth=1.6)
    axs[1, 1].plot(payload["time_s"], 100.0 * payload["decoded_y"], color="tab:blue", linewidth=1.2)
    axs[1, 1].set_title("Y position")
    axs[1, 2].plot(payload["time_s"], 100.0 * payload["x_vel"], color="k", linewidth=1.6)
    axs[1, 2].plot(payload["time_s"], 100.0 * decoded_vx, color="tab:blue", linewidth=1.2)
    axs[1, 2].set_title("X velocity")
    axs[2, 0].plot(payload["time_s"], 100.0 * payload["y_vel"], color="k", linewidth=1.6)
    axs[2, 0].plot(payload["time_s"], 100.0 * decoded_vy, color="tab:blue", linewidth=1.2)
    axs[2, 0].set_title("Y velocity")
    axs[2, 1].plot(payload["time_s"], np.sqrt((payload["decoded_x"] - payload["x_pos"]) ** 2 + (payload["decoded_y"] - payload["y_pos"]) ** 2), color="tab:red", linewidth=1.2)
    axs[2, 1].set_title("Instantaneous path error")
    axs[2, 2].hist(np.sum(payload["spikes"], axis=0), bins=12, color="tab:green", alpha=0.85)
    axs[2, 2].set_title("Spike counts per cell")
    axs[3, 0].axis("off")
    axs[3, 1].axis("off")
    axs[3, 2].axis("off")

    fig = _prepare_figure("plot(time,mean(S_estAll))", figsize=(10.0, 7.0))
    axs = fig.subplots(2, 2)
    axs[0, 0].plot(payload["time_s"], payload["state_true"], color="k", linewidth=1.6, label="True state")
    axs[0, 0].plot(payload["time_s"], 1.0 + (mean_state_prob_2 > 0.5).astype(float), color="tab:blue", linewidth=1.2, label="Mean estimate")
    axs[0, 0].set_yticks([1, 2], ["N", "M"])
    axs[0, 0].legend(loc="best", frameon=False, fontsize=8)
    axs[0, 0].set_title("Average state estimate")
    axs[0, 1].plot(payload["time_s"], mean_state_prob_2, color="tab:blue", linewidth=1.2)
    axs[0, 1].set_title("Average Pr(Movement)")
    axs[1, 0].plot(100.0 * payload["x_pos"], 100.0 * payload["y_pos"], color="k", linewidth=1.6, label="True")
    axs[1, 0].plot(100.0 * mean_decoded_x, 100.0 * mean_decoded_y, color="tab:blue", linewidth=1.2, label="Mean decoded")
    axs[1, 0].legend(loc="best", frameon=False, fontsize=8)
    axs[1, 0].set_title("Average decoded path")
    axs[1, 1].bar(
        ["X RMSE", "Y RMSE"],
        [summary["decode_rmse_x"], summary["decode_rmse_y"]],
        color=["tab:blue", "tab:orange"],
    )
    axs[1, 1].set_title("Single-run decoding RMSE")
    __tracker.finalize()
    """,
]


STIMULUS_2D_NOTE = """\
<!-- parity-note -->
## MATLAB Parity Note
- Source MATLAB helpfile: `StimulusDecode2D.mlx`
- Fidelity status: `partial`
- Remaining justified differences: The notebook reproduces the MATLAB section order, figure inventory, simulated receptive fields, and decoded-trajectory presentation, but the current Python decoder still uses regression-based state recovery instead of MATLAB's symbolic-CIF nonlinear filter.
"""


STIMULUS_2D_CODE = [
    """
    # nSTAT-python notebook example: StimulusDecode2D
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

    from nstat import DecodingAlgorithms
    from nstat.notebook_figures import FigureTracker

    np.random.seed(0)
    OUTPUT_ROOT = REPO_ROOT / "output" / "notebook_images"
    __tracker = FigureTracker(topic='StimulusDecode2D', output_root=OUTPUT_ROOT, expected_count=6)


    def _prepare_figure(matlab_line: str, *, figsize=(8.0, 4.5)):
        fig = __tracker.new_figure(matlab_line)
        fig.clear()
        fig.set_size_inches(*figsize)
        return fig


    def _simulate_decode(seed=19, *, n_cells=24, dt=0.01, tmax=20.0):
        rng = np.random.default_rng(seed)
        time = np.arange(0.0, tmax + dt, dt)
        vel = np.cumsum(rng.normal(0.0, 0.05, size=(time.size, 2)), axis=0)
        vel = 0.18 * vel / np.maximum(np.std(vel, axis=0, ddof=1), 1e-6)
        pos = np.cumsum(vel, axis=0) * dt
        pos = pos - np.mean(pos, axis=0, keepdims=True)
        px = pos[:, 0]
        py = pos[:, 1]
        coeffs = np.column_stack(
            [
                -2.2 - np.abs(rng.normal(0.0, 0.35, size=n_cells)),
                rng.normal(0.0, 1.1, size=n_cells),
                rng.normal(0.0, 1.1, size=n_cells),
                -np.abs(rng.normal(1.6, 0.35, size=n_cells)),
                -np.abs(rng.normal(1.6, 0.35, size=n_cells)),
                rng.normal(0.0, 0.45, size=n_cells),
            ]
        )
        design = np.column_stack([np.ones(time.size), px, py, px * px, py * py, px * py])
        spikes = np.zeros((time.size, n_cells), dtype=float)
        firing_prob = np.zeros_like(spikes)
        for idx in range(n_cells):
            eta = design @ coeffs[idx]
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -20.0, 20.0)))
            firing_prob[:, idx] = p
            spikes[:, idx] = (rng.random(time.size) < p).astype(float)
        grid = np.linspace(-1.4, 1.4, 60)
        gx, gy = np.meshgrid(grid, grid)
        grid_design = np.column_stack([np.ones(gx.size), gx.ravel(), gy.ravel(), gx.ravel() ** 2, gy.ravel() ** 2, gx.ravel() * gy.ravel()])
        fields = []
        for idx in range(n_cells):
            eta = grid_design @ coeffs[idx]
            field = (1.0 / (1.0 + np.exp(-np.clip(eta, -20.0, 20.0)))).reshape(gx.shape)
            fields.append(field)
        subset = max(n_cells // 2, 1)
        dec_x_subset = DecodingAlgorithms.linear_decode(spikes[:, :subset], px)
        dec_y_subset = DecodingAlgorithms.linear_decode(spikes[:, :subset], py)
        dec_x_full = DecodingAlgorithms.linear_decode(spikes, px)
        dec_y_full = DecodingAlgorithms.linear_decode(spikes, py)
        return {
            "time_s": time,
            "px": px,
            "py": py,
            "vx": vel[:, 0],
            "vy": vel[:, 1],
            "spikes": spikes,
            "firing_prob": firing_prob,
            "fields": np.asarray(fields, dtype=float),
            "grid_x": gx,
            "grid_y": gy,
            "decoded_subset_x": dec_x_subset["decoded"],
            "decoded_subset_y": dec_y_subset["decoded"],
            "decoded_full_x": dec_x_full["decoded"],
            "decoded_full_y": dec_y_full["decoded"],
            "rmse_full": float(np.sqrt(np.mean((dec_x_full["decoded"] - px) ** 2 + (dec_y_full["decoded"] - py) ** 2))),
        }


    def _plot_raster(ax, time_s, spikes, *, max_cells=20):
        n_cells = min(int(spikes.shape[1]), max_cells)
        for row in range(n_cells):
            spike_times = np.asarray(time_s, dtype=float)[np.asarray(spikes[:, row], dtype=float) > 0.5]
            if spike_times.size:
                ax.vlines(spike_times, row + 0.6, row + 1.4, color="k", linewidth=0.35)
        ax.set_ylim(0.5, n_cells + 0.5)
        ax.set_ylabel("cell")

    """,
    """
    # SECTION 0: 2-D Stimulus Decode
    # This notebook follows the MATLAB 2-D decoding workflow with simulated spatial receptive fields.
    plt.close("all")
    payload = _simulate_decode()
    print({"num_cells": int(payload["spikes"].shape[1]), "rmse_full": round(float(payload["rmse_full"]), 4)})
    """,
    """
    # SECTION 1: Generate the random receptive fields to simulate different neurons
    fig = _prepare_figure("figure; plot(px,py)", figsize=(6.0, 6.0))
    ax = fig.subplots(1, 1)
    ax.plot(payload["px"], payload["py"], color="tab:blue", linewidth=1.5)
    ax.set_title("Simulated X-Y trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

    fig = _prepare_figure("lambda{i}.plot", figsize=(9.0, 5.0))
    ax = fig.subplots(1, 1)
    show = [0, 1, 2, 3]
    for idx in show:
        ax.plot(payload["time_s"], payload["firing_prob"][:, idx], linewidth=1.2, label=f"Cell {idx + 1}")
    ax.set_title("Example firing probabilities")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("spike probability")
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    fig = _prepare_figure("pcolor(X,Y,placeField{i}), shading interp", figsize=(8.0, 8.0))
    axs = fig.subplots(2, 2, squeeze=False)
    for ax, idx in zip(axs.ravel(), show, strict=False):
        image = ax.imshow(
            payload["fields"][idx],
            origin="lower",
            extent=[float(payload["grid_x"].min()), float(payload["grid_x"].max()), float(payload["grid_y"].min()), float(payload["grid_y"].max())],
            aspect="equal",
            cmap="viridis",
        )
        ax.set_title(f"Cell {idx + 1}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=axs.ravel().tolist(), shrink=0.78)
    """,
    """
    # SECTION 2: Visualize the simulated neural activity
    fig = _prepare_figure("spikeColl.plot", figsize=(9.0, 5.0))
    axs = fig.subplots(2, 1, sharex=True)
    _plot_raster(axs[0], payload["time_s"], payload["spikes"])
    axs[0].set_title("Population raster")
    axs[1].plot(payload["time_s"], np.mean(payload["spikes"], axis=1), color="tab:green", linewidth=1.2)
    axs[1].set_title("Population firing fraction")
    axs[1].set_xlabel("time (s)")
    axs[1].set_ylabel("mean spike/bin")
    """,
    """
    # SECTION 3: Decode the x-y trajectory
    fig = _prepare_figure("plot(x_u(1,:),x_u(2,:),'b',px,py,'k')", figsize=(6.0, 6.0))
    ax = fig.subplots(1, 1)
    ax.plot(payload["px"], payload["py"], color="k", linewidth=1.8, label="True path")
    ax.plot(payload["decoded_subset_x"], payload["decoded_subset_y"], color="tab:orange", linewidth=1.0, label="Subset decode")
    ax.plot(payload["decoded_full_x"], payload["decoded_full_y"], color="tab:blue", linewidth=1.2, label="Full decode")
    ax.set_title("Decoded X-Y trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", frameon=False, fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    fig = _prepare_figure("plot(decoded trajectories)", figsize=(10.0, 5.5))
    axs = fig.subplots(2, 1, sharex=True)
    axs[0].plot(payload["time_s"], payload["px"], color="k", linewidth=1.6, label="True x")
    axs[0].plot(payload["time_s"], payload["decoded_full_x"], color="tab:blue", linewidth=1.2, label="Decoded x")
    axs[0].plot(payload["time_s"], payload["decoded_subset_x"], color="tab:orange", linewidth=1.0, label="Subset x")
    axs[0].legend(loc="best", frameon=False, fontsize=8)
    axs[0].set_ylabel("x")
    axs[1].plot(payload["time_s"], payload["py"], color="k", linewidth=1.6, label="True y")
    axs[1].plot(payload["time_s"], payload["decoded_full_y"], color="tab:blue", linewidth=1.2, label="Decoded y")
    axs[1].plot(payload["time_s"], payload["decoded_subset_y"], color="tab:orange", linewidth=1.0, label="Subset y")
    axs[1].set_ylabel("y")
    axs[1].set_xlabel("time (s)")

    fig = _prepare_figure("decode_rmse", figsize=(7.0, 4.5))
    ax = fig.subplots(1, 1)
    error_full = np.sqrt((payload["decoded_full_x"] - payload["px"]) ** 2 + (payload["decoded_full_y"] - payload["py"]) ** 2)
    error_subset = np.sqrt((payload["decoded_subset_x"] - payload["px"]) ** 2 + (payload["decoded_subset_y"] - payload["py"]) ** 2)
    ax.plot(payload["time_s"], error_full, color="tab:blue", linewidth=1.2, label="Full decode")
    ax.plot(payload["time_s"], error_subset, color="tab:orange", linewidth=1.0, label="Subset decode")
    ax.set_title(f"Pointwise decoding error (RMSE={payload['rmse_full']:.3f})")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Euclidean error")
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
    _write_notebook(
        NOTEBOOK_DIR / "HippocampalPlaceCellExample.ipynb",
        topic="HippocampalPlaceCellExample",
        expected_figures=11,
        markdown_note=HIPPOCAMPAL_NOTE,
        code_cells=HIPPOCAMPAL_CODE,
    )
    _write_notebook(
        NOTEBOOK_DIR / "HybridFilterExample.ipynb",
        topic="HybridFilterExample",
        expected_figures=3,
        markdown_note=HYBRID_NOTE,
        code_cells=HYBRID_CODE,
    )
    _write_notebook(
        NOTEBOOK_DIR / "StimulusDecode2D.ipynb",
        topic="StimulusDecode2D",
        expected_figures=6,
        markdown_note=STIMULUS_2D_NOTE,
        code_cells=STIMULUS_2D_CODE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
