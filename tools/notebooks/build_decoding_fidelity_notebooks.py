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
        cells=[new_markdown_cell(markdown_note), *[new_code_cell(dedent(cell).strip() + "\n") for cell in code_cells]],
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


DECODING_EXAMPLE_NOTE = """\
<!-- parity-note -->
## MATLAB Parity Note
- Source MATLAB helpfile: `DecodingExample.mlx`
- Fidelity status: `high_fidelity`
- Remaining justified differences: Workflow, model fitting, and decoded-stimulus figures now follow the MATLAB helpfile closely; exact traces still depend on stochastic simulation draws and Python plotting defaults.
"""


DECODING_EXAMPLE_CODE = [
    """
    # nSTAT-python notebook example: DecodingExample
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

    from nstat import Analysis, CIF, ConfigColl, CovColl, Covariate, DecodingAlgorithms, Trial, TrialConfig
    from nstat.notebook_figures import FigureTracker

    np.random.seed(0)
    OUTPUT_ROOT = REPO_ROOT / "output" / "notebook_images"
    __tracker = FigureTracker(topic="DecodingExample", output_root=OUTPUT_ROOT, expected_count=5)


    def _prepare_figure(matlab_line: str, *, figsize=(8.0, 4.5)):
        fig = __tracker.new_figure(matlab_line)
        fig.clear()
        fig.set_size_inches(*figsize)
        return fig


    def _plot_raster(ax, spike_coll):
        for row in range(1, spike_coll.numSpikeTrains + 1):
            train = spike_coll.getNST(row)
            spikes = np.asarray(train.getSpikeTimes(), dtype=float).reshape(-1)
            if spikes.size:
                ax.vlines(spikes, row - 0.4, row + 0.4, color="k", linewidth=0.5)
        ax.set_ylabel("Neuron")
        ax.set_ylim(0.5, spike_coll.numSpikeTrains + 0.5)


    def _plot_decoded_ci(ax, time, decoded, cov, stim, title):
        center = np.asarray(decoded, dtype=float).reshape(-1)
        variance = np.asarray(cov, dtype=float).reshape(-1)
        sigma = np.sqrt(np.maximum(variance, 0.0))
        z_val = 3.0
        lower = center - z_val * sigma
        upper = center + z_val * sigma
        ax.plot(time[: center.size], center, "b", linewidth=1.5, label="x_{k|k}(t)")
        ax.plot(time[: center.size], lower, "g", linewidth=1.0, label="x_{k|k}(t)-3σ")
        ax.plot(time[: center.size], upper, "g", linewidth=1.0, label="x_{k|k}(t)+3σ")
        ax.plot(time[: center.size], np.asarray(stim).reshape(-1)[: center.size], "k", linewidth=1.5, label="x(t)")
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.legend(loc="upper right", frameon=False, fontsize=8)


    # SECTION 0: STIMULUS DECODING
    # In this example we decode a univariate stimulus from simulated point-process observations by following the MATLAB DecodingExample workflow.
    """,
    """
    # SECTION 1: Generate the conditional Intensity Function
    plt.close("all")
    delta = 0.001
    Tmax = 10.0
    time = np.arange(0.0, Tmax + delta, delta)
    f = 0.1
    b1 = 1.0
    b0 = -3.0
    x = np.sin(2.0 * np.pi * f * time)
    exp_data = np.exp(b1 * x + b0)
    lambda_data = exp_data / (1.0 + exp_data)
    lambda_cov = Covariate(time, lambda_data / delta, "\\\\Lambda(t)", "time", "s", "Hz", ["lambda_1"])

    numRealizations = 10
    spikeColl = CIF.simulateCIFByThinningFromLambda(lambda_cov, numRealizations=numRealizations)

    fig = _prepare_figure("figure", figsize=(8.0, 5.5))
    axs = fig.subplots(2, 1, sharex=True)
    _plot_raster(axs[0], spikeColl)
    axs[0].set_title("Simulated spike trains from λ(t)")
    axs[1].plot(time, lambda_cov.data[:, 0], color="b", linewidth=2.0)
    axs[1].set_title("Conditional intensity λ(t)")
    axs[1].set_xlabel("time (s)")
    axs[1].set_ylabel("Hz")
    """,
    """
    # SECTION 2: Fit a model to the spikedata to obtain a model CIF
    stim = Covariate(time, x, "Stimulus", "time", "s", "V", ["stim"])
    baseline = Covariate(time, np.ones_like(time), "Baseline", "time", "s", "", ["constant"])
    cc = CovColl([stim, baseline])
    trial = Trial(spikeColl, cc)

    fig = _prepare_figure("figure", figsize=(8.0, 6.0))
    axs = fig.subplots(3, 1, sharex=True)
    _plot_raster(axs[0], spikeColl)
    axs[0].set_title("Trial spike raster")
    axs[1].plot(time, stim.data[:, 0], color="k", linewidth=1.5)
    axs[1].set_title("Stimulus covariate")
    axs[1].set_ylabel("V")
    axs[2].plot(time, baseline.data[:, 0], color="0.3", linewidth=1.5)
    axs[2].set_title("Baseline covariate")
    axs[2].set_ylabel("constant")
    axs[2].set_xlabel("time (s)")

    cfgColl = ConfigColl(
        [
            TrialConfig([["Baseline", "constant"]], 1000.0, [], [], name="Baseline"),
            TrialConfig([["Baseline", "constant"], ["Stimulus", "stim"]], 1000.0, [], [], name="Baseline+Stimulus"),
        ]
    )
    results = Analysis.RunAnalysisForAllNeurons(trial, cfgColl, 0)

    paramEst = np.column_stack([fit.getCoeffs(2)[:2] for fit in results])
    meanParams = np.mean(paramEst, axis=1)
    aic_matrix = np.vstack([fit.AIC for fit in results])
    logll_matrix = np.vstack([fit.logLL for fit in results])
    config_names = results[0].configNames

    fig = _prepare_figure("figure", figsize=(8.0, 4.5))
    axs = fig.subplots(1, 2)
    neuron_idx = np.arange(1, paramEst.shape[1] + 1)
    axs[0].plot(neuron_idx, paramEst[0], "o-", color="tab:blue", label="b0")
    axs[0].axhline(meanParams[0], color="tab:blue", linestyle="--", linewidth=1.0)
    axs[0].set_title("Baseline coefficients")
    axs[0].set_xlabel("Neuron")
    axs[0].set_ylabel("b0")
    axs[1].plot(neuron_idx, paramEst[1], "o-", color="tab:orange", label="b1")
    axs[1].axhline(meanParams[1], color="tab:orange", linestyle="--", linewidth=1.0)
    axs[1].set_title("Stimulus coefficients")
    axs[1].set_xlabel("Neuron")
    axs[1].set_ylabel("b1")

    fig = _prepare_figure("figure", figsize=(8.0, 4.5))
    axs = fig.subplots(1, 2)
    xloc = np.arange(len(config_names))
    axs[0].bar(xloc, np.mean(aic_matrix, axis=0), color=["0.6", "0.3"])
    axs[0].set_xticks(xloc, config_names, rotation=15)
    axs[0].set_title("Mean AIC across neurons")
    axs[1].bar(xloc, np.mean(logll_matrix, axis=0), color=["0.6", "0.3"])
    axs[1].set_xticks(xloc, config_names, rotation=15)
    axs[1].set_title("Mean log-likelihood across neurons")
    """,
    """
    # SECTION 3: Decode the stimulus from the fitted CIF
    b0_est = paramEst[0, :]
    b1_est = paramEst[1, :]
    lambdaCIF = [CIF([b0_est[i], b1_est[i]], ["1", "x"], ["x"], "binomial") for i in range(numRealizations)]

    spikeColl.resample(1.0 / delta)
    dN = spikeColl.dataToMatrix()
    Q = 2.0 * np.std(np.diff(stim.data[:, 0]))
    A = 1.0
    x_p, W_p, x_u, W_u, *_ = DecodingAlgorithms.PPDecodeFilterLinear(A, Q, dN.T, b0_est, b1_est, "binomial", delta)

    fig = _prepare_figure("figure", figsize=(8.0, 4.5))
    ax = fig.subplots(1, 1)
    _plot_decoded_ci(ax, time, x_u, W_u, stim.data[:, 0], f"Decoded stimulus using {numRealizations} cells")
    __tracker.finalize()
    """,
]


DECODING_HISTORY_NOTE = """\
<!-- parity-note -->
## MATLAB Parity Note
- Source MATLAB helpfile: `DecodingExampleWithHist.mlx`
- Fidelity status: `high_fidelity`
- Remaining justified differences: The notebook now mirrors the MATLAB history-aware decoding workflow closely; exact stochastic trajectories and figure styling still vary slightly under Python execution.
"""


DECODING_HISTORY_CODE = [
    """
    # nSTAT-python notebook example: DecodingExampleWithHist
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

    from nstat import CIF, DecodingAlgorithms, History, Covariate, nspikeTrain, nstColl
    from nstat.notebook_figures import FigureTracker

    np.random.seed(0)
    OUTPUT_ROOT = REPO_ROOT / "output" / "notebook_images"
    __tracker = FigureTracker(topic="DecodingExampleWithHist", output_root=OUTPUT_ROOT, expected_count=2)


    def _prepare_figure(matlab_line: str, *, figsize=(8.0, 4.5)):
        fig = __tracker.new_figure(matlab_line)
        fig.clear()
        fig.set_size_inches(*figsize)
        return fig


    def _plot_raster(ax, spike_coll):
        for row in range(1, spike_coll.numSpikeTrains + 1):
            train = spike_coll.getNST(row)
            spikes = np.asarray(train.getSpikeTimes(), dtype=float).reshape(-1)
            if spikes.size:
                ax.vlines(spikes, row - 0.4, row + 0.4, color="k", linewidth=0.5)
        ax.set_ylabel("Neuron")
        ax.set_ylim(0.5, spike_coll.numSpikeTrains + 0.5)


    def _plot_decoded_ci(ax, time, decoded, cov, stim, title):
        center = np.asarray(decoded, dtype=float).reshape(-1)
        spread = np.asarray(cov, dtype=float).reshape(-1)
        z_val = 3.0
        lower = center - z_val * spread
        upper = center + z_val * spread
        ax.plot(time[: center.size], center, "b", linewidth=1.5, label="x_{k|k}(t)")
        ax.plot(time[: center.size], lower, "g", linewidth=1.0, label="x_{k|k}(t)-3σ")
        ax.plot(time[: center.size], upper, "r", linewidth=1.0, label="x_{k|k}(t)+3σ")
        ax.plot(time[: center.size], np.asarray(stim).reshape(-1)[: center.size], "k", linewidth=1.5, label="x(t)")
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.legend(loc="upper right", frameon=False, fontsize=8)


    def _simulate_history_spike_train(time, stim_data, baseline, hist_coeffs, window_times):
        spikes = []
        for idx in range(1, len(time)):
            t = time[idx]
            spike_arr = np.asarray(spikes, dtype=float)
            history_counts = []
            for w_start, w_stop in zip(window_times[:-1], window_times[1:]):
                if spike_arr.size:
                    history_counts.append(np.sum((spike_arr >= t - w_stop) & (spike_arr < t - w_start)))
                else:
                    history_counts.append(0.0)
            eta = baseline + stim_data[idx] + float(np.dot(hist_coeffs, history_counts))
            p = np.exp(np.clip(eta, -20.0, 20.0))
            p = p / (1.0 + p)
            if np.random.rand() < p:
                spikes.append(t)
        return np.asarray(spikes, dtype=float)


    # SECTION 0: 1-D Stimulus Decode with History Effect
    # We simulate neurons with refractory-history effects and compare point-process decoding with and without the correct history terms.
    """,
    """
    # SECTION 1: History-aware decoding workflow
    plt.close("all")
    delta = 0.001
    Tmax = 1.0
    time = np.arange(0.0, Tmax + delta, delta)
    f = 1.0
    b1 = 1.0
    b0 = -2.0
    stimData = b1 * np.sin(2.0 * np.pi * f * time)
    histCoeffs = np.array([-2.0, -2.0, -4.0])
    windowTimes = np.array([0.0, 0.001, 0.002, 0.003])
    histObj = History(windowTimes)
    stim = Covariate(time, stimData, "Stimulus", "time", "s", "Voltage", ["sin"])

    numRealizations = 20
    trains = []
    for idx in range(numRealizations):
        spikes = _simulate_history_spike_train(time, stimData, b0, histCoeffs, windowTimes)
        trains.append(nspikeTrain(spikes, str(idx + 1), delta, 0.0, Tmax, makePlots=-1))
    sC = nstColl(trains)

    fig = _prepare_figure("figure", figsize=(8.0, 5.5))
    axs = fig.subplots(2, 1, sharex=True)
    _plot_raster(axs[0], sC)
    axs[0].set_title("History-dependent simulated spike trains")
    axs[1].plot(time, stim.data[:, 0], color="k", linewidth=1.5)
    axs[1].set_title("Stimulus")
    axs[1].set_xlabel("time (s)")
    axs[1].set_ylabel("Voltage")

    lambdaCIF = [CIF([b0, b1], ["1", "x"], ["x"], "binomial", histCoeffs, histObj) for _ in range(numRealizations)]
    lambdaCIFNoHist = [CIF([b0, b1], ["1", "x"], ["x"], "binomial") for _ in range(numRealizations)]

    sC.resample(1.0 / delta)
    dN = sC.dataToMatrix()
    Q = 2.0 * np.std(np.diff(stim.data[:, 0]))
    Px0 = 0.1
    A = 1.0
    x_p, W_p, x_u, W_u, *_ = DecodingAlgorithms.PPDecodeFilter(A, Q, Px0, dN.T, lambdaCIF, delta)
    x_p_no_hist, W_p_no_hist, x_u_no_hist, W_u_no_hist, *_ = DecodingAlgorithms.PPDecodeFilter(
        A,
        Q,
        Px0,
        dN.T,
        lambdaCIFNoHist,
        delta,
    )

    fig = _prepare_figure("figure", figsize=(8.0, 6.0))
    axs = fig.subplots(2, 1, sharex=True)
    _plot_decoded_ci(axs[0], time, x_u, W_u, stim.data[:, 0], f"Decoded stimulus with history using {numRealizations} cells")
    _plot_decoded_ci(axs[1], time, x_u_no_hist, W_u_no_hist, stim.data[:, 0], f"Decoded stimulus without history using {numRealizations} cells")
    __tracker.finalize()
    """,
]


def main() -> int:
    _write_notebook(
        NOTEBOOK_DIR / "DecodingExample.ipynb",
        topic="DecodingExample",
        expected_figures=5,
        markdown_note=DECODING_EXAMPLE_NOTE,
        code_cells=DECODING_EXAMPLE_CODE,
    )
    _write_notebook(
        NOTEBOOK_DIR / "DecodingExampleWithHist.ipynb",
        topic="DecodingExampleWithHist",
        expected_figures=2,
        markdown_note=DECODING_HISTORY_NOTE,
        code_cells=DECODING_HISTORY_CODE,
    )
    print(NOTEBOOK_DIR / "DecodingExample.ipynb")
    print(NOTEBOOK_DIR / "DecodingExampleWithHist.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
