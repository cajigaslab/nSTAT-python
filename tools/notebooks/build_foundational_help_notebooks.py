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


TRIAL_NOTE = """\
<!-- parity-note -->
## MATLAB Parity Note
- Source MATLAB helpfile: `TrialExamples.mlx`
- Fidelity status: `high_fidelity`
- Remaining justified differences: The notebook now mirrors the MATLAB Trial workflow with executable object construction, masking, history extraction, and plotting; the closing analysis section uses one representative Python `Analysis` run instead of linking out to separate MATLAB help pages.
"""


TRIAL_CODE = [
    """
    # nSTAT-python notebook example: TrialExamples
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

    from nstat import Analysis, ConfigColl, CovColl, Covariate, Events, History, Trial, TrialConfig, nspikeTrain, nstColl
    from nstat.notebook_figures import FigureTracker

    np.random.seed(7)
    OUTPUT_ROOT = REPO_ROOT / "output" / "notebook_images"
    __tracker = FigureTracker(topic='TrialExamples', output_root=OUTPUT_ROOT, expected_count=6)


    def _figure(label: str, *, figsize=(8.5, 3.5)):
        fig = __tracker.new_figure(label)
        fig.clear()
        fig.set_size_inches(*figsize)
        return fig


    def _build_trial():
        length_trial = 1.0
        sample_rate = 1000.0
        time = np.linspace(0.0, length_trial, int(length_trial * sample_rate) + 1)

        position = Covariate(
            time,
            np.column_stack([np.sin(2 * np.pi * time), np.cos(2 * np.pi * time)]),
            "Position",
            "time",
            "s",
            "a.u.",
            ["x", "y"],
        )
        force = Covariate(
            time,
            np.column_stack(
                [
                    0.5 * np.cos(4 * np.pi * time) + 0.15 * np.sin(2 * np.pi * time),
                    0.4 * np.sin(4 * np.pi * time + 0.3),
                ]
            ),
            "Force",
            "time",
            "s",
            "N",
            ["f_x", "f_y"],
        )
        cov_coll = CovColl([position, force])
        cov_coll.setMaxTime(length_trial)

        events = Events([0.18, 0.72], ["E_1", "E_2"])
        history = History([0.0, 0.1, 0.2, 0.4])

        trains = []
        base_grid = np.linspace(0.05, 0.95, 100)
        for neuron_index, phase in enumerate(np.linspace(0.0, np.pi / 2.0, 4), start=1):
            spikes = np.clip(base_grid + 0.008 * np.sin(2 * np.pi * base_grid * (neuron_index + 1) + phase), 0.0, length_trial)
            trains.append(
                nspikeTrain(
                    np.sort(spikes),
                    name=str(neuron_index),
                    minTime=0.0,
                    maxTime=length_trial,
                    makePlots=-1,
                )
            )
        spike_coll = nstColl(trains)
        trial = Trial(spike_coll, cov_coll, events, history)
        return {
            "length_trial": length_trial,
            "sample_rate": sample_rate,
            "history": history,
            "cov_coll": cov_coll,
            "events": events,
            "spike_coll": spike_coll,
            "trial": trial,
        }


    ctx = _build_trial()
    print(
        {
            "trial_duration_s": ctx["length_trial"],
            "num_neurons": ctx["spike_coll"].numSpikeTrains,
            "covariates": ctx["cov_coll"].names,
            "history_windows": ctx["history"].windowTimes.tolist(),
        }
    )
    """,
    """
    # SECTION 1: Example 1: A simple data set
    plt.close("all")
    trial1 = ctx["trial"]
    spikeColl = ctx["spike_coll"]
    cc = ctx["cov_coll"]
    e = ctx["events"]
    h = ctx["history"]
    """,
    """
    # SECTION 2: Create History windows of interest
    fig = _figure("figure; h.plot", figsize=(8.0, 2.5))
    ax = fig.subplots(1, 1)
    h.plot(handle=ax)
    """,
    """
    # SECTION 3: Load Covariates
    fig = _figure("figure; cc.plot", figsize=(8.5, 5.0))
    cc.plot(handle=fig)
    """,
    """
    # SECTION 4: Create trial events
    fig = _figure("figure; e.plot", figsize=(8.0, 2.3))
    ax = fig.subplots(1, 1)
    e.plot(handle=ax)
    """,
    """
    # SECTION 5: Create neural Spike Train Data
    fig = _figure("figure; spikeColl.plot", figsize=(8.5, 3.5))
    ax = fig.subplots(1, 1)
    spikeColl.plot(handle=ax)
    """,
    """
    # SECTION 6: Finally we have everything we need to create a Trial object.
    fig = _figure("figure; trial1.plot", figsize=(9.0, 8.0))
    trial1.plot(handle=fig)
    """,
    """
    # SECTION 7: Mask out some of the data and plot the trial once again
    trial1.setCovMask([["Position", "x"], ["Force", "f_x"]])
    fig = _figure("figure; trial1.plot masked", figsize=(9.0, 8.0))
    trial1.plot(handle=fig)
    hist_cov = trial1.getHistForNeurons([1, 2])
    print({"masked_labels": trial1.getLabelsFromMask(1), "history_covariates": hist_cov.getAllCovLabels()[:4]})
    trial1.resetCovMask()
    """,
    """
    # SECTION 8: Example 2: Analyzing Trial Data
    cfg = TrialConfig([["Position", "x"], ["Force", "f_x"]], sampleRate=ctx["sample_rate"], history=[0.0, 0.05, 0.1], name="Position+Force+History")
    cfgColl = ConfigColl([cfg])
    fit = Analysis.RunAnalysisForNeuron(trial1, 1, cfgColl)
    fit_stats = fit.computeKSStats()
    print(
        {
            "config_name": fit.configNames[0],
            "aic": round(float(fit.AIC[0]), 3),
            "bic": round(float(fit.BIC[0]), 3),
            "ks_stat": round(float(fit_stats["ks_stat"]), 4),
        }
    )
    """,
    """
    # SECTION 9: Related analysis workflows
    print("For larger model-comparison walkthroughs, continue with AnalysisExamples and AnalysisExamples2.")
    __tracker.finalize()
    """,
]


PPSIM_NOTE = """\
<!-- parity-note -->
## MATLAB Parity Note
- Source MATLAB helpfile: `PPSimExample.mlx`
- Fidelity status: `high_fidelity`
- Remaining justified differences: The notebook now follows the MATLAB recursive-CIF workflow with the native Python `CIF.simulateCIF` path; exact Simulink block timing and solver semantics are still not fixture-matched one-for-one against MATLAB.
"""


PPSIM_CODE = [
    """
    # nSTAT-python notebook example: PPSimExample
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

    from nstat import Analysis, CIF, ConfigColl, CovColl, Covariate, FitResSummary, Trial, TrialConfig
    from nstat.notebook_figures import FigureTracker

    np.random.seed(5)
    OUTPUT_ROOT = REPO_ROOT / "output" / "notebook_images"
    __tracker = FigureTracker(topic='PPSimExample', output_root=OUTPUT_ROOT, expected_count=8)


    def _figure(label: str, *, figsize=(8.5, 4.5)):
        fig = __tracker.new_figure(label)
        fig.clear()
        fig.set_size_inches(*figsize)
        return fig


    Ts = 0.001
    tMin = 0.0
    tMax = 50.0
    t = np.arange(tMin, tMax + Ts, Ts)
    mu = -3.0
    H = np.array([-1.0, -2.0, -4.0], dtype=float)
    S = np.array([1.0], dtype=float)
    E = np.array([0.0], dtype=float)
    stimulus_signal = np.sin(2 * np.pi * 1.0 * t)
    stim = Covariate(t, stimulus_signal, "Stimulus", "time", "s", "Voltage", ["sin"])
    ens = Covariate(t, np.zeros_like(t), "Ensemble", "time", "s", "Spikes", ["n1"])
    baseline = Covariate(t, np.ones_like(t), "Baseline", "time", "s", "", ["mu"])
    sC, lambda_cov = CIF.simulateCIF(mu, H, S, E, stim, ens, 5, "binomial", seed=5, return_lambda=True)
    cc = CovColl([stim, baseline])
    trial = Trial(sC, cc)
    print({"duration_s": tMax, "num_realizations": sC.numSpikeTrains, "mean_rate_hz": round(float(np.mean(lambda_cov.data[:, 0])), 3)})
    """,
    """
    # SECTION 1: General Point Process Simulation
    plt.close("all")
    """,
    """
    # SECTION 2: Point Process Sample Path Generation
    print("Using native Python CIF.simulateCIF to mirror the MATLAB recursive-CIF workflow.")
    """,
    """
    # SECTION 3: History Effect
    selfHist = [0.0, 0.001, 0.002, 0.003]
    print({"history_windows_s": selfHist})
    """,
    """
    # SECTION 4: Stimulus Effect
    print({"stimulus_frequency_hz": 1.0, "stimulus_amplitude": 1.0})
    """,
    """
    # SECTION 5: Ensemble Effect
    print({"ensemble_effect": 0.0})
    """,
    """
    # SECTION 6: Generate sample paths
    fig = _figure("figure; subplot(2,1,1); sC.plot; subplot(2,1,2); stim.plot", figsize=(10.0, 5.5))
    axs = fig.subplots(2, 1, sharex=True)
    sC.plot(handle=axs[0])
    axs[0].set_xlim(0.0, tMax / 5.0)
    stim.plot(handle=axs[1])
    axs[1].set_xlim(0.0, tMax / 5.0)
    """,
    """
    # SECTION 7: Inspect the simulated CIF
    fig = _figure("figure; lambda.plot", figsize=(10.0, 4.0))
    ax = fig.subplots(1, 1)
    lambda_cov.getSubSignal(1).plot(handle=ax)
    ax.set_xlim(0.0, tMax / 5.0)
    """,
    """
    # SECTION 8: GLM Model Fitting Setup
    cfg = [
        TrialConfig([["Baseline", "mu"]], sampleRate=1.0 / Ts, name="Baseline"),
        TrialConfig([["Baseline", "mu"], ["Stimulus", "sin"]], sampleRate=1.0 / Ts, name="Stim"),
        TrialConfig([["Baseline", "mu"], ["Stimulus", "sin"]], sampleRate=1.0 / Ts, history=selfHist, name="Stim+Hist"),
    ]
    cfgColl = ConfigColl(cfg)
    """,
    """
    # SECTION 9: Choose the MATLAB-style fitting algorithm
    Algorithm = "BNLRCG"
    print({"algorithm": Algorithm, "binary_representation": bool(sC.getNST(1).isSigRepBinary())})
    """,
    """
    # SECTION 10: GLM Model Fitting and Results
    results = Analysis.RunAnalysisForAllNeurons(trial, cfgColl)
    """,
    """
    # SECTION 11: Results for sample neuron
    fig = _figure("results{1}.plotResults", figsize=(11.0, 8.0))
    results[0].plotResults(handle=fig)
    """,
    """
    # SECTION 12: Baseline-only diagnostic view
    fig = _figure("results{1}.plotResults baseline", figsize=(11.0, 8.0))
    results[0].plotResults(fit_num=1, handle=fig)
    """,
    """
    # SECTION 13: Stimulus model diagnostic view
    fig = _figure("results{2}.plotResults stim", figsize=(11.0, 8.0))
    results[0].plotResults(fit_num=2, handle=fig)
    """,
    """
    # SECTION 14: Stimulus-plus-history diagnostic view
    fig = _figure("results{3}.plotResults hist", figsize=(11.0, 8.0))
    results[0].plotResults(fit_num=3, handle=fig)
    """,
    """
    # SECTION 15: Compare fitted firing rates
    fig = _figure("results.lambda.plot", figsize=(9.5, 4.5))
    ax = fig.subplots(1, 1)
    results[0].lambdaSignal.getSubSignal(3).plot(handle=ax)
    ax.set_xlim(0.0, tMax / 5.0)
    """,
    """
    # SECTION 16: Results across all sample paths
    summary = FitResSummary(results)
    fig = _figure("Summary.plotSummary", figsize=(10.0, 4.5))
    summary.plotSummary(handle=fig)
    print({"fit_names": summary.fitNames, "mean_AIC": np.asarray(summary.AIC, dtype=float).round(3).tolist()})
    """,
    """
    # SECTION 17: Summarize model selection
    fig = _figure("bar(summary.AIC)", figsize=(8.0, 4.5))
    ax = fig.subplots(1, 1)
    ax.bar(np.arange(len(summary.fitNames)), np.asarray(summary.AIC, dtype=float), color=["0.6", "tab:blue", "tab:green"])
    ax.set_xticks(np.arange(len(summary.fitNames)), summary.fitNames, rotation=20)
    ax.set_ylabel("mean AIC")
    ax.set_title("Model comparison across realizations")
    __tracker.finalize()
    """,
]


def main() -> int:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    _write_notebook(
        NOTEBOOK_DIR / "TrialExamples.ipynb",
        topic="TrialExamples",
        expected_figures=6,
        markdown_note=TRIAL_NOTE,
        code_cells=TRIAL_CODE,
    )
    _write_notebook(
        NOTEBOOK_DIR / "PPSimExample.ipynb",
        topic="PPSimExample",
        expected_figures=8,
        markdown_note=PPSIM_NOTE,
        code_cells=PPSIM_CODE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
