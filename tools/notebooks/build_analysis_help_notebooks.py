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


ANALYSIS_NOTE = """\
<!-- parity-note -->
## MATLAB Parity Note
- Source MATLAB helpfile: `AnalysisExamples.mlx`
- Fidelity status: `high_fidelity`
- Remaining justified differences: The notebook now follows the MATLAB standard-GLM workflow with the canonical `glm_data.mat` dataset and real KS/model-visualization figures; coefficient values and styling still vary modestly because the Python GLM backend and plotting defaults differ from MATLAB.
"""


ANALYSIS_CODE = [
    """
    # nSTAT-python notebook example: AnalysisExamples
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

    from nstat import Analysis, Covariate, nspikeTrain
    from nstat.glm import fit_poisson_glm
    from nstat.notebook_data import load_glm_data_for_notebook
    from nstat.notebook_figures import FigureTracker

    GLM_DATA = load_glm_data_for_notebook()
    OUTPUT_ROOT = REPO_ROOT / "output" / "notebook_images"
    __tracker = FigureTracker(topic="AnalysisExamples", output_root=OUTPUT_ROOT, expected_count=4)


    def _prepare_figure(matlab_line: str, *, figsize=(8.0, 4.5)):
        fig = __tracker.new_figure(matlab_line)
        fig.clear()
        fig.set_size_inches(*figsize)
        return fig


    def _poisson_standard_errors(design_matrix, result):
        x = np.asarray(design_matrix, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        x_aug = np.column_stack([np.ones(x.shape[0]), x])
        beta = np.concatenate([[result.intercept], np.asarray(result.coefficients, dtype=float)])
        lam = np.exp(np.clip(x_aug @ beta, -20.0, 20.0))
        cov = np.linalg.pinv(x_aug.T @ (lam[:, None] * x_aug))
        return np.sqrt(np.clip(np.diag(cov), 0.0, None))


    T = np.asarray(GLM_DATA["T"], dtype=float).reshape(-1)
    xN = np.asarray(GLM_DATA["xN"], dtype=float).reshape(-1)
    yN = np.asarray(GLM_DATA["yN"], dtype=float).reshape(-1)
    spikes_binned = np.asarray(GLM_DATA["spikes_binned"], dtype=float).reshape(-1)
    spiketimes = np.asarray(GLM_DATA["spiketimes"], dtype=float).reshape(-1)
    x_at_spiketimes = np.asarray(GLM_DATA["x_at_spiketimes"], dtype=float).reshape(-1)
    y_at_spiketimes = np.asarray(GLM_DATA["y_at_spiketimes"], dtype=float).reshape(-1)
    sample_rate = 1.0 / float(np.median(np.diff(T)))
    nst = nspikeTrain(spiketimes, name="1", minTime=float(T[0]), maxTime=float(T[-1]), makePlots=-1)
    """,
    """
    # SECTION 1: Analysis Examples
    plt.close("all")
    print({"n_samples": int(T.shape[0]), "n_spikes": int(spiketimes.shape[0]), "sample_rate_hz": round(sample_rate, 3)})
    """,
    """
    # SECTION 2: Example 1: Tradition Preliminary Analysis
    x_linear = np.column_stack([xN, yN])
    x_quadratic_centered = np.column_stack(
        [
            xN,
            yN,
            xN**2 - np.mean(xN**2),
            yN**2 - np.mean(yN**2),
            xN * yN - np.mean(xN * yN),
        ]
    )
    x_quadratic = np.column_stack([xN, yN, xN**2, yN**2, xN * yN])
    linear_fit = fit_poisson_glm(x_linear, spikes_binned)
    quadratic_fit = fit_poisson_glm(x_quadratic, spikes_binned)
    centered_fit = fit_poisson_glm(x_quadratic_centered, spikes_binned)
    """,
    """
    # SECTION 3: visualize the raw data
    fig = _prepare_figure("figure; plot(xN,yN,x_at_spiketimes,y_at_spiketimes,'r.')", figsize=(6.5, 6.0))
    ax = fig.subplots(1, 1)
    ax.plot(xN, yN, color="0.65", linewidth=1.0)
    ax.plot(x_at_spiketimes, y_at_spiketimes, "r.", markersize=3.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_title("Rat trajectory with spike locations")
    """,
    """
    # SECTION 4: fit a GLM model to the x and y positions
    fig = _prepare_figure("figure; errorbar(1:length(b), b, stats.se,'.')", figsize=(7.0, 4.5))
    ax = fig.subplots(1, 1)
    centered_beta = np.concatenate([[centered_fit.intercept], np.asarray(centered_fit.coefficients, dtype=float)])
    centered_se = _poisson_standard_errors(x_quadratic_centered, centered_fit)
    xpos = np.arange(centered_beta.size)
    ax.errorbar(xpos, centered_beta, yerr=centered_se, fmt=".", color="tab:blue", capsize=3)
    ax.set_xticks(xpos, ["baseline", "x", "y", "x^2", "y^2", "x*y"])
    ax.set_ylabel("coefficient value")
    ax.set_title("Quadratic GLM coefficients")
    """,
    """
    # SECTION 5: visualize your model
    fig = _prepare_figure("figure; mesh(x_new,y_new,lambda,'AlphaData',0)", figsize=(8.0, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    grid = np.arange(-1.0, 1.01, 0.1)
    x_new, y_new = np.meshgrid(grid, grid)
    X_grid = np.column_stack([x_new.ravel(), y_new.ravel(), x_new.ravel() ** 2, y_new.ravel() ** 2, x_new.ravel() * y_new.ravel()])
    lam_grid = quadratic_fit.predict_rate(X_grid).reshape(x_new.shape)
    lam_grid = np.where((x_new**2 + y_new**2) <= 1.0, lam_grid, np.nan)
    ax.plot_wireframe(x_new, y_new, lam_grid, rstride=1, cstride=1, color="tab:blue", linewidth=0.7)
    theta = np.linspace(-np.pi, np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta), color="k", linewidth=1.0)
    ax.plot(x_at_spiketimes, y_at_spiketimes, np.zeros_like(x_at_spiketimes), "r.", markersize=2.0)
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_zlabel("lambda")
    ax.set_title("Quadratic GLM spatial intensity")
    """,
    """
    # SECTION 6: Compare a linear model versus a Gaussian GLM model
    lambda_linear_hz = linear_fit.predict_rate(x_linear) * sample_rate
    lambda_quadratic_hz = quadratic_fit.predict_rate(x_quadratic) * sample_rate
    lambda_linear = Covariate(T, lambda_linear_hz, "lambda_linear", "time", "s", "Hz", ["Linear"])
    lambda_quadratic = Covariate(T, lambda_quadratic_hz, "lambda_quadratic", "time", "s", "Hz", ["Quadratic"])
    print(
        {
            "linear_mean_rate_hz": round(float(np.mean(lambda_linear_hz)), 4),
            "quadratic_mean_rate_hz": round(float(np.mean(lambda_quadratic_hz)), 4),
        }
    )
    """,
    """
    # SECTION 7: Make the KS Plot
    _, _, x_linear_ks, ks_linear, _ = Analysis.computeKSStats(nst, lambda_linear)
    _, _, x_quadratic_ks, ks_quadratic, _ = Analysis.computeKSStats(nst, lambda_quadratic)
    fig = _prepare_figure("figure; plot(([1:N]-.5)/N, KSSorted, ...)", figsize=(6.5, 5.0))
    ax = fig.subplots(1, 1)
    x_axis = np.asarray(x_linear_ks, dtype=float).reshape(-1)
    ks_linear_arr = np.asarray(ks_linear, dtype=float).reshape(-1)
    ks_quadratic_arr = np.asarray(ks_quadratic, dtype=float).reshape(-1)
    if x_axis.size:
        ci = 1.36 / np.sqrt(x_axis.size)
        ax.plot(x_axis, ks_linear_arr, color="tab:blue", linewidth=1.5, label="Linear")
        ax.plot(x_axis, ks_quadratic_arr, color="tab:orange", linewidth=1.5, label="Quadratic")
        ax.plot([0.0, 1.0], [0.0, 1.0], "g", linewidth=1.0)
        ax.plot(x_axis, np.clip(x_axis + ci, 0.0, 1.0), "r", linewidth=1.0)
        ax.plot(x_axis, np.clip(x_axis - ci, 0.0, 1.0), "r", linewidth=1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Uniform CDF")
    ax.set_ylabel("Empirical CDF of Rescaled ISIs")
    ax.set_title("KS Plot with 95% Confidence Intervals")
    ax.legend(loc="lower right", frameon=False)
    __tracker.finalize()
    """,
]


ANALYSIS2_NOTE = """\
<!-- parity-note -->
## MATLAB Parity Note
- Source MATLAB helpfile: `AnalysisExamples2.mlx`
- Fidelity status: `high_fidelity`
- Remaining justified differences: The notebook now follows the MATLAB toolbox workflow on the canonical `glm_data.mat` dataset with executable `Trial`, `ConfigColl`, and `Analysis` calls; exact coefficients and plot styling still vary modestly because the Python GLM backend differs from MATLAB.
"""


ANALYSIS2_CODE = [
    """
    # nSTAT-python notebook example: AnalysisExamples2
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
    from nstat.glm import fit_poisson_glm
    from nstat.notebook_data import load_glm_data_for_notebook
    from nstat.notebook_figures import FigureTracker

    GLM_DATA = load_glm_data_for_notebook()
    OUTPUT_ROOT = REPO_ROOT / "output" / "notebook_images"
    __tracker = FigureTracker(topic="AnalysisExamples2", output_root=OUTPUT_ROOT, expected_count=5)


    def _prepare_figure(matlab_line: str, *, figsize=(8.0, 4.5)):
        fig = __tracker.new_figure(matlab_line)
        fig.clear()
        fig.set_size_inches(*figsize)
        return fig


    T = np.asarray(GLM_DATA["T"], dtype=float).reshape(-1)
    xN = np.asarray(GLM_DATA["xN"], dtype=float).reshape(-1)
    yN = np.asarray(GLM_DATA["yN"], dtype=float).reshape(-1)
    vxN = np.asarray(GLM_DATA["vxN"], dtype=float).reshape(-1)
    vyN = np.asarray(GLM_DATA["vyN"], dtype=float).reshape(-1)
    spikes_binned = np.asarray(GLM_DATA["spikes_binned"], dtype=float).reshape(-1)
    spiketimes = np.asarray(GLM_DATA["spiketimes"], dtype=float).reshape(-1)
    sample_rate = 1000.0

    nst = nspikeTrain(spiketimes, name="1", minTime=float(T[0]), maxTime=float(T[-1]), makePlots=-1)
    baseline = Covariate(T, np.ones_like(xN), "Baseline", "time", "s", "", ["mu"])
    position = Covariate(T, np.column_stack([xN, yN]), "Position", "time", "s", "m", ["x", "y"])
    velocity = Covariate(T, np.column_stack([vxN, vyN]), "Velocity", "time", "s", "m/s", ["v_x", "v_y"])
    radial = Covariate(T, np.column_stack([xN, yN, xN**2, yN**2, xN * yN]), "Radial", "time", "s", "m", ["x", "y", "x^2", "y^2", "x*y"])
    values_at_spiketimes = position.getValueAt(spiketimes)
    values_at_spiketimes_upsampled = position.resample(1.0 / np.min(np.diff(spiketimes))).getValueAt(spiketimes)
    """,
    """
    # SECTION 1: Analysis Examples 2
    plt.close("all")
    print({"n_samples": int(T.shape[0]), "n_spikes": int(spiketimes.shape[0]), "analysis_sample_rate_hz": sample_rate})
    """,
    """
    # SECTION 2: load the rat trajectory and spiking data
    print({"position_shape": list(position.data.shape), "velocity_shape": list(velocity.data.shape), "radial_shape": list(radial.data.shape)})
    """,
    """
    # SECTION 3: interpolate the covariates at the spike times
    print(
        {
            "direct_spike_position_head": np.asarray(values_at_spiketimes[:3], dtype=float).round(4).tolist(),
            "upsampled_spike_position_head": np.asarray(values_at_spiketimes_upsampled[:3], dtype=float).round(4).tolist(),
        }
    )
    """,
    """
    # SECTION 4: visualize the raw data
    fig = _prepare_figure("figure; plot(position.getSubSignal('x').dataToMatrix,...)", figsize=(6.5, 6.0))
    ax = fig.subplots(1, 1)
    ax.plot(position.getSubSignal("x").dataToMatrix(), position.getSubSignal("y").dataToMatrix(), color="0.6", linewidth=1.0)
    ax.plot(values_at_spiketimes[:, 0], values_at_spiketimes[:, 1], "r.", markersize=3.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_title("Trajectory and interpolated spike locations")
    """,
    """
    # SECTION 5: Create a trial object and define the fits that we want to run
    spikeColl = nstColl([nst])
    covarColl = CovColl([baseline, radial])
    trial = Trial(spikeColl, covarColl)
    tc = [
        TrialConfig([["Baseline", "mu"], ["Radial", "x", "y"]], sampleRate=sample_rate, history=[], name="Linear"),
        TrialConfig([["Baseline", "mu"], ["Radial", "x", "y", "x^2", "y^2", "x*y"]], sampleRate=sample_rate, history=[], name="Quadratic"),
        TrialConfig([["Baseline", "mu"], ["Radial", "x", "y", "x^2", "y^2", "x*y"]], sampleRate=sample_rate, history=[0.0, 1.0 / sample_rate], name="Quadratic+Hist"),
    ]
    tcc = ConfigColl(tc)
    """,
    """
    # SECTION 6: Create our collection of configurations and run the analysis
    fitResults = Analysis.RunAnalysisForAllNeurons(trial, tcc, 0)
    fig = _prepare_figure("fitResults.plotResults", figsize=(11.0, 8.0))
    fitResults.plotResults(handle=fig)
    print({"config_names": fitResults.configNames, "aic": np.asarray(fitResults.AIC, dtype=float).round(3).tolist()})
    """,
    """
    # SECTION 7: Visualize the firing rates as a function of the spatial covariates
    fig = _prepare_figure("mesh(x_new,y_new,lambda)", figsize=(9.0, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    grid = np.arange(-1.0, 1.01, 0.1)
    x_new, y_new = np.meshgrid(grid, grid)
    newData = [np.ones_like(x_new), x_new, y_new, x_new**2, y_new**2, x_new * y_new]
    for fit_index, color in zip(range(1, fitResults.numResults + 1), Analysis.colors, strict=False):
        lambda_eval = fitResults.evalLambda(fit_index, newData)
        ax.plot_wireframe(x_new, y_new, lambda_eval.reshape(x_new.shape), color=color, linewidth=0.5, alpha=0.8)
    ax.plot(values_at_spiketimes[:, 0], values_at_spiketimes[:, 1], np.zeros(values_at_spiketimes.shape[0]), "r.", markersize=2.0)
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_zlabel("lambda")
    ax.set_title("Toolbox-model spatial intensity comparison")
    """,
    """
    # SECTION 8: Toolbox vs. Standard GLM comparison
    standard_fit = fit_poisson_glm(np.column_stack([np.ones_like(xN), xN, yN, xN**2, yN**2, xN * yN]), spikes_binned, include_intercept=False)
    coeff_diff = np.asarray(standard_fit.coefficients - fitResults.getCoeffs(2), dtype=float)
    fig = _prepare_figure("b-fitResults.b{2}", figsize=(7.0, 4.5))
    ax = fig.subplots(1, 1)
    labels = ["mu", "x", "y", "x^2", "y^2", "x*y"]
    ax.bar(np.arange(coeff_diff.size), coeff_diff, color="tab:blue")
    ax.axhline(0.0, color="0.3", linestyle="--", linewidth=1.0)
    ax.set_xticks(np.arange(coeff_diff.size), labels, rotation=20)
    ax.set_ylabel("standard minus toolbox")
    ax.set_title("Coefficient agreement between workflows")
    print({"quadratic_coeff_diff_max_abs": round(float(np.max(np.abs(coeff_diff))), 6)})
    """,
    """
    # SECTION 9: Compute the history effect
    windowTimes = np.arange(0.0, 11.0) / sample_rate
    covLabels = [["Baseline", "mu"], ["Radial", "x", "y", "x^2", "y^2", "x*y"]]
    histResults, histConfigs = Analysis.computeHistLag(trial, 1, windowTimes, covLabels, "GLM", 0, sample_rate, 0)
    histSummary = FitResSummary([histResults])
    fig = _prepare_figure("Analysis.computeHistLag(...)", figsize=(8.5, 4.5))
    ax = fig.subplots(1, 1)
    ax.plot(np.arange(histResults.numResults), np.asarray(histResults.AIC, dtype=float), marker="o", color="tab:green", linewidth=1.2)
    ax.set_xticks(np.arange(histResults.numResults), histResults.configNames, rotation=20)
    ax.set_ylabel("AIC")
    ax.set_title("History-lag model comparison")
    print({"history_config_names": histConfigs.getConfigNames(), "summary_fit_names": histSummary.fitNames})
    __tracker.finalize()
    """,
]


def main() -> int:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    _write_notebook(
        NOTEBOOK_DIR / "AnalysisExamples.ipynb",
        topic="AnalysisExamples",
        expected_figures=4,
        markdown_note=ANALYSIS_NOTE,
        code_cells=ANALYSIS_CODE,
    )
    _write_notebook(
        NOTEBOOK_DIR / "AnalysisExamples2.ipynb",
        topic="AnalysisExamples2",
        expected_figures=5,
        markdown_note=ANALYSIS2_NOTE,
        code_cells=ANALYSIS2_CODE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
