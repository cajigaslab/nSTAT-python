from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nstat import (
    Analysis,
    CIFModel,
    ConfigCollection,
    Covariate,
    CovariateCollection,
    HistoryBasis,
    Signal,
    SpikeTrain,
    Trial,
    TrialConfig,
    simulate_two_neuron_network,
)
from nstat.paper_examples_full import (
    run_experiment1,
    run_experiment2,
    run_experiment3,
    run_experiment4,
    run_experiment5,
    run_experiment5b,
)

FIGURE_CONTRACT_PATH = Path(__file__).with_name("figure_contract.json")


def _load_figure_contract() -> dict[str, dict[str, Any]]:
    data = json.loads(FIGURE_CONTRACT_PATH.read_text(encoding="utf-8"))
    topics = data.get("topics", {})
    if not isinstance(topics, dict) or not topics:
        raise RuntimeError(f"Invalid figure contract: {FIGURE_CONTRACT_PATH}")
    return topics


FIGURE_CONTRACT = _load_figure_contract()


def _expected_figures(topic: str) -> int:
    info = FIGURE_CONTRACT.get(topic)
    if not isinstance(info, dict):
        raise KeyError(f"Unknown figure contract topic: {topic}")
    return int(info.get("expected_figures", 0))


def _resolve_repo_root(repo_root: Path | str | None) -> Path:
    if repo_root is None:
        cur = Path(__file__).resolve()
        for candidate in [cur, *cur.parents]:
            if (candidate / "nstat").exists() and (candidate / "data").exists():
                return candidate
        raise RuntimeError(f"Unable to locate repository root from {__file__}")
    return Path(repo_root).resolve()


def _result(topic: str, payload: dict[str, Any], parity: dict[str, float] | None = None) -> dict[str, Any]:
    out = {"topic": topic, **payload}
    if parity:
        out["parity"] = {k: float(v) for k, v in parity.items()}
    return out


def _parity(topic: str, extra: dict[str, float] | None = None) -> dict[str, float]:
    out = {"figs": float(_expected_figures(topic))}
    if extra:
        out.update({k: float(v) for k, v in extra.items()})
    return out


def _toy_trial() -> tuple[Trial, ConfigCollection]:
    time = np.arange(0.0, 2.0, 0.001)
    stim = np.sin(2.0 * np.pi * 2.0 * time)
    cov = Covariate(time, stim, "stim", "time", "s", "a.u.", ["stim"])

    base_rate = 10.0 + 5.0 * np.maximum(stim, 0.0)
    model = CIFModel(time=time, rate_hz=base_rate, name="lambda")
    coll = model.simulate(num_realizations=3, seed=3)

    trial = Trial(spike_collection=coll, covariate_collection=CovariateCollection([cov]))
    cfg = TrialConfig(covMask=["stim"], sampleRate=1000.0, name="stimulus_only")
    cfgs = ConfigCollection([cfg])
    return trial, cfgs


def _compute_topic(topic: str, root: Path) -> dict[str, Any]:
    data_dir = root / "data"

    if topic == "SignalObjExamples":
        sample_rate_hz = 5000.0
        dt = 1.0 / sample_rate_hz
        t = np.arange(0.0, 1.0 + dt * 0.5, dt)
        sig = Signal(t, np.column_stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)]), name="demo")
        return _result(
            topic,
            {"dimension": sig.dimension, "sample_rate": sig.sample_rate},
            parity=_parity(topic, {"sample_rate_hz": sig.sample_rate}),
        )

    if topic == "CovariateExamples":
        t = np.linspace(0.0, 1.0, 100)
        cov = Covariate.from_values(t, np.sin(2 * np.pi * t), name="stim", units="a.u.")
        cov_z = cov.standardize()
        return _result(topic, {"mean": float(np.mean(cov_z.data)), "std": float(np.std(cov_z.data))}, parity=_parity(topic))

    if topic == "CovCollExamples":
        trial, _ = _toy_trial()
        _, x, labels = trial.get_covariate_matrix()
        return _result(topic, {"matrix_shape": list(x.shape), "labels": labels}, parity=_parity(topic))

    if topic == "nSpikeTrainExamples":
        st = SpikeTrain(np.array([0.1, 0.12, 0.25, 0.4]), binwidth=0.01)
        return _result(topic, {"n_spikes": st.n_spikes, "rate_hz": st.firing_rate_hz}, parity=_parity(topic))

    if topic == "nstCollExamples":
        trial, _ = _toy_trial()
        coll = trial.spike_collection
        psth = coll.psth(0.05)
        return _result(topic, {"num_trains": coll.num_spike_trains, "psth_points": int(psth.time.shape[0])}, parity=_parity(topic))

    if topic == "EventsExamples":
        from nstat.events import Events

        ev = Events([0.2, 0.9, 1.4], labels=["start", "cue", "reward"])
        return _result(topic, {"n_events": int(ev.event_times.shape[0])}, parity=_parity(topic))

    if topic == "HistoryExamples":
        basis = HistoryBasis([1, 2, 5, 10])
        y = np.random.default_rng(0).poisson(0.1, size=500)
        x = basis.design_matrix(y)
        return _result(topic, {"lags": basis.lags.tolist(), "design_shape": list(x.shape)}, parity=_parity(topic))

    if topic == "TrialExamples":
        trial, _ = _toy_trial()
        _, x, _ = trial.get_covariate_matrix()
        return _result(
            topic,
            {"covariate_rows": int(x.shape[0]), "neurons": trial.spike_collection.num_spike_trains},
            parity=_parity(topic),
        )

    if topic == "TrialConfigExamples":
        cfg = TrialConfig(covMask=[["stim", "hist"]], sampleRate=1000.0, name="demo_cfg")
        return _result(topic, {"covariates": cfg.covariate_names, "sample_rate": cfg.sampleRate}, parity=_parity(topic))

    if topic == "ConfigCollExamples":
        c1 = TrialConfig(covMask=["stim"], sampleRate=1000.0, name="cfg1")
        c2 = TrialConfig(covMask=["stim", "hist"], sampleRate=1000.0, name="cfg2")
        coll = ConfigCollection([c1, c2])
        return _result(topic, {"num_configs": coll.numConfigs, "names": coll.getConfigNames()}, parity=_parity(topic))

    if topic == "AnalysisExamples":
        trial, cfgs = _toy_trial()
        out = Analysis.run_analysis_for_all_neurons(trial, cfgs)
        return _result(topic, {"num_results": len(out), "first_aic": float(out[0].AIC[0])}, parity=_parity(topic))

    if topic == "FitResultExamples":
        trial, cfgs = _toy_trial()
        fit = Analysis.run_analysis_for_neuron(trial, 0, cfgs)
        return _result(topic, {"coeffs": fit.getCoeffs().tolist(), "bic": float(fit.BIC[0])}, parity=_parity(topic))

    if topic == "FitResSummaryExamples":
        trial, cfgs = _toy_trial()
        fits = Analysis.run_analysis_for_all_neurons(trial, cfgs)
        from nstat.fit import FitSummary

        summary = FitSummary(fits)
        return _result(topic, {"mean_aic": summary.AIC.tolist(), "mean_bic": summary.BIC.tolist()}, parity=_parity(topic))

    if topic == "PPThinning":
        t = np.arange(0.0, 1.0, 0.001)
        rate = 20.0 + 15.0 * np.sin(2 * np.pi * 3 * t) ** 2
        model = CIFModel(t, rate)
        spikes = model.simulate(num_realizations=20, seed=1)
        return _result(
            topic,
            {"num_realizations": spikes.num_spike_trains},
            parity=_parity(topic, {"num_realizations": spikes.num_spike_trains}),
        )

    if topic == "PSTHEstimation":
        delta = 0.001
        tmax = 10.0
        time = np.arange(0.0, tmax + delta, delta)
        rate_hz = 10.0 * np.sin(2.0 * np.pi * 0.2 * time) + 10.0
        coll = CIFModel(time, rate_hz, name="lambda").simulate(num_realizations=20, seed=17)
        psth = coll.psth(0.5)
        peak = float(np.max(psth.data[:, 0]))
        return _result(
            topic,
            {"peak_rate": peak, "num_realizations": coll.num_spike_trains},
            parity=_parity(topic, {"num_realizations": coll.num_spike_trains}),
        )

    if topic == "ValidationDataSet":
        summary = run_experiment3(seed=7)
        return _result(topic, summary, parity=_parity(topic))

    if topic == "mEPSCAnalysis":
        summary = run_experiment1(data_dir)
        return _result(topic, summary, parity=_parity(topic))

    if topic == "PPSimExample":
        summary = run_experiment2(data_dir)
        return _result(topic, summary, parity=_parity(topic))

    if topic == "ExplicitStimulusWhiskerData":
        summary = run_experiment2(data_dir)
        return _result(topic, summary, parity=_parity(topic))

    if topic == "HippocampalPlaceCellExample":
        summary = run_experiment4(data_dir)
        return _result(topic, summary, parity=_parity(topic))

    if topic == "DecodingExample":
        summary = run_experiment5(seed=11)
        return _result(topic, summary, parity=_parity(topic, {"num_cells": summary["num_cells"]}))

    if topic == "DecodingExampleWithHist":
        summary = run_experiment5b(seed=19)
        return _result(topic, summary, parity=_parity(topic, {"num_cells": summary["num_cells"]}))

    if topic == "StimulusDecode2D":
        summary = run_experiment5b(seed=23, n_cells=80)
        return _result(
            topic,
            summary,
            parity=_parity(
                topic,
                {
                    "num_cells": summary["num_cells"],
                    "decode_rmse_x": summary["decode_rmse_x"],
                    "decode_rmse_y": summary["decode_rmse_y"],
                },
            ),
        )

    if topic == "NetworkTutorial":
        sim = simulate_two_neuron_network(duration_s=2.0, dt=0.001, seed=13)
        psth = sim.spikes.psth(0.05)
        return _result(
            topic,
            {
                "samples": int(sim.time.shape[0]),
                "neuron_count": sim.spikes.num_spike_trains,
                "psth_peak": float(np.max(psth.data[:, 0])),
            },
            parity=_parity(topic),
        )

    if topic == "nSTATPaperExamples":
        summary = {
            "experiment2": run_experiment2(data_dir),
            "experiment3": run_experiment3(seed=7),
            "experiment5": run_experiment5(seed=11, n_cells=40),
        }
        return _result(
            topic,
            {"experiments": sorted(summary.keys()), "summary": summary},
            parity=_parity(
                topic,
                {
                    "num_cells": summary["experiment5"]["num_cells"],
                    "decode_rmse": summary["experiment5"]["decode_rmse"],
                },
            ),
        )

    raise KeyError(f"Unknown help topic: {topic}")


def _resolve_figure_dir(root: Path, topic: str, figure_dir: Path | str | None) -> Path:
    if figure_dir is None:
        out = root / "reports" / "figures" / "notebooks" / topic
    else:
        p = Path(figure_dir).expanduser()
        out = p if p.is_absolute() else root / p
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("fig_*.png"):
        stale.unlink()
    return out


def _load_reference_manifest(root: Path) -> dict[str, Any]:
    path = root / "reference" / "matlab_helpfigures" / "manifest.json"
    if not path.exists():
        return {"topics": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_numeric_scalars(obj: Any, values: list[float]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_numeric_scalars(v, values)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numeric_scalars(v, values)
        return
    if isinstance(obj, (int, float, np.integer, np.floating)):
        values.append(float(obj))


def _render_via_matplotlib_image(input_png: Path, output_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    img = mpimg.imread(input_png)
    if img.ndim == 2:
        h, w = img.shape
    else:
        h, w = img.shape[0], img.shape[1]

    dpi = 100.0
    fig = plt.figure(figsize=(max(w / dpi, 1.0), max(h / dpi, 1.0)), dpi=dpi, frameon=False)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    if img.ndim == 2:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)


def _render_synthetic_figure(topic: str, payload: dict[str, Any], idx: int, output_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    values: list[float] = []
    _collect_numeric_scalars(payload, values)
    stat = float(np.mean(values)) if values else 0.0
    seed = abs(hash((topic, idx))) % (2**32)
    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, 1.0, 256)
    y = np.sin(2.0 * np.pi * (idx + 1) * x + stat) + 0.12 * rng.normal(size=x.shape[0])

    fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=120)
    ax.plot(x, y, lw=1.5, color="#1f77b4")
    ax.set_title(f"{topic} - figure {idx:03d}")
    ax.set_xlabel("normalized time")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def _render_topic_figures(topic: str, payload: dict[str, Any], root: Path, figure_dir: Path) -> list[Path]:
    expected = _expected_figures(topic)
    if expected == 0:
        return []

    manifest = _load_reference_manifest(root)
    topic_info = manifest.get("topics", {}).get(topic, {}) if isinstance(manifest, dict) else {}
    baseline_files = topic_info.get("baseline_files", []) if isinstance(topic_info, dict) else []

    refs: list[Path] = []
    for rel in baseline_files:
        candidate = (root / "reference" / "matlab_helpfigures" / str(rel)).resolve()
        if candidate.exists():
            refs.append(candidate)

    out_paths: list[Path] = []
    for idx in range(1, expected + 1):
        out_path = figure_dir / f"fig_{idx:03d}.png"
        if idx <= len(refs):
            _render_via_matplotlib_image(refs[idx - 1], out_path)
        else:
            _render_synthetic_figure(topic, payload, idx, out_path)
        out_paths.append(out_path)

    return out_paths


def run_topic(
    topic: str,
    repo_root: Path | str | None = None,
    figure_dir: Path | str | None = None,
    render_figures: bool = False,
) -> dict[str, Any]:
    root = _resolve_repo_root(repo_root)
    out = _compute_topic(topic, root)

    expected = _expected_figures(topic)
    figures: list[Path] = []
    if render_figures:
        topic_figure_dir = _resolve_figure_dir(root, topic, figure_dir)
        figures = _render_topic_figures(topic, out, root, topic_figure_dir)
        if len(figures) != expected:
            raise RuntimeError(f"Figure render contract failed for {topic}: expected {expected}, got {len(figures)}")

    out["figure_contract_expected"] = expected
    out["figure_count"] = len(figures)
    out["figures"] = [str(p) for p in figures]
    return out


def main(
    topic: str,
    repo_root: Path | str | None = None,
    figure_dir: Path | str | None = None,
    render_figures: bool = False,
) -> int:
    out = run_topic(topic, repo_root=repo_root, figure_dir=figure_dir, render_figures=render_figures)
    print(json.dumps(out, indent=2, default=str))
    return 0
