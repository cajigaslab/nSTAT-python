#!/usr/bin/env python3
"""Probe mapped MATLAB-compatible methods and record executable coverage."""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from nstat.compat import matlab as M


MISSING = object()


@dataclass(slots=True)
class ProbeResult:
    attempted: bool
    success: bool
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--method-mapping", type=Path, default=Path("parity/method_mapping.yaml"))
    parser.add_argument("--matlab-inventory", type=Path, default=Path("parity/matlab_api_inventory.json"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("parity/method_probe_report.json"),
        help="Output method probe report JSON path.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _build_probe_context() -> dict[str, tuple[Any, dict[str, Any]]]:
    # Shared deterministic fixtures used for lightweight method probing.
    time5 = np.linspace(0.0, 1.0, 5)
    time11 = np.linspace(0.0, 1.0, 11)
    time200 = np.linspace(0.0, 2.0, 200)

    signal = M.SignalObj(time=np.linspace(0.0, 2.0, 5), data=np.array([1.0, 2.0, 3.0, 2.0, 1.0]), name="sig")
    covariate = M.Covariate(
        time=time5,
        data=np.column_stack([time5, time5**2]),
        name="stim",
        labels=["stim1", "stim2"],
    )
    ci = M.ConfidenceInterval(
        time=time5,
        lower=np.array([0.0, 0.4, 0.8, 1.2, 1.6]),
        upper=np.array([0.5, 0.9, 1.3, 1.7, 2.1]),
        level=0.95,
    )
    events = M.Events(times=np.array([0.1, 0.4, 0.9]), labels=["a", "b", "c"])
    history = M.History(bin_edges_s=np.array([0.0, 0.05, 0.1, 0.2]))

    st1 = M.nspikeTrain(spike_times=np.array([0.1, 0.2, 0.25, 0.9]), t_start=0.0, t_end=1.0, name="u1")
    st2 = M.nspikeTrain(spike_times=np.array([0.15, 0.4, 0.8]), t_start=0.0, t_end=1.0, name="u2")
    coll = M.nstColl([st1, st2])

    cov1 = M.Covariate(time=time11, data=np.sin(2 * np.pi * time11), name="sine", labels=["sine"])
    cov2 = M.Covariate(time=time11, data=np.column_stack([time11, time11**2]), name="poly", labels=["t", "t2"])
    covcoll = M.CovColl([cov1, cov2])

    trial_cfg = M.TrialConfig(covariate_labels=["sine", "t", "t2"], sample_rate_hz=10.0, fit_type="poisson", name="cfg")
    cfg_coll = M.ConfigColl([trial_cfg])
    trial = M.Trial(spikes=coll, covariates=covcoll)

    X = np.column_stack([np.sin(2.0 * np.pi * 0.7 * time200), np.cos(2.0 * np.pi * 0.4 * time200)])
    y = np.random.default_rng(2026).poisson(np.exp(-1.0 + 0.3 * X[:, 0] - 0.2 * X[:, 1])).astype(float)
    fit_native = M.Analysis.GLMFit(X, y, "poisson", 1.0)
    fit = M.FitResult.fromStructure(fit_native.to_structure())

    fit2 = M.FitResult(
        coefficients=np.array([0.1, 0.3]),
        intercept=-0.5,
        fit_type="poisson",
        log_likelihood=-20.0,
        n_samples=100,
        n_parameters=3,
        parameter_labels=["stim", "hist"],
    )
    fit_summary = M.FitResSummary([fit, fit2])

    cif = M.CIF(coefficients=np.array([0.8, -0.2]), intercept=-1.0, link="binomial")

    spike_matrix = np.array(
        [
            [0, 0, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1],
        ],
        dtype=float,
    )
    spike_counts = np.array(
        [
            [0, 0, 1, 2, 1, 0],
            [1, 2, 2, 1, 0, 0],
            [0, 0, 0, 1, 2, 2],
        ],
        dtype=float,
    )
    tuning = np.array(
        [
            [1.0, 0.5, 0.1],
            [0.2, 1.1, 0.4],
            [0.1, 0.4, 1.2],
        ],
        dtype=float,
    )
    posterior = M.DecodingAlgorithms.decodeStatePosterior(spike_counts, tuning)[1]

    contexts: dict[str, tuple[Any, dict[str, Any]]] = {
        "SignalObj": (signal, {"structure": signal.dataToStructure()}),
        "Covariate": (covariate, {"structure": covariate.toStructure()}),
        "ConfidenceInterval": (ci, {"structure": ci.toStructure()}),
        "Events": (events, {"structure": events.toStructure()}),
        "History": (history, {"structure": history.toStructure()}),
        "nspikeTrain": (st1, {"structure": st1.toStructure()}),
        "nstColl": (coll, {"structure": coll.toStructure()}),
        "CovColl": (covcoll, {"structure": covcoll.toStructure()}),
        "TrialConfig": (trial_cfg, {"structure": trial_cfg.toStructure(), "trial": trial}),
        "ConfigColl": (cfg_coll, {"structure": cfg_coll.toStructure()}),
        "Trial": (trial, {"structure": trial.toStructure()}),
        "CIF": (cif, {"structure": cif.toStructure(), "X": X, "y": y, "time": time200}),
        "Analysis": (
            M.Analysis,
            {
                "X": X,
                "y": y,
                "trial": trial,
                "config": trial_cfg,
                "fit": fit_native,
                "signals": np.column_stack([np.sin(2 * np.pi * time200), np.cos(2 * np.pi * time200)]),
                "positions": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            },
        ),
        "FitResult": (fit, {"structure": fit.toStructure(), "X": X, "newFitObj": fit2}),
        "FitResSummary": (fit_summary, {"structure": fit_summary.toStructure()}),
        "DecodingAlgorithms": (
            M.DecodingAlgorithms,
            {
                "spike_matrix": spike_matrix,
                "spike_counts": spike_counts,
                "tuning_rates": tuning,
                "tuning_curves": tuning,
                "posterior": posterior,
                "state_values": np.arange(tuning.shape[1]),
                "transition": np.eye(tuning.shape[1], dtype=float),
                "sample": np.random.default_rng(7).uniform(size=50),
            },
        ),
    }
    return contexts


def _infer_arg(name: str, context: dict[str, Any]) -> Any:
    lname = name.lower()
    if name in context:
        return context[name]

    if lname in {"x"}:
        return context.get("X", MISSING)
    if lname in {"y"}:
        return context.get("y", MISSING)
    if "fit" in lname and lname not in {"fittype", "fittype_s"}:
        return context.get("fit", MISSING)
    if "trial" in lname:
        return context.get("trial", MISSING)
    if "config" in lname:
        return context.get("config", MISSING)
    if "structure" in lname or "payload" in lname:
        return context.get("structure", MISSING)
    if "newfitobj" in lname:
        return context.get("newFitObj", MISSING)
    if "sample" == lname:
        return context.get("sample", MISSING)
    if "reference" in lname:
        return context.get("sample", MISSING)
    if lname in {"time", "timegrid_s"}:
        return context.get("time", MISSING)
    if "spike" in lname and "matrix" in lname:
        return context.get("spike_matrix", MISSING)
    if "spike" in lname and "count" in lname:
        return context.get("spike_counts", MISSING)
    if "spike" in lname and "times" in lname:
        return context.get("spike_times", np.array([0.1, 0.2, 0.8]))
    if "tuning" in lname and "curve" in lname:
        return context.get("tuning_curves", MISSING)
    if "tuning" in lname and "rate" in lname:
        return context.get("tuning_rates", MISSING)
    if "posterior" in lname:
        return context.get("posterior", MISSING)
    if "state" in lname and "value" in lname:
        return context.get("state_values", MISSING)
    if "transition" in lname:
        return context.get("transition", MISSING)
    if "position" in lname:
        return context.get("positions", MISSING)
    if "signal" in lname:
        return context.get("signals", MISSING)

    if "binsize" in lname or ("bin" in lname and "size" in lname):
        return 0.1
    if "sample_rate" in lname or "samplerate" in lname:
        return 10.0
    if "alpha" in lname:
        return 0.05
    if "dt" == lname:
        return 1.0
    if "l2" in lname:
        return 0.0
    if "kappa" in lname:
        return 0.0
    if lname in {"k", "maxlag"}:
        return 1
    if "fitnum" in lname or ("num" in lname and "fit" in lname):
        return 1
    if "mode" == lname:
        return "count"
    if "metric" in lname:
        return "aic"
    if "fittype" in lname:
        return "poisson"
    if "name" in lname:
        return "u1"
    if "label" in lname:
        return "stim"
    if "labels" in lname:
        return ["stim"]
    if "selector" in lname or lname in {"ind", "index", "unitindex"}:
        return 0
    if "selectors" in lname:
        return [1]
    if "subfits" in lname:
        return [1]
    if "subfit" in lname:
        return 1
    if "partition" in lname:
        return [0.0, 0.5, 1.0]
    if "window" in lname:
        return (-0.05, 0.05)
    if "t_min" in lname or "mintime" in lname:
        return 0.0
    if "t_max" in lname or "maxtime" in lname:
        return 1.0
    if "newzero" in lname or "targettime" in lname or "shift" in lname:
        return 0.1
    if "mer" in lname:
        return 0.001
    if "delimiter" in lname:
        return ","
    if "level" in lname:
        return 0.95
    if "color" in lname:
        return "r"
    if "value" in lname:
        return 1.0
    return MISSING


def _invoke_member(obj: Any, member_name: str, context: dict[str, Any]) -> ProbeResult:
    if not hasattr(obj, member_name):
        return ProbeResult(attempted=False, success=False, error="missing_member")

    member = getattr(obj, member_name)
    if not callable(member):
        return ProbeResult(attempted=True, success=True)

    try:
        sig = inspect.signature(member)
    except (TypeError, ValueError):
        # Builtins/C-extension callables: best effort call with no args.
        try:
            member()
            return ProbeResult(attempted=True, success=True)
        except Exception as exc:  # noqa: BLE001
            return ProbeResult(attempted=True, success=False, error=type(exc).__name__)

    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for param in sig.parameters.values():
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        inferred = _infer_arg(param.name, context)
        if inferred is MISSING:
            return ProbeResult(attempted=False, success=False, error=f"unresolved_arg:{param.name}")
        if param.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}:
            args.append(inferred)
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs[param.name] = inferred

    try:
        member(*args, **kwargs)
        return ProbeResult(attempted=True, success=True)
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(attempted=True, success=False, error=type(exc).__name__)
    finally:
        plt.close("all")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    mapping = _load_yaml((repo_root / args.method_mapping).resolve())
    matlab_inventory = _load_json((repo_root / args.matlab_inventory).resolve())
    methods_by_class = {str(row["matlab_class"]): list(row["methods"]) for row in matlab_inventory["classes"]}
    contexts = _build_probe_context()

    class_rows: list[dict[str, Any]] = []
    total_methods = 0
    total_attempted = 0
    total_success = 0

    for row in mapping["classes"]:
        matlab_class = str(row["matlab_class"])
        alias = dict(row.get("alias_methods", {}))
        target_obj, context = contexts[matlab_class]

        methods = methods_by_class.get(matlab_class, [])
        success_methods: list[str] = []
        attempted_methods: list[str] = []
        failed_methods: list[dict[str, str]] = []
        skipped_methods: list[dict[str, str]] = []

        for matlab_method in methods:
            total_methods += 1
            mapped = str(alias.get(matlab_method, matlab_method))
            probe = _invoke_member(target_obj, mapped, context)
            if not probe.attempted:
                skipped_methods.append({"mapped_member": mapped, "reason": probe.error})
                continue

            total_attempted += 1
            attempted_methods.append(mapped)
            if probe.success:
                total_success += 1
                success_methods.append(mapped)
            else:
                failed_methods.append({"mapped_member": mapped, "error": probe.error})

        class_rows.append(
            {
                "matlab_class": matlab_class,
                "matlab_method_count": len(methods),
                "attempted_method_count": len(attempted_methods),
                "successful_method_count": len(success_methods),
                "success_ratio_attempted": float(len(success_methods) / max(len(attempted_methods), 1)),
                "success_ratio_total": float(len(success_methods) / max(len(methods), 1)),
                "success_methods": sorted(set(success_methods)),
                "failed_methods": failed_methods[:200],
                "skipped_methods": skipped_methods[:200],
            }
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "summary": {
            "total_methods": total_methods,
            "attempted_methods": total_attempted,
            "successful_methods": total_success,
            "attempt_ratio": float(total_attempted / max(total_methods, 1)),
            "success_ratio_total": float(total_success / max(total_methods, 1)),
            "success_ratio_attempted": float(total_success / max(total_attempted, 1)),
        },
        "class_rows": class_rows,
    }

    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote method probe report to {out_path}")
    print(
        "Method probe summary: "
        f"total={total_methods}, attempted={total_attempted}, successful={total_success}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
