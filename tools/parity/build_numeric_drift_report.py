#!/usr/bin/env python3
"""Build numeric-drift parity report against MATLAB gold fixtures."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.io
import yaml

from nstat.analysis import Analysis
from nstat.decoding import DecodingAlgorithms
from nstat.events import Events
from nstat.signal import Covariate
from nstat.spikes import SpikeTrain, SpikeTrainCollection
from nstat.trial import CovariateCollection, Trial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures-manifest",
        type=Path,
        default=Path("tests/parity/fixtures/matlab_gold/manifest.yml"),
        help="Fixture manifest YAML path.",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("parity/numeric_drift_thresholds.yml"),
        help="Numeric drift threshold YAML path.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("parity/numeric_drift_report.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--equivalence-report",
        type=Path,
        default=Path("parity/function_example_alignment_report.json"),
        help="Equivalence audit JSON used to derive notebook-wide checkpoint metrics.",
    )
    parser.add_argument(
        "--notebook-manifest",
        type=Path,
        default=Path("tools/notebooks/notebook_manifest.yml"),
        help="Notebook manifest listing required topic coverage.",
    )
    parser.add_argument(
        "--fail-on-violation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return non-zero when any metric exceeds threshold.",
    )
    return parser.parse_args()


def _mat(path: Path) -> dict:
    return scipy.io.loadmat(path)


def _vec(m: dict, key: str) -> np.ndarray:
    return np.asarray(m[key], dtype=float).reshape(-1)


def _scalar(m: dict, key: str) -> float:
    return float(_vec(m, key)[0])


def _detect_mepsc_events(trace: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    threshold = -0.12
    refractory = int(round(0.006 / dt))
    candidate = np.where(trace < threshold)[0]
    detected_idx: list[int] = []
    last = -refractory
    for idx in candidate:
        if idx - last >= refractory:
            window_end = min(idx + int(round(0.004 / dt)) + 1, trace.size)
            local = idx + int(np.argmin(trace[idx:window_end]))
            detected_idx.append(local)
            last = local
    det = np.asarray(detected_idx, dtype=int)
    return det * dt, -trace[det]


def _fixture_manifest_index(fixtures_manifest: Path) -> dict[str, dict[str, Path]]:
    payload = yaml.safe_load(fixtures_manifest.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Path]] = {}
    for row in payload.get("fixtures", []):
        topic = str(row.get("topic", row.get("name", ""))).strip()
        if not topic:
            continue
        path = Path(row["path"])
        fixture_type = str(row.get("fixture_type", "")).strip()
        if not fixture_type:
            if path.suffix.lower() == ".json":
                fixture_type = "topic_audit"
            elif path.suffix.lower() == ".mat":
                fixture_type = "numeric"
            else:
                fixture_type = "unknown"
        by_type = out.setdefault(topic, {})
        by_type[fixture_type] = path
    return out


def _load_required_topics(notebook_manifest: Path) -> list[str]:
    payload = yaml.safe_load(notebook_manifest.read_text(encoding="utf-8")) or {}
    return [str(row.get("topic", "")).strip() for row in payload.get("notebooks", []) if str(row.get("topic", "")).strip()]


def _load_equivalence_rows(equivalence_report: Path) -> dict[str, dict]:
    if not equivalence_report.exists():
        return {}
    payload = json.loads(equivalence_report.read_text(encoding="utf-8"))
    rows = payload.get("example_line_alignment_audit", {}).get("topic_rows", [])
    out: dict[str, dict] = {}
    for row in rows:
        topic = str(row.get("topic", "")).strip()
        if topic:
            out[topic] = row
    return out


def _ratio(value: float, threshold: float) -> float:
    if threshold == 0.0:
        return 0.0 if value == 0.0 else float("inf")
    return value / threshold


def _numeric_fixture_paths(fixture_index: dict[str, dict[str, Path]]) -> dict[str, Path]:
    required = [
        "PPSimExample",
        "DecodingExampleWithHist",
        "HippocampalPlaceCellExample",
        "SpikeRateDiffCIs",
        "PSTHEstimation",
        "nstCollExamples",
        "CovCollExamples",
        "TrialExamples",
        "EventsExamples",
        "AnalysisExamples",
        "DecodingExample",
        "ExplicitStimulusWhiskerData",
        "mEPSCAnalysis",
    ]
    out: dict[str, Path] = {}
    for topic in required:
        row = fixture_index.get(topic, {})
        numeric_path = row.get("numeric")
        if numeric_path is None:
            continue
        out[f"{topic}_gold.mat"] = Path(numeric_path)
    return out


def _topic_audit_fixtures(fixture_index: dict[str, dict[str, Path]]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for topic, row in fixture_index.items():
        fixture_path = row.get("topic_audit")
        if fixture_path is None:
            continue
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        out[topic] = payload
    return out


def _evaluate_topic_audit_metrics(
    audit_fixtures: dict[str, dict],
    equivalence_rows: dict[str, dict],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for topic, expected in sorted(audit_fixtures.items()):
        observed = equivalence_rows.get(topic, {})
        if not observed:
            out[topic] = {
                "topic_row_missing_error": 1.0,
            }
            continue

        expected_status = str(expected.get("alignment_status", ""))
        observed_status = str(observed.get("alignment_status", ""))
        expected_matlab_lines = int(expected.get("matlab_code_lines", 0))
        observed_matlab_lines = int(observed.get("matlab_code_lines", 0))
        expected_matlab_refs = int(expected.get("matlab_reference_image_count", 0))
        observed_matlab_refs = int(observed.get("matlab_reference_image_count", 0))
        min_assertions = int(expected.get("min_assertion_count", 0))
        observed_assertions = int(observed.get("assertion_count", 0))
        require_checkpoint = bool(expected.get("require_topic_checkpoint", False))
        observed_checkpoint = bool(observed.get("has_topic_checkpoint", False))
        min_py_images = int(expected.get("min_python_validation_image_count", 0))
        observed_py_images = int(observed.get("python_validation_image_count", 0))
        require_plot = bool(expected.get("require_plot_call", False))
        observed_plot = bool(observed.get("has_plot_call", False))

        out[topic] = {
            "topic_row_missing_error": 0.0,
            "alignment_status_mismatch": 0.0 if observed_status == expected_status else 1.0,
            "matlab_code_lines_abs_error": float(abs(observed_matlab_lines - expected_matlab_lines)),
            "matlab_reference_image_count_abs_error": float(abs(observed_matlab_refs - expected_matlab_refs)),
            "assertion_count_missing_error": 0.0 if observed_assertions >= min_assertions else 1.0,
            "topic_checkpoint_missing_error": 0.0
            if (not require_checkpoint or observed_checkpoint)
            else 1.0,
            "python_validation_image_missing_error": 0.0 if observed_py_images >= min_py_images else 1.0,
            "plot_call_missing_error": 0.0 if (not require_plot or observed_plot) else 1.0,
        }
    return out


def _evaluate_metrics(fixture_paths: dict[str, Path]) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}

    # PPSimExample
    m = _mat(fixture_paths["PPSimExample_gold.mat"])
    X = np.asarray(m["X"], dtype=float)
    y = _vec(m, "y")
    dt = _scalar(m, "dt")
    fit = Analysis.fit_glm(X=X, y=y, fit_type="poisson", dt=dt)
    expected_rate = _vec(m, "expected_rate")
    pred_rate = np.asarray(fit.predict(X), dtype=float).reshape(-1)
    results["PPSimExample"] = {
        "mean_relative_rate_error": float(
            np.mean(np.abs(pred_rate - expected_rate) / np.maximum(expected_rate, 1e-12))
        ),
    }

    # DecodingExampleWithHist
    m = _mat(fixture_paths["DecodingExampleWithHist_gold.mat"])
    decoded, posterior = DecodingAlgorithms.decode_state_posterior(
        spike_counts=np.asarray(m["spike_counts"], dtype=float),
        tuning_rates=np.asarray(m["tuning"], dtype=float),
        transition=np.asarray(m["transition"], dtype=float),
    )
    expected_decoded = np.asarray(m["expected_decoded"], dtype=int).reshape(-1)
    expected_posterior = np.asarray(m["expected_posterior"], dtype=float)
    results["DecodingExampleWithHist"] = {
        "posterior_max_abs_error": float(np.max(np.abs(posterior - expected_posterior))),
        "decoded_mismatch_count": float(np.count_nonzero(decoded != expected_decoded)),
    }

    # HippocampalPlaceCellExample
    m = _mat(fixture_paths["HippocampalPlaceCellExample_gold.mat"])
    decoded = DecodingAlgorithms.decode_weighted_center(
        spike_counts=np.asarray(m["spike_counts_pc"], dtype=float),
        tuning_curves=np.asarray(m["tuning_curves"], dtype=float),
    )
    expected = _vec(m, "expected_decoded_weighted")
    results["HippocampalPlaceCellExample"] = {
        "weighted_center_max_abs_error": float(np.max(np.abs(decoded - expected))),
    }

    # SpikeRateDiffCIs
    m = _mat(fixture_paths["SpikeRateDiffCIs_gold.mat"])
    diff, lo, hi = DecodingAlgorithms.compute_spike_rate_diff_cis(
        spike_matrix_a=np.asarray(m["spike_matrix_a"], dtype=float),
        spike_matrix_b=np.asarray(m["spike_matrix_b"], dtype=float),
        alpha=_scalar(m, "alpha_diff"),
    )
    results["SpikeRateDiffCIs"] = {
        "diff_max_abs_error": float(np.max(np.abs(diff - _vec(m, "expected_diff")))),
        "lo_max_abs_error": float(np.max(np.abs(lo - _vec(m, "expected_lo")))),
        "hi_max_abs_error": float(np.max(np.abs(hi - _vec(m, "expected_hi")))),
    }

    # PSTHEstimation
    m = _mat(fixture_paths["PSTHEstimation_gold.mat"])
    rate, prob_mat, sig_mat = DecodingAlgorithms.compute_spike_rate_cis(
        spike_matrix=np.asarray(m["spike_matrix_psth"], dtype=float),
        alpha=_scalar(m, "alpha_psth"),
    )
    results["PSTHEstimation"] = {
        "rate_max_abs_error": float(np.max(np.abs(rate - _vec(m, "expected_rate_psth")))),
        "prob_max_abs_error": float(
            np.max(np.abs(prob_mat - np.asarray(m["expected_prob_psth"], dtype=float)))
        ),
        "sig_mismatch_count": float(
            np.count_nonzero(sig_mat != np.asarray(m["expected_sig_psth"], dtype=int))
        ),
    }

    # nstCollExamples
    m = _mat(fixture_paths["nstCollExamples_gold.mat"])
    st1 = SpikeTrain(
        spike_times=_vec(m, "spike_times_1"),
        t_start=_scalar(m, "t_start_coll"),
        t_end=_scalar(m, "t_end_coll"),
        name="u1",
    )
    st2 = SpikeTrain(
        spike_times=_vec(m, "spike_times_2"),
        t_start=_scalar(m, "t_start_coll"),
        t_end=_scalar(m, "t_end_coll"),
        name="u2",
    )
    coll = SpikeTrainCollection([st1, st2])
    centers_count, count_mat = coll.to_binned_matrix(
        bin_size_s=_scalar(m, "bin_size_coll"), mode="count"
    )
    _, binary_mat = coll.to_binned_matrix(bin_size_s=_scalar(m, "bin_size_coll"), mode="binary")
    merged = coll.to_spike_train(name="merged").spike_times
    results["nstCollExamples"] = {
        "center_max_abs_error": float(np.max(np.abs(centers_count - _vec(m, "expected_centers")))),
        "count_matrix_max_abs_error": float(
            np.max(np.abs(count_mat - np.asarray(m["expected_count_matrix"], dtype=float)))
        ),
        "binary_mismatch_count": float(
            np.count_nonzero(binary_mat != np.asarray(m["expected_binary_matrix"], dtype=float))
        ),
        "merged_max_abs_error": float(np.max(np.abs(merged - _vec(m, "expected_merged_spikes")))),
    }

    # CovCollExamples
    m = _mat(fixture_paths["CovCollExamples_gold.mat"])
    time = _vec(m, "time_cov")
    stim = Covariate(time=time, data=_vec(m, "cov_stim"), name="stim", labels=["stim"])
    ctx = Covariate(
        time=time,
        data=np.asarray(m["cov_ctx"], dtype=float),
        name="ctx",
        labels=["cosine", "ramp"],
    )
    cov_coll = CovariateCollection([stim, ctx])
    design, _ = cov_coll.design_matrix()
    ctx_only, _ = cov_coll.data_to_matrix_from_names(["ctx"])
    stim_only, _ = cov_coll.data_to_matrix_from_sel([0])
    results["CovCollExamples"] = {
        "design_max_abs_error": float(
            np.max(np.abs(design - np.asarray(m["expected_design_cov"], dtype=float)))
        ),
        "ctx_max_abs_error": float(
            np.max(np.abs(ctx_only - np.asarray(m["expected_ctx_only"], dtype=float)))
        ),
        "stim_max_abs_error": float(
            np.max(np.abs(stim_only.reshape(-1) - _vec(m, "expected_stim_only")))
        ),
    }

    # TrialExamples
    m = _mat(fixture_paths["TrialExamples_gold.mat"])
    time = _vec(m, "time_cov")
    stim = Covariate(time=time, data=_vec(m, "cov_stim"), name="stim", labels=["stim"])
    ctx = Covariate(
        time=time,
        data=np.asarray(m["cov_ctx"], dtype=float),
        name="ctx",
        labels=["cosine", "ramp"],
    )
    trial = Trial(
        spikes=SpikeTrainCollection(
            [
                SpikeTrain(
                    spike_times=_vec(m, "spike_times_trial"),
                    t_start=0.0,
                    t_end=1.0,
                    name="u1",
                )
            ]
        ),
        covariates=CovariateCollection([stim, ctx]),
    )
    t_bins, y, X = trial.aligned_binned_observation(
        bin_size_s=_scalar(m, "bin_size_trial"),
        unit_index=0,
        mode="count",
    )
    results["TrialExamples"] = {
        "t_bins_max_abs_error": float(np.max(np.abs(t_bins - _vec(m, "expected_t_bins_trial")))),
        "y_max_abs_error": float(np.max(np.abs(y.reshape(-1) - _vec(m, "expected_y_trial")))),
        "X_max_abs_error": float(np.max(np.abs(X - np.asarray(m["expected_X_trial"], dtype=float)))),
    }

    # EventsExamples
    m = _mat(fixture_paths["EventsExamples_gold.mat"])
    subset = Events(times=_vec(m, "event_times"), labels=["E1", "E2", "E3"]).subset(
        _scalar(m, "subset_start"),
        _scalar(m, "subset_end"),
    )
    results["EventsExamples"] = {
        "subset_max_abs_error": float(np.max(np.abs(subset.times - _vec(m, "expected_subset_times")))),
    }

    # AnalysisExamples
    m = _mat(fixture_paths["AnalysisExamples_gold.mat"])
    X = np.asarray(m["X_analysis"], dtype=float)
    y = _vec(m, "counts_analysis")
    fit = Analysis.fit_glm(X=X, y=y, fit_type="poisson", dt=_scalar(m, "dt_analysis"))
    pred = np.asarray(fit.predict(X), dtype=float).reshape(-1)
    rmse = float(np.sqrt(np.mean((pred - _vec(m, "true_rate_analysis")) ** 2)))
    results["AnalysisExamples"] = {
        "intercept_abs_error": float(abs(fit.intercept - _vec(m, "b_analysis")[0])),
        "coeff_max_abs_error": float(np.max(np.abs(fit.coefficients - _vec(m, "b_analysis")[1:]))),
        "rate_max_abs_error": float(np.max(np.abs(pred - _vec(m, "expected_rate_analysis")))),
        "rmse_abs_error": float(abs(rmse - _scalar(m, "expected_rmse_analysis"))),
    }

    # DecodingExample
    m = _mat(fixture_paths["DecodingExample_gold.mat"])
    decoded, posterior = DecodingAlgorithms.decode_state_posterior(
        spike_counts=np.asarray(m["spike_counts_dec"], dtype=float),
        tuning_rates=np.asarray(m["tuning_dec"], dtype=float),
        transition=np.asarray(m["transition_dec"], dtype=float),
    )
    latent = _vec(m, "latent_zero_dec").astype(int)
    rmse = float(np.sqrt(np.mean((decoded - latent) ** 2)) / max(np.asarray(m["tuning_dec"]).shape[1] - 1, 1))
    results["DecodingExample"] = {
        "posterior_max_abs_error": float(
            np.max(np.abs(posterior - np.asarray(m["expected_posterior_dec"], dtype=float)))
        ),
        "decoded_mismatch_count": float(
            np.count_nonzero(decoded != np.asarray(m["expected_decoded_dec"], dtype=int).reshape(-1))
        ),
        "rmse_abs_error": float(abs(rmse - _scalar(m, "expected_rmse_dec"))),
    }

    # ExplicitStimulusWhiskerData
    m = _mat(fixture_paths["ExplicitStimulusWhiskerData_gold.mat"])
    stimulus = _vec(m, "stimulus_ws")
    y = _vec(m, "spike_ws")
    fit = Analysis.fit_glm(X=stimulus[:, None], y=y, fit_type="binomial", dt=1.0)
    pred = np.asarray(fit.predict(stimulus[:, None]), dtype=float).reshape(-1)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    b = _vec(m, "b_ws")
    results["ExplicitStimulusWhiskerData"] = {
        "intercept_abs_error": float(abs(fit.intercept - b[0])),
        "coeff_abs_error": float(abs(fit.coefficients[0] - b[1])),
        "prob_max_abs_error": float(np.max(np.abs(pred - _vec(m, "expected_prob_ws")))),
        "rmse_abs_error": float(abs(rmse - _scalar(m, "expected_rmse_ws"))),
    }

    # mEPSCAnalysis
    m = _mat(fixture_paths["mEPSCAnalysis_gold.mat"])
    dt = _scalar(m, "dt_mepsc")
    det_times, det_amps = _detect_mepsc_events(_vec(m, "trace_mepsc"), dt)
    exp_times = _vec(m, "detected_times_mepsc")
    exp_amps = _vec(m, "detected_amps_mepsc")
    exp_count = int(round(_scalar(m, "expected_event_count_mepsc")))
    exp_mean_amp = _scalar(m, "expected_mean_amp_mepsc")
    count_mismatch = abs(int(det_times.size) - exp_count)
    results["mEPSCAnalysis"] = {
        "detected_time_max_abs_error": float(np.max(np.abs(det_times - exp_times))) if det_times.size else 0.0,
        "detected_amp_max_abs_error": float(np.max(np.abs(det_amps - exp_amps))) if det_amps.size else 0.0,
        "mean_amp_abs_error": float(abs(float(np.mean(det_amps)) - exp_mean_amp)) if det_amps.size else 0.0,
        "event_count_mismatch": float(count_mismatch),
    }

    return results


def _build_report(
    metrics_by_topic: dict[str, dict[str, float]],
    thresholds_payload: dict,
    fixtures_manifest: Path,
    thresholds_file: Path,
    required_topics: list[str],
) -> dict:
    thresholds_by_topic = thresholds_payload.get("topics", {})
    default_thresholds = thresholds_payload.get("defaults", {})
    topics_out: dict[str, dict] = {}

    total_checked = 0
    total_failed = 0
    passed_topics = 0

    for topic, metrics in sorted(metrics_by_topic.items()):
        topic_thresholds = thresholds_by_topic.get(topic, {})
        metric_rows: dict[str, dict] = {}
        failed: list[str] = []
        worst_ratio = 0.0
        checked = 0

        for metric_name, value in sorted(metrics.items()):
            threshold_value = topic_thresholds.get(metric_name, default_thresholds.get(metric_name))
            if threshold_value is None:
                continue
            threshold = float(threshold_value)
            passed = value <= threshold
            ratio = _ratio(float(value), threshold)
            metric_rows[metric_name] = {
                "value": float(value),
                "threshold": threshold,
                "pass": bool(passed),
                "ratio_to_threshold": float(ratio),
            }
            checked += 1
            worst_ratio = max(worst_ratio, ratio if np.isfinite(ratio) else float("inf"))
            if not passed:
                failed.append(metric_name)

        topic_pass = len(failed) == 0 and checked > 0
        if topic_pass:
            passed_topics += 1
        total_checked += checked
        total_failed += len(failed)

        topics_out[topic] = {
            "checked_metrics": checked,
            "failed_metrics": failed,
            "worst_ratio_to_threshold": float(worst_ratio),
            "pass": topic_pass,
            "metrics": metric_rows,
        }

    missing_required_topics = sorted(
        topic for topic in required_topics if topic not in topics_out or int(topics_out[topic]["checked_metrics"]) == 0
    )
    checked_required_topics = len(required_topics) - len(missing_required_topics)

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixtures_manifest": fixtures_manifest.as_posix(),
        "thresholds_file": thresholds_file.as_posix(),
        "summary": {
            "topics": len(topics_out),
            "passed_topics": passed_topics,
            "failed_topics": len(topics_out) - passed_topics,
            "checked_metrics": total_checked,
            "failed_metrics": total_failed,
            "required_topics": len(required_topics),
            "required_topics_checked": checked_required_topics,
            "required_topics_missing": missing_required_topics,
        },
        "topics": topics_out,
    }
    return report


def main() -> int:
    args = parse_args()
    fixtures_manifest = args.fixtures_manifest.resolve()
    thresholds_file = args.thresholds.resolve()
    report_out = args.report_out.resolve()
    equivalence_report = args.equivalence_report.resolve()
    notebook_manifest = args.notebook_manifest.resolve()

    fixture_index = _fixture_manifest_index(fixtures_manifest)
    numeric_fixture_paths = _numeric_fixture_paths(fixture_index)
    metrics = _evaluate_metrics(numeric_fixture_paths)
    topic_audit_fixtures = _topic_audit_fixtures(fixture_index)
    equivalence_rows = _load_equivalence_rows(equivalence_report)
    topic_audit_metrics = _evaluate_topic_audit_metrics(topic_audit_fixtures, equivalence_rows)
    for topic, topic_metrics in topic_audit_metrics.items():
        merged = dict(metrics.get(topic, {}))
        merged.update(topic_metrics)
        metrics[topic] = merged

    required_topics = _load_required_topics(notebook_manifest)
    for topic in required_topics:
        merged = dict(metrics.get(topic, {}))
        merged["topic_audit_fixture_missing_error"] = 0.0 if topic in topic_audit_fixtures else 1.0
        metrics[topic] = merged

    thresholds_payload = yaml.safe_load(thresholds_file.read_text(encoding="utf-8")) or {}
    report = _build_report(metrics, thresholds_payload, fixtures_manifest, thresholds_file, required_topics)

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    failed_topics = report["summary"]["failed_topics"]
    failed_metrics = report["summary"]["failed_metrics"]
    missing_required = report["summary"].get("required_topics_missing", [])
    print(f"Wrote numeric drift report: {report_out}")
    print(f"Topics: {report['summary']['topics']}")
    print(f"Failed topics: {failed_topics}")
    print(f"Failed metrics: {failed_metrics}")
    print(
        "Required topics coverage: "
        f"{report['summary'].get('required_topics_checked', 0)}/{report['summary'].get('required_topics', 0)}"
    )
    if missing_required:
        print("Missing required topics:", ", ".join(missing_required))

    if args.fail_on_violation and (failed_topics > 0 or failed_metrics > 0 or len(missing_required) > 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
