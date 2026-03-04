#!/usr/bin/env python3
"""Compare Python benchmark report against MATLAB baseline performance report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def _index_cases(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        out[(str(row["case"]), str(row["tier"]))] = row
    return out


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0.0:
        return float("inf")
    return float(num / den)


def _major_minor(version: Any) -> str:
    text = str(version or "")
    parts = text.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return text


def _is_regression_env_compatible(current: dict[str, Any], previous: dict[str, Any]) -> bool:
    # Performance regressions are only meaningful when runner platform and Python minor line match.
    return (
        str(current.get("platform", "")) == str(previous.get("platform", ""))
        and _major_minor(current.get("python", "")) == _major_minor(previous.get("python", ""))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-report", type=Path, required=True, help="Python benchmark JSON report.")
    parser.add_argument("--matlab-report", type=Path, required=True, help="MATLAB benchmark JSON report.")
    parser.add_argument("--policy", type=Path, default=Path("parity/performance_gate_policy.yml"))
    parser.add_argument(
        "--previous-python-report",
        type=Path,
        default=None,
        help="Optional previous Python benchmark report for regression detection.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("parity/performance_parity_report.json"),
        help="Output comparison JSON path.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("parity/performance_parity_report.csv"),
        help="Output comparison CSV path.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Return non-zero when Python runtime regresses beyond threshold vs previous report.",
    )
    parser.add_argument(
        "--fail-on-matlab-ratio",
        action="store_true",
        help="Return non-zero when Python/MATLAB runtime ratio exceeds policy threshold.",
    )
    args = parser.parse_args()

    py_report = json.loads(args.python_report.read_text(encoding="utf-8"))
    ml_report = json.loads(args.matlab_report.read_text(encoding="utf-8"))
    policy = yaml.safe_load(args.policy.read_text(encoding="utf-8")) or {}

    prev_idx: dict[tuple[str, str], dict[str, Any]] = {}
    regression_env_compatible = True
    if args.previous_python_report and args.previous_python_report.exists():
        prev = json.loads(args.previous_python_report.read_text(encoding="utf-8"))
        regression_env_compatible = _is_regression_env_compatible(
            py_report.get("environment", {}) or {},
            prev.get("environment", {}) or {},
        )
        if regression_env_compatible:
            prev_idx = _index_cases(prev.get("cases", []))
        else:
            print(
                "Skipping regression gating: benchmark environments are not comparable "
                f"(current={py_report.get('environment', {})}, previous={prev.get('environment', {})})"
            )

    py_idx = _index_cases(py_report.get("cases", []))
    ml_idx = _index_cases(ml_report.get("cases", []))

    default_ratio = float(policy.get("default_max_matlab_ratio", 5.0))
    critical = policy.get("critical_case_max_matlab_ratio", {}) or {}
    regression_limit = float(policy.get("max_python_regression_ratio", 1.35))
    min_regression_delta_ms = float(policy.get("min_python_regression_delta_ms", 0.0))

    rows: list[dict[str, Any]] = []
    missing_matlab = 0
    ratio_fail = 0
    regression_fail = 0

    for key, py_case in sorted(py_idx.items()):
        case, tier = key
        ml_case = ml_idx.get(key)
        py_runtime = float(py_case.get("median_runtime_ms", float("nan")))
        py_mem = float(py_case.get("median_peak_memory_mb", float("nan")))

        if ml_case is None:
            missing_matlab += 1
            rows.append(
                {
                    "case": case,
                    "tier": tier,
                    "python_runtime_ms": py_runtime,
                    "matlab_runtime_ms": float("nan"),
                    "python_to_matlab_ratio": float("inf"),
                    "max_allowed_ratio": float(critical.get(case, default_ratio)),
                    "ratio_pass": False,
                    "regression_pass": True,
                    "python_peak_memory_mb": py_mem,
                    "status": "missing_matlab_baseline",
                }
            )
            continue

        ml_runtime = float(ml_case.get("median_runtime_ms", float("nan")))
        ratio = _safe_ratio(py_runtime, ml_runtime)
        max_allowed = float(critical.get(case, default_ratio))
        ratio_pass = bool(ratio <= max_allowed)
        if not ratio_pass:
            ratio_fail += 1

        prev_case = prev_idx.get(key)
        regression_pass = True
        prev_runtime = float("nan")
        py_vs_prev_ratio = float("nan")
        py_vs_prev_delta_ms = float("nan")
        if prev_case is not None:
            prev_runtime = float(prev_case.get("median_runtime_ms", float("nan")))
            py_vs_prev_ratio = _safe_ratio(py_runtime, prev_runtime)
            py_vs_prev_delta_ms = py_runtime - prev_runtime
            ratio_ok = bool(py_vs_prev_ratio <= regression_limit)
            delta_ok = bool(
                math.isnan(py_vs_prev_delta_ms) or py_vs_prev_delta_ms <= min_regression_delta_ms
            )
            regression_pass = bool(ratio_ok or delta_ok)
            if not regression_pass:
                regression_fail += 1

        rows.append(
            {
                "case": case,
                "tier": tier,
                "python_runtime_ms": py_runtime,
                "matlab_runtime_ms": ml_runtime,
                "python_to_matlab_ratio": ratio,
                "max_allowed_ratio": max_allowed,
                "ratio_pass": ratio_pass,
                "python_peak_memory_mb": py_mem,
                "previous_python_runtime_ms": prev_runtime,
                "python_vs_previous_ratio": py_vs_prev_ratio,
                "python_vs_previous_delta_ms": py_vs_prev_delta_ms,
                "regression_pass": regression_pass,
                "status": "ok" if ratio_pass and regression_pass else "needs_attention",
            }
        )

    worst = sorted(
        [r for r in rows if r["python_to_matlab_ratio"] != float("inf")],
        key=lambda r: float(r["python_to_matlab_ratio"]),
        reverse=True,
    )[:5]

    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "policy": {
            "default_max_matlab_ratio": default_ratio,
            "critical_case_max_matlab_ratio": critical,
            "max_python_regression_ratio": regression_limit,
            "min_python_regression_delta_ms": min_regression_delta_ms,
            "regression_env_compatible": regression_env_compatible,
        },
        "python_report": str(args.python_report),
        "matlab_report": str(args.matlab_report),
        "previous_python_report": str(args.previous_python_report) if args.previous_python_report else "",
        "counts": {
            "total_case_tiers": len(rows),
            "missing_matlab_baseline": missing_matlab,
            "ratio_failures": ratio_fail,
            "regression_failures": regression_fail,
        },
        "top_python_vs_matlab_gaps": worst,
        "rows": rows,
    }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "tier",
                "python_runtime_ms",
                "matlab_runtime_ms",
                "python_to_matlab_ratio",
                "max_allowed_ratio",
                "ratio_pass",
                "python_peak_memory_mb",
                "previous_python_runtime_ms",
                "python_vs_previous_ratio",
                "python_vs_previous_delta_ms",
                "regression_pass",
                "status",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote performance parity JSON: {args.report_out}")
    print(f"Wrote performance parity CSV: {args.csv_out}")
    print(
        "Counts: "
        f"total={len(rows)} missing_matlab={missing_matlab} "
        f"ratio_fail={ratio_fail} regression_fail={regression_fail}"
    )

    if args.fail_on_matlab_ratio and ratio_fail > 0:
        return 1
    if args.fail_on_regression and regression_fail > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
