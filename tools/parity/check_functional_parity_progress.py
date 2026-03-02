#!/usr/bin/env python3
"""Enforce functional parity thresholds from equivalence audit output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("parity/function_example_alignment_report.json"),
        help="Functional equivalence audit report JSON path.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("parity/functional_gate_policy.yml"),
        help="Functional gate policy YAML path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    policy = yaml.safe_load(args.policy.read_text(encoding="utf-8"))

    method_summary = report["method_functional_audit"]["summary"]
    class_rows = {row["matlab_class"]: row for row in report["method_functional_audit"]["class_summary"]}
    example_summary = report["example_line_alignment_audit"]["summary"]

    failures: list[str] = []

    method_policy = policy.get("method_thresholds", {})
    min_verified_ratio_overall = float(method_policy.get("min_verified_ratio_overall", 0.0))
    ratio = float(method_summary.get("contract_verified_ratio", 0.0))
    if ratio < min_verified_ratio_overall:
        failures.append(
            f"overall verified ratio {ratio:.3f} below required {min_verified_ratio_overall:.3f}"
        )

    min_eligible_ratio_overall = float(method_policy.get("min_eligible_verified_ratio_overall", 0.0))
    eligible_ratio = float(method_summary.get("eligible_verified_ratio", 0.0))
    if eligible_ratio < min_eligible_ratio_overall:
        failures.append(
            "overall eligible verified ratio "
            f"{eligible_ratio:.3f} below required {min_eligible_ratio_overall:.3f}"
        )

    max_missing = int(method_policy.get("max_missing_symbol_methods", 0))
    missing = int(method_summary.get("missing_symbol_methods", 0))
    if missing > max_missing:
        failures.append(f"missing symbol methods {missing} exceeds allowed {max_missing}")

    for klass, min_count in method_policy.get("class_min_verified_methods", {}).items():
        row = class_rows.get(klass)
        if row is None:
            failures.append(f"missing class in functional report: {klass}")
            continue
        verified = int(row.get("contract_verified_count", 0))
        if verified < int(min_count):
            failures.append(
                f"{klass}: verified method count {verified} below required {int(min_count)}"
            )

    for klass, min_ratio in method_policy.get("class_min_eligible_verified_ratio", {}).items():
        row = class_rows.get(klass)
        if row is None:
            failures.append(f"missing class in functional report: {klass}")
            continue
        ratio_val = float(row.get("eligible_verified_ratio", 0.0))
        if ratio_val < float(min_ratio):
            failures.append(
                f"{klass}: eligible verified ratio {ratio_val:.3f} below required {float(min_ratio):.3f}"
            )

    example_policy = policy.get("example_thresholds", {})
    for key, policy_key in (
        ("missing_artifact_topics", "max_missing_artifact_topics"),
        ("missing_executable_topics", "max_missing_executable_topics"),
        ("pending_manual_review_topics", "max_pending_manual_review_topics"),
    ):
        if policy_key not in example_policy:
            continue
        observed = int(example_summary.get(key, 0))
        maximum = int(example_policy[policy_key])
        if observed > maximum:
            failures.append(f"{key}={observed} exceeds allowed {maximum}")

    if "min_matlab_doc_only_topics" in example_policy:
        observed = int(example_summary.get("matlab_doc_only_topics", 0))
        minimum = int(example_policy["min_matlab_doc_only_topics"])
        if observed < minimum:
            failures.append(f"matlab_doc_only_topics={observed} below required {minimum}")

    if failures:
        print("Functional parity gate FAILED")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("Functional parity gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
