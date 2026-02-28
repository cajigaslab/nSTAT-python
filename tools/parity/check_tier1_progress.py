#!/usr/bin/env python3
"""Enforce Tier-1 parity progress thresholds from a policy file."""

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
        default=Path("parity/parity_gap_report.json"),
        help="Parity gap report JSON path.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("parity/tier1_gate_policy.yml"),
        help="Tier-1 gate policy YAML path.",
    )
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    policy = yaml.safe_load(args.policy.read_text(encoding="utf-8"))

    rows = {row["matlab_class"]: row for row in report["class_coverage"]}

    failures: list[str] = []

    for klass, min_ratio in policy.get("min_coverage_ratio", {}).items():
        if klass not in rows:
            failures.append(f"missing class in report: {klass}")
            continue
        ratio = float(rows[klass]["coverage_ratio"])
        if ratio < float(min_ratio):
            failures.append(
                f"{klass}: coverage_ratio {ratio:.3f} below required {float(min_ratio):.3f}"
            )

    for klass, max_missing in policy.get("max_missing_methods", {}).items():
        if klass not in rows:
            failures.append(f"missing class in report: {klass}")
            continue
        missing = int(rows[klass]["missing_method_count"])
        if missing > int(max_missing):
            failures.append(
                f"{klass}: missing_method_count {missing} exceeds allowed {int(max_missing)}"
            )

    if failures:
        print("Tier-1 parity gate FAILED")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("Tier-1 parity gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
