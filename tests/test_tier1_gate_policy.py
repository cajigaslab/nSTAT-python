from __future__ import annotations

import json
from pathlib import Path

import yaml



def test_tier1_gate_policy_targets_report_classes() -> None:
    policy = yaml.safe_load(Path("parity/tier1_gate_policy.yml").read_text(encoding="utf-8"))
    report = json.loads(Path("parity/parity_gap_report.json").read_text(encoding="utf-8"))
    report_classes = {row["matlab_class"] for row in report["class_coverage"]}

    for klass in policy["min_coverage_ratio"]:
        assert klass in report_classes
    for klass in policy["max_missing_methods"]:
        assert klass in report_classes



def test_tier1_gate_policy_current_report_passes_thresholds() -> None:
    policy = yaml.safe_load(Path("parity/tier1_gate_policy.yml").read_text(encoding="utf-8"))
    report = json.loads(Path("parity/parity_gap_report.json").read_text(encoding="utf-8"))
    rows = {row["matlab_class"]: row for row in report["class_coverage"]}

    for klass, min_ratio in policy["min_coverage_ratio"].items():
        assert float(rows[klass]["coverage_ratio"]) >= float(min_ratio)

    for klass, max_missing in policy["max_missing_methods"].items():
        assert int(rows[klass]["missing_method_count"]) <= int(max_missing)
