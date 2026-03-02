from __future__ import annotations

import json
from pathlib import Path


def test_equivalence_audit_report_exists_and_has_schema() -> None:
    report_path = Path("parity/function_example_alignment_report.json")
    assert report_path.exists(), "parity/function_example_alignment_report.json must exist"

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "method_functional_audit" in payload
    assert "example_line_alignment_audit" in payload

    method_summary = payload["method_functional_audit"]["summary"]
    assert method_summary["total_methods"] >= 1
    assert method_summary["missing_symbol_methods"] == 0

    example_summary = payload["example_line_alignment_audit"]["summary"]
    assert example_summary["total_topics"] >= 1

