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
    assert "strict_line_verified_topics" in example_summary
    assert "strict_line_partial_topics" in example_summary
    assert "strict_line_gap_topics" in example_summary

    topic_rows = payload["example_line_alignment_audit"]["topic_rows"]
    assert topic_rows, "example_line_alignment_audit.topic_rows must not be empty"
    required_topic_fields = {
        "topic",
        "strict_line_status",
        "line_port_coverage",
        "line_port_function_recall",
        "line_port_matched_lines",
        "line_port_matlab_lines",
        "line_port_python_lines",
        "line_port_matlab_function_count",
        "line_port_python_function_count",
    }
    for row in topic_rows:
        missing = required_topic_fields.difference(row)
        assert not missing, f"Missing strict line-port fields for topic {row.get('topic')}: {sorted(missing)}"


def test_top_mismatch_topics_meet_line_port_regression_thresholds() -> None:
    report_path = Path("parity/function_example_alignment_report.json")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    topic_rows = payload["example_line_alignment_audit"]["topic_rows"]
    topic_lookup = {str(row["topic"]): row for row in topic_rows}

    thresholds = {
        "nSTATPaperExamples": (0.45, 0.95),
        "HippocampalPlaceCellExample": (0.35, 0.95),
        "publish_all_helpfiles": (0.90, 0.95),
    }
    allowed_statuses = {"line_port_partial", "line_port_verified"}

    for topic, (min_cov, min_recall) in thresholds.items():
        assert topic in topic_lookup, f"Missing topic row for {topic}"
        row = topic_lookup[topic]
        coverage = float(row["line_port_coverage"])
        recall = float(row["line_port_function_recall"])
        status = str(row["strict_line_status"])
        assert coverage >= min_cov, f"{topic}: coverage {coverage:.4f} < {min_cov:.4f}"
        assert recall >= min_recall, f"{topic}: function recall {recall:.4f} < {min_recall:.4f}"
        assert status in allowed_statuses, f"{topic}: strict status {status} not in {sorted(allowed_statuses)}"
