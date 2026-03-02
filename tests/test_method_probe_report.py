from __future__ import annotations

import json
from pathlib import Path


def test_method_probe_report_exists_and_has_schema() -> None:
    report_path = Path("parity/method_probe_report.json")
    assert report_path.exists(), "parity/method_probe_report.json must exist"

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "summary" in payload
    assert "class_rows" in payload

    summary = payload["summary"]
    assert summary["total_methods"] >= 1
    assert summary["attempted_methods"] >= 1
    assert summary["successful_methods"] >= 1
    assert summary["successful_methods"] <= summary["attempted_methods"]

    class_rows = payload["class_rows"]
    assert len(class_rows) >= 1
    assert all("matlab_class" in row for row in class_rows)
