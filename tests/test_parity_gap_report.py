from __future__ import annotations

import json
from pathlib import Path



def test_parity_gap_report_exists_and_has_schema() -> None:
    report_path = Path("parity/parity_gap_report.json")
    assert report_path.exists(), "parity/parity_gap_report.json must exist"

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "summary" in payload
    assert "issues" in payload
    assert "class_coverage" in payload
    assert "example_coverage" in payload
    assert payload["summary"]["high"] == 0
