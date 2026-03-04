from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_execute_notebooks_smoke_subset(tmp_path: Path) -> None:
    report_path = tmp_path / "notebook_execution_report.json"
    cmd = [
        sys.executable,
        "tools/notebooks/execute_notebooks.py",
        "--group",
        "smoke",
        "--max-notebooks",
        "2",
        "--timeout",
        "600",
        "--out-report",
        str(report_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(
            "Notebook smoke execution failed.\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    assert int(summary.get("total", 0)) == 2
    assert int(summary.get("failed", 1)) == 0

    rows = payload.get("reports", [])
    assert len(rows) == 2
    assert all(bool(row.get("executed_ok", False)) for row in rows)

