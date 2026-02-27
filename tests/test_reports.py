from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_phase0_reports_generation(project_root) -> None:
    if os.environ.get("NSTAT_CI_LIGHT") == "1":
        pytest.skip("Report generation already executed in dedicated CI workflow step")
    subprocess.run([sys.executable, "tools/freeze_port_baseline.py"], cwd=str(project_root), check=True)
    subprocess.run([sys.executable, "tools/generate_method_parity_matrix.py"], cwd=str(project_root), check=True)

    snapshot = json.loads((project_root / "reports/port_baseline_snapshot.json").read_text(encoding="utf-8"))
    matrix = json.loads((project_root / "reports/method_parity_matrix.json").read_text(encoding="utf-8"))

    assert "reports" in snapshot
    assert matrix["summary"]["total"] > 0
    assert len(matrix["classes"]) >= 10
