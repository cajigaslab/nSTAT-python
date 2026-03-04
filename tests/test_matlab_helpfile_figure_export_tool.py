from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_export_matlab_helpfile_figures_tool_dry_run() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        "python",
        "tools/reports/export_matlab_helpfile_figures.py",
        "--source-manifest",
        "parity/help_source_manifest.yml",
        "--output-root",
        "output/matlab_help_images_test",
        "--report-json",
        "output/matlab_help_images_test/report.json",
        "--dry-run",
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": "src:."},
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "Resolved" in proc.stdout
