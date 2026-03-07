from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_python_module_installer_help_exposes_supported_flags() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "nstat.install", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "RuntimeWarning" not in proc.stderr
    assert "--download-example-data" in proc.stdout
    assert "--no-rebuild-doc-search" in proc.stdout
    assert "--clean-user-path-prefs" in proc.stdout


def test_python_module_installer_emits_json_report_without_download() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nstat.install",
            "--download-example-data",
            "never",
            "--no-rebuild-doc-search",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "RuntimeWarning" not in proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["download_example_data"] == "never"
    assert payload["doc_search"]["status"] == "skipped"
    assert payload["example_data"]["is_installed"] in {True, False}


def test_python_module_installer_reports_matlab_compat_noop() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nstat.install",
            "--download-example-data",
            "never",
            "--no-rebuild-doc-search",
            "--clean-user-path-prefs",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "RuntimeWarning" not in proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["path_preferences"]["status"] == "not_applicable"
    assert "ignored in Python" in " ".join(payload["notes"])
