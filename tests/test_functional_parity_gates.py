from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"{' '.join(cmd)} failed:\n{proc.stdout}\n{proc.stderr}"


def test_functional_parity_gate_passes() -> None:
    _run(
        [
            sys.executable,
            "tools/parity/check_functional_parity_progress.py",
            "--report",
            "parity/function_example_alignment_report.json",
            "--policy",
            "parity/functional_gate_policy.yml",
        ]
    )


def test_example_output_spec_gate_passes() -> None:
    _run(
        [
            sys.executable,
            "tools/parity/check_example_output_spec.py",
            "--report",
            "parity/function_example_alignment_report.json",
            "--spec",
            "parity/example_output_spec.yml",
        ]
    )
