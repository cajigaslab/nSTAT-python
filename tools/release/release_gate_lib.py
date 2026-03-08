from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_release_gate_commands(repo: Path | None = None) -> list[list[str]]:
    base = repo_root() if repo is None else repo.resolve()
    return [
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_zernike.py",
            "tests/test_matlab_gold_fixtures.py",
            "tests/test_signalobj_fidelity.py",
            "tests/test_nspiketrain_fidelity.py",
            "tests/test_workflow_fidelity.py",
            "tests/test_class_fidelity_audit.py",
            "tests/test_matlab_symbol_surface.py",
            "tests/test_simulink_fidelity_audit.py",
            "tests/test_parity_report.py",
            "tests/test_cleanroom_boundary.py",
        ],
        [sys.executable, str(base / "tools" / "notebooks" / "run_notebooks.py"), "--group", "parity_core", "--timeout", "900"],
        [sys.executable, str(base / "tools" / "parity" / "build_report.py")],
    ]


def run_release_gate(repo: Path | None = None) -> None:
    base = repo_root() if repo is None else repo.resolve()
    for command in build_release_gate_commands(base):
        subprocess.run(command, cwd=base, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pure-Python fidelity release gate.")
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args(argv)

    if shutil.which(sys.executable) is None:
        raise FileNotFoundError(f"Python executable is not on PATH: {sys.executable}")
    run_release_gate(args.repo_root)
    return 0


__all__ = ["build_release_gate_commands", "main", "repo_root", "run_release_gate"]

