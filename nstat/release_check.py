from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_matlab_repo_root(repo_root: Path | None = None) -> Path:
    base = _repo_root() if repo_root is None else repo_root.resolve()
    return base.parent / "nSTAT"


def _matlab_quote(value: str) -> str:
    return value.replace("'", "''")


def build_release_gate_commands(
    repo_root: Path | None = None,
    *,
    matlab_repo_root: Path | None = None,
    skip_matlab: bool = False,
) -> list[list[str]]:
    base = _repo_root() if repo_root is None else repo_root.resolve()
    matlab_root = default_matlab_repo_root(base) if matlab_repo_root is None else matlab_repo_root.resolve()

    commands: list[list[str]] = [
        [sys.executable, str(base / "tools" / "parity" / "export_matlab_gold_fixtures.py"), "--repo-root", str(base), "--matlab-repo", str(matlab_root)],
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
            "tests/test_matlab_reference.py",
            "tests/test_simulink_fidelity_audit.py",
            "tests/test_parity_report.py",
        ],
        [sys.executable, str(base / "tools" / "notebooks" / "run_notebooks.py"), "--group", "parity_core", "--timeout", "900"],
        [sys.executable, str(base / "tools" / "parity" / "build_report.py")],
    ]
    if not skip_matlab:
        commands.append(
            [
                "matlab",
                "-batch",
                (
                    f"cd('{_matlab_quote(str(matlab_root))}'); "
                    "addpath(pwd); "
                    "addpath(fullfile(pwd,'helpfiles')); "
                    "results = runtests('tests/python_port_fidelity'); "
                    "assertSuccess(results); exit"
                ),
            ]
        )
    return commands


def run_release_gate(
    repo_root: Path | None = None,
    *,
    matlab_repo_root: Path | None = None,
    skip_matlab: bool = False,
) -> None:
    base = _repo_root() if repo_root is None else repo_root.resolve()
    if not skip_matlab and shutil.which("matlab") is None:
        raise FileNotFoundError("`matlab` is not on PATH; rerun with --skip-matlab or install MATLAB CLI access")
    for command in build_release_gate_commands(base, matlab_repo_root=matlab_repo_root, skip_matlab=skip_matlab):
        subprocess.run(command, cwd=base, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Python plus MATLAB fidelity release gate.")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--matlab-repo", type=Path, default=None)
    parser.add_argument("--skip-matlab", action="store_true")
    args = parser.parse_args(argv)

    run_release_gate(args.repo_root, matlab_repo_root=args.matlab_repo, skip_matlab=args.skip_matlab)
    return 0


__all__ = ["build_release_gate_commands", "default_matlab_repo_root", "main", "run_release_gate"]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
