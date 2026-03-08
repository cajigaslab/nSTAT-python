from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
HELPER_DIR = THIS_DIR / "matlab"


def _matlab_quote(value: str) -> str:
    return value.replace("'", "''")


def default_matlab_repo(repo_root: Path | None = None) -> Path:
    base = REPO_ROOT if repo_root is None else repo_root.resolve()
    return base.parent / "nSTAT"


def build_matlab_batch(repo_root: Path, matlab_repo: Path) -> str:
    return (
        f"addpath('{_matlab_quote(str(HELPER_DIR))}'); "
        f"export_matlab_gold_fixtures('{_matlab_quote(str(repo_root))}','{_matlab_quote(str(matlab_repo))}'); "
        "exit"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate MATLAB-derived gold fixtures for Python parity tests.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--matlab-repo", type=Path, default=None)
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    matlab_repo = default_matlab_repo(repo_root) if args.matlab_repo is None else args.matlab_repo.resolve()
    if not matlab_repo.exists():
        raise FileNotFoundError(f"MATLAB reference repo not found at {matlab_repo}")
    if shutil.which("matlab") is None:
        raise FileNotFoundError("`matlab` is not on PATH")

    command = ["matlab", "-batch", build_matlab_batch(repo_root, matlab_repo)]
    subprocess.run(command, cwd=repo_root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
