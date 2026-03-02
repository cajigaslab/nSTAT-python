#!/usr/bin/env python3
"""Regenerate parity artifacts from the current functional audit."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing parity/docs artifacts.",
    )
    parser.add_argument(
        "--matlab-root",
        type=Path,
        default=None,
        help="MATLAB nSTAT root used to regenerate functional audit.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("parity/function_example_alignment_report.json"),
        help="Functional audit report path (relative to --repo-root).",
    )
    parser.add_argument(
        "--method-closure",
        type=Path,
        default=Path("parity/method_closure_sprint.md"),
        help="Method closure sprint markdown output (relative to --repo-root).",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip regenerating parity/function_example_alignment_report.json.",
    )
    parser.add_argument(
        "--skip-help-pages",
        action="store_true",
        help="Skip docs/help regeneration.",
    )
    return parser.parse_args()


def run(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    report_path = (repo_root / args.report).resolve()
    method_closure_path = (repo_root / args.method_closure).resolve()

    if not args.skip_audit:
        if args.matlab_root is None:
            raise SystemExit("--matlab-root is required unless --skip-audit is set.")
        run(
            [
                sys.executable,
                str(repo_root / "tools" / "parity" / "generate_equivalence_audit.py"),
                "--repo-root",
                str(repo_root),
                "--matlab-root",
                str(args.matlab_root.resolve()),
                "--out",
                str(report_path),
            ],
            cwd=repo_root,
        )

    run(
        [
            sys.executable,
            str(repo_root / "tools" / "parity" / "generate_method_closure_sprint.py"),
            "--report",
            str(report_path),
            "--output",
            str(method_closure_path),
        ],
        cwd=repo_root,
    )

    if not args.skip_help_pages:
        run(
            [
                sys.executable,
                str(repo_root / "tools" / "docs" / "generate_help_pages.py"),
            ],
            cwd=repo_root,
        )

    print("Parity artifacts synchronized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
