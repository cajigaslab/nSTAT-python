#!/usr/bin/env python3
"""Build full parity snapshot (inventories + gap report)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--matlab-root", type=Path, required=True)
    parser.add_argument("--fail-on", choices=["none", "low", "medium", "high"], default="high")
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    inventory_cmd = [
        sys.executable,
        str(repo_root / "tools" / "parity" / "generate_api_inventories.py"),
        "--repo-root",
        str(repo_root),
        "--matlab-root",
        str(args.matlab_root.resolve()),
    ]
    report_cmd = [
        sys.executable,
        str(repo_root / "tools" / "parity" / "generate_gap_report.py"),
        "--repo-root",
        str(repo_root),
        "--fail-on",
        args.fail_on,
    ]
    probe_cmd = [
        sys.executable,
        str(repo_root / "tools" / "parity" / "generate_method_probe_report.py"),
        "--repo-root",
        str(repo_root),
    ]
    audit_cmd = [
        sys.executable,
        str(repo_root / "tools" / "parity" / "generate_equivalence_audit.py"),
        "--repo-root",
        str(repo_root),
        "--matlab-root",
        str(args.matlab_root.resolve()),
    ]

    subprocess.run(inventory_cmd, check=True)
    result = subprocess.run(report_cmd, check=False)
    subprocess.run(probe_cmd, check=True)
    subprocess.run(audit_cmd, check=True)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
