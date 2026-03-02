#!/usr/bin/env python3
"""Apply branch protection with required status checks via GitHub API."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=str, required=True, help="owner/repo")
    parser.add_argument("--branch", type=str, default="main")
    parser.add_argument(
        "--required-check",
        action="append",
        default=[],
        help="Required status check context. Repeat flag for multiple checks.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require branch to be up to date before merging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print payload only; do not call GitHub API.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    checks = args.required_check or ["test-and-build", "parity-gate"]
    payload = {
        "required_status_checks": {
            "strict": bool(args.strict),
            "contexts": checks,
        },
        "enforce_admins": False,
        "required_pull_request_reviews": None,
        "restrictions": None,
    }

    print("Applying branch protection:")
    print(json.dumps(payload, indent=2))

    if args.dry_run:
        return 0

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp:
        tmp.write(json.dumps(payload))
        payload_path = Path(tmp.name)

    try:
        cmd = [
            "gh",
            "api",
            "--method",
            "PUT",
            f"repos/{args.repo}/branches/{args.branch}/protection",
            "--input",
            str(payload_path),
            "--header",
            "Accept: application/vnd.github+json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            return proc.returncode
        print("Branch protection updated successfully.")
        return 0
    finally:
        if payload_path.exists():
            payload_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
