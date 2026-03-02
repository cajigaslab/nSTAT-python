#!/usr/bin/env python3
"""Run the full MATLAB-data mirror workflow in one command.

Steps:
1. Build source snapshot manifest.
2. Sync source data into mirrored tree.
3. Regenerate shared-data allowlist entries.
4. Verify mirrored tree against checksum manifest.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Path to MATLAB data directory.")
    parser.add_argument("--version", required=True, help="Dataset version label (for example 20260302).")
    parser.add_argument(
        "--dest-root",
        default="data/shared",
        help="Destination root for mirrored data (default: data/shared).",
    )
    parser.add_argument(
        "--allowlist",
        default="tools/compliance/shared_data_allowlist.yml",
        help="Shared-data allowlist path.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing mirrored dataset directory before sync.",
    )
    return parser.parse_args()


def _run(cmd: list[str], repo_root: Path) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    source_manifest = Path(args.dest_root) / f"matlab_source_{args.version}.manifest.json"
    mirror_manifest = Path(args.dest_root) / f"matlab_gold_{args.version}.manifest.json"

    _run(
        [
            "python",
            "tools/data_mirror/build_manifest.py",
            "--source-root",
            args.source_root,
            "--version",
            args.version,
            "--out",
            source_manifest.as_posix(),
        ],
        repo_root,
    )

    sync_cmd = [
        "python",
        "tools/data_mirror/sync_matlab_data.py",
        "--source-root",
        args.source_root,
        "--version",
        args.version,
        "--dest-root",
        args.dest_root,
    ]
    if args.clean:
        sync_cmd.append("--clean")
    _run(sync_cmd, repo_root)

    _run(
        [
            "python",
            "tools/compliance/update_shared_data_allowlist.py",
            "--manifest",
            mirror_manifest.as_posix(),
            "--allowlist",
            args.allowlist,
        ],
        repo_root,
    )

    _run(
        [
            "python",
            "tools/data_mirror/verify_matlab_data.py",
            "--manifest",
            mirror_manifest.as_posix(),
            "--strict",
        ],
        repo_root,
    )

    print("Mirror workflow completed successfully.")
    print(f"Source manifest: {source_manifest}")
    print(f"Mirror manifest: {mirror_manifest}")
    print(f"Allowlist: {args.allowlist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

