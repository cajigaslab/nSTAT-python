#!/usr/bin/env python3
"""Build a frozen checksum manifest for a MATLAB data directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import build_inventory, manifest_dict, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Path to MATLAB data directory.")
    parser.add_argument("--version", required=True, help="Dataset version label (for example 20260302).")
    parser.add_argument("--out", required=True, help="Manifest output path (JSON).")
    parser.add_argument(
        "--mirror-root",
        default=None,
        help="Optional mirror root path to store in manifest (relative or absolute).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    entries = build_inventory(source_root)
    payload = manifest_dict(
        source_root=source_root,
        mirror_root=args.mirror_root,
        entries=entries,
        version=args.version,
        generated_by="tools/data_mirror/build_manifest.py",
    )
    write_manifest(out, payload)

    print(f"Wrote manifest: {out}")
    print(f"Source root: {source_root}")
    print(f"Files: {payload['file_count']}")
    print(f"Total bytes: {payload['total_size_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

