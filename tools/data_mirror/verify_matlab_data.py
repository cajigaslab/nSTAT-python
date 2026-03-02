#!/usr/bin/env python3
"""Verify mirrored MATLAB example data against a checksum manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import load_manifest, repo_root_from_tools_script, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to checksum manifest JSON.")
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When true, also fail on extra files in mirror root.",
    )
    parser.add_argument("--report-out", default=None, help="Optional JSON report output path.")
    return parser.parse_args()


def _resolve_path(path_arg: str, repo_root: Path) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_tools_script(Path(__file__).resolve())
    manifest_path = _resolve_path(args.manifest, repo_root)
    manifest = load_manifest(manifest_path)

    mirror_root_raw = manifest.get("mirror_root")
    if not mirror_root_raw:
        raise KeyError("Manifest is missing required key: mirror_root")

    mirror_root = _resolve_path(str(mirror_root_raw), repo_root)
    files = manifest.get("files", [])

    missing: list[str] = []
    size_mismatch: list[str] = []
    hash_mismatch: list[str] = []

    expected = set()
    for row in files:
        rel_path = row["relative_path"]
        expected.add(rel_path)
        target = mirror_root / rel_path
        if not target.exists():
            missing.append(rel_path)
            continue
        if target.stat().st_size != int(row["size_bytes"]):
            size_mismatch.append(rel_path)
            continue
        digest = sha256_file(target)
        if digest != row["sha256"]:
            hash_mismatch.append(rel_path)

    extra: list[str] = []
    if args.strict and mirror_root.exists():
        observed = {
            path.relative_to(mirror_root).as_posix()
            for path in mirror_root.rglob("*")
            if path.is_file()
        }
        extra = sorted(observed - expected)

    report = {
        "manifest": str(manifest_path),
        "mirror_root": str(mirror_root),
        "strict": args.strict,
        "expected_file_count": len(files),
        "missing_count": len(missing),
        "size_mismatch_count": len(size_mismatch),
        "hash_mismatch_count": len(hash_mismatch),
        "extra_count": len(extra),
        "missing": missing,
        "size_mismatch": size_mismatch,
        "hash_mismatch": hash_mismatch,
        "extra": extra,
    }

    if args.report_out:
        out_path = _resolve_path(args.report_out, repo_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Manifest: {manifest_path}")
    print(f"Mirror root: {mirror_root}")
    print(f"Expected files: {len(files)}")
    print(f"Missing: {len(missing)}")
    print(f"Size mismatches: {len(size_mismatch)}")
    print(f"Hash mismatches: {len(hash_mismatch)}")
    print(f"Extra files: {len(extra)}")

    has_errors = bool(missing or size_mismatch or hash_mismatch or extra)
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

