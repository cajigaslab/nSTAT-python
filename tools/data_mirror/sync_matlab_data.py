#!/usr/bin/env python3
"""Mirror MATLAB example data into nSTAT-python and verify exact checksums."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from _common import (
    build_inventory,
    manifest_dict,
    repo_root_from_tools_script,
    sha256_file,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Path to MATLAB data directory.")
    parser.add_argument("--version", required=True, help="Dataset version label (for example 20260302).")
    parser.add_argument(
        "--dest-root",
        default="data/shared",
        help="Destination root (default: data/shared, relative to repo root).",
    )
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="Optional manifest output path (default: <dest-root>/matlab_gold_<version>.manifest.json).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete destination mirror root before syncing.",
    )
    return parser.parse_args()


def _resolve_dest_root(dest_root_arg: str, repo_root: Path) -> Path:
    dest_root = Path(dest_root_arg).expanduser()
    if not dest_root.is_absolute():
        dest_root = repo_root / dest_root
    return dest_root.resolve()


def _remove_empty_dirs(root: Path) -> None:
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        if not any(path.iterdir()):
            path.rmdir()


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = repo_root_from_tools_script(script_path)
    source_root = Path(args.source_root).expanduser().resolve()
    dest_root = _resolve_dest_root(args.dest_root, repo_root)

    dataset_dirname = f"matlab_gold_{args.version}"
    mirror_root = dest_root / dataset_dirname

    if args.clean and mirror_root.exists():
        shutil.rmtree(mirror_root)

    entries = build_inventory(source_root)
    expected_paths = {entry.relative_path for entry in entries}

    copied = 0
    skipped = 0
    for entry in entries:
        src = source_root / entry.relative_path
        dst = mirror_root / entry.relative_path
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and sha256_file(dst) == entry.sha256:
            skipped += 1
            continue

        shutil.copy2(src, dst)
        copied += 1

    existing_paths = {
        path.relative_to(mirror_root).as_posix()
        for path in mirror_root.rglob("*")
        if path.is_file()
    }
    extra_paths = sorted(existing_paths - expected_paths)
    for rel_path in extra_paths:
        (mirror_root / rel_path).unlink()
    _remove_empty_dirs(mirror_root)

    # Final checksum verification against source manifest.
    mismatches: list[str] = []
    for entry in entries:
        dst = mirror_root / entry.relative_path
        if not dst.exists():
            mismatches.append(f"missing:{entry.relative_path}")
            continue
        digest = sha256_file(dst)
        if digest != entry.sha256:
            mismatches.append(f"hash:{entry.relative_path}")

    if mismatches:
        sample = ", ".join(mismatches[:5])
        raise RuntimeError(f"Mirror verification failed ({len(mismatches)} mismatches). Sample: {sample}")

    mirror_root_str: str
    try:
        mirror_root_str = mirror_root.relative_to(repo_root).as_posix()
    except ValueError:
        mirror_root_str = str(mirror_root)

    if args.manifest_out is None:
        manifest_out = dest_root / f"{dataset_dirname}.manifest.json"
    else:
        manifest_out = Path(args.manifest_out).expanduser()
        if not manifest_out.is_absolute():
            manifest_out = (repo_root / manifest_out).resolve()

    payload = manifest_dict(
        source_root=source_root,
        mirror_root=mirror_root_str,
        entries=entries,
        version=args.version,
        generated_by="tools/data_mirror/sync_matlab_data.py",
    )
    write_manifest(manifest_out, payload)

    print(f"Repo root: {repo_root}")
    print(f"Source root: {source_root}")
    print(f"Mirror root: {mirror_root}")
    print(f"Manifest: {manifest_out}")
    print(f"Files verified: {payload['file_count']}")
    print(f"Total bytes: {payload['total_size_bytes']}")
    print(f"Copied: {copied}")
    print(f"Skipped: {skipped}")
    print(f"Pruned extra files: {len(extra_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

