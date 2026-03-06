#!/usr/bin/env python3
"""Hydrate pointer-backed MATLAB helpfile assets from a materialized checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab-repo", type=Path, required=True, help="Pointer-backed MATLAB repo checkout.")
    parser.add_argument(
        "--source-help-root",
        type=Path,
        default=None,
        help=(
            "Materialized helpfiles root or MATLAB repo root. "
            "Defaults to $NSTAT_MATLAB_HELPFILES_SOURCE when set."
        ),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("output/matlab_help_images/hydrate_helpfiles_report.json"),
        help="JSON report path.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report actions without copying files.")
    return parser.parse_args()


def _normalize_help_root(path: Path | None) -> Path | None:
    if path is None:
        return None
    candidate = path.expanduser().resolve()
    help_dir = candidate / "helpfiles"
    if help_dir.is_dir():
        return help_dir
    return candidate


def _default_source_help_root() -> Path | None:
    env_path = os.environ.get("NSTAT_MATLAB_HELPFILES_SOURCE", "").strip()
    if env_path:
        return _normalize_help_root(Path(env_path))
    return None


def _read_pointer_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""


def _parse_pointer_metadata(path: Path) -> tuple[str, int] | None:
    text = _read_pointer_text(path)
    if not text.startswith(POINTER_PREFIX):
        return None
    oid = ""
    size = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("oid sha256:"):
            oid = line.split("oid sha256:", 1)[1].strip()
        elif line.startswith("size "):
            try:
                size = int(line.split("size ", 1)[1].strip())
            except ValueError:
                size = 0
    if not oid:
        return None
    return oid, size


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_pointer_files(help_root: Path) -> list[Path]:
    return sorted(path for path in help_root.glob("*.mat") if _parse_pointer_metadata(path) is not None)


def main() -> int:
    args = parse_args()
    matlab_repo = args.matlab_repo.expanduser().resolve()
    target_help_root = _normalize_help_root(matlab_repo / "helpfiles")
    if target_help_root is None or not target_help_root.exists():
        raise FileNotFoundError(f"Missing target helpfiles directory: {matlab_repo / 'helpfiles'}")

    source_help_root = _normalize_help_root(args.source_help_root) if args.source_help_root else _default_source_help_root()
    pointer_files = _iter_pointer_files(target_help_root)

    results: list[dict[str, object]] = []
    failures: list[str] = []
    for pointer_path in pointer_files:
        rel_path = pointer_path.relative_to(target_help_root)
        oid, expected_size = _parse_pointer_metadata(pointer_path) or ("", 0)
        row = {
            "relative_path": str(rel_path),
            "target_path": str(pointer_path),
            "source_path": "",
            "pointer_oid": oid,
            "pointer_size": expected_size,
            "status": "missing_source_root",
            "error": "",
        }

        if source_help_root is None:
            row["error"] = (
                "Set --source-help-root or NSTAT_MATLAB_HELPFILES_SOURCE to a materialized MATLAB helpfiles directory."
            )
            failures.append(f"{rel_path}: {row['error']}")
            results.append(row)
            continue

        if not source_help_root.exists():
            row["source_path"] = str(source_help_root)
            row["error"] = f"Missing source helpfiles directory: {source_help_root}"
            failures.append(f"{rel_path}: {row['error']}")
            results.append(row)
            continue

        source_path = source_help_root / rel_path.name
        row["source_path"] = str(source_path)
        if not source_path.exists():
            row["status"] = "missing_source_file"
            row["error"] = f"Missing materialized source asset: {source_path}"
            failures.append(f"{rel_path}: {row['error']}")
            results.append(row)
            continue

        if _parse_pointer_metadata(source_path) is not None:
            row["status"] = "source_is_pointer"
            row["error"] = f"Source asset is also an LFS pointer: {source_path}"
            failures.append(f"{rel_path}: {row['error']}")
            results.append(row)
            continue

        actual_oid = _sha256(source_path)
        if actual_oid != oid:
            row["status"] = "hash_mismatch"
            row["error"] = f"SHA256 mismatch for {source_path}: expected {oid}, found {actual_oid}"
            failures.append(f"{rel_path}: {row['error']}")
            results.append(row)
            continue

        if not args.dry_run:
            shutil.copy2(source_path, pointer_path)
        row["status"] = "hydrated" if not args.dry_run else "would_hydrate"
        results.append(row)

    payload = {
        "matlab_repo": str(matlab_repo),
        "target_help_root": str(target_help_root),
        "source_help_root": str(source_help_root) if source_help_root else "",
        "pointer_count": len(pointer_files),
        "results": results,
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.report_json)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
