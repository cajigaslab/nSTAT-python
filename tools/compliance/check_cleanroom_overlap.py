#!/usr/bin/env python3
"""Fail if non-data files overlap by SHA256 with the MATLAB nSTAT repository.

This script enforces clean-room constraints for nSTAT-python:
- non-data content hashes must not collide with upstream MATLAB nSTAT
- data file overlaps are allowed only when explicitly allowlisted
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


IGNORED_DIRS = {
    ".git",
    ".github",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "site",
    "docs/_build",
}


@dataclass(frozen=True, slots=True)
class FileDigest:
    relative_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class AllowedDataMatch:
    python_path: str
    upstream_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class Collision:
    sha256: str
    python_path: str
    upstream_path: str



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Path to nSTAT-python repository root.",
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        required=True,
        help="Path to checked-out MATLAB nSTAT repository root.",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=Path(__file__).resolve().parent / "shared_data_allowlist.yml",
        help="Path to shared-data allowlist YAML.",
    )
    return parser.parse_args()



def is_ignored(path: Path) -> bool:
    """Return True if path should be excluded from hash comparison."""

    path_str = path.as_posix()
    for ignored in IGNORED_DIRS:
        if ignored in path.parts or path_str.startswith(f"{ignored}/"):
            return True
    return False



def is_data_path(relative_path: str) -> bool:
    """Return True when path is inside top-level data directory."""

    return relative_path.startswith("data/")



def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if is_ignored(rel):
            continue
        yield path



def build_digest_index(root: Path) -> list[FileDigest]:
    rows: list[FileDigest] = []
    for path in iter_files(root):
        rel = path.relative_to(root).as_posix()
        rows.append(FileDigest(relative_path=rel, sha256=compute_sha256(path)))
    return rows



def load_allowlist(path: Path) -> set[AllowedDataMatch]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    shared_data = payload.get("shared_data") if isinstance(payload, dict) else []
    if shared_data is None:
        shared_data = []

    allowed: set[AllowedDataMatch] = set()
    for row in shared_data:
        if not isinstance(row, dict):
            continue
        python_path = row.get("python_path")
        upstream_path = row.get("upstream_path")
        sha256 = row.get("sha256")
        if python_path and upstream_path and sha256:
            allowed.add(
                AllowedDataMatch(
                    python_path=str(python_path),
                    upstream_path=str(upstream_path),
                    sha256=str(sha256),
                )
            )
    return allowed



def find_collisions(
    python_files: list[FileDigest],
    upstream_files: list[FileDigest],
    allowlist: set[AllowedDataMatch],
) -> list[Collision]:
    upstream_by_hash: dict[str, list[str]] = {}
    for row in upstream_files:
        upstream_by_hash.setdefault(row.sha256, []).append(row.relative_path)

    collisions: list[Collision] = []
    for py_row in python_files:
        upstream_paths = upstream_by_hash.get(py_row.sha256, [])
        if not upstream_paths:
            continue

        for upstream_path in upstream_paths:
            both_data = is_data_path(py_row.relative_path) and is_data_path(upstream_path)
            if both_data:
                allowed = AllowedDataMatch(
                    python_path=py_row.relative_path,
                    upstream_path=upstream_path,
                    sha256=py_row.sha256,
                )
                if allowed in allowlist:
                    continue

            collisions.append(
                Collision(
                    sha256=py_row.sha256,
                    python_path=py_row.relative_path,
                    upstream_path=upstream_path,
                )
            )
    return collisions



def main() -> int:
    args = parse_args()

    if not args.project_root.exists():
        raise FileNotFoundError(f"project root does not exist: {args.project_root}")
    if not args.upstream_root.exists():
        raise FileNotFoundError(f"upstream root does not exist: {args.upstream_root}")
    if not args.allowlist.exists():
        raise FileNotFoundError(f"allowlist does not exist: {args.allowlist}")

    python_files = build_digest_index(args.project_root)
    upstream_files = build_digest_index(args.upstream_root)
    allowlist = load_allowlist(args.allowlist)

    collisions = find_collisions(python_files, upstream_files, allowlist)
    if collisions:
        print("Clean-room compliance check FAILED.")
        print("Detected overlapping file content hashes:")
        for row in collisions:
            print(
                f"  sha256={row.sha256} | python={row.python_path} | upstream={row.upstream_path}"
            )
        return 1

    print("Clean-room compliance check PASSED (no unauthorized hash collisions).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
