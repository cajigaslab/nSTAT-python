"""Shared helpers for mirroring MATLAB example data into nSTAT-python."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

FILE_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class FileEntry:
    """Immutable manifest row for one file in a dataset tree."""

    relative_path: str
    size_bytes: int
    mtime_epoch_s: float
    sha256: str


def sha256_file(path: Path) -> str:
    """Compute SHA256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(FILE_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path) -> list[Path]:
    """Return all files under root in deterministic order."""

    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file())


def build_inventory(root: Path) -> list[FileEntry]:
    """Create full inventory with metadata and checksums."""

    entries: list[FileEntry] = []
    for path in iter_files(root):
        stat = path.stat()
        entries.append(
            FileEntry(
                relative_path=path.relative_to(root).as_posix(),
                size_bytes=stat.st_size,
                mtime_epoch_s=stat.st_mtime,
                sha256=sha256_file(path),
            )
        )
    return entries


def manifest_dict(
    *,
    source_root: Path,
    mirror_root: str | None,
    entries: list[FileEntry],
    version: str,
    generated_by: str,
) -> dict:
    """Build manifest payload."""

    total_size_bytes = sum(entry.size_bytes for entry in entries)
    payload = {
        "schema_version": 1,
        "dataset_name": "matlab_example_data",
        "dataset_version": version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": generated_by,
        "source_root": str(source_root),
        "file_count": len(entries),
        "total_size_bytes": total_size_bytes,
        "files": [
            {
                "relative_path": entry.relative_path,
                "size_bytes": entry.size_bytes,
                "mtime_epoch_s": entry.mtime_epoch_s,
                "sha256": entry.sha256,
            }
            for entry in entries
        ],
    }
    if mirror_root is not None:
        payload["mirror_root"] = mirror_root
    return payload


def write_manifest(path: Path, payload: dict) -> None:
    """Write JSON manifest with stable pretty formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> dict:
    """Load and parse manifest JSON."""

    return json.loads(path.read_text(encoding="utf-8"))


def repo_root_from_tools_script(script_path: Path) -> Path:
    """Resolve repository root from a script located under tools/*."""

    return script_path.resolve().parents[2]

