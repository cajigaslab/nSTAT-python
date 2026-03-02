"""Dataset discovery and download helpers.

Only example data may be shared with the MATLAB nSTAT repository. This module
keeps the policy explicit via checksummed manifests.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve


@dataclass(slots=True)
class DatasetRecord:
    name: str
    version: str
    url: str
    sha256: str
    filename: str


def _manifest_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "datasets_manifest.json"


def _repo_data_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def _local_mirror_roots() -> list[Path]:
    """Return available local mirrored MATLAB dataset roots."""

    shared_root = _repo_data_root() / "shared"
    if not shared_root.exists():
        return []
    roots = [path for path in shared_root.glob("matlab_gold_*") if path.is_dir()]
    roots.sort(key=lambda path: path.name, reverse=True)
    return roots


def get_example_data_root() -> Path | None:
    """Resolve preferred root for example datasets.

    Resolution order:
    1. ``NSTAT_DATA_ROOT`` environment variable (if set and exists).
    2. Latest local mirrored MATLAB dataset in ``data/shared``.
    """

    explicit = os.environ.get("NSTAT_DATA_ROOT")
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if root.exists() and root.is_dir():
            return root

    mirrors = _local_mirror_roots()
    if mirrors:
        return mirrors[0]
    return None


def _load_manifest() -> list[DatasetRecord]:
    data = json.loads(_manifest_path().read_text(encoding="utf-8"))
    return [DatasetRecord(**row) for row in data["datasets"]]


def list_datasets() -> list[str]:
    """Return available dataset names from manifest."""

    return [row.name for row in _load_manifest()]


def get_cache_dir() -> Path:
    """Return dataset cache directory."""

    root = Path(os.environ.get("NSTAT_DATA_CACHE", Path.home() / ".cache" / "nstat" / "data"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_dataset(name: str, version: str | None = None) -> Path:
    """Download dataset artifact and verify checksum."""

    rows = [r for r in _load_manifest() if r.name == name and (version is None or r.version == version)]
    if not rows:
        raise KeyError(f"dataset not found: {name} version={version!r}")

    record = sorted(rows, key=lambda r: r.version)[-1]
    example_root = get_example_data_root()
    if example_root is not None:
        local_candidate = example_root / record.filename
        if local_candidate.exists():
            digest = _sha256(local_candidate)
            if digest == record.sha256:
                return local_candidate

    out = get_cache_dir() / record.filename
    out.parent.mkdir(parents=True, exist_ok=True)

    if not out.exists():
        urlretrieve(record.url, out)

    digest = _sha256(out)
    if digest != record.sha256:
        raise RuntimeError(f"checksum mismatch for {record.filename}: {digest} != {record.sha256}")

    return out
