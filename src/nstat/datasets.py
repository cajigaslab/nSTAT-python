"""Dataset discovery and download helpers.

Only example data may be shared with the MATLAB nSTAT repository. This module
keeps the policy explicit via checksummed manifests.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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


MIRROR_NAME_RE = re.compile(r"^matlab_gold_(\d{8})/(.+)$")


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


def list_matlab_gold_files(version: str | None = None) -> list[str]:
    """List mirrored MATLAB files available in datasets manifest.

    Parameters
    ----------
    version:
        Optional mirror version label (for example ``"20260302"``).
        When omitted, rows across all mirrored versions are returned.
    """

    files: list[str] = []
    for row in _load_manifest():
        match = MIRROR_NAME_RE.match(row.name)
        if not match:
            continue
        row_version = match.group(1)
        rel_path = match.group(2)
        if version is None or version == row_version:
            files.append(rel_path)
    return sorted(set(files))


def latest_matlab_gold_version() -> str | None:
    """Return latest mirrored MATLAB version known by datasets manifest."""

    versions: set[str] = set()
    for row in _load_manifest():
        match = MIRROR_NAME_RE.match(row.name)
        if match:
            versions.add(match.group(1))
    if not versions:
        return None
    return sorted(versions)[-1]


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


def fetch_matlab_gold_file(relative_path: str, version: str | None = None) -> Path:
    """Fetch one file from mirrored MATLAB data via the datasets API.

    Parameters
    ----------
    relative_path:
        File path relative to MATLAB ``data/`` root, for example
        ``"PlaceCellAnimal1Results.mat"``.
    version:
        Optional mirror version (default: latest available).
    """

    selected_version = version or latest_matlab_gold_version()
    if selected_version is None:
        raise KeyError("No mirrored MATLAB dataset version found in datasets manifest.")
    name = f"matlab_gold_{selected_version}/{relative_path}"
    return fetch_dataset(name=name, version=selected_version)
