"""Example data directory resolution and DOI-backed download helpers.

This module keeps raw example assets out of Git while allowing notebooks and
tests to materialize the canonical nSTAT example dataset on demand.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Final


DOI_URL: Final[str] = "https://doi.org/10.6084/m9.figshare.4834640"
DEFAULT_RELATIVE_CACHE: Final[Path] = Path("data_cache") / "nstat_data"
SENTINEL_NAME: Final[str] = ".nstat_data_ok.json"
REQUIRED_SUBDIRS: Final[tuple[str, ...]] = (
    "Explicit Stimulus",
    "Place Cells",
    "mEPSCs",
)
DOWNLOAD_URL_RE: Final[re.Pattern[str]] = re.compile(
    r"https?://(?:www\.)?figshare\.com/ndownloader/files/\d+"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    """Return canonical on-disk example-data directory.

    Resolution order:
    1. ``NSTAT_DATA_DIR`` environment variable.
    2. ``<repo>/data_cache/nstat_data``
    """

    explicit = os.environ.get("NSTAT_DATA_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (_repo_root() / DEFAULT_RELATIVE_CACHE).resolve()


def data_is_present(data_dir: Path) -> bool:
    """Return True when the expected dataset footprint exists."""

    if not data_dir.exists() or not data_dir.is_dir():
        return False
    for subdir in REQUIRED_SUBDIRS:
        if not (data_dir / subdir).exists():
            return False
    return True


def _write_sentinel(data_dir: Path, *, source_url: str) -> None:
    payload = {
        "doi": DOI_URL,
        "source_url": source_url,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (data_dir / SENTINEL_NAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _http_get(url: str, *, timeout: float = 60.0) -> tuple[str, bytes]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "nSTAT-python-data-manager/1.0 (+https://github.com/cajigaslab/nSTAT-python)"
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        final_url = str(resp.geturl())
        body = resp.read()
    return final_url, body


def _resolve_figshare_download_url() -> str:
    final_url, body = _http_get(DOI_URL)
    if DOWNLOAD_URL_RE.search(final_url):
        return final_url
    html = body.decode("utf-8", errors="ignore")
    match = DOWNLOAD_URL_RE.search(html)
    if match:
        return match.group(0)
    raise RuntimeError(
        f"Could not resolve figshare download URL from DOI landing page: {DOI_URL}"
    )


def _stream_download(url: str, destination: Path, *, retries: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "nSTAT-python-data-manager/1.0 (+https://github.com/cajigaslab/nSTAT-python)"
                },
            )
            with urllib.request.urlopen(req, timeout=120.0) as resp, destination.open("wb") as out:
                shutil.copyfileobj(resp, out, length=1024 * 1024)
            return
        except Exception as exc:  # pragma: no cover - network timing dependent
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed to download dataset from {url}") from last_error


def _find_dataset_root(extracted_root: Path) -> Path:
    if data_is_present(extracted_root):
        return extracted_root
    for candidate in extracted_root.rglob("*"):
        if candidate.is_dir() and data_is_present(candidate):
            return candidate
    raise RuntimeError(
        "Downloaded archive did not contain expected nSTAT data folders: "
        + ", ".join(REQUIRED_SUBDIRS)
    )


def _atomic_replace_tree(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f"{destination.name}.bak")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.rename(backup)
    try:
        source.rename(destination)
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        if backup.exists():
            backup.rename(destination)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup)


def ensure_example_data(download: bool = True) -> Path:
    """Ensure the canonical example data exists locally and return its path."""

    data_dir = get_data_dir()
    if data_is_present(data_dir):
        if not (data_dir / SENTINEL_NAME).exists():
            _write_sentinel(data_dir, source_url="local-existing")
        return data_dir

    if not download:
        raise FileNotFoundError(
            f"Example data not found at {data_dir}. "
            "Set NSTAT_DATA_DIR or call ensure_example_data(download=True)."
        )

    # Download to a temp workspace first so partial failures do not pollute
    # the final cache path.
    download_tmp_root = (_repo_root() / "output" / "data_download").resolve()
    download_tmp_root.mkdir(parents=True, exist_ok=True)
    work_root = Path(tempfile.mkdtemp(prefix="nstat_data_", dir=str(download_tmp_root)))
    try:
        archive_path = work_root / "nstat_example_data.zip"
        download_url = _resolve_figshare_download_url()
        _stream_download(download_url, archive_path)
        if not zipfile.is_zipfile(archive_path):
            raise RuntimeError(f"Downloaded file is not a valid zip archive: {archive_path}")
        extracted_root = work_root / "extracted"
        extracted_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extracted_root)
        dataset_root = _find_dataset_root(extracted_root)
        staged = work_root / "staged_data"
        shutil.copytree(dataset_root, staged)
        _atomic_replace_tree(staged, data_dir)
        _write_sentinel(data_dir, source_url=download_url)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    return data_dir
