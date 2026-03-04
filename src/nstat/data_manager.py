"""Example data management for nSTAT notebooks and validation workflows.

This module keeps raw example data out of Git while allowing deterministic
local/CI setup via an on-demand DOI download cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DOI_URL = "https://doi.org/10.6084/m9.figshare.4834640"
FIGSHARE_ARTICLE_FALLBACK = "4834640"
SENTINEL_FILENAME = ".nstat_data_ok.json"
EXPECTED_SUBDIRS = ("mEPSCs", "Explicit Stimulus", "PSTH", "Place Cells")


@dataclass(frozen=True)
class DownloadTarget:
    url: str
    filename: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    """Return the resolved example-data root.

    Resolution order:
    1. ``NSTAT_DATA_DIR`` environment variable.
    2. ``<repo_root>/data_cache/nstat_data``.
    """

    env = os.environ.get("NSTAT_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (_repo_root() / "data_cache" / "nstat_data").resolve()


def _sentinel_path(data_dir: Path) -> Path:
    return data_dir / SENTINEL_FILENAME


def _expected_paths(data_dir: Path) -> list[Path]:
    return [data_dir / name for name in EXPECTED_SUBDIRS]


def data_is_present(data_dir: Path) -> bool:
    """Return True when data dir has expected structure + sentinel."""

    if not data_dir.exists() or not data_dir.is_dir():
        return False
    if not _sentinel_path(data_dir).exists():
        return False
    return all(path.exists() and path.is_dir() for path in _expected_paths(data_dir))


def _build_session() -> requests.Session:
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "nSTAT-python-data-manager/1.0"})
    return session


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_article_ids(url: str) -> list[str]:
    ids = re.findall(r"/articles/(?:dataset|media)/[^/]+/(\d+)", url)
    if not ids:
        ids = re.findall(r"/(\d+)(?:[/?#]|$)", url)
    unique: list[str] = []
    for raw in ids:
        if raw not in unique:
            unique.append(raw)
    if FIGSHARE_ARTICLE_FALLBACK not in unique:
        unique.append(FIGSHARE_ARTICLE_FALLBACK)
    return unique


def _pick_download_target_from_article(session: requests.Session, article_id: str) -> DownloadTarget | None:
    api = f"https://api.figshare.com/v2/articles/{article_id}"
    resp = session.get(api, timeout=60)
    if resp.status_code != 200:
        return None
    payload = resp.json()
    files = payload.get("files", [])
    if not isinstance(files, list) or not files:
        return None

    def sort_key(item: dict) -> tuple[int, int]:
        name = str(item.get("name", "")).lower()
        size = int(item.get("size", 0) or 0)
        zip_pref = 0 if name.endswith(".zip") else 1
        return (zip_pref, -size)

    ordered = sorted((f for f in files if isinstance(f, dict)), key=sort_key)
    for item in ordered:
        url = str(item.get("download_url", "")).strip()
        name = str(item.get("name", "")).strip() or "figshare_data.zip"
        if url:
            return DownloadTarget(url=url, filename=name)
    return None


def _resolve_download_target(session: requests.Session) -> DownloadTarget:
    doi_resp = session.get(DOI_URL, allow_redirects=True, timeout=60)
    doi_resp.raise_for_status()
    final_url = doi_resp.url

    for article_id in _extract_article_ids(final_url):
        target = _pick_download_target_from_article(session, article_id)
        if target is not None:
            return target

    html = doi_resp.text
    links = re.findall(r"https://ndownloader\.figshare\.com/files/\d+", html)
    if links:
        url = links[0]
        return DownloadTarget(url=url, filename=f"figshare_{FIGSHARE_ARTICLE_FALLBACK}.zip")

    raise RuntimeError(f"Could not resolve figshare download target from DOI redirect: {final_url}")


def _download_streaming(session: requests.Session, target: DownloadTarget, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(target.url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _contains_expected_structure(path: Path) -> bool:
    return all((path / name).is_dir() for name in EXPECTED_SUBDIRS)


def _find_extracted_root(extract_dir: Path) -> Path:
    """Find dataset root inside extracted archive.

    Figshare archives sometimes wrap data as:
    - <root>/<EXPECTED_SUBDIRS...>
    - <root>/data/<EXPECTED_SUBDIRS...>
    - <root>/<single-dir>/data/<EXPECTED_SUBDIRS...>
    """

    direct_candidates: list[Path] = [extract_dir]
    if extract_dir.is_dir():
        direct_candidates.extend([p for p in extract_dir.iterdir() if p.is_dir() and p.name != "__MACOSX"])

    for candidate in direct_candidates:
        if _contains_expected_structure(candidate):
            return candidate
        nested_data = candidate / "data"
        if _contains_expected_structure(nested_data):
            return nested_data

    for candidate in extract_dir.rglob("*"):
        if not candidate.is_dir() or candidate.name == "__MACOSX":
            continue
        if _contains_expected_structure(candidate):
            return candidate
        nested_data = candidate / "data"
        if _contains_expected_structure(nested_data):
            return nested_data

    return extract_dir


def _ensure_expected_subdirs(data_dir: Path) -> None:
    missing = [str(path) for path in _expected_paths(data_dir) if not path.exists()]
    if missing:
        raise RuntimeError(
            "Downloaded archive does not contain expected nSTAT example-data structure. "
            f"Missing: {', '.join(missing)}"
        )


def _write_sentinel(data_dir: Path, source_url: str, archive_sha256: str) -> None:
    payload = {
        "doi": DOI_URL,
        "source_url": source_url,
        "archive_sha256": archive_sha256,
        "downloaded_at_utc": datetime.now(tz=UTC).isoformat(),
    }
    _sentinel_path(data_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _atomic_replace_dir(src_dir: Path, dst_dir: Path) -> None:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_dir), str(dst_dir))


def ensure_example_data(download: bool = True) -> Path:
    """Ensure example data are available and return data root.

    Parameters
    ----------
    download:
        When ``True`` (default), download + extract from DOI if missing.
        When ``False``, raise ``FileNotFoundError`` if data are absent.
    """

    data_dir = get_data_dir()
    if data_is_present(data_dir):
        return data_dir

    if not download:
        raise FileNotFoundError(
            f"nSTAT example data not found at {data_dir}. "
            "Set NSTAT_DATA_DIR to an existing dataset root or run with download=True."
        )

    session = _build_session()
    target = _resolve_download_target(session)
    temp_parent = (_repo_root() / "output" / "data_download").resolve()
    temp_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nstat_data_", dir=temp_parent) as tmp_raw:
        tmp_root = Path(tmp_raw)
        archive_path = tmp_root / target.filename
        _download_streaming(session, target, archive_path)

        archive_sha = _sha256(archive_path)
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                bad_entry = zf.testzip()
                if bad_entry is not None:
                    raise RuntimeError(f"Archive integrity check failed at member: {bad_entry}")
                extract_dir = tmp_root / "extract"
                extract_dir.mkdir(parents=True, exist_ok=True)
                zf.extractall(extract_dir)
        except zipfile.BadZipFile as exc:
            raise RuntimeError(f"Downloaded file is not a valid zip archive: {archive_path}") from exc

        extracted_root = _find_extracted_root(tmp_root / "extract")
        _ensure_expected_subdirs(extracted_root)
        _write_sentinel(extracted_root, target.url, archive_sha)
        _atomic_replace_dir(extracted_root, data_dir)

    if not data_is_present(data_dir):
        raise RuntimeError(f"Example data validation failed after download at {data_dir}")

    return data_dir


def iter_data_paths_from_matlab_line(matlab_line: str) -> Iterable[str]:
    for match in re.finditer(r"['\"]([^'\"]*data/[^'\"]+)['\"]", matlab_line, flags=re.IGNORECASE):
        raw = match.group(1)
        marker = raw.lower().find("data/")
        rel = raw[marker + len("data/") :] if marker >= 0 else raw
        rel = rel.lstrip("./")
        rel = rel.replace("\\", "/")
        if rel:
            yield rel
