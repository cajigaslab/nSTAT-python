"""Resolve and materialize the standalone nSTAT-python example dataset."""

from __future__ import annotations

import json
import os
import re
import shutil
import ssl
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import certifi


FIGSHARE_API_URL: Final[str] = "https://api.figshare.com/v2/articles/4834640"
FIGSHARE_DOI_URL: Final[str] = "https://doi.org/10.6084/m9.figshare.4834640.v3"
PAPER_DOI_URL: Final[str] = "https://doi.org/10.1016/j.jneumeth.2012.08.009"
SENTINEL_NAME: Final[str] = ".nstat_data_ok.json"
USER_AGENT: Final[str] = "nSTAT-python-data-manager/1.0 (+https://github.com/cajigaslab/nSTAT-python)"
SSL_CONTEXT: Final[ssl.SSLContext] = ssl.create_default_context(cafile=certifi.where())
DOWNLOAD_URL_RE: Final[re.Pattern[str]] = re.compile(
    r"https?://(?:www\.)?(?:ndownloader|figshare\.com/ndownloader)/files/\d+"
)


@dataclass(frozen=True)
class ExampleDataInfo:
    """Resolved on-disk metadata for the canonical example dataset."""

    root_dir: Path
    data_dir: Path
    figshare_api_url: str
    figshare_doi: str
    paper_doi: str
    required_files: tuple[Path, ...]

    @property
    def is_installed(self) -> bool:
        return all(path.exists() for path in self.required_files)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_cache_dir() -> Path:
    env = os.environ.get("NSTAT_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (_repo_root() / "data_cache" / "nstat_data").resolve()


def get_example_data_info(
    root_dir: str | Path | None = None,
    *,
    treat_as_data_dir: bool = False,
) -> ExampleDataInfo:
    """Return dataset metadata for a repo root or explicit dataset cache path."""

    raw_root = _repo_root() if root_dir is None else Path(root_dir).expanduser().resolve()
    if treat_as_data_dir or (raw_root / "mEPSCs").exists() or raw_root.name == "data":
        data_dir = raw_root
        root = raw_root.parent if raw_root.name == "data" else raw_root
    else:
        root = raw_root
        data_dir = root / "data"
    required = (
        data_dir / "mEPSCs" / "epsc2.txt",
        data_dir / "Explicit Stimulus" / "Dir3" / "Neuron1" / "Stim2" / "trngdataBis.mat",
        data_dir / "PSTH" / "Results.mat",
        data_dir / "Place Cells" / "PlaceCellDataAnimal1.mat",
        data_dir / "PlaceCellAnimal1Results.mat",
        data_dir / "SSGLMExampleData.mat",
    )
    return ExampleDataInfo(
        root_dir=root,
        data_dir=data_dir,
        figshare_api_url=FIGSHARE_API_URL,
        figshare_doi=FIGSHARE_DOI_URL,
        paper_doi=PAPER_DOI_URL,
        required_files=required,
    )


def _write_sentinel(data_dir: Path, *, source_url: str) -> None:
    payload = {
        "figshare_doi": FIGSHARE_DOI_URL,
        "paper_doi": PAPER_DOI_URL,
        "source_url": source_url,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (data_dir / SENTINEL_NAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _http_get(url: str, *, timeout: float = 60.0) -> tuple[str, bytes]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT) as resp:
        final_url = str(resp.geturl())
        body = resp.read()
    return final_url, body


def _resolve_figshare_download_url() -> str:
    _, body = _http_get(FIGSHARE_API_URL)
    payload = json.loads(body.decode("utf-8"))
    files = payload.get("files", [])
    if isinstance(files, list):
        for row in files:
            if not isinstance(row, dict):
                continue
            url = str(row.get("download_url", "")).strip()
            name = str(row.get("name", "")).lower()
            if url and name.endswith(".zip"):
                return url
        for row in files:
            if not isinstance(row, dict):
                continue
            url = str(row.get("download_url", "")).strip()
            if url:
                return url

    final_url, doi_body = _http_get(FIGSHARE_DOI_URL)
    if DOWNLOAD_URL_RE.search(final_url):
        return final_url
    html = doi_body.decode("utf-8", errors="ignore")
    match = DOWNLOAD_URL_RE.search(html)
    if match:
        return match.group(0)
    raise RuntimeError(
        f"Could not resolve figshare download URL from {FIGSHARE_API_URL} or {FIGSHARE_DOI_URL}."
    )


def _stream_download(url: str, destination: Path, *, retries: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(req, timeout=180.0, context=SSL_CONTEXT) as resp, destination.open(
                "wb"
            ) as out:
                shutil.copyfileobj(resp, out, length=1024 * 1024)
            return
        except Exception as exc:  # pragma: no cover - network timing dependent
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed to download dataset from {url}") from last_error


def _find_dataset_root(extracted_root: Path) -> Path:
    for candidate in [extracted_root, extracted_root / "data", *[p for p in extracted_root.rglob("*") if p.is_dir()]]:
        info = get_example_data_info(candidate.parent if candidate.name == "data" else candidate)
        if info.is_installed:
            return info.data_dir
    raise RuntimeError(
        "Downloaded archive did not contain the expected nSTAT example-data layout."
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


def get_data_dir() -> Path:
    """Return the best currently-resolved example-data directory.

    Resolution order:
    1. `NSTAT_DATA_DIR`, if complete.
    2. Repo-local `data/`, if complete.
    3. Cache directory under `data_cache/nstat_data`.
    """

    env = os.environ.get("NSTAT_DATA_DIR")
    if env:
        env_root = Path(env).expanduser().resolve()
        env_info = get_example_data_info(env_root)
        if env_info.is_installed:
            return env_info.data_dir
        return env_root

    repo_info = get_example_data_info(_repo_root())
    if repo_info.is_installed:
        return repo_info.data_dir

    return _default_cache_dir()


def data_is_present(data_dir: Path) -> bool:
    """Return True when the required MATLAB-mirrored example files exist."""

    return get_example_data_info(data_dir, treat_as_data_dir=True).is_installed


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
        staged_parent = work_root / "staged_root"
        staged_parent.mkdir(parents=True, exist_ok=True)
        staged = staged_parent / "data"
        shutil.copytree(dataset_root, staged)
        _atomic_replace_tree(staged, data_dir)
        _write_sentinel(data_dir, source_url=download_url)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    return data_dir
