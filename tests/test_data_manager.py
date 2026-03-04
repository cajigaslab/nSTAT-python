from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from nstat import data_manager


def _write_expected_tree(root: Path) -> None:
    for name in data_manager.EXPECTED_SUBDIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / data_manager.SENTINEL_FILENAME).write_text(
        json.dumps({"doi": data_manager.DOI_URL}), encoding="utf-8"
    )


def _build_fixture_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in data_manager.EXPECTED_SUBDIRS:
            zf.writestr(f"nSTAT_data/{name}/.keep", "")
    return buf.getvalue()


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        url: str = "",
        text: str = "",
        json_payload: dict | None = None,
        content_bytes: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self.url = url
        self.text = text
        self._json_payload = json_payload or {}
        self._content_bytes = content_bytes

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._json_payload

    def iter_content(self, chunk_size: int = 1024 * 1024):
        data = self._content_bytes
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    def __init__(self, zip_bytes: bytes) -> None:
        self.zip_bytes = zip_bytes
        self.headers = {}

    def mount(self, *_args, **_kwargs) -> None:  # pragma: no cover - interface no-op
        return None

    def get(self, url: str, *args, **kwargs):  # noqa: ANN001
        if url.startswith(data_manager.DOI_URL):
            return _FakeResponse(
                status_code=200,
                url="https://figshare.com/articles/dataset/nSTAT_data/4834640",
                text="<html></html>",
            )
        if url.startswith("https://api.figshare.com/v2/articles/4834640"):
            return _FakeResponse(
                status_code=200,
                json_payload={
                    "files": [
                        {
                            "name": "nSTAT_example_data.zip",
                            "download_url": "https://ndownloader.figshare.com/files/123456",
                            "size": len(self.zip_bytes),
                        }
                    ]
                },
            )
        if url.startswith("https://ndownloader.figshare.com/files/123456"):
            return _FakeResponse(status_code=200, content_bytes=self.zip_bytes)
        raise AssertionError(f"Unexpected URL in fake session: {url}")


def test_get_data_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    override = tmp_path / "custom_data_root"
    monkeypatch.setenv("NSTAT_DATA_DIR", str(override))
    assert data_manager.get_data_dir() == override.resolve()


def test_data_is_present_requires_expected_structure(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    assert data_manager.data_is_present(root) is False
    _write_expected_tree(root)
    assert data_manager.data_is_present(root) is True


def test_ensure_example_data_offline_missing_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "missing_data"
    monkeypatch.setenv("NSTAT_DATA_DIR", str(target))
    with pytest.raises(FileNotFoundError):
        data_manager.ensure_example_data(download=False)


def test_ensure_example_data_downloads_and_writes_sentinel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "downloaded_data"
    monkeypatch.setenv("NSTAT_DATA_DIR", str(target))

    zip_bytes = _build_fixture_zip()
    fake_session = _FakeSession(zip_bytes)
    monkeypatch.setattr(data_manager, "_build_session", lambda: fake_session)

    resolved = data_manager.ensure_example_data(download=True)
    assert resolved == target.resolve()
    assert data_manager.data_is_present(resolved)
    payload = json.loads((resolved / data_manager.SENTINEL_FILENAME).read_text(encoding="utf-8"))
    assert payload["doi"] == data_manager.DOI_URL
    assert payload["source_url"].startswith("https://ndownloader.figshare.com/files/")
    assert "archive_sha256" in payload


def test_ensure_example_data_skips_download_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "existing_data"
    _write_expected_tree(target)
    monkeypatch.setenv("NSTAT_DATA_DIR", str(target))

    called = {"value": False}

    def _fail_session() -> _FakeSession:
        called["value"] = True
        raise AssertionError("session should not be created when data are already present")

    monkeypatch.setattr(data_manager, "_build_session", _fail_session)
    resolved = data_manager.ensure_example_data(download=True)
    assert resolved == target.resolve()
    assert called["value"] is False

