"""Regression tests: installing the dataset must not delete unrelated files.

``ensure_example_data(download=True)`` swaps the downloaded dataset into the
data directory with ``_atomic_replace_tree``.  That helper used to delete the
old directory wholesale, so any file that was in the data directory but not
in the downloaded archive was lost -- including the git-tracked
``data_cache/nstat_data/paperHybridFilterExample.{h5,mat}`` that Example 05's
hybrid-filter path reads.  Everything here runs on ``tmp_path``; no network.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import nstat.data_manager as data_manager
from nstat.data_manager import _atomic_replace_tree


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _snapshot(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_replace_keeps_entries_missing_from_source(tmp_path: Path) -> None:
    dest = tmp_path / "nstat_data"
    _write(dest / "paperHybridFilterExample.mat", "tracked-mat")
    _write(dest / "overlap.txt", "old")
    _write(dest / "only_old_dir" / "keep.txt", "keep")
    _write(dest / "shared" / "old_only.txt", "old-only")
    _write(dest / "shared" / "overlap2.txt", "old2")

    src = tmp_path / "staged" / "data"
    _write(src / "overlap.txt", "new")
    _write(src / "shared" / "new.txt", "new-only")
    _write(src / "shared" / "overlap2.txt", "new2")

    _atomic_replace_tree(src, dest)

    assert _snapshot(dest) == {
        "only_old_dir/keep.txt": "keep",           # directory missing from source
        "overlap.txt": "new",                      # source wins
        "paperHybridFilterExample.mat": "tracked-mat",  # file missing from source
        "shared/new.txt": "new-only",
        "shared/old_only.txt": "old-only",         # nested file missing from source
        "shared/overlap2.txt": "new2",             # nested: source wins
    }
    assert not src.exists()
    assert not dest.with_name("nstat_data.bak").exists()


def test_replace_into_missing_destination(tmp_path: Path) -> None:
    dest = tmp_path / "cache" / "nstat_data"
    src = tmp_path / "staged" / "data"
    _write(src / "a.txt", "a")

    _atomic_replace_tree(src, dest)

    assert _snapshot(dest) == {"a.txt": "a"}
    assert not dest.with_name("nstat_data.bak").exists()


def test_replace_rolls_back_when_swap_fails(tmp_path: Path) -> None:
    dest = tmp_path / "nstat_data"
    _write(dest / "paperHybridFilterExample.h5", "tracked-h5")
    _write(dest / "sub" / "x.txt", "x")
    before = _snapshot(dest)

    with pytest.raises(OSError):
        _atomic_replace_tree(tmp_path / "does_not_exist", dest)

    assert _snapshot(dest) == before
    assert not dest.with_name("nstat_data.bak").exists()


def test_replace_rolls_back_when_carry_over_fails(tmp_path: Path, monkeypatch) -> None:
    dest = tmp_path / "nstat_data"
    _write(dest / "paperHybridFilterExample.h5", "tracked-h5")
    _write(dest / "overlap.txt", "old")
    before = _snapshot(dest)
    src = tmp_path / "staged" / "data"
    _write(src / "overlap.txt", "new")

    def failing_carry_over(old: Path, new: Path) -> None:
        raise OSError("simulated copy failure")

    monkeypatch.setattr(data_manager, "_carry_over_missing", failing_carry_over)
    with pytest.raises(OSError, match="simulated copy failure"):
        _atomic_replace_tree(src, dest)

    assert _snapshot(dest) == before
    assert not dest.with_name("nstat_data.bak").exists()


def test_ensure_example_data_download_keeps_tracked_cache_files(
    tmp_path: Path, monkeypatch
) -> None:
    """The documented install path (``nstat-install --download-example-data
    always`` -> ``ensure_example_data(download=True)``) with a fake archive."""
    monkeypatch.delenv("NSTAT_DATA_DIR", raising=False)
    monkeypatch.delenv("NSTAT_OFFLINE", raising=False)
    monkeypatch.setattr(data_manager, "_repo_root", lambda: tmp_path)
    cache = tmp_path / "data_cache" / "nstat_data"
    _write(cache / "paperHybridFilterExample.h5", "tracked-h5")
    _write(cache / "paperHybridFilterExample.mat", "tracked-mat")

    layout = data_manager.get_example_data_info(tmp_path / "layout")
    required = [p.relative_to(layout.data_dir).as_posix() for p in layout.required_files]
    fake_url = "https://example.invalid/ndownloader/files/1"

    def fake_stream_download(url: str, destination: Path, *, retries: int = 3) -> None:
        assert url == fake_url
        with zipfile.ZipFile(destination, "w") as zf:
            for rel in required:
                zf.writestr(f"data/{rel}", "payload")

    monkeypatch.setattr(data_manager, "_resolve_figshare_download_url", lambda: fake_url)
    monkeypatch.setattr(data_manager, "_stream_download", fake_stream_download)

    data_dir = data_manager.ensure_example_data(download=True)

    assert data_dir == cache.resolve()
    assert data_manager.data_is_present(data_dir)
    assert (cache / "paperHybridFilterExample.h5").read_text(encoding="utf-8") == "tracked-h5"
    assert (cache / "paperHybridFilterExample.mat").read_text(encoding="utf-8") == "tracked-mat"
    sentinel = json.loads((cache / data_manager.SENTINEL_NAME).read_text(encoding="utf-8"))
    assert sentinel["source_url"] == fake_url
    assert not cache.with_name("nstat_data.bak").exists()
    assert list((tmp_path / "output" / "data_download").iterdir()) == []
