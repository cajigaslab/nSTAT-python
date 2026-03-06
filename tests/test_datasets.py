from __future__ import annotations

from pathlib import Path

import nstat
import nstat.datasets


def test_dataset_manifest_and_checksums() -> None:
    names = nstat.list_datasets()
    assert names
    assert names == sorted(names)


def test_get_dataset_path_triggers_download_when_data_is_external(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "nstat_data"
    dataset_path = data_root / "mEPSCs" / "epsc2.txt"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text("header\n0 0\n", encoding="utf-8")

    calls: list[bool] = []

    def fake_ensure_example_data(*, download: bool = True) -> Path:
        calls.append(download)
        return data_root

    monkeypatch.setattr(nstat.datasets, "ensure_example_data", fake_ensure_example_data)

    resolved = nstat.get_dataset_path("mepcs_epsc2")
    assert resolved == dataset_path
    assert calls == [True]


def test_verify_checksums_triggers_download_when_data_is_external(monkeypatch, tmp_path: Path) -> None:
    calls: list[bool] = []

    def fake_ensure_example_data(*, download: bool = True) -> Path:
        calls.append(download)
        return tmp_path / "nstat_data"

    monkeypatch.setattr(nstat.datasets, "ensure_example_data", fake_ensure_example_data)

    result = nstat.verify_checksums()
    assert result
    assert all(isinstance(value, bool) for value in result.values())
    assert calls and all(call is True for call in calls)
