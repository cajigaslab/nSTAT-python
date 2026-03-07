from __future__ import annotations

from pathlib import Path

import nstat.data_manager as data_manager
from nstat.compat.matlab import CIF, Covariate, SignalObj, nspikeTrain, nstColl
from nstat.install import nstat_install


def test_matlab_compat_namespace_imports() -> None:
    assert SignalObj is not None
    assert Covariate is not None
    assert CIF is not None
    assert nspikeTrain is not None
    assert nstColl is not None


def test_nstat_install_report_without_download() -> None:
    report = nstat_install(download_example_data=False)
    assert "repo_root" in report
    assert "example_data" in report
    assert report["download_example_data"] == "never"
    assert "required_files" in report["example_data"]


def test_get_paper_data_dirs_api(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"

    def fake_ensure_example_data(*, download: bool = True) -> Path:
        assert download is False
        return data_root

    monkeypatch.setattr(data_manager, "ensure_example_data", fake_ensure_example_data)

    dirs = data_manager.get_paper_data_dirs(download=False)
    assert dirs.data_dir == data_root
    assert dirs.mepsc_dir == data_root / "mEPSCs"
    assert dirs.explicit_stimulus_dir == data_root / "Explicit Stimulus"
    assert dirs.psth_dir == data_root / "PSTH"
    assert dirs.place_cell_data_dir == data_root / "Place Cells"

    tuple_dirs = data_manager.getPaperDataDirs(download=False)
    assert tuple_dirs == (
        data_root,
        data_root / "mEPSCs",
        data_root / "Explicit Stimulus",
        data_root / "PSTH",
        data_root / "Place Cells",
    )
