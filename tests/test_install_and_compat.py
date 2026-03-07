from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import nstat.data_manager as data_manager
import nstat.install as install_module
from nstat.compat.matlab import CIF, Covariate, SignalObj, nspikeTrain, nstColl
from nstat.install import nstat_install


def test_matlab_compat_namespace_imports() -> None:
    assert SignalObj is not None
    assert Covariate is not None
    assert CIF is not None
    assert nspikeTrain is not None
    assert nstColl is not None


def test_nstat_install_report_without_download() -> None:
    report = nstat_install(download_example_data=False, rebuild_doc_search=False)
    assert "repo_root" in report
    assert "example_data" in report
    assert "doc_search" in report
    assert "path_preferences" in report
    assert report["download_example_data"] == "never"
    assert "required_files" in report["example_data"]
    assert report["doc_search"]["status"] == "skipped"
    assert report["path_preferences"]["status"] == "not_applicable"


def test_nstat_install_clean_user_path_prefs_is_documented_noop() -> None:
    report = nstat_install(download_example_data=False, rebuild_doc_search=False, clean_user_path_prefs=True)
    assert report["clean_user_path_prefs"] is True
    assert report["path_preferences"]["requested"] is True
    assert report["path_preferences"]["status"] == "not_applicable"
    assert "ignored in Python" in " ".join(report["notes"])


def test_rebuild_doc_search_reports_success(monkeypatch, tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    html_dir = docs_dir / "_build" / "html"
    (docs_dir / "conf.py").parent.mkdir(parents=True, exist_ok=True)
    (docs_dir / "conf.py").write_text("project='nSTAT'\n", encoding="utf-8")

    monkeypatch.setattr(install_module.importlib.util, "find_spec", lambda name: object())

    def fake_run(cmd, *, cwd, check, capture_output, text):
        assert cmd[:5] == [install_module.sys.executable, "-m", "sphinx", "-q", "-b"]
        assert cwd == tmp_path
        html_dir.mkdir(parents=True, exist_ok=True)
        (html_dir / "searchindex.js").write_text("search", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(install_module.subprocess, "run", fake_run)

    report = install_module._rebuild_doc_search(tmp_path)
    assert report["status"] == "rebuilt"
    assert report["is_available"] is True
    assert report["search_index"] == str(html_dir / "searchindex.js")


def test_rebuild_doc_search_reports_missing_sphinx(monkeypatch, tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "conf.py").write_text("project='nSTAT'\n", encoding="utf-8")

    monkeypatch.setattr(install_module.importlib.util, "find_spec", lambda name: None)

    report = install_module._rebuild_doc_search(tmp_path)
    assert report["status"] == "skipped"
    assert "Sphinx is not installed" in report["reason"]


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
