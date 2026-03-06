from __future__ import annotations

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
