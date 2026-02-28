from __future__ import annotations

from nstat.install import nstat_install



def test_nstat_install_returns_existing_cache_dir() -> None:
    report = nstat_install()
    assert report.package == "nstat"
    assert report.cache_dir.exists()
