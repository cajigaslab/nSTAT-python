from __future__ import annotations

from .install import nstat_install


def nSTAT_Install(**kwargs):
    """MATLAB-style alias that delegates to :func:`nstat.install.nstat_install`."""

    return nstat_install(**kwargs)


__all__ = ["nSTAT_Install"]
