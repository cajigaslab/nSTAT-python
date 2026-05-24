"""Spike-train distance and synchrony metrics not in MATLAB nSTAT.

These metrics belong in ``extras`` (not ``nstat`` core) because they
have no counterpart in upstream MATLAB nSTAT — they're modern
additions that the Python neural-data community has converged on
(Mulansky, Kreuz, Victor-Purpura) and that delegate to specialized,
C-accelerated libraries rather than reimplementing them.

Install:
    pip install nstat-toolbox[metrics]   # PySpike (BSD-2)
"""
from __future__ import annotations

__all__: list[str] = []
