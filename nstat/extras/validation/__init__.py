"""Cross-validation bridges to other Python statistical / decoding toolboxes.

Each submodule fits the same statistical model in nstat and in an
independent reference library (NeMoS, pykalman, statsmodels, ...) so
nstat's MATLAB-faithful estimates can be triangulated against
unrelated implementations.  This is **independent** of the gold-fixture
MATLAB parity in ``tests/parity/fixtures/matlab_gold/`` — the gold
fixtures pin MATLAB↔Python parity; these bridges pin Python↔Python
agreement on the *underlying statistics*.

Install:
    pip install nstat-toolbox[test-parity]  # nemos, pykalman, statsmodels, nitime
"""
from __future__ import annotations

__all__: list[str] = []
