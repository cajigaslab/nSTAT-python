"""Canonical decoding subpackage.

Mirrors the MATLAB ``+nstat/+decoding/`` package layout.  Modules in this
subpackage host individual decoder families (e.g. :mod:`PPLFP`, the
point-process log-likelihood filter family) so each family can be
maintained, documented, and ported in isolation rather than as a single
~8 kLOC flat namespace.

For migration compatibility the legacy public surface previously exposed
from :mod:`nstat.decoding_algorithms` is re-exported here verbatim, so
``from nstat.decoding import DecodingAlgorithms`` and similar imports
continue to work while call sites are updated to the new locations.

Notes
-----
The MATLAB-parity contract (see ``CLAUDE.md`` §"MATLAB parity principle")
applies: names and signatures in :class:`~nstat.decoding.PPLFP.PPLFP`
mirror the MATLAB ``DecodingAlgorithms.m`` ``PPLFP_*`` methods.  Folder
layout may be Python-native; the public method surface is the contract.
"""
from __future__ import annotations

from .PPLFP import PPLFP
from ._suite import DecoderSuite
from .. import decoding_algorithms as _decoding_algorithms

# Re-export the legacy public API surface from nstat.decoding_algorithms so
# downstream code can migrate ``from nstat.decoding_algorithms import X`` to
# ``from nstat.decoding import X`` without losing any symbols.
_legacy_all = getattr(_decoding_algorithms, "__all__", ())
for _name in _legacy_all:
    if _name in globals():
        continue
    globals()[_name] = getattr(_decoding_algorithms, _name)

__all__ = ["PPLFP", "DecoderSuite", *list(_legacy_all)]
