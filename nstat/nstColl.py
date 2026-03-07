from __future__ import annotations

from .trial import SpikeTrainCollection as _SpikeTrainCollection


class nstColl(_SpikeTrainCollection):
    """MATLAB-facing spike-train collection class."""


__all__ = ["nstColl"]
