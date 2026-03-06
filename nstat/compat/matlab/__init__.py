"""MATLAB-style compatibility exports for nSTAT examples and notebooks."""

from __future__ import annotations

from ...ConfigColl import ConfigColl
from ...Covariate import Covariate
from ...FitResSummary import FitResSummary
from ...FitResult import FitResult
from ...SignalObj import SignalObj
from ...TrialConfig import TrialConfig
from ...cif import CIF
from ...nspikeTrain import nspikeTrain
from ...nstColl import nstColl

__all__ = [
    "CIF",
    "ConfigColl",
    "Covariate",
    "FitResSummary",
    "FitResult",
    "SignalObj",
    "TrialConfig",
    "nspikeTrain",
    "nstColl",
]

