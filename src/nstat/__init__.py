"""nSTAT clean-room Python toolbox.

The package preserves high-level MATLAB nSTAT structure while using Python-native
implementations and interfaces.
"""

from .analysis import Analysis
from .cif import CIFModel
from .confidence import ConfidenceInterval
from .decoding import DecodingAlgorithms
from .datasets import fetch_matlab_gold_file, latest_matlab_gold_version, list_matlab_gold_files
from .events import Events
from .fit import FitResult, FitSummary
from .history import HistoryBasis
from .install import InstallReport, nstat_install
from .signal import Covariate, Signal
from .spikes import SpikeTrain, SpikeTrainCollection
from .trial import ConfigCollection, CovariateCollection, Trial, TrialConfig

__all__ = [
    "Analysis",
    "CIFModel",
    "ConfidenceInterval",
    "DecodingAlgorithms",
    "Events",
    "FitResult",
    "FitSummary",
    "HistoryBasis",
    "InstallReport",
    "Signal",
    "Covariate",
    "SpikeTrain",
    "SpikeTrainCollection",
    "CovariateCollection",
    "TrialConfig",
    "ConfigCollection",
    "Trial",
    "nstat_install",
    "list_matlab_gold_files",
    "latest_matlab_gold_version",
    "fetch_matlab_gold_file",
]
