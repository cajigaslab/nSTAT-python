import sys as _sys
from types import ModuleType as _ModuleType

from .ConfidenceInterval import ConfidenceInterval
from .ConfigColl import ConfigColl
from .CovColl import CovColl
from .SignalObj import SignalObj
from .analysis import Analysis, psth
from .cif import CIF, CIFModel
from .data_manager import getPaperDataDirs, get_paper_data_dirs
from .datasets import get_dataset_path, list_datasets, verify_checksums
from .decoding import DecoderSuite
from .decoding_algorithms import DecodingAlgorithms
from .errors import DataNotFoundError, ParityValidationError, UnsupportedWorkflowError
from .events import Events
from .fit import FitResSummary, FitResult, FitSummary
from .glm import PoissonGLMResult, fit_poisson_glm
from .history import History, HistoryBasis
from .matlab_reference import matlab_engine_available, run_point_process_reference, run_simulated_network_reference
from .paper_examples_full import run_full_paper_examples
from .signal import Covariate, Signal
from .simulation import simulate_poisson_from_rate
from .simulators import (
    NetworkSimulationResult,
    PointProcessSimulation,
    simulate_point_process,
    simulate_two_neuron_network,
)
from .spikes import SpikeTrain, SpikeTrainCollection
from .trial import ConfigCollection, CovariateCollection, Trial, TrialConfig
from .nspikeTrain import nspikeTrain
from .nstColl import nstColl

from . import analysis as _analysis_module
from . import cif as _cif_module
from . import events as _events_module
from . import history as _history_module
from . import trial as _trial_module

_sys.modules.setdefault(f"{__name__}.Analysis", _analysis_module)
_sys.modules.setdefault(f"{__name__}.CIF", _cif_module)
_sys.modules.setdefault(f"{__name__}.Events", _events_module)
_sys.modules.setdefault(f"{__name__}.History", _history_module)
_sys.modules.setdefault(f"{__name__}.Trial", _trial_module)


class _NstatModule(_ModuleType):
    def __getattribute__(self, name: str):
        value = super().__getattribute__(name)
        if isinstance(value, _ModuleType) and hasattr(value, name):
            return getattr(value, name)
        return value


_sys.modules[__name__].__class__ = _NstatModule


def __getattr__(name: str):
    if name == "nstat_install":
        from .install import nstat_install as _nstat_install

        return _nstat_install
    if name == "nSTAT_Install":
        from .nstat_install import nSTAT_Install as _nSTAT_Install

        return _nSTAT_Install
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Analysis",
    "CIF",
    "CIFModel",
    "ConfidenceInterval",
    "ConfigColl",
    "ConfigCollection",
    "Covariate",
    "CovColl",
    "CovariateCollection",
    "DataNotFoundError",
    "DecoderSuite",
    "DecodingAlgorithms",
    "Events",
    "FitResSummary",
    "FitResult",
    "FitSummary",
    "History",
    "HistoryBasis",
    "matlab_engine_available",
    "NetworkSimulationResult",
    "ParityValidationError",
    "PointProcessSimulation",
    "PoissonGLMResult",
    "SignalObj",
    "Signal",
    "SpikeTrain",
    "SpikeTrainCollection",
    "Trial",
    "TrialConfig",
    "UnsupportedWorkflowError",
    "fit_poisson_glm",
    "getPaperDataDirs",
    "get_paper_data_dirs",
    "get_dataset_path",
    "list_datasets",
    "nSTAT_Install",
    "nstat_install",
    "psth",
    "run_full_paper_examples",
    "run_point_process_reference",
    "run_simulated_network_reference",
    "simulate_point_process",
    "simulate_poisson_from_rate",
    "simulate_two_neuron_network",
    "nspikeTrain",
    "nstColl",
    "verify_checksums",
]
