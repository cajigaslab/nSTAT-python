"""Minimal Python nSTAT starter toolkit.

This package is a small, dependency-light foundation for Python examples
while the full nSTAT port is under development.
"""

from .analysis import psth, spike_indicator
from .core import Covariate, SpikeTrain
from .glm import PoissonGLMResult, fit_poisson_glm
from .simulation import simulate_cif_from_stimulus, simulate_poisson_from_rate

__all__ = [
    "Covariate",
    "SpikeTrain",
    "psth",
    "spike_indicator",
    "PoissonGLMResult",
    "fit_poisson_glm",
    "simulate_poisson_from_rate",
    "simulate_cif_from_stimulus",
]

