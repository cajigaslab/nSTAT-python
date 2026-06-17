"""Latent-variable / dimensionality-reduction bridges (Python-only).

Wraps community-standard latent-dynamics tools so nstat datasets can be
analysed with Gaussian-Process Factor Analysis (Yu et al. 2009) and
related methods. Each bridge declares its optional dependency; nothing
in this subpackage is imported by ``nstat`` core.
"""
from __future__ import annotations

from nstat.extras.latents.gpfa_bridge import (
    GPFAConfig,
    GPFAResult,
    fit_gpfa,
)

__all__ = ["GPFAConfig", "GPFAResult", "fit_gpfa"]
