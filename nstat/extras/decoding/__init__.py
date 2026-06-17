"""Bayesian point-process state-space decoders (Python-only extensions).

This subpackage hosts decoders that share nSTAT's exact point-process
filter mathematics (PPAF / PPHF) but extend it with observation models
and discrete-state classifiers that postdate the MATLAB toolbox:

- :mod:`nstat.extras.decoding.clusterless_bridge` — a thin bridge to
  `replay_trajectory_classification
  <https://github.com/Eden-Kramer-Lab/replay_trajectory_classification>`_
  (Denovellis et al. 2021, eLife; MIT) for **clusterless** marked
  point-process decoding (spike-waveform features replace spike
  sorting) and **trajectory-type classification** (e.g. replay vs.
  local).
- :mod:`nstat.extras.decoding.place_field_decoder` — pure-core
  one-call wrapper around the canonical 2-D place-cell decoder pattern
  (B-spline encoding + quadratic-CIF refit + PPAF decoding) from
  ``examples/paper/example08_real_place_cells.py``.

Install:

.. code-block:: bash

    pip install nstat-toolbox[clusterless]   # pulls JAX (~200 MB)

The place-field decoder has no opt-deps — it uses only the core nstat
stack (numpy + scipy).
"""
from __future__ import annotations

from nstat.extras.decoding.place_field_decoder import (
    PlaceFieldDecoderConfig,
    PlaceFieldDecoderResult,
    fit_place_field_decoder,
)

__all__: list[str] = [
    "PlaceFieldDecoderConfig",
    "PlaceFieldDecoderResult",
    "fit_place_field_decoder",
]
