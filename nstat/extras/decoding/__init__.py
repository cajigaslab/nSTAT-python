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

Install:

.. code-block:: bash

    pip install nstat-toolbox[clusterless]   # pulls JAX (~200 MB)
"""
from __future__ import annotations

__all__: list[str] = []
