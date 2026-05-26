Extras — opt-in bridges to the Python neuro ecosystem
=====================================================

The :mod:`nstat.extras` namespace ships Python-only features that have
no counterpart in upstream MATLAB nSTAT.  Each subpackage depends on
an optional library declared in ``pyproject.toml`` — install via
``pip install nstat-toolbox[<group>]`` (e.g. ``[neo]``, ``[pynapple]``,
``[nwb]``, ``[metrics]``, ``[nemos]``, ``[test-parity]``, or
``[all-extras]`` for the union).

For the design rationale and stability contract, see
:mod:`nstat.extras` and the
`integration_opportunities <https://github.com/cajigaslab/nSTAT-python/blob/main/parity/integration_opportunities.md>`_
audit.

Narrative usage guides
----------------------

Per-bridge documentation with install commands, API tables, recipes,
gotchas, and links to the runnable demos under ``examples/extras/``.

.. toctree::
   :maxdepth: 1

   extras/interop_neo
   extras/interop_pynapple
   extras/interop_nwb
   extras/validation_nemos
   extras/validation_pykalman
   extras/validation_statsmodels
   extras/metrics_spike_distances
   extras/em_dynamax


.. currentmodule:: nstat.extras


Interop — data-model bridges
----------------------------

.. autosummary::
   :toctree: _autosummary

   interop.neo
   interop.pynapple
   interop.nwb


Validation — cross-validation oracles
-------------------------------------

.. autosummary::
   :toctree: _autosummary

   validation.nemos_bridge
   validation.pykalman_bridge
   validation.statsmodels_bridge


Metrics — modern spike-train distance metrics
---------------------------------------------

.. autosummary::
   :toctree: _autosummary

   metrics.spike_distances


EM — state-space models trained by expectation-maximization
-----------------------------------------------------------

.. autosummary::
   :toctree: _autosummary

   em.dynamax_bridge
