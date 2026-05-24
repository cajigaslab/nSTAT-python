"""Python-only extensions to nstat — opt-in capability that goes beyond MATLAB parity.

This namespace is the home for features that have no counterpart in
upstream MATLAB nSTAT and would dilute the MATLAB-parity contract of
the core :mod:`nstat` package if added there.

Three subpackages ship today:

- :mod:`nstat.extras.interop` — converters between :class:`nstat.nspikeTrain`
  / :class:`nstat.SpikeTrainCollection` / :class:`nstat.Trial` and the
  data models used by **Neo**, **pynapple**, and **pynwb**.
- :mod:`nstat.extras.validation` — Python-side cross-validation bridges
  (start: **NeMoS** Poisson GLM) that triangulate nstat's MATLAB-faithful
  estimates against independent reference implementations.
- :mod:`nstat.extras.metrics` — modern spike-train distance / synchrony
  metrics (ISI-distance, SPIKE-distance, SPIKE-synchronization via
  **PySpike**) that have no MATLAB counterpart.

Stability contract
------------------
- Symbols in :data:`nstat.extras.__all__` follow semantic versioning at
  the **minor** level — minor releases of ``nstat-toolbox`` may add,
  rename, or remove extras-namespace symbols without going to a major
  version bump.
- The core :mod:`nstat` namespace remains under the stricter
  MATLAB-parity contract: removals/renames there require major-version
  bumps.

Decision rule (also documented in :file:`CLAUDE.md` and
:file:`AGENT_GUIDE.md`):

  Goes in core ``nstat.*`` IF:
  - The feature exists in MATLAB nSTAT (``.m`` source file present).
  - Has an entry in ``parity/manifest.yml``.
  - Removing it would break a MATLAB-faithful workflow.

  Goes in ``nstat.extras.*`` IF:
  - Python-only with no MATLAB counterpart.
  - Depends on libraries outside core dependencies (PyTorch,
    SpikeInterface, MNE, Neo, …).
  - Uses Pythonic snake_case naming where the MATLAB-style would clash.
  - Experimental — API may break across minor releases.

Optional dependencies
---------------------
Most extras modules pull in libraries beyond the core ``nstat-toolbox``
dependency set.  Install them via the extras keys declared in
``pyproject.toml``::

    pip install nstat-toolbox[neo]              # neo
    pip install nstat-toolbox[pynapple]         # pynapple
    pip install nstat-toolbox[nwb]              # pynwb
    pip install nstat-toolbox[metrics]          # pyspike
    pip install nstat-toolbox[test-parity]      # nemos, pykalman, statsmodels, nitime
    pip install nstat-toolbox[all-extras]       # install everything

Each extras module raises a clear, actionable ``ImportError`` at import
time when its optional dependency is missing.

Independence
------------
This package is Python-side only.  No runtime coupling to the MATLAB
repository is introduced by anything in ``nstat.extras`` — the
sanctioned MATLAB-Engine bridge module :mod:`nstat.matlab_engine`
remains the only MATLAB-runtime entry point in the package; see
``parity/simulink_fidelity.yml`` for the audit trail.
"""
from __future__ import annotations

# Submodules are not eagerly imported here — each depends on an optional
# library that the user may not have installed.  Users access them via
# explicit imports::
#
#     from nstat.extras.interop.neo import to_neo_spiketrain
#     from nstat.extras.interop.pynapple import to_pynapple_ts
#     from nstat.extras.interop.nwb import read_nwb_path
#     from nstat.extras.validation.nemos_bridge import cross_validate_poisson_glm
#     from nstat.extras.metrics.spike_distances import spike_distance
#
# Importing this top-level package is safe even when no optional deps
# are installed (no eager submodule imports).
#
# The helpfile-freshness checker (``tools/check_helpfile_freshness.py``)
# treats every name in ``__all__`` the same way it treats
# ``nstat.__all__``: it must appear in ``AGENT_GUIDE.md`` and, if a
# class, in ``docs/ClassDefinitions.md``.

__all__: list[str] = []
