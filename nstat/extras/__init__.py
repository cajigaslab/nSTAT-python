"""Python-only extensions to nstat — opt-in capability that goes beyond MATLAB parity.

This namespace is the home for features that have no counterpart in
upstream MATLAB nSTAT and would dilute the MATLAB-parity contract of
the core :mod:`nstat` package if added there.

Seven subpackages ship today:

- :mod:`nstat.extras.interop` — converters between :class:`nstat.nspikeTrain`
  / :class:`nstat.SpikeTrainCollection` / :class:`nstat.Trial` and the
  data models used by **Neo**, **pynapple**, and **pynwb**.
- :mod:`nstat.extras.validation` — Python-side cross-validation bridges
  (**NeMoS** / **statsmodels** Poisson GLM, **pykalman**) that triangulate
  nstat's MATLAB-faithful estimates against independent reference
  implementations.
- :mod:`nstat.extras.metrics` — modern spike-train distance / synchrony
  metrics (ISI-distance, SPIKE-distance, SPIKE-synchronization via
  **PySpike**) that have no MATLAB counterpart.
- :mod:`nstat.extras.em` — EM-trained linear-Gaussian, point-process, and
  hybrid state-space models (the ``KF_EM`` / ``PP_EM`` / ``mPPCO_EM``
  families) via **Dynamax**, with held-out predictive log-likelihood,
  identifiability-gauge canonicalization, and multi-restart selection.
- :mod:`nstat.extras.decoding` — Bayesian point-process decoders that
  extend nSTAT's PPAF / PPHF mathematics, including **clusterless**
  marked point-process decoding via **replay_trajectory_classification**,
  and the pure-core place-field decoder wrapper.
- :mod:`nstat.extras.latents` — latent-dynamics bridges, currently
  Gaussian-Process Factor Analysis (GPFA) via **Elephant**.
- :mod:`nstat.extras.spatial` — spatial and spatiotemporal point-process
  methods (log-Gaussian Cox processes, spatial / marked goodness-of-fit,
  Hawkes and cluster-Cox models); the core needs only NumPy / SciPy, and
  optional heavier bridges (**gpflow**, **tick**, **DPPy**) are opt-in.

Layout convention
-----------------
Use a **subpackage** when a feature area has two or more modules that share
an install key or a private helper module (for example ``interop``, ``em``
and ``spatial``).  Use a **flat module** for a single self-contained feature
that needs only core (non-optional) dependencies.  The flat modules today
are :mod:`nstat.extras.continuous_cif` (continuous-time CIF simulator) and
:mod:`nstat.extras.matlab_rng` (MATLAB-aligned MT19937 stream), plus the
private :mod:`nstat.extras._numba_kernels` and :mod:`nstat.extras._lazy`
helpers.  The split is a convention, not a contract: existing modules are
not relocated to match it, because moving one would change a public import
path.

Stability contract
------------------
- Symbols in :data:`nstat.extras.__all__` follow semantic versioning at
  the **minor** level — minor releases of ``nstat-toolbox`` may add,
  rename, or remove extras-namespace symbols without going to a major
  version bump.
- The core :mod:`nstat` namespace remains under the stricter
  MATLAB-parity contract: removals/renames there require major-version
  bumps.

Decision rule (also documented in :file:`AGENT_GUIDE.md`):

Goes in core ``nstat.*`` if:

- The feature exists in MATLAB nSTAT (``.m`` source file present).
- It has an entry in ``parity/manifest.yml``.
- Removing it would break a MATLAB-faithful workflow.

Goes in ``nstat.extras.*`` if:

- It is Python-only with no MATLAB counterpart.
- It depends on libraries outside core dependencies (PyTorch,
  SpikeInterface, MNE, Neo, …).
- It uses Pythonic snake_case naming where the MATLAB-style would clash.
- It is experimental — the API may break across minor releases.

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
    pip install nstat-toolbox[all-extras]       # the lightweight groups above

The ``all-extras`` key is deliberately **not** a union of every group.  The
heavy or niche groups ``dynamax``, ``clusterless``, ``spatial-gp``,
``hawkes``, ``dpp``, ``latents`` and ``numba`` are excluded for install-size
reasons and must be installed individually, for example
``pip install nstat-toolbox[dynamax]``.

Importing ``nstat.extras`` or one of its subpackages is safe without any
optional dependency installed.  The bridges import their backing library
lazily, inside the function that needs it; calling such a function without
the dependency raises a clear, actionable ``ImportError`` that names the
``pip install nstat-toolbox[<key>]`` line (see :mod:`nstat.extras._lazy`).

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
