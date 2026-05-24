"""Python-only extensions to nstat — opt-in capability that goes beyond MATLAB parity.

This namespace is the home for features that have no counterpart in
upstream MATLAB nSTAT and would dilute the MATLAB-parity contract of
the core :mod:`nstat` package if added there.  Examples (planned, not
yet implemented):

- Bridges to third-party Python neural-data libraries (SpikeInterface,
  Neo, pynapple, NWB).
- Modern decoders that don't exist in MATLAB (deep-learning, wavelet
  synchrosqueeze, adaptive multitaper).
- Pythonic alternatives to MATLAB-style APIs (snake_case wrappers,
  dataclass return types) where the prevailing style would clash with
  the parity-preserved core.

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

    pip install nstat-toolbox[spikeinterface]   # adds SpikeInterface bridge
    pip install nstat-toolbox[neo]              # adds Neo I/O
    pip install nstat-toolbox[nwb]              # adds NWB I/O
    pip install nstat-toolbox[deep-learning]    # adds PyTorch decoders
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

# When concrete extras modules land, append their public symbols to
# this list.  The helpfile-freshness checker
# (``tools/check_helpfile_freshness.py``) treats any name in
# ``nstat.extras.__all__`` the same way it treats ``nstat.__all__``:
# it must appear in ``AGENT_GUIDE.md`` and, if a class, in
# ``docs/ClassDefinitions.md``.
__all__: list[str] = []
