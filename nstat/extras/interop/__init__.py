"""Interoperability bridges between nstat and the wider Python neuro stack.

The submodules here adapt nstat's MATLAB-style primitives
(``nspikeTrain``, ``SpikeTrainCollection``, ``Trial``) into the data
models used by Neo, pynapple, and pynwb — and vice versa — so users can
move data between toolboxes without re-implementing converters.

Each submodule's optional dependency is declared under
``[project.optional-dependencies]`` in ``pyproject.toml``::

    pip install nstat-toolbox[neo]
    pip install nstat-toolbox[pynapple]
    pip install nstat-toolbox[nwb]

Importing a submodule whose backing library is not installed raises
``ImportError`` with the exact pip-install line needed.
"""
from __future__ import annotations

__all__: list[str] = []
