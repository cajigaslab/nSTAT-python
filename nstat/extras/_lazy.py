"""Shared lazy-import helper for ``nstat.extras.*`` bridges.

Every bridge module follows the same pattern: import its optional
backing library inside a small ``_require_X()`` gate that raises a
clear :class:`ImportError` (with the exact ``pip install
nstat-toolbox[<key>]`` line) when the library is absent.  Before this
helper, that contract was hand-rolled in every module — six near-
identical functions across ``nstat/extras/{interop,validation,metrics}/*.py``.

This module centralises that pattern so:

- All bridges share *one* canonical error message format.
- Adding a new bridge needs zero boilerplate: just call
  :func:`require_optional` with the package name and the extras key.
- The "install-hint" contract enforced by
  ``tests/extras/test_extras_namespace.py::*_emits_install_hint_when_missing``
  is impossible to violate by oversight.

Usage
-----

.. code-block:: python

    from nstat.extras._lazy import require_optional

    def my_bridge_function(...):
        neo = require_optional("neo", install_key="neo")
        # ... use neo.<api> normally; the import succeeded.

For modules that need *multiple* packages from the same install key
(e.g., the ``neo`` bridge needs both ``neo`` and ``quantities``), call
:func:`require_optional` once per package — each call validates a
single package against the same install key.
"""
from __future__ import annotations

from importlib import import_module
from types import ModuleType


_BASE_HINT = "pip install nstat-toolbox[{key}]"


def _build_error_message(package: str, install_key: str) -> str:
    """Construct the canonical ImportError message for a missing optional dep."""
    return (
        f"nstat.extras requires the {package!r} package, which is not "
        f"installed.  Install with: {_BASE_HINT.format(key=install_key)}"
    )


def require_optional(package: str, *, install_key: str) -> ModuleType:
    """Import an optional package or raise an actionable :class:`ImportError`.

    Parameters
    ----------
    package
        The PyPI package name to import (e.g., ``"neo"``, ``"pynapple"``).
    install_key
        The ``[project.optional-dependencies]`` group key that ships
        this package (e.g., ``"neo"``, ``"test-parity"``).  Embedded in
        the error message so the user knows exactly which extras key
        to pass to ``pip install``.

    Returns
    -------
    ModuleType
        The imported module.

    Raises
    ------
    ImportError
        If ``package`` cannot be imported.  The message names the
        package, the ``nstat.extras`` namespace, AND the exact
        ``pip install nstat-toolbox[<install_key>]`` line.

    Examples
    --------
    >>> from nstat.extras._lazy import require_optional
    >>> np_mod = require_optional("numpy", install_key="dev")  # always present
    >>> np_mod.__name__
    'numpy'
    """
    try:
        return import_module(package)
    except ImportError as e:  # pragma: no cover — covered by per-bridge tests
        raise ImportError(_build_error_message(package, install_key)) from e


def require_optionals(
    *packages: str, install_key: str
) -> tuple[ModuleType, ...]:
    """Import multiple optional packages from the same install key.

    Convenience for bridges that need >1 package (e.g., neo + quantities).
    All packages are reported in the install_key's group; the failure
    message identifies the first missing one.
    """
    return tuple(require_optional(p, install_key=install_key) for p in packages)


__all__ = ["require_optional", "require_optionals"]
