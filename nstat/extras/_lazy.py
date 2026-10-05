"""Shared lazy-import helper for ``nstat.extras.*`` bridges.

Every bridge module follows the same pattern: import its optional
backing library inside a small ``_require_X()`` gate that raises a
clear :class:`ImportError` (with the exact ``pip install
nstat-toolbox[<key>]`` line) when the library is absent, or installed
but failing to import (e.g. a binary/ABI mismatch).  Before this
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
from importlib.util import find_spec
from types import ModuleType


_BASE_HINT = "pip install nstat-toolbox[{key}]"


def _build_error_message(package: str, install_key: str) -> str:
    """Construct the canonical ImportError message for a missing optional dep."""
    return (
        f"nstat.extras requires the {package!r} package, which is not "
        f"installed.  Install with: {_BASE_HINT.format(key=install_key)}"
    )


def _build_broken_message(package: str, install_key: str, exc: BaseException) -> str:
    """ImportError message for a dep that is installed but fails to import."""
    return (
        f"nstat.extras requires the {package!r} package, which is installed "
        f"but failed to import: {type(exc).__name__}: {exc} (likely a "
        f"binary/ABI incompatibility, e.g. a compiled extension built against "
        f"a different NumPy, or a missing or broken dependency).  Reinstall it "
        f"so it matches this environment; the supported install line is: "
        f"{_BASE_HINT.format(key=install_key)}"
    )


def _is_not_installed(package: str, exc: BaseException) -> bool:
    """True if the import failed because ``package`` itself cannot be found.

    The top-level distribution is looked up with :func:`importlib.util.find_spec`,
    which does not execute the package (``find_spec`` on a dotted name would
    import the parent, and a broken parent would raise again).  A
    ``ModuleNotFoundError`` that names ``package`` or one of its parents also
    counts as "not installed" (e.g. ``tick`` present but ``tick.hawkes``
    absent, or a ``sys.modules[name] = None`` block).  Anything else means the
    package was found but raised while importing.
    """
    if isinstance(exc, ModuleNotFoundError) and exc.name is not None:
        if package == exc.name or package.startswith(exc.name + "."):
            return True
    try:
        return find_spec(package.partition(".")[0]) is None
    except Exception:  # e.g. sys.modules[name].__spec__ is None -> it is present
        return False


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
        ``pip install nstat-toolbox[<install_key>]`` line.  It says
        either that the package is not installed, or that it is
        installed but failed to import (with the original exception type
        and message; usually a binary/ABI incompatibility such as an
        extension built against a different NumPy, which raises
        ``ValueError`` rather than ``ImportError``).  The original
        exception is chained as ``__cause__`` in both cases.

    Examples
    --------
    >>> from nstat.extras._lazy import require_optional
    >>> np_mod = require_optional("numpy", install_key="dev")  # always present
    >>> np_mod.__name__
    'numpy'
    """
    try:
        return import_module(package)
    except Exception as e:
        if _is_not_installed(package, e):
            raise ImportError(_build_error_message(package, install_key)) from e
        raise ImportError(_build_broken_message(package, install_key, e)) from e


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
