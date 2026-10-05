"""Robust optional-dependency probes for the test suite.

``pytest.importorskip`` (and a bare ``except ImportError``) only treat a
*missing* package as skippable.  A package that is installed but broken --
e.g. a wheel built against an older NumPy ABI that raises ``ValueError``
("numpy.dtype size changed") or ``RuntimeError`` at import time -- would turn
into a hard test failure even though the environment, not nstat, is at fault.

The helpers here probe a **third-party** package by actually importing it and
catching ``Exception``, and they distinguish the two situations in the skip
reason:

* not installed:            ``"<pkg> not installed"``
* installed but unusable:   ``"<pkg> installed but failed to import: <ExcType>: <msg>"``

HAZARD -- third-party packages only.  Never route ``import nstat...`` through
these helpers: a genuine nstat import bug must fail the test, not skip it.
"""
from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType
from typing import NamedTuple

import pytest

__all__ = [
    "ProbeResult",
    "probe_optional",
    "importorskip_robust",
    "skip_if_installed",
]


class ProbeResult(NamedTuple):
    """Outcome of :func:`probe_optional`."""

    #: the package imported cleanly
    available: bool
    #: the package is present in the environment (``find_spec`` found it),
    #: whether or not it imports cleanly
    installed: bool
    #: human-readable reason when ``available`` is False, else ``""``
    reason: str
    #: the imported module when ``available`` else ``None``
    module: ModuleType | None


#: Packages that load their submodules lazily: ``import pkg`` succeeds even when
#: the real implementation (and its compiled backends) cannot import, and the
#: failure only surfaces later as a confusing ``AttributeError`` on the lazy
#: module.  Probe the listed submodule too so a broken backend becomes a skip.
#: (pynapple: ``pynapple.core`` pulls in pandas, which may be ABI-broken.)
_DEEP_PROBE: dict[str, str] = {"pynapple": "pynapple.core"}


def probe_optional(pkg: str) -> ProbeResult:
    """Import third-party ``pkg``; classify the failure mode, never raise."""
    try:
        spec = importlib.util.find_spec(pkg)
    except Exception:  # find_spec imports parents of dotted names
        spec = None
    try:
        module = importlib.import_module(pkg)
        if pkg in _DEEP_PROBE:
            importlib.import_module(_DEEP_PROBE[pkg])
    except ImportError as exc:
        if spec is None:
            return ProbeResult(False, False, f"{pkg} not installed", None)
        return ProbeResult(
            False, True,
            f"{pkg} installed but failed to import: {type(exc).__name__}: {exc}",
            None,
        )
    except Exception as exc:  # ABI mismatches surface as ValueError/RuntimeError/...
        return ProbeResult(
            False, True,
            f"{pkg} installed but failed to import: {type(exc).__name__}: {exc}",
            None,
        )
    return ProbeResult(True, True, "", module)


def importorskip_robust(pkg: str) -> ModuleType:
    """Drop-in for ``pytest.importorskip(pkg)`` that also skips on broken installs."""
    __tracebackhide__ = True  # report the skip at the calling test, not here
    result = probe_optional(pkg)
    if not result.available:
        pytest.skip(result.reason, allow_module_level=True)
    assert result.module is not None
    return result.module


def skip_if_installed(pkg: str) -> None:
    """Skip an "import-error path" test when ``pkg`` is present in any state.

    Those tests exercise the message raised when the dependency is *absent*;
    if the package is installed (cleanly or not) that path is unreachable.
    """
    __tracebackhide__ = True  # report the skip at the calling test, not here
    result = probe_optional(pkg)
    if result.available:
        pytest.skip(f"{pkg} is installed; import-error path unreachable")
    if result.installed:
        pytest.skip(result.reason)
