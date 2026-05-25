"""Unit tests for ``nstat.extras._lazy.require_optional``.

The helper centralises the "lazy import or actionable ImportError"
pattern that every ``nstat.extras.*`` bridge uses.  These tests pin
the contract so the per-bridge import-hint tests
(``test_extras_namespace.py::*_emits_install_hint_when_missing``)
can rely on a single canonical error format.
"""
from __future__ import annotations

import pytest

from nstat.extras._lazy import require_optional, require_optionals


def test_require_optional_returns_module_when_installed() -> None:
    """Happy path: numpy is always installed; helper returns the module."""
    np_mod = require_optional("numpy", install_key="dev")
    assert np_mod.__name__ == "numpy"
    assert hasattr(np_mod, "array")


def test_require_optional_raises_actionable_error_when_missing() -> None:
    """ImportError message MUST embed the pip-install hint."""
    with pytest.raises(ImportError) as excinfo:
        require_optional(
            "this_package_does_not_exist_xyz_12345",
            install_key="some-extras-key",
        )
    msg = str(excinfo.value)
    assert "this_package_does_not_exist_xyz_12345" in msg
    assert "pip install nstat-toolbox[some-extras-key]" in msg
    assert "nstat.extras" in msg


def test_require_optional_chains_original_importerror() -> None:
    """``raise ... from e`` must preserve the original exception."""
    with pytest.raises(ImportError) as excinfo:
        require_optional("does_not_exist_abc", install_key="x")
    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_require_optionals_returns_tuple_in_order() -> None:
    """Multi-package import returns modules in argument order."""
    np_mod, sys_mod = require_optionals("numpy", "sys", install_key="dev")
    assert np_mod.__name__ == "numpy"
    assert sys_mod.__name__ == "sys"


def test_require_optionals_short_circuits_on_first_missing() -> None:
    """If the first package is absent, the second is never attempted —
    the error message names the first missing package."""
    with pytest.raises(ImportError) as excinfo:
        require_optionals("does_not_exist_aaa", "numpy", install_key="dev")
    msg = str(excinfo.value)
    assert "does_not_exist_aaa" in msg
    assert "numpy" not in msg


def test_install_hint_format_is_canonical() -> None:
    """Locks in the exact ``pip install nstat-toolbox[KEY]`` shape so
    the per-bridge assertion helper can keep relying on substring match.
    """
    with pytest.raises(ImportError) as excinfo:
        require_optional("nonexistent_zzz", install_key="my-key")
    assert "pip install nstat-toolbox[my-key]" in str(excinfo.value)
