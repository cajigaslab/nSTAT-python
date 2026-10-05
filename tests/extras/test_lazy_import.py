"""Unit tests for ``nstat.extras._lazy.require_optional``.

The helper centralises the "lazy import or actionable ImportError"
pattern that every ``nstat.extras.*`` bridge uses.  These tests pin
the contract so the per-bridge import-hint tests
(``test_extras_namespace.py::*_emits_install_hint_when_missing``)
can rely on a single canonical error format.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nstat.extras import _lazy
from nstat.extras._lazy import require_optional, require_optionals

_ABI_TEXT = (
    "numpy.dtype size changed, may indicate binary incompatibility. "
    "Expected 96 from C header, got 88 from PyObject"
)


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


# ----------------------------------------------------------------------
# Installed-but-broken packages (binary/ABI incompatibility)
# ----------------------------------------------------------------------


def _write_package(root: Path, name: str, init_body: str) -> None:
    pkg = root / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text(init_body, encoding="utf-8")


def _assert_broken_message(msg: str, package: str, exc_text: str, key: str) -> None:
    assert f"nstat.extras requires the {package!r} package" in msg
    assert "installed but failed to import" in msg
    assert exc_text in msg
    assert "likely a binary/ABI incompatibility" in msg
    assert f"pip install nstat-toolbox[{key}]" in msg
    assert "not installed" not in msg


def test_require_optional_reports_abi_broken_package_monkeypatched(monkeypatch) -> None:
    """An installed package whose import raises ValueError (the h5py /
    NumPy ABI failure mode) must surface as an actionable ImportError,
    not the raw ValueError, with the ValueError chained."""
    original = ValueError(_ABI_TEXT)

    def broken_import(name, package=None):
        raise original

    # numpy is installed, so find_spec("numpy") is real and non-None.
    monkeypatch.setattr(_lazy, "import_module", broken_import)
    with pytest.raises(ImportError) as excinfo:
        require_optional("numpy", install_key="nwb")
    _assert_broken_message(str(excinfo.value), "numpy", f"ValueError: {_ABI_TEXT}", "nwb")
    assert excinfo.value.__cause__ is original


def test_require_optional_reports_abi_broken_package_on_disk(tmp_path, monkeypatch) -> None:
    """Same contract through the real import machinery: a package on
    sys.path whose ``__init__`` raises ValueError.  A dotted request
    (``pkg.sub``) is classified by the top-level package, which
    ``find_spec`` locates without executing it."""
    name = "nstat_fake_abi_broken_pkg_q7"
    _write_package(tmp_path, name, f"raise ValueError({_ABI_TEXT!r})\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for requested in (name, f"{name}.sub"):
        with pytest.raises(ImportError) as excinfo:
            require_optional(requested, install_key="nwb")
        _assert_broken_message(str(excinfo.value), requested, f"ValueError: {_ABI_TEXT}", "nwb")
        assert isinstance(excinfo.value.__cause__, ValueError)
        assert str(excinfo.value.__cause__) == _ABI_TEXT
        assert name not in sys.modules


def test_require_optional_reports_installed_package_with_missing_dependency(
    tmp_path, monkeypatch
) -> None:
    """A present package that fails on one of ITS imports is installed
    but broken, not "not installed"."""
    name = "nstat_fake_pkg_missing_dep_q7"
    _write_package(tmp_path, name, "import nstat_fake_absent_dependency_q7\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ImportError) as excinfo:
        require_optional(name, install_key="x")
    _assert_broken_message(
        str(excinfo.value), name,
        "ModuleNotFoundError: No module named 'nstat_fake_absent_dependency_q7'", "x",
    )
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)


def test_require_optional_missing_submodule_is_not_installed(tmp_path, monkeypatch) -> None:
    """A missing submodule of a present package keeps the canonical
    "not installed" message."""
    name = "nstat_fake_pkg_without_sub_q7"
    _write_package(tmp_path, name, "")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ImportError) as excinfo:
        require_optional(f"{name}.sub", install_key="x")
    assert str(excinfo.value) == _lazy._build_error_message(f"{name}.sub", "x")
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)


def test_require_optional_blocked_module_is_not_installed(monkeypatch) -> None:
    """``sys.modules[name] = None`` (how the bridge tests simulate a
    missing dep) is reported as not installed."""
    monkeypatch.setitem(sys.modules, "numpy_blocked_alias_q7", None)
    with pytest.raises(ImportError) as excinfo:
        require_optional("numpy_blocked_alias_q7", install_key="x")
    assert str(excinfo.value) == _lazy._build_error_message("numpy_blocked_alias_q7", "x")


# ----------------------------------------------------------------------
# Bridges that used to hand-roll their own gate now route through
# require_optional (and so inherit the installed-but-broken handling).
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bridge", "gate", "blocked", "package", "key"),
    [
        ("nstat.extras.em.dynamax_bridge", "_require_dynamax",
         ("dynamax",), "dynamax", "dynamax"),
        ("nstat.extras.validation.statsmodels_bridge", "_require_statsmodels",
         ("statsmodels", "statsmodels.api"), "statsmodels.api", "test-parity"),
    ],
)
def test_migrated_bridge_gates_use_canonical_message(
    monkeypatch, bridge, gate, blocked, package, key
) -> None:
    import importlib

    mod = importlib.import_module(bridge)
    for name in blocked:
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError) as excinfo:
        getattr(mod, gate)()
    assert str(excinfo.value) == _lazy._build_error_message(package, key)
    assert isinstance(excinfo.value.__cause__, ImportError)


@pytest.mark.parametrize(
    ("bridge", "gate", "package", "key"),
    [
        ("nstat.extras.em.dynamax_bridge", "_require_dynamax", "dynamax", "dynamax"),
        ("nstat.extras.validation.statsmodels_bridge", "_require_statsmodels",
         "statsmodels.api", "test-parity"),
    ],
)
def test_migrated_bridge_gates_report_abi_broken_dependency(
    monkeypatch, bridge, gate, package, key
) -> None:
    import importlib
    import importlib.util

    mod = importlib.import_module(bridge)
    if importlib.util.find_spec(package.partition(".")[0]) is None:
        pytest.skip(f"{package} is not installed; the installed-but-broken path is unreachable")
    original = ValueError(_ABI_TEXT)

    def broken_import(name, package=None):
        raise original

    monkeypatch.setattr(_lazy, "import_module", broken_import)
    with pytest.raises(ImportError) as excinfo:
        getattr(mod, gate)()
    _assert_broken_message(str(excinfo.value), package, f"ValueError: {_ABI_TEXT}", key)
    assert excinfo.value.__cause__ is original


def test_pynapple_bridge_reports_broken_lazy_backend(monkeypatch) -> None:
    """pynapple loads its submodules lazily: ``import pynapple`` succeeds even
    when its pandas backend is ABI-broken. The bridge must force the core import
    so the user gets the actionable "installed but failed to import" message
    rather than ``AttributeError: '_LazyModule' object has no attribute 'Ts'``."""
    import importlib
    import importlib.util

    if importlib.util.find_spec("pynapple") is None:
        pytest.skip("pynapple is not installed; the installed-but-broken path is unreachable")
    from nstat.extras.interop import pynapple as bridge

    real_import = _lazy.import_module
    original = ImportError("numpy.core.multiarray failed to import")

    def lazy_backend_broken(name, package=None):
        if name == "pynapple.core":
            raise original
        return real_import(name, package)

    monkeypatch.setattr(_lazy, "import_module", lazy_backend_broken)
    with pytest.raises(ImportError) as excinfo:
        bridge._require_pynapple()
    _assert_broken_message(
        str(excinfo.value), "pynapple.core",
        "ImportError: numpy.core.multiarray failed to import", "pynapple",
    )
    assert excinfo.value.__cause__ is original
