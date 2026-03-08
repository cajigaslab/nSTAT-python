from __future__ import annotations

import inspect

from nstat.class_fidelity import EXPECTED_RUNTIME_MEMBERS, resolve_public_symbol


def test_expected_matlab_symbol_surface_exists_and_is_callable() -> None:
    for public_name, expected in EXPECTED_RUNTIME_MEMBERS.items():
        obj = resolve_public_symbol(public_name)
        assert obj is not None, f"{public_name} does not resolve on the Python public surface"
        missing = sorted(name for name in expected if not callable(getattr(obj, name, None)))
        assert not missing, f"{public_name} is missing MATLAB-facing callables: {missing}"


def test_expected_symbol_surface_has_python_runtime_signatures() -> None:
    for public_name, expected in EXPECTED_RUNTIME_MEMBERS.items():
        obj = resolve_public_symbol(public_name)
        assert obj is not None, f"{public_name} does not resolve on the Python public surface"
        for name in expected:
            signature = inspect.signature(getattr(obj, name))
            assert signature is not None
