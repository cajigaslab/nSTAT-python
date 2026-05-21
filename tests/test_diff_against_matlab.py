"""Tests for ``tools/parity/diff_against_matlab.py``.

Runs only when a MATLAB nSTAT checkout is detectable.  Off-CI machines
without the sibling checkout skip cleanly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = REPO_ROOT / "tools" / "parity"

if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))


from diff_against_matlab import (  # noqa: E402
    diff,
    inventory_matlab_checkout,
    inventory_matlab_file,
    load_parity_public_api,
    python_symbol_resolves,
    resolve_matlab_path,
)


def _matlab_path_available() -> bool:
    """Return True if a usable MATLAB checkout is reachable from this machine."""
    try:
        path = resolve_matlab_path(None)
    except FileNotFoundError:
        return False
    return path.exists() and any(path.glob("*.m"))


# ----------------------------------------------------------------------
# Unit tests that don't require the MATLAB repo
# ----------------------------------------------------------------------

def test_python_symbol_resolves_canonical_public_api() -> None:
    """Every entry the parity manifest claims as ``mapped`` must import."""
    api = load_parity_public_api()
    for row in api:
        if row.get("status") != "mapped":
            continue
        target = row.get("python_target")
        if not target:
            continue
        assert python_symbol_resolves(str(target)), (
            f"Parity manifest claims {target!r} is mapped, but the symbol "
            f"does not resolve via importlib.  Either the symbol moved or "
            f"the manifest is stale — run "
            f"`python tools/parity/diff_against_matlab.py` to confirm."
        )


def test_python_symbol_resolves_handles_null_targets() -> None:
    """``None`` / ``null`` / empty string must all return False, not crash."""
    assert python_symbol_resolves("") is False
    assert python_symbol_resolves("null") is False
    assert python_symbol_resolves("None") is False
    assert python_symbol_resolves("nstat.NoSuchSymbol12345") is False


def test_inventory_extracts_classdef_and_function(tmp_path: Path) -> None:
    """Regex-based extraction must catch both classdef and nested functions."""
    fixture = tmp_path / "SyntheticThing.m"
    fixture.write_text(
        "classdef SyntheticThing < handle\n"
        "    methods\n"
        "        function obj = SyntheticThing(x)\n"
        "            obj.x = x;\n"
        "        end\n"
        "        function r = doThing(obj, y)\n"
        "            r = obj.x + y;\n"
        "        end\n"
        "    end\n"
        "end\n",
        encoding="utf-8",
    )
    inv = inventory_matlab_file(fixture)
    assert inv.classes == ["SyntheticThing"]
    assert "SyntheticThing" in inv.functions  # constructor counts
    assert "doThing" in inv.functions
    assert inv.is_class_file is True


def test_inventory_ignores_function_signatures_in_comment_blocks(tmp_path: Path) -> None:
    """``%{ ... %}`` block comments must not contribute spurious functions."""
    fixture = tmp_path / "Foo.m"
    fixture.write_text(
        "classdef Foo\n"
        "    methods\n"
        "        function r = real_method()\n"
        "            r = 1;\n"
        "        end\n"
        "    end\n"
        "end\n"
        "%{\n"
        "function ghost = fakeMethod()\n"
        "    ghost = 0;\n"
        "end\n"
        "%}\n",
        encoding="utf-8",
    )
    inv = inventory_matlab_file(fixture)
    assert "real_method" in inv.functions
    assert "fakeMethod" not in inv.functions


# ----------------------------------------------------------------------
# Integration test — needs the MATLAB checkout
# ----------------------------------------------------------------------

@pytest.mark.skipif(not _matlab_path_available(),
                    reason="MATLAB nSTAT checkout not reachable")
def test_diff_reports_known_drift_against_matlab_v140() -> None:
    """LinearCIF (added in MATLAB v1.4.0) must surface as drift.

    This is a regression test for the diff machinery itself — if it ever
    stops detecting the LinearCIF gap, either the manifest was updated
    or the extractor broke.
    """
    matlab_root = resolve_matlab_path(None)
    matlab_inv = inventory_matlab_checkout(matlab_root)
    parity_api = load_parity_public_api()
    entries = diff(matlab_inv, parity_api)

    new_in_matlab = [e for e in entries if e.category == "new_in_matlab"]
    new_names = {e.matlab_class for e in new_in_matlab}
    # LinearCIF is the v1.4.0 addition with no Python equivalent yet.
    # Once it's ported, this test will need updating — the diff tool's
    # findings are the bellwether for parity gaps.
    if "LinearCIF" in new_names:
        # Gap still present — the tool found it correctly.
        return
    # If LinearCIF has been ported, parity/manifest.yml should now claim
    # it and this assertion should pass that way too:
    parity_by_class = {row.get("matlab"): row for row in parity_api}
    assert "LinearCIF" in parity_by_class, (
        "LinearCIF is neither flagged as new_in_matlab nor present in "
        "parity/manifest.yml — the diff tool may have regressed."
    )
