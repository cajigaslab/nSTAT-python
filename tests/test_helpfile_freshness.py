"""Tests for ``tools/check_helpfile_freshness.py`` + an enforcement smoke check.

The enforcement smoke check (``test_live_helpfiles_cover_all_public_symbols``)
is the durable artifact — every ``pytest`` run from the moment this PR
lands re-validates that every name in ``nstat.__all__`` is mentioned
in ``AGENT_GUIDE.md``, and every class is mentioned in
``docs/ClassDefinitions.md``.

The unit tests below pin the checker's individual behaviors against
synthetic fixtures so genuine logic bugs don't get masked by the
enforcement check passing for the wrong reason.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from textwrap import dedent

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "check_helpfile_freshness.py"

# Make the checker importable as a module (mirrors how the script itself
# self-patches sys.path).
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))


from check_helpfile_freshness import (  # noqa: E402
    _symbol_pattern,
    collect_public_symbols,
    find_missing_symbols,
    main,
    run_checks,
    symbol_is_class,
)


# ----------------------------------------------------------------------
# Enforcement smoke test — the durable artifact
# ----------------------------------------------------------------------

def test_live_helpfiles_cover_all_public_symbols() -> None:
    """Every name in ``nstat.__all__`` (and ``nstat.extras.__all__`` if
    present) must be mentioned in the hand-maintained helpfiles.

    When this fails, run ``make helpfile-check`` (or
    ``python tools/check_helpfile_freshness.py``) locally to see the
    specific missing symbols and add them to ``AGENT_GUIDE.md`` (and
    to ``docs/ClassDefinitions.md`` if they are classes).
    """
    findings = run_checks(REPO_ROOT)

    n_missing_ag = sum(len(v) for v in findings["agent_guide_missing"].values())
    n_missing_cd = sum(len(v) for v in findings["class_definitions_missing"].values())

    assert n_missing_ag == 0, (
        f"AGENT_GUIDE.md is missing {n_missing_ag} symbol(s): "
        f"{findings['agent_guide_missing']}"
    )
    assert n_missing_cd == 0, (
        f"docs/ClassDefinitions.md is missing {n_missing_cd} class(es): "
        f"{findings['class_definitions_missing']}"
    )


# ----------------------------------------------------------------------
# Public-surface discovery
# ----------------------------------------------------------------------

def test_collect_public_symbols_includes_core_namespace() -> None:
    surface = collect_public_symbols()
    assert "nstat" in surface
    # Sanity: a few load-bearing symbols.
    for required in ("SignalObj", "Trial", "DecodingAlgorithms", "LinearCIF"):
        assert required in surface["nstat"], (
            f"Expected {required!r} in nstat.__all__"
        )


def test_collect_public_symbols_includes_extras_when_populated() -> None:
    """When ``nstat.extras.__all__`` has entries, they appear in the surface."""
    surface = collect_public_symbols()
    # As of this commit, extras is an empty stub.  We only assert that
    # the function does NOT crash when extras is present but empty.
    extras_mod = importlib.import_module("nstat.extras")
    if getattr(extras_mod, "__all__", []):
        assert "nstat.extras" in surface


# ----------------------------------------------------------------------
# Symbol classification
# ----------------------------------------------------------------------

def test_symbol_is_class_detects_real_classes() -> None:
    assert symbol_is_class("nstat", "SignalObj") is True
    assert symbol_is_class("nstat", "nspikeTrain") is True
    assert symbol_is_class("nstat", "LinearCIF") is True


def test_symbol_is_class_rejects_functions() -> None:
    assert symbol_is_class("nstat", "psth") is False
    assert symbol_is_class("nstat", "fit_poisson_glm") is False
    assert symbol_is_class("nstat", "simulate_poisson_from_rate") is False


def test_symbol_is_class_rejects_exceptions() -> None:
    """Exceptions are technically classes but cataloged separately in api.rst."""
    assert symbol_is_class("nstat", "DataNotFoundError") is False
    assert symbol_is_class("nstat", "ParityValidationError") is False


def test_symbol_is_class_rejects_unknown_name() -> None:
    assert symbol_is_class("nstat", "NotARealSymbolXYZ") is False


# ----------------------------------------------------------------------
# Symbol pattern (word-boundary regex)
# ----------------------------------------------------------------------

def test_symbol_pattern_matches_whole_word_only() -> None:
    pat = _symbol_pattern("Signal")
    assert pat.search("Signal exists.")
    assert pat.search("- `Signal` — alias")
    assert pat.search("(Signal)")
    # Must NOT match substrings:
    assert not pat.search("SignalObj")
    assert not pat.search("SignalCollection")


def test_symbol_pattern_handles_underscores() -> None:
    pat = _symbol_pattern("fit_poisson_glm")
    assert pat.search("Use `fit_poisson_glm()` for...")
    assert not pat.search("fit_poisson_glm_extended")


# ----------------------------------------------------------------------
# Missing-symbols detection
# ----------------------------------------------------------------------

def test_find_missing_symbols_returns_unmentioned(tmp_path: Path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("# Doc\n\nDocumented: `Alpha`, `Beta`.\n")
    missing = find_missing_symbols(["Alpha", "Beta", "Gamma", "Delta"], target)
    assert missing == ["Gamma", "Delta"]


def test_find_missing_symbols_returns_all_when_file_missing(tmp_path: Path) -> None:
    """If the target file doesn't exist, every symbol is "missing"."""
    missing = find_missing_symbols(["Alpha", "Beta"], tmp_path / "ghost.md")
    assert missing == ["Alpha", "Beta"]


def test_find_missing_symbols_uses_word_boundary(tmp_path: Path) -> None:
    """``Signal`` matches as a whole word — NOT inside ``SignalObj``."""
    target = tmp_path / "doc.md"
    target.write_text("Only SignalObj is documented, not the plain class.")
    missing = find_missing_symbols(["Signal", "SignalObj"], target)
    assert missing == ["Signal"]


# ----------------------------------------------------------------------
# End-to-end CLI smoke test
# ----------------------------------------------------------------------

def test_main_exits_zero_on_clean_baseline(capsys) -> None:
    """The committed state must pass — same assertion as the enforcement test
    but exercises the CLI entry-point."""
    code = main(["--quiet"])
    assert code == 0
