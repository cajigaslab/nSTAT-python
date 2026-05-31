"""Regression test: version strings agree across pyproject.toml, CITATION.cff,
docs/conf.py, and AGENT_GUIDE.md.

A loose contract — we deliberately do NOT also check the latest git tag,
because release tags are pushed AFTER the version bump commit lands.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_pyproject_version() -> str:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    return str(data["project"]["version"])


def _read_citation_version() -> str:
    text = (REPO_ROOT / "CITATION.cff").read_text()
    data = yaml.safe_load(text)
    return str(data["version"])


_CONF_RELEASE_RE = re.compile(r'^release\s*=\s*[\'"]([^\'"]+)[\'"]', re.MULTILINE)
_AGENT_VERSION_RE = re.compile(r"Package version:\s*([0-9][^.\s]*(?:\.[^.\s]+)*)\b")


def _read_conf_release() -> str:
    """The Sphinx ``release`` string in ``docs/conf.py``."""
    text = (REPO_ROOT / "docs" / "conf.py").read_text()
    m = _CONF_RELEASE_RE.search(text)
    assert m is not None, "docs/conf.py: could not find `release = \"...\"` line"
    return m.group(1)


def _read_agent_guide_version() -> str:
    """The ``Package version: X.Y.Z`` string in ``AGENT_GUIDE.md``."""
    text = (REPO_ROOT / "AGENT_GUIDE.md").read_text()
    m = _AGENT_VERSION_RE.search(text)
    assert m is not None, "AGENT_GUIDE.md: could not find `Package version: ...` line"
    return m.group(1).rstrip(".")


def test_pyproject_and_citation_versions_agree() -> None:
    """``pyproject.toml`` and ``CITATION.cff`` must declare the same version.

    The version string is the single most user-visible metadata field after
    the package name.  When they disagree, ``pip show`` and academic
    citations diverge — a confusing experience for downstream users.
    """
    py_version = _read_pyproject_version()
    cff_version = _read_citation_version()
    assert py_version == cff_version, (
        f"Version drift: pyproject.toml says {py_version!r} but "
        f"CITATION.cff says {cff_version!r}.  Update both."
    )


def test_pyproject_version_is_pep440() -> None:
    """``pyproject.toml`` version must satisfy PEP 440 (e.g. ``0.3.1``)."""
    from packaging.version import InvalidVersion, Version

    try:
        Version(_read_pyproject_version())
    except InvalidVersion as exc:  # pragma: no cover - defensive
        pytest.fail(f"pyproject.toml version is not PEP 440-compliant: {exc}")


def test_docs_conf_release_matches_pyproject() -> None:
    """``docs/conf.py``'s ``release`` must match ``pyproject.toml``.

    Sphinx prints the ``release`` string in the rendered docs header.
    Silent drift between the package and the published docs has bitten
    the project before (the conf.py was on 0.3.1 while the package was
    on 0.3.2), and there was no guard for it.
    """
    py_version = _read_pyproject_version()
    conf_version = _read_conf_release()
    assert py_version == conf_version, (
        f"Version drift: pyproject.toml says {py_version!r} but "
        f"docs/conf.py release = {conf_version!r}.  Bump both."
    )


def test_agent_guide_version_matches_pyproject() -> None:
    """The ``Package version: X.Y.Z`` header in ``AGENT_GUIDE.md`` must
    match ``pyproject.toml`` — otherwise the agent-facing orientation
    document misleads downstream AI tools about which API they're seeing.
    """
    py_version = _read_pyproject_version()
    guide_version = _read_agent_guide_version()
    assert py_version == guide_version, (
        f"Version drift: pyproject.toml says {py_version!r} but "
        f"AGENT_GUIDE.md says Package version: {guide_version!r}.  Update both."
    )


def test_release_notes_mentions_current_version() -> None:
    """``RELEASE_NOTES.md`` must include an entry for the current version.

    Catches the failure mode of bumping ``pyproject.toml`` / ``CITATION.cff``
    but forgetting to add a corresponding release-notes section.
    """
    notes_path = REPO_ROOT / "RELEASE_NOTES.md"
    if not notes_path.exists():
        pytest.skip("RELEASE_NOTES.md not present yet")
    text = notes_path.read_text()
    version = _read_pyproject_version()
    assert f"## v{version}" in text, (
        f"RELEASE_NOTES.md has no '## v{version}' section — bump the "
        f"version OR add release notes."
    )
