"""Regression test: version strings agree across pyproject.toml and CITATION.cff.

A loose contract — we deliberately do NOT also check the latest git tag,
because release tags are pushed AFTER the version bump commit lands.
"""
from __future__ import annotations

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
