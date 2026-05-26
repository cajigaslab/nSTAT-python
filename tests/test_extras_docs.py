"""Drift guards for ``docs/extras/*.md`` help files.

Two contracts:

1. Every ``nstat.extras.<subpackage>.<module>.py`` bridge must have a
   matching ``docs/extras/<subpackage>_<module>.md`` help file (where
   ``_bridge`` suffix is stripped to match the doc naming convention).
2. Every help file must be referenced in ``docs/extras.rst``'s toctree
   so it actually renders into the published Sphinx site.

Together these catch the bug class where a new bridge ships but no
human-readable docs are added — the Sphinx auto-summary stub renders
parameter signatures but no narrative usage guide, install command, or
gotchas.
"""
from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRAS_DOCS_DIR = REPO_ROOT / "docs" / "extras"
EXTRAS_RST = REPO_ROOT / "docs" / "extras.rst"
EXTRAS_PKG = REPO_ROOT / "nstat" / "extras"


# Map: bridge module stem → expected docs filename stem (without .md).
# When a module's natural doc name differs from its filename (e.g.,
# nemos_bridge.py → validation_nemos), declare the mapping here.
EXPECTED_DOC_STEM_FOR_BRIDGE = {
    "neo": "interop_neo",
    "pynapple": "interop_pynapple",
    "nwb": "interop_nwb",
    "nemos_bridge": "validation_nemos",
    "pykalman_bridge": "validation_pykalman",
    "statsmodels_bridge": "validation_statsmodels",
    "nitime_bridge": "validation_nitime",
    "spike_distances": "metrics_spike_distances",
    "dynamax": "em_dynamax",
    "dynamax_bridge": "em_dynamax",
}


def _bridge_modules() -> list[Path]:
    """All concrete bridge modules under nstat/extras/, excluding helpers."""
    return [
        p for p in EXTRAS_PKG.rglob("*.py")
        if p.name != "__init__.py" and not p.stem.startswith("_")
    ]


def test_every_bridge_has_a_help_file() -> None:
    """For each concrete extras bridge, ``docs/extras/<stem>.md`` exists."""
    missing: list[tuple[str, str]] = []
    for bridge in _bridge_modules():
        expected = EXPECTED_DOC_STEM_FOR_BRIDGE.get(bridge.stem)
        if expected is None:
            missing.append((str(bridge.relative_to(REPO_ROOT)), "(unmapped)"))
            continue
        doc_path = EXTRAS_DOCS_DIR / f"{expected}.md"
        if not doc_path.exists():
            missing.append(
                (
                    str(bridge.relative_to(REPO_ROOT)),
                    str(doc_path.relative_to(REPO_ROOT)),
                )
            )
    assert not missing, (
        f"Extras bridges without a docs/extras/*.md help file: {missing}. "
        f"Either add the help file or update EXPECTED_DOC_STEM_FOR_BRIDGE "
        f"in this test."
    )


def test_every_help_file_is_in_extras_rst_toctree() -> None:
    """Every ``docs/extras/*.md`` must be referenced in ``docs/extras.rst``.

    Sphinx will silently skip an orphan .md file — it builds but doesn't
    appear in the nav, which defeats the purpose of writing the help.
    """
    if not EXTRAS_RST.exists():
        pytest.skip("docs/extras.rst not present")

    rst_text = EXTRAS_RST.read_text(encoding="utf-8")
    orphans: list[str] = []
    for help_md in EXTRAS_DOCS_DIR.glob("*.md"):
        # The toctree entry uses ``extras/<stem>`` (no extension).
        toctree_entry = f"extras/{help_md.stem}"
        if toctree_entry not in rst_text:
            orphans.append(str(help_md.relative_to(REPO_ROOT)))
    assert not orphans, (
        f"docs/extras/*.md files not in docs/extras.rst toctree: "
        f"{orphans}.  Add ``extras/<stem>`` lines to the toctree."
    )


def test_extras_summary_html_exists_and_is_self_contained() -> None:
    """``docs/extras_summary.html`` ships as a self-contained landing page
    (embedded CSS, no Sphinx wrap) via ``html_extra_path`` in conf.py.

    Contract:
    - File exists.
    - References every shipped bridge subpackage at least once.
    - Is registered in docs/conf.py's html_extra_path.
    """
    summary = REPO_ROOT / "docs" / "extras_summary.html"
    assert summary.exists(), "docs/extras_summary.html missing"

    text = summary.read_text(encoding="utf-8")
    # Every shipped bridge must appear somewhere on the page.
    REQUIRED_BRIDGES = (
        "nstat.extras.interop.neo",
        "nstat.extras.interop.pynapple",
        "nstat.extras.interop.nwb",
        "nstat.extras.validation.nemos_bridge",
        "nstat.extras.validation.pykalman_bridge",
        "nstat.extras.metrics.spike_distances",
    )
    missing = [b for b in REQUIRED_BRIDGES if b not in text]
    assert not missing, (
        f"extras_summary.html is missing references to: {missing}. "
        f"Add a bridge-card for each new shipped bridge."
    )

    # Sphinx must be configured to copy it into the build root.
    conf = (REPO_ROOT / "docs" / "conf.py").read_text(encoding="utf-8")
    assert "extras_summary.html" in conf, (
        "docs/conf.py must list extras_summary.html under html_extra_path "
        "so Sphinx copies it into the published Pages site."
    )


def test_readme_links_to_extras_summary_html() -> None:
    """The README's Related Python projects section must surface the
    visual summary page so users can find it from the GitHub landing.
    """
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "extras_summary.html" in readme, (
        "README.md must link to extras_summary.html in the "
        "'Related Python projects' section."
    )


def test_every_help_file_links_to_its_example_script() -> None:
    """Each help file must link to the corresponding example script.

    Keeps the docs and the demos paired — when a bridge evolves, the
    example and the help file evolve together.
    """
    # Map: doc stem → expected example script stem.
    DOC_TO_EXAMPLE = {
        "interop_neo": "interop_neo_demo",
        "interop_pynapple": "interop_pynapple_demo",
        "interop_nwb": "interop_nwb_demo",
        "validation_nemos": "validation_nemos_demo",
        "validation_pykalman": "validation_pykalman_demo",
        "metrics_spike_distances": "metrics_spike_distances_demo",
        "em_dynamax": "em_dynamax_demo",
    }
    missing_links: list[tuple[str, str]] = []
    for doc_stem, example_stem in DOC_TO_EXAMPLE.items():
        doc_path = EXTRAS_DOCS_DIR / f"{doc_stem}.md"
        if not doc_path.exists():
            continue
        text = doc_path.read_text(encoding="utf-8")
        if f"examples/extras/{example_stem}.py" not in text:
            missing_links.append((doc_stem, example_stem))
    assert not missing_links, (
        f"Help files missing a reference to their example script: "
        f"{missing_links}.  Each docs/extras/X.md should link to "
        f"examples/extras/<corresponding>_demo.py."
    )
