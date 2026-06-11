"""Repository structure-hygiene guards.

These tests lock in the *simple, non-redundant* file layout so that the
specific redundancies cleaned up during the structure simplification cannot
silently creep back in.  Each test encodes one rule about where things live;
the module docstring of each rule explains the canonical structure it
protects, so this file doubles as living documentation of the layout.

Canonical layout (the single home for each concern):

    nstat/                  the packaged Python toolbox (MATLAB-parity surface)
    notebooks/              the ONE source-of-truth for parity notebooks,
                            catalogued 1:1 in tools/notebook_build/notebook_manifest.yml
    tools/notebook_build/        notebook tooling (generators, runners, manifests)
    docs/notebook_galleries/         GENERATED galleries (output; not source notebooks)
    examples/paper/         the MATLAB-faithful paper-example scripts (drive the
                            committed docs/figures/example0N/ with CI parity)
    parity/                 the live MATLAB-parity manifests + reports

The rules below are deliberately strict: they fail loudly on the exact
patterns that previously caused confusion (duplicate paper wrappers, a second
notebook registry, committed Simulink binaries, dated one-off diff snapshots,
notebooks scattered outside the canonical directory).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _tracked_files() -> list[str]:
    """Git-tracked paths only — ignores local build output / untracked junk."""
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def test_source_notebooks_live_only_in_canonical_dirs() -> None:
    """Source ``.ipynb`` files belong under ``notebooks/`` (the parity-notebook
    system) or ``examples/`` (standalone example notebooks).  This prevents
    notebook sprawl into new ad-hoc directories.  ``docs/notebook_galleries/`` holds
    GENERATED galleries (README.md + PNG), never source ``.ipynb``.
    """
    allowed_prefixes = ("notebooks/", "examples/")
    stray = [
        f
        for f in _tracked_files()
        if f.endswith(".ipynb") and not f.startswith(allowed_prefixes)
    ]
    assert not stray, (
        "Source notebooks must live under notebooks/ or examples/; "
        f"found stray notebooks: {stray}"
    )


def test_notebook_manifest_matches_disk_one_to_one() -> None:
    """``tools/notebook_build/notebook_manifest.yml`` is the single authoritative
    notebook catalogue and must stay exactly in sync with ``notebooks/*.ipynb``
    — no dangling entries, no uncatalogued notebooks.
    """
    manifest = yaml.safe_load(
        (REPO_ROOT / "tools" / "notebook_build" / "notebook_manifest.yml").read_text(encoding="utf-8")
    )
    catalogued = {row["topic"] for row in manifest["notebooks"]}
    on_disk = {p.stem for p in (REPO_ROOT / "notebooks").glob("*.ipynb")}
    assert catalogued == on_disk, (
        "notebook_manifest.yml and notebooks/ have drifted.\n"
        f"  on disk but uncatalogued: {sorted(on_disk - catalogued)}\n"
        f"  catalogued but missing:   {sorted(catalogued - on_disk)}"
    )


def test_single_notebook_registry() -> None:
    """There is exactly one notebook registry (the manifest above).  The legacy
    duplicate ``examples/nSTATPaperExamples/manifest.yml`` must not return.
    """
    assert not (REPO_ROOT / "examples" / "nSTATPaperExamples").exists(), (
        "examples/nSTATPaperExamples/ is a removed duplicate notebook registry; "
        "the single source of truth is tools/notebook_build/notebook_manifest.yml"
    )


def test_single_canonical_paper_examples_entry() -> None:
    """The paper examples have one canonical home: ``examples/paper/`` (the
    MATLAB-faithful scripts) plus the ``nstat-paper-examples`` console script
    (``nstat.paper_examples:main``).  The thin ``examples/`` launcher wrappers
    (``nSTATPaperExamples.py`` / ``nstat_paper_examples.py``) are removed
    duplicates of that console script and must not return.
    """
    for stale in ("nSTATPaperExamples.py", "nstat_paper_examples.py"):
        assert not (REPO_ROOT / "examples" / stale).exists(), (
            f"examples/{stale} duplicates the `nstat-paper-examples` console "
            "script; use examples/paper/ scripts or the console entry point"
        )


def test_no_committed_simulink_or_binary_build_artifacts() -> None:
    """MATLAB/Simulink build artifacts (``*.slxc``, ``slprj/``) are local
    intermediates and must never be tracked in the repo.
    """
    offenders = [
        f
        for f in _tracked_files()
        if f.endswith(".slxc") or "/slprj/" in f or f.startswith("slprj/")
    ]
    assert not offenders, f"Simulink build artifacts must not be tracked: {offenders}"


def test_no_dated_one_off_parity_snapshots() -> None:
    """Parity diffs are regenerable via ``tools/parity/diff_against_matlab.py``;
    dated one-off snapshots (e.g. ``parity/matlab_diff_2026-05-21.md``) should
    not accumulate in the tree.
    """
    snapshots = sorted((REPO_ROOT / "parity").glob("matlab_diff_*.md"))
    assert not snapshots, (
        "Dated parity-diff snapshots should not be committed (regenerate via "
        f"tools/parity/diff_against_matlab.py): {[p.name for p in snapshots]}"
    )


def test_no_tracked_python_or_build_caches() -> None:
    """Build / cache output is regenerable and gitignored; it must never be
    tracked.  Guards against accidentally committing __pycache__, compiled
    bytecode, packaging metadata, or the Sphinx build/autosummary output.
    """
    offenders = [
        f
        for f in _tracked_files()
        if (
            "__pycache__/" in f
            or f.endswith(".pyc")
            or ".egg-info/" in f
            or f.startswith("docs/_build/")
            or f.startswith("docs/_autosummary/")
            or ".pytest_cache/" in f
        )
    ]
    assert not offenders, f"Build/cache artifacts must not be tracked: {offenders}"


# Largest legitimately-tracked files are committed figure PNGs (~2.5 MB). The
# figshare neural dataset is distributed separately (tens-to-hundreds of MB),
# so a generous 5 MB ceiling blocks committing dataset dumps while allowing all
# small fixtures (the data_cache offline fixture, nstat/data, MATLAB gold .mat).
_MAX_TRACKED_FILE_KB = 5 * 1024


def test_no_oversized_tracked_files() -> None:
    """No single tracked file may exceed the size ceiling. This is the guard
    against committing the figshare dataset (which ships via ``nstat-install``)
    or other large binary dumps, independent of file extension.
    """
    offenders = []
    for f in _tracked_files():
        p = REPO_ROOT / f
        try:
            kb = p.stat().st_size // 1024
        except OSError:
            continue
        if kb > _MAX_TRACKED_FILE_KB:
            offenders.append(f"{f} ({kb // 1024} MB)")
    assert not offenders, (
        f"Tracked files exceed the {_MAX_TRACKED_FILE_KB // 1024} MB ceiling "
        f"(large data ships via figshare/nstat-install, not git): {offenders}"
    )


def test_no_agent_artifact_directories() -> None:
    """One-off AI-agent scratch dirs (e.g. docs/superpowers/) don't belong in
    the published tree; design notes live in their proper home.
    """
    assert not (REPO_ROOT / "docs" / "superpowers").exists(), (
        "docs/superpowers/ is an AI-agent artifact dir and was removed"
    )


def test_generated_galleries_hold_no_source_notebooks() -> None:
    """docs/notebook_galleries/ is GENERATED output (README.md + PNGs). Source
    notebooks live only in notebooks/; a stray .ipynb here means the source vs
    generated boundary has blurred.
    """
    galleries = REPO_ROOT / "docs" / "notebook_galleries"
    stray = [p.name for p in galleries.rglob("*.ipynb")] if galleries.exists() else []
    assert not stray, f"docs/notebook_galleries/ must not contain source notebooks: {stray}"
