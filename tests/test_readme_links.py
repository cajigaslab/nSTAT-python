"""Tests for ``tools/check_readme_links.py`` + a smoke check on the live README.

The smoke check is the *enforcement* test: every time CI runs, it
re-validates that ``README.md`` has no broken intra-repo links, no
missing images, and no stale code-snippet imports.  The unit tests
below pin the checker's individual behaviors against tiny fixture files.
"""
from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "check_readme_links.py"

# Make the checker importable as a module (mirrors how the script self-
# patches sys.path when invoked directly).
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))


from check_readme_links import (  # noqa: E402
    check_code_snippet_imports,
    check_image_existence,
    check_intra_repo_links,
    main,
)


# ----------------------------------------------------------------------
# Enforcement smoke test — the *whole point* of this file
# ----------------------------------------------------------------------

def test_live_readme_has_no_broken_links_or_imports() -> None:
    """The committed README.md must be self-consistent at all times.

    This is the CI gate that makes "no stale README" enforceable.  When
    it fails, run ``python tools/check_readme_links.py`` locally to see
    the specific findings.
    """
    readme = REPO_ROOT / "README.md"
    assert readme.exists(), "README.md is required at repo root"

    broken_links = check_intra_repo_links(readme, REPO_ROOT)
    broken_images = check_image_existence(readme, REPO_ROOT)
    broken_imports = check_code_snippet_imports(readme)

    assert not broken_links, f"Broken intra-repo links in README.md: {broken_links}"
    assert not broken_images, f"Broken images in README.md: {broken_images}"
    assert not broken_imports, f"Broken code-snippet imports in README.md: {broken_imports}"


# ----------------------------------------------------------------------
# Unit tests: intra-repo link checker
# ----------------------------------------------------------------------

class TestIntraRepoLinks:
    def test_resolves_existing_relative_file(self, tmp_path: Path) -> None:
        (tmp_path / "exists.md").write_text("ok")
        readme = tmp_path / "README.md"
        readme.write_text("See [the doc](exists.md) for details.")
        assert check_intra_repo_links(readme, tmp_path) == []

    def test_flags_missing_relative_file(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text("See [the doc](ghost.md) for details.")
        broken = check_intra_repo_links(readme, tmp_path)
        assert broken == [("the doc", "ghost.md")]

    def test_skips_external_https_urls(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text("[PyPI](https://pypi.org/project/nstat-toolbox/)")
        assert check_intra_repo_links(readme, tmp_path) == []

    def test_skips_mailto_anchors_and_in_page(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text(
            "[Email](mailto:foo@bar.com) [Top](#top) [Page](http://x)"
        )
        assert check_intra_repo_links(readme, tmp_path) == []

    def test_strips_anchor_fragment_before_checking_file(self, tmp_path: Path) -> None:
        """``foo.md#section`` is broken only if ``foo.md`` is missing."""
        (tmp_path / "foo.md").write_text("# Section\n")
        readme = tmp_path / "README.md"
        readme.write_text("[Sec](foo.md#section) [Bad](ghost.md#x)")
        broken = check_intra_repo_links(readme, tmp_path)
        assert broken == [("Bad", "ghost.md#x")]

    def test_deduplicates_repeated_targets(self, tmp_path: Path) -> None:
        """If the same broken target appears N times, only report it once."""
        readme = tmp_path / "README.md"
        readme.write_text("[A](ghost.md) [B](ghost.md) [C](ghost.md)")
        broken = check_intra_repo_links(readme, tmp_path)
        assert len(broken) == 1
        # First-encountered link text wins.
        assert broken[0] == ("A", "ghost.md")

    def test_ignores_image_syntax(self, tmp_path: Path) -> None:
        """``![alt](path)`` is handled by check_image_existence, not this checker."""
        readme = tmp_path / "README.md"
        readme.write_text("![missing](ghost.png)")
        assert check_intra_repo_links(readme, tmp_path) == []


# ----------------------------------------------------------------------
# Unit tests: image existence
# ----------------------------------------------------------------------

class TestImageExistence:
    def test_flags_missing_image(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text("![demo](docs/figures/missing.png)")
        broken = check_image_existence(readme, tmp_path)
        assert broken == [("demo", "docs/figures/missing.png")]

    def test_accepts_existing_image(self, tmp_path: Path) -> None:
        img = tmp_path / "logo.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG magic bytes; just needs to exist
        readme = tmp_path / "README.md"
        readme.write_text("![logo](logo.png)")
        assert check_image_existence(readme, tmp_path) == []

    def test_skips_external_image_urls(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text("![badge](https://img.shields.io/foo.svg)")
        assert check_image_existence(readme, tmp_path) == []


# ----------------------------------------------------------------------
# Unit tests: code-snippet imports
# ----------------------------------------------------------------------

class TestCodeSnippetImports:
    def test_accepts_resolving_from_import(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text(dedent("""
            Setup:

            ```python
            from nstat import nspikeTrain, Trial
            ```
        """).strip())
        assert check_code_snippet_imports(readme) == []

    def test_flags_nonexistent_from_import(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text(dedent("""
            ```python
            from nstat import NotARealSymbol12345
            ```
        """).strip())
        broken = check_code_snippet_imports(readme)
        assert len(broken) == 1
        assert "NotARealSymbol12345" in broken[0][1]

    def test_accepts_resolving_plain_import(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text(dedent("""
            ```python
            import nstat
            ```
        """).strip())
        assert check_code_snippet_imports(readme) == []

    def test_flags_nonexistent_submodule_import(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text(dedent("""
            ```python
            import nstat.ghost_submodule_xyz
            ```
        """).strip())
        broken = check_code_snippet_imports(readme)
        assert len(broken) == 1
        assert "ghost_submodule_xyz" in broken[0][1]

    def test_ignores_non_python_fenced_blocks(self, tmp_path: Path) -> None:
        """A ``bash`` or ``yaml`` fence containing ``from nstat import X`` text is irrelevant."""
        readme = tmp_path / "README.md"
        readme.write_text(dedent("""
            ```bash
            from nstat import not_python
            ```

            ```
            from nstat import also_not_python
            ```
        """).strip())
        assert check_code_snippet_imports(readme) == []

    def test_handles_aliased_imports(self, tmp_path: Path) -> None:
        """``from nstat import X as Y`` should validate ``X``, not ``Y``."""
        readme = tmp_path / "README.md"
        readme.write_text(dedent("""
            ```python
            from nstat import nspikeTrain as nst
            ```
        """).strip())
        assert check_code_snippet_imports(readme) == []


# ----------------------------------------------------------------------
# CLI smoke test
# ----------------------------------------------------------------------

def test_main_exits_zero_on_clean_readme(tmp_path: Path, capsys) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# Title\n\nNo links.")
    code = main(["--readme", str(readme), "--quiet"])
    assert code == 0


def test_main_exits_one_on_broken_link(tmp_path: Path, capsys) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("See [missing](ghost.md).")
    code = main(["--readme", str(readme), "--quiet"])
    assert code == 1
    out = capsys.readouterr()
    assert "ghost.md" in out.out or "ghost.md" in out.err


def test_main_exits_two_on_missing_readme(tmp_path: Path, capsys) -> None:
    code = main(["--readme", str(tmp_path / "does-not-exist.md")])
    assert code == 2
