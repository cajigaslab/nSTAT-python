#!/usr/bin/env python3
"""Check README freshness: intra-repo links, images, and code-snippet imports.

This is the contributor- and CI-side enforcement of the "Keeping README
current" rule documented in ``AGENT_GUIDE.md`` and ``CLAUDE.md``.

What this script catches
------------------------
1. **Broken intra-repo links** — Markdown ``[text](relative/path)`` that
   resolves to a path no longer in the working tree.  Catches renames,
   deletions, and moves.

2. **Missing images** — ``![alt](path)`` syntax where the referenced
   image file no longer exists.

3. **Stale code-snippet imports** — ``from nstat import X`` or
   ``import nstat.X`` inside fenced ``python`` code blocks where the
   symbol no longer resolves via ``importlib``.  Catches renames,
   removals, and additions-not-yet-exported.

What this script does NOT catch
-------------------------------
- External URLs (``http://`` / ``https://``).  Use ``lychee-action`` in
  CI for that — see ``.github/workflows/readme-check.yml``.
- Markdown anchor links inside the README (e.g., ``[Section](#stale)``).
  Heading slugs are GitHub-specific and tooling-dependent; skipping for
  now.

Exit codes
----------
- 0 — clean.  README freshness verified.
- 1 — at least one broken link / image / import detected.
- 2 — script invocation error.

Usage
-----
::

    python tools/check_readme_links.py                       # default: README.md
    python tools/check_readme_links.py --readme docs/foo.md  # check arbitrary md file
    python tools/check_readme_links.py --quiet               # only print on failure

Used by:

- ``.github/workflows/readme-check.yml`` — repo-wide gate.
- ``.pre-commit-config.yaml`` — local fast feedback.
- ``Makefile`` target ``make readme-check`` — manual invocation.
"""
from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path
from typing import Any

# Make the repo importable when invoked as ``python tools/check_readme_links.py``.
# When run as a script, Python places the script's directory on
# ``sys.path[0]`` (which is ``tools/`` here), not the working directory.
# An editable install in some other Python env doesn't help; the explicit
# repo-root injection is the standard pattern for ``tools/`` scripts
# (also used by ``tools/parity/diff_against_matlab.py``).
_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))


# ----------------------------------------------------------------------
# Regexes
# ----------------------------------------------------------------------

# Match Markdown links and images.  Group 1: leading ``!`` for images
# (empty for plain links).  Group 2: link text / alt text.
# Group 3: target URL/path.  The path is non-greedy and stops at the
# first ``)`` (so simple URLs work; ones with parens-in-URL are an
# edge case we deliberately don't handle).
_LINK_RE = re.compile(r"(!?)\[([^\]]*)\]\(([^)]+)\)")

# Match fenced Python code blocks.  Capture group 1: block body.
# Use re.DOTALL so ``.`` matches newlines inside the block.
_PYTHON_FENCE_RE = re.compile(
    r"```python\s*\n(.*?)\n```",
    re.DOTALL,
)

# Match ``from nstat import X[, Y, Z]`` or ``from nstat.module import X``.
# We only handle the ``nstat`` namespace — the README's code snippets
# don't import anything else that's worth checking.
_IMPORT_FROM_RE = re.compile(
    r"^from\s+(nstat(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import\s+(.+)$",
    re.MULTILINE,
)

# Match ``import nstat`` or ``import nstat.module``.
_IMPORT_RE = re.compile(
    r"^import\s+(nstat(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*(?:as\s+\w+)?\s*$",
    re.MULTILINE,
)


# ----------------------------------------------------------------------
# Checkers
# ----------------------------------------------------------------------

def _is_external(target: str) -> bool:
    """Return True for URLs we should not validate locally."""
    lower = target.lower().strip()
    return (
        lower.startswith("http://")
        or lower.startswith("https://")
        or lower.startswith("mailto:")
        or lower.startswith("ftp://")
        or lower.startswith("#")  # in-page anchor only
    )


def check_intra_repo_links(readme_path: Path, repo_root: Path) -> list[tuple[str, str]]:
    """Return [(link_text, broken_target), ...] for intra-repo link breakage.

    A link is considered broken when its target resolves (relative to
    ``readme_path.parent``) to a path that does not exist in the
    working tree.  Anchor fragments (``foo.md#section``) are stripped
    before the existence check — we verify the *file* exists, not the
    anchor inside it.
    """
    text = readme_path.read_text(encoding="utf-8")
    broken: list[tuple[str, str]] = []
    seen: set[str] = set()  # de-dup repeated identical targets
    for match in _LINK_RE.finditer(text):
        is_image, link_text, target = match.groups()
        if is_image:  # images handled by check_image_existence
            continue
        if _is_external(target):
            continue
        if target in seen:
            continue
        seen.add(target)
        # Strip anchor fragment (``foo.md#section`` -> ``foo.md``).
        clean = target.split("#", 1)[0]
        if not clean:  # pure anchor link, no file portion
            continue
        resolved = (readme_path.parent / clean).resolve()
        if not resolved.exists():
            broken.append((link_text, target))
    return broken


def check_image_existence(readme_path: Path, repo_root: Path) -> list[tuple[str, str]]:
    """Return [(alt_text, broken_image_target), ...] for missing images."""
    text = readme_path.read_text(encoding="utf-8")
    broken: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in _LINK_RE.finditer(text):
        is_image, alt_text, target = match.groups()
        if not is_image:
            continue
        if _is_external(target):
            continue
        if target in seen:
            continue
        seen.add(target)
        clean = target.split("#", 1)[0]
        if not clean:
            continue
        resolved = (readme_path.parent / clean).resolve()
        if not resolved.exists():
            broken.append((alt_text, target))
    return broken


def check_code_snippet_imports(readme_path: Path) -> list[tuple[str, str]]:
    """Return [(snippet_excerpt, broken_import), ...] for stale imports.

    Only scans fenced ``python`` blocks.  Resolves every
    ``from nstat[.X] import Y[, Z]`` and ``import nstat[.X]`` against
    the live package; if any name doesn't resolve, it's reported.
    """
    text = readme_path.read_text(encoding="utf-8")
    broken: list[tuple[str, str]] = []

    for block_match in _PYTHON_FENCE_RE.finditer(text):
        block = block_match.group(1)
        block_excerpt = block.strip().splitlines()[0][:80] if block.strip() else "<empty>"

        # ``from nstat import X, Y``
        for imp_match in _IMPORT_FROM_RE.finditer(block):
            module_path, names_blob = imp_match.groups()
            try:
                module = importlib.import_module(module_path)
            except Exception as exc:
                broken.append((block_excerpt, f"{imp_match.group(0)}  # {type(exc).__name__}: {exc}"))
                continue
            # Names may be comma-separated and may include ``as`` clauses.
            names = [
                n.split(" as ")[0].strip()
                for n in names_blob.split(",")
            ]
            for name in names:
                if not name:
                    continue
                if not _has_attribute(module, name):
                    broken.append((block_excerpt, f"from {module_path} import {name}"))

        # ``import nstat[.X]``
        for imp_match in _IMPORT_RE.finditer(block):
            module_path = imp_match.group(1)
            try:
                importlib.import_module(module_path)
            except Exception as exc:
                broken.append((block_excerpt, f"{imp_match.group(0)}  # {type(exc).__name__}: {exc}"))

    return broken


def _has_attribute(module: Any, name: str) -> bool:
    """Resolve ``module.name`` defensively (avoids hasattr's silent excepts)."""
    try:
        getattr(module, name)
    except AttributeError:
        return False
    return True


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("README.md"),
        help="Path to the markdown file to check (default: README.md).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print on failure (default: print summary always).",
    )
    args = parser.parse_args(argv)

    readme = args.readme.resolve()
    if not readme.exists():
        print(f"error: {readme} not found", file=sys.stderr)
        return 2

    # Repo root is whatever's at the top of the working tree containing
    # the readme.  This is permissive — works for sub-directory READMEs too.
    repo_root = readme.parent
    while not (repo_root / ".git").exists() and repo_root.parent != repo_root:
        repo_root = repo_root.parent

    broken_links = check_intra_repo_links(readme, repo_root)
    broken_images = check_image_existence(readme, repo_root)
    broken_imports = check_code_snippet_imports(readme)

    n_broken = len(broken_links) + len(broken_images) + len(broken_imports)

    if not args.quiet or n_broken > 0:
        print(f"README freshness check — {readme.relative_to(repo_root) if readme.is_relative_to(repo_root) else readme}")
        print(f"  intra-repo links checked: {len(broken_links)} broken")
        print(f"  images checked:           {len(broken_images)} broken")
        print(f"  code-snippet imports:     {len(broken_imports)} broken")

    if broken_links:
        print("\nBROKEN INTRA-REPO LINKS:")
        for text, target in broken_links:
            print(f"  - [{text}]({target})")

    if broken_images:
        print("\nBROKEN IMAGES:")
        for alt, target in broken_images:
            print(f"  - ![{alt}]({target})")

    if broken_imports:
        print("\nBROKEN CODE-SNIPPET IMPORTS:")
        for excerpt, broken in broken_imports:
            print(f"  - in snippet starting '{excerpt}...'")
            print(f"      -> {broken}")

    if n_broken > 0:
        print(f"\nREADME freshness FAILED ({n_broken} issue(s)).", file=sys.stderr)
        return 1

    if not args.quiet:
        print("\nREADME freshness OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
