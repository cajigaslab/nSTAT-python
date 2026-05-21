#!/usr/bin/env python3
"""Diff the local MATLAB nSTAT checkout against the Python parity manifest.

Usage
-----
    python tools/parity/diff_against_matlab.py
    python tools/parity/diff_against_matlab.py --matlab-path /path/to/nstat
    python tools/parity/diff_against_matlab.py --json > diff.json

What it does
------------
1. Walks the root-level ``*.m`` files of the MATLAB nSTAT checkout (default:
   ``$NSTAT_MATLAB_PATH`` or ``/Users/iahncajigas/projects/nstat`` as a
   developer-machine fallback).
2. For each ``.m`` file, extracts:
   - the ``classdef <Name>`` declaration (if present), and
   - every ``function`` declaration (these become methods or free functions).
3. Cross-references against the committed ``parity/manifest.yml``
   ``public_api`` list.
4. Reports three categories:

     - NEW IN MATLAB        : classes / methods present in the MATLAB
                              source but not in the Python parity manifest.
     - REMOVED FROM MATLAB  : symbols Python expects but the MATLAB source
                              no longer provides.
     - MISMATCHED PYTHON PATH: symbols where the Python path documented in
                              the manifest no longer points at an importable
                              location.

What it does NOT do
-------------------
- Does not parse method bodies — it only inventories signatures.
- Does not compute behavioural parity (that's what the gold fixtures in
  ``tests/parity/fixtures/matlab_gold/`` and the ``parity/class_fidelity.yml``
  audit are for).
- Does not modify any files; output goes to stdout (or a JSON file when
  ``--json`` is given).

Exit codes
----------
- 0: no drift (the MATLAB checkout's public surface matches the parity
     manifest).
- 1: drift detected (one or more findings in NEW / REMOVED / MISMATCHED).
- 2: invocation error (MATLAB checkout not found, etc.).

Independence guarantee
----------------------
This tool reads the MATLAB checkout as a READ-ONLY filesystem traversal.
It does not import, execute, or modify anything in the MATLAB repo.  The
sanctioned MATLAB-runtime bridge (``nstat.matlab_engine``) is the only
other coupling between the two repos.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Make the repo importable when run as a standalone script.  When invoked
# as ``python tools/parity/diff_against_matlab.py`` from the repo root,
# Python places only the script directory on ``sys.path``, not the working
# directory — so ``import nstat`` fails despite an editable install in a
# sibling Python environment.  Adding the repo root explicitly is the
# standard pattern for tools/ scripts.
_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------

DEFAULT_MATLAB_PATHS = [
    Path("/Users/iahncajigas/projects/nstat"),
    Path.home() / "projects" / "nstat",
    Path("../nstat"),
    Path("../nSTAT"),
]


def resolve_matlab_path(explicit: Path | None) -> Path:
    """Return the canonical MATLAB checkout path.

    Order of resolution:
      1. Explicit ``--matlab-path`` argument.
      2. ``$NSTAT_MATLAB_PATH`` environment variable.
      3. Each path in ``DEFAULT_MATLAB_PATHS``.
    """
    if explicit is not None:
        return explicit.resolve()
    env = os.environ.get("NSTAT_MATLAB_PATH")
    if env:
        return Path(env).resolve()
    for candidate in DEFAULT_MATLAB_PATHS:
        if candidate.exists() and any(candidate.glob("*.m")):
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate the MATLAB nSTAT checkout.  Set NSTAT_MATLAB_PATH "
        "or pass --matlab-path.  Searched: "
        + ", ".join(str(p) for p in DEFAULT_MATLAB_PATHS)
    )


# ----------------------------------------------------------------------
# MATLAB signature extraction
# ----------------------------------------------------------------------

_CLASSDEF_RE = re.compile(r"^\s*classdef\s+([A-Za-z_]\w*)", re.MULTILINE)
_FUNCTION_RE = re.compile(
    r"^\s*function\s+(?:[^=]+=\s*)?([A-Za-z_]\w*)\s*\(",
    re.MULTILINE,
)


@dataclass
class MatlabFile:
    """Inventory of one MATLAB source file."""

    path: Path
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def is_class_file(self) -> bool:
        return any(c.lower() == self.stem.lower() for c in self.classes)


def inventory_matlab_file(path: Path) -> MatlabFile:
    """Extract classdef + function declarations from a MATLAB ``.m`` file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    # Strip block comments (``%{ ... %}``) and line comments to avoid
    # picking up signatures inside docstring blocks.
    text = re.sub(r"%\{.*?%\}", "", text, flags=re.DOTALL)
    text = re.sub(r"%[^\n]*", "", text)

    classes = _CLASSDEF_RE.findall(text)
    functions = _FUNCTION_RE.findall(text)
    return MatlabFile(path=path, classes=classes, functions=functions)


def inventory_matlab_checkout(matlab_root: Path) -> dict[str, MatlabFile]:
    """Inventory every root-level ``.m`` file in the MATLAB checkout.

    Returns a dict mapping filename stem → :class:`MatlabFile`.
    Sub-directories (``+nstat/``, ``helpfiles/``, ``tools/``, ``tests/``,
    ``examples/``) are intentionally not walked — only the canonical
    root-level toolbox files participate in parity tracking.
    """
    out: dict[str, MatlabFile] = {}
    for m_path in sorted(matlab_root.glob("*.m")):
        inv = inventory_matlab_file(m_path)
        out[inv.stem] = inv
    return out


# ----------------------------------------------------------------------
# Python parity manifest
# ----------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "parity" / "manifest.yml"


def load_parity_public_api() -> list[dict[str, Any]]:
    """Return the ``public_api`` list from ``parity/manifest.yml``."""
    data = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    return list(data.get("public_api", []))


def python_symbol_resolves(target: str) -> bool:
    """Return ``True`` if ``target`` (e.g. ``nstat.Analysis``) imports cleanly."""
    if not target or target == "null" or target == "None":
        return False
    parts = target.split(".")
    try:
        module = importlib.import_module(parts[0])
    except Exception:
        return False
    obj: Any = module
    for attr in parts[1:]:
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            return False
    return True


# ----------------------------------------------------------------------
# Diff
# ----------------------------------------------------------------------

@dataclass
class DiffEntry:
    category: str  # "new_in_matlab", "removed_from_matlab", "mismatched_python_path"
    matlab_class: str | None
    matlab_path: str | None
    python_target: str | None
    python_path: str | None
    note: str


def diff(matlab_inv: dict[str, MatlabFile], parity_api: list[dict[str, Any]]) -> list[DiffEntry]:
    """Compute drift between the MATLAB inventory and the parity manifest."""
    out: list[DiffEntry] = []

    # Index of parity entries by MATLAB class name (case-sensitive).
    parity_by_class: dict[str, dict[str, Any]] = {
        str(row.get("matlab", "")): row for row in parity_api
    }

    # 1. NEW IN MATLAB — every .m file whose stem isn't a manifest entry.
    for stem, inv in matlab_inv.items():
        if stem in parity_by_class:
            continue
        # Two .m files are conventionally non-class entry points and may not
        # be tracked individually in the public_api list:
        if stem in {"Contents", "run_tests", "nstatOpenHelpPage"}:
            continue
        out.append(
            DiffEntry(
                category="new_in_matlab",
                matlab_class=stem,
                matlab_path=str(inv.path.name),
                python_target=None,
                python_path=None,
                note=(
                    f"{stem}.m is a {'class' if inv.is_class_file else 'function'} "
                    f"file with {len(inv.functions)} function declaration(s) but "
                    f"has no entry in parity/manifest.yml::public_api."
                ),
            )
        )

    # 2. REMOVED FROM MATLAB — parity entries whose .m file is gone.
    for cls_name, row in parity_by_class.items():
        if cls_name in matlab_inv:
            continue
        if row.get("status") == "not_applicable":
            continue
        out.append(
            DiffEntry(
                category="removed_from_matlab",
                matlab_class=cls_name,
                matlab_path=str(row.get("matlab_path", "")),
                python_target=row.get("python_target"),
                python_path=row.get("python_path"),
                note=(
                    f"parity manifest expects {cls_name}.m but no matching "
                    f"file was found in the MATLAB checkout."
                ),
            )
        )

    # 3. MISMATCHED PYTHON PATH — parity entries whose Python target no
    #    longer resolves at import time.
    for row in parity_api:
        py_target = row.get("python_target")
        if not py_target or row.get("status") == "not_applicable":
            continue
        if not python_symbol_resolves(str(py_target)):
            out.append(
                DiffEntry(
                    category="mismatched_python_path",
                    matlab_class=str(row.get("matlab")),
                    matlab_path=str(row.get("matlab_path", "")),
                    python_target=str(py_target),
                    python_path=str(row.get("python_path", "")),
                    note=(
                        f"Python symbol {py_target!r} does not resolve via "
                        f"importlib (manifest entry is stale)."
                    ),
                )
            )

    return out


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def render_markdown(matlab_root: Path, matlab_inv: dict[str, MatlabFile],
                    entries: list[DiffEntry]) -> str:
    """Render the diff as a human-readable Markdown report."""
    from datetime import date

    lines: list[str] = []
    lines.append(f"# MATLAB ↔ Python parity diff — {date.today().isoformat()}")
    lines.append("")
    lines.append(f"- MATLAB checkout: `{matlab_root}`")
    lines.append(f"- Inventoried `.m` files: **{len(matlab_inv)}**")
    lines.append(f"- Diff entries: **{len(entries)}**")
    lines.append("")

    by_category: dict[str, list[DiffEntry]] = {}
    for e in entries:
        by_category.setdefault(e.category, []).append(e)

    section_titles = {
        "new_in_matlab": "## NEW IN MATLAB (not in `parity/manifest.yml`)",
        "removed_from_matlab": "## REMOVED FROM MATLAB",
        "mismatched_python_path": "## MISMATCHED PYTHON PATH",
    }
    for key, title in section_titles.items():
        rows = by_category.get(key, [])
        lines.append(title)
        lines.append("")
        if not rows:
            lines.append("_(no findings)_")
            lines.append("")
            continue
        for e in rows:
            lines.append(f"- **{e.matlab_class or '?'}** (`{e.matlab_path or '?'}`)")
            if e.python_target:
                lines.append(f"  - Python target: `{e.python_target}`")
            if e.python_path:
                lines.append(f"  - Python path: `{e.python_path}`")
            lines.append(f"  - {e.note}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("**Note**: this report only inspects the public-API surface listed "
                 "in `parity/manifest.yml`.  Behavioural parity is tracked separately "
                 "via gold fixtures in `tests/parity/fixtures/matlab_gold/` and the "
                 "class-fidelity audit in `parity/class_fidelity.yml`.")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matlab-path", type=Path, default=None,
        help="Path to the MATLAB nSTAT checkout (overrides NSTAT_MATLAB_PATH).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of Markdown.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write report to FILE instead of stdout.",
    )
    args = parser.parse_args(argv)

    try:
        matlab_root = resolve_matlab_path(args.matlab_path)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    matlab_inv = inventory_matlab_checkout(matlab_root)
    parity_api = load_parity_public_api()
    entries = diff(matlab_inv, parity_api)

    if args.json:
        payload = {
            "matlab_root": str(matlab_root),
            "matlab_file_count": len(matlab_inv),
            "entries": [
                {
                    "category": e.category,
                    "matlab_class": e.matlab_class,
                    "matlab_path": e.matlab_path,
                    "python_target": e.python_target,
                    "python_path": e.python_path,
                    "note": e.note,
                }
                for e in entries
            ],
        }
        out_text = json.dumps(payload, indent=2)
    else:
        out_text = render_markdown(matlab_root, matlab_inv, entries)

    if args.output is not None:
        args.output.write_text(out_text, encoding="utf-8")
    else:
        sys.stdout.write(out_text)

    return 0 if not entries else 1


if __name__ == "__main__":
    sys.exit(main())
