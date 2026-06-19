#!/usr/bin/env python3
"""Render structured YAML ledgers into the human-readable ``parity/*.md`` files.

Since v6 iter 28, ``parity/matlab_defects.yml`` and
``parity/matlab_pedagogical_gaps.yml`` are the source of truth.  The matching
``.md`` files are regenerated from the YAML by this tool.  Manual edits to
the ``.md`` files will be overwritten — edit the ``.yml`` instead.

Usage
-----
    python tools/parity/render_ledger.py             # render both ledgers
    python tools/parity/render_ledger.py --check     # verify .md matches .yml
    python tools/parity/render_ledger.py --ledger defects

Exit codes
----------
- 0 — render succeeded (or, with ``--check``, the ``.md`` matched).
- 2 — ``--check`` found a mismatch.
- 1 — error (missing YAML, malformed schema).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PARITY_DIR = REPO_ROOT / "parity"

LEDGERS = {
    "defects": {
        "yml": PARITY_DIR / "matlab_defects.yml",
        "md": PARITY_DIR / "matlab_defects.md",
        "header": "# MATLAB defects and Python improvements ledger",
        "section_header": "## Open entries",
        "entry_prefix": "Defect",
    },
    "pedagogical": {
        "yml": PARITY_DIR / "matlab_pedagogical_gaps.yml",
        "md": PARITY_DIR / "matlab_pedagogical_gaps.md",
        "header": "# MATLAB pedagogical gaps — opportunities to enrich the MATLAB toolbox",
        "section_header": "## Open entries",
        "entry_prefix": "Gap",
    },
}


def _join_lines(parts: Iterable[str]) -> str:
    return "\n".join(parts).rstrip() + "\n"


def _format_bullet(label: str, value: str) -> list[str]:
    """Render a `- **Label:** value` bullet with hanging indent on continuations."""
    value = (value or "").rstrip("\n")
    lines = value.split("\n")
    out = [f"- **{label}:** {lines[0]}"]
    for tail in lines[1:]:
        if tail.strip() == "":
            out.append("")
        else:
            out.append(f"  {tail}")
    return out


def _format_list_bullet(label: str, items: list[str]) -> list[str]:
    if not items:
        return [f"- **{label}:** —"]
    if len(items) == 1:
        return [f"- **{label}:** {items[0]}"]
    out = [f"- **{label}:**"]
    for item in items:
        out.append(f"  - {item}")
    return out


# Fields surfaced when an entry has been resolved (typically by upstream
# MATLAB adoption). Rendered as a bullet block after ``Discovered:``.
_RESOLUTION_FIELDS: tuple[tuple[str, str], ...] = (
    ("upstream_status", "Upstream status"),
    ("resolved_in", "Resolved in"),
    ("resolved_iter", "Resolved iter"),
    ("resolved_notes", "Resolved notes"),
)


def _format_resolution_bullets(entry: dict) -> list[str]:
    """Append upstream/resolution bullets (only those present in the YAML)."""
    out: list[str] = []
    for key, label in _RESOLUTION_FIELDS:
        if key in entry and entry[key] not in (None, ""):
            out.extend(_format_bullet(label, str(entry[key])))
    return out


def render_defects(data: dict) -> str:
    parts: list[str] = []
    parts.append("# MATLAB defects and Python improvements ledger")
    parts.append("")
    intro = (data.get("intro") or "").rstrip()
    if intro:
        parts.append(intro)
        parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("## Open entries")
    parts.append("")

    for entry in data.get("entries", []):
        parts.append(f"### {entry.get('defect_class', 'Defect')}: {entry['title']}")
        parts.append("")
        parts.extend(_format_bullet("MATLAB location", entry.get("matlab_location", "")))
        parts.extend(_format_bullet("Defect class", entry.get("defect_class", "")))
        parts.extend(_format_bullet("MATLAB behavior", entry.get("matlab_behavior", "")))
        parts.extend(_format_bullet("Correct behavior", entry.get("correct_behavior", "")))
        impl = entry.get("python_implementation") or []
        if isinstance(impl, str):
            impl = [impl]
        parts.extend(_format_list_bullet("Python implementation", impl))
        parts.extend(_format_bullet("Fixture impact", entry.get("fixture_impact", "")))
        parts.extend(_format_bullet("Discovered", entry.get("discovered", "")))
        parts.extend(_format_resolution_bullets(entry))
        if entry.get("upstream_issue"):
            parts.extend(_format_bullet("Upstream issue", entry["upstream_issue"]))
        parts.append("")
        parts.append("---")
        parts.append("")

    checklist = (data.get("checklist") or "").rstrip()
    if checklist:
        parts.append(checklist)
        parts.append("")

    return _join_lines(parts)


def render_pedagogical(data: dict) -> str:
    parts: list[str] = []
    parts.append("# MATLAB pedagogical gaps — opportunities to enrich the MATLAB toolbox")
    parts.append("")
    intro = (data.get("intro") or "").rstrip()
    if intro:
        parts.append(intro)
        parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("## Open entries")
    parts.append("")

    for entry in data.get("entries", []):
        parts.append(f"### Gap: {entry['title']}")
        parts.append("")
        parts.extend(_format_bullet("Topic / helpfile", f"`{entry['topic']}`"))
        parts.extend(
            _format_bullet("MATLAB count", str(entry.get("matlab_count", "")))
        )
        delta = entry.get("delta_label", "")
        py_count = entry.get("python_count", "")
        parts.extend(
            _format_bullet("Python count", f"{py_count} ({delta})" if delta else str(py_count))
        )
        parts.extend(_format_bullet("What Python adds", entry.get("python_adds", "")))
        parts.extend(
            _format_bullet(
                "Pedagogical justification", entry.get("pedagogical_justification", "")
            )
        )
        parts.extend(
            _format_bullet("MATLAB upstream action", entry.get("matlab_upstream_action", ""))
        )
        parts.extend(
            _format_bullet("Python implementation", entry.get("python_implementation", ""))
        )
        parts.extend(_format_bullet("Discovered", entry.get("discovered", "")))
        parts.extend(_format_resolution_bullets(entry))
        parts.append("")

    deficits = data.get("deficits", [])
    if deficits:
        parts.append("---")
        parts.append("")
        parts.append("## Deficit topics (Python < MATLAB)")
        parts.append("")
        parts.append(
            "These need closure: either close the count gap or document why "
            "MATLAB's extra figures are artifacts (e.g. live-script auto-redraw "
            "duplicates)."
        )
        parts.append("")
        for d in deficits:
            parts.append(f"### Deficit: {d['title']}")
            parts.append("")
            parts.extend(_format_bullet("Topic", f"`{d['topic']}`"))
            parts.extend(_format_bullet("MATLAB count", str(d.get("matlab_count", ""))))
            py = d.get("python_count", "")
            delta = d.get("delta_label", "")
            parts.extend(
                _format_bullet(
                    "Python count", f"{py} (Δ = {delta})" if delta else str(py)
                )
            )
            parts.extend(_format_bullet("MATLAB extras", d.get("matlab_extras", "")))
            parts.extend(
                _format_bullet("Pedagogical impact", d.get("pedagogical_impact", ""))
            )
            parts.extend(_format_bullet("Decision", d.get("decision", "")))
            parts.extend(
                _format_bullet("MATLAB upstream action", d.get("matlab_upstream_action", ""))
            )
            parts.extend(_format_bullet("Discovered", d.get("discovered", "")))
            parts.extend(_format_resolution_bullets(d))
            parts.append("")

    trailer = (data.get("trailer") or "").rstrip()
    if trailer:
        parts.append("---")
        parts.append("")
        parts.append(trailer)
        parts.append("")

    return _join_lines(parts)


RENDERERS = {
    "defects": render_defects,
    "pedagogical": render_pedagogical,
}


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"ERROR: YAML ledger not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_ledger(name: str, *, check: bool = False) -> int:
    cfg = LEDGERS[name]
    data = _load_yaml(cfg["yml"])
    rendered = RENDERERS[name](data)
    md_path: Path = cfg["md"]
    if check:
        existing = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
        if existing != rendered:
            print(
                f"[{name}] MISMATCH: {md_path} differs from rendered YAML.  "
                f"Run `python tools/parity/render_ledger.py` to update.",
                file=sys.stderr,
            )
            return 2
        print(f"[{name}] OK: {md_path} matches {cfg['yml'].name}")
        return 0
    md_path.write_text(rendered, encoding="utf-8")
    n_entries = len(data.get("entries", []))
    n_deficits = len(data.get("deficits", []))
    extra = f", {n_deficits} deficit(s)" if n_deficits else ""
    print(
        f"[{name}] wrote {md_path.relative_to(REPO_ROOT)} "
        f"({n_entries} entries{extra})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        choices=sorted(LEDGERS.keys()) + ["all"],
        default="all",
        help="Which ledger to render (default: all).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the .md matches the .yml without writing.  Exit 2 on mismatch.",
    )
    args = parser.parse_args(argv)

    names = list(LEDGERS) if args.ledger == "all" else [args.ledger]
    rc = 0
    for n in names:
        rc = max(rc, render_ledger(n, check=args.check))
    return rc


if __name__ == "__main__":
    sys.exit(main())
