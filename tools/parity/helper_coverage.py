#!/usr/bin/env python3
"""Report helper coverage across ``notebooks/``.

Scans every notebook for imports/uses of the ``matlab_*`` helpers (and
``apply_matlab_style``) exposed by :mod:`nstat.notebook_figures` and
reports:

- Per-helper: # notebooks using it
- Per-notebook: # helpers used
- Total coverage: notebooks using ≥ 1 helper / total notebooks

CLI
---
    python tools/parity/helper_coverage.py
    python tools/parity/helper_coverage.py --threshold 0.5 --fail-below-threshold
    python tools/parity/helper_coverage.py --json-only

Exit codes
----------
- 0 — coverage meets threshold (or no ``--fail-below-threshold`` flag).
- 2 — coverage below threshold (with the flag set).
- 1 — invocation error (notebooks dir missing, helpers cannot be enumerated).

The JSON output is written to ``.parity-review/helper_coverage.json``.
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
OUTPUT_DIR = REPO_ROOT / ".parity-review"

# Allow direct script invocation from any cwd by ensuring the repo root
# is importable (so ``import nstat.notebook_figures`` resolves to the
# in-tree package even when nstat isn't pip-installed).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _enumerate_helpers() -> list[str]:
    """Discover public ``matlab_*`` (and ``apply_matlab_style``) helpers."""
    module = importlib.import_module("nstat.notebook_figures")
    names: list[str] = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name, None)
        if not callable(obj):
            continue
        if name.startswith("matlab_") or name == "apply_matlab_style":
            names.append(name)
    return sorted(names)


def _notebook_source(path: Path) -> str:
    nb = nbformat.read(path, as_version=4)
    parts: list[str] = []
    for cell in nb.cells:
        if cell.get("cell_type") == "code":
            parts.append(cell.get("source", ""))
    return "\n".join(parts)


def _count_uses(source: str, helper: str) -> int:
    """Count distinct mentions of ``helper`` as an identifier."""
    pattern = re.compile(rf"\b{re.escape(helper)}\b")
    return len(pattern.findall(source))


def scan(notebooks_dir: Path, helpers: list[str]) -> dict:
    notebooks = sorted(p for p in notebooks_dir.glob("*.ipynb") if p.is_file())
    per_helper: dict[str, int] = {h: 0 for h in helpers}
    per_notebook: list[dict] = []

    for nb_path in notebooks:
        try:
            source = _notebook_source(nb_path)
        except Exception as exc:  # pragma: no cover - defensive
            per_notebook.append(
                {
                    "notebook": nb_path.name,
                    "helpers_used": [],
                    "n_helpers_used": 0,
                    "error": str(exc),
                }
            )
            continue
        used: list[str] = []
        for h in helpers:
            if _count_uses(source, h) > 0:
                used.append(h)
                per_helper[h] += 1
        per_notebook.append(
            {
                "notebook": nb_path.name,
                "helpers_used": used,
                "n_helpers_used": len(used),
            }
        )

    n_total = len(notebooks)
    n_using = sum(1 for nb in per_notebook if nb["n_helpers_used"] > 0)
    coverage = (n_using / n_total) if n_total else 0.0

    return {
        "n_total_notebooks": n_total,
        "n_notebooks_using_helpers": n_using,
        "coverage_pct": coverage,
        "helpers": helpers,
        "per_helper": per_helper,
        "per_notebook": per_notebook,
    }


def _print_table(report: dict) -> None:
    helpers = report["helpers"]
    per_helper = report["per_helper"]
    per_notebook = report["per_notebook"]
    n_total = report["n_total_notebooks"]
    n_using = report["n_notebooks_using_helpers"]
    coverage = report["coverage_pct"]

    print("\n=== Per-helper coverage (# notebooks using) ===")
    print(f"{'helper':<32} {'n_notebooks':>12}")
    for h in helpers:
        print(f"{h:<32} {per_helper[h]:>12}")

    print("\n=== Per-notebook helper count ===")
    print(f"{'notebook':<48} {'n_helpers':>10}")
    for nb in per_notebook:
        marker = "*" if nb["n_helpers_used"] > 0 else " "
        print(f"{marker} {nb['notebook']:<46} {nb['n_helpers_used']:>10}")

    print("\n=== Totals ===")
    print(f"notebooks scanned:           {n_total}")
    print(f"notebooks using >=1 helper:  {n_using}")
    print(f"coverage:                    {coverage * 100:.1f}%")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebooks-dir",
        type=Path,
        default=NOTEBOOKS_DIR,
        help=f"Directory containing notebooks (default: {NOTEBOOKS_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "helper_coverage.json",
        help="Path for the JSON report (default: .parity-review/helper_coverage.json).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Coverage threshold (fraction of notebooks using >=1 helper).",
    )
    parser.add_argument(
        "--fail-below-threshold",
        action="store_true",
        help="Exit 2 if coverage is strictly below --threshold.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Skip the console table; just write the JSON.",
    )
    args = parser.parse_args(argv)

    if not args.notebooks_dir.exists():
        print(f"ERROR: notebooks dir not found: {args.notebooks_dir}", file=sys.stderr)
        return 1

    try:
        helpers = _enumerate_helpers()
    except Exception as exc:
        print(f"ERROR: failed to enumerate helpers: {exc}", file=sys.stderr)
        return 1

    if not helpers:
        print(
            "ERROR: no matlab_* helpers discovered in nstat.notebook_figures.",
            file=sys.stderr,
        )
        return 1

    report = scan(args.notebooks_dir, helpers)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.json_only:
        _print_table(report)
    print(f"\nWrote report: {args.output.relative_to(REPO_ROOT)}")

    if args.fail_below_threshold and report["coverage_pct"] < args.threshold:
        print(
            f"FAIL: helper coverage {report['coverage_pct']*100:.1f}% < "
            f"threshold {args.threshold*100:.1f}%",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
