#!/usr/bin/env python3
"""Normalize notebooks to executable code-only form for deterministic CI runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat
import yaml


FORBIDDEN_PREFIXES = ("%", "!")
MAGIC_RE = re.compile(r"^\s*%")
SHELL_RE = re.compile(r"^\s*!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "notebook_manifest.yml",
        help="Notebook manifest path.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any notebook requires normalization.",
    )
    return parser.parse_args()


def _sanitize_code(source: str) -> str:
    sanitized: list[str] = []
    inserted_agg = False
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%matplotlib"):
            if not inserted_agg:
                sanitized.append('import matplotlib')
                sanitized.append('matplotlib.use("Agg")')
                inserted_agg = True
            continue
        if MAGIC_RE.match(line) or SHELL_RE.match(line):
            continue
        sanitized.append(line.rstrip())
    # Keep explicit trailing newline for stable notebook diffs.
    out = "\n".join(sanitized).strip("\n")
    return (out + "\n") if out else ""


def _normalize_notebook(path: Path) -> bool:
    before = path.read_text(encoding="utf-8")
    nb = nbformat.read(path, as_version=4)

    new_cells = []
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        source = _sanitize_code(str(cell.get("source", "")))
        if not source.strip():
            continue
        cell["source"] = source
        cell["outputs"] = []
        cell["execution_count"] = None
        cell["metadata"] = {}
        new_cells.append(cell)

    nb.cells = new_cells
    keep = {"kernelspec", "language_info", "nstat"}
    nb.metadata = {k: v for k, v in nb.metadata.items() if k in keep}

    after = nbformat.writes(nb)
    if after != before:
        path.write_text(after, encoding="utf-8")
        return True
    return False


def _manifest_notebooks(manifest: Path, repo_root: Path) -> list[Path]:
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    out: list[Path] = []
    for row in payload.get("notebooks", []):
        out.append(repo_root / str(row["file"]))
    return out


def main() -> int:
    args = parse_args()
    changed = 0
    notebooks = _manifest_notebooks(args.manifest, args.repo_root)
    for path in notebooks:
        if not path.exists():
            raise FileNotFoundError(f"Notebook missing from manifest: {path}")
        if _normalize_notebook(path):
            changed += 1
            print(f"Normalized {path}")

    print(f"Notebook normalization complete. changed={changed} total={len(notebooks)}")
    if args.check and changed > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

