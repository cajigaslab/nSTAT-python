#!/usr/bin/env python3
"""Embed each notebook's committed gallery PNGs into the notebook as cell
outputs, so the figures render in the ``.ipynb`` on GitHub.

Figures are attributed to their producing cells: a "figure cell" is any code
cell that creates tracked figures (``new_figure(`` / ``_prepare_figure(`` /
``capture(``). Figures are assigned to those cells in order; when a cell
produces figures in a loop (one token, many figures) the trailing figure cell
absorbs the remainder.

This is a fast backfill (no notebook execution). Going forward, executing a
notebook embeds figures automatically — ``FigureTracker`` displays each figure
inline as it is saved — so a normal notebook run keeps these outputs in sync.

Usage:
    python tools/notebook_build/embed_figures.py --all
    python tools/notebook_build/embed_figures.py --topics SignalObjExamples,...
"""
from __future__ import annotations

import argparse
import base64
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[2]
NB_DIR = REPO_ROOT / "notebooks"
GALLERY = REPO_ROOT / "docs" / "notebook_galleries"
TOKENS = ("new_figure(", "_prepare_figure(", "capture(")


def _fig_token_count(src: str) -> int:
    """Count figure-creation tokens in a code cell, ignoring helper defs."""
    n = 0
    for line in src.split("\n"):
        if line.strip().startswith("def "):
            continue
        for tok in TOKENS:
            n += line.count(tok)
    return n


def _image_output(png_bytes: bytes) -> nbformat.NotebookNode:
    return nbformat.v4.new_output(
        "display_data",
        data={"image/png": base64.b64encode(png_bytes).decode("ascii")},
        metadata={},
    )


def _strip_image_outputs(cell: nbformat.NotebookNode) -> None:
    cell["outputs"] = [
        o
        for o in cell.get("outputs", [])
        if not any(k.startswith("image/") for k in o.get("data", {}))
    ]


def embed(topic: str) -> tuple[int, int]:
    nb_path = NB_DIR / f"{topic}.ipynb"
    figs = sorted((GALLERY / topic).glob("fig_*.png"))
    if not figs:
        # No FigureTracker gallery — leave the notebook untouched (e.g.
        # 00_getting_started, which embeds its own figures via inline plotting).
        return 0, 0
    nb = nbformat.read(nb_path, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    for c in code_cells:
        _strip_image_outputs(c)  # idempotent

    fig_cells = [c for c in code_cells if _fig_token_count(c.source) > 0]
    if not fig_cells:
        fig_cells = [code_cells[-1]]

    counts = [_fig_token_count(c.source) for c in fig_cells]
    if sum(counts) != len(figs):
        # loop/helper mismatch: keep leading per-token counts, the last figure
        # cell absorbs the remaining (loop-produced) figures.
        head = counts[:-1]
        counts = head + [max(0, len(figs) - sum(head))]

    idx = 0
    for cell, k in zip(fig_cells, counts):
        for _ in range(k):
            if idx >= len(figs):
                break
            cell.setdefault("outputs", []).append(_image_output(figs[idx].read_bytes()))
            idx += 1
    while idx < len(figs):  # safety: any leftover -> last figure cell
        fig_cells[-1].setdefault("outputs", []).append(_image_output(figs[idx].read_bytes()))
        idx += 1

    nbformat.write(nb, nb_path)
    return idx, len(figs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default="")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    topics = (
        sorted(p.stem for p in NB_DIR.glob("*.ipynb"))
        if args.all
        else [t for t in args.topics.split(",") if t]
    )
    for t in topics:
        embedded, total = embed(t)
        print(f"  {t}: embedded {embedded}/{total} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
