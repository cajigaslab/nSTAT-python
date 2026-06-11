#!/usr/bin/env python3
"""Build GitHub-renderable image galleries from notebook FigureTracker outputs.

For each notebook in the manifest, this script:

1. Executes the notebook in place via ``nbclient`` (no inline output capture —
   the notebooks set ``matplotlib.use("Agg")`` and write figures to
   ``output/notebook_images/<topic>/`` via :class:`nstat.notebook_figures.FigureTracker`).
2. Copies the produced PNGs from ``output/notebook_images/<topic>/`` into
   ``docs/notebook_galleries/<topic>/`` so they're tracked under git and render on
   the GitHub repo browser.
3. Writes a per-notebook gallery ``README.md`` (PNG list with anchors) and
   an index ``docs/notebook_galleries/README.md`` table cross-referencing source
   notebooks and figure galleries.

Run via ``make regen-notebook-galleries`` or directly:

    python tools/notebook_build/build_notebook_galleries.py
    python tools/notebook_build/build_notebook_galleries.py --group smoke
    python tools/notebook_build/build_notebook_galleries.py --skip-execute   # use existing output/
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import nbformat
import yaml
from nbclient import NotebookClient


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_ROOT = REPO_ROOT / "output" / "notebook_images"
GALLERY_ROOT = REPO_ROOT / "docs" / "notebook_galleries"
MANIFEST = REPO_ROOT / "tools" / "notebook_build" / "notebook_manifest.yml"
GROUPS = REPO_ROOT / "tools" / "notebook_build" / "topic_groups.yml"


@dataclass(frozen=True)
class Target:
    topic: str
    path: Path
    run_group: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--groups-file", type=Path, default=GROUPS)
    p.add_argument(
        "--group",
        default="all",
        help="Notebook group to process: smoke, core, full, all, ci_smoke, etc.",
    )
    p.add_argument("--topics", default="", help="Comma-separated subset of topics.")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument(
        "--skip-execute",
        action="store_true",
        help="Skip re-execution; use whatever PNGs are already in output/notebook_images/.",
    )
    return p.parse_args()


def load_targets(manifest_path: Path) -> list[Target]:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    out: list[Target] = []
    for row in payload["notebooks"]:
        out.append(
            Target(
                topic=row["topic"],
                path=(REPO_ROOT / row["file"]).resolve(),
                run_group=row.get("run_group", "full"),
            )
        )
    return out


def load_groups(groups_path: Path) -> dict[str, list[str]]:
    payload = yaml.safe_load(groups_path.read_text(encoding="utf-8"))
    return {k: list(v) for k, v in payload.get("groups", {}).items()}


def select(targets: list[Target], group: str, topics: str, groups: dict[str, list[str]]) -> list[Target]:
    if group in groups:
        wanted = set(groups[group])
        sel = [t for t in targets if t.topic in wanted]
    elif group in {"all"}:
        sel = list(targets)
    elif group in {"full"}:
        sel = list(targets)
    elif group == "smoke":
        sel = [t for t in targets if t.run_group == "smoke"]
    else:
        sel = [t for t in targets if t.run_group == group]
    if topics.strip():
        wanted = {token.strip() for token in topics.split(",") if token.strip()}
        sel = [t for t in sel if t.topic in wanted]
    return sel


def execute_notebook(path: Path, timeout: int) -> None:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()


def has_real_content(png_path: Path) -> bool:
    """A real plot has drawn ink; a FigureTracker placeholder is a near-blank
    canvas with only a title + a monospace annotation line.

    Detect via the fraction of non-white pixels rather than file size: sparse
    but legitimate plots (event-marker rasters, coarse step signal reps)
    compress to small files yet clearly contain data.  Empty placeholders sit
    around 0.004 non-white; the sparsest real plots are >0.01, so 0.008 cleanly
    separates them.
    """
    try:
        from PIL import Image
        import numpy as np

        arr = np.asarray(Image.open(png_path).convert("RGB"))
        non_white = 1.0 - float(np.all(arr >= 245, axis=-1).mean())
        return non_white > 0.008
    except Exception:
        # If Pillow/numpy unavailable, fall back to the old size heuristic.
        try:
            return png_path.stat().st_size > 25_000
        except FileNotFoundError:
            return False


def build_per_notebook_readme(topic: str, gallery_dir: Path, source_rel: str, figures: list[str], real_count: int) -> None:
    lines = [
        f"# {topic} — figure gallery",
        "",
        f"This page is the rendered figure output of",
        f"[`notebooks/{Path(source_rel).name}`](../../notebooks/{Path(source_rel).name}).",
        "Each PNG is an output of the notebook's ``FigureTracker``; placeholder",
        "MATLAB-line annotations look like blank pages with code snippets and indicate the",
        "notebook is a MATLAB-helpfile port rather than a narrative example.",
        "",
        f"- Source notebook: [`notebooks/{Path(source_rel).name}`](../../notebooks/{Path(source_rel).name})",
        f"- Figures: {len(figures)} ({real_count} with substantive plot content)",
        "",
        "## Figures",
        "",
    ]
    for fig in figures:
        anchor = fig.replace(".png", "")
        lines.append(f"### {fig}")
        lines.append("")
        lines.append(f"![{anchor}](./{fig})")
        lines.append("")
    (gallery_dir / "README.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_index_readme(rows: list[dict]) -> None:
    lines = [
        "# Notebook galleries",
        "",
        "Quick visual index of the figures each notebook produces, with links back",
        "to the source `.ipynb`. The PNGs here are deterministic outputs of each",
        "notebook's `FigureTracker` (see `nstat/notebook_figures.py`). They are",
        "regenerated by `make regen-notebook-galleries` and drift-checked in CI.",
        "",
        "Some galleries below are placeholder MATLAB-line annotations (a",
        "MATLAB-helpfile port) rather than substantive plots — those are marked.",
        "",
        "| Notebook | Figures | Substantive | Source |",
        "|---|---|---|---|",
    ]
    for row in rows:
        marker = "yes" if row["real_count"] > 0 else "placeholder only"
        lines.append(
            f"| [`{row['topic']}`]({row['topic']}/) | {row['count']} | {marker} | "
            f"[`notebooks/{row['source_basename']}`](../../notebooks/{row['source_basename']}) |"
        )
    lines.append("")
    lines.append(
        "Regenerate this gallery with `make regen-notebook-galleries` "
        "(executes the smoke group; pass `--group full` for everything)."
    )
    (GALLERY_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    all_targets = load_targets(args.manifest)
    groups = load_groups(args.groups_file)
    targets = select(all_targets, args.group, args.topics, groups)
    if not targets:
        print(f"ERROR: no notebooks selected for --group={args.group!r}", file=sys.stderr)
        return 1
    print(f"Processing {len(targets)} notebook(s) (group={args.group})")

    GALLERY_ROOT.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict] = []
    failures: list[str] = []

    def _row_from_existing(tgt) -> dict | None:
        """Reconstruct an index row from a notebook's already-committed gallery,
        so a subset run (e.g. --skip-execute over only the executed notebooks)
        preserves every notebook's index entry instead of dropping it."""
        existing = sorted((GALLERY_ROOT / tgt.topic).glob("fig_*.png"))
        if not existing:
            return None
        return {
            "topic": tgt.topic,
            "count": len(existing),
            "real_count": sum(1 for f in existing if has_real_content(f)),
            "source_basename": tgt.path.name,
        }

    for tgt in targets:
        if not tgt.path.exists():
            failures.append(f"missing notebook source: {tgt.path}")
            continue

        if not args.skip_execute:
            print(f"  executing [{tgt.run_group}] {tgt.topic}")
            try:
                execute_notebook(tgt.path, timeout=args.timeout)
            except Exception as exc:
                failures.append(f"{tgt.topic}: execute failed — {type(exc).__name__}: {exc}")
                continue

        produced = OUTPUT_ROOT / tgt.topic
        manifest_path = produced / "manifest.json"
        if not manifest_path.exists():
            # Not executed this run (or no FigureTracker): keep its committed
            # gallery and index entry rather than dropping it.
            print(f"  (keep) {tgt.topic}: not executed this run; preserving committed gallery")
            row = _row_from_existing(tgt)
            if row:
                index_rows.append(row)
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        figures = list(manifest.get("images", []))
        if not figures:
            print(f"  (keep) {tgt.topic}: empty figure list; preserving committed gallery")
            row = _row_from_existing(tgt)
            if row:
                index_rows.append(row)
            continue

        gallery_dir = GALLERY_ROOT / tgt.topic
        gallery_dir.mkdir(parents=True, exist_ok=True)
        # Clean stale PNGs that no longer exist in the manifest.
        for stale in gallery_dir.glob("fig_*.png"):
            if stale.name not in figures:
                stale.unlink()
        real_count = 0
        for fig in figures:
            src = produced / fig
            dst = gallery_dir / fig
            shutil.copy2(src, dst)
            if has_real_content(dst):
                real_count += 1

        source_rel = tgt.path.relative_to(REPO_ROOT).as_posix()
        build_per_notebook_readme(tgt.topic, gallery_dir, source_rel, figures, real_count)
        index_rows.append(
            {
                "topic": tgt.topic,
                "count": len(figures),
                "real_count": real_count,
                "source_basename": tgt.path.name,
            }
        )
        print(f"  wrote docs/notebook_galleries/{tgt.topic}/ ({len(figures)} PNG{'s' if len(figures)!=1 else ''}, real={real_count})")

    index_rows.sort(key=lambda r: r["topic"])
    if index_rows:
        build_index_readme(index_rows)
        print(f"wrote docs/notebook_galleries/README.md ({len(index_rows)} entries)")

    if failures:
        print("", file=sys.stderr)
        print("=== Failures ===", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
