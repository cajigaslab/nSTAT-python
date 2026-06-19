"""Build MATLAB↔Python side-by-side composite PNGs (and HTML viewers).

For each notebook topic (e.g. ``mEPSCAnalysis``, ``HippocampalPlaceCellExample``)
this script pairs the MATLAB helpfile figures (``<topic>_NN.png``) with the
matching Python gallery figures (``docs/notebook_galleries/<topic>/fig_NNN.png``)
and writes a single dark-background composite PNG with MATLAB on the left and
Python on the right.

Outputs (under ``.parity-review/``):
    * ``composite_<topic>.png``       — one stacked side-by-side PNG per topic
    * ``topic_<topic>.html``          — per-topic HTML viewer of the same pairs
    * ``parity_index.html``           — clickable index across all topics

Usage::

    python tools/parity/build_composites.py                  # --all (default)
    python tools/parity/build_composites.py --topic mEPSCAnalysis
    python tools/parity/build_composites.py --all

Dependencies: stdlib + Pillow only (no project imports), so this can run in a
bare venv against a checkout of both repos.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

# --- Visual style ----------------------------------------------------------
BG_COLOR = (40, 40, 40)          # #282828
TEXT_COLOR = (220, 220, 220)
SUB_COLOR = (170, 170, 170)
MISSING_BG = (120, 30, 30)       # #781e1e
MISSING_TEXT = (255, 230, 230)
DIVIDER = (90, 90, 90)

CELL_W = 700          # per-side image cell width (px)
CELL_PAD = 8          # padding inside each cell
HEADER_H = 56         # column-header band
ROW_LABEL_H = 36      # per-row caption band
ROW_GAP = 6
COL_GAP = 4

# Default search locations.  Both paths are project-relative — see _autodetect.
DEFAULT_MATLAB_REPO_CANDIDATES = (
    Path.home() / "projects" / "nstat" / "helpfiles",
    Path.home() / "projects" / "nSTAT" / "helpfiles",
)


# --- Topic discovery --------------------------------------------------------
_TOPIC_RE = re.compile(r"^(?P<topic>[A-Za-z0-9]+)_(?P<idx>\d{2,3})\.png$")
_PY_FIG_RE = re.compile(r"^fig_(?P<idx>\d{3})\.png$")


def list_matlab_topics(matlab_helpfiles: Path) -> dict[str, list[Path]]:
    """Return ``{topic: [sorted paths of <topic>_NN.png]}`` from a helpfiles dir."""
    out: dict[str, list[Path]] = {}
    if not matlab_helpfiles.is_dir():
        return out
    for p in matlab_helpfiles.iterdir():
        if not p.is_file():
            continue
        m = _TOPIC_RE.match(p.name)
        if not m:
            continue
        out.setdefault(m.group("topic"), []).append(p)
    for topic in out:
        out[topic].sort(key=lambda q: int(_TOPIC_RE.match(q.name).group("idx")))
    return out


def index_matlab_figs(matlab_helpfiles: Path, topic: str) -> dict[int, Path]:
    """Return ``{NN: path}`` for ``<topic>_NN.png`` figures.

    Strict-index pairing helper: builds a numeric-index → path map so the
    composite renderer can align figure N on the MATLAB side with figure N on
    the Python side, regardless of skips, gaps, or extras on either side.
    """
    paths = list_matlab_topics(matlab_helpfiles).get(topic, [])
    out: dict[int, Path] = {}
    for p in paths:
        m = _TOPIC_RE.match(p.name)
        if not m:
            continue
        out[int(m.group("idx"))] = p
    return out


def list_py_figs(py_galleries: Path, topic: str) -> list[Path]:
    """Return sorted ``fig_NNN.png`` figures for ``py_galleries/<topic>/``."""
    gallery = py_galleries / topic
    if not gallery.is_dir():
        return []
    figs: list[Path] = []
    for p in gallery.iterdir():
        if not p.is_file():
            continue
        if _PY_FIG_RE.match(p.name):
            figs.append(p)
    figs.sort(key=lambda q: int(_PY_FIG_RE.match(q.name).group("idx")))
    return figs


def index_py_figs(py_galleries: Path, topic: str) -> dict[int, Path]:
    """Return ``{NNN: path}`` for ``fig_NNN.png`` figures under ``<topic>/``.

    Strict-index pairing helper: see :func:`index_matlab_figs`.
    """
    out: dict[int, Path] = {}
    for p in list_py_figs(py_galleries, topic):
        m = _PY_FIG_RE.match(p.name)
        if not m:
            continue
        out[int(m.group("idx"))] = p
    return out


# --- Font loading -----------------------------------------------------------
def _load_font(size: int) -> ImageFont.ImageFont:
    """Best-effort font lookup; falls back to PIL default."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


# --- Cell rendering --------------------------------------------------------
def _fit_image(src: Path, max_w: int, max_h: int) -> Image.Image:
    """Open ``src`` and downscale (preserving aspect) to fit ``max_w x max_h``.

    Returns an RGB image; flatten transparency onto white so MATLAB PNGs with
    alpha channels render with their original look.
    """
    im = Image.open(src)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[-1])
        im = bg
    else:
        im = im.convert("RGB")

    w, h = im.size
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        im = im.resize(new_size, Image.LANCZOS)
    return im


def _row_height(ml_path: Path | None, py_path: Path | None, max_h: int) -> int:
    """Compute the row image-band height: tallest of the two scaled images."""
    heights: list[int] = []
    for p in (ml_path, py_path):
        if p is None:
            continue
        try:
            tmp = _fit_image(p, CELL_W - 2 * CELL_PAD, max_h)
            heights.append(tmp.size[1])
        except Exception:
            continue
    if not heights:
        return 120
    return max(heights) + 2 * CELL_PAD


def _draw_cell(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    img_path: Path | None,
    caption: str,
    font_small,
) -> None:
    """Render one side of a row: image or red 'missing' placeholder + caption."""
    cell_box = (x, y, x + w, y + h)
    if img_path is None:
        draw.rectangle(cell_box, fill=MISSING_BG)
        msg = "(missing)"
        tw, th = _text_size(draw, msg, font_small)
        draw.text(
            (x + (w - tw) // 2, y + (h - th) // 2),
            msg,
            fill=MISSING_TEXT,
            font=font_small,
        )
        return

    draw.rectangle(cell_box, fill=BG_COLOR)
    try:
        im = _fit_image(img_path, w - 2 * CELL_PAD, h - 2 * CELL_PAD)
    except Exception as exc:  # corrupt PNG / IO error — degrade gracefully
        draw.rectangle(cell_box, fill=MISSING_BG)
        draw.text(
            (x + CELL_PAD, y + CELL_PAD),
            f"(error: {exc.__class__.__name__})",
            fill=MISSING_TEXT,
            font=font_small,
        )
        return

    iw, ih = im.size
    ix = x + (w - iw) // 2
    iy = y + (h - ih) // 2
    canvas.paste(im, (ix, iy))

    cap = caption
    tw, _th = _text_size(draw, cap, font_small)
    draw.text((x + CELL_PAD, y + h - 18), cap, fill=SUB_COLOR, font=font_small)
    if tw > w:  # nothing to do but log silently
        pass


# --- Composite -------------------------------------------------------------
def build_composite(
    topic: str,
    matlab_helpfiles: Path,
    py_galleries: Path,
    out_path: Path,
    max_rows: int = 12,
) -> dict:
    """Write a single side-by-side composite PNG for ``topic``.

    Parameters
    ----------
    topic:
        Topic name, e.g. ``"mEPSCAnalysis"``.  Matches
        ``<matlab_helpfiles>/<topic>_NN.png`` and
        ``<py_galleries>/<topic>/fig_NNN.png``.
    matlab_helpfiles:
        Directory holding the MATLAB ``<topic>_NN.png`` figures.
    py_galleries:
        Directory holding ``<topic>/fig_NNN.png`` Python gallery subfolders.
    out_path:
        Destination PNG.  Parent directory is created if needed.
    max_rows:
        Hard cap on rendered rows (for very long topics).

    Returns
    -------
    dict
        ``{"ml_count": int, "py_count": int, "rows_shown": int}``.
    """
    ml_by_idx = index_matlab_figs(matlab_helpfiles, topic)
    py_by_idx = index_py_figs(py_galleries, topic)

    # Strict index pairing: a MATLAB figure with index N is paired with the
    # Python figure whose numeric suffix is also N. Either side can be missing
    # for any given index, in which case the cell renders as "(missing)".
    all_indices = sorted(set(ml_by_idx.keys()) | set(py_by_idx.keys()))
    n_rows = min(len(all_indices), max_rows)
    indices = all_indices[:n_rows]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if n_rows == 0:
        # Emit a small placeholder so the CLI never silently fails.
        img = Image.new("RGB", (CELL_W * 2 + COL_GAP, HEADER_H + 80), BG_COLOR)
        d = ImageDraw.Draw(img)
        f = _load_font(18)
        d.text((10, 10), f"{topic}: no figures found", fill=TEXT_COLOR, font=f)
        img.save(out_path, format="PNG")
        return {"ml_count": 0, "py_count": 0, "rows_shown": 0}

    font_h = _load_font(22)
    font_row = _load_font(14)
    font_small = _load_font(12)

    # Per-row heights from a sensible per-row max.  Cap any one row at 1.5 *
    # CELL_W so wide MATLAB figures don't blow up the composite.
    per_row_max_h = int(CELL_W * 1.15)
    row_heights: list[int] = []
    for idx in indices:
        ml = ml_by_idx.get(idx)
        py = py_by_idx.get(idx)
        row_heights.append(_row_height(ml, py, per_row_max_h))

    total_w = CELL_W * 2 + COL_GAP
    total_h = HEADER_H + sum(row_heights) + ROW_GAP * (n_rows - 1)
    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    # Header band -----------------------------------------------------------
    draw.rectangle((0, 0, total_w, HEADER_H), fill=(30, 30, 30))
    title = (
        f"{topic}    MATLAB ({len(ml_by_idx)})  vs.  Python ({len(py_by_idx)})"
    )
    draw.text((12, 14), title, fill=TEXT_COLOR, font=font_h)
    draw.line((0, HEADER_H - 1, total_w, HEADER_H - 1), fill=DIVIDER, width=1)
    draw.line((CELL_W, HEADER_H, CELL_W, total_h), fill=DIVIDER, width=1)

    # Rows ------------------------------------------------------------------
    y = HEADER_H
    for row_i, idx in enumerate(indices):
        rh = row_heights[row_i]
        ml = ml_by_idx.get(idx)
        py = py_by_idx.get(idx)
        ml_cap = ml.name if ml else "(missing)"
        py_cap = py.name if py else "(missing)"

        _draw_cell(canvas, draw, 0, y, CELL_W, rh, ml, ml_cap, font_small)
        _draw_cell(canvas, draw, CELL_W + COL_GAP, y, CELL_W, rh, py, py_cap, font_small)

        # Row index badge along the divider — show the shared numeric index.
        idx_txt = f"{idx:02d}"
        tw, th = _text_size(draw, idx_txt, font_row)
        bx = CELL_W - tw // 2
        by = y + 6
        draw.rectangle((bx - 4, by - 2, bx + tw + 4, by + th + 2), fill=(20, 20, 20))
        draw.text((bx, by), idx_txt, fill=TEXT_COLOR, font=font_row)

        y += rh + ROW_GAP

    canvas.save(out_path, format="PNG", optimize=True)
    return {
        "ml_count": len(ml_by_idx),
        "py_count": len(py_by_idx),
        "rows_shown": n_rows,
    }


# --- HTML emitters ---------------------------------------------------------
def _delta_span(ml: int, py: int) -> str:
    d = py - ml
    if d == 0:
        return '<span style="color:#0f0">OK</span>'
    if d > 0:
        return f'<span style="color:#ff8">+{d}</span>'
    return f'<span style="color:#f55">{d}</span>'


def write_topic_html(
    topic: str,
    matlab_helpfiles: Path,
    py_galleries: Path,
    out_path: Path,
) -> dict:
    """Write a per-topic stacked side-by-side HTML viewer.

    Uses ``file://`` URIs so the page works without a web server.
    """
    ml_by_idx = index_matlab_figs(matlab_helpfiles, topic)
    py_by_idx = index_py_figs(py_galleries, topic)
    indices = sorted(set(ml_by_idx.keys()) | set(py_by_idx.keys()))
    n = len(indices)

    rows: list[str] = []
    for idx in indices:
        ml = ml_by_idx.get(idx)
        py = py_by_idx.get(idx)
        ml_idx = f"{idx:02d}" if ml is not None else "—"
        py_idx = f"{idx:03d}" if py is not None else "—"

        if ml is not None:
            ml_cell = (
                f'<td><div class="cap">{html.escape(ml.name)}</div>'
                f'<img src="file://{ml.resolve()}"></td>'
            )
        else:
            ml_cell = '<td class="missing">(no MATLAB figure)</td>'

        if py is not None:
            py_cell = (
                f'<td><div class="cap">{html.escape(py.name)}</div>'
                f'<img src="file://{py.resolve()}"></td>'
            )
        else:
            py_cell = '<td class="missing">(no Python figure)</td>'

        rows.append(
            f'<tr><th>{idx}<br><small>MATLAB {ml_idx} / Python {py_idx}'
            f"</small></th>{ml_cell}{py_cell}</tr>"
        )

    ml_paths = list(ml_by_idx.values())
    py_paths = list(py_by_idx.values())

    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{html.escape(topic)} parity</title>
<style>
body{{font:14px system-ui;margin:1em;background:#222;color:#eee}}
h1{{margin:0 0 8px 0}}
table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #555;padding:6px;vertical-align:top;text-align:center}}
th{{background:#333;width:60px}}
td.missing{{background:#781e1e;color:#fff;font-weight:bold}}
img{{max-width:48vw;max-height:70vh;background:#fff}}
.cap{{font-size:11px;color:#aaa;margin-bottom:4px}}
.nav a{{color:#9cf;margin-right:1em}}
</style></head><body>
<div class="nav"><a href="parity_index.html">← Index</a></div>
<h1>{html.escape(topic)}</h1>
<table>
  <tr><th></th><th>MATLAB ({len(ml_paths)})</th><th>Python ({len(py_paths)})</th></tr>
  {''.join(rows)}
</table></body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return {"ml_count": len(ml_paths), "py_count": len(py_paths), "rows_shown": n}


def write_index_html(out_path: Path, topic_stats: list[dict]) -> None:
    rows: list[str] = []
    for s in sorted(topic_stats, key=lambda d: d["topic"].lower()):
        rows.append(
            f"<tr><td><a href=\"topic_{html.escape(s['topic'])}.html\">"
            f"{html.escape(s['topic'])}</a></td>"
            f"<td>{s['ml_count']}</td><td>{s['py_count']}</td>"
            f"<td>{_delta_span(s['ml_count'], s['py_count'])}</td>"
            f"<td><a href=\"composite_{html.escape(s['topic'])}.png\">PNG</a></td></tr>"
        )

    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>nSTAT MATLAB↔Python parity index</title>
<style>
body{{font:14px system-ui;margin:1em;background:#222;color:#eee}}
table{{border-collapse:collapse}}
td,th{{border:1px solid #555;padding:6px 12px;text-align:left}}
th{{background:#333}}
a{{color:#9cf}}
</style></head><body>
<h1>MATLAB↔Python figure parity</h1>
<p>Side-by-side gallery comparison. Click a topic to view stacked figures,
or open the PNG for the rendered composite.</p>
<table>
<tr><th>Topic</th><th>MATLAB</th><th>Python</th><th>Δ (Py - ML)</th><th>Composite</th></tr>
{''.join(rows)}
</table></body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")


# --- Top-level orchestration ----------------------------------------------
def _autodetect_matlab_helpfiles(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for c in DEFAULT_MATLAB_REPO_CANDIDATES:
        if c.is_dir():
            return c
    # Fall back to the first candidate — caller will see an empty topic set.
    return DEFAULT_MATLAB_REPO_CANDIDATES[0]


def build_all(matlab_repo: Path, repo_root: Path, out_dir: Path) -> list[dict]:
    """Build composites + HTML for every topic with at least one MATLAB png.

    Parameters
    ----------
    matlab_repo:
        Directory holding ``<topic>_NN.png`` MATLAB helpfile figures.
    repo_root:
        Repo root; ``docs/notebook_galleries/`` is read from here.
    out_dir:
        Destination directory (typically ``.parity-review/``).

    Returns
    -------
    list of dict
        Per-topic ``{"topic", "ml_count", "py_count", "rows_shown",
        "composite", "html"}``.
    """
    py_galleries = repo_root / "docs" / "notebook_galleries"
    topics = sorted(list_matlab_topics(matlab_repo).keys(), key=str.lower)

    results: list[dict] = []
    for topic in topics:
        comp = out_dir / f"composite_{topic}.png"
        html_path = out_dir / f"topic_{topic}.html"
        stats = build_composite(topic, matlab_repo, py_galleries, comp)
        write_topic_html(topic, matlab_repo, py_galleries, html_path)
        results.append(
            {
                "topic": topic,
                "ml_count": stats["ml_count"],
                "py_count": stats["py_count"],
                "rows_shown": stats["rows_shown"],
                "composite": str(comp),
                "html": str(html_path),
            }
        )

    write_index_html(out_dir / "parity_index.html", results)
    return results


# --- CLI -------------------------------------------------------------------
def _repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--topic", help="Build a single topic (e.g. mEPSCAnalysis).")
    grp.add_argument("--all", action="store_true", help="Build every topic (default).")
    parser.add_argument(
        "--matlab-helpfiles",
        type=Path,
        default=None,
        help="MATLAB helpfiles directory (default: autodetect).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repo root (default: derived from this script's path).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <repo-root>/.parity-review).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=12,
        help="Max rows in a single composite (default 12).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = (args.repo_root or _repo_root_from_this_file()).resolve()
    out_dir = (args.out_dir or repo_root / ".parity-review").resolve()
    matlab_helpfiles = _autodetect_matlab_helpfiles(args.matlab_helpfiles).resolve()
    py_galleries = (repo_root / "docs" / "notebook_galleries").resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.topic and not args.all:
        comp = out_dir / f"composite_{args.topic}.png"
        html_path = out_dir / f"topic_{args.topic}.html"
        stats = build_composite(
            args.topic, matlab_helpfiles, py_galleries, comp, max_rows=args.max_rows
        )
        write_topic_html(args.topic, matlab_helpfiles, py_galleries, html_path)
        print(
            f"[{args.topic}] ml={stats['ml_count']} py={stats['py_count']} "
            f"rows={stats['rows_shown']} -> {comp.name}"
        )
        # Refresh the index too so the single-topic build stays clickable.
        existing: list[dict] = []
        for png in sorted(out_dir.glob("composite_*.png")):
            t = png.stem[len("composite_") :]
            ml_paths = list_matlab_topics(matlab_helpfiles).get(t, [])
            py_paths = list_py_figs(py_galleries, t)
            existing.append(
                {
                    "topic": t,
                    "ml_count": len(ml_paths),
                    "py_count": len(py_paths),
                    "rows_shown": max(len(ml_paths), len(py_paths)),
                }
            )
        write_index_html(out_dir / "parity_index.html", existing)
        return 0

    # Default path: build everything.
    results = build_all(matlab_helpfiles, repo_root, out_dir)
    for r in results:
        delta = r["py_count"] - r["ml_count"]
        sign = "+" if delta > 0 else ""
        print(
            f"[{r['topic']}] ml={r['ml_count']} py={r['py_count']} "
            f"({sign}{delta})  -> composite_{r['topic']}.png"
        )
    print(f"\nIndex: {out_dir / 'parity_index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
