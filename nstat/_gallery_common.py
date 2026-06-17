"""Shared low-level helpers for the paper / extras / notebooks galleries.

Internal — not part of public API.  Each gallery's renderer composes
from these primitives.

The three renderers (paper-examples, notebooks, extras) all share the
same embedded CSS, the same landing-page (`docs/galleries.html`), and
the same YAML-loading idiom.  Centralising them here keeps the look-
and-feel of the three gallery pages in sync and avoids the drift class
where one renderer's CSS gets a tweak that the others silently miss.

The per-gallery HTML composition itself stays in the owning module
(`nstat.paper_gallery`, `nstat.extras_gallery`) — those renderers have
distinct anchor schemes and figure-path semantics and are not merged.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import yaml


def _load_gallery_yaml(path: Path) -> Any:
    """Read a YAML manifest/descriptions file with UTF-8 encoding.

    Thin wrapper around ``yaml.safe_load(path.read_text(...))`` so every
    gallery loader shares the same encoding + parser invocation.  Returns
    the raw parsed object — callers decide whether to coerce ``None`` to
    ``{}`` (some loaders historically do, others don't; preserving each
    caller's exact behaviour keeps byte-identical gallery output).
    """
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_descriptions_indexed_by(
    path: Path, list_key: str, index_key: str
) -> dict[str, dict[str, Any]]:
    """Load a YAML descriptions file and index its list by a row key.

    Used by the extras and notebook renderers to turn

        {list_key: [{index_key: "...", ...}, ...]}

    into ``{row[index_key]: row}`` for O(1) lookup while rendering.
    Empty files / missing top-level key both yield ``{}``.
    """
    payload = _load_gallery_yaml(path) or {}
    return {row[index_key]: row for row in payload.get(list_key, [])}


def extract_figure_code(
    script_path: Path, figure_filename: str
) -> str | None:
    """Extract the code region annotated for ``figure_filename`` from a demo script.

    Returns the dedented code chunk between
    ``# === FIGURE: <figure_filename> ===`` and ``# === END FIGURE ===``,
    or ``None`` if no such marker pair is found.

    Marker convention
    -----------------
    Demo scripts wrap each figure-producing block in a pair of comment
    markers so the gallery renderer can extract and embed the exact code
    that produced each PNG:

    ::

        # === FIGURE: fig01_thomas_scatter.png ===
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(thomas_points[:, 0], thomas_points[:, 1])
        fig.savefig(out_dir / "fig01_thomas_scatter.png", dpi=120)
        # === END FIGURE ===

    The opening marker shape is exact: one space after ``#``, three
    ``=``, space, ``FIGURE:``, space, filename, space, three ``=``.  The
    closing marker is the same shape with ``END FIGURE``.  Lines between
    the markers (excluding the markers themselves) are returned with the
    minimum common leading whitespace stripped via :func:`textwrap.dedent`.

    Parameters
    ----------
    script_path
        Absolute path to the demo script.  If the file does not exist
        ``None`` is returned so script-path mismatches are caught by
        other gates rather than tripping a hard failure here.
    figure_filename
        The figure's filename as it appears in the manifest (which is
        also the filename in the opening marker and the ``fig.savefig``
        call).

    Raises
    ------
    ValueError
        If multiple opening markers exist for the same filename, or if
        an opening marker is found without a matching closing marker.
        The build should fail loudly rather than silently truncating.
    """
    if not script_path.exists():
        return None

    text = script_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    open_marker = f"# === FIGURE: {figure_filename} ==="
    close_marker = "# === END FIGURE ==="

    open_indices = [
        i for i, line in enumerate(lines) if line.strip() == open_marker
    ]
    if not open_indices:
        return None
    if len(open_indices) > 1:
        raise ValueError(
            f"extract_figure_code: multiple opening markers for "
            f"{figure_filename!r} in {script_path}: lines {open_indices}"
        )

    open_idx = open_indices[0]
    close_idx: int | None = None
    for j in range(open_idx + 1, len(lines)):
        if lines[j].strip() == close_marker:
            close_idx = j
            break
    if close_idx is None:
        raise ValueError(
            f"extract_figure_code: opening marker for {figure_filename!r} "
            f"at line {open_idx + 1} of {script_path} has no matching "
            f"'# === END FIGURE ===' closer"
        )

    body = "\n".join(lines[open_idx + 1 : close_idx])
    # Dedent to the minimum common leading whitespace; strip the
    # leading/trailing blank lines that often pad the marker region.
    return textwrap.dedent(body).strip("\n")


def _shared_gallery_css() -> str:
    """The CSS used by every auto-generated standalone gallery HTML page."""
    return (
        ":root{--bg:#0f1419;--card:#1a1f29;--ink:#e6e6e6;--ink-2:#9aa4b1;"
        "--accent:#7ee787;--accent-2:#58a6ff;--border:#30363d;}"
        "*{box-sizing:border-box}"
        "body{margin:0;background:var(--bg);color:var(--ink);"
        "font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;}"
        ".wrap{max-width:1100px;margin:0 auto;padding:2rem 1.25rem 3rem;}"
        "header{border-bottom:1px solid var(--border);padding-bottom:1.25rem;margin-bottom:1.5rem;}"
        "h1{margin:0 0 .35rem;font-size:1.6rem;color:var(--accent);}"
        "h2{margin:2rem 0 .75rem;font-size:1.2rem;color:var(--accent-2);"
        "border-top:1px solid var(--border);padding-top:1.25rem;}"
        "h3{margin:1.25rem 0 .35rem;font-size:1rem;color:var(--ink);}"
        "p{margin:.4rem 0;color:var(--ink);}"
        ".muted{color:var(--ink-2);font-size:.9rem;}"
        ".tag{display:inline-block;background:#0b0e14;color:var(--accent);"
        "padding:.05rem .4rem;border-radius:3px;font-size:.75rem;margin-left:.4rem;}"
        "code,pre{font-family:'SF Mono',Menlo,Consolas,monospace;font-size:.85rem;"
        "background:#0b0e14;color:var(--accent);padding:.1rem .3rem;border-radius:3px;}"
        "pre{padding:.75rem 1rem;overflow-x:auto;border:1px solid var(--border);}"
        "nav.toc{background:var(--card);border:1px solid var(--border);"
        "padding:.75rem 1rem;border-radius:6px;margin-bottom:1.5rem;font-size:.9rem;}"
        "nav.toc a{color:var(--accent-2);text-decoration:none;margin-right:.9rem;}"
        "nav.toc a:hover{text-decoration:underline;}"
        ".example{background:var(--card);border:1px solid var(--border);"
        "border-radius:8px;padding:1.25rem 1.5rem;margin:1.25rem 0;}"
        ".gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));"
        "gap:.85rem;margin-top:.75rem;}"
        ".gallery figure{margin:0;background:#0b0e14;border:1px solid var(--border);"
        "border-radius:4px;padding:.5rem;}"
        ".gallery img{width:100%;height:auto;display:block;border-radius:2px;}"
        ".gallery figcaption{font-size:.78rem;color:var(--ink-2);margin-top:.4rem;"
        "word-break:break-word;}"
        ".gallery .analysis{font-size:.78rem;color:var(--ink);margin-top:.3rem;}"
        ".placeholder{padding:1rem;color:var(--ink-2);font-size:.8rem;"
        "background:#0b0e14;border:1px dashed var(--border);border-radius:4px;text-align:center;}"
        ".gallery .code-detail{margin-top:.45rem;font-size:.78rem;}"
        ".gallery .code-detail summary{cursor:pointer;color:var(--accent-2);"
        "padding:.15rem .4rem;background:#0b0e14;border:1px solid var(--border);"
        "border-radius:3px;display:inline-block;list-style:none;"
        "font-size:.75rem;}"
        ".gallery .code-detail summary::-webkit-details-marker{display:none;}"
        ".gallery .code-detail summary:hover{color:var(--accent);"
        "border-color:var(--accent);}"
        ".gallery .code-detail[open] summary{color:var(--accent);"
        "border-color:var(--accent);}"
        ".gallery .code-detail pre{margin:.4rem 0 0;padding:.6rem .8rem;"
        "font-size:.72rem;line-height:1.45;overflow-x:auto;"
        "background:#0b0e14;border:1px solid var(--border);border-radius:4px;"
        "color:var(--ink);}"
        ".gallery .code-detail code.language-python{background:transparent;"
        "color:var(--ink);padding:0;font-size:inherit;}"
        "a{color:var(--accent-2);}"
        "footer{margin-top:2rem;padding-top:1rem;border-top:1px solid var(--border);"
        "color:var(--ink-2);font-size:.85rem;}"
    )


def render_galleries_index_html() -> str:
    """Tiny landing page that links the per-category galleries.

    Auto-generated alongside the per-category pages; lives at
    ``docs/galleries.html`` and is published via ``html_extra_path``.
    """
    css = _shared_gallery_css()
    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width,initial-scale=1">')
    parts.append("<title>nSTAT — Output Galleries</title>")
    parts.append(f"<style>{css}</style>")
    parts.append("</head><body><div class=\"wrap\">")
    parts.append("<header><h1>Output Galleries</h1>")
    parts.append(
        '<p class="muted">Auto-generated landing page linking the three '
        "review-without-diving-in showcases.  Each gallery is "
        "self-contained (embedded CSS, inline images) and re-renders on "
        "every <code>make regen</code>.</p></header>"
    )
    parts.append('<section class="example">')
    parts.append(
        '<h2><a href="paper_examples_gallery.html">Paper Examples</a></h2>'
        '<p class="muted">Every <code>examples/paper/example0N_*.py</code> '
        "script with its committed figure outputs and a one-line description "
        "of what each figure shows.</p>"
    )
    parts.append("</section>")
    parts.append('<section class="example">')
    parts.append(
        '<h2><a href="notebooks_gallery.html">Notebooks</a></h2>'
        '<p class="muted">Every entry in '
        "<code>tools/notebook_build/notebook_manifest.yml</code> with per-figure "
        "captions and analysis notes drawn from "
        "<code>notebook_descriptions.yml</code>.</p>"
    )
    parts.append("</section>")
    parts.append('<section class="example">')
    parts.append(
        '<h2><a href="extras_gallery.html">nstat.extras Demos</a></h2>'
        '<p class="muted">Every <code>examples/extras/*_demo.py</code> script '
        "registered in <code>examples/extras/manifest.yml</code> with per-figure "
        "captions and analysis notes drawn from "
        "<code>tools/extras_build/extras_descriptions.yml</code>.  "
        "Figure-bearing demos show their <code>--export-figures</code> outputs; "
        "validation/interop demos document the text-based assertions they exercise.</p>"
    )
    parts.append("</section>")
    parts.append("</div></body></html>")
    return "\n".join(parts) + "\n"
