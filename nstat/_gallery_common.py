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
