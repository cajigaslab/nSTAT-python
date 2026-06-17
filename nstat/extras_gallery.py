"""Self-contained ``docs/extras_gallery.html`` showcase generator.

Mirrors the notebook-gallery two-file pattern shipped in
``nstat.paper_gallery``:

- ``examples/extras/manifest.yml`` is the canonical list of demos (one
  per ``examples/extras/*_demo.py`` script) with the title, question,
  short description, and the figure filenames each demo emits when run
  with ``--export-figures``.
- ``tools/extras_build/extras_descriptions.yml`` holds the per-demo
  overview, goal, and per-figure caption + analysis prose.  The set of
  ``demo_id`` keys in this file MUST equal the set in the manifest —
  enforced by ``tests/test_extras_docs.py``.

The rendered ``docs/extras_gallery.html`` is committed to the repo and
served via ``html_extra_path`` in ``docs/conf.py``.  Every ``make regen``
re-runs ``tools/extras_build/build_extras_gallery.py`` which calls
:func:`write_extras_gallery_outputs` here.

The renderer also regenerates ``docs/galleries.html`` so the landing
page links the new gallery alongside paper-examples and notebooks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from nstat._gallery_common import (
    _load_descriptions_indexed_by,
    _load_gallery_yaml,
    _shared_gallery_css,
    render_galleries_index_html,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_extras_manifest(repo_root: Path | None = None) -> list[dict[str, Any]]:
    """Load ``examples/extras/manifest.yml`` and return the ``extras`` list."""
    base = _repo_root() if repo_root is None else repo_root.resolve()
    path = base / "examples" / "extras" / "manifest.yml"
    payload = _load_gallery_yaml(path) or {}
    return list(payload.get("extras", []))


def load_extras_descriptions(repo_root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load ``tools/extras_build/extras_descriptions.yml``, keyed by ``demo_id``."""
    base = _repo_root() if repo_root is None else repo_root.resolve()
    path = base / "tools" / "extras_build" / "extras_descriptions.yml"
    return _load_descriptions_indexed_by(
        path, list_key="extras", index_key="demo_id"
    )


def render_extras_html(repo_root: Path | None = None) -> str:
    """Self-contained HTML showcase of every ``examples/extras/*_demo.py`` script.

    Reads the manifest + descriptions, then renders one ``<section>``
    per demo.  Figure-bearing demos get a ``.gallery`` grid of
    ``<figure>`` cards; figure-free demos get a single ``.placeholder``
    block explaining that the demo emits console output only.  Image
    ``src`` paths are relative to ``docs/`` so the page works both
    locally and on the deployed GitHub Pages site.
    """
    base = _repo_root() if repo_root is None else repo_root.resolve()
    extras = load_extras_manifest(base)
    descriptions = load_extras_descriptions(base)

    css = _shared_gallery_css()

    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width,initial-scale=1">')
    parts.append("<title>nstat.extras — runnable demo gallery</title>")
    parts.append(f"<style>{css}</style>")
    parts.append("</head><body><div class=\"wrap\">")
    parts.append("<header>")
    parts.append("<h1>nstat.extras — runnable demo gallery</h1>")
    parts.append(
        "<p class=\"muted\">Every <code>examples/extras/*_demo.py</code> script "
        "registered in <code>examples/extras/manifest.yml</code>, with the figure "
        "outputs each demo produces (when applicable) and a short description of "
        "what every figure shows.  Auto-generated from "
        "<code>tools/extras_build/extras_descriptions.yml</code> — re-render with "
        "<code>python tools/extras_build/build_extras_gallery.py</code> after "
        "editing the descriptions.</p>"
    )
    parts.append("</header>")

    parts.append('<nav class="toc"><strong>Demos:</strong> ')
    nav_links = " · ".join(
        f'<a href="#{row["demo_id"]}">{row["demo_id"]}</a>'
        for row in extras
    )
    parts.append(nav_links + "</nav>")

    for row in extras:
        demo_id = row["demo_id"]
        script = row["script"]
        title = row.get("title", demo_id)
        question = row.get("question", "").strip()
        description = row.get("description", "").strip()
        manifest_figures = row.get("figures") or []
        desc = descriptions.get(demo_id, {})
        overview = desc.get("overview", "").strip()
        goal = desc.get("goal", "").strip()
        figures = desc.get("figures") or []

        has_figures = bool(manifest_figures)
        run_cmd = (
            f"python {script} --export-figures"
            if has_figures
            else f"python {script}"
        )

        parts.append(f'<section class="example" id="{demo_id}">')
        parts.append(f"<h2>{title}</h2>")
        if question:
            parts.append(f"<p><strong>Question:</strong> {question}</p>")
        if goal:
            parts.append(f"<p><strong>Goal:</strong> {goal}</p>")
        if overview:
            parts.append(f'<p class="muted">{overview}</p>')
        elif description:
            parts.append(f'<p class="muted">{description}</p>')
        parts.append(
            f'<p><strong>Run:</strong> <code>{run_cmd}</code> · '
            f'<a href="https://github.com/cajigaslab/nSTAT-python/blob/main/'
            f'{script}">Script</a></p>'
        )

        if not has_figures:
            parts.append(
                '<div class="placeholder">'
                "(no figure output — this is a text-based validation/interop "
                "demo; run the script for console output)"
                "</div>"
            )
            parts.append("</section>")
            continue

        # Index description figures by filename so we can attach
        # caption + analysis to each manifest entry.
        desc_by_filename = {
            fig.get("filename", ""): fig for fig in figures
        }
        parts.append('<div class="gallery">')
        for fig in manifest_figures:
            filename = fig.get("filename", "")
            desc_fig = desc_by_filename.get(filename, {})
            caption = desc_fig.get("caption", "").strip()
            analysis = desc_fig.get("analysis", "").strip()
            rel = f"figures/extras/{demo_id}/{filename}"
            png_path = base / "docs" / "figures" / "extras" / demo_id / filename
            if png_path.exists():
                parts.append(
                    f'<figure><img src="{rel}" alt="{filename}" loading="lazy">'
                    f'<figcaption><strong>{filename}</strong>'
                    + (f"<br>{caption}" if caption else "")
                    + "</figcaption>"
                    + (f'<div class="analysis">{analysis}</div>' if analysis else "")
                    + "</figure>"
                )
            else:
                parts.append(
                    f'<figure><div class="placeholder">'
                    f"<code>{filename}</code><br>(run "
                    f"<code>python {script} --export-figures</code> "
                    f"to generate this figure)</div>"
                    f'<figcaption><strong>{filename}</strong>'
                    + (f"<br>{caption}" if caption else "")
                    + "</figcaption>"
                    + (f'<div class="analysis">{analysis}</div>' if analysis else "")
                    + "</figure>"
                )
        parts.append("</div>")
        parts.append("</section>")

    parts.append("<footer>")
    parts.append(
        '<p>Source: <a href="https://github.com/cajigaslab/nSTAT-python">'
        "cajigaslab/nSTAT-python</a> · License: GPL-2.0 · "
        'Paper: <a href="https://pubmed.ncbi.nlm.nih.gov/22981419/">'
        "Cajigas, Malik &amp; Brown 2012, J Neurosci Methods</a></p>"
    )
    parts.append("</footer>")
    parts.append("</div></body></html>")
    return "\n".join(parts) + "\n"


def write_extras_gallery_outputs(
    repo_root: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Write ``docs/extras_gallery.html`` and regenerate ``docs/galleries.html``.

    Returns ``(manifest_path, html_path, galleries_index_path)``.
    """
    base = _repo_root() if repo_root is None else repo_root.resolve()
    manifest_path = base / "examples" / "extras" / "manifest.yml"
    docs_dir = base / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    html_path = docs_dir / "extras_gallery.html"
    html_path.write_text(render_extras_html(base), encoding="utf-8")

    # Always regenerate the landing page so its three-card layout stays
    # in sync regardless of which build script ran most recently.
    galleries_index_path = docs_dir / "galleries.html"
    galleries_index_path.write_text(render_galleries_index_html(), encoding="utf-8")

    return manifest_path, html_path, galleries_index_path


__all__ = [
    "load_extras_descriptions",
    "load_extras_manifest",
    "render_extras_html",
    "write_extras_gallery_outputs",
]
