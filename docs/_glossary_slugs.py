"""MyST heading-slug override for ``concepts/glossary.md`` (see ``conf.py``).

Each glossary term is a ``###`` heading preceded by an ``<a id="x"></a>``
anchor.  The anchor lines are the single source of truth: they give browsers
the short fragment ids, and this module reads the same lines so MyST resolves
links such as ``glossary.md#local-field-potential``.  The slug returned here
only drives MyST link resolution; the HTML id Sphinx emits for a heading is
the docutils id, unaffected by it.
"""
from __future__ import annotations

import re
from pathlib import Path

from myst_parser.mdit_to_docutils.base import default_slugify

_GLOSSARY = Path(__file__).resolve().parent / "concepts" / "glossary.md"


def _key(title: str) -> str:
    # MyST passes only the text/code children of a heading to the slug
    # function, so inline math (``$...$``) is absent there; drop it from the
    # glossary titles too so both sides compare equal.
    return re.sub(r"\$[^$]*\$", "", title).strip()


_SLUGS: dict[str, str] = {
    _key(title): slug
    for slug, title in re.findall(
        r'<a id="([a-z0-9-]+)"></a>\n### (.+)', _GLOSSARY.read_text(encoding="utf-8")
    )
}


def heading_slug(title: str) -> str:
    return _SLUGS.get(_key(title)) or default_slugify(title)
