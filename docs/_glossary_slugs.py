"""MyST heading-slug override for ``concepts/glossary.md`` (see ``conf.py``).

Each glossary term is a ``###`` heading preceded by a ``<!-- slug: x -->``
marker; the heading's anchor becomes ``x`` instead of the long auto-slug, so
links such as ``glossary.md#local-field-potential`` resolve.
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
        r"<!-- slug: ([a-z0-9-]+) -->\n### (.+)", _GLOSSARY.read_text(encoding="utf-8")
    )
}


def heading_slug(title: str) -> str:
    return _SLUGS.get(_key(title)) or default_slugify(title)
