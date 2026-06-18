"""Extract structured content from MATLAB Live Editor (.mlx) files.

A .mlx file is a zip archive containing ``matlab/document.xml``, which uses
WordprocessingML (the same OOXML schema as Word). Paragraphs (``<w:p>``) carry
a style via ``<w:pStyle w:val="...">`` — observed styles in the nSTAT helpfiles
are ``title``, ``heading``, ``text``, ``code``, and ``ListParagraph``.
``matlab/output.xml`` (execution results) is skipped on purpose.

We split the document into sections at each ``heading``-styled paragraph (the
title paragraph, when present, becomes a leading section). Each section
collects:

- ``title`` — the heading (or title) text
- ``text_md`` — a markdown rendering of the prose / list paragraphs
- ``code`` — the concatenated MATLAB code blocks

The output schema matches what the parity-review pipeline expects::

    {
        "sections": [{"title": str, "text_md": str, "code": str}, ...],
        "metadata": {"sha": str, "source": str},
    }

CLI usage::

    python tools/parity/extract_mlx.py <topic> <path/to/file.mlx>
    python tools/parity/extract_mlx.py all <matlab-repo-root>

Writes ``.parity-review/mlx_extract/<topic>.json``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def _qn(tag: str) -> str:
    """Qualify a tag with the WordprocessingML namespace."""
    return f"{{{W_NS}}}{tag}"


def _para_style(p: ET.Element) -> str:
    """Return the ``w:pStyle`` value of a ``w:p`` element (or empty string)."""
    pPr = p.find("w:pPr", NS)
    if pPr is None:
        return ""
    style = pPr.find("w:pStyle", NS)
    if style is None:
        return ""
    return style.get(_qn("val"), "")


def _para_numbered(p: ET.Element) -> bool:
    """True if the paragraph is part of a numbered/bulleted list."""
    pPr = p.find("w:pPr", NS)
    if pPr is None:
        return False
    return pPr.find("w:numPr", NS) is not None


def _run_text(r: ET.Element) -> str:
    """Concatenate every ``w:t`` (including CDATA) inside a run."""
    parts: list[str] = []
    for t in r.iter(_qn("t")):
        if t.text:
            parts.append(t.text)
    return "".join(parts)


def _run_is_bold(r: ET.Element) -> bool:
    rPr = r.find("w:rPr", NS)
    return rPr is not None and rPr.find("w:b", NS) is not None


def _run_is_italic(r: ET.Element) -> bool:
    rPr = r.find("w:rPr", NS)
    return rPr is not None and rPr.find("w:i", NS) is not None


def _para_text_markdown(p: ET.Element) -> str:
    """Render a non-code paragraph to inline markdown, honoring bold/italic."""
    out: list[str] = []
    # Iterate direct children so hyperlinks (which wrap runs) are flattened.
    for child in p.iter():
        if child.tag != _qn("r"):
            continue
        txt = _run_text(child)
        if not txt:
            continue
        if _run_is_bold(child):
            txt = f"**{txt}**"
        elif _run_is_italic(child):
            txt = f"*{txt}*"
        out.append(txt)
    return "".join(out).strip()


def _para_code_text(p: ET.Element) -> str:
    """Render a ``code``-styled paragraph as raw MATLAB source (preserves CDATA)."""
    parts: list[str] = []
    for r in p.findall("w:r", NS):
        parts.append(_run_text(r))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

def _new_section(title: str = "") -> dict:
    return {"title": title, "_text_lines": [], "_code_lines": []}


def _finalize(section: dict) -> dict:
    text_md = "\n\n".join(line for line in section["_text_lines"] if line).strip()
    code = "\n".join(section["_code_lines"]).strip("\n")
    return {"title": section["title"], "text_md": text_md, "code": code}


def _iter_paragraphs(body: ET.Element) -> Iterable[ET.Element]:
    return body.findall("w:p", NS)


def _split_sections(body: ET.Element) -> list[dict]:
    """Walk the body, splitting into sections at each heading paragraph."""
    sections: list[dict] = []
    current = _new_section()

    for p in _iter_paragraphs(body):
        style = _para_style(p)

        if style == "title":
            # Title becomes the first section's title; flush any preceding empty.
            if current["title"] or current["_text_lines"] or current["_code_lines"]:
                sections.append(current)
            current = _new_section(_para_text_markdown(p) or "(untitled)")
            continue

        if style == "heading":
            sections.append(current)
            current = _new_section(_para_text_markdown(p) or "(untitled heading)")
            continue

        if style == "code":
            code = _para_code_text(p)
            # Preserve the original line breaks; the CDATA block usually carries them.
            current["_code_lines"].append(code.rstrip("\n"))
            continue

        # text / ListParagraph / anything else → prose
        text = _para_text_markdown(p)
        if not text:
            continue
        if style == "ListParagraph" or _para_numbered(p):
            text = f"- {text}"
        current["_text_lines"].append(text)

    sections.append(current)
    # Drop sections that are completely empty (no title, text, or code).
    return [
        _finalize(s)
        for s in sections
        if s["title"] or s["_text_lines"] or s["_code_lines"]
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_mlx(mlx_path: Path) -> dict:
    """Parse a ``.mlx`` and return a structured section breakdown.

    Returns ``{"sections": [...], "metadata": {"sha": str, "source": str}}``.
    Falls back to a single section if the document has no heading markers.
    """
    mlx_path = Path(mlx_path)
    raw = mlx_path.read_bytes()
    sha = hashlib.sha1(raw).hexdigest()

    with zipfile.ZipFile(mlx_path) as zf:
        with zf.open("matlab/document.xml") as fh:
            tree = ET.parse(fh)

    root = tree.getroot()
    body = root.find("w:body", NS)
    if body is None:
        body = root  # extremely defensive

    sections = _split_sections(body)
    if not sections:
        sections = [{"title": mlx_path.stem, "text_md": "", "code": ""}]

    return {
        "sections": sections,
        "metadata": {"sha": sha, "source": str(mlx_path)},
    }


def _discover_topics(matlab_repo_root: Path) -> dict[str, Path]:
    """Return {topic_name: mlx_path} for every ``.mlx`` under ``helpfiles/``."""
    helpfiles = matlab_repo_root / "helpfiles"
    if not helpfiles.exists():
        # Allow callers to point directly at a directory of .mlx files.
        helpfiles = matlab_repo_root
    topics: dict[str, Path] = {}
    for mlx in sorted(helpfiles.glob("*.mlx")):
        topics[mlx.stem] = mlx
    return topics


def extract_all(repo_root: Path, matlab_repo_root: Path) -> dict[str, dict]:
    """Extract every ``.mlx`` topic under ``matlab_repo_root/helpfiles``.

    ``repo_root`` is accepted for API symmetry with the rest of the parity
    tooling; it is not consulted directly here.
    """
    del repo_root  # currently unused; reserved for future cross-references.
    topics = _discover_topics(Path(matlab_repo_root))
    return {name: extract_mlx(path) for name, path in topics.items()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _output_dir(repo_root: Path) -> Path:
    out = repo_root / ".parity-review" / "mlx_extract"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_topic(repo_root: Path, topic: str, data: dict) -> Path:
    out_path = _output_dir(repo_root) / f"{topic}.json"
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return out_path


def _print_summary(topic: str, data: dict) -> None:
    sections = data["sections"]
    print(f"[{topic}] {len(sections)} sections, sha={data['metadata']['sha'][:10]}")
    for i, s in enumerate(sections):
        n_text = len(s["text_md"])
        n_code = len(s["code"])
        title = s["title"] or "(untitled)"
        print(f"  {i:>2}. {title!s:<50}  text={n_text:>5}  code={n_code:>5}")


def main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parents[2]

    if len(argv) < 2:
        print(__doc__)
        return 2

    target = argv[1]

    if target == "all":
        if len(argv) < 3:
            print("usage: extract_mlx.py all <matlab-repo-root>", file=sys.stderr)
            return 2
        matlab_repo_root = Path(argv[2])
        results = extract_all(repo_root, matlab_repo_root)
        for topic, data in results.items():
            path = _write_topic(repo_root, topic, data)
            print(f"wrote {path}")
        return 0

    # Single-topic mode: <topic> <path/to/.mlx>
    if len(argv) < 3:
        print("usage: extract_mlx.py <topic> <path/to/file.mlx>", file=sys.stderr)
        return 2

    topic = target
    mlx_path = Path(argv[2])
    data = extract_mlx(mlx_path)
    out = _write_topic(repo_root, topic, data)
    _print_summary(topic, data)
    print(f"wrote {out}")
    # Also dump JSON to stdout for piping / quick eyeballing.
    print(json.dumps(data, indent=2, ensure_ascii=False)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
