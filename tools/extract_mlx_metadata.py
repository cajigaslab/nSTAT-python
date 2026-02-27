from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATLAB_HELPFILES = Path(
    "/Users/iahncajigas/Library/CloudStorage/Dropbox/Research/Matlab/nSTAT_currentRelease_Local/helpfiles"
)
DEFAULT_OUT = PROJECT_ROOT / "examples" / "help_topics" / "matlab_mlx_metadata.json"

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _paragraph_style(p: ET.Element) -> str:
    ppr = p.find(f"{W_NS}pPr")
    if ppr is None:
        return ""
    pstyle = ppr.find(f"{W_NS}pStyle")
    if pstyle is None:
        return ""
    return str(pstyle.attrib.get(f"{W_NS}val", "") or pstyle.attrib.get("w:val", "")).strip().lower()


def _paragraph_text(p: ET.Element) -> str:
    parts = [t.text or "" for t in p.iter(f"{W_NS}t")]
    return " ".join("".join(parts).split())


def _extract_one(mlx_path: Path, repo_root: Path) -> dict[str, object]:
    with zipfile.ZipFile(mlx_path, "r") as zf:
        raw = zf.read("matlab/document.xml")

    root = ET.fromstring(raw)
    body = root.find(f"{W_NS}body")
    if body is None:
        return {
            "file": str(mlx_path),
            "title": mlx_path.stem,
            "intro": "",
            "headings": [],
        }

    title = mlx_path.stem
    intro = ""
    headings: list[str] = []

    for p in body.findall(f"{W_NS}p"):
        style = _paragraph_style(p)
        text = _paragraph_text(p)
        if not text:
            continue

        if style == "title" and title == mlx_path.stem:
            title = text
            continue

        if style == "heading":
            headings.append(text)
            continue

        if style == "text" and not intro:
            intro = text

    rel_path = str(mlx_path.resolve().relative_to(repo_root.resolve()))
    return {
        "file": rel_path,
        "title": title,
        "intro": intro,
        "headings": headings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract title/heading metadata from MATLAB help .mlx files")
    parser.add_argument("--matlab-helpfiles", default=str(DEFAULT_MATLAB_HELPFILES))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    help_dir = Path(args.matlab_helpfiles).expanduser().resolve()
    if not help_dir.exists() or not help_dir.is_dir():
        raise RuntimeError(f"MATLAB helpfiles directory not found: {help_dir}")

    repo_root = help_dir.parents[1]
    rows: dict[str, dict[str, object]] = {}
    for mlx in sorted(help_dir.glob("*.mlx")):
        rows[mlx.stem] = _extract_one(mlx, repo_root=repo_root)

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_helpfiles": str(help_dir),
        "topics": rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"topics": len(rows), "output": str(out.relative_to(PROJECT_ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
