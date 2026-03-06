#!/usr/bin/env python3
"""Sanitize notebooks so they present as Python-native examples."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat
import yaml


MATLAB_LINE_RE = re.compile(r"^\s*# MATLAB L\d+:.*$")
MATLAB_CALL_RE = re.compile(r"^\s*_matlab\(.*\)\s*$")
MATLAB_HELPER_RE = re.compile(
    r"\n?def _matlab\(line: str\) -> None:\n"
    r"(?:    .*\n)+?"
    r"    return\n",
    re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--notebook-dir", type=Path, default=Path("notebooks"))
    parser.add_argument("--manifest", type=Path, default=Path("tools/notebooks/notebook_manifest.yml"))
    return parser.parse_args()


def _load_run_groups(path: Path) -> dict[str, str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, str] = {}
    for row in payload.get("notebooks", []):
        topic = str(row.get("topic", "")).strip()
        if topic:
            out[topic] = str(row.get("run_group", "")).strip()
    return out


def _collapse_blank_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    blank_streak = 0
    for line in lines:
        if line.strip():
            blank_streak = 0
            out.append(line.rstrip())
            continue
        blank_streak += 1
        if blank_streak <= 1:
            out.append("")
    while out and not out[-1].strip():
        out.pop()
    return out


def _sanitize_source(source: str, topic: str) -> str:
    text = source.replace("_load_matlab_globals", "_load_example_globals")
    text = MATLAB_HELPER_RE.sub("\n", text)
    raw_lines = text.splitlines()
    cleaned: list[str] = []
    replaced_banner = False
    for line in raw_lines:
        if not replaced_banner and line.startswith("# AUTO-GENERATED FROM MATLAB "):
            cleaned.append(f"# nSTAT-python notebook example: {topic}")
            replaced_banner = True
            continue
        if MATLAB_LINE_RE.match(line):
            continue
        if MATLAB_CALL_RE.match(line):
            continue
        cleaned.append(line)
    return "\n".join(_collapse_blank_lines(cleaned))


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    notebook_dir = (repo_root / args.notebook_dir).resolve()
    run_groups = _load_run_groups((repo_root / args.manifest).resolve())

    updated = 0
    for path in sorted(notebook_dir.glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        topic = path.stem
        changed = False

        for cell in nb.cells:
            new_source = _sanitize_source(cell.source, topic)
            if new_source != cell.source:
                cell.source = new_source
                changed = True

        nstat_meta = dict(nb.metadata.get("nstat", {}))
        new_meta = {
            "topic": topic,
            "run_group": run_groups.get(topic, str(nstat_meta.get("run_group", "")).strip()),
            "expected_figures": int(nstat_meta.get("expected_figures", 0)),
            "style": "python-example",
        }
        if nb.metadata.get("nstat") != new_meta:
            nb.metadata["nstat"] = new_meta
            changed = True

        if changed:
            nbformat.write(nb, path)
            updated += 1

    print(f"sanitized {updated} notebook(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
