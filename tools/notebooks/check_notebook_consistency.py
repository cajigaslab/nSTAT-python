#!/usr/bin/env python3
"""Check staleness of committed help notebooks against MATLAB source hashes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import nbformat
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("parity/help_source_manifest.yml"),
        help="Help-source manifest with topic/source/notebook paths.",
    )
    parser.add_argument(
        "--hash-manifest",
        type=Path,
        default=Path("parity/notebook_source_hashes.json"),
        help="Sidecar hash manifest used for staleness checks.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Rewrite hash manifest from current sources/notebooks.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML at {path}")
    return payload


def _load_generator_module(repo_root: Path):
    mod_path = repo_root / "tools" / "notebooks" / "generate_helpfile_notebooks.py"
    spec = importlib.util.spec_from_file_location("generate_helpfile_notebooks", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import notebook generator from {mod_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_source_lines(source_path: Path, source_type: str, generator_mod: Any) -> str:
    sections = generator_mod._extract_sections(source_path, source_type)
    lines: list[str] = []
    for section in sections:
        lines.append(f"%%SECTION::{section.index}::{section.title.strip()}")
        for row in section.lines:
            raw = row.raw.rstrip()
            lines.append(raw)
    return "\n".join(lines).strip() + "\n"


def _normalize_notebook_code(notebook_path: Path) -> str:
    nb = nbformat.read(notebook_path, as_version=4)
    chunks: list[str] = []
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = str(cell.get("source", ""))
        normalized = "\n".join(line.rstrip() for line in src.splitlines()).strip("\n")
        chunks.append(normalized)
    return "\n\n".join(chunks).strip() + "\n"


def _resolve_path(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest_path = (repo_root / args.manifest).resolve()
    hash_manifest_path = (repo_root / args.hash_manifest).resolve()

    payload = _load_yaml(manifest_path)
    rows = payload.get("topics", [])
    if not isinstance(rows, list):
        raise ValueError("help_source_manifest.yml must contain topics list")

    generator_mod = _load_generator_module(repo_root)
    current: dict[str, dict[str, str]] = {}

    for row in rows:
        if not isinstance(row, dict):
            continue
        topic = str(row.get("topic", "")).strip()
        source_path = _resolve_path(str(row.get("source_path", "")).strip(), repo_root)
        source_type = str(row.get("source_type", "m")).strip().lower() or "m"
        notebook_path = _resolve_path(str(row.get("notebook_output_path", "")).strip(), repo_root)
        if not topic or not source_path.exists() or not notebook_path.exists():
            continue

        source_norm = _normalize_source_lines(source_path, source_type, generator_mod)
        notebook_norm = _normalize_notebook_code(notebook_path)
        current[topic] = {
            "source_path": str(source_path),
            "notebook_path": str(notebook_path),
            "source_hash": _sha256(source_norm),
            "notebook_hash": _sha256(notebook_norm),
        }

    if args.update or not hash_manifest_path.exists():
        hash_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        hash_manifest_path.write_text(json.dumps({"topics": current}, indent=2), encoding="utf-8")
        print(json.dumps({"status": "updated", "topics": len(current)}, indent=2))
        return 0

    recorded_payload = json.loads(hash_manifest_path.read_text(encoding="utf-8"))
    recorded = recorded_payload.get("topics", {}) if isinstance(recorded_payload, dict) else {}
    if not isinstance(recorded, dict):
        recorded = {}

    mismatches: list[dict[str, str]] = []
    for topic, cur in sorted(current.items()):
        rec = recorded.get(topic)
        if not isinstance(rec, dict):
            mismatches.append({"topic": topic, "reason": "missing_from_hash_manifest"})
            continue
        for key in ("source_hash", "notebook_hash"):
            if str(rec.get(key, "")) != str(cur.get(key, "")):
                mismatches.append({"topic": topic, "reason": f"{key}_mismatch"})
                break

    missing_topics = sorted(set(recorded.keys()) - set(current.keys()))
    for topic in missing_topics:
        mismatches.append({"topic": str(topic), "reason": "missing_from_current_manifest"})

    summary = {"topics_checked": len(current), "mismatches": mismatches}
    print(json.dumps(summary, indent=2))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
