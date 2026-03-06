#!/usr/bin/env python3
"""Check notebook/source consistency and freeze source hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nbformat
import yaml



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-manifest", type=Path, default=Path("parity/help_source_manifest.yml"))
    parser.add_argument("--out-json", type=Path, default=Path("parity/notebook_source_hashes.json"))
    parser.add_argument("--topics", default="")
    return parser.parse_args()



def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest = yaml.safe_load((repo_root / args.source_manifest).read_text(encoding="utf-8")) or {}
    rows = manifest.get("topics", [])
    if args.topics.strip():
        wanted = {token.strip() for token in args.topics.split(",") if token.strip()}
        rows = [row for row in rows if str(row.get("topic", "")).strip() in wanted]
        if not rows:
            raise RuntimeError(f"No topics matched --topics={args.topics!r}")

    results = []
    failures = []
    for row in rows:
        topic = str(row["topic"])
        source_path = Path(str(row["source_path"])).expanduser().resolve()
        notebook_path = Path(str(row["notebook_output_path"])).expanduser().resolve()
        expected_sections = int(row.get("expected_section_count", 0))

        result = {
            "topic": topic,
            "source_path": str(source_path),
            "notebook_path": str(notebook_path),
            "source_sha256": _sha256(source_path) if source_path.exists() else "",
            "notebook_sha256": _sha256(notebook_path) if notebook_path.exists() else "",
            "expected_sections": expected_sections,
            "status": "ok",
        }
        if not source_path.exists():
            result["status"] = "missing_source"
            failures.append(f"{topic}: missing source {source_path}")
        elif not notebook_path.exists():
            result["status"] = "missing_notebook"
            failures.append(f"{topic}: missing notebook {notebook_path}")
        else:
            nb = nbformat.read(notebook_path, as_version=4)
            if len(nb.cells) != expected_sections:
                result["status"] = "cell_count_mismatch"
                failures.append(
                    f"{topic}: notebook has {len(nb.cells)} cell(s), expected {expected_sections}"
                )
            source_file_meta = str(nb.metadata.get("nstat", {}).get("source_file", "")).strip()
            if source_file_meta and Path(source_file_meta).name != source_path.name:
                result["status"] = "source_mismatch"
                failures.append(
                    f"{topic}: notebook metadata source {source_file_meta} does not match {source_path.name}"
                )
        results.append(result)

    payload = {
        "topics": results,
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }
    out_path = (repo_root / args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out_path)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
