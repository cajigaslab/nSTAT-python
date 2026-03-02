#!/usr/bin/env python3
"""Update shared-data allowlist entries from a mirrored MATLAB data manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to matlab_gold manifest JSON.")
    parser.add_argument(
        "--allowlist",
        default="tools/compliance/shared_data_allowlist.yml",
        help="Path to shared-data allowlist YAML.",
    )
    parser.add_argument(
        "--upstream-repo",
        default="cajigaslab/nSTAT",
        help="Upstream MATLAB repository identifier.",
    )
    parser.add_argument(
        "--upstream-data-prefix",
        default="data",
        help="Upstream data root prefix in repository.",
    )
    return parser.parse_args()


def _resolve(path_arg: str, repo_root: Path) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_allowlist(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "shared_data": []}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"version": 1, "shared_data": []}
    payload.setdefault("version", 1)
    payload.setdefault("shared_data", [])
    return payload


def _normalize_shared_data(rows: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        python_path = row.get("python_path")
        upstream_path = row.get("upstream_path")
        sha256 = row.get("sha256")
        upstream_repo = row.get("upstream_repo")
        if not (python_path and upstream_path and sha256):
            continue
        clean_row = {
            "python_path": str(python_path),
            "upstream_repo": str(upstream_repo) if upstream_repo else "cajigaslab/nSTAT",
            "upstream_path": str(upstream_path),
            "sha256": str(sha256),
        }
        source_manifest = row.get("source_manifest")
        if source_manifest:
            clean_row["source_manifest"] = str(source_manifest)
        normalized.append(clean_row)
    return normalized


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    args = parse_args()
    manifest_path = _resolve(args.manifest, repo_root)
    allowlist_path = _resolve(args.allowlist, repo_root)

    manifest = _load_manifest(manifest_path)
    mirror_root = manifest.get("mirror_root")
    files = manifest.get("files")
    if not isinstance(mirror_root, str) or not mirror_root:
        raise KeyError("manifest is missing required key: mirror_root")
    if not isinstance(files, list):
        raise KeyError("manifest is missing required key: files")

    payload = _load_allowlist(allowlist_path)
    existing = _normalize_shared_data(payload.get("shared_data", []))

    generated_rows: list[dict] = []
    mirror_prefix = f"{mirror_root}/"
    for row in files:
        rel = row["relative_path"]
        generated_rows.append(
            {
                "python_path": f"{mirror_root}/{rel}",
                "upstream_repo": args.upstream_repo,
                "upstream_path": f"{args.upstream_data_prefix}/{rel}",
                "sha256": row["sha256"],
                "source_manifest": manifest_path.relative_to(repo_root).as_posix(),
            }
        )

    # Drop previous generated entries for this mirror root, keep other legacy/handwritten rows.
    retained = [row for row in existing if not row["python_path"].startswith(mirror_prefix)]
    combined = retained + generated_rows

    # Deduplicate exact keys, then sort deterministically by python path.
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict] = []
    for row in combined:
        key = (row["python_path"], row["upstream_path"], row["sha256"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    deduped.sort(key=lambda row: row["python_path"])

    out_payload = {
        "version": payload.get("version", 1),
        "shared_data": deduped,
    }
    allowlist_path.parent.mkdir(parents=True, exist_ok=True)
    allowlist_path.write_text(
        yaml.safe_dump(out_payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )

    print(f"Manifest: {manifest_path}")
    print(f"Allowlist: {allowlist_path}")
    print(f"Generated entries: {len(generated_rows)}")
    print(f"Total allowlist entries: {len(deduped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

