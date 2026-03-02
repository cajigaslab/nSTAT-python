#!/usr/bin/env python3
"""Regenerate data/datasets_manifest.json entries from a MATLAB mirror manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to matlab_gold manifest JSON.")
    parser.add_argument(
        "--datasets-manifest",
        default="data/datasets_manifest.json",
        help="Path to nSTAT datasets manifest JSON.",
    )
    parser.add_argument(
        "--upstream-raw-prefix",
        default="https://raw.githubusercontent.com/cajigaslab/nSTAT/master/data/",
        help="Raw URL prefix for upstream MATLAB data files.",
    )
    parser.add_argument(
        "--keep-legacy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep non-mirror legacy dataset rows (default: true).",
    )
    return parser.parse_args()


def _resolve(path_arg: str, repo_root: Path) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _mirror_name(version: str, rel_path: str) -> str:
    return f"matlab_gold_{version}/{rel_path}"


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    mirror_manifest_path = _resolve(args.manifest, repo_root)
    datasets_manifest_path = _resolve(args.datasets_manifest, repo_root)

    mirror = _load_json(mirror_manifest_path)
    datasets_payload = _load_json(datasets_manifest_path)

    version = str(mirror["dataset_version"])
    files = mirror.get("files", [])
    if not isinstance(files, list):
        raise KeyError("Mirror manifest missing files list.")

    generated_rows: list[dict] = []
    for row in files:
        rel_path = str(row["relative_path"])
        url = args.upstream_raw_prefix + quote(rel_path, safe="/")
        generated_rows.append(
            {
                "name": _mirror_name(version, rel_path),
                "version": version,
                "url": url,
                "sha256": str(row["sha256"]),
                "filename": rel_path,
            }
        )

    current_rows = list(datasets_payload.get("datasets", []))
    if args.keep_legacy:
        retained = [
            row
            for row in current_rows
            if not str(row.get("name", "")).startswith("matlab_gold_")
        ]
    else:
        retained = []

    all_rows = retained + generated_rows
    all_rows.sort(key=lambda r: (str(r["name"]), str(r["version"])))

    datasets_payload["datasets"] = all_rows
    if "schema_version" not in datasets_payload:
        datasets_payload["schema_version"] = 1
    if "notes" not in datasets_payload:
        datasets_payload["notes"] = (
            "Only shared example data are listed here. "
            "Non-data assets are never shared with MATLAB nSTAT."
        )

    _dump_json(datasets_manifest_path, datasets_payload)

    print(f"Mirror manifest: {mirror_manifest_path}")
    print(f"Datasets manifest: {datasets_manifest_path}")
    print(f"Generated mirror dataset entries: {len(generated_rows)}")
    print(f"Total dataset entries: {len(all_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

