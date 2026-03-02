#!/usr/bin/env python3
"""Fail when mirrored-data artifacts are out of sync.

Checks:
1. Every `matlab_gold_<version>.manifest.json` has matching source manifest.
2. Source/mirror manifests agree on relative_path + size + sha256.
3. Shared-data allowlist has one entry per mirrored file with matching checksum.
4. data/datasets_manifest.json has one mirror dataset row per mirrored file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shared-root",
        default="data/shared",
        help="Directory containing matlab_source/matlab_gold manifests.",
    )
    parser.add_argument(
        "--allowlist",
        default="tools/compliance/shared_data_allowlist.yml",
        help="Shared-data allowlist YAML.",
    )
    parser.add_argument(
        "--datasets-manifest",
        default="data/datasets_manifest.json",
        help="Datasets manifest JSON.",
    )
    return parser.parse_args()


def _resolve(path_arg: str, repo_root: Path) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_allowlist(path: Path) -> list[dict]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = payload.get("shared_data", [])
    return rows if isinstance(rows, list) else []


def _file_map(files: list[dict]) -> dict[str, tuple[int, str]]:
    out: dict[str, tuple[int, str]] = {}
    for row in files:
        out[str(row["relative_path"])] = (int(row["size_bytes"]), str(row["sha256"]))
    return out


def _extract_version_from_manifest_name(name: str) -> str:
    # matlab_gold_20260302.manifest.json -> 20260302
    stem = name.replace(".manifest.json", "")
    return stem.removeprefix("matlab_gold_")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    args = parse_args()

    shared_root = _resolve(args.shared_root, repo_root)
    allowlist_path = _resolve(args.allowlist, repo_root)
    datasets_manifest_path = _resolve(args.datasets_manifest, repo_root)

    allowlist_rows = _load_allowlist(allowlist_path)
    dataset_rows = _load_json(datasets_manifest_path).get("datasets", [])

    allow_idx: dict[tuple[str, str], dict] = {}
    for row in allowlist_rows:
        py_path = str(row.get("python_path", ""))
        sha = str(row.get("sha256", ""))
        if py_path and sha:
            allow_idx[(py_path, sha)] = row

    ds_idx: dict[tuple[str, str], dict] = {}
    for row in dataset_rows:
        name = str(row.get("name", ""))
        sha = str(row.get("sha256", ""))
        if name and sha:
            ds_idx[(name, sha)] = row

    errors: list[str] = []

    gold_manifests = sorted(shared_root.glob("matlab_gold_*.manifest.json"))
    if not gold_manifests:
        errors.append(f"No matlab_gold manifests found in {shared_root}")

    for gold_path in gold_manifests:
        gold = _load_json(gold_path)
        version = str(gold.get("dataset_version") or _extract_version_from_manifest_name(gold_path.name))
        source_path = shared_root / f"matlab_source_{version}.manifest.json"
        if not source_path.exists():
            errors.append(f"Missing source manifest for version {version}: {source_path}")
            continue

        source = _load_json(source_path)
        gold_files = gold.get("files", [])
        source_files = source.get("files", [])
        if not isinstance(gold_files, list) or not isinstance(source_files, list):
            errors.append(f"Malformed files list in manifests for version {version}")
            continue

        gmap = _file_map(gold_files)
        smap = _file_map(source_files)
        if gmap != smap:
            errors.append(f"Source/mirror manifest mismatch for version {version}")

        mirror_root = str(gold.get("mirror_root", f"data/shared/matlab_gold_{version}"))
        source_manifest_rel = gold_path.relative_to(repo_root).as_posix()

        for rel_path, (_, sha) in gmap.items():
            py_path = f"{mirror_root}/{rel_path}"
            arow = allow_idx.get((py_path, sha))
            if arow is None:
                errors.append(f"Allowlist missing: {py_path}")
            else:
                if str(arow.get("source_manifest", "")) != source_manifest_rel:
                    errors.append(f"Allowlist source_manifest mismatch for {py_path}")
                expected_upstream = f"data/{rel_path}"
                if str(arow.get("upstream_path", "")) != expected_upstream:
                    errors.append(f"Allowlist upstream_path mismatch for {py_path}")

            ds_name = f"matlab_gold_{version}/{rel_path}"
            drow = ds_idx.get((ds_name, sha))
            if drow is None:
                errors.append(f"datasets_manifest missing row: {ds_name}")
            else:
                if str(drow.get("filename", "")) != rel_path:
                    errors.append(f"datasets_manifest filename mismatch for {ds_name}")

    if errors:
        print("Mirror regeneration check FAILED:")
        for err in errors[:200]:
            print(f"  - {err}")
        if len(errors) > 200:
            print(f"  ... and {len(errors) - 200} more")
        return 1

    print("Mirror regeneration check PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

