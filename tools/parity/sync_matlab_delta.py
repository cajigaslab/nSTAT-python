#!/usr/bin/env python3
"""Generate a machine-readable MATLAB delta report for nSTAT-python syncs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--matlab-repo", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, default=Path("parity/matlab_reference.yml"))
    parser.add_argument("--port-mapping", type=Path, default=Path("parity/port_mapping.yml"))
    parser.add_argument("--out-json", type=Path, default=Path("parity/matlab_delta_report.json"))
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML payload in {path}")
    return payload


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def _mapping_index(mapping_payload: dict) -> dict[str, dict]:
    idx: dict[str, dict] = {}
    for row in mapping_payload.get("entries", []):
        if not isinstance(row, dict):
            continue
        rel = str(row.get("matlab_relpath", "")).strip()
        if rel:
            idx[rel] = row
    return idx


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    matlab_repo = args.matlab_repo.resolve()
    ref_cfg = _load_yaml((repo_root / args.reference_config).resolve())
    mapping = _load_yaml((repo_root / args.port_mapping).resolve())

    reference = ref_cfg.get("reference", {})
    if not isinstance(reference, dict):
        raise ValueError("reference config must contain a reference mapping")
    previous_ref = str(reference.get("previous_ref") or reference.get("ref") or "").strip()
    current_ref = _git(matlab_repo, "rev-parse", "HEAD")
    if not previous_ref:
        raise ValueError("matlab_reference.yml must define reference.previous_ref or reference.ref")

    diff_text = _git(matlab_repo, "diff", "--name-status", f"{previous_ref}..{current_ref}", "--", "*.m", "*.mlx")
    changed_rows = []
    mapping_idx = _mapping_index(mapping)
    requires_notebook_regeneration: list[str] = []
    requires_fixture_update: list[str] = []

    for line in [row for row in diff_text.splitlines() if row.strip()]:
        status, relpath = line.split(maxsplit=1)
        relpath = relpath.strip()
        mapped = mapping_idx.get(relpath)
        if mapped is None:
            mapped = mapping_idx.get(f"helpfiles/{Path(relpath).name}")
        row = {
            "status": status,
            "matlab_relpath": relpath,
            "mapping": mapped or None,
            "requires_notebook_regeneration": relpath.startswith("helpfiles/") or relpath.endswith("nSTATPaperExamples.mlx"),
            "requires_fixture_update": relpath.endswith(".m") and not relpath.startswith("helpfiles/"),
        }
        changed_rows.append(row)
        if row["requires_notebook_regeneration"]:
            requires_notebook_regeneration.append(relpath)
        if row["requires_fixture_update"]:
            requires_fixture_update.append(relpath)

    payload = {
        "matlab_repo": str(matlab_repo),
        "previous_ref": previous_ref,
        "current_ref": current_ref,
        "changed_files": changed_rows,
        "requires_notebook_regeneration": sorted(set(requires_notebook_regeneration)),
        "requires_fixture_update": sorted(set(requires_fixture_update)),
    }
    out_path = (repo_root / args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
