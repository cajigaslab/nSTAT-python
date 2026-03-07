#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).resolve().parent / "notebook_manifest.yml"
GROUPS_PATH = Path(__file__).resolve().parent / "topic_groups.yml"
NOTEBOOK_INFRA_PATTERNS = (
    "tools/notebooks/",
    "nstat/notebook_",
    "parity/notebook_fidelity.yml",
    "parity/report.md",
)


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> dict[str, str]:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    return {str(row["file"]): str(row["topic"]) for row in payload.get("notebooks", [])}


def load_group(name: str, groups_path: Path = GROUPS_PATH) -> list[str]:
    payload = yaml.safe_load(groups_path.read_text(encoding="utf-8")) or {}
    groups = payload.get("groups", {})
    group = groups.get(name, [])
    return [str(item) for item in group]


def infer_topics_from_paths(paths: list[str], manifest: dict[str, str], fallback_group: list[str]) -> list[str]:
    changed = [path.strip() for path in paths if path.strip()]
    direct_topics = sorted({manifest[path] for path in changed if path in manifest})
    if direct_topics:
        return direct_topics
    if any(path.startswith(NOTEBOOK_INFRA_PATTERNS) for path in changed):
        return sorted(set(fallback_group))
    return []


def changed_paths(base_sha: str, head_sha: str, repo_root: Path = REPO_ROOT) -> list[str]:
    proc = subprocess.run(
        ["git", "diff", "--name-only", base_sha, head_sha],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--groups-file", type=Path, default=GROUPS_PATH)
    parser.add_argument("--fallback-group", default="parity_core")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    fallback = load_group(args.fallback_group, args.groups_file)
    topics = infer_topics_from_paths(changed_paths(args.base_sha, args.head_sha), manifest, fallback)
    print(",".join(topics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
