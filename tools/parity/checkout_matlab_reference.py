#!/usr/bin/env python3
"""Resolve or checkout pinned MATLAB nSTAT reference repo."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("parity/matlab_reference.yml"))
    parser.add_argument("--dest", type=Path, default=Path("/tmp/upstream-nstat"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    return parser.parse_args()


def _load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid config: {path}")
    return payload


def _clone(remote_url: str, ref: str, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.run(["git", "clone", remote_url, str(dest)], check=True)
    if ref:
        subprocess.run(["git", "-C", str(dest), "checkout", ref], check=True)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    cfg = _load_config((repo_root / args.config).resolve())
    ref = cfg.get("reference", {})
    if not isinstance(ref, dict):
        raise ValueError("reference config must include a 'reference' mapping")

    local_path = str(ref.get("local_path", "")).strip()
    remote_url = str(ref.get("remote_url", "")).strip()
    git_ref = str(ref.get("ref", "")).strip()

    dest = args.dest.resolve()
    source_dir: Path | None = None

    if local_path:
        local = Path(local_path)
        if not local.is_absolute():
            local = (repo_root / local).resolve()
        if local.exists():
            source_dir = local

    if source_dir is None and remote_url:
        _clone(remote_url, git_ref, dest)
        source_dir = dest

    if source_dir is None:
        raise FileNotFoundError(
            "Could not resolve MATLAB reference. Set reference.local_path or reference.remote_url."
        )

    help_subdir = str(ref.get("helpfiles_subdir", "helpfiles")).strip() or "helpfiles"
    help_root = source_dir / help_subdir
    if not help_root.exists():
        raise FileNotFoundError(f"Helpfiles directory missing: {help_root}")

    print(str(source_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
