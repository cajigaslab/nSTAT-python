#!/usr/bin/env python3
"""Checkout pinned MATLAB nSTAT reference repo at an immutable commit."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    env = os.environ.copy()
    env.setdefault("GIT_LFS_SKIP_SMUDGE", "1")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("parity/matlab_reference.yml"),
        help="Pinned MATLAB reference config YAML.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("/tmp/upstream-nstat"),
        help="Destination directory for checked-out MATLAB repo.",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=Path("parity/matlab_reference_checkout.json"),
        help="Optional JSON metadata output path.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    repo_url = str(cfg.get("repo_url", "")).strip()
    ref = str(cfg.get("ref", "")).strip()
    if not repo_url or not ref:
        raise ValueError("Config must define non-empty repo_url and ref")

    dest = args.dest.resolve()
    if dest.exists():
        shutil.rmtree(dest)
    _run(["git", "clone", "--depth", "1", "--no-tags", repo_url, str(dest)])
    _run(["git", "fetch", "--depth", "1", "origin", ref], cwd=dest)
    _run(["git", "checkout", "--detach", "--force", "FETCH_HEAD"], cwd=dest)
    sha = _run(["git", "rev-parse", "HEAD"], cwd=dest)

    if sha.lower() != ref.lower():
        raise RuntimeError(
            f"Pinned checkout mismatch: expected {ref}, resolved {sha}. "
            "Reference is not immutable."
        )

    metadata = {
        "repo_url": repo_url,
        "requested_ref": ref,
        "resolved_sha": sha,
        "dest": str(dest),
    }
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Checked out MATLAB reference at {sha} -> {dest}")
    print(f"Wrote metadata: {args.metadata_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
