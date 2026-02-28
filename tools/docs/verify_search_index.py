#!/usr/bin/env python3
"""Verify Sphinx search index includes key nSTAT help topics/classes."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


REQUIRED_CLASS_TOKENS = [
    "signalobj",
    "covariate",
    "confidenceinterval",
    "events",
    "history",
    "nspiketrain",
    "nstcoll",
    "covcoll",
    "trialconfig",
    "configcoll",
    "trial",
    "cif",
    "analysis",
    "fitresult",
    "fitressummary",
    "decodingalgorithms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-index",
        type=Path,
        default=Path("docs/_build/html/searchindex.js"),
        help="Path to built Sphinx searchindex.js",
    )
    parser.add_argument(
        "--notebook-manifest",
        type=Path,
        default=Path("tools/notebooks/notebook_manifest.yml"),
        help="Notebook manifest for expected example topics",
    )
    return parser.parse_args()



def main() -> int:
    args = parse_args()

    if not args.search_index.exists():
        raise FileNotFoundError(
            f"search index not found: {args.search_index}. Build docs first with sphinx-build."
        )

    payload = args.search_index.read_text(encoding="utf-8").lower()
    manifest = yaml.safe_load(args.notebook_manifest.read_text(encoding="utf-8"))

    required_topics = [row["topic"].lower() for row in manifest["notebooks"]]
    required_tokens = sorted(set(REQUIRED_CLASS_TOKENS + required_topics))

    missing = [token for token in required_tokens if token not in payload]
    if missing:
        print("Search index verification FAILED.")
        for token in missing:
            print(f"  - missing token: {token}")
        return 1

    print(
        "Search index verification PASSED "
        f"({len(required_tokens)} class/topic tokens found in search index)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
