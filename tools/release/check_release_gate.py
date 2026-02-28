#!/usr/bin/env python3
"""Release gate checks for nSTAT-python v1 readiness."""

from __future__ import annotations

from pathlib import Path

import yaml


REQUIRED_CLASS_HELP = [
    "Analysis",
    "CIF",
    "ConfidenceInterval",
    "ConfigColl",
    "CovColl",
    "Covariate",
    "DecodingAlgorithms",
    "Events",
    "FitResSummary",
    "FitResult",
    "History",
    "SignalObj",
    "Trial",
    "TrialConfig",
    "nspikeTrain",
    "nstColl",
]



def main() -> int:
    missing: list[str] = []

    for klass in REQUIRED_CLASS_HELP:
        page = Path("docs/help/classes") / f"{klass}.md"
        if not page.exists():
            missing.append(f"missing class help page: {page}")

    manifest = yaml.safe_load(Path("tools/notebooks/notebook_manifest.yml").read_text(encoding="utf-8"))
    for row in manifest["notebooks"]:
        notebook = Path(row["file"])
        if not notebook.exists():
            missing.append(f"missing notebook: {notebook}")

    if missing:
        print("Release gate failed:")
        for msg in missing:
            print(f"  - {msg}")
        return 1

    print("Release gate checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
