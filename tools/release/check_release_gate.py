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

    class_manifest = yaml.safe_load(
        Path("baseline/class_method_inventory.yml").read_text(encoding="utf-8")
    )
    expected_classes = sorted(
        {row["matlab_class"] for row in class_manifest["classes"]} | set(REQUIRED_CLASS_HELP)
    )

    for klass in expected_classes:
        page = Path("docs/help/classes") / f"{klass}.md"
        if not page.exists():
            missing.append(f"missing class help page: {page}")

    manifest = yaml.safe_load(Path("tools/notebooks/notebook_manifest.yml").read_text(encoding="utf-8"))
    help_toc = yaml.safe_load(Path("docs/help/helptoc.yml").read_text(encoding="utf-8"))
    toc_text = str(help_toc).lower()

    for row in manifest["notebooks"]:
        topic = row["topic"]
        notebook = Path(row["file"])
        help_page = Path("docs/help/examples") / f"{topic}.md"

        if not notebook.exists():
            missing.append(f"missing notebook: {notebook}")
        if not help_page.exists():
            missing.append(f"missing example help page: {help_page}")

        if row.get("run_group") not in {"smoke", "full"}:
            missing.append(f"invalid run_group for {topic}: {row.get('run_group')}")

        if topic.lower() not in toc_text:
            missing.append(f"topic missing from help TOC: {topic}")

    notebooks_index = Path("docs/notebooks.md")
    if not notebooks_index.exists():
        missing.append("missing notebook catalog page: docs/notebooks.md")

    if missing:
        print("Release gate failed:")
        for msg in missing:
            print(f"  - {msg}")
        return 1

    print("Release gate checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
