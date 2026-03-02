#!/usr/bin/env python3
"""Generate clean-room help pages, TOC, and cross-links for nSTAT-python."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_NOTEBOOK_BASE = "https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks"
PAPER_OVERVIEW = "../paper_overview.md"
API_PAGE = "../../api.md"
REPO_PARITY_BASE = "https://github.com/cajigaslab/nSTAT-python/blob/main/parity"


CLASS_MAP = [
    ("SignalObj", "nstat.signal.Signal"),
    ("Covariate", "nstat.signal.Covariate"),
    ("ConfidenceInterval", "nstat.confidence.ConfidenceInterval"),
    ("Events", "nstat.events.Events"),
    ("History", "nstat.history.HistoryBasis"),
    ("nspikeTrain", "nstat.spikes.SpikeTrain"),
    ("nstColl", "nstat.spikes.SpikeTrainCollection"),
    ("CovColl", "nstat.trial.CovariateCollection"),
    ("TrialConfig", "nstat.trial.TrialConfig"),
    ("ConfigColl", "nstat.trial.ConfigCollection"),
    ("Trial", "nstat.trial.Trial"),
    ("CIF", "nstat.cif.CIFModel"),
    ("Analysis", "nstat.analysis.Analysis"),
    ("FitResult", "nstat.fit.FitResult"),
    ("FitResSummary", "nstat.fit.FitSummary"),
    ("DecodingAlgorithms", "nstat.decoding.DecodingAlgorithms"),
]

CLASS_NOTEBOOKS = {
    "SignalObj": ["SignalObjExamples", "AnalysisExamples"],
    "Covariate": ["CovariateExamples", "TrialExamples"],
    "ConfidenceInterval": ["DecodingExample", "FitResSummaryExamples"],
    "Events": ["EventsExamples", "NetworkTutorial"],
    "History": ["HistoryExamples", "DecodingExampleWithHist"],
    "nspikeTrain": ["nSpikeTrainExamples", "PPSimExample"],
    "nstColl": ["nstCollExamples", "TrialExamples"],
    "CovColl": ["CovCollExamples", "TrialExamples"],
    "TrialConfig": ["TrialConfigExamples", "AnalysisExamples"],
    "ConfigColl": ["ConfigCollExamples", "AnalysisExamples"],
    "Trial": ["TrialExamples", "AnalysisExamples"],
    "CIF": ["PPSimExample", "PPThinning"],
    "Analysis": ["AnalysisExamples", "nSTATPaperExamples"],
    "FitResult": ["FitResultExamples", "nSTATPaperExamples"],
    "FitResSummary": ["FitResSummaryExamples", "nSTATPaperExamples"],
    "DecodingAlgorithms": ["DecodingExample", "StimulusDecode2D"],
}


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def _latest_snapshot(parity_root: Path) -> Path | None:
    candidates = sorted(parity_root.glob("matlab_gold_snapshot_*.yml"))
    if not candidates:
        return None
    return candidates[-1]


def generate_class_page(help_root: Path, matlab_name: str, python_target: str) -> None:
    related = CLASS_NOTEBOOKS.get(matlab_name, [])
    related_lines = "\n".join(
        line
        for topic in related
        for line in [
            f"- [{topic} notebook]({REPO_NOTEBOOK_BASE}/{topic}.ipynb)",
            f"- [{topic} help page](../examples/{topic}.md)",
        ]
    )

    if not related_lines:
        related_lines = "- See [Examples Index](../examples_index.md)."

    content = f"""# {matlab_name}

Python implementation: `{python_target}`

## Purpose
This class preserves MATLAB-facing structure while using a Python-native,
fully independent implementation in `nSTAT-python`.

## References
- [API reference]({API_PAGE})
- [Paper overview]({PAPER_OVERVIEW})
- DOI: `10.1016/j.jneumeth.2012.08.009`
- PMID: `22981419`

## Related learning resources
{related_lines}
"""
    write_text(help_root / "classes" / f"{matlab_name}.md", content)


def generate_example_page(help_root: Path, topic: str, run_group: str) -> None:
    content = f"""# {topic}

Python-native tutorial page for `{topic}`.

## Notebook
- [{topic}.ipynb]({REPO_NOTEBOOK_BASE}/{topic}.ipynb)
- Execution group: `{run_group}`

## Linked references
- [Examples index](../examples_index.md)
- [Class definitions](../class_definitions.md)
- [Paper overview](../paper_overview.md)
- DOI: `10.1016/j.jneumeth.2012.08.009`
- PMID: `22981419`
"""
    write_text(help_root / "examples" / f"{topic}.md", content)


def generate_examples_index(help_root: Path, topics: list[str]) -> None:
    links = "\n".join([f"- [{topic}](examples/{topic}.md)" for topic in topics])
    content = f"""# Examples Index

The topics below map clean-room Python workflows to executable notebooks.

{links}

All notebooks are available in the repository's `notebooks/` directory and are
validated in CI.
"""
    write_text(help_root / "examples_index.md", content)


def generate_class_definitions(help_root: Path) -> None:
    rows = "\n".join(
        [f"| {matlab_name} | {python_target} |" for matlab_name, python_target in CLASS_MAP]
    )
    content = f"""# Class Definitions

Each MATLAB nSTAT class has a corresponding Python help page and implementation.

| MATLAB class | Python target |
|---|---|
{rows}
"""
    write_text(help_root / "class_definitions.md", content)


def generate_help_toc(help_root: Path, topics: list[str]) -> None:
    entries: dict[str, object] = {
        "root": "help/index.md",
        "entries": [
            {"title": "Home", "target": "help/index.md"},
            {"title": "Class Definitions", "target": "help/class_definitions.md"},
            {"title": "Examples Index", "target": "help/examples_index.md"},
            {"title": "Paper Overview", "target": "help/paper_overview.md"},
            {"title": "Parity Dashboard", "target": "help/parity_dashboard.md"},
            {
                "title": "Classes",
                "children": [
                    {
                        "title": matlab_name,
                        "target": f"help/classes/{matlab_name}.md",
                    }
                    for matlab_name, _ in CLASS_MAP
                ],
            },
            {
                "title": "Examples",
                "children": [
                    {
                        "title": topic,
                        "target": f"help/examples/{topic}.md",
                    }
                    for topic in topics
                ],
            },
        ],
    }
    write_text(help_root / "helptoc.yml", yaml.safe_dump(entries, sort_keys=False))


def generate_help_home(help_root: Path, topics: list[str]) -> None:
    class_refs = "\n".join([f"classes/{matlab_name}" for matlab_name, _ in CLASS_MAP])
    topic_refs = "\n".join([f"examples/{topic}" for topic in topics])

    content = f"""# nSTAT-python Help Home

Welcome to the clean-room Python help system for `nSTAT-python`.

This site preserves class/workflow structure of MATLAB nSTAT while keeping
all implementation, docs, and tooling Python-specific.

## Navigation
- [Class Definitions](class_definitions.md)
- [Examples Index](examples_index.md)
- [Paper Overview](paper_overview.md)
- [Parity Dashboard](parity_dashboard.md)

```{{toctree}}
:maxdepth: 2

class_definitions
examples_index
paper_overview
parity_dashboard
{class_refs}
{topic_refs}
```
"""
    write_text(help_root / "index.md", content)


def generate_parity_dashboard(help_root: Path, repo_root: Path) -> None:
    parity_root = repo_root / "parity"
    gap = _read_json(parity_root / "parity_gap_report.json").get("summary", {})
    functional = _read_json(parity_root / "function_example_alignment_report.json")
    numeric = _read_json(parity_root / "numeric_drift_report.json").get("summary", {})
    example_spec = _read_yaml(parity_root / "example_output_spec.yml")
    snapshot_path = _latest_snapshot(parity_root)
    snapshot = _read_yaml(snapshot_path) if snapshot_path is not None else {}

    method_summary = functional.get("method_functional_audit", {}).get("summary", {})
    example_summary = functional.get("example_line_alignment_audit", {}).get("summary", {})
    out_of_scope_topics = example_spec.get("out_of_scope_topics", [])

    snapshot_name = snapshot_path.name if snapshot_path is not None else "-"
    snapshot_id = snapshot.get("snapshot_id", "-")
    snapshot_date = snapshot.get("captured_on", "-")
    source_sha = snapshot.get("source", {}).get("manifest_sha256", "-")
    mirror_sha = snapshot.get("mirror", {}).get("manifest_sha256", "-")
    mirror_count = snapshot.get("mirror", {}).get("file_count", "-")

    oos_lines = "\n".join([f"- `{topic}`" for topic in out_of_scope_topics]) or "- None"

    content = f"""# Parity Dashboard

This dashboard summarizes current MATLAB-to-Python parity status from generated
artifacts in the `parity/` directory.

## Structural parity
| Metric | Value |
|---|---:|
| High gaps | {int(gap.get("high", 0))} |
| Medium gaps | {int(gap.get("medium", 0))} |
| Low gaps | {int(gap.get("low", 0))} |
| Total gaps | {int(gap.get("total", 0))} |

## Functional parity (methods)
| Metric | Value |
|---|---:|
| Total methods | {int(method_summary.get("total_methods", 0))} |
| Contract-verified | {int(method_summary.get("contract_verified_methods", 0))} |
| Contract-explicit verified | {int(method_summary.get("contract_explicit_verified_methods", 0))} |
| Probe-verified | {int(method_summary.get("probe_verified_methods", 0))} |
| Excluded methods | {int(method_summary.get("excluded_methods", 0))} |
| Missing symbols | {int(method_summary.get("missing_symbol_methods", 0))} |
| Unverified behavior | {int(method_summary.get("unverified_behavior_methods", 0))} |

## Example parity
| Metric | Value |
|---|---:|
| Total topics | {int(example_summary.get("total_topics", 0))} |
| Validated topics | {int(example_summary.get("validated_topics", 0))} |
| MATLAB doc-only topics | {int(example_summary.get("matlab_doc_only_topics", 0))} |
| Pending manual review topics | {int(example_summary.get("pending_manual_review_topics", 0))} |
| Missing executable topics | {int(example_summary.get("missing_executable_topics", 0))} |

### Out-of-scope example topics
{oos_lines}

## Numeric drift
| Metric | Value |
|---|---:|
| Topics checked | {int(numeric.get("topics", 0))} |
| Topics passed | {int(numeric.get("passed_topics", 0))} |
| Topics failed | {int(numeric.get("failed_topics", 0))} |
| Metrics checked | {int(numeric.get("checked_metrics", 0))} |
| Metrics failed | {int(numeric.get("failed_metrics", 0))} |

## Frozen MATLAB data snapshot
| Metric | Value |
|---|---|
| Snapshot file | `{snapshot_name}` |
| Snapshot id | `{snapshot_id}` |
| Snapshot date | `{snapshot_date}` |
| Mirror file count | `{mirror_count}` |
| Source manifest SHA256 | `{source_sha}` |
| Mirror manifest SHA256 | `{mirror_sha}` |

## Artifact links
- [parity_gap_report.json]({REPO_PARITY_BASE}/parity_gap_report.json)
- [function_example_alignment_report.json]({REPO_PARITY_BASE}/function_example_alignment_report.json)
- [numeric_drift_report.json]({REPO_PARITY_BASE}/numeric_drift_report.json)
- [example_output_spec.yml]({REPO_PARITY_BASE}/example_output_spec.yml)
- [method_closure_sprint.md]({REPO_PARITY_BASE}/method_closure_sprint.md)
"""
    write_text(help_root / "parity_dashboard.md", content)


def generate_notebook_index(repo_root: Path, manifest_rows: list[dict[str, str]]) -> None:
    lines = []
    for row in manifest_rows:
        topic = row["topic"]
        run_group = row["run_group"]
        notebook_url = f"{REPO_NOTEBOOK_BASE}/{topic}.ipynb"
        lines.append(f"- [{topic}.ipynb]({notebook_url}) (`{run_group}`)")

    content = (
        "# Notebook Catalog\n\n"
        "The notebooks below are generated from a clean-room manifest and executed in CI.\n\n"
        + "\n".join(lines)
        + "\n"
    )
    write_text(repo_root / "docs" / "notebooks.md", content)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    help_root = repo_root / "docs" / "help"
    notebook_manifest = repo_root / "tools" / "notebooks" / "notebook_manifest.yml"

    manifest = yaml.safe_load(notebook_manifest.read_text(encoding="utf-8"))
    rows = manifest["notebooks"]
    topics = [row["topic"] for row in rows]
    run_group = {row["topic"]: row["run_group"] for row in rows}

    for matlab_name, python_target in CLASS_MAP:
        generate_class_page(help_root=help_root, matlab_name=matlab_name, python_target=python_target)

    for topic in topics:
        generate_example_page(help_root=help_root, topic=topic, run_group=run_group[topic])

    generate_examples_index(help_root=help_root, topics=topics)
    generate_class_definitions(help_root=help_root)
    generate_help_toc(help_root=help_root, topics=topics)
    generate_help_home(help_root=help_root, topics=topics)
    generate_parity_dashboard(help_root=help_root, repo_root=repo_root)
    generate_notebook_index(repo_root=repo_root, manifest_rows=rows)

    mapping = {"classes": [{"matlab": m, "python": p} for m, p in CLASS_MAP], "topics": topics}
    write_text(repo_root / "baseline" / "help_mapping.json", json.dumps(mapping, indent=2) + "\n")

    print("Generated help pages, TOC, and mapping artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
