from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT if (PROJECT_ROOT / "helpfiles").exists() else PROJECT_ROOT.parent
TOC_PATH = REPO_ROOT / "helpfiles" / "helptoc.xml"
DOCS_ROOT = PROJECT_ROOT / "docs"
TOPICS_DIR = DOCS_ROOT / "topics"
PAPER_NOMENCLATURE = PROJECT_ROOT / "examples" / "help_topics" / "paper_nomenclature.json"
MLX_METADATA = PROJECT_ROOT / "examples" / "help_topics" / "matlab_mlx_metadata.json"

CLASS_API_MAP = {
    "SignalObj": "nstat.signal.Signal",
    "Covariate": "nstat.signal.Covariate",
    "CovColl": "nstat.trial.CovariateCollection",
    "nSpikeTrain": "nstat.spikes.SpikeTrain",
    "nspikeTrain": "nstat.spikes.SpikeTrain",
    "nstColl": "nstat.spikes.SpikeTrainCollection",
    "Events": "nstat.events.Events",
    "History": "nstat.history.HistoryBasis",
    "Trial": "nstat.trial.Trial",
    "TrialConfig": "nstat.trial.TrialConfig",
    "ConfigColl": "nstat.trial.ConfigCollection",
    "Analysis": "nstat.analysis.Analysis",
    "FitResult": "nstat.fit.FitResult",
    "FitResSummary": "nstat.fit.FitSummary",
    "PPThinning": "nstat.cif.CIFModel.simulate",
    "PSTHEstimation": "nstat.spikes.SpikeTrainCollection.psth",
    "DecodingExample": "nstat.decoding.DecoderSuite",
    "DecodingExampleWithHist": "nstat.decoding.DecoderSuite",
    "StimulusDecode2D": "nstat.decoding.DecoderSuite",
}


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "topic"


def _iter_topics(root: ET.Element):
    for item in root.iter("tocitem"):
        target = item.attrib.get("target", "").strip()
        title = " ".join((item.text or "").split())
        if not title:
            title = Path(target).stem
        if not target:
            continue
        yield title, target


def _mapping_for_target(target: str) -> tuple[str, str]:
    stem = Path(target).stem
    base = stem[:-8] if (stem.endswith("Examples") and stem != "Examples") else stem
    matlab_api = base
    python_api = CLASS_API_MAP.get(base, "nstat (canonical module by topic)")
    return matlab_api, python_api


def _load_json(path: Path, default: dict[str, object]) -> dict[str, object]:
    if not path.exists():
        return default
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else default


def _topic_body(title: str, target: str) -> str:
    is_example = "example" in target.lower() or target.lower().endswith("examples.html")
    notebook_name = Path(target).stem
    matlab_api, python_api = _mapping_for_target(target)
    paper_meta = _load_json(PAPER_NOMENCLATURE, default={})
    topic_alignment = {}
    if isinstance(paper_meta.get("topic_alignment"), dict):
        topic_alignment = paper_meta["topic_alignment"].get(notebook_name, {})  # type: ignore[index]
    if not isinstance(topic_alignment, dict):
        topic_alignment = {}

    mlx_meta = _load_json(MLX_METADATA, default={})
    mlx_topic = {}
    if isinstance(mlx_meta.get("topics"), dict):
        mlx_topic = mlx_meta["topics"].get(notebook_name, {})  # type: ignore[index]
    if not isinstance(mlx_topic, dict):
        mlx_topic = {}
    lines = [
        title,
        "=" * len(title),
        "",
        f"MATLAB help target: ``{target}``",
        "",
        "Concept",
        "-------",
        "This page mirrors the corresponding MATLAB help topic and documents the Python standalone equivalent.",
        "",
        "API Mapping (MATLAB -> Python)",
        "------------------------------",
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - MATLAB API",
        "     - Python API",
        f"   * - ``{matlab_api}``",
        f"     - ``{python_api}``",
        "",
        "Migration Callout",
        "-----------------",
        "- MATLAB-style compatibility adapters remain importable for one major cycle and emit ``DeprecationWarning``.",
        "- Prefer canonical Python modules under ``nstat`` for new code.",
        "",
        "Python Usage",
        "------------",
        ".. code-block:: python",
        "",
        "   import nstat",
        "   print(nstat.__all__[:5])",
        "",
        "Data Requirements",
        "-----------------",
        "Use ``nstat.datasets.list_datasets()`` and ``nstat.datasets.get_dataset_path(...)`` to access bundled datasets.",
        "",
        "Expected Outputs",
        "----------------",
        "This topic should execute without MATLAB and produce deterministic summary metrics where applicable.",
        "",
        "Debugging Notes",
        "---------------",
        "- Confirm dataset paths with ``nstat.datasets.list_datasets()`` and ``nstat.datasets.get_dataset_path(...)``.",
        "- For notebook execution in CI/headless runs, set ``MPLBACKEND=Agg``.",
        "- If parity checks fail, inspect generated reports under ``reports/`` for topic-level details.",
        "",
        "Known Differences",
        "-----------------",
        "- Some legacy plotting helpers are represented via notebooks/docs instead of full method parity.",
        "- Numerical outputs may vary if random seeds, bin widths, or sample rates differ from MATLAB defaults.",
        "",
    ]

    if is_example:
        paper_sections = topic_alignment.get("paper_sections", [])
        if not isinstance(paper_sections, list):
            paper_sections = []
        paper_sections = [str(section).strip() for section in paper_sections if str(section).strip()]

        paper_terms = topic_alignment.get("paper_terms", [])
        if not isinstance(paper_terms, list):
            paper_terms = []
        paper_terms = [str(term).strip() for term in paper_terms if str(term).strip()]

        narrative_focus = str(topic_alignment.get("narrative_focus", "")).strip()

        has_mlx_payload = bool(mlx_topic)
        mlx_title = str(mlx_topic.get("title", title)).strip() or title
        if has_mlx_payload:
            mlx_file = str(mlx_topic.get("file", f"helpfiles/{notebook_name}.mlx")).strip() or f"helpfiles/{notebook_name}.mlx"
        else:
            mlx_file = f"helpfiles/{notebook_name}.m"
        mlx_headings = mlx_topic.get("headings", [])
        if not isinstance(mlx_headings, list):
            mlx_headings = []
        mlx_headings = [str(heading).strip() for heading in mlx_headings if str(heading).strip()][:6]

        lines.extend(
            [
                "Example Utility",
                "---------------",
                f"This example demonstrates how `{matlab_api}` workflows map to standalone Python execution and why the resulting outputs are useful for model debugging and interpretation.",
                "",
                "Paper Nomenclature",
                "------------------",
                "Use terminology consistent with Cajigas et al. (2012): conditional intensity function (CIF), point process generalized linear model (PP-GLM), maximum likelihood estimation (MLE), and related decoding/network terms where applicable.",
                "Primary paper URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC3491120/",
                "",
                "Workflow Summary",
                "----------------",
                "1. Load data (or deterministic synthetic fallback) and configure the example pipeline.",
                "2. Execute the Python topic workflow from ``examples/help_topics``.",
                "3. Review structured outputs and generated notebook figures.",
                "4. Compare behavior against MATLAB intent using parity reports when needed.",
                "",
                "MATLAB MLX Alignment",
                "--------------------",
                f"Reference Live Script: ``{mlx_file}``",
                f"MATLAB Live Script title: ``{mlx_title}``",
            ]
        )
        if not has_mlx_payload:
            lines.extend(
                [
                    "No ``.mlx`` metadata was found for this topic in the synced MATLAB helpfiles; alignment is based on the MATLAB help script naming convention.",
                    "",
                ]
            )
        if mlx_headings:
            lines.append("Key Live Script headings:")
            for heading in mlx_headings:
                lines.append(f"- {heading}")
            lines.append("")

        lines.extend(
            [
                "Paper Section Alignment",
                "-----------------------",
            ]
        )
        if paper_sections:
            for section in paper_sections:
                lines.append(f"- {section}")
        else:
            lines.append("- Section 2.2.1 (Class descriptions)")
        lines.append("")

        if paper_terms:
            lines.append("Topic-specific paper terms:")
            for term in paper_terms:
                lines.append(f"- {term}")
            lines.append("")

        if narrative_focus:
            lines.extend(
                [
                    "Section-aligned Interpretation",
                    "-----------------------------",
                    narrative_focus,
                    "",
                ]
            )

        lines.extend(
            [
                "Notebook",
                "--------",
                f"A generated executable notebook is available at ``notebooks/helpfiles/{notebook_name}.ipynb``.",
                "",
            ]
        )

    return "\n".join(lines)


def generate_docs() -> list[Path]:
    if not TOC_PATH.exists():
        raise FileNotFoundError(f"Missing TOC file: {TOC_PATH}")

    tree = ET.parse(TOC_PATH)
    toc = tree.getroot()

    TOPICS_DIR.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    topic_entries: list[tuple[str, str, Path]] = []
    seen_slugs: set[str] = set()

    for title, target in _iter_topics(toc):
        if target.startswith("http://") or target.startswith("https://"):
            continue
        slug = _slugify(Path(target).stem)
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        out = TOPICS_DIR / f"{slug}.rst"
        out.write_text(_topic_body(title, target), encoding="utf-8")
        created.append(out)
        topic_entries.append((title, target, out))

    topic_index_lines = [
        "Help Topics",
        "===========",
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        "",
    ]
    for _, _, out in topic_entries:
        topic_index_lines.append(f"   topics/{out.stem}")

    (DOCS_ROOT / "help_topics.rst").write_text("\n".join(topic_index_lines) + "\n", encoding="utf-8")

    api_lines = [
        "API Reference",
        "=============",
        "",
        ".. code-block:: python",
        "",
        "   import nstat",
        "   print(nstat.__all__)",
        "",
        "Canonical modules include:",
        "",
        "- ``nstat.signal``",
        "- ``nstat.spikes``",
        "- ``nstat.trial``",
        "- ``nstat.analysis``",
        "- ``nstat.fit``",
        "- ``nstat.cif``",
        "- ``nstat.decoding``",
        "- ``nstat.datasets``",
    ]
    (DOCS_ROOT / "api.rst").write_text("\n".join(api_lines) + "\n", encoding="utf-8")

    index_lines = [
        "nSTAT Python Documentation",
        "==========================",
        "",
        "Standalone Python port of nSTAT with MATLAB-help topic coverage and executable notebooks.",
        "",
        ".. toctree::",
        "   :maxdepth: 2",
        "",
        "   api",
        "   help_topics",
        "   parity_runbook",
        "   notebook_figure_parity",
        "   repo_split_status",
    ]
    (DOCS_ROOT / "index.rst").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    return created


def main() -> int:
    created = generate_docs()
    print(f"generated_topics={len(created)}")
    print(f"docs_root={DOCS_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
