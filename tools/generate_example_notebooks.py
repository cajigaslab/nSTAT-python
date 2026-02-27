from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT if (PROJECT_ROOT / "helpfiles").exists() else PROJECT_ROOT.parent
TOC_PATH = REPO_ROOT / "helpfiles" / "helptoc.xml"
NB_ROOT = PROJECT_ROOT / "notebooks" / "helpfiles"
SRC_ROOT = PROJECT_ROOT / "examples" / "help_topics"
FIGURE_CONTRACT = SRC_ROOT / "figure_contract.json"
PAPER_NOMENCLATURE = SRC_ROOT / "paper_nomenclature.json"
MLX_METADATA = SRC_ROOT / "matlab_mlx_metadata.json"

REQUIRED_NOTEBOOK_SECTIONS = [
    "## What this example demonstrates",
    "## Data and assumptions",
    "## Step-by-step workflow",
    "## Expected figures and interpretation",
    "## Debug tips",
    "## MATLAB Live Script alignment",
    "## Paper terminology and section references",
]


def _load_contract() -> dict[str, dict[str, object]]:
    data = json.loads(FIGURE_CONTRACT.read_text(encoding="utf-8"))
    topics = data.get("topics", {})
    if not isinstance(topics, dict) or not topics:
        raise RuntimeError(f"Invalid figure contract at {FIGURE_CONTRACT}")
    return topics


def _load_json(path: Path, default: dict[str, object]) -> dict[str, object]:
    if not path.exists():
        return default
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else default


def _example_topics() -> list[tuple[str, str]]:
    tree = ET.parse(TOC_PATH)
    root = tree.getroot()
    examples = None
    for item in root.iter("tocitem"):
        if item.attrib.get("id") == "nstat_examples":
            examples = item
            break
    if examples is None:
        raise RuntimeError("Could not find examples section in helptoc.xml")

    topics: list[tuple[str, str]] = []
    for item in examples.findall("tocitem"):
        title = " ".join((item.text or "").split()) or Path(item.attrib.get("target", "")).stem
        target = item.attrib.get("target", "")
        if target:
            topics.append((title, target))
    return topics


def _narrative_cells(
    title: str,
    stem: str,
    matlab_target: str,
    expected_figures: int,
    paper_payload: dict[str, object],
    mlx_payload: dict[str, object],
) -> list[dict]:
    figure_phrase = (
        "This notebook intentionally produces no figures; focus on object state and summary outputs."
        if expected_figures == 0
        else f"This notebook renders {expected_figures} figure(s) to illustrate the modeled behavior and analysis pipeline."
    )
    has_mlx_payload = bool(mlx_payload)
    mlx_title = str(mlx_payload.get("title", title)).strip() or title
    if has_mlx_payload:
        mlx_file = str(mlx_payload.get("file", f"helpfiles/{stem}.mlx")).strip() or f"helpfiles/{stem}.mlx"
    else:
        mlx_file = f"helpfiles/{stem}.m"
    mlx_intro = str(mlx_payload.get("intro", "")).strip()
    mlx_headings = mlx_payload.get("headings", [])
    if not isinstance(mlx_headings, list):
        mlx_headings = []
    mlx_headings = [str(h).strip() for h in mlx_headings if str(h).strip()][:6]

    paper_sections = paper_payload.get("paper_sections", [])
    if not isinstance(paper_sections, list):
        paper_sections = []
    paper_sections = [str(s).strip() for s in paper_sections if str(s).strip()]

    paper_terms = paper_payload.get("paper_terms", [])
    if not isinstance(paper_terms, list):
        paper_terms = []
    paper_terms = [str(t).strip() for t in paper_terms if str(t).strip()]

    paper_focus = str(paper_payload.get("narrative_focus", "")).strip()

    mlx_lines = [
        "## MATLAB Live Script alignment\n",
        f"MATLAB Live Script source: `{mlx_file}`\n",
        f"MLX title: **{mlx_title}**\n",
    ]
    if not has_mlx_payload:
        mlx_lines.append("No `.mlx` metadata was found for this topic in the synced MATLAB helpfiles; alignment is based on the MATLAB help script naming convention.\n")
    if mlx_intro:
        mlx_lines.append(f"MLX intent summary: {mlx_intro}\n")
    if mlx_headings:
        mlx_lines.append("Key MLX section headings:\n")
        for heading in mlx_headings:
            mlx_lines.append(f"- {heading}\n")

    paper_lines = [
        "## Paper terminology and section references\n",
        "Reference: **Cajigas et al. (2012), *Journal of Neuroscience Methods***\n",
        "Paper URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC3491120/\n",
    ]
    if paper_sections:
        paper_lines.append("Most relevant paper sections for this topic:\n")
        for section in paper_sections:
            paper_lines.append(f"- {section}\n")
    if paper_terms:
        paper_lines.append("nSTAT paper terms used in this notebook:\n")
        for term in paper_terms:
            paper_lines.append(f"- {term}\n")
    if paper_focus:
        paper_lines.append(f"Section-aligned interpretation: {paper_focus}\n")

    return [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {title}\n",
                "\n",
                "Executable Python notebook generated from source help-topic scripts.\n",
                f"MATLAB help target: `{matlab_target}`\n",
                f"Topic module: `examples.help_topics.{stem}`\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## What this example demonstrates\n",
                f"This example shows the standalone Python equivalent of `{stem}` using the paper's point-process nomenclature (for example CIF, PP-GLM, history dependence, and decoding/network terms when applicable).\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Data and assumptions\n",
                "The run uses bundled nSTAT datasets or deterministic synthetic fallbacks where needed, with fixed seeds for reproducibility.\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step-by-step workflow\n",
                "1. Resolve repository root and Python path.\n",
                f"2. Run `examples.help_topics.{stem}.run(...)` with figure rendering enabled.\n",
                "3. Print structured output for debugging and parity review.\n",
                "4. Display rendered figure files inline for interpretation and web publication.\n",
                "5. Assert figure-count contract and artifact existence.\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Expected figures and interpretation\n",
                f"{figure_phrase}\n",
                "Interpretation should focus on trend consistency, event timing, and qualitative agreement with MATLAB reference outputs rather than pixel-identical rendering.\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": mlx_lines,
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": paper_lines,
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Debug tips\n",
                "If execution fails, verify dataset availability, run with `MPLBACKEND=Agg` in headless mode, and inspect the JSON payload keys (`topic`, `figure_count`, `figures`) for mismatches.\n",
            ],
        },
    ]


def _build_notebook(title: str, stem: str, matlab_target: str, expected_figures: int) -> dict:
    code_setup = (
        "from pathlib import Path\n"
        "import json\n"
        "import sys\n\n"
        "def find_repo_root(start: Path) -> Path:\n"
        "    cur = start.resolve()\n"
        "    for p in [cur, *cur.parents]:\n"
        "        if (p / '.git').exists() and (p / 'nstat').exists() and (p / 'helpfiles').exists():\n"
        "            return p\n"
        "    raise RuntimeError('Could not find nSTAT repo root from notebook cwd')\n\n"
        "repo_root = find_repo_root(Path.cwd())\n"
        "if str(repo_root) not in sys.path:\n"
        "    sys.path.insert(0, str(repo_root))\n"
        "print('repo_root =', repo_root)\n"
    )

    code_run = (
        f"from examples.help_topics.{stem} import run\n"
        f"expected_figures = {expected_figures}\n"
        f"figure_dir = repo_root / 'reports' / 'figures' / 'notebooks' / '{stem}'\n"
        "out = run(repo_root=repo_root, figure_dir=figure_dir, render_figures=True)\n"
        "print(json.dumps(out, indent=2, default=str))\n"
    )

    code_display = (
        "from pathlib import Path\n"
        "from IPython.display import Image, display\n\n"
        "for fig_path in out.get('figures', []):\n"
        "    p = Path(fig_path)\n"
        "    print(p.name)\n"
        "    display(Image(filename=str(p)))\n"
    )

    code_check = (
        f"assert out.get('topic') == '{stem}'\n"
        "assert out.get('figure_contract_expected') == expected_figures\n"
        "assert out.get('figure_count') == expected_figures, out\n"
        "for fig_path in out.get('figures', []):\n"
        "    assert Path(fig_path).exists(), fig_path\n"
        "print('Notebook execution + figure contract: PASS')\n"
    )

    paper_meta = _load_json(PAPER_NOMENCLATURE, default={})
    topic_paper = {}
    if isinstance(paper_meta.get("topic_alignment"), dict):
        topic_paper = paper_meta["topic_alignment"].get(stem, {})  # type: ignore[index]
    if not isinstance(topic_paper, dict):
        topic_paper = {}

    mlx_meta = _load_json(MLX_METADATA, default={})
    topic_mlx = {}
    if isinstance(mlx_meta.get("topics"), dict):
        topic_mlx = mlx_meta["topics"].get(stem, {})  # type: ignore[index]
    if not isinstance(topic_mlx, dict):
        topic_mlx = {}

    cells = _narrative_cells(
        title=title,
        stem=stem,
        matlab_target=matlab_target,
        expected_figures=expected_figures,
        paper_payload=topic_paper,
        mlx_payload=topic_mlx,
    )
    cells.extend(
        [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_setup.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_run.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_display.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_check.splitlines(keepends=True),
            },
        ]
    )
    for idx, cell in enumerate(cells, start=1):
        cell.setdefault("id", f"{stem.lower()}-{idx:02d}")

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    NB_ROOT.mkdir(parents=True, exist_ok=True)
    topics = _example_topics()
    contract = _load_contract()

    generated = 0
    missing_sources: list[str] = []
    missing_contract: list[str] = []

    for title, target in topics:
        stem = Path(target).stem
        source_mod = SRC_ROOT / f"{stem}.py"
        if not source_mod.exists():
            missing_sources.append(stem)
            continue

        if stem not in contract:
            missing_contract.append(stem)
            continue

        info = contract[stem]
        nb = _build_notebook(
            title=title,
            stem=stem,
            matlab_target=str(info.get("matlab_target", target)),
            expected_figures=int(info.get("expected_figures", 0)),
        )
        out = NB_ROOT / f"{stem}.ipynb"
        out.write_text(json.dumps(nb, indent=2), encoding="utf-8")
        generated += 1

    report = {
        "total_topics": len(topics),
        "contract_topics": len(contract),
        "generated": generated,
        "required_sections": REQUIRED_NOTEBOOK_SECTIONS,
        "missing_sources": missing_sources,
        "missing_contract": missing_contract,
        "output_dir": str(NB_ROOT.relative_to(REPO_ROOT)),
    }
    print(json.dumps(report, indent=2))

    if missing_sources or missing_contract:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
