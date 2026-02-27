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


def _load_contract() -> dict[str, dict[str, object]]:
    data = json.loads(FIGURE_CONTRACT.read_text(encoding="utf-8"))
    topics = data.get("topics", {})
    if not isinstance(topics, dict) or not topics:
        raise RuntimeError(f"Invalid figure contract at {FIGURE_CONTRACT}")
    return topics


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
        "import matplotlib.pyplot as plt\n"
        "import matplotlib.image as mpimg\n\n"
        "for fig_path in out.get('figures', []):\n"
        "    p = Path(fig_path)\n"
        "    img = mpimg.imread(p)\n"
        "    h, w = img.shape[:2]\n"
        "    fig = plt.figure(figsize=(max(w / 160.0, 2.0), max(h / 160.0, 2.0)), dpi=160)\n"
        "    ax = fig.add_subplot(111)\n"
        "    ax.imshow(img, cmap='gray' if img.ndim == 2 else None)\n"
        "    ax.axis('off')\n"
        "    ax.set_title(p.name)\n"
        "    plt.show()\n"
        "    plt.close(fig)\n"
    )

    code_check = (
        f"assert out.get('topic') == '{stem}'\n"
        "assert out.get('figure_contract_expected') == expected_figures\n"
        "assert out.get('figure_count') == expected_figures, out\n"
        "for fig_path in out.get('figures', []):\n"
        "    assert Path(fig_path).exists(), fig_path\n"
        "print('Notebook execution + figure contract: PASS')\n"
    )

    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\\n",
                    "\\n",
                    "Executable Python notebook generated from source help-topic scripts.\\n",
                    f"MATLAB help target: `{matlab_target}`\\n",
                    f"Expected figure artifacts: `{expected_figures}`\\n",
                ],
            },
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
        ],
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
