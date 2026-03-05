from __future__ import annotations

import re
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"
CATALOG_PATH = REPO_ROOT / "examples" / "nSTATPaperExamples" / "manifest.yml"


FEATURED_HEADINGS = [
    "### Example 1 — Single sinusoid: signal + multitaper spectrum + spectrogram",
    "### Example 2 — Time-varying CIF over 10 seconds (single-frequency sinusoid)",
    "### Example 3 — Spike train collection raster from Example 2",
]

FEATURED_RUN_COMMANDS = [
    "python examples/readme_examples/example1_multitaper_and_spectrogram.py",
    "python examples/readme_examples/example2_simulate_cif_spiketrain_10s.py",
    "python examples/readme_examples/example3_nstcoll_raster_from_example2.py",
]


def _extract_examples_block(text: str) -> str:
    match = re.search(r"## Examples\n(.*?)\n## Documentation\n", text, flags=re.S)
    if not match:
        raise AssertionError("README is missing an Examples block bounded by '## Examples' and '## Documentation'.")
    return match.group(1)


def test_readme_featured_examples_are_preserved_in_order() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    block = _extract_examples_block(readme)

    heading_positions = []
    for heading in FEATURED_HEADINGS:
        pos = block.find(heading)
        assert pos >= 0, f"Missing featured heading: {heading}"
        heading_positions.append(pos)
    assert heading_positions == sorted(heading_positions), "Featured examples must remain in the original order."

    for cmd in FEATURED_RUN_COMMANDS:
        assert cmd in block, f"Missing featured run command: {cmd}"


def test_readme_includes_complete_nstatpaperexamples_catalog_once() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    block = _extract_examples_block(readme)
    assert "### nSTATPaperExamples" in block, "README Examples section is missing the nSTATPaperExamples catalog header."

    manifest = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8")) or {}
    entries = manifest.get("examples", [])
    assert entries, "nSTATPaperExamples manifest has no entries."

    for row in entries:
        name = str(row["name"])
        rel_path = str(row["relative_path"])
        link = f"[{name}]({rel_path})"
        count = block.count(link)
        assert count == 1, f"Catalog entry must appear exactly once in README: {link} (found {count})."


def test_readme_examples_section_has_no_other_example_groups() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    block = _extract_examples_block(readme)

    headings = re.findall(r"^###\s+.+$", block, flags=re.M)
    expected = FEATURED_HEADINGS + ["### nSTATPaperExamples"]
    assert headings == expected, (
        "README Examples section must contain only the three featured examples "
        "followed by the nSTATPaperExamples catalog header."
    )
