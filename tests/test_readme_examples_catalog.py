from __future__ import annotations

import re
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"
PAPER_MANIFEST_PATH = REPO_ROOT / "examples" / "paper" / "manifest.yml"

SUPPLEMENTARY_EXAMPLES = [
    (
        "python examples/readme_examples/example1_multitaper_and_spectrogram.py",
        "[PNG](examples/readme_examples/images/readme_example1_multitaper_and_spectrogram.png)",
    ),
    (
        "python examples/readme_examples/example2_simulate_cif_spiketrain_10s.py",
        "[PNG](examples/readme_examples/images/readme_example2_simulate_cif_spiketrain_10s.png)",
    ),
    (
        "python examples/readme_examples/example3_nstcoll_raster_from_example2.py",
        "[PNG](examples/readme_examples/images/readme_example3_nstcoll_raster.png)",
    ),
]

EXPECTED_CANONICAL_QUESTIONS = {
    "example01_mepsc_poisson": "Do mEPSCs follow constant vs piecewise Poisson firing under Mg2+ washout?",
    "example02_whisker_stimulus_thalamus": "How do explicit whisker stimulus and spike history improve thalamic GLM fits?",
    "example03_psth_and_ssglm": "How do PSTH and SSGLM capture within-trial and across-trial dynamics?",
    "example04_place_cells_continuous_stimulus": "Which receptive-field basis (Gaussian vs Zernike) better fits place cells?",
    "example05_decoding_ppaf_pphf": "How well do adaptive/hybrid point-process filters decode stimulus and reach state?",
}


def _extract_examples_block(text: str) -> str:
    match = re.search(r"## Examples\n(.*?)\n## Documentation\n", text, flags=re.S)
    if not match:
        raise AssertionError("README is missing an Examples block bounded by '## Examples' and '## Documentation'.")
    return match.group(1)


def test_readme_examples_headings_match_gallery_layout() -> None:
    block = _extract_examples_block(README_PATH.read_text(encoding="utf-8"))
    headings = re.findall(r"^###\s+.+$", block, flags=re.M)
    assert headings == [
        "### Paper Examples (Self-Contained)",
        "### Supplementary Examples",
    ]
    assert "python tools/paper_examples/build_gallery.py" in block


def test_readme_paper_example_rows_track_manifest() -> None:
    block = _extract_examples_block(README_PATH.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(PAPER_MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    entries = manifest.get("examples", [])
    assert entries, "Paper example manifest has no entries."

    for index, row in enumerate(entries, start=1):
        script = str(row["script"])
        question = str(row["question"])
        example_label = f"Example {index:02d}"
        thumbnail = str(row["figure_files"][0])
        assert example_label in block
        assert question in block
        assert f"![{example_label}]({thumbnail})" in block
        assert f"`python {script}`" in block
        assert f"[Script]({script})" in block
        assert f"[Figures](docs/figures/{row['example_id']}/)" in block


def test_paper_example_manifest_questions_match_matlab_gallery_wording() -> None:
    manifest = yaml.safe_load(PAPER_MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    entries = manifest.get("examples", [])
    questions = {str(row["name"]): str(row["question"]) for row in entries}
    assert questions == EXPECTED_CANONICAL_QUESTIONS


def test_readme_supplementary_examples_are_preserved() -> None:
    block = _extract_examples_block(README_PATH.read_text(encoding="utf-8"))

    positions: list[int] = []
    for command, png_link in SUPPLEMENTARY_EXAMPLES:
        pos = block.find(command)
        assert pos >= 0, f"Missing supplementary example command: {command}"
        positions.append(pos)
        assert png_link in block
    assert positions == sorted(positions), "Supplementary examples must remain in README order."
