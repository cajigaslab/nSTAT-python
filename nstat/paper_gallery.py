from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_paper_example_manifest(repo_root: Path | None = None) -> dict[str, Any]:
    base = _repo_root() if repo_root is None else repo_root.resolve()
    path = base / "examples" / "paper" / "manifest.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_gallery_manifest(repo_root: Path | None = None) -> dict[str, Any]:
    base = _repo_root() if repo_root is None else repo_root.resolve()
    payload = load_paper_example_manifest(base)
    examples: list[dict[str, Any]] = []
    for row in payload["examples"]:
        figure_dir = f"docs/figures/{row['example_id']}"
        examples.append(
            {
                "example_id": row["example_id"],
                "title": row["title"],
                "source_script": row["script"],
                "matlab_source": row["matlab_source"],
                "description": row["description"],
                "question": row["question"],
                "run_command": f"python {row['script']}",
                "figure_dir": figure_dir,
                "figure_files": list(row["figure_files"]),
                "thumbnail_file": row["figure_files"][0],
                "sections": list(row["sections"]),
            }
        )
    return {
        "figure_root": "docs/figures",
        "examples": examples,
    }


def render_paper_examples_markdown(repo_root: Path | None = None) -> str:
    manifest = build_gallery_manifest(repo_root)
    lines = [
        "# nSTAT Python Paper Examples",
        "",
        "This page mirrors the MATLAB paper-example index for the standalone Python port.",
        "",
        "Canonical source files:",
        "- `examples/paper/*.py`",
        "- `nstat/paper_examples_full.py`",
        "",
        "## Run Everything",
        "",
        "```bash",
        "python tools/paper_examples/build_gallery.py",
        "```",
        "",
        "Outputs:",
        "- Figure metadata: `docs/figures/manifest.json`",
        "- Gallery page: `docs/paper_examples.md`",
        "- Figures: `docs/figures/example01/` ... `docs/figures/example05/`",
        "",
        "## Example Index",
        "",
        "| ID | Thumbnail | Standalone source | Question | Run command | Figure gallery |",
        "|---|---|---|---|---|---|",
    ]
    for row in manifest["examples"]:
        lines.append(
            f"| `{row['example_id']}` | ![{row['example_id'].replace('example', 'Example ')}]({row['thumbnail_file'].replace('docs/', '')}) | "
            f"[{Path(row['source_script']).name}](../{row['source_script']}) | {row['question']} | "
            f"`{row['run_command']}` | [gallery page](./figures/{row['example_id']}/README.md) |"
        )

    lines.extend(
        [
            "",
            "```{toctree}",
            ":hidden:",
            "",
        ]
    )
    for row in manifest["examples"]:
        lines.append(f"figures/{row['example_id']}/README")
    lines.extend(["```", "", "## Gallery", ""])
    for row in manifest["examples"]:
        lines.extend(
            [
                f"### {row['example_id'].replace('example', 'Example ')}: {row['title']}",
                "",
                f"Question: {row['question']}",
                "",
                f"Run command: `{row['run_command']}`",
                "",
                f"![{row['example_id'].replace('example', 'Example ')}]({row['thumbnail_file'].replace('docs/', '')})",
                "",
                "Expected figure files:",
            ]
        )
        for fig in row["figure_files"]:
            lines.append(f"- `{fig}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_readme_examples_markdown(repo_root: Path | None = None) -> str:
    manifest = build_gallery_manifest(repo_root)
    lines = [
        "## Examples",
        "",
        "### Paper Examples (Self-Contained)",
        "",
        "Canonical source files:",
        "- `examples/paper/*.py`",
        "- `nstat/paper_examples_full.py`",
        "",
        "Single command to regenerate the paper-example gallery metadata:",
        "",
        "```bash",
        "python tools/paper_examples/build_gallery.py",
        "```",
        "",
        "This writes `docs/paper_examples.md`, `docs/figures/manifest.json`, and",
        "refreshes the canonical README paper-example table from",
        "`examples/paper/manifest.yml`.",
        "",
        "| Example | Thumbnail | What question it answers | Run command | Links |",
        "|---|---|---|---|---|",
    ]
    for index, row in enumerate(manifest["examples"], start=1):
        label = f"Example {index:02d}"
        lines.append(
            f"| {label} | ![{label}]({row['thumbnail_file']}) | {row['question']} | "
            f"`{row['run_command']}` | [Script]({row['source_script']}) · [Figures]({row['figure_dir']}/) |"
        )

    lines.extend(
        [
            "",
            "Expanded paper-example index and figure gallery:",
            "- [docs/paper_examples.md](docs/paper_examples.md)",
            "",
            "### Supplementary Examples",
            "",
            "These smaller demos remain useful as quick install and plotting checks.",
            "",
            "| Example | Run command | Output |",
            "|---|---|---|",
            "| Multitaper spectrum + spectrogram | `python examples/readme_examples/example1_multitaper_and_spectrogram.py` | [PNG](examples/readme_examples/images/readme_example1_multitaper_and_spectrogram.png) |",
            "| Simulated CIF spike train | `python examples/readme_examples/example2_simulate_cif_spiketrain_10s.py` | [PNG](examples/readme_examples/images/readme_example2_simulate_cif_spiketrain_10s.png) |",
            "| Spike-train raster | `python examples/readme_examples/example3_nstcoll_raster_from_example2.py` | [PNG](examples/readme_examples/images/readme_example3_nstcoll_raster.png) |",
            "",
        ]
    )
    return "\n".join(lines)


def ensure_gallery_dirs(repo_root: Path | None = None) -> list[Path]:
    base = _repo_root() if repo_root is None else repo_root.resolve()
    manifest = load_paper_example_manifest(base)
    created: list[Path] = []
    for row in manifest["examples"]:
        directory = base / "docs" / "figures" / row["example_id"]
        directory.mkdir(parents=True, exist_ok=True)
        readme = directory / "README.md"
        if not readme.exists():
            readme.write_text(
                f"# {row['example_id']}\n\nGenerated figure outputs for `{row['name']}` are written here.\n",
                encoding="utf-8",
            )
        created.append(directory)
    return created


def write_gallery_outputs(repo_root: Path | None = None) -> tuple[Path, Path, Path]:
    base = _repo_root() if repo_root is None else repo_root.resolve()
    ensure_gallery_dirs(base)
    docs_dir = base / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = docs_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = figures_dir / "manifest.json"
    manifest_path.write_text(json.dumps(build_gallery_manifest(base), indent=2), encoding="utf-8")

    markdown_path = docs_dir / "paper_examples.md"
    markdown_path.write_text(render_paper_examples_markdown(base), encoding="utf-8")

    readme_path = base / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8")
    start = readme_text.index("## Examples")
    end = readme_text.index("## Documentation")
    readme_path.write_text(
        readme_text[:start] + render_readme_examples_markdown(base) + readme_text[end:],
        encoding="utf-8",
    )
    return manifest_path, markdown_path, readme_path


__all__ = [
    "build_gallery_manifest",
    "ensure_gallery_dirs",
    "load_paper_example_manifest",
    "render_paper_examples_markdown",
    "render_readme_examples_markdown",
    "write_gallery_outputs",
]
