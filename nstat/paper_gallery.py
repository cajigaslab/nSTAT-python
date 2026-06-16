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
        "## Paper examples",
        "",
        "The five canonical examples from Cajigas, Malik & Brown (2012), each",
        "reproduced as a self-contained Python script with a generated figure gallery.",
        "",
        "Regenerate the gallery metadata after editing any paper-example script:",
        "",
        "```bash",
        "python tools/paper_examples/build_gallery.py",
        "```",
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
            "[docs/paper_examples.md](docs/paper_examples.md).",
            "",
            "The figshare paper dataset is distributed separately from the Git repository:",
            "[DOI 10.6084/m9.figshare.4834640.v3](https://doi.org/10.6084/m9.figshare.4834640.v3)",
            "(`nstat-install --download-example-data always` fetches it; `NSTAT_OFFLINE=1`",
            "forces offline mode).",
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


def render_paper_examples_html(repo_root: Path | None = None) -> str:
    """Self-contained HTML showcase of the paper-example gallery.

    Mirrors the style of ``docs/extras_summary.html``: embedded CSS, no
    Sphinx wrap, suitable for ``html_extra_path``.  Each example gets a
    section with its title, the question, run command, and inline
    ``<img>`` references to every figure in the gallery (paths are
    relative to ``docs/`` so the page works both locally and on
    GitHub Pages).
    """
    manifest = build_gallery_manifest(repo_root)
    examples = manifest["examples"]

    css = (
        ":root{--bg:#0f1419;--card:#1a1f29;--ink:#e6e6e6;--ink-2:#9aa4b1;"
        "--accent:#7ee787;--accent-2:#58a6ff;--border:#30363d;}"
        "*{box-sizing:border-box}"
        "body{margin:0;background:var(--bg);color:var(--ink);"
        "font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;}"
        ".wrap{max-width:1100px;margin:0 auto;padding:2rem 1.25rem 3rem;}"
        "header{border-bottom:1px solid var(--border);padding-bottom:1.25rem;margin-bottom:1.5rem;}"
        "h1{margin:0 0 .35rem;font-size:1.6rem;color:var(--accent);}"
        "h2{margin:2rem 0 .75rem;font-size:1.2rem;color:var(--accent-2);"
        "border-top:1px solid var(--border);padding-top:1.25rem;}"
        "h3{margin:1.25rem 0 .35rem;font-size:1rem;color:var(--ink);}"
        "p{margin:.4rem 0;color:var(--ink);}"
        ".muted{color:var(--ink-2);font-size:.9rem;}"
        "code,pre{font-family:'SF Mono',Menlo,Consolas,monospace;font-size:.85rem;"
        "background:#0b0e14;color:var(--accent);padding:.1rem .3rem;border-radius:3px;}"
        "pre{padding:.75rem 1rem;overflow-x:auto;border:1px solid var(--border);}"
        "nav.toc{background:var(--card);border:1px solid var(--border);"
        "padding:.75rem 1rem;border-radius:6px;margin-bottom:1.5rem;font-size:.9rem;}"
        "nav.toc a{color:var(--accent-2);text-decoration:none;margin-right:.9rem;}"
        "nav.toc a:hover{text-decoration:underline;}"
        ".example{background:var(--card);border:1px solid var(--border);"
        "border-radius:8px;padding:1.25rem 1.5rem;margin:1.25rem 0;}"
        ".gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));"
        "gap:.85rem;margin-top:.75rem;}"
        ".gallery figure{margin:0;background:#0b0e14;border:1px solid var(--border);"
        "border-radius:4px;padding:.5rem;}"
        ".gallery img{width:100%;height:auto;display:block;border-radius:2px;}"
        ".gallery figcaption{font-size:.78rem;color:var(--ink-2);margin-top:.4rem;"
        "word-break:break-word;}"
        "a{color:var(--accent-2);}"
        "footer{margin-top:2rem;padding-top:1rem;border-top:1px solid var(--border);"
        "color:var(--ink-2);font-size:.85rem;}"
    )

    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width,initial-scale=1">')
    parts.append("<title>nSTAT Paper Examples — Output Gallery</title>")
    parts.append(f"<style>{css}</style>")
    parts.append("</head><body><div class=\"wrap\">")
    parts.append("<header>")
    parts.append("<h1>Paper Examples — Output Gallery</h1>")
    parts.append(
        "<p class=\"muted\">Every <code>examples/paper/*.py</code> script registered "
        "in <code>examples/paper/manifest.yml</code>, with the figure outputs each "
        "produces.  Auto-generated from the manifest — re-render with "
        "<code>python tools/paper_examples/build_gallery.py</code>.</p>"
    )
    parts.append("</header>")

    parts.append('<nav class="toc"><strong>Examples:</strong> ')
    nav_links = " · ".join(
        f'<a href="#{row["example_id"]}">{row["example_id"].replace("example", "Example ")}</a>'
        for row in examples
    )
    parts.append(nav_links + "</nav>")

    for row in examples:
        example_label = row["example_id"].replace("example", "Example ")
        parts.append(f'<section class="example" id="{row["example_id"]}">')
        parts.append(f"<h2>{example_label}: {row['title']}</h2>")
        parts.append(f"<p><strong>Question:</strong> {row['question']}</p>")
        parts.append(f"<p class=\"muted\">{row['description']}</p>")
        parts.append(
            f'<p><strong>Run:</strong> <code>{row["run_command"]}</code> · '
            f'<a href="https://github.com/cajigaslab/nSTAT-python/blob/main/'
            f'{row["source_script"]}">Script</a> · '
            f'<a href="figures/{row["example_id"]}/">Figures directory</a></p>'
        )
        parts.append('<div class="gallery">')
        for fig_path in row["figure_files"]:
            # fig_path is "docs/figures/exampleNN/figXX_name.png" — strip "docs/"
            # so the src is relative to the HTML file's location (docs/).
            rel = fig_path.replace("docs/", "", 1)
            filename = Path(fig_path).name
            parts.append(
                f'<figure><img src="{rel}" alt="{filename}" loading="lazy">'
                f'<figcaption>{filename}</figcaption></figure>'
            )
        parts.append("</div>")
        parts.append("</section>")

    parts.append("<footer>")
    parts.append(
        '<p>Source: <a href="https://github.com/cajigaslab/nSTAT-python">'
        "cajigaslab/nSTAT-python</a> · License: GPL-2.0 · "
        'Paper: <a href="https://pubmed.ncbi.nlm.nih.gov/22981419/">'
        "Cajigas, Malik &amp; Brown 2012, J Neurosci Methods</a></p>"
    )
    parts.append("</footer>")
    parts.append("</div></body></html>")
    return "\n".join(parts) + "\n"


def write_gallery_outputs(repo_root: Path | None = None) -> tuple[Path, Path, Path, Path]:
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

    html_path = docs_dir / "paper_examples_gallery.html"
    html_path.write_text(render_paper_examples_html(base), encoding="utf-8")

    readme_path = base / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8")
    start = readme_text.index("## Paper examples")
    end = readme_text.index("Plot-style policy")
    gallery_text = render_readme_examples_markdown(base)
    readme_path.write_text(
        readme_text[:start] + gallery_text + "\n" + readme_text[end:],
        encoding="utf-8",
    )
    return manifest_path, markdown_path, html_path, readme_path


__all__ = [
    "build_gallery_manifest",
    "ensure_gallery_dirs",
    "load_paper_example_manifest",
    "render_paper_examples_html",
    "render_paper_examples_markdown",
    "render_readme_examples_markdown",
    "write_gallery_outputs",
]
