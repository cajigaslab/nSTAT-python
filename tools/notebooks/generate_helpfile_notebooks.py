#!/usr/bin/env python3
"""Generate helpfile-aligned notebooks with strict MATLAB section-to-cell mapping.

Rules:
- Split each MATLAB helpfile on section markers (%%), treating pre-%% content as section 1.
- Emit exactly one Jupyter *code* cell per MATLAB section.
- Preserve section order.
- Keep narrative as Python comments inside each section code cell.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nbformat as nbf
import yaml


@dataclass(frozen=True)
class Section:
    title: str
    lines: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tools/notebooks/notebook_manifest.yml"),
        help="Notebook topic/run-group manifest.",
    )
    parser.add_argument(
        "--helpfile-map",
        type=Path,
        default=Path("parity/notebook_to_helpfile_map.yml"),
        help="Topic to MATLAB helpfile mapping.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root.",
    )
    parser.add_argument(
        "--matlab-help-root",
        type=Path,
        default=None,
        help="Optional explicit MATLAB helpfiles root.",
    )
    parser.add_argument(
        "--out-helpfile-manifest",
        type=Path,
        default=Path("parity/helpfile_notebook_manifest.yml"),
        help="Output manifest including section/cell/figure counts.",
    )
    parser.add_argument(
        "--rewrite-notebook-manifest",
        action="store_true",
        help="Rewrite tools/notebooks/notebook_manifest.yml notebook paths to notebooks/helpfiles/*.ipynb.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Run notebook cleaner after generation for deterministic formatting.",
    )
    return parser.parse_args()


def _load_generate_notebooks_module(repo_root: Path) -> Any:
    module_path = repo_root / "tools" / "notebooks" / "generate_notebooks.py"
    spec = importlib.util.spec_from_file_location("nstat_generate_notebooks", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected mapping YAML at {path}")
    return payload


def resolve_matlab_help_root(repo_root: Path, provided: Path | None) -> Path:
    candidates: list[Path] = []
    if provided is not None:
        candidates.append(provided)

    env_help = os.environ.get("NSTAT_MATLAB_HELP_ROOT")
    if env_help:
        candidates.append(Path(env_help))

    candidates.extend(
        [
            Path("/tmp/upstream-nstat/helpfiles"),
            repo_root / ".." / "nSTAT_currentRelease_Local" / "helpfiles",
            Path.home()
            / "Library"
            / "CloudStorage"
            / "Dropbox"
            / "Research"
            / "Matlab"
            / "nSTAT_currentRelease_Local"
            / "helpfiles",
        ]
    )

    for cand in candidates:
        resolved = cand.expanduser().resolve()
        if resolved.is_dir():
            return resolved
    checked = "\n".join(f"- {str(p.expanduser())}" for p in candidates)
    raise RuntimeError(f"Could not resolve MATLAB help root. Checked:\n{checked}")


def split_helpfile_sections(helpfile_text: str) -> list[Section]:
    lines = helpfile_text.splitlines()
    if not lines:
        return [Section(title="(empty helpfile)", lines=[])]

    sections: list[Section] = []
    current_lines: list[str] = []
    current_title = "Preamble"

    for line in lines:
        if re.match(r"^\s*%%", line):
            if current_lines:
                sections.append(Section(title=current_title, lines=current_lines))
            marker = re.sub(r"^\s*%%\s*", "", line).strip()
            current_title = marker if marker else "Section"
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(Section(title=current_title, lines=current_lines))

    if not sections:
        sections.append(Section(title="Preamble", lines=lines))
    return sections


def matlab_lines_to_python_comments(lines: list[str]) -> str:
    out: list[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            out.append("#")
            continue
        stripped = line.lstrip()
        if stripped.startswith("%"):
            text = stripped[1:].lstrip()
            out.append(f"# {text}" if text else "#")
        else:
            out.append(f"# MATLAB: {line.rstrip()}")
    return "\n".join(out)


def _build_cell_source(
    *,
    topic: str,
    section: Section,
    section_index: int,
    section_count: int,
    header_code: str,
    setup_code: str,
    execution_blob: str,
) -> str:
    parts: list[str] = []
    parts.append(f"# MATLAB section {section_index}/{section_count} for {topic}: {section.title}")
    parts.append(matlab_lines_to_python_comments(section.lines))

    if section_count == 1:
        parts.append("# Python translation bootstrap + execution for single-section helpfile.")
        parts.append(header_code)
        parts.append(setup_code)
        parts.append(execution_blob)
        return "\n\n".join(parts)

    if section_index == 1:
        parts.append("# Python translation bootstrap for this helpfile.")
        parts.append(header_code)
        parts.append(setup_code)
    elif section_index == section_count:
        parts.append("# Python translation execution block for this helpfile.")
        parts.append(execution_blob)
    else:
        parts.append("# Python translation note: deterministic execution is consolidated in final section cell.")
        parts.append(f"section_index = {section_index}")
        parts.append("_ = section_index")

    return "\n\n".join(parts)


def _notebook_metadata(topic: str, run_group: str, family: str, section_count: int, matlab_helpfile: str) -> dict[str, Any]:
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
        },
        "nstat": {
            "topic": topic,
            "run_group": run_group,
            "family": family,
            "source_helpfile": matlab_helpfile,
            "section_count": section_count,
        },
    }


def _line_port_snapshot(module: Any, topic: str, repo_root: Path) -> str:
    snapshot = module.line_port_snapshot_cell(topic, repo_root)
    return snapshot.strip() if isinstance(snapshot, str) else ""


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    module = _load_generate_notebooks_module(repo_root)

    base_manifest = _read_yaml(args.manifest)
    base_rows = [dict(row) for row in base_manifest.get("notebooks", [])]
    if not base_rows:
        raise RuntimeError(f"No notebook rows found in {args.manifest}")

    run_group_by_topic = {str(row["topic"]): str(row["run_group"]) for row in base_rows}

    map_payload = _read_yaml(args.helpfile_map)
    map_rows = [dict(row) for row in map_payload.get("mappings", [])]
    if not map_rows:
        raise RuntimeError(f"No mappings found in {args.helpfile_map}")

    matlab_help_root = resolve_matlab_help_root(repo_root, args.matlab_help_root)

    help_manifest_rows: list[dict[str, Any]] = []
    rewritten_rows: list[dict[str, Any]] = []

    for row in map_rows:
        topic = str(row["topic"])
        matlab_helpfile = str(row["matlab_helpfile"])
        run_group = run_group_by_topic.get(topic, "full")
        family = module.classify_topic(topic)

        helpfile_path = matlab_help_root / matlab_helpfile
        if not helpfile_path.exists():
            raise RuntimeError(f"Missing MATLAB helpfile for topic {topic}: {helpfile_path}")

        help_text = helpfile_path.read_text(encoding="utf-8", errors="ignore")
        sections = split_helpfile_sections(help_text)

        output_rel = Path("notebooks") / "helpfiles" / f"{topic}.ipynb"
        output_path = repo_root / output_rel
        output_path.parent.mkdir(parents=True, exist_ok=True)

        header_code = module.code_header_cell(topic, run_group, family)
        setup_code = module.code_cell_setup(topic, family)
        template_code = module.template_for_topic(topic, family)
        execution_parts = [template_code, module.ASSERTION_CELL]
        execution_blob = "\n\n".join(part for part in execution_parts if part and part.strip())

        notebook = nbf.v4.new_notebook()
        notebook.metadata.update(_notebook_metadata(topic, run_group, family, len(sections), matlab_helpfile))

        cells = []
        for idx, section in enumerate(sections, start=1):
            cell_source = _build_cell_source(
                topic=topic,
                section=section,
                section_index=idx,
                section_count=len(sections),
                header_code=header_code,
                setup_code=setup_code,
                execution_blob=execution_blob,
            )
            cell = nbf.v4.new_code_cell(cell_source.rstrip() + "\n")
            cells.append(cell)
        notebook.cells = cells
        nbf.write(notebook, output_path)

        help_manifest_rows.append(
            {
                "topic": topic,
                "file": str(output_rel.as_posix()),
                "run_group": run_group,
                "matlab_helpfile": matlab_helpfile,
                "section_count": int(len(sections)),
                "cell_count": int(len(cells)),
                "expected_figure_count": 1,
            }
        )
        rewritten_rows.append(
            {
                "topic": topic,
                "file": str(output_rel.as_posix()),
                "run_group": run_group,
            }
        )

    out_help_payload = {"version": 1, "notebooks": help_manifest_rows}
    args.out_helpfile_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_helpfile_manifest.write_text(yaml.safe_dump(out_help_payload, sort_keys=False), encoding="utf-8")

    if args.rewrite_notebook_manifest:
        out_manifest_payload = {"version": 1, "notebooks": rewritten_rows}
        args.manifest.write_text(yaml.safe_dump(out_manifest_payload, sort_keys=False), encoding="utf-8")

    if args.normalize:
        cleaner = repo_root / "tools" / "notebooks" / "clean_notebooks.py"
        subprocess.run(
            [
                sys.executable,
                str(cleaner),
                "--manifest",
                str(args.manifest),
                "--repo-root",
                str(repo_root),
            ],
            check=True,
            cwd=repo_root,
        )

    print(f"Generated {len(help_manifest_rows)} helpfile notebooks under notebooks/helpfiles")
    print(f"MATLAB help root: {matlab_help_root}")
    print(f"Wrote helpfile manifest: {args.out_helpfile_manifest}")
    if args.rewrite_notebook_manifest:
        print(f"Rewrote notebook manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
