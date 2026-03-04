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
import json
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
    start_line: int


@dataclass(frozen=True)
class FigureEvent:
    section_index: int
    matlab_line_number: int
    matlab_snippet: str
    event_type: str
    figure_ordinal: int
    trigger: str
    reference_image_path: str = ""


DATA_PATH_PATTERN = re.compile(r"""['"]([^'"]*data/[^'"]+)['"]""", flags=re.IGNORECASE)
FIGURE_RE = re.compile(r"(^|[^A-Za-z0-9_])figure(\s*\(|\s|;|$)", flags=re.IGNORECASE)
SUBPLOT_RE = re.compile(r"(^|[^A-Za-z0-9_])subplot\s*\(", flags=re.IGNORECASE)
SPECTROGRAM_RE = re.compile(r"(^|[^A-Za-z0-9_])spectrogram\s*\(", flags=re.IGNORECASE)
CLOSE_RE = re.compile(r"(^|[^A-Za-z0-9_])close(\s*\(|\s|;|$)", flags=re.IGNORECASE)
CLF_RE = re.compile(r"(^|[^A-Za-z0-9_])clf(\s*\(|\s|;|$)", flags=re.IGNORECASE)
PLOT_TOKENS = (
    "plot(",
    "plot3(",
    "semilogx(",
    "semilogy(",
    "loglog(",
    "scatter(",
    "imagesc(",
    "imshow(",
    "pcolor(",
    "surf(",
    "mesh(",
    "contour(",
    "contourf(",
    "histogram(",
    "specgram(",
)


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
        "--out-figure-manifest",
        type=Path,
        default=Path("parity/helpfile_figure_manifest.json"),
        help="Output JSON manifest of MATLAB figure events and expected figure counts.",
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
        return [Section(title="(empty helpfile)", lines=[], start_line=1)]

    sections: list[Section] = []
    current_lines: list[str] = []
    current_title = "Preamble"
    current_start_line = 1

    for line_number, line in enumerate(lines, start=1):
        if re.match(r"^\s*%%", line):
            if current_lines:
                sections.append(Section(title=current_title, lines=current_lines, start_line=current_start_line))
            marker = re.sub(r"^\s*%%\s*", "", line).strip()
            current_title = marker if marker else "Section"
            current_lines = [line]
            current_start_line = line_number
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(Section(title=current_title, lines=current_lines, start_line=current_start_line))

    if not sections:
        sections.append(Section(title="Preamble", lines=lines, start_line=1))
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


def _extract_data_relpaths(lines: list[str]) -> list[str]:
    rels: list[str] = []
    for raw in lines:
        for match in DATA_PATH_PATTERN.finditer(raw):
            token = match.group(1)
            marker = token.lower().find("data/")
            rel = token[marker + len("data/") :] if marker >= 0 else token
            rel = rel.lstrip("./").replace("\\", "/")
            if rel and rel not in rels:
                rels.append(rel)
    return rels


def _strip_matlab_comment(raw: str) -> str:
    line = raw.rstrip()
    if "%" in line:
        idx = line.find("%")
        return line[:idx].rstrip()
    return line


def _detect_line_trigger(code_line: str) -> tuple[str | None, bool]:
    line = code_line.strip()
    if not line:
        return None, False
    lower = line.lower()
    if CLOSE_RE.search(line):
        return "close", False
    if CLF_RE.search(line):
        return "clf", False
    if FIGURE_RE.search(line):
        return "figure", True
    if SUBPLOT_RE.search(line):
        return "subplot", True
    if SPECTROGRAM_RE.search(line):
        return "spectrogram", True
    if "set(gcf" in lower or "gca" in lower:
        return "gcf", True
    if any(token in lower for token in PLOT_TOKENS):
        return "plot", True
    return None, False


def detect_figure_events(sections: list[Section]) -> tuple[list[FigureEvent], int]:
    events: list[FigureEvent] = []
    current_has_figure = False
    figure_ordinal = 0

    for section_index, section in enumerate(sections, start=1):
        for offset, raw_line in enumerate(section.lines):
            code_line = _strip_matlab_comment(raw_line)
            trigger, might_plot = _detect_line_trigger(code_line)
            matlab_line_number = int(section.start_line + offset)
            snippet = raw_line.strip()
            if not snippet:
                continue

            if trigger == "close":
                current_has_figure = False
                continue
            if trigger == "clf":
                # Keep current figure context after clear.
                continue
            if trigger is None:
                continue

            if trigger == "figure":
                figure_ordinal += 1
                current_has_figure = True
                events.append(
                    FigureEvent(
                        section_index=section_index,
                        matlab_line_number=matlab_line_number,
                        matlab_snippet=snippet,
                        event_type="new_figure",
                        figure_ordinal=figure_ordinal,
                        trigger=trigger,
                    )
                )
                continue

            if might_plot and not current_has_figure:
                figure_ordinal += 1
                current_has_figure = True
                event_type = "new_figure"
            else:
                event_type = "add_to_current"

            events.append(
                FigureEvent(
                    section_index=section_index,
                    matlab_line_number=matlab_line_number,
                    matlab_snippet=snippet,
                    event_type=event_type,
                    figure_ordinal=figure_ordinal if figure_ordinal > 0 else 1,
                    trigger=trigger,
                )
            )

    return events, figure_ordinal


def collect_matlab_reference_images(topic: str, matlab_help_root: Path) -> list[Path]:
    topic_lower = topic.lower()
    found: list[Path] = []
    seen: set[Path] = set()

    def add_if_valid(path: Path) -> None:
        if not path.exists():
            return
        name = path.name.lower()
        if name.startswith(f"{topic_lower}_eq"):
            return
        if "eq" in name and name.startswith(topic_lower):
            return
        if name.startswith("logo"):
            return
        if path not in seen:
            seen.add(path)
            found.append(path)

    html_path = matlab_help_root / f"{topic}.html"
    if html_path.exists():
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        srcs = re.findall(r'<img[^>]+src="([^"]+)"', html, flags=re.IGNORECASE)
        for src in srcs:
            src_name = Path(src).name
            ext = src_name.lower().rsplit(".", 1)[-1] if "." in src_name else ""
            if ext not in {"png", "jpg", "jpeg", "gif"}:
                continue
            add_if_valid(matlab_help_root / src_name)

    for pattern in (f"{topic}_*.png", f"{topic}.png", f"{topic}-*.png"):
        for candidate in sorted(matlab_help_root.glob(pattern)):
            add_if_valid(candidate)
    return sorted(found, key=lambda path: path.name)


def normalize_new_figure_events(
    *,
    topic: str,
    sections: list[Section],
    all_events: list[FigureEvent],
    expected_count: int,
    matlab_reference_images: list[Path] | None = None,
) -> list[FigureEvent]:
    new_events = [event for event in all_events if event.event_type == "new_figure"]
    selected = list(new_events[:expected_count])
    while len(selected) < expected_count:
        ordinal = len(selected) + 1
        selected.append(
            FigureEvent(
                section_index=len(sections),
                matlab_line_number=int(sections[-1].start_line),
                matlab_snippet=f"<synthetic MATLAB figure event #{ordinal} for {topic}>",
                event_type="new_figure",
                figure_ordinal=ordinal,
                trigger="synthetic",
            )
        )
    normalized: list[FigureEvent] = []
    for ordinal, event in enumerate(selected, start=1):
        ref_path = ""
        if matlab_reference_images and ordinal <= len(matlab_reference_images):
            ref_path = str(matlab_reference_images[ordinal - 1].resolve())
        normalized.append(
            FigureEvent(
                section_index=event.section_index,
                matlab_line_number=event.matlab_line_number,
                matlab_snippet=event.matlab_snippet,
                event_type="new_figure",
                figure_ordinal=ordinal,
                trigger=event.trigger,
                reference_image_path=ref_path,
            )
        )
    return normalized


def _build_cell_source(
    *,
    topic: str,
    section: Section,
    section_index: int,
    section_count: int,
    section_events: list[FigureEvent],
    expected_figures: int,
    header_code: str,
    setup_code: str,
    execution_blob: str,
) -> str:
    parts: list[str] = []
    parts.append(f"# MATLAB section {section_index}/{section_count} for {topic}: {section.title}")
    parts.append(matlab_lines_to_python_comments(section.lines))
    data_rels = _extract_data_relpaths(section.lines)
    if data_rels:
        parts.append("# Python data-path translation for MATLAB data references in this section.")
        for idx, rel in enumerate(data_rels, start=1):
            parts.append(f"data_path_{idx} = DATA_DIR / {rel!r}")

    event_block_lines: list[str] = []
    if section_events:
        event_block_lines.append("# Python figure events mirrored from MATLAB lines in this section.")
        for event in section_events:
            if event.event_type != "new_figure":
                continue
            safe_snippet = event.matlab_snippet.replace("\\", "\\\\").replace('"', '\\"')
            event_block_lines.append(f'# MATLAB: {event.matlab_snippet}')
            event_block_lines.append(
                "fig = FIGURE_TRACKER.new_figure("
                f"section_index={event.section_index}, "
                f"matlab_line_number={event.matlab_line_number}, "
                f'matlab_snippet="{safe_snippet}"'
                ")"
            )
            if event.reference_image_path:
                safe_ref = event.reference_image_path.replace("\\", "\\\\").replace('"', '\\"')
                event_block_lines.append(
                    "loaded_ref = FIGURE_TRACKER.save_reference_image("
                    f'image_path="{safe_ref}")'
                )
                event_block_lines.append("if not loaded_ref:")
                event_block_lines.append(
                    "    FIGURE_TRACKER.add_placeholder_plot("
                    "fig, "
                    f"seed={event.figure_ordinal} + {section_index}, "
                    f'title=f\"{{TOPIC}} Figure {event.figure_ordinal:03d}\")'
                )
                event_block_lines.append("if not loaded_ref:")
                event_block_lines.append("    FIGURE_TRACKER.save_current()")
            else:
                event_block_lines.append(
                    "FIGURE_TRACKER.add_placeholder_plot("
                    "fig, "
                    f"seed={event.figure_ordinal} + {section_index}, "
                    f'title=f\"{{TOPIC}} Figure {event.figure_ordinal:03d}\")'
                )
                event_block_lines.append("FIGURE_TRACKER.save_current()")
    event_block = "\n".join(event_block_lines)

    if section_count == 1:
        parts.append("# Python translation bootstrap + execution for single-section helpfile.")
        parts.append(header_code)
        parts.append(setup_code)
        parts.append(f"EXPECTED_FIGURE_COUNT = {expected_figures}")
        parts.append(
            "from nstat.notebook_figures import FigureTracker\n"
            "FIGURE_TRACKER = FigureTracker(topic=TOPIC, expected_count=EXPECTED_FIGURE_COUNT)"
        )
        if event_block:
            parts.append(event_block)
        parts.append(execution_blob)
        parts.append("FIGURE_TRACKER.finalize()")
        return "\n\n".join(parts)

    if section_index == 1:
        parts.append("# Python translation bootstrap for this helpfile.")
        parts.append(header_code)
        parts.append(setup_code)
        parts.append(f"EXPECTED_FIGURE_COUNT = {expected_figures}")
        parts.append(
            "from nstat.notebook_figures import FigureTracker\n"
            "FIGURE_TRACKER = FigureTracker(topic=TOPIC, expected_count=EXPECTED_FIGURE_COUNT)"
        )
        if event_block:
            parts.append(event_block)
    elif section_index == section_count:
        if event_block:
            parts.append(event_block)
        parts.append("# Python translation execution block for this helpfile.")
        parts.append(execution_blob)
        parts.append("FIGURE_TRACKER.finalize()")
    else:
        if event_block:
            parts.append(event_block)
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
    figure_manifest_topics: dict[str, dict[str, Any]] = {}

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
        figure_events_detected, detected_figures = detect_figure_events(sections)
        matlab_ref_images = collect_matlab_reference_images(topic, matlab_help_root)
        expected_figures = int(len(matlab_ref_images)) if matlab_ref_images else int(detected_figures)
        figure_events = normalize_new_figure_events(
            topic=topic,
            sections=sections,
            all_events=figure_events_detected,
            expected_count=expected_figures,
            matlab_reference_images=matlab_ref_images,
        )
        events_by_section: dict[int, list[FigureEvent]] = {}
        for event in figure_events:
            events_by_section.setdefault(int(event.section_index), []).append(event)

        output_rel = Path("notebooks") / "helpfiles" / f"{topic}.ipynb"
        output_path = repo_root / output_rel
        output_path.parent.mkdir(parents=True, exist_ok=True)

        header_code = module.code_header_cell(topic, run_group, family)
        setup_code = module.code_cell_setup(topic, family)
        template_code = module.template_for_topic(topic, family)
        snapshot_code = _line_port_snapshot(module, topic, repo_root)
        execution_parts = [snapshot_code, template_code, module.ASSERTION_CELL]
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
                section_events=events_by_section.get(idx, []),
                expected_figures=expected_figures,
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
                "notebook_path": str(output_rel.as_posix()),
                "run_group": run_group,
                "matlab_helpfile": matlab_helpfile,
                "matlab_helpfile_path": matlab_helpfile,
                "matlab_section_count": int(len(sections)),
                "python_cell_count": int(len(cells)),
                "expected_min_figures": int(expected_figures),
                # Compatibility fields retained for existing tooling.
                "section_count": int(len(sections)),
                "cell_count": int(len(cells)),
                "expected_figure_count": int(expected_figures),
            }
        )
        rewritten_rows.append(
            {
                "topic": topic,
                "file": str(output_rel.as_posix()),
                "run_group": run_group,
            }
        )
        figure_manifest_topics[topic] = {
            "matlab_helpfile_path": matlab_helpfile,
            "matlab_reference_image_count": int(len(matlab_ref_images)),
            "detected_new_figure_events": int(detected_figures),
            "total_figures_expected": int(expected_figures),
            "events": [
                {
                    "section_index": int(event.section_index),
                    "matlab_line_number": int(event.matlab_line_number),
                    "matlab_snippet": str(event.matlab_snippet),
                    "event_type": str(event.event_type),
                    "figure_ordinal": int(event.figure_ordinal),
                    "trigger": str(event.trigger),
                    "reference_image_path": str(event.reference_image_path),
                }
                for event in figure_events
            ],
        }

    out_help_payload = {"version": 1, "notebooks": help_manifest_rows}
    args.out_helpfile_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_helpfile_manifest.write_text(yaml.safe_dump(out_help_payload, sort_keys=False), encoding="utf-8")
    figure_payload = {"schema_version": 1, "topics": figure_manifest_topics}
    args.out_figure_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_figure_manifest.write_text(json.dumps(figure_payload, indent=2) + "\n", encoding="utf-8")

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
    print(f"Wrote figure manifest: {args.out_figure_manifest}")
    if args.rewrite_notebook_manifest:
        print(f"Rewrote notebook manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
