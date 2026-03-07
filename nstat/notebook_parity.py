from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nbformat
import yaml


PARITY_NOTES_RELATIVE_PATH = Path("tools") / "notebooks" / "parity_notes.yml"
NOTEBOOK_IMAGE_ROOT = Path("output") / "notebook_images"
FIGURE_MANIFEST_NAME = "manifest.json"
FIGURE_TRACKER_RE = re.compile(
    r"FigureTracker\(\s*topic=['\"](?P<topic>[^'\"]+)['\"]\s*,\s*output_root=OUTPUT_ROOT\s*,\s*expected_count=(?P<count>\d+)\s*\)",
    re.DOTALL,
)
PLACEHOLDER_RE = re.compile(r"(^|\n)\s*pass\b|TODO|FIXME", re.IGNORECASE)


@dataclass(frozen=True)
class NotebookFigureContract:
    topic: str
    expected_count: int
    has_finalize_call: bool

    def topic_dir(self, repo_root: Path) -> Path:
        return repo_root / NOTEBOOK_IMAGE_ROOT / self.topic

    def manifest_path(self, repo_root: Path) -> Path:
        return self.topic_dir(repo_root) / FIGURE_MANIFEST_NAME


@dataclass(frozen=True)
class NotebookPlaceholderAudit:
    placeholder_cells: int
    tracker_only_cells: int
    contains_placeholders: bool
    contains_tracker_only_cells: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_notebook_parity_notes(repo_root: Path | None = None) -> list[dict[str, Any]]:
    base = _repo_root() if repo_root is None else repo_root.resolve()
    payload = yaml.safe_load((base / PARITY_NOTES_RELATIVE_PATH).read_text(encoding="utf-8")) or {}
    return list(payload.get("notes", []))


def summarize_notebook_fidelity(notes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in notes:
        status = str(row.get("fidelity_status", "")).strip()
        if not status:
            continue
        counts[status] = counts.get(status, 0) + 1
    return counts


def iter_outstanding_notebook_fidelity(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in notes if row.get("fidelity_status") == "partial"]


def extract_figure_contract(notebook_path: Path) -> NotebookFigureContract | None:
    notebook = nbformat.read(notebook_path, as_version=4)
    text = "\n".join(str(cell.get("source", "")) for cell in notebook.cells)
    match = FIGURE_TRACKER_RE.search(text)
    if not match:
        return None
    return NotebookFigureContract(
        topic=match.group("topic"),
        expected_count=int(match.group("count")),
        has_finalize_call="__tracker.finalize()" in text,
    )


def audit_notebook_placeholders(notebook_path: Path) -> NotebookPlaceholderAudit:
    notebook = nbformat.read(notebook_path, as_version=4)
    placeholder_cells = 0
    tracker_only_cells = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = str(cell.get("source", ""))
        if PLACEHOLDER_RE.search(source):
            placeholder_cells += 1
        non_empty = [line for line in source.splitlines() if line.strip()]
        non_comment = [line for line in non_empty if not line.lstrip().startswith("#")]
        if non_comment and all(
            line.lstrip().startswith("__tracker.") or line.lstrip().startswith("plt.close(")
            for line in non_comment
        ) and any(line.lstrip().startswith("__tracker.") for line in non_comment):
            tracker_only_cells += 1
    return NotebookPlaceholderAudit(
        placeholder_cells=placeholder_cells,
        tracker_only_cells=tracker_only_cells,
        contains_placeholders=placeholder_cells > 0,
        contains_tracker_only_cells=tracker_only_cells > 0,
    )


def reset_notebook_figure_artifacts(repo_root: Path, contract: NotebookFigureContract) -> None:
    topic_dir = contract.topic_dir(repo_root.resolve())
    if not topic_dir.exists():
        return
    for path in topic_dir.glob("fig_*.png"):
        path.unlink()
    manifest = contract.manifest_path(repo_root.resolve())
    if manifest.exists():
        manifest.unlink()


def validate_notebook_figure_artifacts(
    repo_root: Path,
    contract: NotebookFigureContract,
    *,
    expected_topic: str | None = None,
) -> None:
    base = repo_root.resolve()
    if expected_topic is not None and contract.topic != expected_topic:
        raise AssertionError(
            f"Notebook figure contract topic {contract.topic!r} does not match manifest topic {expected_topic!r}"
        )
    if not contract.has_finalize_call:
        raise AssertionError(f"{contract.topic}: notebook uses FigureTracker but never calls __tracker.finalize()")

    topic_dir = contract.topic_dir(base)
    manifest_path = contract.manifest_path(base)
    if not manifest_path.exists():
        raise AssertionError(f"{contract.topic}: missing notebook figure manifest at {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    produced = int(payload.get("produced_count", -1))
    expected = int(payload.get("expected_count", -1))
    images = [str(item) for item in payload.get("images", [])]

    if payload.get("topic") != contract.topic:
        raise AssertionError(f"{contract.topic}: figure manifest topic mismatch")
    if expected != contract.expected_count:
        raise AssertionError(
            f"{contract.topic}: figure manifest expected_count={expected} does not match notebook contract {contract.expected_count}"
        )
    if produced != contract.expected_count:
        raise AssertionError(
            f"{contract.topic}: figure manifest produced_count={produced} does not match expected_count={contract.expected_count}"
        )
    if len(images) != contract.expected_count:
        raise AssertionError(
            f"{contract.topic}: figure manifest lists {len(images)} image(s), expected {contract.expected_count}"
        )

    disk_images = sorted(topic_dir.glob("fig_*.png"))
    if len(disk_images) != contract.expected_count:
        raise AssertionError(
            f"{contract.topic}: output directory contains {len(disk_images)} figure(s), expected {contract.expected_count}"
        )

    missing = [path for path in images if not (topic_dir / path).exists()]
    if missing:
        raise AssertionError(f"{contract.topic}: manifest references missing figure files: {missing}")


__all__ = [
    "FIGURE_MANIFEST_NAME",
    "NOTEBOOK_IMAGE_ROOT",
    "NotebookFigureContract",
    "NotebookPlaceholderAudit",
    "audit_notebook_placeholders",
    "extract_figure_contract",
    "iter_outstanding_notebook_fidelity",
    "load_notebook_parity_notes",
    "reset_notebook_figure_artifacts",
    "summarize_notebook_fidelity",
    "validate_notebook_figure_artifacts",
]
