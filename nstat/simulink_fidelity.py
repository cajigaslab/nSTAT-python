from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


STATUS_LEGEND = (
    "exact_native_python",
    "high_fidelity_native_python",
    "generated_code_wrapped",
    "packaged_runtime",
    "matlab_engine_reference",
    "reference_only",
    "unsupported",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_simulink_fidelity_audit(repo_root: Path | None = None) -> dict[str, Any]:
    base = _repo_root() if repo_root is None else repo_root.resolve()
    path = base / "parity" / "simulink_fidelity.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def normalize_simulink_status(row: dict[str, Any]) -> str:
    explicit = str(row.get("status", "")).strip()
    if explicit:
        return explicit

    strategy = str(row.get("python_strategy", "")).strip()
    current = str(row.get("current_python_status", "")).strip()
    if strategy == "reference_only" or current == "reference_only":
        return "reference_only"
    if strategy == "generated_code_wrapped":
        return "generated_code_wrapped"
    if strategy == "packaged_runtime":
        return "packaged_runtime"
    if strategy == "matlab_engine_fallback":
        return "matlab_engine_reference"
    if strategy == "unsupported" or current == "unsupported":
        return "unsupported"
    if strategy == "native_python" and current == "exact":
        return "exact_native_python"
    if strategy == "native_python" and current == "high_fidelity":
        return "high_fidelity_native_python"
    return current or strategy or "unsupported"


def summarize_simulink_strategies(payload: dict[str, Any]) -> dict[str, int]:
    counts = {status: 0 for status in payload.get("strategy_legend", [])}
    for row in payload.get("items", []):
        strategy = str(row.get("python_strategy", "")).strip()
        if strategy not in counts:
            counts[strategy] = 0
        counts[strategy] += 1
    return counts


def summarize_simulink_statuses(payload: dict[str, Any]) -> dict[str, int]:
    counts = {status: 0 for status in payload.get("status_legend", STATUS_LEGEND)}
    for row in payload.get("items", []):
        status = normalize_simulink_status(row)
        if status not in counts:
            counts[status] = 0
        counts[status] += 1
    return counts


def iter_outstanding_simulink_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in payload.get("items", [])
        if normalize_simulink_status(row) in {"unsupported"}
    ]


__all__ = [
    "iter_outstanding_simulink_items",
    "load_simulink_fidelity_audit",
    "normalize_simulink_status",
    "STATUS_LEGEND",
    "summarize_simulink_statuses",
    "summarize_simulink_strategies",
]
