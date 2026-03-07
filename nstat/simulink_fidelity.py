from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_simulink_fidelity_audit(repo_root: Path | None = None) -> dict[str, Any]:
    base = _repo_root() if repo_root is None else repo_root.resolve()
    path = base / "parity" / "simulink_fidelity.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def summarize_simulink_strategies(payload: dict[str, Any]) -> dict[str, int]:
    counts = {status: 0 for status in payload.get("strategy_legend", [])}
    for row in payload.get("items", []):
        strategy = str(row.get("python_strategy", "")).strip()
        if strategy not in counts:
            counts[strategy] = 0
        counts[strategy] += 1
    return counts


def iter_outstanding_simulink_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in payload.get("items", [])
        if row.get("current_python_status") in {"missing", "partial", "unsupported"}
    ]


__all__ = [
    "iter_outstanding_simulink_items",
    "load_simulink_fidelity_audit",
    "summarize_simulink_strategies",
]
