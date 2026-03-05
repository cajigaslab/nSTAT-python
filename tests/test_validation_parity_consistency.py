from __future__ import annotations

import json
from pathlib import Path

import pytest


def _strict_failure_topics(strict_summary: dict[str, object]) -> set[str]:
    failures = strict_summary.get("failures", [])
    out: set[str] = set()
    if not isinstance(failures, list):
        return out
    for item in failures:
        text = str(item).strip()
        if not text:
            continue
        topic = text.split(":", 1)[0].strip()
        if topic:
            out.add(topic)
    return out


def _pdf_failure_topics(pdf_summary: dict[str, object]) -> set[str]:
    rows = pdf_summary.get("topics", [])
    out: set[str] = set()
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        topic = str(row.get("topic", "")).strip()
        parity_pass = row.get("parity_pass")
        if topic and parity_pass is False:
            out.add(topic)
    return out


def test_pdf_image_mode_parity_matches_strict_ordinal_summary_when_artifacts_present() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    strict_path = repo_root / "output" / "pdf" / "image_mode_parity" / "summary_full.json"
    pdf_path = repo_root / "output" / "pdf" / "validation_report_latest.json"

    if not strict_path.exists() or not pdf_path.exists():
        pytest.skip("Validation parity artifacts not present locally")

    strict_summary = json.loads(strict_path.read_text(encoding="utf-8"))
    pdf_summary = json.loads(pdf_path.read_text(encoding="utf-8"))

    strict_topics = _strict_failure_topics(strict_summary)
    pdf_topics = _pdf_failure_topics(pdf_summary)

    assert strict_topics == pdf_topics
    assert int(pdf_summary.get("parity_failures", -1)) == len(pdf_topics)
