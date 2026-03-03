from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPORT_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "reports" / "generate_validation_pdf.py"
SPEC = importlib.util.spec_from_file_location("generate_validation_pdf", REPORT_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

NotebookReport = MODULE.NotebookReport
_uniqueness_violations = MODULE._uniqueness_violations


def _report(
    topic: str,
    image_hashes: list[str],
    *,
    unique_image_count: int | None = None,
) -> NotebookReport:
    unique_count = len(set(image_hashes)) if unique_image_count is None else unique_image_count
    return NotebookReport(
        topic=topic,
        file=Path(f"{topic}.ipynb"),
        run_group="smoke",
        executed=True,
        duration_s=0.1,
        image_paths=[],
        unique_image_paths=[],
        image_hashes=image_hashes,
        image_count=len(image_hashes),
        unique_image_count=unique_count,
        duplicate_image_count=max(0, len(image_hashes) - unique_count),
        text_snippet="",
        error="",
        matlab_ref_images=[],
        similarity_score=None,
        parity_pass=None,
        alignment_status=None,
        matched_python_image=None,
        matched_matlab_image=None,
        parity_metrics=None,
    )


def test_uniqueness_stats_and_no_violations_at_lenient_thresholds() -> None:
    reports = [
        _report("ExampleA", ["h1", "h1", "h2"]),
        _report("ExampleB", ["h2", "h3"]),
    ]
    violations, stats = _uniqueness_violations(
        reports=reports,
        min_unique_images_per_topic=1,
        max_cross_topic_reuse_ratio=1.0,
    )

    assert violations == []
    assert stats["total_image_instances"] == 5
    assert stats["total_unique_hashes"] == 3
    assert stats["cross_topic_reused_hashes"] == 1
    assert stats["repeated_instances"] == 2
    assert float(stats["cross_topic_reuse_ratio"]) == 1.0 / 3.0


def test_uniqueness_violations_for_topic_and_cross_topic_reuse() -> None:
    reports = [
        _report("ExampleA", ["h1", "h1", "h2"], unique_image_count=1),
        _report("ExampleB", ["h2", "h3"]),
    ]
    violations, stats = _uniqueness_violations(
        reports=reports,
        min_unique_images_per_topic=2,
        max_cross_topic_reuse_ratio=0.2,
    )

    assert any("ExampleA: unique_images=1 < min_required=2" in row for row in violations)
    assert any("cross_topic_reuse_ratio=" in row for row in violations)
    assert float(stats["cross_topic_reuse_ratio"]) > 0.2
