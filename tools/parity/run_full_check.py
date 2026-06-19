#!/usr/bin/env python3
"""Top-level MATLAB↔Python parity-check orchestrator.

Drives the full parity-sweep pipeline end-to-end:

1. Extract MATLAB helpfile figures from `.mlx` sources (if available).
2. Execute the notebook fleet.
3. Rebuild `docs/notebook_galleries/` from executed notebooks.
4. Score visual parity (SSIM) against MATLAB references.
5. Build side-by-side composite PNGs (if helper available).
6. Run section-aligned code-structure diff (MATLAB helpfiles ↔ notebooks).
7. Run class-method parity audit (MATLAB classes ↔ Python classes).
8. Write a summary Markdown report under ``.parity-review/``.

Two modes
---------
- *full* (default): every stage above (~30 minutes).
- ``--quick``: skip extraction + notebook execution AND skip the
  code-structure / class-method parity stages; composite + SSIM only
  against the current gallery state (~30 seconds).

Exit code
---------
- ``0`` — every required stage succeeded and no SSIM threshold failed.
- ``2`` — at least one SSIM threshold failed.
- ``1`` — a required pipeline stage errored out.

Optional helpers (``tools/parity/extract_mlx.py``,
``tools/parity/build_composites.py``) are skipped gracefully when missing;
their absence is recorded in the summary but does not fail the run.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = REPO_ROOT / ".parity-review"
RESULTS_JSON = REPO_ROOT / "parity" / "visual_fidelity_results.json"


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], *, stage: str, log_path: Path) -> tuple[int, str]:
    """Run ``cmd`` from the repo root, capturing combined stdout/stderr.

    Returns ``(returncode, tail)`` where ``tail`` is the last ~40 lines of
    the captured output (for the summary report).  Full output is written
    to ``log_path``.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== [{stage}] {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        log_path.write_text(f"FileNotFoundError: {exc}\n", encoding="utf-8")
        return 127, str(exc)
    combined = (proc.stdout or "") + (proc.stderr or "")
    log_path.write_text(combined, encoding="utf-8")
    tail = "\n".join(combined.splitlines()[-40:])
    if proc.returncode != 0:
        print(f"    [{stage}] returned exit code {proc.returncode}")
    return proc.returncode, tail


def _try_extract_mlx(review_dir: Path) -> dict:
    """Run ``extract_mlx.extract_all()`` if the module is present."""
    try:
        from tools.parity import extract_mlx  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        return {
            "stage": "extract_mlx",
            "status": "skipped",
            "reason": f"tools/parity/extract_mlx.py not importable: {exc}",
        }
    try:
        result = extract_mlx.extract_all()  # type: ignore[attr-defined]
    except Exception as exc:
        (review_dir / "extract_mlx.log").write_text(
            f"extract_all raised: {exc}\n", encoding="utf-8"
        )
        return {"stage": "extract_mlx", "status": "error", "reason": str(exc)}
    return {"stage": "extract_mlx", "status": "ok", "result": repr(result)[:200]}


def _try_build_composites(review_dir: Path) -> dict:
    """Run ``tools/parity/build_composites.py --all`` if it exists."""
    script = REPO_ROOT / "tools" / "parity" / "build_composites.py"
    if not script.exists():
        return {
            "stage": "build_composites",
            "status": "skipped",
            "reason": "tools/parity/build_composites.py not present",
        }
    rc, tail = _run(
        [sys.executable, str(script), "--all"],
        stage="build_composites",
        log_path=review_dir / "build_composites.log",
    )
    return {
        "stage": "build_composites",
        "status": "ok" if rc == 0 else "error",
        "exit_code": rc,
        "tail": tail,
    }


def _run_code_structure_diff(review_dir: Path) -> dict:
    """Run ``tools/parity/code_structure_diff.py --all`` if present."""
    script = REPO_ROOT / "tools" / "parity" / "code_structure_diff.py"
    if not script.exists():
        return {
            "stage": "code_structure_diff",
            "status": "skipped",
            "reason": "tools/parity/code_structure_diff.py not present",
        }
    rc, tail = _run(
        [sys.executable, str(script), "--all"],
        stage="code_structure_diff",
        log_path=review_dir / "code_structure_diff.log",
    )
    return {
        "stage": "code_structure_diff",
        "status": "ok" if rc == 0 else "error",
        "exit_code": rc,
        "tail": tail,
    }


def _run_class_method_parity(review_dir: Path) -> dict:
    """Run ``tools/parity/class_method_parity.py --all`` if present."""
    script = REPO_ROOT / "tools" / "parity" / "class_method_parity.py"
    if not script.exists():
        return {
            "stage": "class_method_parity",
            "status": "skipped",
            "reason": "tools/parity/class_method_parity.py not present",
        }
    rc, tail = _run(
        [sys.executable, str(script), "--all"],
        stage="class_method_parity",
        log_path=review_dir / "class_method_parity.log",
    )
    return {
        "stage": "class_method_parity",
        "status": "ok" if rc == 0 else "error",
        "exit_code": rc,
        "tail": tail,
    }


def _load_ssim_results() -> list[dict]:
    if not RESULTS_JSON.exists():
        return []
    try:
        payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return list(payload.get("entries", []))


def _summarize_by_topic(entries: list[dict]) -> list[dict]:
    """Group SSIM entries by topic for the report table."""
    by_topic: dict[str, dict] = {}
    for e in entries:
        topic = e.get("topic", "<unknown>")
        slot = by_topic.setdefault(
            topic,
            {"topic": topic, "scored": 0, "passed": 0, "failed": 0, "skipped": 0, "scores": []},
        )
        if e.get("skipped"):
            slot["skipped"] += 1
            continue
        if e.get("ssim") is None:
            slot["failed"] += 1
            continue
        slot["scored"] += 1
        slot["scores"].append(float(e["ssim"]))
        if e.get("passed"):
            slot["passed"] += 1
        else:
            slot["failed"] += 1
    return [by_topic[k] for k in sorted(by_topic)]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _write_summary(
    review_dir: Path,
    *,
    timestamp: str,
    mode: str,
    stages: list[dict],
    ssim_entries: list[dict],
) -> Path:
    review_dir.mkdir(parents=True, exist_ok=True)
    out = review_dir / f"SUMMARY_run_{timestamp}.md"

    lines: list[str] = []
    lines.append(f"# Parity check — run {timestamp} UTC")
    lines.append("")
    lines.append(f"- Mode: `{mode}`")
    lines.append(f"- Repo: `{REPO_ROOT}`")
    lines.append("")
    lines.append("## Pipeline stages")
    lines.append("")
    lines.append("| Stage | Status | Notes |")
    lines.append("|---|---|---|")
    for s in stages:
        notes = s.get("reason") or s.get("tail", "")
        notes = " ".join(str(notes).split())[:160] or "—"
        lines.append(f"| `{s['stage']}` | {s['status']} | {notes} |")
    lines.append("")

    n_pass = sum(1 for e in ssim_entries if e.get("ssim") is not None and e.get("passed"))
    n_fail = sum(
        1
        for e in ssim_entries
        if not e.get("skipped") and (e.get("ssim") is None or not e.get("passed"))
    )
    n_skip = sum(1 for e in ssim_entries if e.get("skipped"))
    n_total = len(ssim_entries)
    lines.append("## SSIM overall")
    lines.append("")
    lines.append(f"- Entries: **{n_total}**")
    lines.append(f"- Passed: **{n_pass}**")
    lines.append(f"- Failed: **{n_fail}**")
    lines.append(f"- Skipped: **{n_skip}**")
    lines.append("")

    grouped = _summarize_by_topic(ssim_entries)
    if grouped:
        lines.append("## Per-topic SSIM")
        lines.append("")
        lines.append("| Topic | Scored | Passed | Failed | Skipped | Min | Mean | Max |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for g in grouped:
            scores = g["scores"]
            if scores:
                mn = f"{min(scores):.3f}"
                mx = f"{max(scores):.3f}"
                mean = f"{sum(scores) / len(scores):.3f}"
            else:
                mn = mx = mean = "—"
            lines.append(
                f"| `{g['topic']}` | {g['scored']} | {g['passed']} | "
                f"{g['failed']} | {g['skipped']} | {mn} | {mean} | {mx} |"
            )
        lines.append("")

    failing = [
        e
        for e in ssim_entries
        if not e.get("skipped") and e.get("ssim") is not None and not e.get("passed")
    ]
    if failing:
        lines.append("## Failing entries")
        lines.append("")
        lines.append("| Topic | Figure | SSIM | Threshold | Notes |")
        lines.append("|---|---|---:|---:|---|")
        for e in failing:
            notes = " ".join(str(e.get("notes", "")).split())[:120] or "—"
            lines.append(
                f"| `{e.get('topic', '')}` | {e.get('figure_description', '')} | "
                f"{e.get('ssim', 0):.3f} | {e.get('threshold', 0):.2f} | {notes} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "Generated by `tools/parity/run_full_check.py`. "
        "Per-stage logs live alongside this file as `<stage>.log`."
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip extraction + notebook execution; composite + SSIM only.",
    )
    args = parser.parse_args(argv)

    mode = "quick" if args.quick else "full"
    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    stages: list[dict] = []
    hard_failure = False

    if not args.quick:
        # 1. Extract MATLAB .mlx figures (optional module).
        stages.append(_try_extract_mlx(REVIEW_DIR))

        # 2. Execute notebooks.
        rc, tail = _run(
            [
                sys.executable,
                "tools/notebook_build/run_notebooks.py",
                "--group",
                "all",
                "--timeout",
                "1500",
            ],
            stage="run_notebooks",
            log_path=REVIEW_DIR / "run_notebooks.log",
        )
        stages.append(
            {
                "stage": "run_notebooks",
                "status": "ok" if rc == 0 else "error",
                "exit_code": rc,
                "tail": tail,
            }
        )
        if rc != 0:
            hard_failure = True

        # 3. Rebuild galleries from the executed notebooks.
        rc, tail = _run(
            [
                sys.executable,
                "tools/notebook_build/build_notebook_galleries.py",
                "--skip-execute",
            ],
            stage="build_notebook_galleries",
            log_path=REVIEW_DIR / "build_notebook_galleries.log",
        )
        stages.append(
            {
                "stage": "build_notebook_galleries",
                "status": "ok" if rc == 0 else "error",
                "exit_code": rc,
                "tail": tail,
            }
        )
        if rc != 0:
            hard_failure = True

    # 4. SSIM scoring (runs in both modes).
    rc, tail = _run(
        [sys.executable, "tools/parity/build_visual_comparison.py"],
        stage="build_visual_comparison",
        log_path=REVIEW_DIR / "build_visual_comparison.log",
    )
    stages.append(
        {
            "stage": "build_visual_comparison",
            "status": "ok" if rc == 0 else "error",
            "exit_code": rc,
            "tail": tail,
        }
    )
    if rc != 0:
        # The SSIM-scoring script itself failing (missing manifest, etc.)
        # is a hard failure; per-entry threshold failures are surfaced
        # separately from the results JSON.
        hard_failure = True

    # 5. Composite generation (optional helper).
    stages.append(_try_build_composites(REVIEW_DIR))

    # 6. Code-structure diff + class-method parity audit (default mode only).
    if not args.quick:
        stages.append(_run_code_structure_diff(REVIEW_DIR))
        stages.append(_run_class_method_parity(REVIEW_DIR))

    # 7. Summarize.
    ssim_entries = _load_ssim_results()
    summary_path = _write_summary(
        REVIEW_DIR,
        timestamp=timestamp,
        mode=mode,
        stages=stages,
        ssim_entries=ssim_entries,
    )
    print(f"\nWrote summary: {summary_path}")

    # Exit code: 1 for pipeline hard failure, 2 for SSIM regressions, 0 otherwise.
    n_threshold_fail = sum(
        1
        for e in ssim_entries
        if not e.get("skipped") and e.get("ssim") is not None and not e.get("passed")
    )
    if hard_failure:
        return 1
    if n_threshold_fail > 0:
        print(f"SSIM threshold failures: {n_threshold_fail}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
