from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sync_matlab_reference_figures import _load_contract, _topic_candidates

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_helpfiles() -> Path:
    env = os.environ.get("NSTAT_MATLAB_HELPFILES", "").strip()
    if env:
        return Path(env).expanduser()
    return PROJECT_ROOT / ".." / "nSTAT" / "helpfiles"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate expected MATLAB help-topic figure counts against local helpfiles PNG assets"
    )
    parser.add_argument(
        "--matlab-helpfiles",
        default=str(_default_helpfiles()),
        help="Path to MATLAB nSTAT helpfiles directory",
    )
    parser.add_argument(
        "--report",
        default=str(PROJECT_ROOT / "reports" / "matlab_topic_figure_count_validation.json"),
        help="Output JSON report path",
    )
    args = parser.parse_args()

    helpfiles = Path(args.matlab_helpfiles).expanduser().resolve()
    if not helpfiles.exists() or not helpfiles.is_dir():
        raise RuntimeError(f"MATLAB helpfiles directory not found: {helpfiles}")

    contract = _load_contract()
    rows: list[dict[str, object]] = []
    ok = 0

    for topic, info in sorted(contract.items()):
        expected = int(info.get("expected_figures", 0))
        found = len(_topic_candidates(helpfiles, topic))
        match = found >= expected
        if match:
            ok += 1
        rows.append(
            {
                "topic": topic,
                "expected_figures": expected,
                "candidate_pngs_found": found,
                "ok": match,
                "matlab_target": info.get("matlab_target", ""),
            }
        )

    report = Path(args.report).expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    summary = {"topics": len(rows), "ok": ok, "pass": ok == len(rows), "helpfiles": str(helpfiles)}
    report.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "report": str(report.relative_to(PROJECT_ROOT))}, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
