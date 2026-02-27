from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT if (PROJECT_ROOT / "helpfiles").exists() else PROJECT_ROOT.parent
REPORT_PATH = PROJECT_ROOT / "reports" / "python_vs_matlab_similarity_report.json"
BASELINE_PATH = PROJECT_ROOT / "reports" / "python_vs_matlab_similarity_baseline.json"


def main() -> int:
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"Missing report: {REPORT_PATH}")

    payload = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    baseline = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "report": payload,
    }
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(f"wrote {BASELINE_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
