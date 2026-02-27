from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT if (PROJECT_ROOT / "helpfiles").exists() else PROJECT_ROOT.parent
REPORT_DIR = PROJECT_ROOT / "reports"

SOURCE_REPORTS = {
    "mfile_parity": "mfile_parity_report.json",
    "examples_notebooks": "examples_notebook_verification.json",
    "matlab_smoke_input": "matlab_smoke_input.json",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reports": {},
    }

    for key, filename in SOURCE_REPORTS.items():
        report_path = REPORT_DIR / filename
        if not report_path.exists():
            payload["reports"][key] = {
                "path": str(report_path.relative_to(REPO_ROOT)),
                "exists": False,
            }
            continue

        data = _read_json(report_path)
        entry: dict[str, Any] = {
            "path": str(report_path.relative_to(REPO_ROOT)),
            "exists": True,
            "sha256": _sha256(report_path),
            "size_bytes": report_path.stat().st_size,
        }
        if isinstance(data, dict) and "summary" in data:
            entry["summary"] = data["summary"]
        payload["reports"][key] = entry

    return payload


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "port_baseline_snapshot.json"
    snapshot = build_snapshot()
    out.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out.relative_to(REPO_ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
