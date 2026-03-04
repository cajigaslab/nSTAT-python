#!/usr/bin/env python3
"""Execute notebooks deterministically and emit a machine-readable report."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import yaml
from nbclient import NotebookClient


THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class NotebookTarget:
    topic: str
    path: Path
    run_group: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "notebook_manifest.yml",
        help="Notebook manifest path.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root.",
    )
    parser.add_argument(
        "--group",
        choices=["smoke", "full", "all"],
        default="smoke",
        help="Execution group from notebook manifest.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-cell timeout in seconds.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=180,
        help="Kernel startup timeout in seconds.",
    )
    parser.add_argument(
        "--max-notebooks",
        type=int,
        default=0,
        help="Optional cap on executed notebooks (0 means all selected).",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=Path("output/notebooks/notebook_execution_report.json"),
        help="JSON report output path.",
    )
    parser.add_argument(
        "--executed-dir",
        type=Path,
        default=Path("output/notebooks/executed"),
        help="Directory for executed notebook copies and extracted images.",
    )
    parser.add_argument(
        "--write-executed",
        action="store_true",
        help="Persist executed notebooks under --executed-dir.",
    )
    return parser.parse_args()


def _set_deterministic_env() -> None:
    for key in THREAD_ENV:
        os.environ.setdefault(key, "1")
    os.environ.setdefault("MPLBACKEND", "Agg")


def _load_targets(manifest_path: Path, repo_root: Path) -> list[NotebookTarget]:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    targets: list[NotebookTarget] = []
    for row in payload.get("notebooks", []):
        targets.append(
            NotebookTarget(
                topic=str(row["topic"]),
                path=repo_root / str(row["file"]),
                run_group=str(row["run_group"]),
            )
        )
    return targets


def _select_targets(targets: list[NotebookTarget], group: str, max_notebooks: int) -> list[NotebookTarget]:
    if group in {"full", "all"}:
        selected = targets
    else:
        selected = [target for target in targets if target.run_group == "smoke"]
    if max_notebooks > 0:
        return selected[:max_notebooks]
    return selected


def _extract_images(notebook: nbformat.NotebookNode, image_dir: Path) -> list[Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    image_idx = 0
    for cell_idx, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code":
            continue
        for out_idx, output in enumerate(cell.get("outputs", [])):
            data = output.get("data", {})
            if not isinstance(data, dict) or "image/png" not in data:
                continue
            blob = data["image/png"]
            if isinstance(blob, str):
                raw = base64.b64decode(blob.encode("utf-8"))
            else:
                continue
            path = image_dir / f"cell{cell_idx:03d}_out{out_idx:03d}_{image_idx:03d}.png"
            path.write_bytes(raw)
            out.append(path)
            image_idx += 1
    return out


def _execute_one(
    target: NotebookTarget,
    timeout: int,
    startup_timeout: int,
    executed_dir: Path,
    write_executed: bool,
) -> dict[str, object]:
    notebook = nbformat.read(target.path, as_version=4)
    start = time.perf_counter()
    error = ""
    image_paths: list[Path] = []
    ok = True

    try:
        client = NotebookClient(
            notebook,
            timeout=timeout,
            startup_timeout=startup_timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(target.path.parent)}},
        )
        client.execute()
        image_paths = _extract_images(notebook, executed_dir / target.topic / "images")
        if write_executed:
            out_nb = executed_dir / target.topic / target.path.name
            out_nb.parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(notebook, out_nb)
    except Exception as exc:  # noqa: BLE001
        ok = False
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - start
    return {
        "topic": target.topic,
        "path": str(target.path),
        "run_group": target.run_group,
        "executed_ok": ok,
        "duration_s": elapsed,
        "image_count": len(image_paths),
        "image_paths": [str(path) for path in image_paths],
        "error": error,
    }


def main() -> int:
    args = parse_args()
    _set_deterministic_env()

    selected = _select_targets(_load_targets(args.manifest, args.repo_root), args.group, args.max_notebooks)
    if not selected:
        raise RuntimeError(f"No notebooks selected for group={args.group}")

    start = time.perf_counter()
    rows: list[dict[str, object]] = []
    for target in selected:
        if not target.path.exists():
            rows.append(
                {
                    "topic": target.topic,
                    "path": str(target.path),
                    "run_group": target.run_group,
                    "executed_ok": False,
                    "duration_s": 0.0,
                    "image_count": 0,
                    "image_paths": [],
                    "error": "FileNotFoundError: notebook path missing",
                }
            )
            continue
        print(f"Executing [{target.run_group}] {target.topic}: {target.path}")
        rows.append(
            _execute_one(
                target=target,
                timeout=args.timeout,
                startup_timeout=args.startup_timeout,
                executed_dir=args.executed_dir,
                write_executed=args.write_executed,
            )
        )

    total_time = time.perf_counter() - start
    failed = [row for row in rows if not bool(row["executed_ok"])]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "group": args.group,
        "max_notebooks": int(args.max_notebooks),
        "summary": {
            "total": len(rows),
            "passed": len(rows) - len(failed),
            "failed": len(failed),
            "duration_s": total_time,
        },
        "environment": {key: os.environ.get(key, "") for key in THREAD_ENV},
        "reports": rows,
    }

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote notebook execution report: {args.out_report}")

    if failed:
        print("Notebook execution failures:")
        for row in failed:
            print(f"  - {row['topic']}: {row['error']}")
        return 1

    print(f"Notebook execution passed for {len(rows)} notebook(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

