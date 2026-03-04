#!/usr/bin/env python3
"""Run deterministic Python performance benchmarks for MATLAB parity tracking."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import scipy

try:
    from nstat.performance_workloads import CASE_ORDER, TIER_ORDER, run_python_workload
except ModuleNotFoundError:  # pragma: no cover - fallback for non-installed local runs
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from nstat.performance_workloads import CASE_ORDER, TIER_ORDER, run_python_workload


def _git_sha(repo_root: Path) -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=True,
            )
            .stdout.strip()
        )
    except Exception:
        return "unknown"


def _collect_env() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "omp_num_threads": os.getenv("OMP_NUM_THREADS", ""),
        "mkl_num_threads": os.getenv("MKL_NUM_THREADS", ""),
        "openblas_num_threads": os.getenv("OPENBLAS_NUM_THREADS", ""),
        "veclib_maximum_threads": os.getenv("VECLIB_MAXIMUM_THREADS", ""),
    }


def _median(vals: list[float]) -> float:
    return float(statistics.median(vals)) if vals else float("nan")


def _run_case(case: str, tier: str, repeats: int, warmup: int, seed: int) -> dict[str, Any]:
    runtimes_ms: list[float] = []
    peak_mem_mb: list[float] = []
    summary: dict[str, float] = {}

    for rep in range(warmup + repeats):
        run_seed = int(seed + rep)
        if rep >= warmup:
            tracemalloc.start()
        t0 = time.perf_counter()
        summary = run_python_workload(case=case, tier=tier, seed=run_seed)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if rep >= warmup:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            runtimes_ms.append(float(elapsed_ms))
            peak_mem_mb.append(float(peak / (1024.0 * 1024.0)))

    return {
        "case": case,
        "tier": tier,
        "repeats": int(repeats),
        "warmup": int(warmup),
        "median_runtime_ms": _median(runtimes_ms),
        "mean_runtime_ms": float(np.mean(runtimes_ms)),
        "std_runtime_ms": float(np.std(runtimes_ms)),
        "median_peak_memory_mb": _median(peak_mem_mb),
        "summary": summary,
        "samples_runtime_ms": runtimes_ms,
        "samples_peak_memory_mb": peak_mem_mb,
    }


def _write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "tier",
        "repeats",
        "median_runtime_ms",
        "mean_runtime_ms",
        "std_runtime_ms",
        "median_peak_memory_mb",
        "summary",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case": row["case"],
                    "tier": row["tier"],
                    "repeats": row["repeats"],
                    "median_runtime_ms": row["median_runtime_ms"],
                    "mean_runtime_ms": row["mean_runtime_ms"],
                    "std_runtime_ms": row["std_runtime_ms"],
                    "median_peak_memory_mb": row["median_peak_memory_mb"],
                    "summary": json.dumps(row["summary"], sort_keys=True),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiers",
        type=str,
        default="S,M,L",
        help="Comma-separated tier list from {S,M,L}.",
    )
    parser.add_argument("--repeats", type=int, default=7, help="Measured repeats per case/tier.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup repeats per case/tier.")
    parser.add_argument("--seed", type=int, default=20260303, help="Base deterministic seed.")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("output/performance/python_performance_report.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("output/performance/python_performance_report.csv"),
        help="Output CSV report path.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root for metadata.",
    )
    args = parser.parse_args()

    tiers = [t.strip().upper() for t in args.tiers.split(",") if t.strip()]
    unknown = [t for t in tiers if t not in TIER_ORDER]
    if unknown:
        raise ValueError(f"Unsupported tiers: {unknown}")

    rows: list[dict[str, Any]] = []
    for case in CASE_ORDER:
        for tier in tiers:
            rows.append(_run_case(case=case, tier=tier, repeats=args.repeats, warmup=args.warmup, seed=args.seed))

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "implementation": "python",
        "repo_root": str(args.repo_root.resolve()),
        "git_sha": _git_sha(args.repo_root.resolve()),
        "tiers": tiers,
        "cases": rows,
        "environment": _collect_env(),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(rows, args.out_csv)

    print(f"Wrote Python performance JSON: {args.out_json}")
    print(f"Wrote Python performance CSV: {args.out_csv}")
    print(f"Benchmarked case-tier pairs: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
