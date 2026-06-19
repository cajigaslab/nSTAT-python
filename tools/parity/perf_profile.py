#!/usr/bin/env python3
"""Per-path cProfile harness for performance findings.

Runs :mod:`cProfile` over each hot path defined in :mod:`perf_check` (the
v11 performance baseline) and dumps a top-30 cumulative-time pstats listing
to ``.parity-review/perf_profile_<name>.txt``.  No optimisation happens
here — this is a diagnosis tool.  Iter 55 reads the findings and acts.

Usage
-----
::

    python tools/parity/perf_profile.py                       # all 5 paths
    python tools/parity/perf_profile.py --paths kalman_filter
    python tools/parity/perf_profile.py --runs 5

For each path the output file contains:

* a ``# header`` block with the path name, the underlying Python function,
  the input-size description and number of runs aggregated;
* a ``TOP-10 BY CUMULATIVE TIME`` machine-readable table — one row per
  hot function, tab-separated;
* the full ``pstats.Stats.print_stats(30)`` cumulative dump.

Design notes
------------
- We import the timer closures from :mod:`perf_check` so the inputs match
  the baseline exactly.  The closures include their own input-construction
  cost; that cost shows up in the profile but is consistent across runs.
- Multiple repetitions are aggregated into a single ``cProfile.Profile``
  via ``enable()``/``disable()`` bracketing each call.  This amortises
  sampling jitter without renormalising the per-call counts (so ``ncalls``
  reflects N runs).
- We deliberately do NOT measure MATLAB here.  ``perf_check.py`` already
  captures the MATLAB side; cProfile is a Python-side tool.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import re
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# perf_check.py lives next to us; make tools/parity importable as a package-less dir.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import perf_check  # noqa: E402  (after sys.path manipulation)

BASELINE_PATH = _REPO_ROOT / "parity" / "performance_baseline.yml"
OUT_DIR = _REPO_ROOT / ".parity-review"


# ---------------------------------------------------------------------------
# Path closures (reuse perf_check's HotPath registry)
# ---------------------------------------------------------------------------


def _path_closures() -> dict[str, Callable[[], float]]:
    """Map hot-path name -> python_timer closure from :mod:`perf_check`."""
    return {p.name: p.python_timer for p in perf_check._build_paths()}


# ---------------------------------------------------------------------------
# Profile + extract
# ---------------------------------------------------------------------------


def _format_func_label(func: tuple[str, int, str]) -> str:
    """Render a cProfile (file, lineno, name) tuple as ``file:line:name``."""
    filename, lineno, name = func
    # Trim the user-home prefix so output stays readable across machines.
    short = filename
    home = str(Path.home())
    if short.startswith(home):
        short = "~" + short[len(home):]
    return f"{short}:{lineno}:{name}"


def _extract_top_functions(stats: pstats.Stats, n: int = 10) -> list[dict[str, Any]]:
    """Return the top-``n`` functions by cumulative time as plain dicts.

    Each dict has: ``func``, ``ncalls``, ``tottime``, ``cumtime``,
    ``pct_total`` (cumtime / total runtime, percent).
    """
    # pstats exposes per-function stats in stats.stats: {func: (cc, nc, tt, ct, callers)}
    raw = stats.stats  # type: ignore[attr-defined]
    total = float(stats.total_tt) or 1.0  # type: ignore[attr-defined]

    # Sort by cumulative time descending.
    items = sorted(raw.items(), key=lambda kv: kv[1][3], reverse=True)
    top: list[dict[str, Any]] = []
    for func, (cc, nc, tt, ct, _callers) in items[:n]:
        top.append(
            {
                "func": _format_func_label(func),
                "ncalls": int(nc),
                "primitive_calls": int(cc),
                "tottime": round(float(tt), 6),
                "cumtime": round(float(ct), 6),
                "pct_total": round(100.0 * float(ct) / total, 2),
            }
        )
    return top


def profile_path(name: str, n_runs: int = 3) -> dict[str, Any]:
    """Profile ``name`` across ``n_runs`` invocations of its closure."""
    closures = _path_closures()
    if name not in closures:
        raise KeyError(f"unknown path: {name!r}; known: {sorted(closures)}")
    closure = closures[name]

    prof = cProfile.Profile()
    for _ in range(n_runs):
        prof.enable()
        closure()
        prof.disable()

    buf = io.StringIO()
    stats = pstats.Stats(prof, stream=buf).strip_dirs()
    stats.sort_stats("cumulative").print_stats(30)
    return {
        "raw": buf.getvalue(),
        "top": _extract_top_functions(pstats.Stats(prof), 10),
        "total_tt": float(pstats.Stats(prof).total_tt),  # type: ignore[attr-defined]
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _resolve_meta(name: str) -> dict[str, str]:
    """Pull the python_function + input_size for ``name`` from the baseline YAML."""
    try:
        doc = yaml.safe_load(BASELINE_PATH.read_text())
    except FileNotFoundError:
        return {"python_function": "<unknown>", "input_size": "<unknown>"}
    for entry in doc.get("paths", []):
        if entry.get("name") == name:
            return {
                "python_function": entry.get("python_function", "<unknown>"),
                "input_size": entry.get("input_size", "<unknown>"),
            }
    return {"python_function": "<unknown>", "input_size": "<unknown>"}


def _write_output(name: str, n_runs: int, result: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"perf_profile_{name}.txt"
    meta = _resolve_meta(name)

    lines: list[str] = []
    lines.append(f"# perf_profile: {name}")
    lines.append(f"# python_function: {meta['python_function']}")
    lines.append(f"# input_size: {meta['input_size']}")
    lines.append(f"# n_runs: {n_runs}")
    lines.append(f"# aggregate_total_tt: {result['total_tt']:.6f}s")
    lines.append("")
    lines.append("TOP-10 BY CUMULATIVE TIME (cumtime,tottime in seconds)")
    lines.append("rank\tncalls\ttottime\tcumtime\tpct_total\tfunction")
    for i, fn in enumerate(result["top"], 1):
        lines.append(
            f"{i}\t{fn['ncalls']}\t{fn['tottime']}\t{fn['cumtime']}\t"
            f"{fn['pct_total']}\t{fn['func']}"
        )
    lines.append("")
    lines.append("=" * 78)
    lines.append("FULL pstats (cumulative, top 30)")
    lines.append("=" * 78)
    lines.append(result["raw"])

    out_path.write_text("\n".join(lines))
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--paths",
        nargs="+",
        default=None,
        help="restrict profiling to the named paths (default: all baseline paths)",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=3,
        help="repetitions per path aggregated into one profile (default 3)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    doc = yaml.safe_load(BASELINE_PATH.read_text())
    all_names = [p["name"] for p in doc["paths"]]

    if args.paths:
        unknown = [n for n in args.paths if n not in all_names]
        if unknown:
            print(f"unknown path(s): {unknown}", file=sys.stderr)
            return 2
        targets = args.paths
    else:
        targets = all_names

    for name in targets:
        print(f"[perf_profile] {name}: profiling ({args.runs} runs)...", file=sys.stderr)
        result = profile_path(name, args.runs)
        out_path = _write_output(name, args.runs, result)
        # Print a one-line summary so the run is informative without opening files.
        top3 = result["top"][:3]
        top3_str = " | ".join(
            f"{fn['pct_total']:.1f}% {re.sub(r'^.*:', '', fn['func'])}" for fn in top3
        )
        print(f"  -> {out_path.name}: top3 {top3_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
