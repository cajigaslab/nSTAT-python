"""Upstream-MATLAB change detector.

Hashes the canonical `.m` files in the cajigaslab/nSTAT checkout that
nstat-python's parity audit tracks. Compares against a committed baseline
(`parity/upstream_watch_baseline.yml`). If any hashes differ, writes a
JSON summary that the CI workflow turns into a GitHub issue.

This runs in CI without MATLAB installed — it only reads `.m` source files
and computes SHA-256 hashes.

Modes:
    --matlab-repo PATH --baseline-file YAML --changes-output JSON
        Compute current hashes; if differ from baseline, write changes JSON.
        Always update $GITHUB_STEP_SUMMARY if provided.

    --file-issue JSON_PATH
        Read the changes JSON and open a GitHub issue summarizing them.
        Uses `gh` CLI under the hood.

    --update-baseline PATH
        Recompute baseline from the current upstream and commit to YAML.
        Use after each successful reconciliation cycle to re-anchor.

    --audit-image-pair MATLAB_PNG PYTHON_PNG
        Run the multi-signal content-score heuristic on a MATLAB/Python
        figure pair (see ``image_content_audit.content_score``) and print
        the verdict.  Use this during reconciliation BEFORE filing a
        "degenerate Python figure" issue — the legacy non-white-pixel
        threshold produced recurring false positives on schematics,
        trajectory plots, and dot-scatters.

Usage:
    python tools/parity/upstream_watch.py \\
        --matlab-repo /path/to/nSTAT \\
        --baseline-file parity/upstream_watch_baseline.yml \\
        --changes-output /tmp/changes.json

    python tools/parity/upstream_watch.py \\
        --audit-image-pair MATLAB_PNG PYTHON_PNG
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import yaml

# Files we care about. These are the canonical MATLAB function bodies
# that the parity audit's gold fixtures, drift recipes, and visual
# comparisons depend on. When any of these changes upstream, a
# v10-style reconciliation cycle is warranted.
#
# Paths are relative to the MATLAB-repo root.
TRACKED_FILES = [
    # Core classes that gold fixtures exercise (at MATLAB-repo root)
    "Analysis.m",
    "SignalObj.m",
    "Covariate.m",
    "nspikeTrain.m",
    "Events.m",
    "History.m",
    "TrialConfig.m",
    "Trial.m",
    "FitResult.m",
    "FitResSummary.m",
    "CIF.m",
    # Decoding family — moved to +nstat/+decoding/ namespace in v10 upstream reorg
    "+nstat/+decoding/PPAF.m",
    "+nstat/+decoding/PPHF.m",
    "+nstat/+decoding/PPLFP.m",
    "+nstat/+decoding/PPSS.m",
    # Helpfile sources (visual parity)
    "helpfiles/AnalysisExamples.m",
    "helpfiles/CovariateExamples.m",
    "helpfiles/DecodingExample.m",
    "helpfiles/ExplicitStimulusWhiskerData.m",
    "helpfiles/HippocampalPlaceCellExample.m",
    "helpfiles/HybridFilterExample.m",
    "helpfiles/NetworkTutorial.m",
    "helpfiles/PPSimExample.m",
    "helpfiles/PPThinning.m",
    "helpfiles/PSTHEstimation.m",
    "helpfiles/SignalObjExamples.m",
    "helpfiles/StimulusDecode2D.m",
    "helpfiles/TrialExamples.m",
    "helpfiles/mEPSCAnalysis.m",
    "helpfiles/nSTATPaperExamples.m",
    "helpfiles/nSpikeTrainExamples.m",
    "helpfiles/nstCollExamples.m",
]


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def compute_current_hashes(matlab_repo: Path) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for rel in TRACKED_FILES:
        full = matlab_repo / rel
        if not full.exists():
            out[rel] = None
        else:
            out[rel] = hash_file(full)
    return out


def load_baseline(path: Path) -> dict[str, str | None]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return data.get("hashes", {})


def diff_hashes(
    baseline: dict[str, str | None],
    current: dict[str, str | None],
) -> dict[str, dict[str, str | None]]:
    """Return per-file change dict for files that differ."""
    changes: dict[str, dict[str, str | None]] = {}
    all_keys = set(baseline) | set(current)
    for k in sorted(all_keys):
        b = baseline.get(k)
        c = current.get(k)
        if b != c:
            changes[k] = {"baseline": b, "current": c}
    return changes


def write_summary(changes: dict, fh) -> None:
    if not changes:
        fh.write("## parity-upstream-watch\n\n")
        fh.write("No changes detected in tracked upstream MATLAB files.\n\n")
        fh.write(f"Files checked: {len(TRACKED_FILES)}\n")
        return
    fh.write("## parity-upstream-watch — changes detected\n\n")
    fh.write(f"**{len(changes)} of {len(TRACKED_FILES)} tracked files differ.**\n\n")
    fh.write("| File | Baseline hash | Current hash |\n")
    fh.write("|---|---|---|\n")
    for path, h in changes.items():
        b = h["baseline"] or "_(not present)_"
        c = h["current"] or "_(deleted)_"
        fh.write(f"| `{path}` | `{b}` | `{c}` |\n")
    fh.write("\nRun `docs/parity/runbook.md` reconciliation procedure.\n")


def file_issue(changes_json_path: Path) -> None:
    """Open a GitHub issue summarizing the changes via `gh` CLI."""
    data = json.loads(changes_json_path.read_text())
    changes = data["changes"]
    if not changes:
        print("No changes — not filing issue", file=sys.stderr)
        return

    from datetime import date

    title = f"[parity-watch] Upstream MATLAB changes detected {date.today().isoformat()}"
    body_lines = [
        "Automated detection: `cajigaslab/nSTAT@main` content hash changed for tracked files.",
        "",
        f"**{len(changes)} of {len(TRACKED_FILES)} tracked files differ.**",
        "",
        "| File | Baseline hash | Current hash |",
        "|---|---|---|",
    ]
    for path, h in changes.items():
        b = h["baseline"] or "_(not present)_"
        c = h["current"] or "_(deleted)_"
        body_lines.append(f"| `{path}` | `{b}` | `{c}` |")
    body_lines.append("")
    body_lines.append("## Next steps")
    body_lines.append(
        "Run the post-upstream-merge reconciliation procedure documented in "
        "[`docs/parity/runbook.md`](docs/parity/runbook.md)."
    )
    body_lines.append("")
    body_lines.append(
        "After verifying / addressing, run "
        "`python tools/parity/upstream_watch.py --update-baseline <repo>` "
        "to re-anchor the watch baseline."
    )

    body = "\n".join(body_lines)

    subprocess.run(
        [
            "gh",
            "issue",
            "create",
            "--title",
            title,
            "--body",
            body,
            "--label",
            "parity-watch",
        ],
        check=True,
    )


def update_baseline(matlab_repo: Path, baseline_file: Path) -> None:
    current = compute_current_hashes(matlab_repo)
    payload = {
        "version": 1,
        "updated_utc": subprocess.check_output(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]
        ).decode().strip(),
        "matlab_repo": "cajigaslab/nSTAT@main",
        "tracked_files": len(TRACKED_FILES),
        "hashes": current,
    }
    baseline_file.write_text(yaml.safe_dump(payload, sort_keys=False))
    present = sum(1 for v in current.values() if v is not None)
    missing = sum(1 for v in current.values() if v is None)
    print(f"updated {baseline_file}: {present} hashed, {missing} missing")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matlab-repo", type=Path)
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=Path("parity/upstream_watch_baseline.yml"),
    )
    parser.add_argument("--changes-output", type=Path)
    parser.add_argument("--output-summary", type=Path)
    parser.add_argument("--file-issue", type=Path)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument(
        "--audit-image-pair",
        nargs=2,
        metavar=("MATLAB_PNG", "PYTHON_PNG"),
        type=Path,
        help=(
            "Multi-signal content-score audit on a MATLAB/Python PNG pair. "
            "Use before filing a 'degenerate figure' issue."
        ),
    )
    args = parser.parse_args()

    if args.audit_image_pair:
        # Import the sibling helper without requiring a packaged ``tools.``
        # namespace — the parity tools historically run as plain scripts.
        sys.path.insert(0, str(Path(__file__).parent))
        from image_content_audit import audit_pair  # noqa: E402

        verdict = audit_pair(*args.audit_image_pair)
        print(f"MATLAB: {verdict.matlab.as_dict()}")
        print(f"PYTHON: {verdict.python.as_dict()}")
        print(f"VERDICT: {verdict.verdict}")
        return 0

    if args.file_issue:
        file_issue(args.file_issue)
        return 0

    if args.update_baseline:
        if not args.matlab_repo:
            parser.error("--update-baseline requires --matlab-repo")
        update_baseline(args.matlab_repo, args.baseline_file)
        return 0

    if not args.matlab_repo:
        parser.error("--matlab-repo is required for change detection")

    current = compute_current_hashes(args.matlab_repo)
    baseline = load_baseline(args.baseline_file)
    changes = diff_hashes(baseline, current)

    # Always write the summary
    if args.output_summary:
        with args.output_summary.open("a") as fh:
            write_summary(changes, fh)
    else:
        write_summary(changes, sys.stdout)

    # Write the changes JSON only if there are changes
    if args.changes_output:
        if changes:
            args.changes_output.write_text(
                json.dumps({"changes": changes, "tracked_files": TRACKED_FILES}, indent=2)
            )
        else:
            # Ensure the file doesn't exist (the workflow checks for its presence)
            if args.changes_output.exists():
                args.changes_output.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
