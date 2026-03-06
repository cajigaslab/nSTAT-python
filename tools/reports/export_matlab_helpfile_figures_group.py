#!/usr/bin/env python3
"""Export MATLAB helpfile figures topic-by-topic with clean process isolation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--matlab-repo", type=Path, required=True)
    parser.add_argument(
        "--hydrate-helpfiles-from",
        type=Path,
        default=None,
        help=(
            "Optional materialized MATLAB helpfiles root or repo root. "
            "Defaults to $NSTAT_MATLAB_HELPFILES_SOURCE when set."
        ),
    )
    parser.add_argument("--source-manifest", type=Path, default=Path("parity/help_source_manifest.yml"))
    parser.add_argument("--notebook-manifest", type=Path, default=Path("tools/notebooks/notebook_manifest.yml"))
    parser.add_argument("--groups-file", type=Path, default=Path("tools/notebooks/topic_groups.yml"))
    parser.add_argument("--group", default="", help="Optional topic group (smoke, core, all, full).")
    parser.add_argument("--topics", default="", help="Optional comma-separated topic subset.")
    parser.add_argument("--output-root", type=Path, default=Path("output/matlab_help_images"))
    parser.add_argument("--report-json", type=Path, default=Path("output/matlab_help_images/report_group.json"))
    parser.add_argument("--batch-timeout-seconds", type=int, default=900)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML payload in {path}")
    return payload


def _load_group_topics(group: str, *, notebook_manifest: Path, groups_file: Path) -> list[str]:
    if not group:
        return []

    if groups_file.exists():
        payload = _load_yaml(groups_file)
        groups = payload.get("groups", {})
        if isinstance(groups, dict) and group in groups and isinstance(groups[group], list):
            return [str(item).strip() for item in groups[group] if str(item).strip()]

    payload = _load_yaml(notebook_manifest)
    rows = payload.get("notebooks", [])
    if group in {"all", "full"}:
        return [str(row.get("topic", "")).strip() for row in rows if str(row.get("topic", "")).strip()]
    return [
        str(row.get("topic", "")).strip()
        for row in rows
        if str(row.get("run_group", "")).strip() == group and str(row.get("topic", "")).strip()
    ]


def _selected_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    manifest = _load_yaml((args.repo_root / args.source_manifest).resolve())
    rows = manifest.get("topics", [])
    wanted: set[str] = set()
    if args.group.strip():
        wanted.update(
            _load_group_topics(
                args.group.strip(),
                notebook_manifest=(args.repo_root / args.notebook_manifest).resolve(),
                groups_file=(args.repo_root / args.groups_file).resolve(),
            )
        )
    if args.topics.strip():
        wanted.update(token.strip() for token in args.topics.split(",") if token.strip())
    if wanted:
        rows = [row for row in rows if str(row.get("topic", "")).strip() in wanted]
    if not rows:
        detail = args.topics if args.topics.strip() else args.group
        raise RuntimeError(f"No topics selected for export ({detail!r}).")
    return rows


def _ensure_matlab_data_link(matlab_repo: Path, data_dir: Path) -> None:
    repo_data = matlab_repo / "data"
    if repo_data.is_symlink():
        current = repo_data.resolve()
        if current == data_dir.resolve():
            return
        repo_data.unlink()
    elif repo_data.exists():
        backup = matlab_repo / "data.lfs-skip.bak"
        if backup.exists():
            if backup.is_symlink():
                backup.unlink()
            elif backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()
        repo_data.rename(backup)
    repo_data.symlink_to(data_dir.resolve(), target_is_directory=True)


def _kill_stale_export_matlab() -> None:
    proc = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in proc.stdout.splitlines():
        text = line.strip()
        if "MATLAB_maca64 -batch" not in text:
            continue
        if "export_helpfile_figures(" not in text:
            continue
        pid_text = text.split(maxsplit=1)[0]
        try:
            os.kill(int(pid_text), signal.SIGTERM)
        except Exception:
            continue


def _hydrate_helpfile_assets(
    *,
    repo_root: Path,
    matlab_repo: Path,
    output_root: Path,
    source_help_root: Path | None,
) -> Path:
    report_path = output_root / "hydrate_helpfiles_report.json"
    cmd = [
        "python3",
        "tools/reports/hydrate_matlab_helpfile_assets.py",
        "--matlab-repo",
        str(matlab_repo),
        "--report-json",
        str(report_path),
    ]
    if source_help_root is not None:
        cmd.extend(["--source-help-root", str(source_help_root)])
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "helpfile hydration failed").strip()
        raise RuntimeError(detail)
    return report_path


def _verify_topic_output(row: dict[str, object], output_root: Path) -> dict[str, object]:
    topic = str(row.get("topic", "")).strip()
    expected = int(row.get("expected_figure_count", 0))
    img_paths = sorted((output_root / topic).glob("fig_*.png"))
    produced = len(img_paths)
    status = "ok" if produced == expected else "count_mismatch"
    error = "" if status == "ok" else f"produced={produced} expected={expected}"
    if not img_paths and not bool(row.get("no_figure_utility", False)):
        status = "missing_output"
        error = f"no figure files written for {topic}"
    return {
        "topic": topic,
        "source_path": str(row.get("source_path", "")),
        "source_type": str(row.get("source_type", "")),
        "expected_figures": expected,
        "produced_figures": produced,
        "status": status,
        "error": error,
    }


def _reset_topic_output(output_root: Path, topic: str) -> None:
    topic_dir = output_root / topic
    if topic_dir.exists():
        shutil.rmtree(topic_dir)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    matlab_repo = args.matlab_repo.resolve()
    output_root = (repo_root / args.output_root).resolve()
    report_json = (repo_root / args.report_json).resolve()
    hydrate_source = args.hydrate_helpfiles_from.expanduser().resolve() if args.hydrate_helpfiles_from else None

    data_proc = subprocess.run(
        [
            "python3",
            "-c",
            "from nstat.data_manager import ensure_example_data; print(ensure_example_data())",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    data_dir = Path(data_proc.stdout.strip().splitlines()[-1]).expanduser().resolve()
    _ensure_matlab_data_link(matlab_repo, data_dir)

    rows = _selected_rows(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "topics": [str(row.get("topic", "")) for row in rows],
                    "data_dir": str(data_dir),
                    "hydrate_helpfiles_from": str(hydrate_source) if hydrate_source else os.environ.get("NSTAT_MATLAB_HELPFILES_SOURCE", ""),
                },
                indent=2,
            )
        )
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    hydrate_report = _hydrate_helpfile_assets(
        repo_root=repo_root,
        matlab_repo=matlab_repo,
        output_root=output_root,
        source_help_root=hydrate_source,
    )
    env = {**os.environ, "NSTAT_DATA_DIR": str(data_dir)}

    results: list[dict[str, object]] = []
    failures: list[str] = []
    for idx, row in enumerate(rows, start=1):
        topic = str(row.get("topic", "")).strip()
        topic_report = output_root / f"report_{topic}.json"
        topic_log_dir = output_root / "logs" / topic
        _reset_topic_output(output_root, topic)
        cmd = [
            "python3",
            "tools/reports/export_matlab_helpfile_figures.py",
            "--source-manifest",
            str((repo_root / args.source_manifest).resolve()),
            "--output-root",
            str(output_root),
            "--report-json",
            str(topic_report),
            "--log-dir",
            str(topic_log_dir),
            "--topics",
            topic,
            "--topics-batch-size",
            "1",
            "--batch-timeout-seconds",
            str(args.batch_timeout_seconds),
        ]
        print(f"[{idx}/{len(rows)}] Exporting MATLAB figures for {topic}")
        _kill_stale_export_matlab()
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        if proc.returncode != 0:
            result = {
                "topic": topic,
                "source_path": str(row.get("source_path", "")),
                "source_type": str(row.get("source_type", "")),
                "expected_figures": int(row.get("expected_figure_count", 0)),
                "produced_figures": 0,
                "status": "error",
                "error": (proc.stderr or proc.stdout or f"export command failed for {topic}").strip(),
            }
        else:
            result = _verify_topic_output(row, output_root)
        results.append(result)
        if result["status"] != "ok":
            failures.append(f"{topic}: {result['status']} {result['error']}".strip())
            if not args.keep_going:
                break
        _kill_stale_export_matlab()

    payload = {
        "matlab_repo": str(matlab_repo),
        "data_dir": str(data_dir),
        "hydrate_helpfiles_report": str(hydrate_report),
        "topic_count": len(results),
        "results": results,
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(report_json)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
