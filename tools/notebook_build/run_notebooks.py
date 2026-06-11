#!/usr/bin/env python3
"""Execute nSTAT-python notebooks deterministically for CI validation."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import nbformat
import yaml
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nstat.notebook_parity import (
    extract_figure_contract,
    reset_notebook_figure_artifacts,
    validate_notebook_figure_artifacts,
)


@dataclass(frozen=True)
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
        help="Notebook manifest path",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root",
    )
    parser.add_argument(
        "--group",
        default="smoke",
        help="Execution group: smoke, core, full, all, or a custom group from the groups file.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-cell timeout in seconds",
    )
    parser.add_argument(
        "--topics",
        default="",
        help="Optional comma-separated topic subset to execute.",
    )
    parser.add_argument(
        "--groups-file",
        type=Path,
        default=Path(__file__).resolve().parent / "topic_groups.yml",
        help="Optional topic-group mapping file.",
    )
    return parser.parse_args()


def load_targets(manifest_path: Path, repo_root: Path) -> list[NotebookTarget]:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
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


def load_topic_groups(groups_file: Path) -> dict[str, list[str]]:
    if not groups_file.exists():
        return {}
    payload = yaml.safe_load(groups_file.read_text(encoding="utf-8")) or {}
    groups = payload.get("groups", {})
    out: dict[str, list[str]] = {}
    if not isinstance(groups, dict):
        return out
    for key, value in groups.items():
        if not isinstance(value, list):
            continue
        out[str(key)] = [str(item).strip() for item in value if str(item).strip()]
    return out


def select_targets(targets: list[NotebookTarget], group: str) -> list[NotebookTarget]:
    if group in {"full", "all"}:
        return targets
    return [target for target in targets if target.run_group == "smoke"]


def execute_notebook(path: Path, timeout: int) -> None:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    all_targets = load_targets(args.manifest, args.repo_root)
    groups = load_topic_groups(args.groups_file)
    if args.group in groups:
        wanted = set(groups[args.group])
        targets = [target for target in all_targets if target.topic in wanted]
    else:
        targets = select_targets(all_targets, args.group)

    if args.topics.strip():
        wanted = {token.strip() for token in args.topics.split(",") if token.strip()}
        targets = [target for target in targets if target.topic in wanted]
        if not targets:
            raise RuntimeError(f"No notebooks selected for --topics={args.topics!r}")

    if not targets:
        raise RuntimeError(f"No notebooks selected for group={args.group}")

    failures: list[str] = []
    for target in targets:
        if not target.path.exists():
            failures.append(f"missing notebook: {target.path}")
            continue
        print(f"Executing [{target.run_group}] {target.topic}: {target.path}")
        figure_contract = extract_figure_contract(target.path)
        try:
            if figure_contract is not None:
                reset_notebook_figure_artifacts(args.repo_root, figure_contract)
            execute_notebook(target.path, timeout=args.timeout)
            if figure_contract is not None:
                validate_notebook_figure_artifacts(
                    args.repo_root,
                    figure_contract,
                    expected_topic=target.topic,
                )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{target.path}: {exc}")

    if failures:
        print("Notebook execution failures:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print(f"Notebook execution passed for {len(targets)} notebook(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
