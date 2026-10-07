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
    parser.add_argument(
        "--require-dataset",
        action="store_true",
        help=(
            "Fail fast with an actionable message if the figshare paper "
            "dataset is not installed, instead of letting dataset-dependent "
            "notebook cells fail deep into a kernel run (or silently skip, "
            "which is how HippocampalPlaceCellExample's broken cells 5-6 "
            "went unnoticed for months)."
        ),
    )
    parser.add_argument(
        "--kernel-name",
        default="python3",
        help=(
            "Jupyter kernel name to execute notebooks with (default: python3). "
            "Pass a dedicated name (e.g. nstat-check) when the caller installed "
            "an isolated kernelspec rather than overwriting the user's own "
            "'python3' kernel -- see `make notebooks-check`."
        ),
    )
    parser.add_argument(
        "--warnings-as-errors",
        action="store_true",
        default=os.environ.get("NSTAT_NOTEBOOK_WARNINGS_AS_ERRORS", "").strip().lower()
        in ("1", "true", "yes"),
        help=(
            "Turn matplotlib.MatplotlibDeprecationWarning into an error inside "
            "the notebook kernel (also settable via "
            "NSTAT_NOTEBOOK_WARNINGS_AS_ERRORS=1). Catches a removed-API break "
            "before it ships, the way a plain pytest filterwarnings never can "
            "for code that only runs inside a notebook kernel."
        ),
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


_WARNINGS_AS_ERRORS_SOURCE = (
    "import warnings as _nstat_warnings\n"
    "try:\n"
    "    from matplotlib import MatplotlibDeprecationWarning as _MplDeprecationWarning\n"
    "except ImportError:\n"
    "    from matplotlib._api.deprecation import (\n"
    "        MatplotlibDeprecationWarning as _MplDeprecationWarning,\n"
    "    )\n"
    "_nstat_warnings.filterwarnings('error', category=_MplDeprecationWarning)\n"
)


def execute_notebook(
    path: Path,
    timeout: int,
    *,
    warnings_as_errors: bool = False,
    kernel_name: str = "python3",
) -> None:
    notebook = nbformat.read(path, as_version=4)
    if warnings_as_errors:
        # nbclient/ipykernel run in a separate process from this script, so a
        # warnings.filterwarnings() call here never reaches the kernel.
        # Inject it as the notebook's own first cell instead (in-memory only
        # — the file on disk is never modified).
        guard_cell = nbformat.v4.new_code_cell(source=_WARNINGS_AS_ERRORS_SOURCE)
        notebook.cells.insert(0, guard_cell)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()


def _check_dataset_present() -> str | None:
    """Return an actionable failure message, or None if the dataset is present."""
    from nstat.data_manager import data_is_present, get_data_dir

    data_dir = get_data_dir()
    if data_is_present(data_dir):
        return None
    return (
        f"Figshare paper dataset not found at {data_dir}.\n"
        "Install it first:\n"
        "  nstat-install --download-example-data always\n"
        "or point NSTAT_DATA_DIR at an existing local copy."
    )


def main() -> int:
    args = parse_args()
    if args.require_dataset:
        message = _check_dataset_present()
        if message is not None:
            print(message, file=sys.stderr)
            return 1
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
            execute_notebook(
                target.path,
                timeout=args.timeout,
                warnings_as_errors=args.warnings_as_errors,
                kernel_name=args.kernel_name,
            )
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
