"""Python installation helper for the standalone nSTAT package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .data_manager import (
    FIGSHARE_DOI_URL,
    PAPER_DOI_URL,
    ensure_example_data,
    get_data_dir,
    get_example_data_info,
)


def _normalize_download_mode(raw_mode: str | bool) -> str:
    if isinstance(raw_mode, bool):
        return "always" if raw_mode else "never"
    mode = str(raw_mode).strip().lower()
    if mode in {"always", "prompt", "never"}:
        return mode
    raise ValueError("download_example_data must be true/false or one of: always, prompt, never")


def _should_prompt_for_example_data(info: dict[str, Any]) -> bool:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False
    prompt = (
        "nSTAT example data was not found.\n\n"
        f"Download the paper-example dataset into:\n{info['data_dir']}\n\n"
        f"Dataset DOI: {FIGSHARE_DOI_URL}\n"
        "Download now? [y/N]: "
    )
    try:
        answer = input(prompt)
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


def _apply_example_data_info(report: dict[str, Any], info: Any) -> None:
    report["example_data"]["data_dir"] = str(info.data_dir)
    report["example_data"]["is_installed"] = bool(info.is_installed)
    report["example_data"]["required_files"] = [str(path) for path in info.required_files]


def nstat_install(
    *,
    rebuild_doc_search: bool = True,
    clean_user_path_prefs: bool = False,
    download_example_data: str | bool = "prompt",
) -> dict[str, Any]:
    """Configure the Python package and optionally install example data."""

    mode = _normalize_download_mode(download_example_data)
    repo_root = Path(__file__).resolve().parents[1]
    repo_info = get_example_data_info(repo_root)
    data_dir = get_data_dir()
    data_info = get_example_data_info(data_dir, treat_as_data_dir=True)

    report: dict[str, Any] = {
        "repo_root": str(repo_root),
        "package_root": str(Path(__file__).resolve().parent),
        "rebuild_doc_search": bool(rebuild_doc_search),
        "clean_user_path_prefs": bool(clean_user_path_prefs),
        "download_example_data": mode,
        "example_data": {
            "data_dir": str(data_dir),
            "is_installed": bool(repo_info.is_installed or data_info.is_installed),
            "figshare_doi": FIGSHARE_DOI_URL,
            "paper_doi": PAPER_DOI_URL,
            "required_files": [str(path) for path in data_info.required_files],
        },
        "notes": [],
    }

    try:
        if repo_info.is_installed:
            _apply_example_data_info(report, repo_info)
            report["notes"].append("Example data already present.")
        elif data_info.is_installed:
            _apply_example_data_info(report, data_info)
            report["notes"].append("Example data already present.")
        elif mode == "always":
            path = ensure_example_data(download=True)
            _apply_example_data_info(report, get_example_data_info(path, treat_as_data_dir=True))
            report["notes"].append("Downloaded example data.")
        elif mode == "prompt":
            if _should_prompt_for_example_data(report["example_data"]):
                path = ensure_example_data(download=True)
                _apply_example_data_info(report, get_example_data_info(path, treat_as_data_dir=True))
                report["notes"].append("Downloaded example data after prompt.")
            else:
                report["notes"].append(
                    "Example data was not preinstalled; paper-example and dataset APIs will download it on first use."
                )
        else:
            report["notes"].append(
                "Example data was not preinstalled; paper-example and dataset APIs will download it on first use."
            )
    except Exception as exc:  # noqa: BLE001
        report["example_data"]["error"] = str(exc)
        report["notes"].append("Example data installation failed.")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-example-data",
        default="prompt",
        help="true/false or always/prompt/never",
    )
    parser.add_argument("--no-rebuild-doc-search", action="store_true")
    parser.add_argument("--clean-user-path-prefs", action="store_true")
    args = parser.parse_args()

    raw_mode: str | bool
    mode_text = str(args.download_example_data).strip().lower()
    if mode_text in {"true", "1", "yes", "on"}:
        raw_mode = True
    elif mode_text in {"false", "0", "no", "off"}:
        raw_mode = False
    else:
        raw_mode = args.download_example_data

    report = nstat_install(
        rebuild_doc_search=not args.no_rebuild_doc_search,
        clean_user_path_prefs=args.clean_user_path_prefs,
        download_example_data=raw_mode,
    )
    print(json.dumps(report, indent=2))
    return 0 if "error" not in report.get("example_data", {}) else 1
