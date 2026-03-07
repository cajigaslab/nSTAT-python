"""Python installation helper for the standalone nSTAT package."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
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


def _rebuild_doc_search(repo_root: Path) -> dict[str, Any]:
    docs_dir = repo_root / "docs"
    output_dir = docs_dir / "_build" / "html"
    search_index = output_dir / "searchindex.js"
    report: dict[str, Any] = {
        "requested": True,
        "source_dir": str(docs_dir),
        "output_dir": str(output_dir),
        "search_index": str(search_index),
    }

    if not (docs_dir / "conf.py").exists():
        report["status"] = "skipped"
        report["reason"] = "Docs configuration was not found."
        return report

    if importlib.util.find_spec("sphinx") is None:
        report["status"] = "skipped"
        report["reason"] = "Sphinx is not installed; install nstat-toolbox[dev] to rebuild docs search."
        return report

    proc = subprocess.run(
        [sys.executable, "-m", "sphinx", "-q", "-b", "html", "docs", str(output_dir)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        report["status"] = "failed"
        report["error"] = proc.stderr.strip() or proc.stdout.strip() or "Unknown sphinx build failure."
        return report

    report["status"] = "rebuilt"
    report["is_available"] = search_index.exists()
    if not search_index.exists():
        report["status"] = "failed"
        report["error"] = f"Expected search index was not created: {search_index}"
    return report


def nstat_install(
    *,
    rebuild_doc_search: bool = True,
    clean_user_path_prefs: bool = False,
    download_example_data: str | bool = "prompt",
) -> dict[str, Any]:
    """Configure the Python package and optionally install example data."""

    mode = _normalize_download_mode(download_example_data)
    repo_root = Path(__file__).resolve().parents[1]
    info = get_example_data_info(repo_root)
    data_dir = get_data_dir()

    report: dict[str, Any] = {
        "repo_root": str(repo_root),
        "package_root": str(Path(__file__).resolve().parent),
        "rebuild_doc_search": bool(rebuild_doc_search),
        "clean_user_path_prefs": bool(clean_user_path_prefs),
        "path_preferences": {
            "requested": bool(clean_user_path_prefs),
            "status": "not_applicable",
            "reason": "Python packaging/import resolution replaces MATLAB user path preference cleanup.",
        },
        "download_example_data": mode,
        "example_data": {
            "data_dir": str(data_dir),
            "is_installed": bool(info.is_installed or get_example_data_info(data_dir).is_installed),
            "figshare_doi": FIGSHARE_DOI_URL,
            "paper_doi": PAPER_DOI_URL,
            "required_files": [str(path) for path in info.required_files],
        },
        "notes": [],
    }

    if rebuild_doc_search:
        doc_search = _rebuild_doc_search(repo_root)
        report["doc_search"] = doc_search
        if doc_search["status"] == "rebuilt":
            report["notes"].append("Rebuilt docs HTML search index.")
        elif doc_search["status"] == "skipped":
            report["notes"].append(f"Docs search rebuild skipped: {doc_search['reason']}")
        else:
            report["notes"].append("Docs search rebuild failed.")
    else:
        report["doc_search"] = {
            "requested": False,
            "status": "skipped",
            "reason": "Disabled by caller.",
        }

    if clean_user_path_prefs:
        report["notes"].append("Clean user path preferences is a MATLAB-only option and is ignored in Python.")

    try:
        if info.is_installed:
            report["example_data"]["is_installed"] = True
            report["example_data"]["data_dir"] = str(info.data_dir)
            report["notes"].append("Example data already present.")
        elif mode == "always":
            path = ensure_example_data(download=True)
            report["example_data"]["is_installed"] = True
            report["example_data"]["data_dir"] = str(path)
            report["notes"].append("Downloaded example data.")
        elif mode == "prompt":
            if _should_prompt_for_example_data(report["example_data"]):
                path = ensure_example_data(download=True)
                report["example_data"]["is_installed"] = True
                report["example_data"]["data_dir"] = str(path)
                report["notes"].append("Downloaded example data after prompt.")
            else:
                report["notes"].append("Example data not installed; run with download_example_data=True to install.")
        else:
            report["notes"].append("Example data not installed; run with download_example_data=True to install.")
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
    parser.add_argument(
        "--clean-user-path-prefs",
        action="store_true",
        help="MATLAB-compatibility no-op; Python does not maintain MATLAB-style user path preferences.",
    )
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
    example_data_failed = "error" in report.get("example_data", {})
    doc_search_failed = report.get("doc_search", {}).get("status") == "failed"
    return 0 if not (example_data_failed or doc_search_failed) else 1
