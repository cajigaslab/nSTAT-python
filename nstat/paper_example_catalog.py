from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data_manager import ensure_example_data
from .paper_figures import default_export_dir, export_named_paper_figures
from .paper_examples_full import (
    _default_repo_root,
    run_experiment1,
    run_experiment2,
    run_experiment3,
    run_experiment3b,
    run_experiment4,
    run_experiment5,
    run_experiment5b,
    run_experiment6,
)


def run_named_paper_example(
    example_id: str, repo_root: Path, *, return_payload: bool = False
) -> dict[str, dict[str, float]] | tuple[dict[str, dict[str, float]], dict[str, dict[str, object]]]:
    repo_root = repo_root.resolve()

    if example_id == "example01":
        data_dir = ensure_example_data(download=True)
        if not return_payload:
            return {"experiment1": run_experiment1(data_dir)}
        summary, payload = run_experiment1(data_dir, return_payload=True)
        return {"experiment1": summary}, {"experiment1": payload}
    if example_id == "example02":
        data_dir = ensure_example_data(download=True)
        if not return_payload:
            return {"experiment2": run_experiment2(data_dir)}
        summary, payload = run_experiment2(data_dir, return_payload=True)
        return {"experiment2": summary}, {"experiment2": payload}
    if example_id == "example03":
        data_dir = ensure_example_data(download=True)
        if not return_payload:
            return {
                "experiment3": run_experiment3(),
                "experiment3b": run_experiment3b(data_dir),
            }
        summary3, payload3 = run_experiment3(return_payload=True)
        summary3b, payload3b = run_experiment3b(data_dir, return_payload=True)
        return {
            "experiment3": summary3,
            "experiment3b": summary3b,
        }, {
            "experiment3": payload3,
            "experiment3b": payload3b,
        }
    if example_id == "example04":
        data_dir = ensure_example_data(download=True)
        if not return_payload:
            return {"experiment4": run_experiment4(data_dir)}
        summary4, payload4 = run_experiment4(data_dir, return_payload=True)
        return {"experiment4": summary4}, {"experiment4": payload4}
    if example_id == "example05":
        if not return_payload:
            return {
                "experiment5": run_experiment5(),
                "experiment5b": run_experiment5b(),
                "experiment6": run_experiment6(repo_root),
            }
        summary5, payload5 = run_experiment5(return_payload=True)
        summary5b, payload5b = run_experiment5b(return_payload=True)
        summary6, payload6 = run_experiment6(repo_root, return_payload=True)
        return {
            "experiment5": summary5,
            "experiment5b": summary5b,
            "experiment6": summary6,
        }, {
            "experiment5": payload5,
            "experiment5b": payload5b,
            "experiment6": payload6,
        }
    raise ValueError(f"Unknown paper example id: {example_id}")


def main_for(example_id: str) -> int:
    parser = argparse.ArgumentParser(description=f"Run canonical nSTAT Python paper example {example_id}")
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--export-figures", action="store_true", help="Export canonical figure files for this example.")
    parser.add_argument("--export-dir", type=Path, default=None, help="Destination directory for exported figure files.")
    args = parser.parse_args()

    if args.export_figures:
        results, payloads = run_named_paper_example(example_id, args.repo_root, return_payload=True)
        export_dir = (args.export_dir or default_export_dir(args.repo_root, example_id)).resolve()
        if len(results) == 1:
            section_name = next(iter(results))
            figure_summary = results[section_name]
            figure_payload = payloads[section_name]
        else:
            figure_summary = results
            figure_payload = payloads
        saved_paths = export_named_paper_figures(
            example_id,
            summary=figure_summary,
            payload=figure_payload,
            export_dir=export_dir,
        )
    else:
        results = run_named_paper_example(example_id, args.repo_root)
        saved_paths = []

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    if saved_paths:
        print("\nGenerated figures:")
        for path in saved_paths:
            print(str(path))
    return 0


__all__ = ["main_for", "run_named_paper_example"]
