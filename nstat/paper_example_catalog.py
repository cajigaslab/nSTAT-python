from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data_manager import ensure_example_data
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


def run_named_paper_example(example_id: str, repo_root: Path) -> dict[str, dict[str, float]]:
    repo_root = repo_root.resolve()
    data_dir = ensure_example_data(download=True)

    if example_id == "example01":
        return {"experiment1": run_experiment1(data_dir)}
    if example_id == "example02":
        return {"experiment2": run_experiment2(data_dir)}
    if example_id == "example03":
        return {
            "experiment3": run_experiment3(),
            "experiment3b": run_experiment3b(data_dir),
        }
    if example_id == "example04":
        return {"experiment4": run_experiment4(data_dir)}
    if example_id == "example05":
        return {
            "experiment5": run_experiment5(),
            "experiment5b": run_experiment5b(),
            "experiment6": run_experiment6(repo_root),
        }
    raise ValueError(f"Unknown paper example id: {example_id}")


def main_for(example_id: str) -> int:
    parser = argparse.ArgumentParser(description=f"Run canonical nSTAT Python paper example {example_id}")
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    results = run_named_paper_example(example_id, args.repo_root)
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


__all__ = ["main_for", "run_named_paper_example"]
