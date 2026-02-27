from __future__ import annotations

import argparse

from ._common import main as _main


def run(repo_root=None, figure_dir=None, render_figures=False):
    from ._common import run_topic

    return run_topic("FitResultExamples", repo_root=repo_root, figure_dir=figure_dir, render_figures=render_figures)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FitResultExamples Python help-topic workflow")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--figure-dir", default=None)
    parser.add_argument("--render-figures", action="store_true")
    args = parser.parse_args()
    return _main("FitResultExamples", repo_root=args.repo_root, figure_dir=args.figure_dir, render_figures=args.render_figures)


if __name__ == "__main__":
    raise SystemExit(main())
