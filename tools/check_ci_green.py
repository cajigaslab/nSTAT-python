#!/usr/bin/env python3
"""Confirm `ci.yml` is green for the current tree, before releasing.

2026-10 EM / release cycle: CI on `main` went red (a removed matplotlib API
+ a BLAS-dependent singular-matrix test) and nobody noticed for a stretch,
because `ci.yml` is manual-trigger (`workflow_dispatch` only) rather than
push-triggered. No local gate caught it. This script closes that gap:
it asks GitHub (read-only, via `gh`) for the newest `ci.yml` run whose head
commit's TREE matches the current `HEAD`'s tree, and fails loudly — with
the exact dispatch command — if that run isn't a green, completed run.

Comparing trees rather than commit SHAs means a run from a *different* but
content-identical commit (e.g. after a rebase, or a merge commit with the
same tree as its parent) still counts: what matters is "has this exact
source tree been proven green", not "has this exact commit been run".

Never dispatches a workflow itself — read-only `gh` calls only.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

WORKFLOW = "ci.yml"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_SLUG = "cajigaslab/nstat-python"
# How many of the most recent ci.yml runs to inspect before giving up.
MAX_RUNS_CHECKED = 30


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, **kwargs)


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


def _current_branch_for_dispatch() -> str:
    """Best-effort branch name to suggest in the dispatch command.

    Falls back to the repo's default branch name when HEAD is detached
    (e.g. a CI checkout, or a worktree checked out at a bare commit).
    """
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if branch and branch != "HEAD":
        return branch
    try:
        head_ref = _git("symbolic-ref", "--short", "-q", "refs/remotes/origin/HEAD")
        return head_ref.rsplit("/", 1)[-1]
    except subprocess.CalledProcessError:
        return "main"


def _repo_slug() -> str:
    try:
        url = _git("remote", "get-url", "origin")
    except subprocess.CalledProcessError:
        return DEFAULT_REPO_SLUG
    url = url.removesuffix(".git")
    for marker in ("github.com/", "github.com:"):
        if marker in url:
            return url.split(marker, 1)[1]
    return DEFAULT_REPO_SLUG


def _dispatch_hint(branch: str) -> str:
    return f"gh workflow run {WORKFLOW} --ref {branch}"


def check(*, tree_sha: str, branch_for_hint: str, repo_slug: str) -> tuple[bool, str]:
    """Return (ok, message)."""
    if shutil.which("gh") is None:
        return False, (
            "`gh` (GitHub CLI) is not installed or not on PATH — cannot verify "
            f"CI status. Install it, then run:\n  {_dispatch_hint(branch_for_hint)}"
        )

    auth = _run(["gh", "auth", "status"])
    if auth.returncode != 0:
        return False, (
            "`gh` is not authenticated — cannot verify CI status. Run "
            "`gh auth login`, then:\n  " + _dispatch_hint(branch_for_hint)
        )

    runs = _run(
        [
            "gh",
            "run",
            "list",
            "--repo",
            repo_slug,
            "--workflow",
            WORKFLOW,
            "--limit",
            str(MAX_RUNS_CHECKED),
            "--json",
            "databaseId,headSha,status,conclusion,createdAt",
        ]
    )
    if runs.returncode != 0:
        return False, (
            f"`gh run list` failed (repo={repo_slug}): {runs.stderr.strip() or runs.stdout.strip()}\n"
            f"Verify repo access, then:\n  {_dispatch_hint(branch_for_hint)}"
        )

    try:
        run_rows = json.loads(runs.stdout or "[]")
    except json.JSONDecodeError:
        return False, (
            f"Could not parse `gh run list` output for {WORKFLOW}.\n"
            f"  {_dispatch_hint(branch_for_hint)}"
        )

    # Newest first (gh already sorts this way, but don't rely on it).
    run_rows.sort(key=lambda r: r.get("createdAt", ""), reverse=True)

    for row in run_rows:
        head_sha = row.get("headSha")
        if not head_sha:
            continue
        commit = _run(
            [
                "gh",
                "api",
                f"repos/{repo_slug}/commits/{head_sha}",
                "--jq",
                ".commit.tree.sha",
            ]
        )
        if commit.returncode != 0:
            continue  # commit may have been force-pushed away; skip it
        run_tree_sha = commit.stdout.strip()
        if run_tree_sha != tree_sha:
            continue

        status = row.get("status")
        conclusion = row.get("conclusion")
        if status != "completed":
            return False, (
                f"Newest {WORKFLOW} run for this tree (headSha {head_sha[:10]}, "
                f"run {row.get('databaseId')}) is still '{status}' — wait for it "
                "to finish, then re-check. Do not dispatch another run for the "
                "same tree while one is in flight."
            )
        if conclusion != "success":
            return False, (
                f"Newest {WORKFLOW} run for this tree (headSha {head_sha[:10]}, "
                f"run {row.get('databaseId')}) concluded '{conclusion}', not "
                f"'success'. Fix it, then:\n  {_dispatch_hint(branch_for_hint)}"
            )
        return True, (
            f"{WORKFLOW} is green for this tree (run {row.get('databaseId')}, "
            f"headSha {head_sha[:10]})."
        )

    return False, (
        f"No {WORKFLOW} run among the last {MAX_RUNS_CHECKED} matches this tree "
        f"({tree_sha[:10]}). Dispatch one:\n  {_dispatch_hint(branch_for_hint)}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tree-sha",
        default=None,
        help="Override the tree SHA to check (default: HEAD^{tree} of this repo).",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Branch name to use in the suggested dispatch command (default: current branch).",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="owner/repo slug (default: parsed from the `origin` remote).",
    )
    args = parser.parse_args(argv)

    tree_sha = args.tree_sha or _git("rev-parse", "HEAD^{tree}")
    branch = args.branch or _current_branch_for_dispatch()
    repo_slug = args.repo or _repo_slug()

    ok, message = check(tree_sha=tree_sha, branch_for_hint=branch, repo_slug=repo_slug)
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
