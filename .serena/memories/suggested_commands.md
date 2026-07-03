# nstat-python — Suggested Commands

All commands below are Makefile targets (`make help` lists them); each is a
thin wrapper, no hidden logic. Override interpreter with `PY=python3.12 make ...`.

## Install
- `make install` → `pip install -e ".[dev]"`.
- `nstat-install --download-example-data always` fetches the ~150 MB figshare
  paper dataset (not in git). `NSTAT_OFFLINE=1` forces offline mode (fail-fast
  instead of hanging on network).

## Test
- `make test` → `pytest -q --ignore=tests/test_paper_example_scripts.py`
  (~25s full suite, excludes slow paper-example subprocess tests).
- `make test-smoke` / `make test-fast` → targeted `-k` selection
  (`test_repo_layout or test_api_surface or test_release_check or test_version_sync`),
  ~1s, for a quick structural sanity check.
- `make test-datasets` → `pytest -q tests/test_datasets.py` (figshare manifest
  hash checks only).
- `make test-no-paper` → same as `make test` (alias).
- Full paper-example subprocess tests (`tests/test_paper_example_scripts.py`)
  are deliberately excluded from `make test` — run explicitly when touching
  `examples/paper/`.

## Lint / format / typecheck (non-enforcing)
- `make format` / `make lint` / `make typecheck` → run ruff/ruff/mypy **only if
  installed on PATH**, else print "not installed; skipping" and exit 0. Do not
  rely on these to catch anything in CI — CI does not gate on them.

## Docs
- `make docs` → `sphinx -b html docs docs/_build/html`.
- `make docs-strict` → two-pass build, second pass with `-W` (warnings as
  errors) — matches CI's docs-build job. First pass populates
  `docs/_autosummary` so the strict pass doesn't trip on cold-start warnings.
- `make docs-open` → build + open in browser (macOS `open` / Linux `xdg-open`).

## Freshness / drift gates (mirror CI, run before pushing surface changes)
- `make readme-check` → `tools/check_readme_links.py` (README intra-repo
  links/images/code-snippet imports resolve).
- `make helpfile-check` → `tools/check_helpfile_freshness.py` (every
  `nstat.__all__` symbol documented in `AGENT_GUIDE.md` + `ClassDefinitions.md`).
- `make freshness-check` → both of the above.
- `make drift-check` → regenerates gallery + parity-report artifacts, then
  `git diff --exit-code` against committed state (no commit made) — catches
  the "forgot to `make regen`" class of CI failure.
- `make ci-local` → `freshness-check test docs-strict drift-check`, the
  deterministic PR gates that don't need the figshare dataset or JAX extras.
  Notebook-execution / figure-regen / extras-{dynamax,clusterless} jobs still
  only run on GitHub Actions.

## Regenerate artifacts (after touching manifests/figures — CI diffs these)
- `make regen` → gallery + extras-gallery + parity-report + notebook-fidelity
  (the standard set CI drift-checks).
- `make regen-figures` — ~30 min, needs figshare dataset, regenerates every
  paper-example PNG.
- `make regen-visual-parity` — needs a sibling MATLAB checkout at `../nSTAT`.

## Release
- `make version-check` → `pytest -q tests/test_version_sync.py` (pyproject.toml
  / CITATION.cff / RELEASE_NOTES.md versions agree).
- `make sanity` → quick import + entry-point smoke check (no pytest).
- `make release-check` → `version-check freshness-check test docs-strict regen`
  — full pre-release gauntlet.

## Performance parity
- `make perf-check` (~30s, Python only) / `make perf-check-full` (~5-10 min,
  10 runs/side, needs MATLAB) / `make perf-check-capture` (re-baselines
  `parity/performance_baseline.yml`).

## macOS/Darwin notes
- No Darwin-specific command forms needed; `make docs-open` already branches
  on `open` (macOS) vs `xdg-open` (Linux) inside the Makefile.
