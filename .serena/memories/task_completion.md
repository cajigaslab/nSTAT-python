# nstat-python — Task Completion Gate

Minimum gate before calling a change "done" in this repo:

1. `make test` (pytest -q, excludes slow paper-example subprocess tests, ~25s).
   For a quick sanity pass mid-work use `make test-smoke` (~1s), but it is not
   a substitute for the full run before finishing.
2. If the change touched `examples/paper/manifest.yml`, any `parity/*.yml`,
   or any committed figure under `docs/figures/`: run `make regen` (or the
   specific `regen-*` target) and commit the regenerated artifacts — CI does
   `git diff --exit-code` against committed state and fails on drift.
   `make drift-check` reproduces this locally without committing.
3. If the change touched `README.md`, `docs/`, `examples/`, `notebooks/`,
   `nstat.__all__`, `pyproject.toml`, or `CITATION.cff`: run
   `make freshness-check` (= `readme-check` + `helpfile-check`).
4. If the change added/removed a public symbol: it needs an entry in
   `nstat/__init__.py` `__all__`, a row in `tests/test_api_surface.py`, a
   recipe in `AGENT_GUIDE.md`, and (for classes) `docs/ClassDefinitions.md` —
   `helpfile-check` enforces the doc side.
5. If the change touched `nstat/` public API or docs, run `make docs-strict`
   (matches CI's `-W` docs-build job) before considering docs changes safe.
6. For the closest full local mirror of the deterministic CI PR gates (no
   figshare dataset or JAX needed), run `make ci-local`
   (= freshness-check + test + docs-strict + drift-check).
7. Never treat `ruff`/`mypy` output as a completion signal — `make
   format`/`lint`/`typecheck` are no-ops unless those tools happen to be
   installed; CI does not gate on them.
8. A failing `tests/parity/fixtures/matlab_gold/*.mat` comparison always means
   the code change is wrong — fix the code, never the gold fixture.
9. Full release gate (only when actually cutting a release):
   `make release-check` (version-check + freshness-check + test + docs-strict
   + regen).

Post-merge (only after merging something touching `nstat/**`, `docs/`,
`README.md`, `AGENT_GUIDE.md`, `CITATION.cff`, or `pyproject.toml`): verify
`deploy-docs.yml` actually rebuilt GitHub Pages green for the merge commit —
local `make docs-strict` only proves the build compiles, not that Pages
redeployed. Check via `gh run list --workflow=deploy-docs.yml --limit 1
--json conclusion,headSha,createdAt` (exact recipe in the repo's own
`CLAUDE.md`, not duplicated here since it's git-ignored/local).
