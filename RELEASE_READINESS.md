# Release Readiness — next-version plan

> Working / pre-tag doc. Tracks what's queued for the **next** release
> and a few one-time chores that survive across releases. When the
> maintainer is ready to cut a tag, copy the relevant "What's in scope"
> section into `RELEASE_NOTES.md`, bump the four version-bearing files,
> and run `make release-check`.
>
> _Last refresh: 2026-06-10 — post v0.4.5 (documentation hygiene, theme
> polish, CI billing conservation, onboarding additions)._

## PyPI publishing — operational

`v0.4.1` shipped the OIDC Trusted-Publisher pattern
(`.github/workflows/publish.yml`), and every release since (v0.4.2 →
v0.4.5) has auto-published on tag push. The one-time PyPI account /
Trusted-Publisher / `pypi` GitHub-Environment configuration is **done** —
no further setup needed. The workflow asserts `git tag ==
pyproject.toml version` at build time, so a forgotten version bump
cannot ship.

## Next release — v0.4.6 (small/medium docs follow-up; ship when ready)

The natural follow-up to v0.4.5's hygiene work. Likely scope (in priority
order):

- **Glossary first-mention cross-linking** across the 14 concepts pages.
  Each first occurrence of a defined term (CIF, KS test, PPAF, multitaper,
  etc.) becomes `[term](glossary.md#term)`. Bulk-mechanical, high value
  for students.
- **`notebooks/00_getting_started.ipynb`** — a runnable end-to-end notebook
  mirroring `intro.html`. Closes the gap between the API-tour HTML and the
  29 reference notebooks.
- **Notebook ↔ concepts crosswalk** — a table mapping each concepts page
  to the notebooks/tutorials that demonstrate it. Either in `docs/Examples.md`
  or a new `docs/notebooks.md`. 29 notebooks need a goal-oriented index.
- **`notebooks/` index page** if the crosswalk grows large enough.

## Future release — v0.5.0 (queued; ship when upstream is ready)

The Tier 3 extras need upstream code + license verification before
shipping. Each lives in `nstat.extras` with its own opt-dep group.

- **Tier 3.1 — CLDS** (Geadah NeurIPS 2025): condition-dependent linear
  dynamics; builds on the EM bridge. Needs upstream-code availability
  + LICENSE verification.
- **Tier 3.2 — NPNR**: Bayesian nonparametric (non-)renewal point
  processes (variational GP). Complements the marked time-rescaling test
  from Tier 1.1 by directly characterizing the rescaled-ISI structure
  the test only checks.
- **Tier 3.3 — GP-based GLM coupling/history filters**: nonparametric
  filter inference.

See [`parity/methods_roadmap.md`](parity/methods_roadmap.md) for the
full rationale and references.

## Standing release checklist

When ready to tag, follow the same flow used since v0.4.1:

1. Merge queued PRs to `main` in dependency order.
2. Bump versions to the new value in **all four files**:
   - `pyproject.toml` → `version = "X.Y.Z"`
   - `CITATION.cff` → `version: "X.Y.Z"` and `date-released: "YYYY-MM-DD"`
   - `docs/conf.py` → `release = "X.Y.Z"`
   - `AGENT_GUIDE.md` → `Package version: X.Y.Z.` and `Updated:` line.
3. Add a `## vX.Y.Z — <date>` section at the top of `RELEASE_NOTES.md`.
4. Write `docs/changes/<date>-vX.Y.Z-<slug>.html` (per-iteration change page,
   light theme — copy a recent neighbor as the template).
5. Add a top-of-list entry to `docs/changes/whats_new.html` pointing at
   the new change page.
6. Run `make release-check` (= `version-check + freshness-check + test +
   docs-strict + regen`). Locally a few unrelated env-mismatch failures
   may show — fine to tag based on green CI.
7. Open PR, merge to `main`.
8. `git tag vX.Y.Z <merge-sha>` and `git push origin vX.Y.Z`. The
   `publish.yml` workflow auto-uploads to PyPI. If the GitHub
   environment requires approval, click "Approve" on the Actions run.
9. After merge, `deploy-docs.yml` is **manual** (as of v0.4.5 CI
   conservation): trigger it from the Actions tab so the new release's
   docs go live on GitHub Pages.
10. Verify: `pip index versions nstat-toolbox`, plus `curl -sI` against
    a representative new page on the live docs site.

## Out of scope / tracked elsewhere

- **Multi-restart selection + EM hardening** (Tier 0.3 + 0.3 follow-ups):
  shipped in v0.4.0 / v0.4.2.
- **Friendly intro page**: live at `intro.html` (v0.4.0 / refreshed
  v0.4.5 with concepts + tutorials cards).
- **Concepts learning track + tutorials**: shipped in v0.4.4.
- **Internal-curriculum scrub**: shipped in v0.4.5.
- **GitHub Actions Node 20 → Node 24 bumps**: shipped in v0.4.5
  follow-up (`actions/checkout@v6`, `setup-python@v6`,
  `upload-artifact@v7`, etc.). Removes the deprecation warning that
  GitHub will enforce on 2026-09-16.
- **`_lazywhere` local-test failures**: resolved in v0.4.1 by pinning
  `statsmodels>=0.15`.
