# Release Readiness — next-version plan

> Working / pre-tag doc.  Tracks what's queued for the **next** release
> and a few one-time chores that survive across releases.  When the
> maintainer is ready to cut a tag, copy the relevant "What's in scope"
> section into `RELEASE_NOTES.md`, bump the four version-bearing files,
> and run `make release-check`.
>
> _Last refresh: 2026-05-31 — post v0.4.1 polish._

## One-time PyPI publishing setup (required before first PyPI release)

`v0.4.1` is the first release configured for PyPI publication via the
new `.github/workflows/publish.yml` (OIDC Trusted Publisher pattern —
no API tokens stored in the repo).  Before the next tag push will
actually publish, the maintainer needs to perform three one-time
clicks-only chores:

1. **Reserve the project name on PyPI** (creates an empty project).
   - Visit https://pypi.org/account/register/ if you don't already
     have a PyPI account.
   - The package name `nstat-toolbox` is what `pyproject.toml` declares.
     PyPI will auto-create the project under the uploader's account on
     the first successful publish — but only if the name isn't already
     squatted.  If `pip install nstat-toolbox` returns 404 the name is
     free; reserve it via the standard PyPI account flow before tagging
     to avoid a race.

2. **Configure GitHub as a Trusted Publisher** at
   https://pypi.org/manage/project/nstat-toolbox/settings/publishing/ —
   add a publisher with these exact fields:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `nstat-toolbox` |
   | Owner | `cajigaslab` |
   | Repository name | `nSTAT-python` |
   | Workflow filename | `publish.yml` |
   | Environment name | `pypi` |

3. **(Recommended) Create the `pypi` GitHub Environment** in this
   repo's Settings → Environments → New environment → `pypi`.  Add a
   required-reviewer rule so each publish requires an explicit
   approval click in the Actions UI before it fires.  Belt-and-suspenders
   against an accidental tag push.

Once those three are in place, any `git push` of a `v*` tag whose
version matches `pyproject.toml` will trigger:

- Build sdist + wheel (Python 3.12).
- `twine check` the artifacts.
- Upload to PyPI via OIDC (no API token).

The workflow asserts `git tag == pyproject.toml version` at build time
so a forgotten version bump cannot ship.

## Next release — v0.5.0 (queued; ship when ready)

Likely scope candidates (in priority order; pick what makes sense
based on user-facing demand):

- **Tier 0.3 follow-ups** (deferred from #119): data-driven PP_EM init
  from the log-empirical rate, plus a ridge on the `A` / `Q` M-step.
  Improves the weak-observability collapse documented in
  `docs/extras/em_dynamax.md`.  Estimated thin-to-moderate.
- **Tier 3.1 — CLDS** (Geadah NeurIPS 2025): condition-dependent
  linear dynamics; builds on the EM bridge.  Needs upstream-code
  availability + LICENSE verification.
- **Tier 3.2 — NPNR**: Bayesian nonparametric (non-)renewal point
  processes (variational GP).  Complements Tier 1.1's GOF tests by
  directly characterizing the rescaled-ISI structure they only test.
- **Tier 3.3 — GP-based GLM coupling/history filters**: nonparametric
  filter inference.  All Tier-3 items live in `extras`; each needs an
  opt-dep group + LICENSE check.

See [`parity/methods_roadmap.md`](parity/methods_roadmap.md) for the
full rationale and references.

## Standing release checklist

When ready to tag, follow the same flow used for v0.4.0 / v0.4.1:

1. Merge queued PRs to `main` in dependency order.
2. Bump versions to the new value in **all four files**:
   - `pyproject.toml` → `version = "X.Y.Z"`
   - `CITATION.cff` → `version: "X.Y.Z"`
   - `docs/conf.py` → `release = "X.Y.Z"`
   - `AGENT_GUIDE.md` → `Package version: X.Y.Z.`
3. Copy the "What's in scope" section into `RELEASE_NOTES.md` as the
   published `## vX.Y.Z — <date>` entry.
4. Run `make release-check` (= `version-check + freshness-check + test
   + docs-strict + regen`).  In CI everything passes; locally the few
   unrelated env-mismatch failures (if any) will show — fine to tag
   based on green CI.
5. `git tag vX.Y.Z` + push.  The `publish.yml` workflow auto-uploads
   to PyPI.
6. Create a GitHub Release at the tag (UI or `gh release create`).

## Out of scope / tracked elsewhere

- **Multi-restart selection** for PP_EM is shipped (Tier 0.3, PR #119).
- **Friendly intro page** is live at `intro.html` (PR #122).
- **`_lazywhere` local-test failures** are resolved in v0.4.1 by
  pinning `statsmodels>=0.15`.
