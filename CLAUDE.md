# CLAUDE.md — repository playbook for AI coding agents

> Read this first when working in `nstat-python`.  It is the *repo
> maintenance* counterpart to [AGENT_GUIDE.md](AGENT_GUIDE.md) — that file
> teaches an agent how to *use* the toolbox; this file teaches an agent
> how to *modify* the toolbox and its surrounding artifacts.
>
> Last updated: 2026-05-19.  Package version: 0.3.0.

---

## Canonical references

When you need to refer to the upstream sources, **use these URLs verbatim**:

| Resource | Canonical URL |
|---|---|
| Original paper (Cajigas, Malik & Brown 2012) | https://pubmed.ncbi.nlm.nih.gov/22981419/ |
| Paper DOI | https://doi.org/10.1016/j.jneumeth.2012.08.009 |
| MATLAB toolbox (reference implementation) | https://github.com/cajigaslab/nSTAT |
| Python toolbox (this repository) | https://github.com/cajigaslab/nSTAT-python |
| Python docs (GitHub Pages) | https://cajigaslab.github.io/nSTAT-python/ |
| PyPI package | https://pypi.org/project/nstat-toolbox/ |
| Paper dataset (figshare) | https://doi.org/10.6084/m9.figshare.4834640.v3 |

### Naming convention

- **The toolbox is `nSTAT`** (CamelCase) — as in the 2012 paper and the
  GitHub repository names.  Both repos use the CamelCase form:
  `cajigaslab/nSTAT` (MATLAB) and `cajigaslab/nSTAT-python` (Python).
- **The Python package** is `nstat` (lowercase) — the importable module
  name (`import nstat`), the PyPI distribution name (`nstat-toolbox`,
  installed via `pip install nstat-toolbox`), and the CLI commands
  (`nstat-install`, `nstat-paper-examples`).
- **Rule of thumb:** use `nSTAT` (CamelCase) when referring to the
  toolbox by name, the paper, or the GitHub repositories.  Use `nstat`
  (lowercase) only when writing Python code, import statements,
  package-distribution names, or CLI invocations.

---

## TL;DR — what this repo is

`nstat-python` is the Python port of the MATLAB **nSTAT** neural spike
train analysis toolbox.  Public on GitHub at
[cajigaslab/nSTAT-python](https://github.com/cajigaslab/nSTAT-python).
Distributed on PyPI as `nstat-toolbox`.  Docs at
https://cajigaslab.github.io/nSTAT-python/.

Repository facts (from `gh repo view` + GitHub landing page):

| Field | Value |
|---|---|
| Visibility | public |
| Default branch | `main` |
| Latest release | v0.3.0 ("Full Matlab Parity", 2026-03-11) |
| License | GPL-2.0 (declared in README; `licenseInfo` field empty in GitHub metadata — fix later if needed) |
| Issues | enabled — currently zero open |
| Wiki | enabled (but appears unused) |
| Discussions | disabled |
| `delete_branch_on_merge` | `false` (merged branches stay until manually deleted) |
| Docs site | GitHub Pages → `docs/_build/html`, served by `deploy-docs.yml` |
| PyPI name | `nstat-toolbox` (not `nstat-python`) |

---

## Independence from the MATLAB repo

This is a **hard rule, set by the maintainer**:

> *"Keep the nSTAT MATLAB repo and the nstat-python repo completely
> independent.  The only prior dependence is the Simulink model used in
> CIF simulation by thinning."*

What that means in practice:

- Do **not** add cross-repo runtime imports, GitHub URLs into
  `cajigaslab/nSTAT` (the MATLAB repo) in manifests, or new
  parity-tracking fields that point at MATLAB source paths.
- The README's narrative attribution of the MATLAB origin is fine —
  that's history, not coupling.
- The `parity/` directory and `compat/matlab/` subpackage *are*
  allowed because they record *audit results* and *MATLAB-style import
  aliases* within this package — they don't depend on the MATLAB repo
  at runtime.
- For correctness verification, you may consult a local MATLAB checkout
  (if available) **as a reference** — but never cite it in committed
  code or docs.

The local MATLAB checkout at
`/Users/iahncajigas/Dropbox/Research/Matlab/nSTAT_currentRelease_Local`
historically contained Dropbox cloud-sync stubs (0-byte files); always
verify file sizes before relying on it.  The canonical authority for
parity is the MATLAB-side `.mat` gold fixtures in
`tests/parity/fixtures/matlab_gold/`.

---

## Repository layout

```
nstat/                       50+ modules, ~24 kLOC — the package
  __init__.py                  public API surface (`__all__`, 53 symbols)
  core.py                      SignalObj + Covariate classes
  _spike_train_impl.py         nspikeTrain class (extracted from core.py in v0.3.1)
  trial.py                     CovariateCollection, SpikeTrainCollection, Trial
  _trial_config_impl.py        TrialConfig + ConfigCollection (extracted from trial.py)
  analysis.py                  Analysis, GLMFit (returns GLMFitResult dataclass), psth
  cif.py                       Conditional intensity functions (sympy-backed)
  fit.py                       FitResult + FitResSummary
  decoding_algorithms.py       PPAF/PPHF/Kalman/EM static methods
  data_manager.py              figshare dataset fetch + NSTAT_OFFLINE
  plot_style.py                modern/legacy plot policy
  compat/matlab/               MATLAB-style import aliases
  {SignalObj,nspikeTrain,...}.py   thin re-export shims for MATLAB-style imports

examples/
  paper/                       5 canonical paper-example scripts (Cajigas 2012)
    example01..05_*.py
    manifest.yml               -> drives docs/figures/manifest.json
    regenerate_all_figures.py  -> CI calls this; writes to docs/figures/
  readme_examples/             4 short scripts (use nstat.compat.matlab)
  basic_data_workflow.py
  fit_poisson_glm.py
  simulate_population_psth.py

docs/                          Sphinx + MyST
  index.rst, api.rst
  PaperOverview.md, ClassDefinitions.md, Examples.md
  figures/example0{1..5}/      CANONICAL paper-figure tree
  paper_examples.md            auto-generated (drift-checked in CI)

notebooks/                     30+ Jupyter notebooks; many are MATLAB-help ports
                               9 are tracker-only stubs (banner-marked)

parity/                        MATLAB↔Python audit
  manifest.yml, report.md
  class_fidelity.yml, function_contracts.yml, class_contracts.yml
  notebook_fidelity.yml, simulink_fidelity.yml
  README.md                    how to regenerate

tests/                         48 files, 268 tests, all green on main
  parity/fixtures/matlab_gold/   exact-numerical .mat fixtures

tools/
  paper_examples/build_gallery.py    docs/paper_examples.md regenerator
  parity/build_report.py             parity/report.md regenerator
  parity/build_notebook_fidelity_audit.py
  notebooks/                         notebook execution helpers
  release/                           release-gate scripts

.github/workflows/             5 CI workflows (see § "CI gates" below)
```

Files at the repo root worth knowing about:

- [README.md](README.md) — user-facing landing page (PyPI install, paper
  examples table, paper citation).
- [AGENT_GUIDE.md](AGENT_GUIDE.md) — *toolbox-usage* guide for AI agents.
- [AUDIT_REPORT.md](AUDIT_REPORT.md) — frozen 2026-03 cross-toolbox audit
  result (bug catalog + missing methods).  Update only on a real new
  audit pass.
- [PORTING_MAP.md](PORTING_MAP.md) — frozen method-by-method porting map.
- [CITATION.cff](CITATION.cff) — software-citation metadata
  (`version: 0.3.0`, `date-released: 2026-05-19`).
- [pyproject.toml](pyproject.toml) — build/runtime/dev deps; entry-point
  scripts `nstat-paper-examples`, `nstat-install`.

---

## Commands cheat sheet

### Install / setup

```bash
python -m pip install -e .[dev]          # editable install + dev extras
nstat-install --download-example-data always   # ~150 MB figshare dataset
```

For offline / air-gapped: `export NSTAT_OFFLINE=1` and have data
pre-staged at `$NSTAT_DATA_DIR`.

### Run tests

```bash
pytest -q                                  # full suite (~3 min, 263 pass + 5 skip)
pytest -q -k "test_repo_layout or test_api_surface"   # smoke (~1 sec)
pytest -q tests/test_datasets.py           # data-integrity only
pytest -q --ignore=tests/test_paper_example_scripts.py   # skip slow paper-example subproc tests
```

### Regenerate artifacts (CI drift-checks these — keep them in sync)

```bash
python tools/paper_examples/build_gallery.py        # docs/paper_examples.md, docs/figures/manifest.json
python tools/parity/build_report.py                 # parity/report.md
python tools/parity/build_notebook_fidelity_audit.py   # parity/notebook_fidelity.yml
python examples/paper/regenerate_all_figures.py     # docs/figures/example0{1..5}/*.png
```

### Build docs

```bash
python -m sphinx -W -b html docs docs/_build/html
open docs/_build/html/index.html                    # macOS preview
```

The `-W` flag turns warnings into errors (matches CI `docs-build` job).

### Install / sanity checks

```bash
python -m nstat.install --download-example-data never --no-rebuild-doc-search
python -c "import nstat; print(len(nstat.__all__))"   # should be 53
```

---

## CI gates (.github/workflows/)

When you push to a PR, these gates must pass.  Knowing them prevents
"green locally, red in CI" surprises.

| Workflow | Triggered on | What it checks |
|---|---|---|
| `ci.yml` | PR + push to `main` | unit tests (py 3.11 and 3.12), gallery-artifact drift, parity-report drift, sphinx docs build with `-W`, installer smoke (`--download-example-data never`), data-integrity, notebook smoke |
| `deploy-docs.yml` | push to `main` | Builds Sphinx site and publishes to GitHub Pages |
| `notebook-full-fidelity.yml` | PRs touching notebooks/parity, weekly cron (Sat 07:15 UTC) | Full notebook execution + fidelity audit |
| `performance-parity.yml` | PR + daily cron 06:30 UTC | Performance stability and parity tests |

**Drift checks** are easy to trip — if you change `examples/paper/manifest.yml`,
`parity/*.yml`, or any committed figure under `docs/figures/`, you must
regenerate the dependent artifacts (see commands above) *and commit
them*.  The drift jobs `git diff --exit-code` against the committed
state.

---

## Conventions

### Naming

- **Class names follow MATLAB conventions**: `nspikeTrain` (lowercase
  `n`), `nstColl` (lowercase `n`, both caps inner), `CovColl`,
  `TrialConfig`, `SignalObj`.  Do not "Pythonize" these.
- **Method names also follow MATLAB**: `GLMFit`, `computeKSStats`,
  `setMinTime`, `findNearestTimeIndex`.  Python-style aliases exist
  in some cases (`get_nst`, `add_config`) but the MATLAB-style name is
  always the public API surface in `__all__`.
- **Module-level shims** like `nstat/SignalObj.py`, `nstat/nspikeTrain.py`
  exist for MATLAB-style import paths
  (`from nstat.SignalObj import SignalObj`).  Don't delete them; tests
  in `tests/test_api_surface.py` enforce their presence.
- **Private internal modules** use a leading underscore:
  `nstat/_spike_train_impl.py`, `nstat/_trial_config_impl.py`.
  These are implementation details — re-exported from `core.py`/`trial.py`.

### Time and units

- All time vectors are in **seconds**.
- All sample rates in **Hz**.
- Spike times are stored sorted ascending on construction.
- For MATLAB `start:step:stop` parity, prefer the `_matlab_colon` pattern
  (compute element count from `floor((stop-start)/step)+1`) over
  `np.arange` to avoid float-error length drift.

### Imports and module structure

- `from __future__ import annotations` everywhere — no exceptions for
  new code.
- `nstat.core` keeps `SignalObj` + `Covariate`; `_spike_train_impl.py`
  has `nspikeTrain`.  All three are re-exported by `nstat.core` for
  back-compat.
- `nstat.trial` keeps `CovariateCollection` / `SpikeTrainCollection` /
  `Trial`; `_trial_config_impl.py` has `TrialConfig` / `ConfigCollection`.
- Don't add new `matplotlib.use("Agg")` at module top.  The user's
  backend choice must survive `import nstat`.  If a method genuinely
  needs Agg (e.g. a CI-only path), set the backend inside the method.
- **Lazy-import** SciPy / scikit-image / heavy deps inside the functions
  that need them; module-level imports should be cheap.

### Tests / parity

- The `tests/parity/fixtures/matlab_gold/*.mat` files are the ground
  truth.  If a "bug fix" makes one of these fail, **the fix is wrong** —
  the Python port is intentionally faithful to MATLAB idiosyncrasies
  (e.g. `numSpikesPerBurst + 1.0` in `nspikeTrain.computeStatistics`,
  the hybrid `logLL` in `Analysis.GLMFit`).  Revert and document the
  MATLAB convention with a code comment.
- The auto-generated artifacts (`docs/paper_examples.md`,
  `parity/report.md`, `parity/notebook_fidelity.yml`,
  `docs/figures/manifest.json`) are checked into git but regenerated
  by tools.  **Always regenerate them via the script** rather than
  hand-editing.
- New public-API symbols need an entry in `nstat/__init__.py` `__all__`
  *and* a row in `tests/test_api_surface.py`.

### Code style

- Type hints: `float | None` (not `float = None`).  `Optional[...]` is
  fine but the `X | None` form is preferred.
- Docstrings: NumPy style with Parameters / Returns sections, plus a
  brief Matlab cross-reference for paper-paper methods, e.g.
  `"""Compute KS statistics (Matlab ``FitResult.computeKSStats``)."""`.
- The repo does **not** enforce ruff / black / mypy in CI.  Be
  conservative: don't reformat untouched lines.  Match local style.

---

## Branch and commit conventions

From recent PR history (`git log --oneline -30 | grep "Merge pull request"`):

- Branches are named like `fix/<topic>` or `feat/<topic>`:
  `fix/example02-fig01-xlim`, `fix/figure-visual-parity`,
  `fix/deploy-github-pages`, `fix/example05-figures-matlab-parity`.
- Commits are short imperative summaries:
  `"Fix figure visual parity with MATLAB toolbox"`,
  `"Add GitHub Pages deployment workflow with sphinx_rtd_theme"`.
- PRs are squash- or merge-committed via the GitHub UI (merge commits
  with `Merge pull request #NN from cajigaslab/<branch>` are visible).
- `delete_branch_on_merge: false` — merged branches stay; clean up
  manually with `git push origin --delete <branch>` after release if
  desired.

For agent-authored work, **follow the same pattern**: branch
`fix/<area>` or `feat/<area>`, descriptive imperative subject, body
explains *why*.

---

## Common workflows for an agent

### "Add a new public API"

1. Implement in the appropriate module (e.g. a new simulator in
   `nstat/simulation.py`).
2. Add to that module's `__all__`.
3. Add `from .simulation import <name>` in `nstat/__init__.py`.
4. Add to `__all__` in `nstat/__init__.py`.
5. Add a row to `tests/test_api_surface.py`.
6. Add a usage recipe to [AGENT_GUIDE.md](AGENT_GUIDE.md).
7. Update [docs/api.rst](docs/api.rst) — currently hand-maintained.

### "Fix a bug in a class"

1. Find the canonical class location (after the v0.3.1 splits, that's
   either `nstat/core.py`, `nstat/_spike_train_impl.py`,
   `nstat/trial.py`, or `nstat/_trial_config_impl.py`).
2. Check the gold-fixture tests (`tests/test_matlab_gold_fixtures.py`,
   `tests/parity/fixtures/matlab_gold/*.mat`) — if your "fix" changes
   a value any fixture expects, **the bug is in your reading**, not the
   code.
3. Write the smallest possible diff.  Add a code comment explaining
   *why* if the change looks counter-intuitive.
4. Run the full test suite — not just the touched area.

### "Add a paper example or regenerate figures"

1. Paper-example scripts live in `examples/paper/example0N_*.py`.
2. Each script accepts `--export-figures`, `--export-dir`, `--no-display`,
   `--plot-style {modern,legacy}`.
3. The canonical figure tree is `docs/figures/example0N/`.
4. `examples/paper/figures/` is a *deprecated* byte-identical mirror;
   see [examples/paper/figures/README.md](examples/paper/figures/README.md).
5. After editing a script, run
   `python examples/paper/regenerate_all_figures.py` and commit the
   updated PNGs.  CI will fail otherwise (gallery-artifact drift check).
6. The notebook-stub conventions are documented in
   [AGENT_GUIDE.md](AGENT_GUIDE.md) §5.4.

### "Touch an example dataset path"

The figshare paper dataset (~150 MB) is *not* in git.
`nstat.data_manager.ensure_example_data(download=True)` fetches it on
demand; `NSTAT_OFFLINE=1` forces offline mode.  CI sets
`NSTAT_DATA_DIR` to a cache path.

---

## Things to **NOT** do

- ❌ Don't add `matlab_source` fields, or any path/URL pointing into
  `cajigaslab/nSTAT` (MATLAB repo) inside this repo's manifests or YAML
  files.  See "Independence" section above.
- ❌ Don't hand-edit `parity/report.md`, `parity/notebook_fidelity.yml`,
  `docs/paper_examples.md`, or `docs/figures/manifest.json` — they're
  regenerated.  CI will diff-fail.
- ❌ Don't break `tests/parity/fixtures/matlab_gold/*.mat` parity.  The
  MATLAB idiosyncrasies (`+1.0` on burst stats, hybrid `logLL`) are
  intentional.  Document, don't "fix".
- ❌ Don't add `matplotlib.use("Agg")` at module top in any `nstat/`
  file.
- ❌ Don't delete or rename the MATLAB-style shim files
  (`SignalObj.py`, `nspikeTrain.py`, etc.) — `tests/test_api_surface.py`
  imports from them directly.
- ❌ Don't reach into `nstat.core._<private>` from outside `nstat/`.
- ❌ Don't add new examples under `examples/nSTATPaperExamples/` — that
  manifest is legacy.  New paper examples go in `examples/paper/`.
- ❌ Don't introduce a hard dependency on a particular matplotlib
  version.  The repo currently supports matplotlib 3.8+ via the
  `_boxplot_labels()` shim pattern in `example04`.
- ❌ Don't add `np.random.rand`-style legacy random calls.  Use
  `np.random.default_rng(seed)` for any new randomness.

---

## Quick reference: where things live

| Question | Where to look |
|---|---|
| Where's the canonical class definition? | `git grep -l "^class <name>" nstat/` |
| What's exported? | `python -c "import nstat; print(sorted(nstat.__all__))"` |
| Did MATLAB do X this way? | `tests/parity/fixtures/matlab_gold/*.mat` (gold fixtures) > `parity/report.md` > `PORTING_MAP.md` > MATLAB repo |
| Is method X ported? | `PORTING_MAP.md` |
| Why does X look weird? | `AUDIT_REPORT.md` §1–2 (bug catalog) |
| Where do figures go? | `docs/figures/exampleNN/` |
| What CI will run? | `.github/workflows/` |
| Where's the data installer? | `nstat/install.py` + `nstat/data_manager.py` |
| What versions of Python? | `pyproject.toml` (>=3.10), CI matrix runs 3.11 and 3.12 |

---

## When in doubt

- The README at the root is the canonical user-facing introduction.
- [AGENT_GUIDE.md](AGENT_GUIDE.md) is the canonical *toolbox-usage*
  guide for AI agents (recipes, API surface, limitations).
- [AUDIT_REPORT.md](AUDIT_REPORT.md) is the canonical bug catalog.
- [PORTING_MAP.md](PORTING_MAP.md) is the canonical method-coverage
  reference.
- This file (CLAUDE.md) is the canonical *repo-maintenance* playbook.

If any of those four disagree, prefer the most specific one for the
task at hand, and flag the contradiction in your response.

Lab websites:
- Neuroscience Statistics Research Laboratory: https://www.neurostat.mit.edu
- RESToRe Lab: https://www.med.upenn.edu/cajigaslab/
