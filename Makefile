# Makefile — convenience entry points for nstat-python development.
#
# Mirrors the "Commands cheat sheet" in CLAUDE.md / AGENT_GUIDE.md so a
# contributor doesn't have to memorize the shell commands.  Each target
# is a thin wrapper around the canonical command — no logic.
#
# Conventions
# -----------
# - Use the editable install (``pip install -e .[dev]``) before running
#   targets that touch the package.
# - Slow targets (paper-figure regeneration, full notebook fidelity) are
#   tagged with ``.PHONY`` to make their cost explicit.
# - Most targets accept ``PY=python3.12`` etc. to override the Python
#   interpreter; default is whatever ``python`` resolves to.

PY        ?= python
PYTEST    ?= $(PY) -m pytest
PIP       ?= $(PY) -m pip
SPHINX    ?= $(PY) -m sphinx
REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null || pwd)

.PHONY: help install test test-smoke test-fast test-datasets test-no-paper \
        regen regen-gallery regen-parity regen-figures regen-notebook-fidelity \
        regen-notebook-galleries regen-visual-parity \
        docs docs-strict docs-open refresh-intersphinx-inv \
        diff-matlab readme-check helpfile-check freshness-check \
        format lint typecheck \
        version-check sanity clean release-check \
        ci-local drift-check

# --- help ------------------------------------------------------------

help:  ## Show this help.
	@echo "nstat-python development targets:"
	@echo
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Override the Python interpreter via PY=python3.12 (etc.)."

# --- install ---------------------------------------------------------

install:  ## Editable install with dev extras (sphinx + scikit-image + pytest).
	$(PIP) install -e ".[dev]"

# --- tests -----------------------------------------------------------

test:  ## Full pytest suite (~25s; excludes the slow paper-example subprocess tests).
	$(PYTEST) -q --ignore=tests/test_paper_example_scripts.py

test-fast: test-smoke  ## Alias for test-smoke.

test-smoke:  ## Fast structural tests only (~1s).
	$(PYTEST) -q -k "test_repo_layout or test_api_surface or test_release_check or test_version_sync"

test-datasets:  ## Dataset-integrity tests only (figshare manifest hash checks).
	$(PYTEST) -q tests/test_datasets.py

test-no-paper:  ## Full suite minus slow paper-example tests.
	$(PYTEST) -q --ignore=tests/test_paper_example_scripts.py

# --- regenerated artifacts ------------------------------------------

regen: regen-gallery regen-parity regen-notebook-fidelity  ## Regenerate all CI-drift-checked artifacts.

regen-gallery:  ## docs/paper_examples.md + docs/figures/manifest.json + README table.
	$(PY) tools/paper_examples/build_gallery.py

regen-parity:  ## parity/report.md from parity manifests.
	$(PY) tools/parity/build_report.py

regen-notebook-fidelity:  ## parity/notebook_fidelity.yml from tools/notebook_build/parity_notes.yml.
	$(PY) tools/parity/build_notebook_fidelity_audit.py

regen-figures:  ## ~30 min — regenerate every paper-example PNG (needs figshare dataset).
	@echo "Note: this requires the figshare paper dataset.  Set NSTAT_OFFLINE=1 to fail-fast."
	$(PY) examples/paper/regenerate_all_figures.py

regen-notebook-galleries:  ## Execute every notebook + rebuild docs/notebook_galleries/<topic>/README.md + PNGs.
	@echo "Note: executes notebooks via nbclient; needs the figshare paper dataset for full coverage."
	$(PY) tools/notebook_build/build_notebook_galleries.py --group full

regen-visual-parity:  ## SSIM scores for parity/visual_fidelity.yml entries (needs sibling MATLAB checkout).
	@echo "Note: requires the cajigaslab/nSTAT checkout at ../nSTAT for MATLAB PNG comparison."
	$(PY) tools/parity/build_visual_comparison.py

# --- docs ------------------------------------------------------------

docs:  ## Build the Sphinx site (writes to docs/_build/html).
	$(SPHINX) -b html docs docs/_build/html

docs-strict:  ## Build docs with -W (warnings as errors, matches CI).
	# Two-pass build: the first run populates docs/_autosummary so the
	# strict run doesn't trip on cold-start "stub file not found"
	# warnings.  Mirrors the CI docs-build job.
	$(SPHINX) -b html docs docs/_build/html
	$(SPHINX) -W -b html docs docs/_build/html

docs-open: docs  ## Build docs and open in default browser (macOS / linux).
	@command -v open >/dev/null && open docs/_build/html/index.html || \
		(command -v xdg-open >/dev/null && xdg-open docs/_build/html/index.html) || \
		echo "Built docs/_build/html/index.html — open it manually"

refresh-intersphinx-inv:  ## Refresh vendored intersphinx fallback inventories in docs/_inv/.
	$(PY) tools/refresh_intersphinx_inv.py

# --- MATLAB parity diff (Phase 4.1) ---------------------------------

diff-matlab:  ## Diff MATLAB nSTAT checkout against parity/manifest.yml.
	$(PY) tools/parity/diff_against_matlab.py

# --- README freshness check -----------------------------------------

readme-check:  ## Verify README intra-repo links, images, and code-snippet imports.
	$(PY) tools/check_readme_links.py

helpfile-check:  ## Verify every nstat.__all__ symbol is documented in AGENT_GUIDE + ClassDefinitions.
	$(PY) tools/check_helpfile_freshness.py

freshness-check: readme-check helpfile-check  ## Run both README and helpfile freshness checks.

# --- formatting / linting (no enforcement; recommendations only) ----

format:  ## Run ruff format if installed (no-op otherwise — non-enforcing).
	@command -v ruff >/dev/null && ruff format . || echo "ruff not installed; skipping"

lint:  ## Run ruff check if installed (no-op otherwise).
	@command -v ruff >/dev/null && ruff check . || echo "ruff not installed; skipping"

typecheck:  ## Run mypy if installed (no-op otherwise).
	@command -v mypy >/dev/null && mypy nstat/ || echo "mypy not installed; skipping"

# --- release / sanity ------------------------------------------------

version-check:  ## Assert pyproject.toml, CITATION.cff, RELEASE_NOTES.md versions agree.
	$(PYTEST) -q tests/test_version_sync.py

sanity:  ## Quick "is the package importable + entry points wired?" check.
	@$(PY) -c "import nstat; print(f'nstat OK — __all__ has {len(nstat.__all__)} symbols')"
	@$(PY) -c "from nstat.install import main; print('nstat-install entry point OK')"
	@$(PY) -c "from nstat.paper_examples import main; print('nstat-paper-examples entry point OK')"

release-check: version-check freshness-check test docs-strict regen  ## Pre-release verification gauntlet.
	@echo "Release check passed — ready to tag."

# --- local CI mirror -------------------------------------------------
# Reproduce the *deterministic* PR gates locally — everything that does
# NOT need the 150 MB figshare dataset or the heavy JAX extras.  Run
# this before pushing so PRs land green on the first GitHub Actions run
# instead of burning billing minutes on avoidable failures.
#
# Coverage vs. the GitHub workflows:
#   freshness-check  -> readme-check.yml + helpfile-check.yml
#   test             -> ci.yml: unit-lint, cleanroom-compliance,
#                       symbol-surface-audit, data-integrity
#   docs-strict      -> ci.yml: docs-build  (+ deploy-docs.yml build)
#   drift-check      -> ci.yml: paper-gallery-artifacts,
#                       parity-report-artifacts
#
# NOT covered locally (need the dataset / JAX / a real runner): the
# notebook-execution jobs, regenerate-figures, extras-{dynamax,clusterless}.
# Those still run on GitHub when you actually have minutes.

ci-local: freshness-check test docs-strict drift-check  ## Run the deterministic PR gates locally (no dataset/JAX needed).
	@echo
	@echo "ci-local PASSED — the deterministic PR gates are green."
	@echo "Notebook / figure-regen / extras jobs still run on GitHub Actions."

drift-check:  ## Regenerate deterministic CI-checked artifacts and fail if they drift (no commit).
	@echo "Regenerating gallery + parity-report artifacts..."
	@$(PY) tools/paper_examples/build_gallery.py
	@$(PY) tools/parity/build_report.py
	@echo "Checking for drift..."
	@git diff --exit-code -- \
		README.md \
		docs/paper_examples.md \
		docs/figures/manifest.json \
		docs/figures/example01/README.md \
		docs/figures/example02/README.md \
		docs/figures/example03/README.md \
		docs/figures/example04/README.md \
		docs/figures/example05/README.md \
		parity/report.md \
	&& echo "No artifact drift." \
	|| { echo "DRIFT: regenerated artifacts differ — commit the regenerated files above."; exit 1; }
	@echo "Note: parity/notebook_fidelity.yml is environment-coupled (it records the"
	@echo "      local MATLAB-checkout path) and is validated on GitHub Actions, not here."

clean:  ## Remove built artifacts (__pycache__, .pytest_cache, docs/_build, dist).
	rm -rf docs/_build dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned build artifacts."
