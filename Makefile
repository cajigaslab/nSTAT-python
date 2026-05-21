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
        docs docs-strict docs-open \
        diff-matlab \
        format lint typecheck \
        version-check sanity clean release-check

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

regen-notebook-fidelity:  ## parity/notebook_fidelity.yml from tools/notebooks/parity_notes.yml.
	$(PY) tools/parity/build_notebook_fidelity_audit.py

regen-figures:  ## ~30 min — regenerate every paper-example PNG (needs figshare dataset).
	@echo "Note: this requires the figshare paper dataset.  Set NSTAT_OFFLINE=1 to fail-fast."
	$(PY) examples/paper/regenerate_all_figures.py

# --- docs ------------------------------------------------------------

docs:  ## Build the Sphinx site (writes to docs/_build/html).
	$(SPHINX) -b html docs docs/_build/html

docs-strict:  ## Build docs with -W (warnings as errors, matches CI).
	$(SPHINX) -W -b html docs docs/_build/html

docs-open: docs  ## Build docs and open in default browser (macOS / linux).
	@command -v open >/dev/null && open docs/_build/html/index.html || \
		(command -v xdg-open >/dev/null && xdg-open docs/_build/html/index.html) || \
		echo "Built docs/_build/html/index.html — open it manually"

# --- MATLAB parity diff (Phase 4.1) ---------------------------------

diff-matlab:  ## Diff MATLAB nSTAT checkout against parity/manifest.yml.
	$(PY) tools/parity/diff_against_matlab.py

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

release-check: version-check test docs-strict regen  ## Pre-release verification gauntlet.
	@echo "Release check passed — ready to tag."

clean:  ## Remove built artifacts (__pycache__, .pytest_cache, docs/_build, dist).
	rm -rf docs/_build dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned build artifacts."
