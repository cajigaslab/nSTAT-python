"""Sphinx configuration for nstat-python.

Builds a Sphinx HTML site at ``docs/_build/html``.  Wired into CI via
``.github/workflows/ci.yml::docs-build`` and to GitHub Pages via
``.github/workflows/deploy-docs.yml``.

Documentation conventions
-------------------------
- Docstrings use NumPy style — parsed by ``sphinx.ext.napoleon``.
- API reference is auto-generated from the package via
  ``sphinx.ext.autodoc`` + ``sphinx.ext.autosummary``.  No manual list
  of symbols — adding a name to ``nstat.__all__`` makes it appear in
  the rendered docs automatically.
- Cross-references to numpy / scipy / matplotlib / Python stdlib are
  resolved by ``sphinx.ext.intersphinx``.

Bump ``release`` here when ``pyproject.toml`` version moves.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make ``nstat`` importable so autodoc can introspect it without
# requiring a separate ``pip install -e .`` step in every doc build
# environment.  The repo root is one directory up from this conf.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Force matplotlib headless during docs build (autodoc imports modules
# that may instantiate figures).
os.environ.setdefault("MPLBACKEND", "Agg")


# -- Project information -----------------------------------------------------

project = "nSTAT Python"
author = "Cajigas Lab"
release = "0.3.1"


# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",                  # Markdown-in-RST
    "sphinx.ext.autodoc",           # Pull docstrings from Python source
    "sphinx.ext.autosummary",       # Auto-generate API reference stubs
    "sphinx.ext.napoleon",          # Parse NumPy/Google-style docstrings
    "sphinx.ext.intersphinx",       # Cross-link to numpy/scipy/python docs
    "sphinx.ext.viewcode",          # "[source]" links from docs to GitHub
    "sphinx.ext.todo",              # ``.. todo::`` directives surfaced
]

exclude_patterns = ["_build", "_autosummary", "Thumbs.db", ".DS_Store", "superpowers"]
master_doc = "index"


# -- autosummary / autodoc ---------------------------------------------------

# Auto-generate API stubs on every build.  Output is written to
# ``docs/_autosummary/`` and is in ``exclude_patterns`` so the raw stubs
# aren't published — only the rendered HTML is.
autosummary_generate = True
autosummary_imported_members = False  # only document symbols defined in nstat

# Document members by default, including dunders only when explicitly
# documented in the source.  ``inherited-members`` would balloon every
# class to show every Python object method; keep it off.
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"   # render type hints in the Parameters block, not the signature
autodoc_member_order = "bysource"
autodoc_class_signature = "mixed"   # show __init__ signature inline with class signature

# When a member's docstring is missing, autodoc falls back to a stub —
# don't fail the build, just emit a warning.  CI's ``docs-build`` job
# runs with -W (warnings as errors) so genuine doc gaps DO surface, but
# missing-docstring warnings are noisy enough that we exempt them.
suppress_warnings = ["autodoc.import_object"]


# -- napoleon (NumPy-style docstrings) ---------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True


# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
}


# -- todo directives ---------------------------------------------------------

todo_include_todos = False  # don't render TODOs in published docs


# -- HTML theme --------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}

# Copy standalone HTML pages into the build root.  Used for
# docs/extras_summary.html — a self-contained landing page (embedded
# CSS, no Sphinx wrap) that lives at
# https://cajigaslab.github.io/nSTAT-python/extras_summary.html.
html_extra_path = ["extras_summary.html"]
