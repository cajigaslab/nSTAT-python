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
release = "0.6.0"


# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",                  # Markdown-in-RST
    "sphinx.ext.autodoc",           # Pull docstrings from Python source
    "sphinx.ext.autosummary",       # Auto-generate API reference stubs
    "sphinx.ext.napoleon",          # Parse NumPy/Google-style docstrings
    "sphinx.ext.intersphinx",       # Cross-link to numpy/scipy/python docs
    "sphinx.ext.viewcode",          # "[source]" links from docs to GitHub
    "sphinx.ext.todo",              # ``.. todo::`` directives surfaced
    "sphinx.ext.mathjax",           # Render LaTeX math (client-side MathJax)
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "superpowers", "notebook_galleries", "parity"]
master_doc = "index"

# MyST: generate slug anchors for headings (h1–h3) so cross-page links to a
# section — e.g. ``self_check.md`` → ``goodness_of_fit_and_decoding.md#check-your-understanding``
# — resolve under the strict ``-W`` build.
myst_heading_anchors = 3


# MyST resolves ``glossary.md#slug`` links only against heading slugs, and
# explicit ``(target)=`` / ``{#id}`` markup is not consulted for that form.
# The glossary terms have long headings ("Local field potential (LFP)") but
# are linked by short slugs (``#local-field-potential``), so
# ``docs/_glossary_slugs.py`` overrides the slug for exactly those headings
# (it reads the ``<a id="slug"></a>`` anchors in glossary.md).  This only
# drives MyST link resolution; the HTML id of each heading is unchanged.
# The module lives next to conf.py and is imported by its bare name (the docs
# dir is put on sys.path below) because MyST's config is pickled into
# Sphinx's environment cache, and a function defined in conf.py cannot be;
# a ``docs.`` package prefix would collide with any installed top-level
# ``docs`` package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
myst_heading_slug_func = "_glossary_slugs.heading_slug"

# MyST: enable LaTeX math via ``$...$`` (inline) and ``$$...$$`` (block), plus
# ``\begin{align}...\end{align}`` AMS environments. Same syntax that GitHub's
# native MathJax integration uses for README math — one source renders on both
# github.com and the Sphinx site.
myst_enable_extensions = ["dollarmath", "amsmath"]


# -- autosummary / autodoc ---------------------------------------------------

# Auto-generate API stubs on every build.  Output is written to
# ``docs/_autosummary/``.  The stubs must NOT be in ``exclude_patterns``:
# Sphinx only renders (and links) the autosummary table entries for stub
# documents that are part of the build, so excluding them leaves every
# api.rst / extras.rst entry unlinked and emits "stub file not found"
# warnings.  The generated stubs are git-ignored.
autosummary_generate = True
autosummary_imported_members = False  # only document symbols defined in nstat


def _build_class_targets() -> dict:
    """Map each public ``nstat`` name that is a class to its real home.

    Consumed by ``_templates/autosummary/class.rst``: ``module``/``qualname``
    locate the class behind a public name and ``is_alias`` says whether the
    public name is an alias of it (``nstat.CovColl = CovariateCollection``).
    Plain data (no functions) so Sphinx can pickle it into its config cache.
    """
    import inspect

    import nstat

    targets: dict = {}
    for name in nstat.__all__:
        obj = getattr(nstat, name, None)
        if inspect.isclass(obj):
            targets[f"nstat.{name}"] = {
                "module": obj.__module__,
                "qualname": obj.__qualname__,
                "is_alias": obj.__name__ != name,
            }
    return targets


autosummary_context = {"class_targets": _build_class_targets()}
# Case-insensitive filesystems (macOS default) cannot hold both
# ``nstat.nstat_install.rst`` and ``nstat.nSTAT_Install.rst``; give the
# second a distinct stub name so the toctree finds both everywhere.
autosummary_filename_map = {"nstat.nSTAT_Install": "nstat.nSTAT_Install-case"}

# Document members by default, including dunders only when explicitly
# documented in the source.  Inherited members are NOT on by default (it
# would add every ``object`` method to every class); the autosummary class
# template turns on ``:inherited-members:`` for the non-alias ``autoclass`` so
# thin subclasses (``nstColl``, ``Covariate``, ...) list their inherited API.
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
# False: the autosummary class template already emits
# ``.. automethod:: __init__``; letting napoleon also inject ``__init__``
# documents it twice ("duplicate object description" warnings).
napoleon_include_init_with_doc = False
# Render NumPy "Attributes" sections as :ivar: fields rather than
# ``.. attribute::`` directives: autodoc also documents dataclass fields as
# members, so the directive form registered every field twice
# ("duplicate object description").
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True


# -- intersphinx -------------------------------------------------------------

# Each target carries a vendored fallback inventory under ``docs/_inv/``.
# Sphinx tries the live ``objects.inv`` first (so cross-refs stay current),
# then falls back to the committed copy if the host is unreachable.  This
# keeps the strict ``-W`` docs build / GitHub Pages deploy from failing on
# a transient network hiccup (e.g. a ``docs.scipy.org`` timeout), since a
# successful fallback emits no warning.  ``intersphinx_timeout`` bounds the
# wait per host so an outage can't stall the build for minutes.
# Refresh the vendored inventories with ``make refresh-intersphinx-inv``
# (or re-download ``<uri>objects.inv``) when upstream targets move.
intersphinx_timeout = 10  # seconds per inventory fetch

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", (None, "_inv/python.inv")),
    "numpy": ("https://numpy.org/doc/stable/", (None, "_inv/numpy.inv")),
    "scipy": ("https://docs.scipy.org/doc/scipy/", (None, "_inv/scipy.inv")),
    "matplotlib": ("https://matplotlib.org/stable/", (None, "_inv/matplotlib.inv")),
    "sympy": ("https://docs.sympy.org/latest/", (None, "_inv/sympy.inv")),
}


# -- todo directives ---------------------------------------------------------

todo_include_todos = False  # don't render TODOs in published docs


# -- HTML theme --------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "logo_only": True,                      # show the brand logo, hide the text title
    "style_nav_header_background": "#0e1117",  # deep charcoal to match the dark theme (see _static/custom.css)
}

# Brand assets (the official nSTAT wordmark, sourced from the MATLAB
# toolbox helpfiles).  ``nstat-logo-light.png`` is a white variant that
# reads cleanly on the dark/blue sidebar header; the favicon is the
# neuron-"n" monogram cropped from the same mark.
html_logo = "_static/nstat-logo-light.png"
html_favicon = "_static/favicon.png"

# Custom stylesheet that restyles the RTD theme into the dark
# "oscilloscope / lab-instrument" look shared with the hand-built landing
# pages (intro.html, extras_summary.html) — phosphor-green signal accents
# on deep charcoal, with matching typography, tables, and code blocks.
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = f"nSTAT Python {release}"

# Light Pygments tokens to match the light Read-the-Docs theme.
pygments_style = "friendly"

# Copy standalone HTML pages into the build root.  Sphinx flattens a
# directory entry's *contents* into the output root (the entry's own
# name is dropped), so the docs/changes/ files publish at the site root:
#   docs/extras_summary.html            -> /extras_summary.html
#   docs/paper_examples_gallery.html    -> /paper_examples_gallery.html
#   docs/changes/whats_new.html         -> /whats_new.html   (What's New index)
#   docs/changes/2026-*-*.html          -> /2026-*-*.html
# These are self-contained pages (embedded CSS, no Sphinx wrap).  The
# landing file is named whats_new.html — NOT index.html — so it does not
# clobber Sphinx's generated /index.html.
html_extra_path = [
    "extras_summary.html",
    "paper_examples_gallery.html",
    "notebooks_gallery.html",
    "extras_gallery.html",
    "galleries.html",
    "intro.html",
    "changes",
]
