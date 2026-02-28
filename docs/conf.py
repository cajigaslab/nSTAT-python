from __future__ import annotations

import os
import sys


sys.path.insert(0, os.path.abspath("../src"))

project = "nSTAT-python"
author = "Cajigas Lab"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

source_suffix = {
    ".md": "markdown",
}

autosummary_generate = True
autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "nSTAT-python Documentation"

# PubMed/PMC frequently block automated HEAD/GET requests with 403 in CI.
# We keep these references in docs but exclude them from linkcheck enforcement.
linkcheck_ignore = [
    r"https://pubmed\.ncbi\.nlm\.nih\.gov/.*",
    r"https://pmc\.ncbi\.nlm\.nih\.gov/articles/.*",
]
