"""Regenerate ``docs/extras_gallery.html`` (and ``docs/galleries.html``).

Mirrors ``tools/paper_examples/build_gallery.py`` but for the
``examples/extras/*_demo.py`` scripts.  Re-run after editing
``examples/extras/manifest.yml`` or
``tools/extras_build/extras_descriptions.yml`` — ``make regen`` wires
this in automatically.
"""
from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nstat.extras_gallery import write_extras_gallery_outputs


if __name__ == "__main__":
    manifest_path, html_path, galleries_index_path = write_extras_gallery_outputs(REPO_ROOT)
    print(manifest_path)
    print(html_path)
    print(galleries_index_path)
