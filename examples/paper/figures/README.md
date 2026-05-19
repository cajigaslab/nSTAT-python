# Deprecated figure tree

The canonical paper-example figure outputs now live at:

    docs/figures/exampleNN/

The PNGs in this directory are **byte-identical duplicates** kept here for
backward compatibility while downstream tools migrate to the canonical
path.  Running

    python examples/paper/regenerate_all_figures.py

writes only to `docs/figures/`.  Individual paper scripts default
their `--export-dir` to `docs/figures/exampleNN/` as well.

This directory will be removed in a future release.  To clean it up
locally:

    git rm -r examples/paper/figures/
