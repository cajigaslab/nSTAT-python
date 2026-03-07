# Parity Inventory

This directory tracks MATLAB-to-Python parity for the standalone port.

Current inventory source:
- [`manifest.yml`](./manifest.yml)

Current headline status:
- Public API coverage is close, but `Events` is not exported from the package root, `nSTAT_Install` is only available through a submodule alias, and `getPaperDataDirs` is still missing.
- Help/notebook parity is mostly present by filename, but `ConfidenceIntervalOverview` is missing.
- Canonical paper examples are only partially represented in Python because the repo does not yet expose the five standalone example scripts, figure gallery, or manifest structure used by the MATLAB repo.
- GitHub-facing docs/gallery parity is mostly absent.
- The Python repo still has two package trees in practice: the real `nstat/` package and vestigial `src/nstat/` files referenced by tests.

Next actions:
1. Collapse the repo to one canonical package layout.
2. Close the remaining public API gaps.
3. Add the five canonical paper example scripts and generated docs gallery.
4. Expand parity verification so CI fails when the manifest drifts.
