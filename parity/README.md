# Parity Inventory

This directory tracks MATLAB-to-Python parity for the standalone port.

Current inventory source:
- [`manifest.yml`](./manifest.yml)

Current headline status:
- Public API coverage is close, but `Events` is not exported from the package root, `nSTAT_Install` is only available through a submodule alias, and `getPaperDataDirs` is still missing.
- Help/notebook parity is mostly present by filename, but `ConfidenceIntervalOverview` is missing.
- Canonical paper examples are only partially represented in Python because the repo does not yet expose the five standalone example scripts, figure gallery, or manifest structure used by the MATLAB repo.
- GitHub-facing docs/gallery parity is mostly absent.
- The canonical package layout is now the root `nstat/` package only.

Next actions:
1. Close the remaining public API gaps.
2. Add the five canonical paper example scripts and generated docs gallery.
3. Expand parity verification so CI fails when the manifest drifts.
