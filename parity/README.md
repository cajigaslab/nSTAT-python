# Parity Inventory

This directory tracks MATLAB-to-Python parity for the standalone port.

Current inventory source:
- [`manifest.yml`](./manifest.yml)

Current headline status:
- Public API coverage now matches the MATLAB inventory except for the explicitly non-applicable `nstatOpenHelpPage`.
- Help/notebook parity now covers the full MATLAB helpfile notebook surface by filename.
- Canonical paper examples are only partially represented in Python because the repo does not yet expose the five standalone example scripts, figure gallery, or manifest structure used by the MATLAB repo.
- GitHub-facing docs/gallery parity is mostly absent.
- The canonical package layout is now the root `nstat/` package only.

Next actions:
1. Add the five canonical paper example scripts and generated docs gallery.
2. Expand parity verification so CI fails when the manifest drifts.
