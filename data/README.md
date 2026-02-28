# Shared Example Data Policy

This directory tracks only shared example datasets.

Rules:
- Data may be shared with MATLAB nSTAT when explicitly listed in `datasets_manifest.json`.
- Every dataset record must include immutable version and SHA256 checksum.
- No non-data files are shared across repositories.

Recommended flow:
1. Publish immutable data artifacts (for example, GitHub Release assets).
2. Add records to `datasets_manifest.json` with URL and checksum.
3. Update `tools/compliance/shared_data_allowlist.yml` with approved overlaps.
